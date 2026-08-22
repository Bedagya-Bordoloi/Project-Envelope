"""
scripts/check_local_provider.py

Same purpose as scripts/check_secondary_provider.py, but for a LOCAL
OpenAI-compatible model server (Ollama, vLLM, LM Studio, etc.) instead of
a cloud secondary provider. The real risk with a local model isn't
volume (no rate limit) -- it's whether it reliably honors the SAME forced
tool_choice this project's whole control loop depends on:

    tool_choice={"type": "function", "function": {"name": "set_hvac"}}

agents/strategist.py expects every successful call to return a tool call
with a parseable 'setpoint_c' (or 'setpoint') key. A model that ignores
forced tool_choice, replies in plain text, or gets the argument name
wrong doesn't crash anything (the existing retry + fallback logic in
strategist.py catches that) -- but it silently trades "AI (Fallback) from
rate limits" for "AI (Fallback) from bad tool calls," which is the same
failure from the dashboard's point of view. Better to find that out here,
against N repeated calls with realistic varying inputs, than live.

This script does NOT touch config/building_policy.yaml or .env -- it
takes the local server's base_url/model/api_key as CLI args (with
sensible Ollama defaults) so you can test before deciding whether to
wire it into the real secondary_provider config.

Usage (Ollama defaults):
    python scripts/check_local_provider.py --model llama3.1:8b-instruct

Usage (custom local server):
    python scripts/check_local_provider.py \\
        --base-url http://localhost:11434/v1 \\
        --model mistral:7b-instruct \\
        --api-key ollama \\
        --trials 30

Usage (fallback to tool_choice="auto" if forced fails a lot):
    python scripts/check_local_provider.py --model llama3.2:3b-instruct --tool-choice auto

Exit code 0 if the success rate meets --min-success-rate (default 0.9).
1 otherwise -- read the per-trial detail either way.
"""

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bms_mcp.tools import TOOL_SCHEMAS

# A spread of realistic (t_in, t_out) pairs covering winter / shoulder /
# summer conditions -- the same inputs agents/strategist.py's
# _build_prompt() would see across a real year, not just one easy case.
TEST_CASES = [
    {"t_in": 20.2, "t_out": -11.1, "season": "deep winter"},
    {"t_in": 20.4, "t_out": -2.2,  "season": "winter"},
    {"t_in": 21.0, "t_out": 8.3,   "season": "shoulder"},
    {"t_in": 22.5, "t_out": 18.0,  "season": "mild"},
    {"t_in": 24.0, "t_out": 27.5,  "season": "summer"},
    {"t_in": 24.5, "t_out": 33.0,  "season": "peak summer"},
]


def build_prompt(t_in, t_out):
    """Deliberately mirrors the shape/tone of agents/strategist.py's
    real _build_prompt() (goal framing, situation block, forced-JSON
    instruction) without importing it directly -- this script must stay
    runnable with zero EnergyPlus/policy/tool_context dependencies so it
    can be run as a 30-second standalone check."""
    return (
        f"You are the 'Eco-Loop' BMS Strategist. GOAL: Beat 22.0C Baseline profit.\n\n"
        f"[SITUATION]\n"
        f"- Outdoor: {t_out:.1f}C | Indoor: {t_in:.1f}C\n"
        f"- Stability: Do not change setpoint unless weather trends shift significantly.\n\n"
        f"[TASK]\nPropose a setpoint that maximizes profit while ensuring 100% Gate Approval. "
        f"You MUST call the 'set_hvac' tool.\n\n"
        f"IMPORTANT: Output valid JSON tool calls only. Do not provide setpoints as plain text."
    )


def run_trial(client, model, tool_choice_mode, t_in, t_out, timeout):
    prompt = build_prompt(t_in, t_out)
    tool_choice = (
        {"type": "function", "function": {"name": "set_hvac"}}
        if tool_choice_mode == "forced" else "auto"
    )
    start = time.perf_counter()
    try:
        resp = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model=model,
            tools=TOOL_SCHEMAS,
            tool_choice=tool_choice,
            timeout=timeout,
        )
    except Exception as e:
        return {"ok": False, "latency_s": time.perf_counter() - start,
                "detail": f"Request error: {type(e).__name__}: {e}"}

    latency_s = time.perf_counter() - start
    message = resp.choices[0].message

    if not message.tool_calls:
        text_preview = (message.content or "")[:120]
        return {"ok": False, "latency_s": latency_s,
                "detail": f"No tool call. Model replied in text instead: {text_preview!r}"}

    try:
        args = json.loads(message.tool_calls[0].function.arguments)
    except (json.JSONDecodeError, IndexError, AttributeError) as e:
        return {"ok": False, "latency_s": latency_s,
                "detail": f"Tool call present but arguments didn't parse: {e}"}

    val = args.get("setpoint_c") or args.get("setpoint")
    if val is None:
        return {"ok": False, "latency_s": latency_s,
                "detail": f"Tool call parsed but missing 'setpoint_c'/'setpoint'. Args: {args}"}

    try:
        val = float(val)
    except (TypeError, ValueError):
        return {"ok": False, "latency_s": latency_s,
                "detail": f"'setpoint_c' present but not a number: {val!r}"}

    return {"ok": True, "latency_s": latency_s, "setpoint": val,
            "detail": f"setpoint={val}, confidence={args.get('confidence')}, "
                      f"reason={args.get('reason')!r}"}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base-url", default="http://localhost:11434/v1",
                         help="OpenAI-compatible base URL of your local server (default: Ollama's default port).")
    parser.add_argument("--model", required=True,
                         help="Model name as your local server identifies it (e.g. llama3.1:8b-instruct for Ollama).")
    parser.add_argument("--api-key", default="ollama",
                         help="API key value -- most local servers (Ollama included) ignore the actual value "
                              "but the OpenAI client requires SOMETHING non-empty to be passed.")
    parser.add_argument("--trials", type=int, default=30,
                         help="Total number of test calls to make, spread evenly across TEST_CASES "
                              "(default 30 -- enough to get a meaningful success-rate estimate).")
    parser.add_argument("--timeout", type=float, default=30.0,
                         help="Per-call timeout in seconds. Local models on CPU can be much slower than "
                              "Groq/Gemini -- raise this if you see spurious timeout failures before "
                              "concluding tool-calling itself is broken.")
    parser.add_argument("--tool-choice", choices=["forced", "auto"], default="forced",
                         help="'forced' matches agents/strategist.py's real behavior exactly (recommended "
                              "first pass). If forced fails often, re-run with 'auto' to see whether that's "
                              "the actual fix needed, before assuming the model can't tool-call at all.")
    parser.add_argument("--min-success-rate", type=float, default=0.9,
                         help="Exit code 0 requires at least this fraction of trials to succeed (default 0.90 "
                              "-- roughly 'reliable enough to trust for a live demo without constant fallback').")
    args = parser.parse_args()

    from openai import OpenAI
    client = OpenAI(api_key=args.api_key, base_url=args.base_url)

    print(f"Local provider check")
    print(f"  base_url:    {args.base_url}")
    print(f"  model:       {args.model}")
    print(f"  tool_choice: {args.tool_choice}")
    print(f"  trials:      {args.trials}")
    print("-" * 70)

    results = []
    for i in range(args.trials):
        case = TEST_CASES[i % len(TEST_CASES)]
        r = run_trial(client, args.model, args.tool_choice, case["t_in"], case["t_out"], args.timeout)
        r["case"] = case
        results.append(r)
        status = "PASS" if r["ok"] else "FAIL"
        print(f"[{i+1:>2}/{args.trials}] {status}  ({case['season']:<12} t_in={case['t_in']:>5.1f} "
              f"t_out={case['t_out']:>5.1f})  {r['latency_s']:.2f}s  {r['detail']}")

    successes = [r for r in results if r["ok"]]
    n = len(results)
    success_rate = len(successes) / n if n else 0.0
    avg_latency = sum(r["latency_s"] for r in results) / n if n else 0.0
    avg_success_latency = (sum(r["latency_s"] for r in successes) / len(successes)) if successes else 0.0

    print("-" * 70)
    print(f"Success rate:            {len(successes)}/{n}  ({success_rate:.0%})")
    print(f"Avg latency (all calls): {avg_latency:.2f}s")
    if successes:
        print(f"Avg latency (successes): {avg_success_latency:.2f}s")

    # Rough real-run time estimate: this project makes ~2,920 real decide()
    # calls for a full year at the default cadence_steps=12, before
    # counting the second call a correction pass can add on a REJECTED tick.
    est_full_year_min = (2920 * avg_success_latency) / 60.0 if successes else float("nan")
    print(f"Rough full-year estimate: ~2,920 calls x {avg_success_latency:.2f}s "
          f"~= {est_full_year_min:.0f} min wall-clock (ignores correction-pass calls; "
          f"use --run-period-days for a faster real check instead of trusting this estimate blindly).")

    print("-" * 70)
    if success_rate >= args.min_success_rate:
        print(f"[PASS] {success_rate:.0%} >= required {args.min_success_rate:.0%}. "
              f"This model/tool_choice combo looks reliable enough to wire into "
              f"config/building_policy.yaml's strategist.secondary_provider (or promote to primary).")
        sys.exit(0)
    else:
        print(f"[FAIL] {success_rate:.0%} < required {args.min_success_rate:.0%}.")
        if args.tool_choice == "forced":
            print("Try re-running with --tool-choice auto -- some local models handle unforced "
                  "tool calling much better than a hard-forced function name. If auto passes, "
                  "you'll need to relax strategist.py's tool_choice for the local tier specifically.")
        else:
            print("Even 'auto' tool-calling isn't reliable for this model. Try a larger model "
                  "(8B instead of 3B) before concluding the local-model approach doesn't work.")
        sys.exit(1)


if __name__ == "__main__":
    main()
