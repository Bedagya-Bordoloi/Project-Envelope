"""
scripts/check_llm_connectivity.py

Round-2 audit finding: the year-long control log showed exactly two
confidence values (0.5 and 0.9) across all 28 "AI" rows -- both hardcoded
constants from agents/strategist.py's fallback/defensive-parsing paths,
never a genuine LLM-returned confidence score. That's strong evidence the
Groq call was failing on (almost) every attempt, silently, the whole run.

This is the "2 minutes to confirm" check the audit recommended, now as a
real script instead of an inline print() you'd have to add and remove by
hand. It calls Strategist.decide() directly, a handful of times with
varied inputs, and reports plainly whether each call actually reached
Groq and parsed a response, or fell through to the hardcoded fallback --
using the llm_ok/error diagnostics fields agents/strategist.py now
returns.

Usage:
    python scripts/check_llm_connectivity.py
    python scripts/check_llm_connectivity.py --calls 10

Exit code is 0 if every call reached the LLM, 1 if any call fell back
(so this is CI/pre-demo-rehearsal friendly).
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv

from agents.strategist import Strategist

load_dotenv()


SAMPLE_INPUTS = [
    # (t_in, t_out, forecast, carbon)
    (22.0, 10.0, [9.0, 8.5, 8.0], "Low"),      # winter
    (22.0, 25.0, [26.0, 27.0, 28.0], "High"),  # summer, forecast spike
    (22.0, 18.0, [18.5, 19.0, 19.0], "Medium"),  # shoulder
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calls", type=int, default=len(SAMPLE_INPUTS),
                         help="How many decide() calls to make (cycles through SAMPLE_INPUTS).")
    parser.add_argument("--model", default=None,
                         help="Override the model string (defaults to config/building_policy.yaml's strategist.model).")
    args = parser.parse_args()

    if not os.environ.get("GROQ_API_KEY"):
        print("FAIL: GROQ_API_KEY is not set in the environment / .env file. "
              "Nothing to test -- every call will raise before it even reaches Groq.")
        sys.exit(1)

    model = args.model
    if model is None:
        import yaml
        with open("config/building_policy.yaml") as f:
            model = yaml.safe_load(f)["strategist"]["model"]

    print(f"Model: {model}")
    print(f"GROQ_API_KEY: set (first 6 chars: {os.environ['GROQ_API_KEY'][:6]}...)\n")

    strategist = Strategist(model=model)

    ok_count = 0
    fallback_count = 0

    for i in range(args.calls):
        t_in, t_out, forecast, carbon = SAMPLE_INPUTS[i % len(SAMPLE_INPUTS)]
        proposal = strategist.decide(t_in, t_out, forecast, carbon)
        llm_ok = proposal.get("llm_ok", True)
        tag = "LLM OK  " if llm_ok else "FALLBACK"
        print(f"[{i+1}/{args.calls}] {tag} | setpoint={proposal['setpoint']:.2f} "
              f"confidence={proposal['confidence']:.2f} latency={strategist.last_latency_s:.2f}s "
              f"tool_calls={strategist.last_tool_calls}")
        if not llm_ok:
            print(f"           error: {proposal.get('error')}")
            fallback_count += 1
        else:
            ok_count += 1

    print(f"\n{ok_count}/{args.calls} calls reached the LLM successfully; "
          f"{fallback_count}/{args.calls} fell through to the hardcoded fallback.")

    if fallback_count:
        print(
            "\nSome/all calls fell back. Common causes: expired/invalid "
            "GROQ_API_KEY, wrong model string for your account, network "
            "egress to api.groq.com blocked, or a tool-schema mismatch "
            "the model can't satisfy. The 'error' lines above are the raw "
            "exception/parse-failure message for each failed call -- start "
            "there. Until this shows 0 fallbacks, your control log's "
            "\"AI\" rows may include fallback decisions -- check the "
            "\"llm_ok\"/\"fallback_error\" fields main.py now writes to "
            "each log line."
        )
        sys.exit(1)
    else:
        print("\nAll calls reached the LLM. The Strategist is genuinely being used.")
        sys.exit(0)


if __name__ == "__main__":
    main()
