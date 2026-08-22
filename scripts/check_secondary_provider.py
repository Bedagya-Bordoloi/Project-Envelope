"""
scripts/check_secondary_provider.py

Priority 3b caveat (blueprint Section 1.6: "exact response-object
compatibility across providers isn't guaranteed by this SDK trick alone
-- worth a quick smoke test against whichever provider you pick before
relying on it for a demo"). agents/strategist.py only ever reaches the
secondary provider (config/building_policy.yaml's
strategist.secondary_provider) when the PRIMARY Groq call fails with a
rate limit -- which means the normal control flow could go an entire
demo without ever exercising the secondary path, and the first time it's
actually called for real could be live in front of judges.

This script calls the secondary provider DIRECTLY -- no primary call,
no rate-limit trigger needed -- using the exact same forced tool_choice
agents/strategist.py uses ({"type":"function","function":{"name":
"set_hvac"}}), so a "yes/no" on whether that forced form actually works
against your chosen provider (e.g. Gemini's OpenAI-compatibility layer)
takes one command instead of waiting to get rate-limited on stage.

Usage:
    python scripts/check_secondary_provider.py

Exit code 0 if the secondary provider is configured AND returns a
usable tool call. 1 if it's not configured, misconfigured, or the call
fails/returns something unusable -- read the printed detail either way.
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yaml
from dotenv import load_dotenv

from bms_mcp.tools import TOOL_SCHEMAS

load_dotenv()


def load_policy(path="config/building_policy.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def main():
    policy = load_policy()
    cfg = policy.get("strategist", {}).get("secondary_provider", {}) or {}

    if not cfg.get("enabled", False):
        print("[check_secondary_provider] strategist.secondary_provider.enabled is false "
              "in config/building_policy.yaml -- nothing to check. Set it to true first.")
        sys.exit(1)

    key_env = cfg.get("api_key_env", "SECONDARY_LLM_API_KEY")
    api_key = os.environ.get(key_env)
    base_url = cfg.get("base_url")
    model = cfg.get("model")

    print(f"Provider base_url: {base_url}")
    print(f"Model: {model}")
    print(f"API key env var: {key_env} -- {'set' if api_key else 'NOT SET'} "
          f"(first 6 chars: {api_key[:6] + '...' if api_key else 'n/a'})")

    if not api_key or not base_url or not model:
        print("[check_secondary_provider] Missing api key / base_url / model -- fix "
              "config/building_policy.yaml and/or your .env, then re-run.")
        sys.exit(1)

    # Same client agents/strategist.py's Strategist.__init__ uses for the
    # secondary tier. NOTE: this is the real `openai` package, not the
    # Groq SDK pointed at a different base_url -- that used to be the
    # approach here, but groq-python hardcodes the request path as
    # "{base_url}/openai/v1/chat/completions", which only lines up with
    # Groq's own bare base_url (https://api.groq.com, no path segment).
    # Gemini's documented OpenAI-compat base_url already ends in
    # ".../v1beta/openai/", so the Groq client was posting to
    # ".../v1beta/openai/openai/v1/chat/completions" -- doubled, 404,
    # every time, not just under rate-limit load. See Strategist.__init__
    # for the full writeup and https://ai.google.dev/gemini-api/docs/openai
    # for Google's own openai-package example against this same base_url.
    from openai import OpenAI

    client = OpenAI(api_key=api_key, base_url=base_url)

    prompt = (
        "You are a building HVAC strategist. The current indoor temperature is "
        "21.5C, outdoor is 3.0C. Propose a cooling/heating setpoint using the "
        "set_hvac tool."
    )

    try:
        resp = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model=model,
            tools=TOOL_SCHEMAS,
            tool_choice={"type": "function", "function": {"name": "set_hvac"}},
            timeout=15,
        )
    except Exception as e:
        print(f"[FAIL] Request itself errored: {type(e).__name__}: {e}")
        print("This is the exact failure mode you'd otherwise only discover live during "
              "a Groq rate limit -- fix it now, not during the demo.")
        sys.exit(1)

    message = resp.choices[0].message
    if not message.tool_calls:
        print("[FAIL] No tool call in the response -- the forced tool_choice form this "
              "project uses may not be supported by this provider's OpenAI-compatibility "
              "layer. Response message:")
        print(f"  {message}")
        print("Consider whether your provider needs tool_choice='auto' instead of a "
              "forced function name, or check that provider's OpenAI-compat tool-calling docs.")
        sys.exit(1)

    try:
        args = json.loads(message.tool_calls[0].function.arguments)
    except (json.JSONDecodeError, IndexError, AttributeError) as e:
        print(f"[FAIL] Tool call present but arguments didn't parse as JSON: {e}")
        sys.exit(1)

    val = args.get("setpoint_c") or args.get("setpoint")
    if val is None:
        print(f"[FAIL] Tool call parsed but has no 'setpoint_c'/'setpoint' key. Args: {args}")
        sys.exit(1)

    print(f"[PASS] Secondary provider returned a usable tool call: setpoint={val}, "
          f"confidence={args.get('confidence')}, reason={args.get('reason')!r}")
    print("Secondary tier is confirmed working -- agents/strategist.py will be able to "
          "use it when the primary Groq call hits a rate limit.")
    sys.exit(0)


if __name__ == "__main__":
    main()
