"""
scripts/check_both_providers.py

You want a straight answer to "is Groq rate-limited right now, is Gemini
rate-limited right now, or are both actually fine" -- WITHOUT going
through Strategist.decide()'s full retry/backoff/shared-pacer/secondary-
fallback chain, because that chain can mask which specific provider is
the problem (a Groq rate limit silently triggers a Gemini attempt, which
might succeed OR fail, and decide() only gives you the end result).

This script makes exactly ONE minimal request to each provider, directly,
and reports a plain verdict for each:
    OK            -- key works, provider responded with a usable answer
    RATE LIMITED  -- key is valid, but you're currently over quota (429)
    AUTH FAILED   -- key is missing/invalid/revoked (401/403)
    NOT FOUND     -- wrong URL/model for that provider (404) -- a config
                     problem, not a quota problem
    FAILED        -- something else (network, timeout, etc.) -- see detail

Reads the same config/building_policy.yaml + .env your app uses, so this
checks the exact same keys/models/base_urls the real run will use.

Usage:
    python scripts/check_both_providers.py

Exit code: 0 if both providers are OK, 1 otherwise (so this is
CI/pre-demo-rehearsal friendly, same convention as the other check_*
scripts in this folder).
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yaml
from dotenv import load_dotenv

load_dotenv()


def load_policy(path="config/building_policy.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def classify_error(e):
    """Best-effort classification shared by both providers -- both the
    groq and openai SDKs raise exceptions whose class name and/or
    .status_code follow the same rough shape (they're both built on top
    of an OpenAI-compatible HTTP contract), so one classifier works for
    either."""
    status = getattr(e, "status_code", None)
    name = type(e).__name__

    if status == 429 or "RateLimitError" in name:
        return "RATE LIMITED", (
            "Key is valid but you're over quota right now. This clears "
            "on its own -- per-minute limits reset within a minute, "
            "daily limits reset at the provider's own daily boundary. "
            "No new key needed."
        )
    if status in (401, 403) or "AuthenticationError" in name or "PermissionDeniedError" in name:
        return "AUTH FAILED", (
            "Key is missing, invalid, or revoked. Double check the .env "
            "value (no stray quotes/whitespace) and that it hasn't been "
            "deleted in the provider's console."
        )
    if status == 404 or "NotFoundError" in name:
        return "NOT FOUND", (
            "The request reached the server but the URL/model combo is "
            "wrong -- a config problem, not a quota problem. Check "
            "base_url and model in config/building_policy.yaml."
        )
    return "FAILED", f"{name}: {e}"


def check_groq(policy):
    print("=" * 60)
    print("PRIMARY -- Groq")
    print("=" * 60)

    api_key = os.environ.get("GROQ_API_KEY")
    model = policy.get("strategist", {}).get("model", "openai/gpt-oss-20b")

    print(f"Model: {model}")
    print(f"GROQ_API_KEY: {'set (first 6: ' + api_key[:6] + '...)' if api_key else 'NOT SET'}")

    if not api_key:
        print("Verdict: AUTH FAILED -- GROQ_API_KEY not set in environment/.env.")
        return False

    from groq import Groq

    client = Groq(api_key=api_key)

    try:
        resp = client.chat.completions.create(
            messages=[{"role": "user", "content": "Reply with the single word OK."}],
            model=model,
            max_tokens=5,
            timeout=15,
        )
        text = resp.choices[0].message.content
        print(f"Verdict: OK -- response: {text!r}")
        return True
    except Exception as e:
        verdict, detail = classify_error(e)
        print(f"Verdict: {verdict}")
        print(f"Detail: {detail}")
        return False


def check_gemini(policy):
    print()
    print("=" * 60)
    print("SECONDARY -- Gemini")
    print("=" * 60)

    cfg = policy.get("strategist", {}).get("secondary_provider", {}) or {}

    if not cfg.get("enabled", False):
        print("secondary_provider.enabled is false in config/building_policy.yaml.")
        print("Verdict: SKIPPED -- nothing to check until you set enabled: true.")
        return True  # not a failure -- it's deliberately off

    key_env = cfg.get("api_key_env", "SECONDARY_LLM_API_KEY")
    api_key = os.environ.get(key_env)
    base_url = cfg.get("base_url")
    model = cfg.get("model")

    print(f"base_url: {base_url}")
    print(f"Model: {model}")
    print(f"{key_env}: {'set (first 6: ' + api_key[:6] + '...)' if api_key else 'NOT SET'}")

    if not api_key or not base_url or not model:
        print("Verdict: AUTH FAILED -- missing api key / base_url / model.")
        return False

    from openai import OpenAI

    client = OpenAI(api_key=api_key, base_url=base_url)

    try:
        resp = client.chat.completions.create(
            messages=[{"role": "user", "content": "Reply with the single word OK."}],
            model=model,
            max_tokens=5,
            timeout=15,
        )
        text = resp.choices[0].message.content
        print(f"Verdict: OK -- response: {text!r}")
        return True
    except Exception as e:
        verdict, detail = classify_error(e)
        print(f"Verdict: {verdict}")
        print(f"Detail: {detail}")
        return False


def main():
    policy = load_policy()

    groq_ok = check_groq(policy)
    gemini_ok = check_gemini(policy)

    print()
    print("=" * 60)
    print(f"Groq (primary):    {'OK' if groq_ok else 'NOT OK'}")
    print(f"Gemini (secondary): {'OK' if gemini_ok else 'NOT OK'}")
    print("=" * 60)

    if groq_ok and gemini_ok:
        print("Both providers are reachable right now.")
        sys.exit(0)
    elif not groq_ok and not gemini_ok:
        print("BOTH providers are currently unavailable -- decide() calls will "
              "fall through to the hardcoded physical-target fallback until "
              "at least one clears.")
        sys.exit(1)
    else:
        print("One provider is down but the other is up -- the app's "
              "fallback-to-secondary path should still keep you off the "
              "hardcoded fallback for now.")
        sys.exit(1)


if __name__ == "__main__":
    main()
