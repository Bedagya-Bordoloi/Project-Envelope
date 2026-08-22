"""
core/provider_pool.py

"""

import threading
import time

# How long a provider's daily request/token budget is assumed to take to
# refill if we can't tell from the 429's own text. Groq's docs describe
# this as "every 24 hours" rather than a fixed UTC boundary, so this
# pool tracks each provider's OWN rolling 24h window (starting from its
# first real call) rather than assuming an aligned UTC-midnight reset --
# safer to slightly over-wait than to retry a still-exhausted provider.
DEFAULT_DAILY_WINDOW_S = 24 * 60 * 60.0


class _RateLimiter:
    """Thread-safe minimum-interval pacer -- same mechanism as the
    process-wide _RateLimiter that already existed in
    agents/strategist.py (kept deliberately identical; the blueprint's
    Section 2 says to keep this mechanism, not replace it). The
    difference here is cardinality: agents/strategist.py used ONE shared
    instance for the single Groq account every Strategist/zone talked
    to. A pool of independently-owned providers needs ONE limiter PER
    PROVIDER, keyed by name -- pacing a Gemini call against Groq's
    per-minute budget (or vice versa) would be wrong, since they're
    different accounts with different budgets.
    """

    def __init__(self, max_per_minute):
        self._min_interval = 60.0 / max(1, max_per_minute)
        self._lock = threading.Lock()
        self._next_allowed = 0.0

    def wait(self):
        with self._lock:
            now = time.perf_counter()
            delay = self._next_allowed - now
            if delay > 0:
                time.sleep(delay)
                now = time.perf_counter()
            self._next_allowed = now + self._min_interval


class ProviderPool:
    """Round-robins across independently-owned providers (NOT multiple
    keys on the same provider/account -- see the module docstring's ToS
    note), skipping any provider currently in cooldown from a recent
    429, and recovering it automatically once its cooldown expires.

    Each entry in `providers` is a plain dict with at least:
        {"name": str, "client": <an object with .chat.completions.create(...)>,
         "model": str, "rpm": int}
    and OPTIONALLY:
        {"rpd": int}  # requests-per-day budget; 0/absent = not proactively tracked
    `client`/`model` are opaque to this class -- agents/strategist.py's
    decide() is what actually calls provider["client"].chat.completions.
    create(model=provider["model"], ...). This class only ever reads
    provider["name"], provider["rpm"], and provider["rpd"].
    """

    def __init__(self, providers):
        if not providers:
            raise ValueError(
                "ProviderPool requires at least one provider (the primary "
                "provider is always included -- see "
                "agents/strategist.py's _build_provider_pool())."
            )
        names = [p["name"] for p in providers]
        if len(set(names)) != len(names):
            raise ValueError(f"ProviderPool got duplicate provider names: {names}")

        self.providers = list(providers)
        self._limiters = {p["name"]: _RateLimiter(p.get("rpm", 25)) for p in self.providers}
        self._cooldown_until = {p["name"]: 0.0 for p in self.providers}
        # ROUND 7 -- proactive daily-budget tracking (see module docstring).
        # rpd: 0 means "no configured daily budget for this provider" --
        # _daily_available() always returns True in that case, i.e. purely
        # reactive (mark_daily_exhausted()) protection only.
        self._daily_limit = {p["name"]: int(p.get("rpd", 0) or 0) for p in self.providers}
        self._daily_count = {p["name"]: 0 for p in self.providers}
        # 0.0 means "window not started yet" -- the first record_request()
        # call for a provider opens its rolling 24h window.
        self._daily_window_start = {p["name"]: 0.0 for p in self.providers}
        self._lock = threading.Lock()
        self._idx = 0

    def __len__(self):
        return len(self.providers)

    def names(self):
        return [p["name"] for p in self.providers]

    def _daily_available_locked(self, name, now):
        """Must be called with self._lock held. Rolls the window over if
        it's expired (treated as fully available again, same as any
        rolling-window budget), otherwise compares the running count
        against the configured rpd."""
        limit = self._daily_limit.get(name, 0)
        if limit <= 0:
            return True  # not proactively tracked for this provider
        window_start = self._daily_window_start[name]
        if window_start == 0.0 or (now - window_start) >= DEFAULT_DAILY_WINDOW_S:
            return True  # window hasn't started, or has rolled over
        return self._daily_count[name] < limit

    def next_available(self):
        """Returns the next provider dict (round-robin order) whose
        cooldown has cleared AND whose own proactively-tracked daily
        budget (if configured) isn't already exhausted, and advances the
        internal pointer past it. Returns None if EVERY provider in the
        pool is currently cooling down / daily-exhausted -- the caller
        (Strategist.decide()) falls through to the hardcoded
        physical-target fallback in that case, exactly as the old logic
        did when both Groq and Gemini had failed."""
        n = len(self.providers)
        now = time.time()
        with self._lock:
            for i in range(n):
                candidate_idx = (self._idx + i) % n
                p = self.providers[candidate_idx]
                name = p["name"]
                if now >= self._cooldown_until[name] and self._daily_available_locked(name, now):
                    self._idx = (candidate_idx + 1) % n
                    return p
            return None

    def rate_limiter_for(self, name):
        """The per-provider pacer -- call .wait() on this immediately
        before the real network call, same pattern as the pre-Phase-4
        single shared limiter (paces REAL calls proactively so the
        account's own per-minute budget is never outrun in the first
        place, rather than only reacting after a 429 -- see
        agents/strategist.py's module docstring for the original bug
        this fixed)."""
        return self._limiters[name]

    def record_request(self, name):
        """ROUND 7: call this immediately before (or right after) a REAL
        network call is actually sent to `name`, so the proactive daily
        budget (if `rpd` was configured for this provider) stays
        accurate. Opens/rolls the provider's own 24h window as needed.
        A no-op (still counts, harmlessly) for providers with no
        configured `rpd` -- cheap enough not to bother branching on it.
        """
        now = time.time()
        with self._lock:
            window_start = self._daily_window_start[name]
            if window_start == 0.0 or (now - window_start) >= DEFAULT_DAILY_WINDOW_S:
                self._daily_window_start[name] = now
                self._daily_count[name] = 0
            self._daily_count[name] += 1

    def mark_rate_limited(self, name, backoff_s):
        """Puts `name` into a SHORT cooldown for backoff_s seconds --
        use this for an ordinary per-minute/per-request-burst 429 that's
        expected to clear quickly. For a 429 that's actually a daily
        quota exhaustion, use mark_daily_exhausted() instead (see module
        docstring for why these need different treatment). No separate
        'mark recovered' call is needed -- next_available() re-checks
        time.time() against this deadline on every call, so the
        provider becomes eligible again automatically once it passes."""
        with self._lock:
            self._cooldown_until[name] = time.time() + max(0.0, backoff_s)

    def mark_daily_exhausted(self, name, reset_hint_s=None):
        """ROUND 7: puts `name` into cooldown until its rolling 24h
        window is expected to clear -- NOT the short per-minute backoff
        mark_rate_limited() uses. Call this specifically when a 429's
        error text indicates a per-day (RPD/TPD) limit rather than a
        per-minute one (see agents/strategist.py's
        _looks_like_daily_quota_error()).

        `reset_hint_s`, if given (e.g. parsed from the provider's own
        Retry-After for this error), is honored directly. Otherwise this
        estimates the remaining time in `name`'s own rolling 24h window
        (opening one now if none was ever recorded -- a provider that's
        already daily-exhausted has obviously made real calls before,
        but defends against the edge case anyway) -- see
        DEFAULT_DAILY_WINDOW_S. Also defensively marks the provider's
        proactive daily counter as exhausted, so next_available() agrees
        even if `rpd` wasn't configured precisely.
        """
        now = time.time()
        with self._lock:
            window_start = self._daily_window_start.get(name, 0.0)
            if window_start == 0.0:
                window_start = now
                self._daily_window_start[name] = window_start
            remaining = max(0.0, DEFAULT_DAILY_WINDOW_S - (now - window_start))
            wait_s = reset_hint_s if (reset_hint_s and reset_hint_s > 0) else remaining
            self._cooldown_until[name] = now + wait_s
            limit = self._daily_limit.get(name, 0)
            if limit > 0:
                self._daily_count[name] = limit  # force _daily_available_locked() to agree

    def status(self):
        """Diagnostic snapshot for logging/dashboards: {name: {"cooldown_s":
        seconds remaining in cooldown (0.0 if available), "daily_used":
        current rolling-24h request count, "daily_limit": configured rpd
        (0 if not tracked)}}. Used by scripts that want to show pool
        health without tripping a real call (mirrors the intent of
        scripts/check_both_providers.py, generalized to N providers)."""
        now = time.time()
        with self._lock:
            return {
                p["name"]: {
                    "cooldown_s": max(0.0, self._cooldown_until[p["name"]] - now),
                    "daily_used": self._daily_count[p["name"]],
                    "daily_limit": self._daily_limit[p["name"]],
                }
                for p in self.providers
            }
