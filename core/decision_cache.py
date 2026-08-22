"""
core/decision_cache.py

Rework Blueprint 5.5 -- Phase 1: state-bin decision cache.

Generalizes the existing "HOLD: change too small" idea (SentinelGate's
hysteresis dwell logic, core/sentinel_gate.py) from "identical to the
last decision" to "close enough to a decision we already made and the
gate already approved recently." Before agents/strategist.py's
Strategist.decide() is called on a cadence tick, main.py checks this
cache for a bin close to the current (t_in, t_out, hour_of_day, season)
tuple; a hit skips the real LLM call entirely and reuses the cached
proposal.

IMPORTANT -- this cache never bypasses SentinelGate. A cache hit still
goes through gate.check() for THIS tick's actual measurements exactly
like a fresh proposal would (see main.py's ProjectEnvelope.decide()).
The cache only ever removes the LLM call from the critical path, never
the safety/comfort scoring. That's also why put() below only ever
accepts an already-APPROVED, post-gate proposal (Blueprint Section 7's
explicit risk mitigation: "only caching approved (post-gate)
trajectories, never raw LLM proposals" -- caching a rejected or
never-scored proposal would let a bad decision get silently replayed
onto future similar-looking ticks without ever being re-scored).

Binning dimensions, and why "occupancy flag" isn't one of them: the
blueprint's 5.5 spec bins on (indoor-temp x outdoor-temp x time-of-day x
occupancy flag). This codebase has no occupancy signal anywhere --
bms_mcp/tools.py and core/energyplus_bridge.py expose weather, carbon
intensity, and zone temperature, but nothing occupancy-related. Rather
than invent one, hour-of-day is combined with the existing
WINTER/SUMMER/SHOULDER label from core/seasonality.classify_season(),
which is already the closest real "what regime am I in" signal this
system has -- it's literally the same value SentinelGate's opt-in
baseline-direction guardrail keys off of.

TTL is expressed in CADENCE TICKS, not raw EnergyPlus physical steps --
the same convention config/building_policy.yaml's
hysteresis.min_dwell_steps already uses, and for the same reason: a
raw-step TTL silently means something different every time
strategist.cadence_steps is retuned (see that key's own comment in the
YAML for the exact bug class this avoids -- min_dwell_steps used to
mean 9.4 DAYS at cadence_steps=300 and 9 HOURS at cadence_steps=12 for
the *same* configured number, because nobody noticed the unit was
"ticks" not "steps"). decision_cache.ttl_ticks: 4 at the default
cadence_steps: 12 means "reuse a bin for up to 4 * 12 * 15min = 12
hours," not 4 raw 15-minute steps.
"""


class DecisionCache:
    """Deliberately NOT a general LRU. The binned state space here is
    small and bounded (a handful of temperature buckets x a handful of
    time-of-day buckets x 3 seasons), so a plain dict that lazily prunes
    an expired entry the next time it's looked up is simpler than an
    eviction policy and just as effective for this size of key space."""

    def __init__(self, policy):
        cache_cfg = (policy or {}).get("strategist", {}).get("decision_cache", {}) or {}
        self.enabled = bool(cache_cfg.get("enabled", False))
        self.indoor_bin_c = float(cache_cfg.get("indoor_bin_c", 0.5))
        self.outdoor_bin_c = float(cache_cfg.get("outdoor_bin_c", 1.0))
        self.hour_bin_h = float(cache_cfg.get("hour_bin_h", 3.0))
        self.ttl_ticks = int(cache_cfg.get("ttl_ticks", 4))
        self._store = {}  # bin_key -> (tick_index_written, proposal_dict)
        self.hits = 0
        self.misses = 0

    def _bin_key(self, t_in, t_out, hour_of_day, season):
        i_bin = round(t_in / self.indoor_bin_c) if self.indoor_bin_c > 0 else round(t_in, 1)
        o_bin = round(t_out / self.outdoor_bin_c) if self.outdoor_bin_c > 0 else round(t_out, 1)
        h_bin = int(hour_of_day // self.hour_bin_h) if self.hour_bin_h > 0 else int(hour_of_day)
        return (i_bin, o_bin, h_bin, season)

    def get(self, t_in, t_out, hour_of_day, season, tick_index):
        """Returns a shallow copy of the cached proposal dict (with
        "cache_hit": True stamped onto it) if a close-enough, still-fresh
        bin exists, else None. Never raises -- a miss or a disabled
        cache is always a normal, silent event; the caller falls
        straight through to a real Strategist call, same as before this
        cache existed."""
        if not self.enabled:
            return None
        key = self._bin_key(t_in, t_out, hour_of_day, season)
        entry = self._store.get(key)
        if entry is None:
            self.misses += 1
            return None
        cached_tick, proposal = entry
        if tick_index - cached_tick > self.ttl_ticks:
            del self._store[key]  # stale -- prune now (lazy eviction)
            self.misses += 1
            return None
        self.hits += 1
        cached = dict(proposal)
        cached["cache_hit"] = True
        return cached

    def put(self, t_in, t_out, hour_of_day, season, tick_index, proposal):
        """Only ever call this with an APPROVED, post-gate proposal --
        see the module docstring's Section-7 risk-mitigation note."""
        if not self.enabled:
            return
        key = self._bin_key(t_in, t_out, hour_of_day, season)
        self._store[key] = (tick_index, dict(proposal))

    def invalidate(self, t_in, t_out, hour_of_day, season):
        """Drops a bin immediately (rather than waiting out its TTL) --
        called when a cache hit's proposal turns out to be REJECTED by
        the gate on replay: the state has drifted enough that this bin's
        cached decision is no longer trustworthy, so don't keep serving
        it to other ticks that still land in the same bin."""
        if not self.enabled:
            return
        self._store.pop(self._bin_key(t_in, t_out, hour_of_day, season), None)

    @property
    def hit_rate(self):
        total = self.hits + self.misses
        return (self.hits / total) if total else 0.0
