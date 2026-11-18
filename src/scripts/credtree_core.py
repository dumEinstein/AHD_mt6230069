"""CredTree scheduler and credibility state (design/round6/CREDIBILITY_TREE.md).

Pure Python: no skydiscover, no LLM, no I/O. The controller (credtree.py) owns generation and
evaluation; this module only decides WHERE the next evaluations go and updates the credibility
of the nodes they land on. analysis/calib/test_core.py drives it with the offline simulator's
world and checks that it makes the same decisions as sim.run_tree.

Tree: root -> Framing -> Attempt. An attempt is a small population refined by leaf edits.
Credibility is age-corrected: every attempt carries discounted observed successes s and
discounted expected successes E (the sum of the pooled hazard h(k) at each of its own ages),
and the draw is Gamma(a0 + s, a0 + E), "1 = as good as an average attempt of its age".

Constants below are chosen once and identical on all five tasks (spec section 9).
"""
import math
import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

# ---------------------------------------------------------------- constants (spec section 9)
GAMMA = 0.8                 # discount on s and E
BLOCK, BLOCK_EXT, BLOCK_MAX = 2, 1, 6      # evaluations per step, +1 per success, cap
WIDEN = 1.0                 # a new child is on offer while K < ceil(WIDEN * sqrt(n)), both levels
EXPLOIT = 0.3               # share of non-probe blocks sent straight to the global-best attempt
A0 = 2.0                    # prior weight, in expected successes
PROBE_VALID, PROBE_MAX = 6, 9      # a new attempt is guaranteed 6 valid evaluations, at most 9
N_START = 2                 # framings generated at the start, besides framing 0
HAZ_PRIOR = (1.0, 4.0)      # Beta prior per age bucket of the pooled hazard h(k)
CLO_PRIOR = (1.0, 49.0)     # Beta prior per gap bucket of the global-record rate pi(g)
AGE_BUCKETS = [(1, 1), (2, 3), (4, 7), (8, 15), (16, 30), (31, 10 ** 9)]
GAP_EDGES = [0.0, 0.05, 0.2, 0.5]          # gap buckets {0}, (0,.05], (.05,.2], (.2,.5], >.5
P_PARENT_BEST = 0.8         # leaf-edit parent: the attempt's best with this probability ...
NEAR_BEST_BAND = 0.1        # ... else uniform among programs within this normalised gap
DEDUP_RETRIES = 5           # consecutive duplicate framings that exhaust the root
REOPEN_AFTER = 60           # evaluations with no global record after which the root reopens
REPAIR_MIN_TRIED, REPAIR_MIN_RATE = 10, 0.15   # seed repair switches off below this valid rate
EMPTY_SLOTS_DEAD = 3        # consecutive slots that produced no candidate end an attempt's probe
# v2 early cut: a probe ends once the attempt has EARLY_CUT_AFTER valid evaluations and its best is still more
# than EARLY_CUT_GAP of the way from G back to b0. The v1 pilot's rewrite seeds scored 0-1 against b0 21.75 on
# prism; a 6-valid probe on such an attempt is mostly wasted. 0 disables (the sim equivalence tests).
EARLY_CUT_AFTER, EARLY_CUT_GAP = 3, 0.5
EPS = 1e-12

DEFAULTS: Dict[str, Any] = dict(
    gamma=GAMMA, block=BLOCK, block_ext=BLOCK_EXT, block_max=BLOCK_MAX, widen=WIDEN, exploit=EXPLOIT, a0=A0,
    probe=PROBE_VALID, probe_max=PROBE_MAX, haz_prior=HAZ_PRIOR, clo_prior=CLO_PRIOR, reopen=REOPEN_AFTER,
    repair_min_tried=REPAIR_MIN_TRIED, repair_min_rate=REPAIR_MIN_RATE, empty_dead=EMPTY_SLOTS_DEAD,
    early_cut_after=EARLY_CUT_AFTER, early_cut_gap=EARLY_CUT_GAP)


def constants() -> Dict[str, Any]:
    """The constants in force, for the startup log."""
    return {**DEFAULTS, "n_start": N_START, "dedup_retries": DEDUP_RETRIES, "p_parent_best": P_PARENT_BEST,
            "near_best_band": NEAR_BEST_BAND}


def age_bucket(k: int) -> int:
    for i, (lo, hi) in enumerate(AGE_BUCKETS):
        if lo <= k <= hi:
            return i
    return 0


def gap_bucket(g: float) -> int:
    if g <= EPS:
        return 0
    for i in range(1, len(GAP_EDGES)):
        if g <= GAP_EDGES[i]:
            return i
    return len(GAP_EDGES)


def pick_parent(members: Sequence[Tuple[Any, float, bool]], G: float, b0: float, rng: random.Random,
                p_best: float = P_PARENT_BEST, band: float = NEAR_BEST_BAND) -> Any:
    """Leaf-edit parent among an attempt's programs, given as (key, score, valid).

    With probability p_best the attempt's best; otherwise uniform among the valid programs whose
    normalised gap (best_A - x) / (G - b0) is at most `band` (falling back to the best)."""
    valid = [m for m in members if m[2]]
    pool = valid or list(members)
    best = max(pool, key=lambda m: m[1])
    if rng.random() < p_best:
        return best[0]
    denom = max(G - b0, EPS)
    near = [m for m in valid if (best[1] - m[1]) / denom <= band]
    return rng.choice(near)[0] if near else best[0]


# ---------------------------------------------------------------- nodes
@dataclass
class Attempt:
    id: int
    fid: int
    plan: str = ""
    best: Optional[float] = None
    s: float = 0.0          # discounted observed successes
    E: float = 0.0          # discounted expected successes
    s0: float = 0.0         # prior (a0 * optimistic mean)
    E0: float = 0.0
    sraw: float = 0.0       # discounted successes / failures, logged only
    fraw: float = 0.0
    n: int = 0              # evaluations spent here (seed, repair and edits)
    valid: int = 0
    age: int = 0            # leaf edits evaluated here
    probe_done: bool = False
    cut: bool = False       # probe ended early by the v2 early cut
    empty: int = 0          # consecutive slots that produced no candidate
    initial: bool = False
    seed_valid: Optional[bool] = None
    data: Dict[str, Any] = field(default_factory=dict)    # controller extras


@dataclass
class Framing:
    id: int
    card: Dict[str, Any] = field(default_factory=dict)
    attempts: List[Attempt] = field(default_factory=list)
    data: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Outcome:
    valid: bool
    success: bool = False
    record: bool = False        # new global best
    age: int = 0
    h: float = 0.0              # hazard h(age) used in the update
    gap_bucket: int = 0


@dataclass
class Target:
    kind: str                   # "block" | "new_framing" | "new_attempt"
    attempt: Optional[Attempt] = None
    framing: Optional[Framing] = None       # the framing a new attempt goes under / the chosen framing
    how: str = ""               # "probe" | "exploit" | "draw" | "new"
    d: Optional[float] = None   # winning draw
    d_framing: Optional[float] = None


class Block:
    """One step on an attempt: BLOCK evaluations, one more after each success, up to BLOCK_MAX."""

    def __init__(self, tree: "Tree", a: Attempt):
        self.tree, self.a = tree, a
        self.size, self.k, self.done = tree.c["block"], 0, False

    def pending(self) -> int:
        return 0 if self.done else max(0, self.size - self.k)

    def after_edit(self, out: Outcome):
        """Call after tree.record(a, ..., "edit")."""
        c, a = self.tree.c, self.a
        self.k += 1
        if out.success and c["block_ext"]:
            self.size = min(c["block_max"], self.size + c["block_ext"])
        cut = (c["early_cut_after"] and a.valid >= c["early_cut_after"]
               and self.tree.gap(a) > c["early_cut_gap"])
        if not a.probe_done and (a.valid >= c["probe"] or a.n >= c["probe_max"] or cut):
            a.probe_done = True
            a.cut = bool(cut and a.valid < c["probe"] and a.n < c["probe_max"])
            if self.k >= 2 or a.cut:
                self.done = True

    def abort(self):
        self.done = True


class Tree:
    def __init__(self, b0: float, rng: Optional[random.Random] = None, **cfg):
        self.c: Dict[str, Any] = {**DEFAULTS, **cfg}
        self.rng = rng or random.Random()
        self.b0 = self.G = float(b0)
        self.N = 0                                  # evaluations recorded (the initial program excluded)
        self.framings: List[Framing] = []
        self.attempts: List[Attempt] = []
        self.haz = [[0, 0] for _ in AGE_BUCKETS]    # pooled own-success by age bucket: [successes, n]
        self.clo = [[0, 0] for _ in range(len(GAP_EDGES) + 1)]   # global records by gap bucket
        self.exhausted = False                      # root: dedup could not produce a new framing
        self.exhausted_N = 0
        self.last_record_N = 0
        self.rep = [0, 0]                           # seed repairs: [valid, tried]
        self.seeds = [0, 0]                         # first-try seeds: [valid, tried]

    # ------------------------------------------------------------ credibility pieces
    def h(self, k: int) -> float:
        s, n = self.haz[age_bucket(k)]
        a, b = self.c["haz_prior"]
        return (s + a) / (n + a + b)

    def pi(self, bucket: int) -> float:
        s, n = self.clo[bucket]
        a, b = self.c["clo_prior"]
        return (s + a) / (n + a + b)

    def gap(self, a: Attempt) -> float:
        if self.G - self.b0 <= EPS or a.best is None:
            return 0.0
        return max(0.0, (self.G - a.best) / (self.G - self.b0))

    def mean_rho(self, a: Attempt) -> float:
        return (a.s0 + a.s) / (a.E0 + a.E)

    def _prior_mean(self, pool: Sequence[Attempt]) -> float:
        pool = [a for a in pool if a.best is not None]
        if not pool:
            return 1.0
        return max(1.0, self.mean_rho(max(pool, key=lambda a: a.best)))

    def w(self, a: Attempt) -> float:
        """Closeness weight, capped at 1; 1 while G = b0."""
        if self.G - self.b0 <= EPS:
            return 1.0
        return min(1.0, self.pi(gap_bucket(self.gap(a))) / self.pi(0))

    def draw(self, a: Attempt) -> float:
        shape, rate = a.s0 + a.s, a.E0 + a.E
        return self.rng.gammavariate(shape, 1.0 / rate) * self.w(a)

    def prior_draw(self, pool: Sequence[Attempt]) -> float:
        m = self._prior_mean(pool)
        return self.rng.gammavariate(self.c["a0"] * m, 1.0 / self.c["a0"])

    # ------------------------------------------------------------ construction
    def add_framing(self, card: Optional[Dict[str, Any]] = None) -> Framing:
        f = Framing(len(self.framings), dict(card or {}))
        self.framings.append(f)
        return f

    def add_attempt(self, f: Framing, plan: str = "", initial: bool = False) -> Attempt:
        # A new node starts at the optimistic mean of its best sibling (best in the framing, else in the tree).
        sib = f.attempts or [x for g in self.framings for x in g.attempts]     # ties go to the older framing
        m = self._prior_mean(sib)
        a = Attempt(len(self.attempts), f.id, plan, initial=initial)
        a.s0, a.E0 = self.c["a0"] * m, self.c["a0"]
        if initial:
            a.best, a.seed_valid = self.b0, True
        self.attempts.append(a)
        f.attempts.append(a)
        return a

    def n_framing(self, f: Framing) -> int:
        return sum(a.n for a in f.attempts)

    def best_attempt(self, f: Optional[Framing] = None) -> Attempt:
        pool = f.attempts if f is not None else self.attempts
        return max(pool, key=lambda a: a.best if a.best is not None else -math.inf)

    # ------------------------------------------------------------ scheduling
    def repair_enabled(self) -> bool:
        valid, tried = self.rep
        return not (tried >= self.c["repair_min_tried"] and valid / tried < self.c["repair_min_rate"])

    def mark_exhausted(self):
        self.exhausted, self.exhausted_N = True, self.N

    def _maybe_reopen(self):
        if self.exhausted and self.N - max(self.exhausted_N, self.last_record_N) >= self.c["reopen"]:
            self.exhausted = False

    def next_target(self, no_new: bool = False) -> Target:
        """Where the next block goes. `no_new` suppresses both "new" options (used after a failed creation)."""
        c = self.c
        self._maybe_reopen()
        # 1. unfinished probes, oldest first
        for a in self.attempts:
            if not a.probe_done:
                return Target("block", a, self.framings[a.fid], "probe")
        # 1b. exploit share
        if self.rng.random() < c["exploit"]:
            a = self.best_attempt()
            return Target("block", a, self.framings[a.fid], "exploit")
        # 2. root level
        N = self.N
        opts: List[Tuple[float, str, Optional[Framing]]] = []
        if (not no_new and not self.exhausted
                and len(self.framings) < math.ceil(c["widen"] * math.sqrt(N))):
            opts.append((self.prior_draw(self.attempts), "newF", None))
        for f in self.framings:
            if f.attempts:
                opts.append((self.draw(self.best_attempt(f)), "F", f))
        if not opts:
            return Target("new_framing", how="new")
        d, kind, f = max(opts, key=lambda o: o[0])
        if kind == "newF":
            return Target("new_framing", how="new", d=d)
        d_framing = d
        # 3. framing level
        nF = self.n_framing(f)
        opts2: List[Tuple[float, str, Optional[Attempt]]] = []
        if not no_new and len(f.attempts) < math.ceil(c["widen"] * math.sqrt(nF)):
            opts2.append((self.prior_draw(f.attempts), "newA", None))
        for a in f.attempts:
            opts2.append((self.draw(a), "A", a))
        d, kind, a = max(opts2, key=lambda o: o[0])
        if kind == "newA":
            return Target("new_attempt", framing=f, how="new", d=d, d_framing=d_framing)
        return Target("block", a, f, "draw", d=d, d_framing=d_framing)

    def begin_block(self, a: Attempt) -> Block:
        return Block(self, a)

    # ------------------------------------------------------------ outcomes
    def record(self, a: Attempt, valid: bool, score: float, role: str) -> Outcome:
        """One evaluation landed on attempt `a`. role: "seed" | "repair" | "edit".

        Seeds and repairs set the attempt's best but leave its credibility alone (a seed has no
        age and no expectation). Edits update the pooled hazard, the closeness curve and (s, E)."""
        self.N += 1
        a.n += 1
        a.valid += int(bool(valid))
        a.empty = 0
        out = Outcome(valid=bool(valid))
        if role in ("seed", "repair"):
            if role == "seed":
                self.seeds[0] += int(bool(valid)); self.seeds[1] += 1
            else:
                self.rep[0] += int(bool(valid)); self.rep[1] += 1
            if valid:
                a.best = score
                a.seed_valid = True
            out.record = self._global(valid, score)
            return out
        # leaf edit
        gb = gap_bucket(self.gap(a))
        a.age += 1
        out.age, out.gap_bucket = a.age, gb
        out.success = bool(valid) and a.best is not None and score > a.best
        out.record = self._global(valid, score)
        out.h = self.h(a.age)                                       # before this evaluation is pooled
        hb = self.haz[age_bucket(a.age)]
        hb[0] += int(out.success); hb[1] += 1
        self.clo[gb][0] += int(out.record); self.clo[gb][1] += 1
        g = self.c["gamma"]
        a.s = g * a.s + out.success
        a.E = g * a.E + out.h
        a.sraw = g * a.sraw + out.success
        a.fraw = g * a.fraw + (not out.success)
        if out.success:
            a.best = score
        return out

    def _global(self, valid: bool, score: float) -> bool:
        if valid and score > self.G + EPS:
            self.G = score
            self.last_record_N = self.N
            return True
        return False

    def seed_done(self, a: Attempt):
        """The seed (and its repair, if any) is over. A seed that never produced a valid program
        leaves the attempt with a copy of the global best and its plan (spec section 6.4)."""
        if a.best is None:
            a.best, a.seed_valid = self.G, False

    def note_empty(self, a: Attempt):
        """A slot on `a` produced no candidate (no evaluation spent). Enough in a row end its probe."""
        a.empty += 1
        if a.empty >= self.c["empty_dead"]:
            a.probe_done = True

    # ------------------------------------------------------------ introspection
    def node_stats(self, a: Attempt) -> Dict[str, Any]:
        f = self.framings[a.fid]
        return {
            "attempt": {"id": a.id, "s": round(a.s, 4), "f": round(a.fraw, 4), "E": round(a.E, 4),
                        "n": a.n, "valid": a.valid, "age": a.age, "best": a.best,
                        "s0": round(a.s0, 4), "rho": round(self.mean_rho(a), 4), "w": round(self.w(a), 4),
                        "gap": round(self.gap(a), 4), "probe_done": a.probe_done, "cut": a.cut},
            "framing": {"id": f.id, "n": self.n_framing(f), "attempts": len(f.attempts),
                        "best": self.best_attempt(f).best},
            "root": {"N": self.N, "G": self.G, "framings": len(self.framings),
                     "attempts": len(self.attempts), "exhausted": self.exhausted},
        }
