"""BANYAN scheduler and credibility state (design/round7/BANYAN.md, sections 3-7 and 16).

Pure Python: no skydiscover, no LLM, no I/O. The controller (banyan.py) owns generation and evaluation; this
module only decides WHERE the next evaluations go, updates the credibility of the nodes they land on, and
re-roots the tree when a branch sets a global record. analysis/calib/test_banyan_core.py drives it with the
offline simulator's world and checks that it makes the same decisions as banyan_sim.run_banyan.

Tree: the TRUNK is the edit loop on the incumbent (the attempt that holds the global best G). Framings hold
BRANCHES (attempts): each is a graft on the incumbent and stores anchor_G, the incumbent score when it was
grafted. Credibility is the chance that an evaluation in a node raises the incumbent:
  success (trunk)  x > G + eps          success (branch)  x > anchor_G + eps   (and x beats the node's own best)
  s <- gamma*s + success,  E <- gamma*E + h(age),  rho ~ Gamma(a0 + s, a0 + E)
with the pooled age hazard h(k) (the hazard counts the evaluation just made, as banyan_sim does). Every draw is
the plain ratio rho: no hazard factor is applied to a draw, so the trunk (always the oldest node) is not marked
down for its age.

Root draw, two stages (spec 5.2):
  stage 1  trunk rho_T against ONE pooled rho_B ~ Gamma(a0 + sum s, a0 + sum E) over every open branch since the
           last re-root (prior Gamma(a0, a0) with none); the trunk wins ties;
  stage 2  otherwise the argmax over the open framings and, while K < ceil(c*sqrt(N)) and the root is not
           exhausted, a "new framing" drawn from the neutral prior Gamma(a0, a0); then the same inside the
           chosen framing, where a "new attempt" competes while K_F < ceil(c*sqrt(n_F)).

A record in a branch RE-ROOTS: that attempt becomes the trunk, every other open attempt is frozen, frozen
branches whose posterior says productive (P(rho > 1) >= 0.7 with >= probe valid evaluations) are queued for
re-graft, and the exhaustion flag resets. A record in the trunk moves G and every open branch's anchor_G.

Constants are module constants, identical on all five tasks (handout 2.3). Options that exist only so the
simulator equivalence test can run the sim's own behaviour are marked (sim).
"""
import math
import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from credtree_core import AGE_BUCKETS, age_bucket, pick_parent   # noqa: F401  (pick_parent re-exported)

# ---------------------------------------------------------------- constants (spec 4-7)
GAMMA = 0.8
BLOCK, BLOCK_EXT, BLOCK_MAX = 2, 1, 6
WIDEN = 1.0                 # a new child is on offer while K < ceil(WIDEN * sqrt(n)), both levels
A0 = 2.0                    # prior weight, in expected successes
PROBE_VALID, PROBE_MAX = 3, 5
CUT_GAP = 0.1               # a probe ends early when the attempt is > 0.1 (normalised) below anchor_G
N_START = 2                 # framings generated at the start, besides framing 0
HAZ_PRIOR = (1.0, 4.0)
MAX_FRAMINGS = 60
REGRAFT_P = 0.7             # a frozen branch is re-grafted when P(rho > 1) is at least this
LABEL_PRODUCTIVE, LABEL_SATURATED, SATURATED_N = 0.7, 0.2, 8       # spec 6
DEDUP_RETRIES = 5
REOPEN_AFTER = 60           # evaluations with no global record after which the root reopens
EMPTY_SLOTS_DEAD = 3
EPS = 1e-12                 # floor of the success / record margin (banyan_sim's 1e-12)

DEFAULTS: Dict[str, Any] = dict(
    gamma=GAMMA, block=BLOCK, block_ext=BLOCK_EXT, block_max=BLOCK_MAX, widen=WIDEN, a0=A0, probe=PROBE_VALID,
    probe_max=PROBE_MAX, cut=CUT_GAP, haz_prior=HAZ_PRIOR, max_framings=MAX_FRAMINGS, regraft=True,
    regraft_p=REGRAFT_P, reopen=REOPEN_AFTER, empty_dead=EMPTY_SLOTS_DEAD, two_stage=True, anchor_follow=True,
    # ablations (env in banyan.py; the spec is the default):
    reroot=True,                # BANYAN_P2=0 sets False (with two_stage=False): a branch record moves G and every open
                                # anchor but freezes nothing and leaves the trunk where it is (a standing tree)
    alloc="thompson",           # BANYAN_ALLOC=random: every draw is uniform on [0, 1), so allocation ignores credibility
    # (sim) options whose default here is the BANYAN spec; the equivalence test also runs banyan_sim's own value
    framing_draw="best",        # a framing's draw: its best open attempt's draw ("best") | max over its attempts' draws ("max")
    regraft_trunk=False,        # re-graft the old trunk's framing at a re-root (spec: frozen BRANCHES only)
    prob="exact",               # P(rho > 1): exact incomplete gamma | Monte Carlo with 32 draws on the shared rng ("mc")
)


def constants() -> Dict[str, Any]:
    """The constants in force, for the startup log."""
    return {**DEFAULTS, "n_start": N_START, "dedup_retries": DEDUP_RETRIES, "label_productive": LABEL_PRODUCTIVE,
            "label_saturated": LABEL_SATURATED, "saturated_n": SATURATED_N}


# ---------------------------------------------------------------- P(rho > 1)
def gamma_q(a: float, x: float) -> float:
    """Regularised upper incomplete gamma Q(a, x) = P(X > x) for X ~ Gamma(shape a, scale 1)."""
    if x <= 0.0:
        return 1.0
    lg = math.lgamma(a)
    if x < a + 1.0:                                   # series for P(a, x)
        term = total = 1.0 / a
        n = a
        for _ in range(500):
            n += 1.0
            term *= x / n
            total += term
            if abs(term) < abs(total) * 1e-15:
                break
        return max(0.0, 1.0 - total * math.exp(-x + a * math.log(x) - lg))
    tiny = 1e-300                                     # continued fraction for Q(a, x) (modified Lentz)
    b = x + 1.0 - a
    c = 1.0 / tiny
    d = 1.0 / b
    h = d
    for i in range(1, 500):
        an = -i * (i - a)
        b += 2.0
        d = an * d + b
        d = tiny if abs(d) < tiny else d
        c = b + an / c
        c = tiny if abs(c) < tiny else c
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < 1e-15:
            break
    return min(1.0, math.exp(-x + a * math.log(x) - lg) * h)


def p_rho_gt1(shape: float, rate: float) -> float:
    """P(rho > 1) for rho ~ Gamma(shape, rate)."""
    return gamma_q(shape, rate)


# ---------------------------------------------------------------- nodes
@dataclass
class Attempt:
    id: int
    fid: int
    plan: str = ""
    best: Optional[float] = None
    s: float = 0.0          # discounted successes
    E: float = 0.0          # discounted expected successes
    n: int = 0              # evaluations spent here (seed and edits)
    valid: int = 0
    age: int = 0            # leaf edits evaluated here
    probe_done: bool = False
    cut: bool = False       # probe ended early by the cut
    empty: int = 0          # consecutive slots that produced no candidate
    initial: bool = False
    anchor: float = 0.0     # anchor_G: the incumbent score this branch is judged against
    frozen: bool = False
    frozen_at: int = 0      # re-root number that froze it
    seed_valid: Optional[bool] = None
    data: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Framing:
    id: int
    card: Dict[str, Any] = field(default_factory=dict)
    attempts: List[Attempt] = field(default_factory=list)
    regraft: bool = False   # queued for a re-graft on the new incumbent
    closed: bool = False    # no live attempt: out of the root draw
    data: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Outcome:
    valid: bool
    success: bool = False
    record: bool = False        # x > G + margin: a new incumbent
    rerooted: bool = False      # the record came from a branch and moved the root
    age: int = 0
    h: float = 0.0              # hazard used in the update
    bar: float = 0.0            # what the node was judged against (G for the trunk, anchor_G for a branch)
    was_trunk: bool = False
    frozen: int = 0             # open attempts frozen by this re-root
    regrafts: List[int] = field(default_factory=list)   # framings queued for re-graft by this re-root


@dataclass
class Target:
    kind: str                   # "block" | "new_framing" | "new_attempt" | "regraft"
    attempt: Optional[Attempt] = None
    framing: Optional[Framing] = None
    how: str = ""               # "probe" | "trunk" | "branch" | "new" | "regraft" | "fallback"
    d: Optional[float] = None   # winning draw
    d_trunk: Optional[float] = None
    d_branch: Optional[float] = None
    stage1: str = ""            # "trunk" | "branch" | "" (no draw: probe / regraft)


class Block:
    """One step on an attempt: BLOCK evaluations, one more after each success, up to BLOCK_MAX."""

    def __init__(self, tree: "Tree", a: Attempt):
        self.tree, self.a = tree, a
        self.size, self.k, self.done = tree.c["block"], 0, False

    def pending(self) -> int:
        return 0 if self.done or self.a.frozen else max(0, self.size - self.k)

    def after_edit(self, out: Outcome):
        """Call after tree.record(a, ..., "edit")."""
        t, c, a = self.tree, self.tree.c, self.a
        self.k += 1
        if out.success and c["block_ext"]:
            self.size = min(c["block_max"], self.size + c["block_ext"])
        if not a.probe_done:
            denom = t.G - t.b0
            cut = a.valid >= 1 and denom > 0 and a.best is not None and (a.anchor - a.best) > c["cut"] * denom
            if a.valid >= c["probe"] or a.n >= c["probe_max"] or cut:
                a.probe_done = True
                a.cut = bool(cut and a.valid < c["probe"] and a.n < c["probe_max"])
                if self.k >= 2:
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
        self.haz = [[0, 0] for _ in AGE_BUCKETS]    # pooled success by own-age bucket: [successes, n]
        self.trunk: Optional[Attempt] = None
        self.eps = 0.0                              # noise margin (absolute), set by the controller; see margin()
        self.exhausted = False
        self.exhausted_N = 0
        self.last_record_N = 0
        self.reroot_N = 0                           # N at the last re-root
        self.stats = {"reroots": 0, "regrafts": 0, "trunk_evals": 0, "branch_evals": 0, "stage1_trunk": 0,
                      "stage1_branch": 0}
        self.last_draw: Dict[str, Any] = {}

    # ------------------------------------------------------------ credibility pieces
    def margin(self) -> float:
        """A success or a record must beat its bar by this much: the floor EPS plus the measured noise."""
        return EPS + max(0.0, self.eps)

    def h(self, k: int) -> float:
        s, n = self.haz[age_bucket(max(k, 1))]
        a, b = self.c["haz_prior"]
        return (s + a) / (n + a + b)

    def mean_rho(self, a: Attempt) -> float:
        return (self.c["a0"] + a.s) / (self.c["a0"] + a.E)

    def p_gt1(self, a: Attempt) -> float:
        return p_rho_gt1(self.c["a0"] + a.s, self.c["a0"] + a.E)

    def label(self, a: Attempt) -> str:
        """Spec 6: untried (probe not finished) | productive (P(rho > 1) >= 0.7) | saturated (n >= 8 valid and
        P(rho > 1) <= 0.2) | "" (no label)."""
        if not a.probe_done and not a.initial:
            return "untried"
        p = self.p_gt1(a)
        if p >= LABEL_PRODUCTIVE:
            return "productive"
        if a.valid >= SATURATED_N and p <= LABEL_SATURATED:
            return "saturated"
        return ""

    def draw(self, a: Attempt) -> float:
        if self.c["alloc"] == "random":
            return self.rng.random()
        return self.rng.gammavariate(self.c["a0"] + a.s, 1 / (self.c["a0"] + a.E))

    def prior_draw(self) -> float:
        """The neutral prior of a node with no data: mean 1, weight a0."""
        if self.c["alloc"] == "random":
            return self.rng.random()
        return self.rng.gammavariate(self.c["a0"] * 1.0, 1 / self.c["a0"])

    def _p_gt1_mc(self, a: Attempt, n: int = 32) -> float:
        sh, rt = self.c["a0"] + a.s, self.c["a0"] + a.E
        return sum(self.rng.gammavariate(sh, 1 / rt) > 1 for _ in range(n)) / n

    # ------------------------------------------------------------ construction
    def add_framing(self, card: Optional[Dict[str, Any]] = None) -> Framing:
        f = Framing(len(self.framings), dict(card or {}))
        self.framings.append(f)
        return f

    def add_attempt(self, f: Framing, plan: str = "", initial: bool = False) -> Attempt:
        a = Attempt(len(self.attempts), f.id, plan, anchor=self.G, initial=initial, probe_done=initial)
        if initial:
            a.best, a.seed_valid = self.b0, True
            if self.trunk is None:
                self.trunk = a
        self.attempts.append(a)
        f.attempts.append(a)
        return a

    def n_framing(self, f: Framing) -> int:
        return sum(a.n for a in f.attempts)

    def best_attempt(self, f: Optional[Framing] = None, open_only: bool = False) -> Optional[Attempt]:
        pool = self.open_attempts(f) if open_only else (f.attempts if f is not None else self.attempts)
        return max(pool, key=lambda a: a.best if a.best is not None else -math.inf) if pool else None

    def open_attempts(self, f: Optional[Framing] = None) -> List[Attempt]:
        """Branches that can still take blocks: not frozen and not the trunk."""
        return [a for a in self.attempts if not a.frozen and a is not self.trunk and (f is None or a.fid == f.id)]

    def live_framings(self) -> List[Framing]:
        return [f for f in self.framings if not f.closed]

    def close_framings(self):
        """A framing with no live attempt (every attempt frozen, none the trunk) and no queued re-graft leaves the draw."""
        for f in self.framings:
            f.closed = not any((not a.frozen) or a is self.trunk for a in f.attempts) and not f.regraft

    # ------------------------------------------------------------ scheduling
    def mark_exhausted(self):
        self.exhausted, self.exhausted_N = True, self.N

    def _maybe_reopen(self):
        if self.exhausted and self.N - max(self.exhausted_N, self.last_record_N) >= self.c["reopen"]:
            self.exhausted = False

    def next_target(self, no_new: bool = False) -> Target:
        """Where the next step goes. `no_new` suppresses both "new" options (used after a failed creation)."""
        c = self.c
        self._maybe_reopen()
        # 0. queued re-grafts, then unfinished probes (oldest first)
        for f in self.framings:
            if f.regraft:
                f.regraft = False
                f.closed = False
                self.stats["regrafts"] += 1
                return Target("regraft", framing=f, how="regraft")
        for a in self.attempts:
            if not a.probe_done and not a.frozen:
                return Target("block", a, self.framings[a.fid], "probe")
        t = self.trunk
        N = self.N
        live = self.live_framings()
        # 1. stage 1: trunk or branch out
        d_t = d_b = None
        if c["two_stage"]:
            ob = self.open_attempts()
            S, E = sum(a.s for a in ob), sum(a.E for a in ob)
            d_b = self.rng.random() if c["alloc"] == "random" else self.rng.gammavariate(c["a0"] + S, 1 / (c["a0"] + E))
            d_t = self.draw(t)
            self.last_draw = {"rho_T": d_t, "rho_B": d_b, "open_branches": len(ob), "S": S, "E": E}
            if d_t >= d_b:
                self.stats["stage1_trunk"] += 1
                return Target("block", t, self.framings[t.fid], "trunk", d=d_t, d_trunk=d_t, d_branch=d_b,
                              stage1="trunk")
            self.stats["stage1_branch"] += 1
        opts: List[Tuple[float, str, Any]] = [] if c["two_stage"] else [(self.draw(t), "T", t)]
        # 2. stage 2: which framing (or a new one)
        if (not no_new and not self.exhausted
                and len(live) < min(c["max_framings"], math.ceil(c["widen"] * math.sqrt(N)))):
            opts.append((self.prior_draw(), "newF", None))
        for f in live:
            oa = self.open_attempts(f)
            if not oa:
                d = self.prior_draw()
            elif c["framing_draw"] == "max":
                d = max(self.draw(a) for a in oa)
            else:
                d = self.draw(self.best_attempt(f, open_only=True))
            opts.append((d, "F", f))
        if not opts:
            return Target("block", t, self.framings[t.fid], "fallback", d_trunk=d_t, d_branch=d_b)
        d, kind, x = max(opts, key=lambda o: o[0])
        if kind == "T":
            return Target("block", t, self.framings[t.fid], "trunk", d=d, d_trunk=d, stage1="trunk")
        if kind == "newF":
            return Target("new_framing", how="new", d=d, d_trunk=d_t, d_branch=d_b, stage1="branch")
        f = x
        d_framing = d
        # 3. inside the framing: an open attempt, or a new attempt
        oa = self.open_attempts(f)
        nF = sum(a.n for a in oa)
        o2: List[Tuple[float, str, Any]] = [(self.draw(a), "A", a) for a in oa]
        if (not oa or len(oa) < math.ceil(c["widen"] * math.sqrt(nF))) and not no_new:
            o2.append((self.prior_draw(), "newA", f))
        if not o2:
            return Target("block", t, self.framings[t.fid], "fallback", d_trunk=d_t, d_branch=d_b)
        d, kind, y = max(o2, key=lambda o: o[0])
        if kind == "newA":
            return Target("new_attempt", framing=f, how="new", d=d, d_trunk=d_t, d_branch=d_b, stage1="branch")
        return Target("block", y, f, "branch", d=d, d_trunk=d_t, d_branch=d_b, stage1="branch")

    def begin_block(self, a: Attempt) -> Block:
        return Block(self, a)

    # ------------------------------------------------------------ outcomes
    def record(self, a: Attempt, valid: bool, score: float, role: str) -> Outcome:
        """One evaluation landed on attempt `a`. role: "seed" | "edit".

        A seed (the graft that opens a branch) sets the attempt's best and counts as an evaluation but leaves its
        credibility alone: it has no age and no expectation. An edit updates the pooled hazard and (s, E). A
        record in a branch re-roots; a record in the trunk moves G and every open branch's anchor_G."""
        c = self.c
        self.N += 1
        a.n += 1
        a.valid += int(bool(valid))
        a.empty = 0
        out = Outcome(valid=bool(valid))
        margin = self.margin()
        was_trunk = a is self.trunk
        out.was_trunk = was_trunk
        bar = self.G if was_trunk else a.anchor
        out.bar = bar
        if role == "seed":
            if valid:
                a.best = score
                a.seed_valid = True
            out.record = bool(valid) and score > self.G + margin
            if out.record:
                self._new_record(score)
                if c["reroot"]:
                    self._reroot(a, out)
                else:
                    self._follow()
            return out
        # leaf edit
        a.age += 1
        out.age = a.age
        prev = a.best
        own = bool(valid) and prev is not None and score > prev
        out.success = bool(own and score > bar + margin)
        out.record = bool(valid) and score > self.G + margin
        if out.record:
            self._new_record(score)
        self.stats["trunk_evals" if was_trunk else "branch_evals"] += 1
        hb = self.haz[age_bucket(a.age)]
        hb[0] += int(out.success)
        hb[1] += 1
        out.h = self.h(a.age)                       # after pooling, as banyan_sim
        a.s = c["gamma"] * a.s + out.success
        a.E = c["gamma"] * a.E + out.h
        if own:
            a.best = score
        if out.record and not was_trunk and c["reroot"]:
            self._reroot(a, out)
        elif out.record and c["anchor_follow"]:
            self._follow()
        return out

    def _follow(self):
        """Every open node's anchor_G rises to the new G (spec 4: success means "raised the incumbent")."""
        for b in self.attempts:
            if not b.frozen:
                b.anchor = self.G

    def _new_record(self, score: float):
        self.G = score
        self.last_record_N = self.N

    def _reroot(self, a: Attempt, out: Outcome):
        c = self.c
        old = self.trunk
        self.stats["reroots"] += 1
        self.reroot_N = self.N
        for b in self.attempts:
            if b is a or b.frozen:
                continue
            b.frozen, b.frozen_at = True, self.stats["reroots"]
            out.frozen += 1
            if not c["regraft"] or (b is old and not c["regraft_trunk"]) or b.valid < c["probe"]:
                continue
            if c["prob"] == "mc":
                ok = self.rng.gammavariate(c["a0"] + b.s, 1 / (c["a0"] + b.E)) > 0 and \
                    self._p_gt1_mc(b) >= c["regraft_p"]
            else:
                ok = self.p_gt1(b) >= c["regraft_p"]
            if ok and not self.framings[b.fid].regraft:
                self.framings[b.fid].regraft = True
                out.regrafts.append(b.fid)
        a.anchor = self.G
        self.trunk = a
        out.rerooted = True
        self.exhausted = False                      # dedup counters and exhaustion reset (spec 7)

    def seed_done(self, a: Attempt):
        """The seed is over. A seed that never produced a valid program leaves the attempt with a copy of the
        incumbent and its plan."""
        if a.best is None:
            a.best, a.seed_valid = self.G, False

    def note_empty(self, a: Attempt):
        """A slot on `a` produced no candidate (no evaluation spent). Enough in a row end its probe."""
        a.empty += 1
        if a.empty >= self.c["empty_dead"]:
            a.probe_done = True

    # ------------------------------------------------------------ resume
    @classmethod
    def restore(cls, dump: Dict[str, Any], rows: Sequence[Dict[str, Any]], events: Sequence[Dict[str, Any]],
                rng: Optional[random.Random] = None) -> "Tree":
        """Rebuild the tree of a run that died, from the controller's per-step dump (banyan_tree.json, written after
        every step), its per-evaluation log (tree_stats.jsonl) and its event log (tree_events.jsonl). The nodes'
        programs and edit histories are left empty for the controller to refill from its database.
        Not restored: the rng state (re-seeded by the caller) and the last stage-1 draw (log only)."""
        t = cls(dump["b0"], rng=rng)
        t.G, t.N, t.eps, t.exhausted = dump["G"], dump["N"], dump["eps"], dump["exhausted"]
        t.haz = [list(h) for h in dump["haz"]]
        t.stats = dict(dump["stats"])
        if t.exhausted:
            t.exhausted_N = max((e["N"] for e in events if e["event"] == "exhausted"), default=0)
        t.last_record_N = max((r["nodes"]["root"]["N"] for r in rows if r["global_record"]), default=0)
        t.reroot_N = max((r["nodes"]["root"]["N"] for r in rows if r["rerooted"]), default=0)
        pending: List[int] = []                         # re-grafts queued by a re-root and not yet taken
        for e in events:
            if e["event"] == "reroot":
                pending += [fid for fid in e.get("regrafts") or [] if fid not in pending]
            elif e["event"] in ("regraft", "regraft_dup") and e.get("framing") in pending:
                pending.remove(e["framing"])
        for fd in dump["framings"]:
            f = Framing(fd["id"], dict(fd["card"]), closed=fd["closed"], regraft=fd["id"] in pending)
            t.framings.append(f)
            for ad in fd["attempts"]:
                a = Attempt(ad["id"], f.id, ad["plan"], best=ad["best"], s=ad["s"], E=ad["E"], n=ad["n"],
                            valid=ad["valid"], age=ad["age"], probe_done=ad["probe_done"], cut=ad["cut"],
                            initial=ad["id"] == 0, anchor=ad["anchor"], frozen=ad["frozen"], frozen_at=ad["frozen_at"],
                            seed_valid=ad["seed_valid"], data={"programs": [], "history": []})
                f.attempts.append(a)
                t.attempts.append(a)
        t.attempts.sort(key=lambda a: a.id)
        assert [a.id for a in t.attempts] == list(range(len(t.attempts))), "attempt ids are not 0..K-1"
        t.trunk = t.attempts[dump["trunk"]]
        return t

    # ------------------------------------------------------------ introspection
    def labels(self) -> Dict[str, int]:
        """Label counts over the live nodes (open branches and the trunk), for the pilot log."""
        out = {"productive": 0, "saturated": 0, "untried": 0, "unlabelled": 0}
        for a in self.attempts:
            if a.frozen:
                continue
            out[self.label(a) or "unlabelled"] += 1
        return out

    def node_stats(self, a: Attempt) -> Dict[str, Any]:
        f = self.framings[a.fid]
        return {
            "attempt": {"id": a.id, "s": round(a.s, 4), "E": round(a.E, 4), "n": a.n, "valid": a.valid,
                        "age": a.age, "best": a.best, "anchor": a.anchor, "rho": round(self.mean_rho(a), 4),
                        "p_gt1": round(self.p_gt1(a), 4), "label": self.label(a), "probe_done": a.probe_done,
                        "cut": a.cut, "frozen": a.frozen, "trunk": a is self.trunk},
            "framing": {"id": f.id, "n": self.n_framing(f), "attempts": len(f.attempts)},
            "root": {"N": self.N, "G": self.G, "framings": len(self.framings), "live_framings": len(self.live_framings()),
                     "attempts": len(self.attempts), "open_branches": len(self.open_attempts()),
                     "exhausted": self.exhausted, "reroots": self.stats["reroots"], "eps": self.eps},
        }
