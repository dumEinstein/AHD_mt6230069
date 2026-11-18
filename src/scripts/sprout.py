"""SPROUT: strategy portfolio with protected one-shot trials (arm 4).

Specification: design/round2/design_combined.md (section 3 is the algorithm; R1-R9 below
refer to its rules). Opt-in from run_capped.py with SPROUT=1 and `search.type: sprout`.

One step of the loop:
  1. birth, if the champion (incumbent) is exhausted and no challenger is alive   (R1, R2)
  2. pick two jobs: lane A refines the champion, lane B a challenger;            (R5)
     while a newborn incubates both lanes belong to it                          (R3)
  3. generate both concurrently (LLM calls are free); every candidate passes
     the pre-evaluation filter, failures are retried with the reason            (3.5)
  4. evaluate the survivors together (each costs one of the 500 evaluations)
  5. record per lineage, then gate the newborn and close stale challengers      (R4, R6, R7, R9)

Within a lineage, parent and context sampling are AdaEvolve's own components
(UnifiedArchive + AdaptiveState + AdaEvolveContextBuilder), one instance per lineage
and nothing shared across lineages: no migration, no global-top context, no seeding
by copy (R8). Every program is stored in the database whether or not it stays in its
lineage's archive, so validation selection sees all of them.

Hyper-parameters are module constants: identical on all five tasks (handout 2.3).
"""
import ast
import asyncio
import collections
import hashlib
import json
import logging
import math
import os
import re
import statistics
import time
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from skydiscover.optimize.config import AdaEvolveDatabaseConfig, _DB_CONFIG_BY_TYPE
from skydiscover.optimize.context_builder.adaevolve import AdaEvolveContextBuilder
from skydiscover.optimize.search.adaevolve.adaptation import AdaptiveState
from skydiscover.optimize.search.adaevolve.archive.diversity import create_diversity_strategy
from skydiscover.optimize.search.adaevolve.archive.unified_archive import ArchiveConfig, UnifiedArchive
from skydiscover.optimize.search.adaevolve.database import EXPLOIT_LABEL, EXPLORE_LABEL
from skydiscover.optimize.search.base_database import Program, ProgramDatabase
from skydiscover.optimize.search.default_discovery_controller import DiscoveryController
from skydiscover.optimize.search.registry import register_controller, register_database
from skydiscover.optimize.search.utils.discovery_utils import SerializableResult
from skydiscover.optimize.utils.code_utils import extract_diffs, format_diff_summary

logger = logging.getLogger("sprout")

# ---------------------------------------------------------------- hyper-parameters (3.6)
EVAL_CAP = int(os.environ.get("EVAL_CAP", "500"))
BATCH = 2                 # evaluations per step (two lanes)
FAST_FAIL = 4             # close a newborn with nothing admissible after this many
# A born lineage gets TRIAL own evaluations to take the global best, then closes. In the
# seven round-2 runs every one of the overtakes happened within a newborn's first four
# evaluations, and 0 of 572 evaluations spent on a lineage at own index >= 5 while it was
# still a challenger ever took the best. SPROUT_TRIAL=8 is the A-TRIAL ablation;
# SPROUT_TRIAL=legacy restores the round-2 gate and challenger tier byte for byte.
_T = os.environ.get("SPROUT_TRIAL", "4")
LEGACY = _T == "legacy"
TRIAL = FAST_FAIL if _T == "4" else (int(_T) if not LEGACY else 8)
B_INC = 8                 # legacy only: protected incubation evaluations
R_NEAR = 0.95             # legacy only: promote outright at this ratio to the best
R_FLOOR0, R_FLOOR_SLOPE = 0.1, 0.6     # legacy only: gate floor r >= 0.1 + 0.6 p
GAINS_MIN = 1             # legacy only
THETA = 0.005             # meaningful gain = THETA * delta_star
L_MIN = int(os.environ.get("SPROUT_L_MIN", "12"))   # exhaustion floor; override for pilots ONLY
L_MAX = int(os.environ.get("SPROUT_L_MAX", "48"))   # exhaustion ceiling; override for pilots ONLY
ALPHA = 0.25
PRIOR_A, PRIOR_B = 1, 19                # Beta(1,19) prior on the champion's gain rate
C_MAX = 2                 # legacy only: live challengers
N_MAX = int(os.environ.get("SPROUT_N_MAX", "20"))   # births per run (A0 ablation: 0)
P_NO_NEW, P_POLISH = 0.85, 0.95
N_WRITE, N_DRAFT = 3, 3
COPY_JACCARD = 0.9        # verbatim-copy screen for one-shot drafts (5-gram Jaccard)
NUM_CONTEXT = 2
HYBRID_EVERY, HYBRID_MIN_RATIO = 3, 0.9

# ---------------------------------------------------------------- SIFT (opt-in: SIFT=1)
# Surrogate-Informed Filtering of Trials. The arrival rate of new global bests per
# *admissible* evaluation is flat against every search lever we have measured; the number
# of admissible evaluations is not. 3-57% of a run's evaluations score zero, and on the
# low-admissibility tasks those zeros are predictable from the code alone.
#
# So: after every evaluation, record (features(program), score > 0). Refit a logistic
# regression on the run's own spent evaluations, calibrate a rejection threshold on a
# held-out tail of them, and refuse to spend an evaluation on a candidate the model
# condemns -- regenerating instead, which costs only a (free) LLM call.
#
# Nothing is task-specific: the code is identical on all five tasks and learns each
# task's failure modes from its own run. On tasks where admissibility is already high,
# or where no threshold meets the precision target, the gate never fires.
SIFT = os.environ.get("SIFT") == "1"
SIFT_TARGET = float(os.environ.get("SIFT_TARGET", "0.98"))   # held-out precision required
SIFT_MAXREJ = float(os.environ.get("SIFT_MAXREJ", "0.25"))   # never condemn more than this share
SIFT_MIN_TRAIN, SIFT_REFIT = 80, 25

# ---------------------------------------------------------------- HARVEST (opt-in: HARVEST=1)
# Three mechanisms sharing one substrate, each independently ablatable.
#
#  M3  structural descriptor. Programs are clustered by binary AST features (attributes
#      called, names bound, imports). Measured: this breaks the dominant algorithm family
#      from 95% of programs down to a largest cluster of 18-26%, explains only 5-22% of
#      score variance (so it is not a restatement of fitness), and under-represented
#      regions are not worse than the dominant one.
#
#  M2  region yield. Each cluster's distinct-outcome rate over a recent window, where an
#      "outcome" is the per-instance score vector from the evaluator feedback. Measured
#      forward, with no leakage: the next program drawn from a top-quartile-yield cluster
#      sets a new global best 1.93% of the time against 0.90% from the bottom quartile
#      (11x on cloudcast and prism). This is the signal fitness cannot see -- prism's
#      largest cluster has the highest mean score and a 1.1% yield.
#
#  M1  operator credit. Per-operator windows of percentile ranks against a shared
#      reference pool (so operators applied at very different rates stay comparable),
#      decayed so top ranks dominate (scale-free but extreme-sensitive), and divided by
#      the operator's mean cost in evaluations.
HARVEST = os.environ.get("HARVEST", "0") in ("1", "2")   # 1 = as first shipped, 2 = corrected M2
HV_YIELD = HARVEST and os.environ.get("HARVEST_YIELD", "1") == "1"   # M2
HV_OPS = HARVEST and os.environ.get("HARVEST_OPS", "0") == "1"       # M1, default OFF
# M1's only call site arbitrates repair-vs-oneshot, a branch reached when a lineage is
# still scoring zero. Pooled over 7984 evaluations in 16 runs that was refine 96.9%,
# oneshot 3.0%, repair 0.1% -- eight repair calls in all. The bandit was credited and
# logged but governed 3.1% of decisions, so it is off by default and kept for ablation.
HV_BIRTH = HARVEST and os.environ.get("HARVEST_BIRTH", "1") == "1"   # M3 birth targeting
HV_K = int(os.environ.get("HARVEST_K", "12"))        # clusters
HV_MIN = int(os.environ.get("HARVEST_MIN", "60"))    # programs before the first fit
HV_REFIT = 50            # refit cadence, in evaluations
HV_WINDOW = 40           # per-region yield window, in programs
HV_PRIOR = 8.0           # pseudo-observations pulling a thin region to the pooled rate
HV_BIAS = 0.7            # probability that parent choice is yield-biased rather than stock
HV_OPWIN = 30            # per-operator window of percentile ranks
HV_DECAY = 0.4           # rank decay: ->0 extreme-weighted, ->1 mean-like
HV_C = 0.5               # UCB exploration scale
HV2 = HARVEST and os.environ.get("HARVEST", "0") == "2"   # corrected M2 (see design/round5)
HV_BETA = float(os.environ.get("HARVEST_BETA", "1"))      # yield exponent on region choice
# beta=1 makes the region tilt equal to the yield ratio itself. Removing the
# per-program (occupancy) weighting is already most of the correction; beta>1 was
# measured at a 9.8x tilt on a 2.4x signal, which over-concentrates.
HV_TOPFRAC = 0.5         # within the chosen cluster, sample from this top score fraction
ATTEMPTS = 3              # LLM attempts per job before it yields no candidate
UNIMPLEMENTABLE = 5       # consecutive empty jobs that close a newborn
MAX_LEN = 60000
CAP_REFINE, CAP_OTHER = 30000, 20000   # prompt caps in characters
LEDGER_CHARS, NOTES_CHARS, FEEDBACK_CHARS = 3000, 1200, 2000
COPY_REL = 1e-9

INCUBATING, INCUMBENT, CHALLENGER, CLOSED = "INCUBATING", "INCUMBENT", "CHALLENGER", "CLOSED"

_HERE = os.path.dirname(os.path.abspath(__file__))
PROMPTS = os.environ.get("SPROUT_PROMPTS", os.path.join(os.path.dirname(_HERE), "prompts", "v4"))


def _tpl(name: str) -> str:
    with open(os.path.join(PROMPTS, f"sprout_{name}.txt")) as fh:
        return fh.read()


def _score(metrics: Optional[Dict[str, Any]]) -> float:
    try:
        v = float((metrics or {}).get("combined_score", 0.0))
    except (TypeError, ValueError):
        return 0.0
    return v if v == v else 0.0


def _wall_seconds() -> float:
    """Task wall-clock limit from LIMIT (run_task.sh passes e.g. 5h, 10h, 16m)."""
    raw = os.environ.get("SPROUT_WALL") or os.environ.get("LIMIT") or "5h"
    m = re.fullmatch(r"\s*([\d.]+)\s*([smhd]?)\s*", raw)
    if not m:
        return 5 * 3600.0
    return float(m.group(1)) * {"": 1, "s": 1, "m": 60, "h": 3600, "d": 86400}[m.group(2)]


# ---------------------------------------------------------------- code utilities
_BLOCK_START, _BLOCK_END = "# EVOLVE-BLOCK-START", "# EVOLVE-BLOCK-END"


def _split_block(code: str) -> Optional[Tuple[str, str, str]]:
    i, j = code.find(_BLOCK_START), code.find(_BLOCK_END)
    if i < 0 or j < 0 or j < i:
        return None
    i_end = code.find("\n", i)
    i_end = len(code) if i_end < 0 else i_end + 1
    return code[:i_end], code[i_end:j], code[j:]


def _top_level_names(tree: ast.Module) -> set:
    names = set()
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for t in targets:
                for n in ast.walk(t):
                    if isinstance(n, ast.Name):
                        names.add(n.id)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            for a in node.names:
                names.add((a.asname or a.name).split(".")[0])
    return names


def _norm_hash(code: str) -> str:
    """Hash of the AST, so comments and formatting do not make a copy look new."""
    try:
        key = ast.dump(ast.parse(code), annotate_fields=False)
    except SyntaxError:
        key = "\n".join(l.rstrip() for l in code.splitlines() if l.strip())
    return hashlib.sha1(key.encode()).hexdigest()


def _ngrams(text: str, n: int, words: bool) -> set:
    toks = re.findall(r"\w+", text.lower()) if words else re.findall(r"\w+|[^\w\s]", text)
    return {tuple(toks[i:i + n]) for i in range(max(0, len(toks) - n + 1))}


def _jaccard(a: set, b: set) -> float:
    return len(a & b) / len(a | b) if (a or b) else 1.0


def _fenced(text: str) -> Optional[str]:
    """The longest fenced code block; no raw-text fallback (3.5)."""
    blocks = re.findall(r"```(?:python|py)?[ \t]*\n(.*?)```", text, re.DOTALL)
    return max(blocks, key=len).strip("\n") + "\n" if blocks else None


def _strict_apply(parent: str, response: str) -> Tuple[Optional[str], Optional[str]]:
    """Apply SEARCH/REPLACE blocks; fail if any block does not match (stock apply_diff skips it)."""
    blocks = extract_diffs(response)
    if not blocks:
        return None, None
    lines = parent.split("\n")
    for k, (search, replace) in enumerate(blocks, 1):
        s_lines, r_lines = search.split("\n"), replace.split("\n")
        for i in range(len(lines) - len(s_lines) + 1):
            if lines[i:i + len(s_lines)] == s_lines:
                lines[i:i + len(s_lines)] = r_lines
                break
        else:
            return None, (f"SEARCH block {k} of {len(blocks)} does not match the program exactly; "
                          "copy the lines to replace verbatim, including indentation")
    return "\n".join(lines), None


# ---------------------------------------------------------------- HARVEST machinery
class Structure:
    """M3 + M2: cluster programs by code structure, and track each cluster's yield.

    The descriptor is deliberately NOT derived from score. Every diversity mechanism we
    tried on the per-instance score vector failed because that vector has an effective
    dimensionality of about 1.8 -- it is fitness in disguise. Code structure is not.
    """

    def __init__(self):
        self.X: List[Dict[str, float]] = []      # feature dicts of every scored program
        self.ids: List[str] = []                 # program id, parallel to X
        self.outs: List[tuple] = []              # outcome fingerprint, parallel to X
        self.keys: List[str] = []
        self.centroids = None
        self.hist: Dict[int, List[tuple]] = collections.defaultdict(list)  # cluster -> [(outcome, is_new)]
        self.of_program: Dict[str, int] = {}
        self.fits = 0

    @staticmethod
    def features(code: str) -> Dict[str, float]:
        d: Dict[str, float] = collections.Counter()
        try:
            t = ast.parse(code)
        except SyntaxError:
            return {"unparsed": 1.0}
        for n in ast.walk(t):
            if isinstance(n, ast.Attribute):
                d["a:" + n.attr] += 1
            elif isinstance(n, ast.Name):
                d["n:" + n.id] += 1
            elif isinstance(n, (ast.Import, ast.ImportFrom)):
                for a in n.names:
                    d["i:" + (a.name or "").split(".")[0]] += 1
            elif isinstance(n, (ast.For, ast.While)):
                d["loop"] += 1
        return dict(d)

    @staticmethod
    def outcome(prog: Program) -> tuple:
        """Behavioural fingerprint: the per-instance score vector the evaluator reports."""
        fb = (prog.artifacts or {}).get("feedback") or ""
        return tuple(round(float(x), 9) for x in re.findall(r"score=([\d.eE+-]+)", fb))

    def _vec(self, f):
        import numpy as np
        return np.array([1.0 if f.get(k) else 0.0 for k in self.keys])

    def maybe_fit(self, n_eval: int):
        if not HARVEST or len(self.X) < HV_MIN or n_eval % HV_REFIT:
            return
        try:
            import numpy as np
        except ImportError:
            return
        df = collections.Counter(k for f in self.X for k in f)
        lo, hi = 0.02 * len(self.X), 0.98 * len(self.X)
        self.keys = [k for k, c in df.items() if lo < c < hi]
        if len(self.keys) < 4:
            self.centroids = None
            return
        M = np.array([self._vec(f) for f in self.X])
        k = min(HV_K, max(2, len(M) // 10))
        rng = np.random.default_rng(0)                       # fixed: refits stay comparable
        C = M[rng.choice(len(M), k, replace=False)]
        for _ in range(25):
            lab = np.argmin(((M[:, None, :] - C[None, :, :]) ** 2).sum(2), axis=1)
            for j in range(k):
                if (lab == j).any():
                    C[j] = M[lab == j].mean(0)
        self.centroids = C
        self.fits += 1
        # The feature basis and the k-means init both change at every refit, so cluster
        # index j after this fit is unrelated to index j before it. Relabel the whole
        # history; otherwise hist[j] silently blends up to nine different regions.
        lab = np.argmin(((M[:, None, :] - C[None, :, :]) ** 2).sum(2), axis=1)
        self.of_program = {pid: int(c) for pid, c in zip(self.ids, lab)}
        self.hist = collections.defaultdict(list)
        for pid, o in zip(self.ids, self.outs):
            c = self.of_program[pid]
            self.hist[c].append((o, o not in {x for x, _ in self.hist[c]}))
        for c in list(self.hist):
            if len(self.hist[c]) > HV_WINDOW:
                self.hist[c] = self.hist[c][-HV_WINDOW:]
        logger.warning("harvest: refit %d clusters on %d programs (%d features), history relabelled",
                       k, len(M), len(self.keys))

    def assign(self, code: str) -> Optional[int]:
        if self.centroids is None:
            return None
        try:
            import numpy as np
        except ImportError:
            return None
        return int(np.argmin(((self._vec(self.features(code)) - self.centroids) ** 2).sum(1)))

    def record(self, prog: Program, code: str):
        o = self.outcome(prog)
        self.X.append(self.features(code))
        self.ids.append(prog.id)
        self.outs.append(o)
        c = self.assign(code)
        if c is None:
            return
        self.of_program[prog.id] = c
        seen = {x for x, _ in self.hist[c]}
        self.hist[c].append((o, o not in seen))
        if len(self.hist[c]) > HV_WINDOW:
            self.hist[c] = self.hist[c][-HV_WINDOW:]

    def _pooled(self) -> float:
        """Novelty rate over every region, the prior a thinly-sampled region shrinks to."""
        k = sum(1 for h in self.hist.values() for _, new in h if new)
        n = sum(len(h) for h in self.hist.values())
        return (k / n) if n else 0.5

    def yield_of(self, cluster: Optional[int]) -> float:
        """Share of recent programs from this region that produced an unseen outcome,
        shrunk toward the pooled rate with the weight of HV_PRIOR observations.

        The first version returned 1.0 for any region holding fewer than 8 programs. With
        the per-program weighting that was harmless -- a 4-program region carried little
        mass whatever its yield. Weighting per region removes that damping, so a flat 1.0
        would make the emptiest region the most attractive object in the pool and turn a
        productivity signal into a rarity bonus. Nothing measured supports that: the 2.4x
        gradient was measured on regions with observations, and says nothing about regions
        without. Shrinkage keeps an unmeasured region at the pooled rate until it earns a
        different one."""
        h = self.hist.get(cluster) or []
        mu = self._pooled()
        k = sum(1 for _, new in h if new)
        return (k + HV_PRIOR * mu) / (len(h) + HV_PRIOR)

    def yields(self) -> Dict[int, float]:
        return {c: self.yield_of(c) for c in self.hist}

    def occupancy(self) -> Dict[int, int]:
        return {c: len(v) for c, v in self.hist.items()}

    def stats(self) -> Dict[str, Any]:
        y = {c: round(self.yield_of(c), 3) for c in sorted(self.hist)}
        return {"fits": self.fits, "clusters": len(self.hist), "yield": y,
                "occupancy": self.occupancy()}


class Operators:
    """M1: cost-normalised, rank-based operator credit.

    Each outcome is scored by its percentile rank against a SHARED pool of recent
    outcomes, so operators applied 450 times and 8 times per run stay comparable. An
    operator's credit is the decayed mean of its own recent ranks, divided by its mean
    cost in evaluations. Selection is UCB over credit-per-cost.
    """

    NAMES = ("refine", "oneshot", "repair")

    def __init__(self):
        self.pool: List[float] = []                                   # shared reference scores
        self.win: Dict[str, List[float]] = collections.defaultdict(list)
        self.cost: Dict[str, List[float]] = collections.defaultdict(list)
        self.n: Dict[str, int] = collections.Counter()

    def record(self, op: str, score: float, cost: float = 1.0):
        self.pool.append(score)
        if len(self.pool) > 200:
            self.pool = self.pool[-200:]
        pool = sorted(self.pool)
        rank = (sum(1 for x in pool if x <= score) / max(len(pool), 1)) if pool else 0.5
        w = self.win[op]
        w.append(rank)
        if len(w) > HV_OPWIN:
            del w[0]
        self.cost[op].append(cost)
        if len(self.cost[op]) > HV_OPWIN:
            del self.cost[op][0]
        self.n[op] += 1

    def credit(self, op: str) -> float:
        w = self.win.get(op) or []
        if not w:
            return 1.0                                    # untried operators look attractive
        r = sorted(w, reverse=True)
        wts = [HV_DECAY ** i for i in range(len(r))]       # decay: top ranks dominate
        val = sum(a * b for a, b in zip(r, wts)) / sum(wts)
        c = sum(self.cost[op]) / max(len(self.cost[op]), 1) or 1.0
        return val / max(c, 1e-9)

    def choose(self, options: List[str]) -> str:
        tot = max(sum(self.n[o] for o in options), 1)
        best, pick = -1.0, options[0]
        for o in options:
            u = self.credit(o) + HV_C * math.sqrt(math.log(tot + 1) / (self.n[o] + 1))
            if u > best:
                best, pick = u, o
        return pick

    def stats(self) -> Dict[str, Any]:
        return {o: {"n": self.n[o], "credit": round(self.credit(o), 4)} for o in self.NAMES}


# ---------------------------------------------------------------- SIFT surrogate
class Surrogate:
    """Online admissibility predictor: numpy-only logistic regression over AST features.

    Trained only on evaluations already spent, so it can never see a label it has not
    paid for. `threshold` is None until a calibration window proves that some rejection
    fraction reaches SIFT_TARGET precision on held-out spent data; while it is None the
    gate is inert.
    """

    def __init__(self):
        self.X: List[Dict[str, float]] = []
        self.y: List[int] = []
        self.keys: List[str] = []
        self.w = self.mu = self.sd = None
        self.threshold: Optional[float] = None
        self.rejected = self.fits = 0

    @staticmethod
    def features(code: str, parent_score: float) -> Dict[str, float]:
        d: Dict[str, float] = collections.Counter()
        try:
            t = ast.parse(code)
            d["parses"] = 1.0
            for n in ast.walk(t):
                d["n_" + type(n).__name__] += 1
                if isinstance(n, (ast.Import, ast.ImportFrom)):
                    for a in n.names:
                        d["imp_" + (a.name or "").split(".")[0]] += 1
                elif isinstance(n, ast.Attribute):
                    d["at_" + n.attr] += 1
                elif isinstance(n, ast.Name):
                    d["id_" + n.id] += 1
        except SyntaxError:
            d["parses"] = 0.0
        d["len"] = len(code) / 1000.0
        d["pscore"] = parent_score
        d["padm"] = float(parent_score > 0)
        return dict(d)

    def record(self, code: str, parent_score: float, admissible: bool):
        self.X.append(self.features(code, parent_score))
        self.y.append(int(admissible))

    # -- numpy helpers -------------------------------------------------
    def _matrix(self, rows, keys):
        import numpy as np
        M = np.zeros((len(rows), len(keys) + 1))
        M[:, -1] = 1.0
        idx = {k: i for i, k in enumerate(keys)}
        for i, f in enumerate(rows):
            for k, v in f.items():
                j = idx.get(k)
                if j is not None:
                    M[i, j] = v
        M[:, :-1] = np.sign(M[:, :-1]) * np.log1p(np.abs(M[:, :-1]))
        return M

    @staticmethod
    def _fit(X, y, iters=400, lr=0.5, l2=0.01):
        import numpy as np
        w = np.zeros(X.shape[1])
        for _ in range(iters):
            pr = 1.0 / (1.0 + np.exp(-np.clip(X @ w, -30, 30)))
            w -= lr * (X.T @ (pr - y) / len(y) + l2 * w)
        return w

    @staticmethod
    def _pred(X, w):
        import numpy as np
        return 1.0 / (1.0 + np.exp(-np.clip(X @ w, -30, 30)))

    def maybe_refit(self, n_eval: int):
        """Recalibrate on spent data. Sets self.threshold, or clears it when unsafe."""
        if not SIFT or n_eval < SIFT_MIN_TRAIN or n_eval % SIFT_REFIT or len(self.y) < SIFT_MIN_TRAIN:
            return
        try:
            import numpy as np
        except ImportError:
            return
        y = np.array(self.y)
        if y.min() == y.max():
            self.threshold = None
            return
        self.keys = sorted({k for f in self.X for k in f})
        M = self._matrix(self.X, self.keys)
        mu, sd = M[:, :-1].mean(0), M[:, :-1].std(0) + 1e-9
        M[:, :-1] = (M[:, :-1] - mu) / sd
        c = int(0.7 * len(y))
        self.threshold = None
        if c < 20 or y[:c].min() == y[:c].max() or y[c:].min() == y[c:].max():
            return
        wv = self._fit(M[:c], y[:c])
        pv, yv = self._pred(M[c:], wv), y[c:]
        order = np.argsort(pv)
        frac = None
        f = SIFT_MAXREJ
        while f >= 0.05:                       # largest rejection fraction that is still safe
            k = int(f * len(yv))
            if k >= 8 and (1 - yv[order][:k].mean()) >= SIFT_TARGET:
                frac = f
                break
            f -= 0.05
        if frac is None:
            return
        self.w = self._fit(M, y)               # refit on everything, then set the cut there
        self.mu, self.sd = mu, sd
        pall = np.sort(self._pred(M, self.w))
        self.threshold = float(pall[max(0, int(frac * len(pall)) - 1)])
        self.fits += 1
        logger.warning("sift: refit on %d evaluations, condemning the lowest %.0f%% "
                       "(held-out precision >= %.2f)", len(y), 100 * frac, SIFT_TARGET)

    def condemns(self, code: str, parent_score: float) -> bool:
        if not SIFT or self.threshold is None or self.w is None:
            return False
        try:
            import numpy as np
        except ImportError:
            return False
        M = self._matrix([self.features(code, parent_score)], self.keys)
        M[:, :-1] = (M[:, :-1] - self.mu) / self.sd
        return bool(self._pred(M, self.w)[0] <= self.threshold)


# ---------------------------------------------------------------- lineage state
class Lineage:
    def __init__(self, lid: int, kind: str, archive: UnifiedArchive, adapt: AdaptiveState,
                 text: Optional[str] = None, parents: Tuple[int, ...] = ()):
        self.id, self.kind, self.parents, self.text = lid, kind, list(parents), text
        self.status, self.close_reason = INCUBATING, None
        self.archive, self.adapt = archive, adapt
        self.programs: List[str] = []
        self.n = self.k = self.dry = self.gains = 0
        self.best = 0.0
        self.best_id: Optional[str] = None
        self.curve8: Optional[float] = None
        self.drafts: List[Dict[str, Any]] = []
        self.overtook = False
        self.empty_jobs = 0
        self.zero_ids: List[str] = []         # programs scoring 0 (repair material)
        self.last_failure = ""

    @property
    def title(self) -> str:
        m = re.search(r"TITLE:\s*(.+)", self.text or "")
        return m.group(1).strip()[:100] if m else ("initial program" if self.kind == "seed" else f"lineage {self.id}")

    def summary(self) -> Dict[str, Any]:
        return {k: getattr(self, k) for k in ("id", "kind", "parents", "status", "close_reason", "n", "k",
                                              "dry", "gains", "best", "best_id", "curve8")} | {
            "title": self.title, "text": self.text, "n_programs": len(self.programs)}


class SproutDatabase(ProgramDatabase):
    """Keeps every program (lineage archives do the sampling)."""

    def add(self, program: Program, iteration: Optional[int] = None, **kwargs) -> str:
        self.programs[program.id] = program
        if iteration is not None:
            self.last_iteration = max(self.last_iteration, iteration)
        if self.config.db_path:
            self._save_program(program)
        self._update_best_program(program)
        return program.id

    def sample(self, num_context_programs: Optional[int] = 4, **kwargs):
        progs = sorted(self.programs.values(), key=lambda p: _score(p.metrics), reverse=True)
        return progs[0], progs[1:1 + (num_context_programs or 0)]


# ---------------------------------------------------------------- controller
class SproutController(DiscoveryController):
    def __init__(self, controller_input):
        super().__init__(controller_input)
        self.context_builder = AdaEvolveContextBuilder(self.config)
        self.db_cfg = self.config.search.database
        self.rng = self.database.rng
        self.t_start = time.time()
        self.T = _wall_seconds()
        self.E = 0
        self.lineages: List[Lineage] = []
        self.inc: Optional[Lineage] = None
        self.challengers: List[Lineage] = []
        self.nursery: Optional[Lineage] = None
        self.births = 0
        self.ref8: List[float] = []
        self.b0 = 0.0
        self.g_star = 0.0
        self.cache: Dict[str, float] = {}
        self.sift = Surrogate()
        self.struct = Structure()
        self.ops = Operators()
        self.notes: Dict[str, List[Any]] = {}       # class -> [count, verbatim example]
        self.hybridised: set = set()
        self.rr = 0                                  # lane-B round robin
        self.iteration = 0
        self.initial_solution = ""
        self.required: List[str] = []
        self.system = self.config.context_builder.system_message or ""
        self._stats_path: Optional[str] = None
        self._lineage_path: Optional[str] = None
        self._evaluator_src = ""
        try:
            with open(self.evaluation_file) as fh:
                self._evaluator_src = fh.read()
        except OSError:
            pass

    # ------------------------------------------------------------ setup
    def _new_lineage(self, kind: str, text: Optional[str] = None, parents: Tuple[int, ...] = ()) -> Lineage:
        c = self.db_cfg
        hib = getattr(c, "higher_is_better", {}) or {}
        archive = UnifiedArchive(
            config=ArchiveConfig(
                max_size=getattr(c, "population_size", 20), k_neighbors=getattr(c, "k_neighbors", 5),
                elite_ratio=getattr(c, "archive_elite_ratio", 0.2), pareto_weight=getattr(c, "pareto_weight", 0.4),
                fitness_weight=getattr(c, "fitness_weight", 1.0), novelty_weight=getattr(c, "novelty_weight", 0.0),
                higher_is_better=hib, fitness_key=getattr(c, "fitness_key", None)),
            diversity_strategy=create_diversity_strategy(getattr(c, "diversity_strategy", "code"), higher_is_better=hib),
            rng=self.rng)
        adapt = AdaptiveState(decay=getattr(c, "decay", 0.9), intensity_min=getattr(c, "intensity_min", 0.15),
                              intensity_max=getattr(c, "intensity_max", 0.5))
        lin = Lineage(len(self.lineages), kind, archive, adapt, text, parents)
        self.lineages.append(lin)
        return lin

    def _required_names(self) -> List[str]:
        """Entry points the evaluator calls, plus in-block names the frozen code uses."""
        req = set(re.findall(r"\bprog\.(\w+)\s*\(", self._evaluator_src))
        parts = _split_block(self.initial_solution)
        if parts:
            pre, block, post = parts
            try:
                defined = _top_level_names(ast.parse(block))
                frozen = ast.parse(pre + "\n" + post)
                used = {n.id for n in ast.walk(frozen) if isinstance(n, ast.Name)}
                req |= defined & used
            except SyntaxError:
                pass
        return sorted(req)

    def _contract(self) -> Dict[str, List[str]]:
        """{class: [methods]} that the evaluator calls on objects the program supplies.

        The required-name check protects top-level names, but an evaluator can also call a
        method on the object a program returns: cloudcast does `bc.set_num_partitions(k)`
        (evaluator.py:50). A candidate that drops it scores zero deterministically -- 190 of
        500 programs in the hacked round-2 run and 183 of 500 in the patched one, every one
        of them zero. Screening it costs an LLM retry instead of an evaluation. The rule is
        derived, not hard-coded: it resolves to {BroadCastTopology: [set_num_partitions]} on
        cloudcast and to {} on the other four tasks, in both kits.
        """
        if os.environ.get("SPROUT_CONTRACT") == "0":
            return {}
        try:
            ev = ast.parse(self._evaluator_src)
            init = ast.parse(self.initial_solution)
        except SyntaxError:
            return {}
        attrs = {n.attr for n in ast.walk(ev) if isinstance(n, ast.Attribute)}
        out = {}
        for n in init.body:
            if isinstance(n, ast.ClassDef):
                m = sorted({x.name for x in n.body
                            if isinstance(x, (ast.FunctionDef, ast.AsyncFunctionDef))} & attrs)
                if m:
                    out[n.name] = m
        return out

    @staticmethod
    def _provides(tree: ast.Module, name: str) -> bool:
        """True if the program still supplies `name`: as a def, an assignment, a keyword
        argument, or a literal string (setattr/type() construction)."""
        for n in ast.walk(tree):
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == name:
                return True
            if isinstance(n, ast.Name) and n.id == name and isinstance(n.ctx, ast.Store):
                return True
            if isinstance(n, ast.keyword) and n.arg == name:
                return True
            if isinstance(n, ast.Constant) and n.value == name:
                return True
        return False

    def _setup_logs(self):
        out = self.output_dir or "."
        os.makedirs(out, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._stats_path = os.path.join(out, f"sprout_iteration_stats_{ts}.jsonl")
        self._lineage_path = os.path.join(out, "lineages.json")

    # ------------------------------------------------------------ derived quantities
    def progress(self) -> float:
        return max(self.E / EVAL_CAP, (time.time() - self.t_start) / (0.9 * self.T))

    def delta_star(self) -> float:
        return max(self.g_star - self.b0, 0.01 * abs(self.g_star), 1e-12)

    def ratio(self, s: Lineage) -> float:
        return 1.0 if self.g_star <= 0 else s.best / self.g_star

    def window(self, s: Lineage) -> int:
        if s is not self.inc:
            return L_MIN
        h = (s.k + PRIOR_A) / (s.n + PRIOR_A + PRIOR_B)
        w = math.ceil(math.log(ALPHA) / math.log(1 - h)) if 0 < h < 1 else L_MIN
        return int(min(L_MAX, max(L_MIN, w)))

    def exhausted(self, s: Lineage) -> bool:
        return s.dry >= self.window(s)

    def birth_due(self, p: float) -> bool:                                         # R1
        return (p < P_NO_NEW and self.births < N_MAX and self.inc is not None
                and self.exhausted(self.inc) and not self.challengers)

    # ------------------------------------------------------------ main loop
    async def run_discovery(self, start_iteration: int, max_iterations: int, checkpoint_callback=None):
        self._setup_logs()
        self.iteration = start_iteration - 1
        seed = self.database.get_best_program()
        if seed is None:
            raise RuntimeError("SPROUT needs the evaluated initial program in the database")
        self.initial_solution = seed.solution
        self.required = self._required_names()
        self.contract = self._contract()
        self.b0 = self.g_star = _score(seed.metrics)
        self.E = 1                                       # the runner's seed evaluation
        s0 = self._new_lineage("seed")
        s0.status = INCUMBENT
        self.inc = s0
        seed.metadata = dict(seed.metadata or {}, strategy_id=0, role="seed")
        self._record_into(s0, seed, _score(seed.metrics))
        self.cache[_norm_hash(seed.solution)] = _score(seed.metrics)
        self._note_feedback(seed)
        logger.warning("sprout: b0=%.6g required=%s contract=%s T=%.0fs (LIMIT=%s) cap=%d N_MAX=%d "
                       "L_MIN=%d L_MAX=%d TRIAL=%s FAST_FAIL=%d SIFT=%s HARVEST=%s prompts=%s",
                       self.b0, self.required, self.contract or "{}", self.T, os.environ.get("LIMIT"),
                       EVAL_CAP, N_MAX, L_MIN, L_MAX, "legacy" if LEGACY else TRIAL, FAST_FAIL,
                       f"on(target={SIFT_TARGET},maxrej={SIFT_MAXREJ})" if SIFT else "off",
                       (f"on(K={HV_K},yield={int(HV_YIELD)},ops={int(HV_OPS)},birth={int(HV_BIRTH)})"
                        if HARVEST else "off"), PROMPTS)

        empty = 0
        while self.E < EVAL_CAP and not self.shutdown_event.is_set():
            e0 = self.E
            try:
                await self._step(checkpoint_callback)
            except Exception as e:                       # one bad step must not end the run
                logger.exception("sprout step failed: %s", e)
            self._write_lineages()
            # A step that evaluates nothing costs no budget, but an unreachable or
            # rejecting LLM server would otherwise be hammered in a tight loop.
            empty = 0 if self.E > e0 else empty + 1
            if empty >= 3:
                await asyncio.sleep(min(60.0, 2.0 ** min(empty - 3, 6)))
            if empty >= 400:
                logger.error("sprout: %d consecutive steps without an evaluation, giving up", empty)
                break
        self._write_lineages()
        logger.warning("sprout: finished E=%d births=%d champion=lineage %s best=%.6g "
                       "sift_refits=%d sift_rejected=%d harvest_fits=%d clusters=%d",
                       self.E, self.births, self.inc.id if self.inc else None, self.g_star,
                       self.sift.fits, self.sift.rejected, self.struct.fits, len(self.struct.hist))
        return self.database.get_best_program()

    async def _step(self, checkpoint_callback):
        p = self.progress()
        k = min(BATCH, EVAL_CAP - self.E)
        if self.nursery is None and self.birth_due(p):
            self.nursery = await self._birth()
            if self.nursery is None:
                self.inc.dry = 0                           # nothing usable was written
        if self.nursery is not None:
            jobs = self._incubation_jobs(self.nursery, k)
        else:
            jobs = [("refine", self.inc, None)]
            if k == 2:
                jobs.append(("refine", self._slot_b(p) or self.inc, None))
        cands = await asyncio.gather(*(self._generate(job) for job in jobs))
        for job, c in zip(jobs, cands):
            lin = job[1]
            if c is None:
                lin.empty_jobs += 1
            else:
                lin.empty_jobs = 0
        seen, uniq = set(), []
        for c in cands:
            if c is not None and c["hash"] not in seen:
                seen.add(c["hash"])
                uniq.append(c)
        cands = uniq
        if self.nursery is not None and self.nursery.empty_jobs >= UNIMPLEMENTABLE and self.nursery.best <= 0:
            self._close(self.nursery, "unimplementable")
            self._end_incubation()
        if not cands or self.shutdown_event.is_set():
            return
        t0 = time.time()
        results = await asyncio.gather(*(self.evaluator.evaluate_program(c["code"], c["id"]) for c in cands))
        eval_time = time.time() - t0
        for c, res in zip(cands, results):
            self._record(c, res, eval_time, checkpoint_callback)                    # R6
        if self.nursery is not None:
            self._maybe_gate(self.nursery, self.progress())                         # R4
        if self.progress() < P_POLISH:
            for c in list(self.challengers):
                if self.exhausted(c):
                    self._close(c, "exhausted")                                     # R7

    def _slot_b(self, p: float) -> Optional[Lineage]:                               # R5
        if not self.challengers:
            return None
        if p >= P_POLISH:
            near = [c for c in self.challengers if self.ratio(c) >= R_NEAR]
            return max(near, key=lambda c: c.best) if near else None
        self.rr = (self.rr + 1) % len(self.challengers)
        return self.challengers[self.rr]

    def _incubation_jobs(self, s: Lineage, k: int) -> List[Tuple[str, Lineage, Any]]:   # R3
        if s.n == 0:
            jobs = []
            for _ in range(k):
                jobs.append(("draft", s, s.drafts.pop(0)) if s.drafts else ("fresh", s, None))
            return jobs
        if s.best <= 0:
            if HV_OPS and k == 1:
                pick = self.ops.choose(["repair", "oneshot"])
                return [("repair", s, None)] if pick == "repair" else [("fresh", s, None)]
            return [("repair", s, None), ("fresh", s, None)][:k]
        return [("refine", s, None) for _ in range(k)]

    # ------------------------------------------------------------ recording (R6)
    def _record_into(self, s: Lineage, prog: Program, score: float):
        s.programs.append(prog.id)
        s.archive.add(prog)
        s.adapt.record_evaluation(score)
        s.n += 1
        old = s.best
        if score > s.best:
            s.best, s.best_id = score, prog.id
        if score - old >= THETA * self.delta_star() and score > old:
            s.dry = 0
            s.k += 1
            s.gains += int(old > 0)
        else:
            s.dry += 1
        if score <= 0:
            s.zero_ids.append(prog.id)
        if s.kind == "seed" and s.n == B_INC:
            s.curve8 = s.best
            self.ref8.append(s.best)

    def _record(self, c: Dict[str, Any], res, eval_time: float, checkpoint_callback):
        s: Lineage = c["lineage"]
        metrics = dict(res.metrics or {})
        score = _score(metrics)
        self.E += 1
        self.iteration += 1
        self.cache[c["hash"]] = score
        parent = c.get("parent")
        prog = Program(
            id=c["id"], solution=c["code"], language=self.config.language, metrics=metrics,
            iteration_found=self.iteration, parent_id=parent.id if parent else None,
            other_context_ids=c.get("context_ids") or [],
            generation=(parent.generation + 1) if parent else 0,
            metadata={"changes": c.get("changes"), "parent_metrics": parent.metrics if parent else None,
                      "strategy_id": s.id, "role": c["role"], "lineage_kind": s.kind},
            artifacts=res.artifacts or {})
        prev_best = self.g_star
        if c["role"] == "oneshot" and self.inc is not None and score > 0 and \
                abs(score - self.inc.best) <= COPY_REL * max(1.0, abs(self.inc.best)):
            logger.warning("sprout: behavioural copy: one-shot %s in lineage %d scores exactly the champion's %.6g",
                           c["id"][:8], s.id, score)
        if score <= 0 and c["role"] == "oneshot":
            errs = [l for l in ((res.artifacts or {}).get("feedback") or "").splitlines() if "error" in l or "timeout" in l]
            s.last_failure = "\n".join(errs[:6]) or "it scored zero"
        self._record_into(s, prog, score)
        self.sift.record(c["code"], _score(parent.metrics) if parent else 0.0, score > 0)
        self.sift.maybe_refit(self.E)
        if HARVEST:
            self.struct.record(prog, c["code"])
            self.struct.maybe_fit(self.E)
            if HV_OPS:
                # a one-shot carries the birth trial's share of the cost; a refine is 1
                self.ops.record(c["role"] if c["role"] in Operators.NAMES else "refine",
                                score, cost=2.0 if c["role"] == "oneshot" else 1.0)
        self._note_feedback(prog)
        if score > self.g_star:
            self.g_star = score
            if s is not self.inc:
                old = self.inc
                s.status = INCUMBENT
                if s is self.nursery:
                    s.overtook = True
                # self.inc must be reassigned before _close, which refuses to close the champion.
                self.inc = s
                if LEGACY:
                    old.status, old.dry = CHALLENGER, 0
                    self.challengers.append(old)
                    if s in self.challengers:
                        self.challengers.remove(s)
                    self._trim_challengers()
                else:
                    self._close(old, "overtaken")
                logger.warning("sprout: lineage %d overtakes (%.6g > %.6g); lineage %d %s",
                               s.id, score, prev_best, old.id,
                               "becomes a challenger" if LEGACY else "closes")
        result = SerializableResult(
            child_program_dict=prog.to_dict(), parent_id=prog.parent_id, other_context_ids=prog.other_context_ids,
            iteration_time=c["llm_time"] + eval_time, llm_generation_time=c["llm_time"], eval_time=eval_time,
            prompt=c.get("prompt"), llm_response=c.get("response"), iteration=self.iteration)
        self._process_iteration_result(result, self.iteration, checkpoint_callback)
        self._log_stats(prog, s, c["role"])

    # ------------------------------------------------------------ gate (R4, R9)
    def _maybe_gate(self, s: Lineage, p: float):
        if s.overtook:
            logger.warning("sprout: newborn %d graduates by overtaking at age %d", s.id, s.n)
            s.curve8 = s.best
            self.ref8.append(s.best)
            self._end_incubation()
            return
        if s.n >= FAST_FAIL and s.best <= 0:
            self._close(s, "inadmissible")
            self._end_incubation()
            return
        if not LEGACY:
            # The only way out of the nursery is to take the global best, which is handled
            # above. A lineage that has not done so by TRIAL evaluations closes: no promotion,
            # no challenger tier, no ratio gate.
            if s.n >= TRIAL:
                self._close(s, "no overtake")
                self._end_incubation()
            return
        if s.n < B_INC:
            return
        s.curve8 = s.best
        floor = R_FLOOR0 + R_FLOOR_SLOPE * p
        level = bool(self.ref8) and s.curve8 >= statistics.median(self.ref8)
        r = self.ratio(s)
        if self._behavioural_copy(s):
            self._close(s, "copy")
        elif s.best > 0 and (r >= R_NEAR or ((level or s.gains >= GAINS_MIN) and r >= floor)):
            s.status, s.dry = CHALLENGER, 0
            self.challengers.append(s)
            self.ref8.append(s.curve8)
            logger.warning("sprout: newborn %d promoted (r=%.3f floor=%.3f level=%s gains=%d)",
                           s.id, r, floor, level, s.gains)
            self._trim_challengers()
        else:
            self._close(s, f"below gate (r={r:.3f}, floor={floor:.3f}, level={level}, gains={s.gains})")
        self._end_incubation()

    def _behavioural_copy(self, s: Lineage) -> bool:
        if s.gains > 0 or s.best <= 0:
            return False
        others = [self.b0] + [l.best for l in self.lineages if l is not s and l.n > 0]
        return any(abs(s.best - o) <= COPY_REL * max(1.0, abs(o)) for o in others)

    def _trim_challengers(self):
        while len(self.challengers) > C_MAX:
            self._close(min(self.challengers, key=self.ratio), "displaced")

    def _close(self, s: Lineage, reason: str):
        if s is self.inc:
            return                                       # the champion is never closed
        s.status, s.close_reason = CLOSED, reason
        if s in self.challengers:
            self.challengers.remove(s)
        logger.warning("sprout: lineage %d closed: %s (n=%d best=%.6g)", s.id, reason, s.n, s.best)

    def _end_incubation(self):
        if self.inc is not None:
            self.inc.dry = 0
        self.nursery = None

    # ------------------------------------------------------------ birth (R2)
    async def _birth(self) -> Optional[Lineage]:
        self.births += 1
        if self.inc.text is None:
            self.inc.text = await self._describe(self.inc)
        partner = None
        if self.births % HYBRID_EVERY == 0:
            pool = [l for l in self.lineages if l is not self.inc and l.best > 0 and l.best >= HYBRID_MIN_RATIO * self.g_star
                    and (min(l.id, self.inc.id), max(l.id, self.inc.id)) not in self.hybridised]
            if pool:
                partner = max(pool, key=lambda l: l.best)
                if partner.text is None:
                    partner.text = await self._describe(partner)
        kind = "hybrid" if partner else "novel"
        text = await self._write_strategy(partner)
        if text is None:
            logger.warning("sprout: birth %d produced no usable strategy", self.births)
            return None
        parents = (self.inc.id, partner.id) if partner else (self.inc.id,)
        if partner:
            self.hybridised.add((min(partner.id, self.inc.id), max(partner.id, self.inc.id)))
        s = self._new_lineage(kind, text, parents)
        s.drafts = await self._one_shot_drafts(s)
        if HV_BIRTH and self.struct.centroids is not None:
            # Diagnostic only. This block never steered birth: it computed the thinnest
            # region and logged it, and nothing consumed the value. Retired as a
            # mechanism rather than left in as decoration.
            occ = self.struct.occupancy()
            if occ:
                thin = min(occ, key=lambda c: occ[c])
                logger.warning("harvest: thinnest structural cluster %d holds %d programs, yields %.2f",
                               thin, occ[thin], self.struct.yield_of(thin))
        logger.warning("sprout: birth %d -> lineage %d (%s) '%s' with %d screened drafts at E=%d p=%.3f",
                       self.births, s.id, kind, s.title, len(s.drafts), self.E, self.progress())
        return s

    async def _llm(self, user: str, cap: int = CAP_OTHER) -> Tuple[Optional[str], float]:
        if len(user) > cap:
            logger.warning("sprout: prompt %d chars over the %d cap", len(user), cap)
        t0 = time.time()
        try:
            r = await self._call_llm(self.system, user)
            return r.text, time.time() - t0
        except Exception as e:
            logger.warning("sprout: LLM call failed: %s", e)
            return None, time.time() - t0

    @staticmethod
    def _parse_strategy(text: Optional[str]) -> Optional[str]:
        if not text:
            return None
        m = re.search(r"<<<STRATEGY>>>(.*?)(?:<<<END>>>|$)", text, re.DOTALL)
        body = (m.group(1) if m else "").strip()
        if not body or "TITLE:" not in body:
            return None
        return body[:1100]

    async def _describe(self, s: Lineage) -> Optional[str]:
        pid = s.best_id or (s.programs[0] if s.programs else None)
        best = self.database.get(pid) if pid else None
        if best is None:
            return None
        text, _ = await self._llm(_tpl("describe").format(score=f"{s.best:.6g}", program=best.solution[:CAP_OTHER - 2000]))
        return self._parse_strategy(text)

    async def _write_strategy(self, partner: Optional[Lineage]) -> Optional[str]:
        inc_best = self.database.get(self.inc.best_id) if self.inc.best_id else None
        fb = ((inc_best.artifacts or {}).get("feedback") or "") if inc_best else ""
        if partner:
            mode = _tpl("mode_hybrid").format(text_a=self.inc.text or "(no description)",
                                              text_b=partner.text or "(no description)")
        else:
            mode = _tpl("mode_novel")
        user = _tpl("write").format(
            required=", ".join(self.required) or "(see system message)", inc_score=f"{self.g_star:.6g}",
            inc_text=self.inc.text or "(no description)", inc_feedback=fb[:FEEDBACK_CHARS] or "(none)",
            ledger=self._ledger(), notes=self._notes(), mode_block=mode)
        for _ in range(2):                                   # retry once if nothing survives
            outs = await asyncio.gather(*(self._llm(user) for _ in range(N_WRITE)))
            titles = {re.sub(r"\W+", " ", l.title.lower()).strip() for l in self.lineages}
            texts = [l.text for l in self.lineages if l.text]
            cands = []
            for text, _ in outs:
                body = self._parse_strategy(text)
                if not body:
                    continue
                m = re.search(r"TITLE:\s*(.+)", body)
                if m and re.sub(r"\W+", " ", m.group(1).lower()).strip() in titles:
                    continue
                g = _ngrams(body, 3, words=True)
                sim = max((_jaccard(g, _ngrams(t, 3, words=True)) for t in texts), default=0.0)
                cands.append((sim, body))
            if cands:
                return min(cands, key=lambda x: x[0])[1]
        return None

    async def _one_shot_drafts(self, s: Lineage, previous_failure: str = "") -> List[Dict[str, Any]]:
        outs = await asyncio.gather(*(self._one_shot(s, previous_failure) for _ in range(N_DRAFT)))
        drafts, seen = [], set()
        for d in outs:
            if d and d["hash"] not in seen:
                seen.add(d["hash"])
                drafts.append(d)
        if len(drafts) > 2:                                   # most mutually dissimilar pair first
            grams = [_ngrams(d["code"], 5, words=False) for d in drafts]
            pairs = [(i, j) for i in range(len(drafts)) for j in range(i + 1, len(drafts))]
            i, j = min(pairs, key=lambda ij: _jaccard(grams[ij[0]], grams[ij[1]]))
            drafts = [drafts[i], drafts[j]] + [d for x, d in enumerate(drafts) if x not in (i, j)]
        return drafts

    async def _one_shot(self, s: Lineage, previous_failure: str = "") -> Optional[Dict[str, Any]]:
        prev = f"\n## Your previous attempt at this strategy failed\n{previous_failure[:FEEDBACK_CHARS]}\n" if previous_failure else ""
        base = _tpl("oneshot").format(strategy=s.text, required=", ".join(self.required) or "(see system message)",
                                      interface=self.initial_solution, notes=self._notes(), previous_failure=prev)
        error, llm_time = None, 0.0
        for _ in range(ATTEMPTS):
            user = base + (f"\n## Your previous answer was rejected before evaluation\n{error}\n" if error else "")
            text, dt = await self._llm(user)
            llm_time += dt
            if not text:
                error = "empty response"
                continue
            code = _fenced(text)
            if code is None:
                error = "no ```python fenced block found; return the complete program in one fenced block"
                continue
            code, error = self._filter(code, parent_code=self.initial_solution, splice=True)
            if error:
                continue
            error = self._condemned(code, self.b0)
            if error:
                continue
            refs = [self.initial_solution] + [self.database.get(l.best_id).solution for l in self.lineages
                                              if l.best_id and self.database.get(l.best_id) is not None]
            blk = lambda c: (_split_block(c) or ("", c, ""))[1]
            g = _ngrams(blk(code), 5, words=False)
            if any(_jaccard(g, _ngrams(blk(r), 5, words=False)) >= COPY_JACCARD for r in refs):
                error = ("the program is nearly identical to an existing one; write a new implementation "
                         "of the strategy instead of copying")
                continue
            return {"code": code, "hash": _norm_hash(code), "llm_time": llm_time, "response": text,
                    "prompt": {"system": self.system, "user": user}}
        s.last_failure = error or ""
        return None

    # ------------------------------------------------------------ generation of one job
    async def _generate(self, job) -> Optional[Dict[str, Any]]:
        kind, s, payload = job
        try:
            if kind == "draft":
                d = payload
                if d["hash"] in self.cache:
                    return None
                return self._cand(s, "oneshot", d["code"], d["hash"], None, d["llm_time"], d["prompt"],
                                  d["response"], "one-shot from strategy")
            if kind == "fresh":
                d = s.drafts.pop(0) if s.drafts else await self._one_shot(s, s.last_failure)
                if d is None or d["hash"] in self.cache:
                    return None
                return self._cand(s, "oneshot", d["code"], d["hash"], None, d["llm_time"], d["prompt"],
                                  d["response"], "one-shot from strategy")
            if kind == "repair":
                return await self._repair(s)
            return await self._refine(s)
        except Exception as e:
            logger.exception("sprout: job %s on lineage %d failed: %s", kind, s.id, e)
            return None

    def _cand(self, s, role, code, h, parent, llm_time, prompt, response, changes, context_ids=None):
        return {"id": str(uuid.uuid4()), "lineage": s, "role": role, "code": code, "hash": h,
                "parent": parent, "llm_time": llm_time, "prompt": prompt, "response": response,
                "changes": changes, "context_ids": context_ids or []}

    def _sample_parent(self, s: Lineage) -> Tuple[Program, str]:                     # R8
        pop = s.archive.get_all()
        if HV_YIELD and self.struct.centroids is not None and len(pop) >= 3 and self.rng.random() < HV_BIAS:
            # M2: prefer parents from structural regions still producing new outcomes.
            # Measured forward on HARVEST's own 16 runs: within a progress band, the next
            # program from the high-yield half of regions sets a record 1.29% of the time
            # against 0.53% -- a 2.4x gradient that survives stratification by progress.
            groups: Dict[Any, List[Program]] = collections.defaultdict(list)
            for prog in pop:
                c = self.struct.of_program.get(prog.id)
                if c is None:
                    c = self.struct.assign(prog.solution)
                groups[c].append(prog)
            if HV2 and len(groups) >= 2:
                # Weight the REGION, not each program in it. Weighting per program makes a
                # region's mass yield x occupancy; occupancy spans 4-40x against yield's
                # 2x, so the tilt the first version realised was 1.7x on a 2.4x signal.
                cs = list(groups)
                w = [max(self.struct.yield_of(c), 0.02) ** HV_BETA for c in cs]
                r = self.rng.random() * sum(w)
                acc, pick = 0.0, cs[-1]
                for c, wi in zip(cs, w):
                    acc += wi
                    if acc >= r:
                        pick = c
                        break
                # Score still matters inside a region: top-quintile parents set records at
                # 1.84%/eval against 0.74%. Sample from the better half of the region.
                sub = sorted(groups[pick], key=lambda q: _score(q.metrics), reverse=True)
                sub = sub[:max(1, int(len(sub) * HV_TOPFRAC))]
                return self.rng.choice(sub), EXPLOIT_LABEL
            w = [max(self.struct.yield_of(c), 0.02) for c in
                 (self.struct.of_program.get(p0.id, self.struct.assign(p0.solution)) for p0 in pop)]
            r = self.rng.random() * sum(w)
            acc = 0.0
            for prog, wi in zip(pop, w):
                acc += wi
                if acc >= r:
                    return prog, EXPLOIT_LABEL
            return pop[-1], EXPLOIT_LABEL
        intensity = s.adapt.get_search_intensity()
        u = self.rng.random()
        if u < intensity:
            return s.archive.sample_parent("exploration"), EXPLORE_LABEL
        if u < intensity + 0.7 * (1 - intensity):
            top = sorted(pop, key=lambda p: _score(p.metrics), reverse=True)
            return self.rng.choice(top[:max(1, len(top) // 4)]), EXPLOIT_LABEL
        return s.archive.sample_parent("balanced"), ""

    def _refine_prompt(self, s, parent, label, error):
        ctx_progs = s.archive.sample_other_context_programs(parent, NUM_CONTEXT)
        siblings = s.archive.get_children(parent.id)[-5:]
        while True:
            context = {"program_metrics": parent.metrics, "other_context_programs": {"": ctx_progs},
                       "paradigm": None, "siblings": siblings, "error_context": error}
            prompt = self.context_builder.build_prompt({label: parent}, context)
            if s.text:
                block = _tpl("inject").format(strategy=s.text, notes_block=self._notes_block())
                user = prompt.get("user", "")
                i = user.rfind("\n# Task")
                prompt["user"] = (user[:i] + block + user[i:]) if i >= 0 else (user + block)
            if len(prompt.get("user", "")) <= CAP_REFINE or (not ctx_progs and not siblings):
                return prompt, ctx_progs
            if ctx_progs:
                ctx_progs = ctx_progs[:-1]
            else:
                siblings = siblings[1:]

    async def _refine(self, s: Lineage) -> Optional[Dict[str, Any]]:
        parent, label = self._sample_parent(s)
        if parent is None:
            return None
        error, llm_time = None, 0.0
        for _ in range(ATTEMPTS):
            prompt, ctx = self._refine_prompt(s, parent, label, error)
            if len(prompt.get("user", "")) > CAP_REFINE:
                logger.warning("sprout: refine prompt %d chars > cap even without context", len(prompt["user"]))
            t0 = time.time()
            try:
                r = await self._call_llm(prompt["system"], prompt["user"])
                text = r.text
            except Exception as e:
                text, error = None, f"LLM call failed: {e}"
            llm_time += time.time() - t0
            if not text:
                error = error or "empty response"
                continue
            code, changes, error = self._parse_edit(text, parent.solution)
            if error:
                continue
            code, error = self._filter(code, parent_code=parent.solution, splice=False)
            if error:
                continue
            # Never starve a slot: on the last attempt the candidate is evaluated even if
            # the surrogate condemns it, so a mis-calibrated gate costs LLM calls, not progress.
            if _ < ATTEMPTS - 1:
                error = self._condemned(code, _score(parent.metrics))
                if error:
                    continue
            return self._cand(s, "refine", code, _norm_hash(code), parent, llm_time, prompt, text, changes,
                              [p.id for p in ctx])
        return None

    async def _repair(self, s: Lineage) -> Optional[Dict[str, Any]]:
        progs = [self.database.get(i) for i in s.zero_ids]
        progs = [p for p in progs if p is not None]
        if not progs:
            return None

        def ok_count(p):
            fb = (p.artifacts or {}).get("feedback") or ""
            return len(re.findall(r": ok,", fb))
        target = max(progs, key=ok_count)
        if len(target.solution) > CAP_REFINE - 6000:
            return None                                       # too long to repair inside the window
        fb = ((target.artifacts or {}).get("feedback") or "")[:FEEDBACK_CHARS]
        base = _tpl("repair").format(strategy=s.text, program=target.solution, feedback=fb or "(none)", notes=self._notes())
        error, llm_time = None, 0.0
        for _ in range(ATTEMPTS):
            user = base + (f"\n## Your previous answer was rejected before evaluation\n{error}\n" if error else "")
            text, dt = await self._llm(user, cap=CAP_REFINE)
            llm_time += dt
            if not text:
                error = "empty response"
                continue
            code, changes, error = self._parse_edit(text, target.solution)
            if error:
                continue
            code, error = self._filter(code, parent_code=target.solution, splice=changes == "Full rewrite")
            if error:
                continue
            if _ < ATTEMPTS - 1:
                error = self._condemned(code, _score(target.metrics))
                if error:
                    continue
            return self._cand(s, "repair", code, _norm_hash(code), target, llm_time,
                              {"system": self.system, "user": user}, text, changes)
        return None

    def _parse_edit(self, text: str, parent_code: str) -> Tuple[Optional[str], Optional[str], Optional[str]]:
        if extract_diffs(text):
            code, err = _strict_apply(parent_code, text)
            return (code, format_diff_summary(extract_diffs(text)), None) if code else (None, None, err)
        code = _fenced(text)
        if code is None:
            return None, None, ("no SEARCH/REPLACE edits and no ```python fenced program found; "
                                "answer with edits in the required format")
        return code, "Full rewrite", None

    def _condemned(self, code: str, parent_score: float) -> Optional[str]:
        """SIFT: refuse to spend an evaluation the surrogate predicts will score zero.
        Returns a retry message, or None to let the candidate through."""
        if not self.sift.condemns(code, parent_score):
            return None
        self.sift.rejected += 1
        return ("this program resembles ones that have already scored zero in this run; "
                "change the part that would violate a constraint, not the surface")

    # ------------------------------------------------------------ pre-evaluation filter (3.5)
    def _filter(self, code: str, parent_code: str, splice: bool) -> Tuple[Optional[str], Optional[str]]:
        if splice:
            init_parts, cand_parts = _split_block(self.initial_solution), _split_block(code)
            if init_parts and cand_parts:
                code = init_parts[0] + cand_parts[1] + init_parts[2]
        if len(code) > MAX_LEN:
            return None, f"program is {len(code)} characters, over the {MAX_LEN} limit"
        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            return None, f"the program does not parse: {e.msg} at line {e.lineno}"
        missing = [n for n in self.required if n not in _top_level_names(tree)]
        if missing:
            return None, f"the program no longer defines {', '.join(missing)}, which the evaluator calls"
        for cls, methods in self.contract.items():
            for m in methods:
                if not self._provides(tree, m):
                    return None, (f"the program no longer provides {cls}.{m}, which the evaluator calls on "
                                  f"the object your function returns; keep it defined")
        if code.strip() == (parent_code or "").strip():
            return None, "the edit did not change the program"
        if _norm_hash(code) in self.cache:
            return None, "this exact program has already been evaluated; make a real change"
        return code, None

    # ------------------------------------------------------------ deterministic notes and ledger
    def _note_feedback(self, prog: Program):
        fb = (prog.artifacts or {}).get("feedback") or ""
        for line in fb.splitlines():
            if ": error" not in line and ": timeout" not in line:
                continue
            msg = line.split("(", 1)[1].rstrip(")") if "(" in line else line
            key = re.sub(r"\d+(\.\d+)?", "N", msg)
            key = re.sub(r"'[^']*'|\"[^\"]*\"", "S", key)[:120]
            entry = self.notes.setdefault(key, [0, msg.strip()[:200]])
            entry[0] += 1

    def _notes(self) -> str:
        if not self.notes:
            return "(nothing rejected yet)"
        rows = sorted(self.notes.values(), key=lambda e: -e[0])[:6]
        text = "\n".join(f"- {n}x: {ex}" for n, ex in rows)
        return text[:NOTES_CHARS]

    def _notes_block(self) -> str:
        return "" if not self.notes else "\n## Evaluator rejections so far in this run\n" + self._notes() + "\n"

    def _ledger(self) -> str:
        rows = []
        for l in self.lineages:
            rows.append(f"- [{l.status.lower()}] {l.title} ({l.kind}): {l.n} evaluations, best {l.best:.6g}, "
                        f"{self.ratio(l):.2f} of the best" + (f", closed: {l.close_reason}" if l.close_reason else ""))
        text = "\n".join(rows)
        return text if len(text) <= LEDGER_CHARS else text[:LEDGER_CHARS] + "\n- ..."

    # ------------------------------------------------------------ logs
    def _log_stats(self, prog: Program, s: Lineage, role: str):
        if not self._stats_path:
            return
        rec = {
            "iteration": self.iteration, "timestamp": datetime.now().isoformat(),
            "evaluations": self.E, "progress": round(self.progress(), 4),
            "global": {"global_best_score": self.g_star},
            "iteration_result": {"success": True, "child_program": {
                "id": prog.id, "metrics": prog.metrics, "parent_id": prog.parent_id}},
            "sprout": {"lineage": s.id, "role": role, "status": s.status, "incumbent": self.inc.id if self.inc else None,
                       "challengers": [c.id for c in self.challengers],
                       "nursery": self.nursery.id if self.nursery else None, "births": self.births,
                       "window_incumbent": self.window(self.inc) if self.inc else None,
                       "dry_incumbent": self.inc.dry if self.inc else None,
                       "sift_rejected": self.sift.rejected, "sift_fits": self.sift.fits,
                       "sift_threshold": self.sift.threshold,
                       "harvest": ({**self.struct.stats(), "ops": self.ops.stats()}) if HARVEST else None,
                       "cluster": self.struct.of_program.get(prog.id) if HARVEST else None},
        }
        try:
            with open(self._stats_path, "a") as fh:
                fh.write(json.dumps(rec, default=str) + "\n")
        except OSError as e:
            logger.warning("sprout: stats log failed: %s", e)

    def _write_lineages(self):
        if not self._lineage_path:
            return
        data = {"E": self.E, "births": self.births, "g_star": self.g_star, "b0": self.b0,
                "incumbent": self.inc.id if self.inc else None, "ref8": self.ref8,
                "notes": self.notes, "lineages": [l.summary() for l in self.lineages]}
        tmp = self._lineage_path + ".tmp"
        try:
            with open(tmp, "w") as fh:
                json.dump(data, fh, indent=1, default=str)
            os.replace(tmp, self._lineage_path)
        except OSError as e:
            logger.warning("sprout: lineage dump failed: %s", e)


def install():
    """Register search type `sprout`. Called from run_capped.py when SPROUT=1."""
    _DB_CONFIG_BY_TYPE["sprout"] = AdaEvolveDatabaseConfig     # archive/intensity defaults
    register_database("sprout", SproutDatabase)
    register_controller("sprout", SproutController)
    from skydiscover.optimize.search.route import _CONTROLLER_REGISTRY
    assert _CONTROLLER_REGISTRY.get("sprout") is SproutController, "sprout controller not registered"
    logger.warning("sprout installed: prompts=%s cap=%d N_MAX=%d", PROMPTS, EVAL_CAP, N_MAX)
