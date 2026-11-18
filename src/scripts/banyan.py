"""BANYAN: an incumbent-rooted credibility tree (design/round7/BANYAN.md; section numbers below refer to it).

Opt-in from run_capped.py with BANYAN=1 and `search.type: banyan`. It builds on CredTree (credtree.py: the same
SkyDiscover hooks, database, candidate pipeline, evaluation and logging machinery are imported and subclassed, and
CredTree's own behaviour is untouched) and replaces the scheduler (banyan_core.py), the prompts (prompts/banyan),
the diagnoser gate (banyan_diag.py) and the commit/log step.

Where the evaluations go (banyan_core.Tree.next_target):
  trunk   edits on the incumbent (its parent: the node's best 0.8 of the time, else a near-best program)
  branch  framings whose attempts are SEARCH/REPLACE grafts on the incumbent; each stores anchor_G
  stage 1 draws the trunk's credibility against one pooled "branch out" draw, stage 2 picks the framing / a new one;
  a record in a branch RE-ROOTS: that attempt becomes the trunk, everything else freezes, productive frozen
  branches are queued for re-graft.
Every evaluation spends one of the 500; every other LLM call (framings, plans, hypotheses, reviews, diagnoses) is
free. Hyper-parameters are module constants of banyan_core: identical on all five tasks (handout 2.3). The only
task-specific inputs are the prompts and what is read from the evaluator source (its TIME_LIMIT, its entry points).

What is new relative to CredTree v3 (all switchable by env, defaults below):
  clock pacing      code-writing prompts state the evaluator's per-call limit L and require an anytime main loop
                    stopping at t0 + D, D = min(0.6*L, 0.5*T/cap/waves); edit prompts show the parent's headroom (6)
  map + inspiration every code / idea prompt shows the tree with a label per node and one other branch's change (6)
  symbol check      every loaded name must be defined, imported or builtin, or the candidate is rejected free (8)
  noise margin      eps = 2 x median |re-run - logged| of evaluated programs (>= 3 samples), else 1e-9 relative (4, 9)
  diagnoser         ON (BANYAN_DIAG=1) with the same-output replay gate, so clock-paced programs can be diagnosed (9)
  lessons           OFF (BANYAN_LESSONS=0): an ablation, the principle behind them has no evidence
"""
import ast
import asyncio
import builtins
import difflib
import glob
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

import banyan_core as core
import banyan_diag
import banyan_replay as replay
import credtree as ct
from credtree import (CredTreeDatabase, CredTreeController, _error_text, _feedback, _norm_hash, _parse_json, _score,
                      _split_block, _wall_seconds)
from skydiscover.optimize.config import AdaEvolveDatabaseConfig, _DB_CONFIG_BY_TYPE
from skydiscover.optimize.search.base_database import Program
from skydiscover.optimize.search.registry import register_controller, register_database
from skydiscover.optimize.search.utils.discovery_utils import SerializableResult, load_evaluator_code

logger = logging.getLogger("banyan")

# ---------------------------------------------------------------- flags and constants
EVALSRC = os.environ.get("BANYAN_EVALSRC", "1") == "1"       # the generator sees the evaluator source (v2)
DIAG = os.environ.get("BANYAN_DIAG", "1") == "1"             # diagnoser on by default (the user's decision)
LESSONS = os.environ.get("BANYAN_LESSONS", "0") == "1"       # per-framing lessons: off, an ablation
CLOCK_FRAC = float(os.environ.get("BANYAN_CLOCK_FRAC", "0.6"))   # the deadline never exceeds CLOCK_FRAC * L
# Budget-aware deadline: a program may use at most BUDGET_FRAC of the per-evaluation share of the wall clock,
# T / EVAL_CAP, spread over the evaluator's waves of instances, so that clock pacing leaves room for EVAL_CAP
# evaluations inside the wall clock. D = min(CLOCK_FRAC * L, BUDGET_FRAC * T / EVAL_CAP / waves).
BUDGET_FRAC = float(os.environ.get("BANYAN_BUDGET_FRAC", "0.5"))
# Resume a run that died (jobs/22_resume.pbs): the runner reloads the program database from the last checkpoint
# (--checkpoint) and the controller rebuilds its tree and state from the run's own logs (_resume), keeping its
# deadline D, its wall clock T and its evaluation count. Off: a fresh run, unchanged.
RESUME = os.environ.get("BANYAN_RESUME", "0") == "1"
# Ablations (design/round7/BANYAN_ABLATIONS.md); defaults are the spec.
#   BANYAN_P2=0           standing tree: no stage-1 trunk/branch draw (the trunk is one option among the framings)
#                         and no re-rooting; new grafts go on the global best program
#   BANYAN_ALLOC=random   every draw is uniform: the tree, probes and prompts stay, allocation ignores credibility
#   (BANYAN_DIAG=0        diagnoser off, above)
#   BANYAN_FLAT=1         flat tree: no framing -> attempt levels. Every node is a child of the root (the incumbent),
#                         written straight from it by the edit prompts with no framing card and no plan; the same
#                         draws pick a child or a new one (widening), records re-root; no re-grafts (they are per framing)
P2 = os.environ.get("BANYAN_P2", "1") == "1"
ALLOC = os.environ.get("BANYAN_ALLOC", "thompson")
assert ALLOC in ("thompson", "random"), ALLOC
FLAT = os.environ.get("BANYAN_FLAT", "0") == "1"
FLAT_FRAMING = "(none: this tree has no framings; every node is a child program written directly from the incumbent)"
FLAT_PLAN = "A new child of the incumbent."
COMPUTE_HEADROOM = 0.3                                        # a compute move is allowed above this headroom
RERUN_PARALLEL = int(os.environ.get("BANYAN_RERUN_PARALLEL", "4"))
EPS_MIN_SAMPLES = 3
EPS_NOISE_MULT = 2.0
EPS_DEFAULT_REL = 1e-9
MAP_CHARS, INSPIRATION_CHARS = 2600, 1800
TEMPLATES = ("framing_describe", "framing_new", "framing_judge", "plan", "seed_graft", "micro_idea_diag", "edit_diag",
             "review", "diagnoser", "verify")

CARD_FIELDS = ("problem_class", "decision_variables", "objective_restatement", "assumptions_relaxed",
               "why_it_could_beat_current_best", "search_space", "inner_search", "time_use") + \
    (("evaluator_audit",) if EVALSRC else ())
VALIDITY = ("the program must pass every check the evaluator actually performs, on every input it may receive; "
            "the evaluator source decides what is valid (see the evaluator audit)") if EVALSRC else \
    "every check and constraint in your system message must hold on every input the program may receive"
REVIEW_BASIS = ("every check the evaluator source below actually performs, and every environment limit in your system "
                "message (installed packages, time limits); a rule the evaluator does not enforce is not a defect") \
    if EVALSRC else "every check, constraint and environment limit in your system message"
AUDIT_FIELD = (',\n  "evaluator_audit": "from the evaluator source: what it actually checks, what it does not '
               'check, and how this formulation uses that"') if EVALSRC else ""
_NAME_ERROR_RE = re.compile(r"NameError|UnboundLocalError")
SYMBOL_MSG = "are not defined, imported or builtin"

_HERE = os.path.dirname(os.path.abspath(__file__))


def _prompt_dir() -> str:
    for d in (os.environ.get("BANYAN_PROMPTS"), os.path.join(os.path.dirname(_HERE), "prompts", "banyan"),
              os.path.join(_HERE, "prompts", "banyan")):
        if d and os.path.isdir(d):
            return d
    raise FileNotFoundError("prompts/banyan not found (set BANYAN_PROMPTS)")


_TPL: Dict[str, str] = {}


def _tpl(name: str) -> str:
    if name not in _TPL:
        with open(os.path.join(_prompt_dir(), f"{name}.txt")) as fh:
            _TPL[name] = fh.read()
    return _TPL[name]


_DEFAULT_BLANK = ("evaluator", "clock", "headroom", "map", "inspiration", "saturated", "rejection", "clock_check",
                  "diagnosis", "lessons")


def _fill(name: str, **kw) -> str:
    """Substitute {key} for the given keys only; any other brace (JSON, code) is left alone."""
    kw.setdefault("validity", VALIDITY)
    kw.setdefault("review_basis", REVIEW_BASIS)
    kw.setdefault("audit_field", AUDIT_FIELD)
    for k in _DEFAULT_BLANK:
        kw.setdefault(k, "")
    return re.sub(r"\{([a-z_0-9]+)\}", lambda m: str(kw[m.group(1)]) if m.group(1) in kw else m.group(0), _tpl(name))


# ---------------------------------------------------------------- helpers read from the evaluator / program
def time_limit(evaluator_src: str) -> Optional[float]:
    """The per-call limit L: the evaluator's TIME_LIMIT constant (also unpacked from `TIME_LIMIT, GRACE = a, b`)."""
    try:
        tree = ast.parse(evaluator_src)
    except SyntaxError:
        return None
    for n in tree.body:
        if not isinstance(n, ast.Assign):
            continue
        for t in n.targets:
            pairs = []
            if isinstance(t, ast.Name):
                pairs = [(t, n.value)]
            elif isinstance(t, ast.Tuple) and isinstance(n.value, ast.Tuple) and len(t.elts) == len(n.value.elts):
                pairs = list(zip(t.elts, n.value.elts))
            for name, val in pairs:
                if isinstance(name, ast.Name) and name.id == "TIME_LIMIT":
                    try:
                        v = float(ast.literal_eval(val))
                    except (ValueError, TypeError):
                        continue
                    if v > 0:
                        return v
    return None


def undefined_names(tree: ast.Module) -> Tuple[List[str], List[str]]:
    """(loaded names that are never bound anywhere in the module nor builtin, those of them that are called).

    Scope-free on purpose: a name bound anywhere (assignment, def, class, import, parameter, loop or comprehension
    target, except / with / walrus / match target, global declaration) counts as defined, so this only flags the
    NameErrors that no reading of the program can excuse (a helper that was never written, a missing import)."""
    bound = set(dir(builtins)) | {"__file__", "__name__", "__doc__", "__package__", "__spec__", "__loader__",
                                  "__builtins__", "__class__", "__debug__"}
    loads: List[ast.Name] = []
    called = set()
    for n in ast.walk(tree):
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            bound.add(n.name)
        elif isinstance(n, ast.arg):
            bound.add(n.arg)
        elif isinstance(n, ast.Name):
            if isinstance(n.ctx, ast.Load):
                loads.append(n)
            else:
                bound.add(n.id)
        elif isinstance(n, ast.ExceptHandler) and n.name:
            bound.add(n.name)
        elif isinstance(n, (ast.Import, ast.ImportFrom)):
            for al in n.names:
                if al.name == "*":
                    return [], []                       # a star import binds unknown names: cannot judge
                bound.add((al.asname or al.name).split(".")[0])
        elif isinstance(n, (ast.Global, ast.Nonlocal)):
            bound.update(n.names)
        elif isinstance(n, ast.Call) and isinstance(n.func, ast.Name):
            called.add(n.func.id)
        for attr in ("name", "rest"):                    # match-statement captures
            if isinstance(n, (ast.MatchAs, ast.MatchStar, ast.MatchMapping)) and getattr(n, attr, None):
                bound.add(getattr(n, attr))
    missing = sorted({n.id for n in loads if n.id not in bound})
    return missing, [m for m in missing if m in called]


def _fmt_card(card: Dict[str, str]) -> str:
    return (f"Problem class: {card.get('problem_class', '')}\n"
            f"Decision variables: {card.get('decision_variables', '')}\n"
            f"Objective restatement: {card.get('objective_restatement', '')}\n"
            f"Assumptions relaxed: {card.get('assumptions_relaxed', '')}\n"
            f"Why it could beat the incumbent: {card.get('why_it_could_beat_current_best', '')}\n"
            f"Search space (at run time): {card.get('search_space', '')}\n"
            f"Inner search: {card.get('inner_search', '')}\n"
            f"Time use: {card.get('time_use', '')}"
            + (f"\nEvaluator audit: {card['evaluator_audit']}" if card.get("evaluator_audit") else ""))


def _parse_card(text: Optional[str]) -> Optional[Dict[str, str]]:
    d = _parse_json(text)
    if not d:
        return None
    card = {}
    for k in CARD_FIELDS:
        v = d.get(k)
        if isinstance(v, (list, tuple)):
            v = "; ".join(map(str, v))
        if not isinstance(v, str) or not v.strip():
            return None
        card[k] = v.strip()[:600]
    return card


def _title(card: Dict[str, str], n: int = 70) -> str:
    return re.sub(r"\s+", " ", card.get("problem_class") or "?")[:n]


# ---------------------------------------------------------------- controller
class BanyanController(CredTreeController):
    def __init__(self, controller_input):
        super().__init__(controller_input)
        self.tree: Optional[core.Tree] = None
        self.L = time_limit(self._evaluator_src)               # per-call limit, read from the evaluator
        self.eval_parallel = max(1, int(os.environ.get("EVAL_PARALLEL", "8")))
        self.D = CLOCK_FRAC * self.L if self.L else None       # deadline per call; budget-aware once b0 is known
        self.overhead = 0.0                                    # wall of the (instant) initial program: fork, import, load
        self.timing: Dict[str, Dict[str, float]] = {}          # program id -> {"wall", "inst", "headroom"}
        self.eps_samples: List[float] = []                     # |re-run score - logged score|, raw units
        self.best_id: Optional[str] = None                     # best valid program by score (AdaEvolve's "best", for P2)
        self.best_score = 0.0
        self._diag_tasks: Dict[str, asyncio.Future] = {}
        self._noise_tasks: set = set()
        self._cur: Dict[str, Any] = {}
        self._stage: Tuple[str, Optional[float], Optional[float]] = ("", None, None)   # stage-1 outcome of the creating step
        self._eval_view = ""
        if EVALSRC:
            try:
                self._eval_view = load_evaluator_code(self.evaluation_file)[:ct.EVAL_SRC_CHARS]
            except Exception as e:
                logger.warning("banyan: evaluator source unavailable: %s", e)
                self._eval_view = self._evaluator_src[:ct.EVAL_SRC_CHARS]
        self.counters.update({"symbol_rejections": 0, "name_errors": 0, "reroots": 0, "regrafts_started": 0,
                              "regraft_dups": 0, "graft_seeds": 0, "graft_seeds_valid": 0, "diag_shown": 0,
                              "diag_missing": 0, "hypotheses": 0, "noise_reruns": 0})

    # ------------------------------------------------------------ logs
    def _setup_logs(self):
        super()._setup_logs()
        out = self.output_dir or "."
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._stats_path = os.path.join(out, f"banyan_iteration_stats_{ts}.jsonl")
        if RESUME:                                                    # a resumed run appends to its own stats file
            self._stats_path = sorted(glob.glob(os.path.join(out, "banyan_iteration_stats_*.jsonl")))[-1]
        self._dump_path = os.path.join(out, "banyan_tree.json")      # tree_stats.jsonl / tree_events.jsonl keep their names

    def _write_tree(self):
        if not self._dump_path or not self.tree:
            return
        t = self.tree
        data = {"E": self.E, "N": t.N, "G": t.G, "b0": self.b0, "eps": t.eps, "eps_samples": self.eps_samples,
                "trunk": t.trunk.id if t.trunk else None, "stats": t.stats, "labels": t.labels(),
                "exhausted": t.exhausted, "counters": self.counters, "haz": t.haz, "L": self.L,
                "clock_frac": CLOCK_FRAC, "overhead": self.overhead,
                "diag_paths": getattr(self.diag, "paths", None),
                "framings": [{"id": f.id, "card": f.card, "n": t.n_framing(f), "closed": f.closed,
                              "attempts": [{"id": a.id, "plan": a.plan, "best": a.best, "n": a.n, "valid": a.valid,
                                            "age": a.age, "s": a.s, "E": a.E, "anchor": a.anchor, "frozen": a.frozen,
                                            "frozen_at": a.frozen_at, "trunk": a is t.trunk, "label": t.label(a),
                                            "p_gt1": t.p_gt1(a), "probe_done": a.probe_done, "cut": a.cut,
                                            "seed_valid": a.seed_valid, "programs": len(a.data.get("programs", []))}
                                           for a in f.attempts]} for f in t.framings]}
        tmp = self._dump_path + ".tmp"
        try:
            with open(tmp, "w") as fh:
                json.dump(data, fh, indent=1, default=str)
            os.replace(tmp, self._dump_path)
        except OSError as e:
            logger.warning("banyan: tree dump failed: %s", e)

    # ------------------------------------------------------------ text blocks read by the prompts
    def _clock(self) -> str:
        if not self.L:
            return ""
        L, D = self.L, self.D
        return ("\n## Time limit and the anytime loop (applies to every program you write)\n"
                f"The evaluator allows {L:g} seconds per call of the program's entry point; a call that runs longer scores "
                f"0. Write the program as an ANYTIME search paced by the clock: take t0 = time.perf_counter() at the start "
                f"of the entry call and stop the main loop at deadline = t0 + {D:.3g} seconds, "
                "checking the clock inside the loop (not only between restarts). Keep the best valid result found so far "
                "and always return it, however the loop ends. A fixed iteration or restart count alone must not decide how "
                "long the program runs: the deadline does, and the design question is how to spend that time (restarts "
                "against iterations, construction then polish). This replaces any advice in your system message to let "
                "the result depend on the work done rather than on elapsed time: the margin that advice asks for is the "
                f"{(1 - D / L) * 100:.0f}% of the limit left after the deadline.\n")

    def _clock_check(self) -> str:
        if not self.L:
            return ""
        return (f", and that the main loop stops at deadline = t0 + {self.D:.3g} s measured with "
                "time.perf_counter() and returns the best valid result found so far")

    def _headroom(self, prog: Optional[Program]) -> str:
        """The parent's measured time use and whether a compute move is allowed (spec 6)."""
        if not self.L:
            return ""
        tm = self.timing.get(prog.id) if prog else None
        if tm is None:
            return ("\n## Time used by this program\n(not measured for this program; treat the headroom as unknown: a "
                    "compute move is only justified if the diagnosis shows the search is cut off by time)\n")
        h = tm["headroom"]
        rule = ("A COMPUTE MOVE is allowed: the change may spend more of the deadline on the inner search, or re-split "
                "how the time is used (restarts against iterations, construction then polish)." if h > COMPUTE_HEADROOM
                else f"Headroom is not above {COMPUTE_HEADROOM:g}: do not add compute; change the logic, not the time.")
        return ("\n## Time used by this program (measured on its last evaluation)\n"
                f"about {tm['inst']:.1f} s of the {self.D:.3g} s deadline per call (limit {self.L:g} s): headroom = "
                f"1 - used/deadline = {h:.2f}. "
                f"{rule}\n")

    def _framing(self, f: core.Framing) -> str:
        return FLAT_FRAMING if FLAT else _fmt_card(f.card)

    def _tree_cfg(self) -> Dict[str, Any]:
        return dict(two_stage=P2, reroot=P2, alloc=ALLOC, **(dict(max_framings=1, regraft=False) if FLAT else {}))

    # ---- the map (P5a)
    def _child_line(self, a: core.Attempt) -> str:
        """BANYAN_FLAT=1: one map line per child of the root."""
        t = self.tree
        state = "TRUNK (the incumbent's line)" if a is t.trunk else f"frozen at re-root {a.frozen_at}" if a.frozen else "open"
        best = a.best if a.best is None else format(a.best, ".6g")
        return (f"[child {a.id}] {a.plan[:90]} | {state} | {a.n} evals, best {best}, rho {t.mean_rho(a):.2f}, "
                f"P(rho>1) {t.p_gt1(a):.2f} | {t.label(a) or 'no label'}")

    def _node_line(self, f: core.Framing) -> str:
        t = self.tree
        atts = [a for a in f.attempts if a.best is not None]
        best = max((a.best for a in atts), default=None)
        live = [a for a in f.attempts if not a.frozen or a is t.trunk]
        pick = max(live or f.attempts, key=lambda a: (t.p_gt1(a), a.n)) if f.attempts else None
        lab = (t.label(pick) or "no label") if pick is not None else "untried"
        if f.attempts and t.trunk in f.attempts:
            state = "TRUNK (the incumbent's line)"
        elif live:
            state = "open"
        else:
            fz = max((a.frozen_at for a in f.attempts), default=0)
            state = f"frozen at re-root {fz}"
        rho = f", rho {t.mean_rho(pick):.2f}, P(rho>1) {t.p_gt1(pick):.2f}" if pick is not None else ""
        return (f"[framing {f.id}] {_title(f.card)} | {state} | {t.n_framing(f)} evals, {len(f.attempts)} attempts, best "
                f"{best if best is None else format(best, '.6g')}{rho} | {lab}")

    def _map(self) -> str:
        t = self.tree
        if t is None or t.trunk is None:
            return "(no tree yet)"
        order = sorted(t.framings, key=lambda f: (0 if t.trunk in f.attempts else 1 if any(not a.frozen for a in f.attempts)
                                                   else 2, -t.n_framing(f)))
        rows = [f"Incumbent score {t.G:.6g} (initial program {self.b0:.6g}); {t.stats['reroots']} re-roots so far. "
                "Credibility rho = how often an evaluation in the node raises the incumbent, compared with what its age "
                "predicts (1 = as expected); labels: productive P(rho>1) >= 0.7, saturated = >= 8 valid evaluations and "
                "P(rho>1) <= 0.2, untried = probe unfinished."]
        if FLAT:
            kids = sorted(t.attempts, key=lambda a: (0 if a is t.trunk else 2 if a.frozen else 1, -a.n))
            rows += [self._child_line(a) for a in kids[:14]]
        else:
            rows += [self._node_line(f) for f in order[:14]]
        text = "\n".join(rows)
        return text if len(text) <= MAP_CHARS else text[:MAP_CHARS] + "\n..."

    def _saturated_note(self) -> str:
        t = self.tree
        rows = []
        for f in t.framings:
            for a in f.attempts:
                if t.label(a) == "saturated":
                    rows.append(f"{_title(f.card, 60)} (tried on incumbent {a.anchor:.6g})")
        if not rows:
            return ""
        return ("\nFamilies labelled saturated were tried with no gain on the incumbent shown in brackets; do not propose "
                "them again unless the new incumbent changes what they would do: " + "; ".join(rows[:8]) + "\n")

    def _inspiration(self, a: Optional[core.Attempt]) -> str:
        """The change (diff against the incumbent) carried by the best program of the most credible OTHER branch."""
        t = self.tree
        cands = [b for b in t.open_attempts() if b is not a and any(p["valid"] for p in b.data.get("programs", []))]
        if not cands:
            return ""
        b = max(cands, key=lambda b: (t.mean_rho(b), b.best if b.best is not None else -math.inf))
        best = max((p for p in b.data["programs"] if p["valid"]), key=lambda p: p["score"])
        prog, inc = self._program(best["id"]), self._incumbent()
        if prog is None or inc is None or prog.id == inc.id:
            return ""
        diff = "\n".join(l for l in difflib.unified_diff(inc.solution.splitlines(), prog.solution.splitlines(), lineterm="",
                                                           n=1) if not l.startswith(("---", "+++")))
        if not diff.strip():
            return ""
        if len(diff) > INSPIRATION_CHARS:
            diff = diff[:INSPIRATION_CHARS] + "\n... (diff truncated)"
        f = t.framings[b.fid]
        name = f"child {b.id}: {b.plan[:60]}" if FLAT else _title(f.card, 60)
        return (f"\n## One change being tried elsewhere in the tree\nBest program of the most credible other branch "
                f"({name}; score {best['score']:.6g} against the incumbent {t.G:.6g}); diff against the "
                f"incumbent:\n```diff\n{diff}\n```\n")

    def _incumbent(self) -> Optional[Program]:
        """The incumbent program I: the best program of the trunk (BANYAN_P2=0: the global best, which a standing
        tree's trunk need not hold)."""
        if not P2:
            return self._program(self.g_id)
        tr = self.tree.trunk if self.tree else None
        members = [p for p in (tr.data.get("programs", []) if tr else []) if p["valid"]]
        if members:
            return self._program(max(members, key=lambda p: p["score"])["id"])
        return self._program(self.g_id)

    # ------------------------------------------------------------ framings (7)
    def _cards(self, exclude_ids=()) -> str:
        t = self.tree
        rows = []
        for f in t.framings:
            best = t.best_attempt(f)
            rows.append(f"[framing {f.id}] best score reached {best.best if best and best.best is not None else float('nan'):.6g}, "
                        f"{t.n_framing(f)} evaluations spent"
                        f"{' (frozen history)' if f.closed else ''}\n" + _fmt_card(f.card))
        text = "\n\n".join(rows)
        return text if len(text) <= ct.CARDS_CHARS else text[-ct.CARDS_CHARS:]

    def _live_framings(self, exclude: Optional[int] = None) -> List[core.Framing]:
        """The branches grown from the CURRENT incumbent: dedup compares with these only (spec 7)."""
        return [f for f in self.tree.framings if not f.closed and f.id != exclude]

    async def _describe_initial(self, seed: Program) -> Dict[str, str]:
        user = _fill("framing_describe", score=f"{self.b0:.6g}", program=self._view(seed.solution),
                     evaluator=self._evaluator_section(), clock=self._clock())
        for _ in range(3):
            text, _ = await self._llm(user)
            card = _parse_card(text)
            if card:
                return card
        c = {"problem_class": "the formulation of the initial program", "decision_variables": "as in the initial program",
             "objective_restatement": "as in the task statement", "assumptions_relaxed": "unknown",
             "why_it_could_beat_current_best": "n/a (this is the formulation of the current program)",
             "search_space": "as in the initial program", "inner_search": "as in the initial program",
             "time_use": "as in the initial program"}
        if EVALSRC:
            c["evaluator_audit"] = "not written (the description call failed)"
        return c

    async def _judge(self, card: Dict[str, str], exclude: Optional[int] = None) -> Tuple[bool, str, Optional[int]]:
        """Free dedup judge against the branches of the current incumbent only."""
        short = "\n\n".join(f"[framing {f.id}]\nProblem class: {f.card.get('problem_class', '')}\n"
                            f"Decision variables: {f.card.get('decision_variables', '')}"
                            for f in self._live_framings(exclude))
        if not short:
            return False, "", None
        cand = f"Problem class: {card['problem_class']}\nDecision variables: {card['decision_variables']}"
        self.counters["dedup_calls"] += 1
        text, _ = await self._llm(_fill("framing_judge", candidate=cand, cards=short))
        d = _parse_json(text)
        if not d or "duplicate" not in d:
            return False, "", None
        dup = d["duplicate"] is True or str(d["duplicate"]).lower() == "true"
        m = d.get("matches")
        return dup, str(d.get("reason", ""))[:300], (m if isinstance(m, int) else None)

    async def _propose_framing(self) -> Optional[Dict[str, str]]:
        """Generator + judge, up to DEDUP_RETRIES tries. None means the root is exhausted (spec 7)."""
        inc = self._incumbent()
        diag = None
        if self.diag is not None:
            diag = await self._diagnosis(inc, wait=True)          # what binds the incumbent (verified), if it can be had
            self.counters["diag_shown" if diag else "diag_missing"] += 1
        rejection = ""
        for k in range(core.DEDUP_RETRIES):
            user = _fill("framing_new", score=f"{self.tree.G:.6g}", program=self._view(inc.solution),
                         cards=self._cards(), rejection=rejection, evaluator=self._evaluator_section(),
                         diagnosis=diag or self._fallback(inc), map=self._map(), saturated=self._saturated_note(),
                         clock=self._clock())
            text, _ = await self._llm(user)
            card = _parse_card(text)
            if card is None:
                rejection = (f"\n## Your previous answer was rejected\nIt was not a JSON object with the {len(CARD_FIELDS)} "
                             "required string fields; answer with exactly that object.\n")
                self._event("framing_try", k=k, result="unparsable")
                continue
            dup, reason, match = await self._judge(card)
            self._event("framing_try", k=k, result="duplicate" if dup else "new", match=match, reason=reason,
                        problem_class=card["problem_class"])
            if not dup:
                return card
            self.counters["dedup_dups"] += 1
            rejection = (f"\n## Your previous answer was rejected as a duplicate\n"
                         f"Previous framing: {card['problem_class']} / {card['decision_variables']}\n"
                         f"Judge: {reason}\nReformulate: the new framing must differ in problem class or decision "
                         f"variables from every live framing above, and from the rejected one.\n")
        return None

    async def _make_plan(self, card: Dict[str, str], f: Optional[core.Framing]) -> Optional[str]:
        inc = self._incumbent()
        sib = "(none yet)"
        if f is not None and f.attempts:
            sib = "\n".join(f"- ({a.best:.6g}{', on an older incumbent' if a.frozen else ''}) {a.plan[:500]}"
                            for a in f.attempts)
        user = _fill("plan", framing=_fmt_card(card), score=f"{self.tree.G:.6g}", program=self._view(inc.solution),
                     siblings=sib, map=self._map(), clock=self._clock())
        for _ in range(3):
            text, _ = await self._llm(user)
            if not text:
                continue
            m = re.search(r"PLAN:\s*(.+)", text, re.S)
            plan = re.sub(r"\s+", " ", (m.group(1) if m else text).strip().strip("`").strip())[:1200]
            if len(plan) >= 40:
                return plan
        return None

    async def _grow_framing(self) -> bool:
        card = await self._propose_framing()
        if card is None:
            self.tree.mark_exhausted()
            logger.warning("banyan: root exhausted after %d duplicate framings in a row at E=%d (reopens after %d "
                           "evaluations without a record, or at a re-root)", core.DEDUP_RETRIES, self.E, core.REOPEN_AFTER)
            self._event("exhausted")
            return False
        plan = await self._make_plan(card, None)
        if plan is None:
            logger.warning("banyan: no plan could be written for the new framing; dropped")
            return False
        f = self.tree.add_framing(card)
        logger.warning("banyan: framing %d = %s | %s | search: %s", f.id, card["problem_class"], card["decision_variables"],
                       card.get("search_space"))
        self._event("framing", framing=f.id, card=card, source="generated")
        await self._start_attempt(f, plan, "new")
        return True

    async def _grow_attempt(self, f: core.Framing) -> bool:
        plan = await self._make_plan(f.card, f)
        if plan is None:
            logger.warning("banyan: no plan could be written under framing %d; skipped", f.id)
            return False
        await self._start_attempt(f, plan, "new")
        return True

    async def _regraft(self, f: core.Framing) -> bool:
        """A productive frozen branch re-grafted on the new incumbent: a free plan call, a dedup check against the
        branches of the current incumbent, then a normal probe."""
        plan = await self._make_plan(f.card, f)
        if plan is None:
            return False
        dup, reason, match = await self._judge(f.card, exclude=f.id)
        if dup:
            self.counters["regraft_dups"] += 1
            self._event("regraft_dup", framing=f.id, match=match, reason=reason)
            return False
        self.counters["regrafts_started"] += 1
        self._event("regraft", framing=f.id, G=self.tree.G)
        await self._start_attempt(f, plan, "regraft")
        return True

    # ------------------------------------------------------------ attempts and seeds
    async def _start_attempt(self, f: core.Framing, plan: str, how: str):
        a = self.tree.add_attempt(f, plan)
        a.data = {"programs": [], "history": []}
        logger.warning("banyan: attempt %d under framing %d (anchor_G=%.6g) at E=%d: %s", a.id, f.id, a.anchor, self.E,
                       plan[:160])
        self._event("attempt", attempt=a.id, framing=f.id, plan=plan, anchor=a.anchor, how=how)
        st = self._stage
        self._cur = {"how": how, "d": None, "d_trunk": st[1], "d_branch": st[2], "stage1": st[0]}
        await self._seed(a, f)

    async def _seed(self, a: core.Attempt, f: core.Framing):
        """The graft: SEARCH/REPLACE edits on the incumbent that carry out plan + framing; 1 evaluation. No repair."""
        parent = self._incumbent()
        hr, ins, mp = self._headroom(parent), self._inspiration(a), self._map()

        def build(error: str, lean: bool) -> str:
            return _fill("seed_graft", framing=_fmt_card(f.card), plan=a.plan, score=f"{_score(parent.metrics):.6g}",
                         program=parent.solution, required=", ".join(self.required) or "(see system message)",
                         notes="(omitted)" if lean else self._notes(), rejected=error, clock=self._clock(),
                         headroom="" if lean else hr, map="(omitted)" if lean else mp,
                         inspiration="" if lean else ins)

        self.counters["seed_tries"] += 1
        self.counters["graft_seeds"] += 1
        cand = await self._candidate(build, parent, "seed", a)
        await self._finish_seed(a, f, parent, cand)

    async def _grow_child(self) -> bool:
        """BANYAN_FLAT=1: a new child of the root. The edit prompts (hypothesis, then SEARCH/REPLACE edits) write it
        straight from the incumbent, with no framing card and no plan; 1 evaluation, logged as the child's seed. Its
        first change becomes the child's plan for the edits that follow."""
        f, parent = self.tree.framings[0], self._incumbent()
        a = self.tree.add_attempt(f, FLAT_PLAN)
        a.data = {"programs": [], "history": []}
        st = self._stage
        self._cur = {"how": "new", "d": None, "d_trunk": st[1], "d_branch": st[2], "stage1": st[0]}
        self.counters["seed_tries"] += 1
        self.counters["graft_seeds"] += 1
        cand = await self._make_edit(a, f, parent=parent)
        if cand is not None and cand.get("micro_idea"):
            a.plan = f"Child of the incumbent; its first change: {cand['micro_idea']}"
        logger.warning("banyan: child %d of the root (anchor_G=%.6g) at E=%d: %s", a.id, a.anchor, self.E, a.plan[:160])
        self._event("attempt", attempt=a.id, framing=f.id, plan=a.plan, anchor=a.anchor, how="new")
        await self._finish_seed(a, f, parent, cand)
        return True

    async def _finish_seed(self, a: core.Attempt, f: core.Framing, parent: Program, cand: Optional[Dict[str, Any]]):
        if cand is None:
            self.counters["empty_slots"] += 1
            logger.warning("banyan: attempt %d: no graft could be written; it starts from the incumbent", a.id)
        else:
            res = (await self._evaluate_many([cand]))[0]
            out = self._commit(cand, res, a, f, "seed")
            self.counters["graft_seeds_valid"] += int(out.valid)
        self.tree.seed_done(a)
        if a.seed_valid is False:                       # the plan survives; the programs are a copy of the incumbent
            a.data["programs"].append({"id": parent.id, "score": _score(parent.metrics), "valid": True})
            self._event("seed_fallback", attempt=a.id, copy_of=parent.id)

    # ------------------------------------------------------------ diagnosis (9): same-output replay gate
    def _on_noise_sample(self, pid: str, delta: float, live: float, logged: float):
        self.eps_samples.append(delta)
        self.counters["noise_reruns"] += 1
        self._event("noise_sample", program=pid, delta=delta, live=live, logged=logged, n=len(self.eps_samples))
        self._update_eps()

    async def _setup_diag(self, seed: Program):
        """Build the run's diagnoser (ReplayDiagnoser). With a zero-scoring initial program there is nothing to
        reproduce yet: the module is installed after the static check only and stays untrusted until it reproduces
        valid programs."""
        self._diag_sem = asyncio.Semaphore(1)
        wd = os.path.join(self.output_dir or ".", "diag")
        os.makedirs(wd, exist_ok=True)

        async def llm(user: str) -> Optional[str]:
            text, _ = await self._llm(user)
            return text

        def build_prompt(sources: str, previous: str, failure: str) -> str:
            fail = (f"\n## Your previous module was rejected\n```python\n{previous[:6000]}\n```\nWhy: {failure}\n"
                    "Fix exactly this problem.\n") if previous else ""
            return _fill("diagnoser", sources=sources, program=self._view(self.initial_solution), failure=fail)

        self.diag = banyan_diag.ReplayDiagnoser(
            self.evaluation_file, wd, llm, build_prompt, timeout=ct.DIAG_TIMEOUT_FRAC * self.T,
            log=lambda kind, **kw: self._event(kind, **kw), parallel=RERUN_PARALLEL, on_sample=self._on_noise_sample)
        if not DIAG:
            return
        refs = [(seed.id, seed.solution, (seed.artifacts or {}).get("feedback") or "", self.b0)] if self.b0 > 0 else []
        ok = await self.diag.build(refs)
        logger.warning("banyan: diagnoser %s (v%d, %d reference programs, %.0fs of re-runs); entry points %s; replay "
                       "paths so far %s", "installed" if ok else f"NOT built: {self.diag.last_failure[:200]}",
                       self.diag.version, len(refs), self.diag.seconds, self.diag.entries, self.diag.paths)

    async def _diagnosis(self, prog: Optional[Program], wait: bool = False) -> Optional[str]:
        """Verified diagnosis of an evaluated program, or None. Cached: returned at once. Not cached: the replay run
        starts in the background (one at a time) and, unless `wait`, the caller goes on without it (a clock-paced
        program takes up to a deadline to re-run, and an edit must not stand idle for that)."""
        d = self.diag
        if not DIAG or d is None or prog is None:
            return None
        if (d.path is None and not self._rebuilding and d.rebuilds < ct.DIAG_REBUILDS
                and len({round(_score(p.metrics), 9) for p in self.database.programs.values()
                         if _score(p.metrics) > 0}) >= banyan_diag.diagmod.TRUST_MIN * (d.rebuilds + 1)):
            self._spawn(self._rebuild_diag())
        if d.path is None:
            return None
        fb, sc = (prog.artifacts or {}).get("feedback") or "", _score(prog.metrics)
        c = d.cache.get(prog.id)
        if c is not None and c["version"] == d.version:
            return await asyncio.to_thread(d.diagnose, prog.id, prog.solution, fb, sc)       # cache hit
        if d.seconds > ct.DIAG_FRAC * self.T:
            return None
        task = self._diag_tasks.get(prog.id)
        if task is None or (task.done() and (d.cache.get(prog.id) or {}).get("version") != d.version):
            task = asyncio.ensure_future(self._diag_job(prog, fb, sc))
            self._diag_tasks[prog.id] = task
        return await asyncio.shield(task) if wait else None

    async def _diag_job(self, prog: Program, fb: str, sc: float) -> Optional[str]:
        d = self.diag
        try:
            async with self._diag_sem:
                text = await asyncio.to_thread(d.diagnose, prog.id, prog.solution, fb, sc)
            if d.needs_rebuild() and not self._rebuilding:
                self._spawn(self._rebuild_diag())
            return text
        except Exception as e:
            logger.warning("banyan: diagnosis job failed: %s", e)
            return None

    async def _noise_job(self, prog: Program):
        """Re-run the incumbent (already evaluated, train split, no new candidate) until there are
        EPS_MIN_SAMPLES noise samples: the margin eps is 2 x median |re-run - logged| (spec 4, 9)."""
        d, sc = self.diag, _score(prog.metrics)
        try:
            for _ in range(2 * EPS_MIN_SAMPLES):          # a failed re-run yields no sample: do not loop for ever
                if len(self.eps_samples) >= EPS_MIN_SAMPLES or sc <= 0 or self._stop():
                    break
                async with self._diag_sem:
                    await asyncio.to_thread(d.noise_run, prog.id, prog.solution, sc)
        except Exception as e:
            logger.warning("banyan: noise re-run failed: %s", e)

    def _maybe_measure_noise(self, prog: Optional[Program]):
        if (DIAG and self.diag is not None and prog is not None and len(self.eps_samples) < EPS_MIN_SAMPLES
                and not self._noise_tasks and self.diag.seconds <= ct.DIAG_FRAC * self.T):
            t = asyncio.ensure_future(self._noise_job(prog))
            self._noise_tasks.add(t)
            t.add_done_callback(self._noise_tasks.discard)

    def _update_eps(self):
        tiny = EPS_DEFAULT_REL * abs(self.tree.G)
        eps = max(EPS_NOISE_MULT * statistics.median(self.eps_samples), tiny) \
            if len(self.eps_samples) >= EPS_MIN_SAMPLES else tiny
        if eps != self.tree.eps:
            self.tree.eps = eps
            self._event("eps", eps=eps, samples=len(self.eps_samples), G=self.tree.G)

    # ------------------------------------------------------------ blocks and leaf edits
    async def _make_edit(self, a: core.Attempt, f: core.Framing, parent: Optional[Program] = None) -> Optional[Dict[str, Any]]:
        """One leaf edit on `a` (its parent drawn from the attempt's programs), or, given `parent`, an edit of it."""
        try:
            if parent is None:
                members = [(p["id"], p["score"], p["valid"]) for p in a.data["programs"]]
                parent = self._program(core.pick_parent(members, self.tree.G, self.b0, self.rng))
            outcomes = "\n".join(f"- {h}" for h in a.data["history"][-ct.HISTORY:]) or "(no edits evaluated yet)"
            diag = await self._diagnosis(parent, wait=False)
            self.counters["diag_shown" if diag else "diag_missing"] += 1
            shown = diag or self._fallback(parent)
            hr, ins, mp = self._headroom(parent), self._inspiration(a), self._map()
            hyp = await self._hypothesis(a, f, parent, outcomes, shown, hr, ins, mp)

            def build(error: str, lean: bool) -> str:
                return _fill("edit_diag", framing=self._framing(f), plan=a.plan, cause=hyp.get("cause") or "(not stated)",
                             idea=hyp.get("change") or "(none: use your judgement)",
                             predict=hyp.get("predict") or "(not stated)", diagnosis="(omitted)" if lean else shown,
                             lessons=self._lessons(f), outcomes=outcomes, score=f"{_score(parent.metrics):.6g}",
                             program=parent.solution, notes="(omitted)" if lean else self._notes(), rejected=error,
                             required=", ".join(self.required) or "(see system message)", clock=self._clock(),
                             headroom="" if lean else hr, inspiration="" if lean else ins)

            cand = await self._candidate(build, parent, "edit", a)
            if cand is not None:
                cand["micro_idea"] = hyp.get("change")
                cand["hyp"] = hyp
                if LESSONS:
                    cand["hypothesis"] = hyp           # the inherited block loop verifies (and learns from) these
                cand["parent_diag"] = diag
                c0 = self.diag.cache.get(parent.id) if self.diag is not None else None
                cand["diag_path"] = (c0 or {}).get("path") or ("pending" if (self.diag is not None and DIAG and self.diag.path)
                                                               else "none")
                cand["headroom_shown"] = (self.timing.get(parent.id) or {}).get("headroom")
            return cand
        except Exception as e:
            logger.exception("banyan: edit on attempt %d failed: %s", a.id, e)
            return None

    async def _hypothesis(self, a, f, parent: Program, outcomes: str, diagnosis: str, headroom: str = "",
                          inspiration: str = "", tree_map: str = "") -> Dict[str, str]:
        user = _fill("micro_idea_diag", framing=self._framing(f), plan=a.plan, score=f"{_score(parent.metrics):.6g}",
                     program=self._view(parent.solution), diagnosis=diagnosis, lessons=self._lessons(f), outcomes=outcomes,
                     headroom=headroom, inspiration=inspiration, map=tree_map or self._map())
        text, _ = await self._llm(user)
        hyp = {}
        for key in ("CAUSE", "CHANGE", "PREDICT"):
            m = re.search(rf"{key}:\s*(.+)", text or "")
            hyp[key.lower()] = re.sub(r"\s+", " ", m.group(1)).strip()[:300] if m else ""
        if not hyp["change"]:
            hyp["change"] = re.sub(r"\s+", " ", next((l for l in (text or "").splitlines() if l.strip()), ""))[:300]
        self.counters["hypotheses"] += int(bool(hyp["cause"] and hyp["predict"]))
        return hyp

    # ------------------------------------------------------------ checks before an eval (8)
    def _static(self, code: str, parent_code: str) -> Optional[str]:
        """CredTree's static pre-check plus the symbol check: every loaded name is defined, imported or builtin."""
        err = super()._static(code, parent_code)
        if err is not None:
            return err
        try:
            tree = ast.parse(code)
        except SyntaxError:
            return None
        missing, called = undefined_names(tree)
        if missing:
            self.counters["symbol_rejections"] += 1
            shown = ", ".join(f"{m}{' (called)' if m in called else ''}" for m in missing[:8])
            return (f"these names {SYMBOL_MSG}: {shown}. Define them in the program or import them (numpy and scipy are "
                    "installed; nothing else may be imported)")
        return None

    async def _review(self, code: str, errors: List[str]) -> Tuple[Optional[str], float]:
        user = _fill("review", program=code, errors="\n".join(f"- {e}" for e in errors) or "(none)",
                     evaluator=self._evaluator_section(audit=False), clock_check=self._clock_check())
        if len(user) > ct.PROMPT_CHAR_BUDGET:
            return None, 0.0
        self.counters["reviews"] += 1
        text, dt = await self._llm(user)
        if not text or not re.search(r"VERDICT:\s*FIX", text, re.I):
            return None, dt
        new, _ = ct._strict_apply(code, text)
        return (new if new and new.strip() != code.strip() else None), dt

    # ------------------------------------------------------------ evaluation and recording
    def _timing(self, c: Dict[str, Any], metrics: Dict[str, Any]) -> Optional[Dict[str, float]]:
        wall = c.get("eval_time")
        if wall is None:
            return None
        n_inst = metrics.get("n_instances") or 1
        waves = max(1, math.ceil(float(n_inst) / self.eval_parallel))
        inst = max(0.0, wall - self.overhead) / waves
        return {"wall": wall, "inst": inst, "headroom": (1.0 - inst / self.D) if self.D else 1.0}

    def _commit(self, c: Dict[str, Any], res, a: core.Attempt, f: core.Framing, role: str,
                first_error: str = "") -> core.Outcome:
        """Record one evaluated candidate: database, core credibility (re-rooting), the three logs."""
        metrics = dict(res.metrics or {})
        score = _score(metrics)
        valid = score > 0 and math.isfinite(score)
        self.E += 1
        self.iteration += 1
        self.cache.add(c["hash"])
        parent = c["parent"]
        t = self.tree
        prev_best, prev_G, prev_anchor, was_trunk = a.best, t.G, a.anchor, a is t.trunk
        since_reroot = t.N - t.reroot_N                  # evaluations since the last re-root (re-root burst accounting)
        inc_before = self.best_id
        mem = [p for p in a.data.get("programs", []) if p["valid"]]
        att_best_id = max(mem, key=lambda p: p["score"])["id"] if mem else None
        draw = dict(t.last_draw)
        out = t.record(a, valid, score, role)
        prog = Program(
            id=c["id"], solution=c["code"], language=self.config.language, metrics=metrics,
            iteration_found=self.iteration, parent_id=parent.id if parent else None, other_context_ids=[],
            generation=(parent.generation + 1) if parent else 0,
            metadata={"changes": c.get("changes"), "parent_metrics": parent.metrics if parent else None,
                      "framing_id": f.id, "attempt_id": a.id, "role": role, "micro_idea": c.get("micro_idea"),
                      "success": out.success, "global_record": out.record, "anchor_G": prev_anchor,
                      "was_trunk": was_trunk},
            artifacts=res.artifacts or {})
        if valid:
            a.data["programs"].append({"id": prog.id, "score": score, "valid": True})
            if score > self.best_score:
                self.best_id, self.best_score = prog.id, score
        if out.record:
            self.g_id, self.g_score = prog.id, score
        tm = self._timing(c, metrics) if valid else None
        if tm:
            self.timing[prog.id] = tm
        pscore = _score(parent.metrics) if parent else 0.0
        err = "" if valid else _error_text(metrics, res.artifacts or {})
        name_error = bool(err and _NAME_ERROR_RE.search(err))
        self.counters["name_errors"] += int(name_error)
        sym_rej = sum(1 for r in (c.get("rejections") or []) if SYMBOL_MSG in r)
        if role == "edit":
            idea = (c.get("micro_idea") or "")[:100]
            msg = (f"{'improved the attempt best' if out.success else 'no gain'}: score {score:.6g} "
                   f"({score - pscore:+.4g} vs its parent {pscore:.6g})") if valid else f"invalid, scored 0: {err}"
            a.data["history"].append(f"[{idea}] {msg}" if idea else msg)
        self._note_feedback(prog)
        result = SerializableResult(
            child_program_dict=prog.to_dict(), parent_id=prog.parent_id, other_context_ids=[],
            iteration_time=c["llm_time"] + c.get("eval_time", 0.0), llm_generation_time=c["llm_time"],
            eval_time=c.get("eval_time", 0.0), prompt=c.get("prompt"), llm_response=c.get("response"),
            iteration=self.iteration)
        self._process_iteration_result(result, self.iteration, self._ckpt)
        now = datetime.now().isoformat()
        cur = self._cur
        how = cur.get("how")
        node_role = (("regraft_seed" if how == "regraft" else "graft_seed") if role == "seed"
                     else "probe" if how == "probe" else "trunk" if was_trunk else "branch")
        self._append(self._tree_path, {
            "eval": self.E, "iteration": self.iteration, "timestamp": now, "framing": f.id, "attempt": a.id,
            "role": role, "node_role": node_role, "how": how, "parent": prog.parent_id, "program": prog.id,
            "score": score, "valid": valid, "success": out.success, "global_record": out.record, "G": t.G,
            "prev_best": prev_best, "prev_G": prev_G, "anchor_G": prev_anchor, "bar": out.bar, "age": out.age,
            "h": out.h, "was_trunk": was_trunk, "parent_is_incumbent": bool(parent and parent.id == inc_before),
            "parent_is_attempt_best": bool(parent and parent.id == att_best_id),
            "stage1": {"outcome": cur.get("stage1") or ("probe" if how == "probe" else how), "rho_T": cur.get("d_trunk"),
                       "rho_B": cur.get("d_branch"), "draw": draw if role == "edit" else None},
            "rerooted": out.rerooted, "frozen": out.frozen, "regrafts_queued": out.regrafts, "since_reroot": since_reroot,
            "reroots": t.stats["reroots"], "labels": t.labels(), "eps": t.eps, "eps_samples": len(self.eps_samples),
            "eval_wall": c.get("eval_time"), "inst_time": (tm or {}).get("inst"), "headroom": (tm or {}).get("headroom"),
            "parent_headroom": c.get("headroom_shown"), "name_error": name_error, "symbol_rejections": sym_rej,
            "diag_shown": bool(c.get("parent_diag")) if role == "edit" else None, "diag_path": c.get("diag_path"),
            "micro_idea": c.get("micro_idea"), "hypothesis": c.get("hyp"), "prechecks": c.get("rejections"),
            "error": err or first_error or None, "nodes": t.node_stats(a)})
        self._append(self._stats_path, {
            "iteration": self.iteration, "timestamp": now, "evaluations": self.E,
            "global": {"global_best_score": t.G},
            "iteration_result": {"success": True, "child_program": {
                "id": prog.id, "metrics": prog.metrics, "parent_id": prog.parent_id}},
            "banyan": {"framing": f.id, "attempt": a.id, "role": role, "node_role": node_role, "valid": valid,
                       "success": out.success, "global_record": out.record, "rerooted": out.rerooted,
                       "framings": len(t.framings), "attempts": len(t.attempts), "N": t.N, "eps": t.eps}})
        logger.warning("banyan: E=%d F%d/A%d %s%s score=%.6g (parent %.6g) best_A=%.6g G=%.6g%s%s%s", self.E, f.id, a.id,
                       node_role, f" age={out.age}" if role == "edit" else "", score, pscore,
                       a.best if a.best is not None else float("nan"), t.G, " SUCCESS" if out.success else "",
                       " RECORD" if out.record else "", " REROOT" if out.rerooted else "")
        if out.rerooted:
            self.counters["reroots"] += 1
            self._event("reroot", attempt=a.id, framing=f.id, G=t.G, frozen=out.frozen, regrafts=out.regrafts,
                        n_reroot=t.stats["reroots"], card=_title(f.card, 120))
        if out.record:
            self._maybe_measure_noise(prog)
        self._update_eps()
        return out

    # ------------------------------------------------------------ main loop
    async def _start(self):
        """A fresh run: framing 0 from the initial program, the diagnoser, the overhead, N_START more framings."""
        seed = self.database.get_best_program()
        if seed is None:
            raise RuntimeError("BANYAN needs the evaluated initial program in the database")
        self.initial_solution = seed.solution
        self.required = ct.required_names(self._evaluator_src, self.initial_solution)
        self.contract = ct.contract(self._evaluator_src, self.initial_solution)
        self.banned = ct.banned_imports(self.system)
        self.has_markers = _split_block(self.initial_solution) is not None
        self.b0 = self.g_score = _score(seed.metrics)
        self.g_id = self.best_id = seed.id
        self.best_score = self.b0
        self.E = 1                                   # the runner's evaluation of the initial program
        self.cache.add(_norm_hash(seed.solution))
        self.tree = core.Tree(self.b0, rng=self.rng, **self._tree_cfg())
        self._update_eps()
        if self.L:
            waves = max(1, math.ceil(float((seed.metrics or {}).get("n_instances") or 1) / self.eval_parallel))
            self.D = min(CLOCK_FRAC * self.L, BUDGET_FRAC * self.T / max(1, ct.EVAL_CAP) / waves)
            logger.warning("banyan: deadline D=%.3gs per call = min(%g*L=%g, %g*T/cap/waves=%g*%.0f/%d/%d)", self.D,
                           CLOCK_FRAC, CLOCK_FRAC * self.L, BUDGET_FRAC, BUDGET_FRAC, self.T, ct.EVAL_CAP, waves)
        ev_in_system = bool(self._evaluator_src.strip()) and self._evaluator_src.strip()[:200] in self.system
        logger.warning(
            "banyan: b0=%.6g required=%s contract=%s banned_imports=%s evolve_markers=%s T=%.0fs (LIMIT=%s) cap=%d "
            "evaluator_source_in_system_prompt=%s evalsrc=%s(%d chars) per-call limit L=%s clock_frac=%g diag=%s "
            "lessons=%s p2=%s alloc=%s flat=%s constants=%s prompts=%s", self.b0, self.required, self.contract or "{}", sorted(self.banned),
            self.has_markers, self.T, os.environ.get("LIMIT"), ct.EVAL_CAP, ev_in_system, EVALSRC, len(self._eval_view),
            self.L, CLOCK_FRAC, DIAG, LESSONS, P2, ALLOC, FLAT, json.dumps(self.tree.c, default=str), _prompt_dir())
        self._event("start", b0=self.b0, constants=core.constants(), cap=ct.EVAL_CAP, T=self.T, L=self.L,
                    clock_frac=CLOCK_FRAC, budget_frac=BUDGET_FRAC, deadline=self.D, diag=DIAG, lessons=LESSONS, evalsrc=EVALSRC,
                    p2=P2, alloc=ALLOC, flat=FLAT, tree_cfg=self.tree.c)

        # framing 0 is the initial program's own framing (BANYAN_FLAT=1: the one framing, no card); the trunk starts as
        # the initial program
        card0 = {"problem_class": "flat tree (no framings)"} if FLAT else await self._describe_initial(seed)
        f0 = self.tree.add_framing(card0)
        a0 = self.tree.add_attempt(f0, "Refine the incumbent as it is, keeping its approach.", initial=True)
        a0.data = {"programs": [{"id": seed.id, "score": self.b0, "valid": True}], "history": []}
        seed.metadata = dict(seed.metadata or {}, framing_id=0, attempt_id=0, role="initial")
        logger.warning("banyan: framing 0 (trunk) = %s", card0.get("problem_class"))
        self._event("framing", framing=0, card=card0, source="initial")
        try:
            await self._setup_diag(seed)
        except Exception as e:                       # without a diagnoser the prompts fall back to the plain feedback
            logger.exception("banyan: diagnoser setup failed: %s", e)
            self.diag = None
        await self._measure_overhead(seed)

        for _ in range(core.N_START):                # two more framings, deduplicated, probed first (7)
            if self._stop():
                break
            self._stage = ("start", None, None)
            await (self._grow_child() if FLAT else self._grow_framing())
            self.tree.close_framings()

    async def _resume(self):
        """Continue a run that died. The runner has reloaded its database from the last checkpoint; the tree, the
        evaluation count, the clock and the per-program state come from the run's own logs. Evaluations made after
        that checkpoint keep their effect on the tree and the budget, but their programs are gone; evaluations made
        after the dump (EVAL_DONE) count against the budget only. The diagnoser is
        rebuilt (free LLM calls, re-runs of evaluated programs), as at the start of a run."""
        out = self.output_dir or "."
        load = lambda name: [json.loads(l) for l in open(os.path.join(out, name)) if l.strip()]
        dump = json.load(open(self._dump_path))
        rows, events = load("tree_stats.jsonl"), load("tree_events.jsonl")
        start = next(e for e in events if e["event"] == "start")
        progs = self.database.programs
        seed = next(p for p in progs.values() if not p.parent_id)
        self.initial_solution = seed.solution
        self.required = ct.required_names(self._evaluator_src, self.initial_solution)
        self.contract = ct.contract(self._evaluator_src, self.initial_solution)
        self.banned = ct.banned_imports(self.system)
        self.has_markers = _split_block(self.initial_solution) is not None
        self.b0 = dump["b0"]
        self.E = max(dump["E"], int(os.environ.get("EVAL_DONE") or 0))     # evaluations spent after the dump count too
        self.iteration = max([self.iteration] + [r["iteration"] for r in rows])
        if abs(self.T - start["T"]) > 1:
            raise RuntimeError(f"resume needs the original wall clock T={start['T']} (LIMIT), got {self.T}")
        self.D, self.overhead = start["deadline"], dump.get("overhead", 0.0)
        elapsed = (datetime.fromisoformat(rows[-1]["timestamp"]) - datetime.fromisoformat(start["timestamp"])).total_seconds()
        self.t_start = time.time() - elapsed
        self.counters.update(dump.get("counters") or {})
        self.eps_samples = list(dump.get("eps_samples") or [])
        self.rng.seed(f"resume:{self.E}:{self.b0}")       # the dead run's rng state is not saved: a fresh, logged stream
        self.tree = t = core.Tree.restore(dump, rows, events, rng=self.rng)
        t.c.update(self._tree_cfg())                     # restore() starts from the defaults: re-apply the ablations

        # node programs and edit histories, in evaluation order; lost programs (after the checkpoint) are skipped
        score = {p.id: _score(p.metrics) for p in progs.values()}
        t.attempts[0].data["programs"].append({"id": seed.id, "score": self.b0, "valid": True})
        fallbacks = {}
        for e in events:
            if e["event"] == "seed_fallback":
                fallbacks.setdefault(e["N"], []).append(e)
        lost = 0
        for r in rows:
            a = t.attempts[r["attempt"]]
            if r["valid"]:
                if r["program"] in progs:
                    a.data["programs"].append({"id": r["program"], "score": r["score"], "valid": True})
                else:
                    lost += 1
            if r["role"] == "edit":
                idea = (r.get("micro_idea") or "")[:100]
                ps = score.get(r["parent"], 0.0)
                msg = (f"{'improved the attempt best' if r['success'] else 'no gain'}: score {r['score']:.6g} "
                       f"({r['score'] - ps:+.4g} vs its parent {ps:.6g})") if r["valid"] else \
                    f"invalid, scored 0: {r.get('error') or ''}"
                a.data["history"].append(f"[{idea}] {msg}" if idea else msg)
            if r.get("headroom") is not None:
                self.timing[r["program"]] = {"wall": r.get("eval_wall"), "inst": r.get("inst_time"),
                                             "headroom": r["headroom"]}
            for e in fallbacks.pop(r["nodes"]["root"]["N"], []):       # a failed seed leaves a copy of the incumbent
                if e["copy_of"] in progs:
                    t.attempts[e["attempt"]].data["programs"].append(
                        {"id": e["copy_of"], "score": score[e["copy_of"]], "valid": True})
        for es in fallbacks.values():                                   # fallbacks logged before any evaluation
            for e in es:
                if e["copy_of"] in progs:
                    t.attempts[e["attempt"]].data["programs"].append(
                        {"id": e["copy_of"], "score": score[e["copy_of"]], "valid": True})

        for p in sorted(progs.values(), key=lambda p: p.iteration_found or 0):
            self.cache.add(_norm_hash(p.solution))
            self._note_feedback(p)
        valid = [p for p in progs.values() if score[p.id] > 0]
        best = max(valid, key=lambda p: score[p.id])
        self.best_id, self.best_score = best.id, score[best.id]
        inc = self._incumbent()
        self.g_id, self.g_score = (inc.id, _score(inc.metrics)) if inc else (best.id, score[best.id])
        if abs(self.g_score - t.G) > 1e-12:
            logger.warning("banyan: resume: the incumbent G=%.6g was lost after the checkpoint; prompts use %.6g", t.G,
                           self.g_score)
        logger.warning("banyan: RESUMED E=%d N=%d G=%.6g trunk=A%d framings=%d attempts=%d eps=%.3g D=%.3gs elapsed=%.0fs "
                       "of T=%.0fs; %d programs in the database, %d evaluated programs lost after the checkpoint",
                       self.E, t.N, t.G, t.trunk.id, len(t.framings), len(t.attempts), t.eps, self.D, elapsed, self.T,
                       len(progs), lost)
        self._event("resume", programs=len(progs), lost=lost, elapsed=elapsed, G=t.G, g_score=self.g_score)
        try:
            await self._setup_diag(seed)
        except Exception as e:
            logger.exception("banyan: diagnoser setup failed: %s", e)
            self.diag = None

    async def run_discovery(self, start_iteration: int, max_iterations: int, checkpoint_callback=None):
        self._setup_logs()
        self.iteration = start_iteration - 1
        self._ckpt = checkpoint_callback
        if RESUME:
            await self._resume()
        else:
            await self._start()

        empty, no_new = 0, False
        while not self._stop():
            e0 = self.E
            try:
                t = self.tree.next_target(no_new=no_new)
                no_new = False
                self._stage = (t.stage1, t.d_trunk, t.d_branch)
                if t.kind == "regraft":
                    if not await self._regraft(t.framing):
                        no_new = True
                elif t.kind == "new_framing":
                    if not await self._grow_framing():
                        no_new = True
                elif t.kind == "new_attempt":
                    if not await (self._grow_child() if FLAT else self._grow_attempt(t.framing)):
                        no_new = True
                else:
                    self._cur = {"how": t.how, "d": t.d, "d_trunk": t.d_trunk, "d_branch": t.d_branch,
                                 "stage1": t.stage1}
                    await self._run_block(t.attempt)
            except Exception as e:                   # one bad step must not end the run
                logger.exception("banyan step failed: %s", e)
                no_new = True
            self.tree.close_framings()
            self._write_tree()
            empty = 0 if self.E > e0 else empty + 1
            if empty >= 3:
                await asyncio.sleep(min(60.0, 2.0 ** min(empty - 3, 6)))
            if empty >= 400:
                logger.error("banyan: %d consecutive steps without an evaluation, giving up", empty)
                break
        for task in list(self._bg) + list(self._noise_tasks):     # re-runs and verifications do not outlive the run
            task.cancel()
        if self.diag:
            self.counters.update(diag_version=self.diag.version, diag_rebuilds=self.diag.rebuilds,
                                 diag_seconds=round(self.diag.seconds), diag_trusted=self.diag.trusted,
                                 diag_paths=dict(self.diag.paths))
        self._write_tree()
        t = self.tree
        logger.warning("banyan: finished E=%d N=%d framings=%d attempts=%d best=%.6g G=%.6g reroots=%d eps=%.3g "
                       "elapsed=%.0fs counters=%s stats=%s", self.E, t.N, len(t.framings), len(t.attempts),
                       self.best_score, t.G, t.stats["reroots"], t.eps, self._elapsed(), json.dumps(self.counters),
                       json.dumps(t.stats))
        self._event("finish", best=self.best_score, G=t.G, counters=self.counters, stats=t.stats)
        return self.database.get_best_program()

    async def _measure_overhead(self, seed: Program):
        """Wall of the evaluator on the (instant) initial program, measured warm (the second of two evaluations in one
        process): fork, import and data-load cost per evaluation, which is subtracted from an evaluation's wall time
        to estimate the program's own time per instance. (A cold first run, 0.7 s in the first pilot, over-subtracted:
        it made every headroom 1.00.)"""
        try:
            wd = os.path.join(self.output_dir or ".", "diag")
            os.makedirs(wd, exist_ok=True)
            path = os.path.join(wd, "initial_program.py")
            with open(path, "w") as fh:
                fh.write(seed.solution)
            r = await asyncio.to_thread(replay.run_eval, self.evaluation_file, path, 0.05 * self.T, self.eval_parallel, 2)
            if r.get("ok"):
                self.overhead = float(r.get("eval_seconds") or 0.0)
            logger.warning("banyan: evaluator overhead on the initial program %.2fs (ok=%s)", self.overhead, r.get("ok"))
            self._event("overhead", seconds=self.overhead, ok=r.get("ok"), error=r.get("error"))
        except Exception as e:
            logger.warning("banyan: overhead measurement failed: %s", e)


def install():
    """Register search type `banyan`. Called from run_capped.py when BANYAN=1."""
    for name in TEMPLATES:                            # fail loudly on a missing prompt, not at the first call
        _tpl(name)
    _DB_CONFIG_BY_TYPE["banyan"] = AdaEvolveDatabaseConfig       # archive/intensity defaults, unused
    register_database("banyan", CredTreeDatabase)
    register_controller("banyan", BanyanController)
    from skydiscover.optimize.search.route import _CONTROLLER_REGISTRY
    assert _CONTROLLER_REGISTRY.get("banyan") is BanyanController, "banyan controller not registered"
    logger.warning("banyan installed: prompts=%s cap=%d diag=%s lessons=%s clock_frac=%g constants=%s", _prompt_dir(),
                   ct.EVAL_CAP, DIAG, LESSONS, CLOCK_FRAC, json.dumps(core.constants(), default=str))
