"""CredTree: framing -> attempt tree with Thompson budget flow on age-corrected credibility.

Specification: design/round6/CREDIBILITY_TREE.md (section numbers below refer to it). Opt-in from
run_capped.py with CREDTREE=1 and `search.type: credtree`. A sibling of sprout.py: same SkyDiscover
hooks (DiscoveryController subclass, own database that keeps every program, evaluator called directly,
SerializableResult through _process_iteration_result), a different loop.

The scheduler is credtree_core.Tree (pure Python, no skydiscover). This module generates programs and
spends evaluations where the core says. One step of the loop:
  1. core.next_target(): a block on an attempt (probe / exploit / Thompson draw), or a NEW framing, or
     a NEW attempt under a framing                                                         (5)
  2. new framing: free generator call + free dedup judge (<= 5 tries, then the root is exhausted)  (7)
     new attempt: free plan call; its SEED is a full rewrite of the CURRENT GLOBAL BEST under
     plan + framing, 1 evaluation, repaired once if it crashes (seeds only, auto-off)         (6)
  3. block: 2 leaf edits (diffs on the attempt's best, p 0.8, else a near-best program), each with one
     micro-idea from a free call that sees the parent and the attempt's last 3 outcomes; evaluated
     together; +1 evaluation after each success, up to 6                                      (5, 6)
  4. every candidate passes a static pre-check (regenerate <= 2) and a free self-review (<= 1 revise)
     before it costs an evaluation                                                            (6)

Only evaluations count against the 500 cap (EVAL_CAP); every LLM call that is not evaluated is free.
Hyper-parameters are module constants of credtree_core: identical on all five tasks (handout 2.3).
"""
import ast
import asyncio
import hashlib
import json
import logging
import math
import os
import re
import time
import uuid
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Tuple

import credtree_core as core
import diagnoser as diagmod
from skydiscover.optimize.config import AdaEvolveDatabaseConfig, _DB_CONFIG_BY_TYPE
from skydiscover.optimize.search.base_database import Program, ProgramDatabase
from skydiscover.optimize.search.default_discovery_controller import DiscoveryController
from skydiscover.optimize.search.registry import register_controller, register_database
from skydiscover.optimize.search.utils.discovery_utils import SerializableResult, load_evaluator_code
from skydiscover.optimize.utils.code_utils import extract_diffs, format_diff_summary

logger = logging.getLogger("credtree")

# ---------------------------------------------------------------- controller constants
EVAL_CAP = int(os.environ.get("EVAL_CAP", "500"))
REGEN = 2                  # regenerations after a failed static pre-check (6.1)
REVIEW_ROUNDS = 1          # free self-review revise rounds (6.2)
HISTORY = 3                # "last 3 outcomes" in the micro-idea and edit prompts
MAX_LEN = 60000            # program length cap (the task configs' max_solution_length)
PROMPT_CHAR_BUDGET = int(os.environ.get("PROMPT_CHAR_BUDGET", "45000"))
VIEW_CHARS = 14000         # program view in prompts that do not need the whole file
FEEDBACK_CHARS, NOTES_CHARS, CARDS_CHARS = 2000, 1200, 6000
WALL_FRAC = 0.985          # stop starting work at this share of the wall-clock limit
# v2 (defaults ON; set to 0 for the v1 ablations):
#  CREDTREE_EVALSRC  the framing generator (and framing 0's description) sees the evaluator source, as
#                    AdaEvolve's paradigm generator does, and every card carries an evaluator audit: what
#                    the evaluator checks and what it does not. Plans, seeds and edits see the card.
#  CREDTREE_GRAFT    a seed implements the plan as SEARCH/REPLACE edits on the global best; a full rewrite
#                    only when the plan cannot be expressed as edits (the v1 rewrites scored 0-1 vs 21.75).
EVALSRC = os.environ.get("CREDTREE_EVALSRC", "1") == "1"
GRAFT = os.environ.get("CREDTREE_GRAFT", "1") == "1"
EVAL_SRC_CHARS = 12000
# v3 (default OFF, so v2 runs are unchanged; spec section 14):
#  CREDTREE_DIAG     the generator writes a diagnoser from the evaluator source (scripts/diagnoser.py); every edit
#                    starts from a verified logic-level diagnosis of its parent, states CAUSE / CHANGE / PREDICT,
#                    and after evaluation a free verify call turns the outcome into a lesson for that framing.
DIAG = os.environ.get("CREDTREE_DIAG", "0") == "1"
DIAG_FRAC = 0.2            # share of the wall-clock limit diagnosis runs may use; after it, cached diagnoses only
DIAG_TIMEOUT_FRAC = 0.05   # one diagnosis run (a program on every training instance, possibly twice)
DIAG_REBUILDS = 3
DIAG_BACKGROUND = 6        # verifications in flight; more are skipped
LESSONS_SHOWN = 6
TEMPLATES = ("framing_describe", "framing_new", "framing_judge", "plan", "seed", "seed_graft", "micro_idea",
             "edit", "review", "repair") + (("diagnoser", "micro_idea_diag", "edit_diag", "verify") if DIAG else ())

_BLOCK_START, _BLOCK_END = "# EVOLVE-BLOCK-START", "# EVOLVE-BLOCK-END"
CARD_FIELDS = ("problem_class", "decision_variables", "objective_restatement", "assumptions_relaxed",
               "why_it_could_beat_current_best") + (("evaluator_audit",) if EVALSRC else ())
# What "valid" means in every template's {validity} slot. With the evaluator visible, the evaluator's own checks
# decide: a rule the task statement gives but the evaluator never enforces is not a defect (course policy: unpatched
# evaluator loopholes are allowed and are reported as such). Without it, the system message's written checks decide.
VALIDITY = ("the program must pass every check the evaluator actually performs, on every input it may receive; "
            "the evaluator source decides what is valid (see the evaluator audit)") if EVALSRC else \
    "every check and constraint in your system message must hold on every input the program may receive"
REVIEW_BASIS = ("every check the evaluator source below actually performs, and every environment limit in your system "
                "message (installed packages, time limits); a rule the evaluator does not enforce is not a defect") \
    if EVALSRC else "every check, constraint and environment limit in your system message"
# extra JSON line for the card templates' {audit_field} slot
AUDIT_FIELD = (',\n  "evaluator_audit": "from the evaluator source: what it actually checks, what it does not '
               'check, and how this formulation uses that"') if EVALSRC else ""

_HERE = os.path.dirname(os.path.abspath(__file__))


def _prompt_dir() -> str:
    for d in (os.environ.get("CREDTREE_PROMPTS"), os.path.join(os.path.dirname(_HERE), "prompts", "credtree"),
              os.path.join(_HERE, "prompts", "credtree")):
        if d and os.path.isdir(d):
            return d
    raise FileNotFoundError("prompts/credtree not found (set CREDTREE_PROMPTS)")


_TPL: Dict[str, str] = {}


def _tpl(name: str) -> str:
    if name not in _TPL:
        with open(os.path.join(_prompt_dir(), f"{name}.txt")) as fh:
            _TPL[name] = fh.read()
    return _TPL[name]


def _fill(name: str, **kw) -> str:
    """Substitute {key} for the given keys only; any other brace (JSON, code) is left alone."""
    kw.setdefault("validity", VALIDITY)
    kw.setdefault("review_basis", REVIEW_BASIS)
    kw.setdefault("evaluator", "")
    kw.setdefault("audit_field", AUDIT_FIELD)
    return re.sub(r"\{([a-z_0-9]+)\}", lambda m: str(kw[m.group(1)]) if m.group(1) in kw else m.group(0), _tpl(name))


# ---------------------------------------------------------------- helpers shared with sprout.py (copied)
def _score(metrics: Optional[Dict[str, Any]]) -> float:
    try:
        v = float((metrics or {}).get("combined_score", 0.0))
    except (TypeError, ValueError):
        return 0.0
    return v if v == v else 0.0


def _wall_seconds() -> float:
    """Task wall-clock limit from LIMIT (run_task.sh passes e.g. 5h, 10h, 16m)."""
    raw = os.environ.get("CREDTREE_WALL") or os.environ.get("LIMIT") or "5h"
    m = re.fullmatch(r"\s*([\d.]+)\s*([smhd]?)\s*", raw)
    if not m:
        return 5 * 3600.0
    return float(m.group(1)) * {"": 1, "s": 1, "m": 60, "h": 3600, "d": 86400}[m.group(2)]


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


def _fenced(text: str) -> Optional[str]:
    """The longest fenced code block; no raw-text fallback."""
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


def _provides(tree: ast.Module, name: str) -> bool:
    """True if the program still supplies `name`: as a def, an assignment, a keyword argument, or a
    literal string (setattr/type() construction)."""
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


def required_names(evaluator_src: str, initial_solution: str) -> List[str]:
    """Entry points the evaluator calls, plus in-block names the frozen code uses."""
    req = set(re.findall(r"\bprog\.(\w+)\s*\(", evaluator_src))
    parts = _split_block(initial_solution)
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


def contract(evaluator_src: str, initial_solution: str) -> Dict[str, List[str]]:
    """{class: [methods]} the evaluator calls on objects the program supplies (cloudcast: set_num_partitions)."""
    if os.environ.get("CREDTREE_CONTRACT") == "0":
        return {}
    try:
        ev = ast.parse(evaluator_src)
        init = ast.parse(initial_solution)
    except SyntaxError:
        return {}
    attrs = {n.attr for n in ast.walk(ev) if isinstance(n, ast.Attribute)}
    out = {}
    for n in init.body:
        if isinstance(n, ast.ClassDef):
            m = sorted({x.name for x in n.body if isinstance(x, (ast.FunctionDef, ast.AsyncFunctionDef))} & attrs)
            if m:
                out[n.name] = m
    return out


def banned_imports(system: str) -> set:
    """Modules the task prompt says are not installed ("Not installed: a, b (c), d -- ..."). Derived from
    the prompt text, so the rule holds on any task and is empty when the prompt lists none."""
    m = re.search(r"Not installed:\s*(.*?)\s*(?:--|\.\s|\n\n)", system or "", re.S | re.I)
    out = set()
    if m:
        for part in m.group(1).split(","):
            w = re.match(r"\s*([A-Za-z_][\w.]*)", part)
            if w:
                out.add(w.group(1).split(".")[0])
    return out


def _unguarded_imports(tree: ast.Module) -> set:
    guarded = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.Try) and n.handlers:
            for s in n.body:
                for x in ast.walk(s):
                    guarded.add(id(x))
    mods = set()
    for n in ast.walk(tree):
        if id(n) in guarded:
            continue
        if isinstance(n, ast.Import):
            mods |= {a.name.split(".")[0] for a in n.names}
        elif isinstance(n, ast.ImportFrom) and n.module and not n.level:
            mods.add(n.module.split(".")[0])
    return mods


# ---------------------------------------------------------------- cards and parsing
def _parse_json(text: Optional[str]) -> Optional[dict]:
    if not text:
        return None
    t = re.sub(r"```(?:json)?", "", text)
    i, j = t.find("{"), t.rfind("}")
    if i < 0 or j < i:
        return None
    try:
        d = json.loads(t[i:j + 1])
    except ValueError:
        return None
    return d if isinstance(d, dict) else None


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


def _fmt_card(card: Dict[str, str]) -> str:
    return (f"Problem class: {card.get('problem_class', '')}\n"
            f"Decision variables: {card.get('decision_variables', '')}\n"
            f"Objective restatement: {card.get('objective_restatement', '')}\n"
            f"Assumptions relaxed: {card.get('assumptions_relaxed', '')}\n"
            f"Why it could beat the current best: {card.get('why_it_could_beat_current_best', '')}"
            + (f"\nEvaluator audit: {card['evaluator_audit']}" if card.get("evaluator_audit") else ""))


def _feedback(prog: Optional[Program]) -> str:
    return (((prog.artifacts or {}).get("feedback") or "") if prog else "")[:FEEDBACK_CHARS] or "(none)"


def _error_text(metrics: Dict[str, Any], artifacts: Dict[str, Any]) -> str:
    lines = []
    for v in (metrics.get("error"), (artifacts or {}).get("error")):
        if isinstance(v, str) and v.strip():
            lines.append(v.strip())
    fb = (artifacts or {}).get("feedback") or ""
    lines += [l.strip() for l in fb.splitlines() if ": error" in l or ": timeout" in l or "Error" in l]
    return " | ".join(lines[:3])[:500] or "the evaluator scored it zero without an error message"


# ---------------------------------------------------------------- database
class CredTreeDatabase(ProgramDatabase):
    """Keeps every program (the tree does the sampling), so validation selection sees all of them."""

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
class CredTreeController(DiscoveryController):
    def __init__(self, controller_input):
        super().__init__(controller_input)
        self.rng = self.database.rng
        self.t_start = time.time()
        self.T = _wall_seconds()
        self.E = 0                                   # evaluations spent, the initial program included
        self.iteration = 0
        self.tree: Optional[core.Tree] = None
        self.b0 = 0.0
        self.g_id: Optional[str] = None              # current global best program
        self.g_score = 0.0
        self.cache: set = set()                      # AST hashes of everything evaluated
        self.notes: Dict[str, List[Any]] = {}        # evaluator rejections seen so far
        self.initial_solution = ""
        self.required: List[str] = []
        self.contract: Dict[str, List[str]] = {}
        self.banned: set = set()
        self.has_markers = False
        self.system = self.config.context_builder.system_message or ""
        self._evaluator_src = ""
        try:
            with open(self.evaluation_file) as fh:
                self._evaluator_src = fh.read()
        except OSError:
            pass
        self._eval_view = ""
        if EVALSRC:                                   # the same loader AdaEvolve's paradigm generator uses
            try:
                self._eval_view = load_evaluator_code(self.evaluation_file)[:EVAL_SRC_CHARS]
            except Exception as e:
                logger.warning("credtree: evaluator source unavailable: %s", e)
                self._eval_view = self._evaluator_src[:EVAL_SRC_CHARS]
        self._stats_path = self._tree_path = self._events_path = self._dump_path = None
        self._cur: Dict[str, Any] = {}               # scheduling info of the block being run
        self.counters = {"prechecks": 0, "regens": 0, "reviews": 0, "revised": 0, "dedup_dups": 0,
                         "dedup_calls": 0, "seed_tries": 0, "seed_valid": 0, "seed_crashed": 0,
                         "repair_tried": 0, "repair_valid": 0, "empty_slots": 0, "llm_failures": 0}
        self.diag: Optional[diagmod.Diagnoser] = None
        self.lessons: Dict[int, List[str]] = {}      # framing id -> verified lessons (v3)
        self._diag_locks: Dict[str, asyncio.Lock] = {}
        self._diag_sem: Optional[asyncio.Semaphore] = None
        self._bg: set = set()
        self._rebuilding = False
        if DIAG:
            self.counters.update({"diag_shown": 0, "diag_missing": 0, "hypotheses": 0, "lessons": 0,
                                  "verify_skipped": 0, "confirmed": 0, "refuted": 0})

    # ------------------------------------------------------------ logs
    def _setup_logs(self):
        out = self.output_dir or "."
        os.makedirs(out, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._stats_path = os.path.join(out, f"credtree_iteration_stats_{ts}.jsonl")
        self._tree_path = os.path.join(out, "tree_stats.jsonl")           # one line per evaluation (8)
        self._events_path = os.path.join(out, "tree_events.jsonl")        # framings, attempts, dedup, repair switches
        self._dump_path = os.path.join(out, "credtree_tree.json")

    @staticmethod
    def _append(path: Optional[str], rec: Dict[str, Any]):
        if not path:
            return
        try:
            with open(path, "a") as fh:
                fh.write(json.dumps(rec, default=str) + "\n")
        except OSError as e:
            logger.warning("credtree: log write failed: %s", e)

    def _event(self, kind: str, **kw):
        self._append(self._events_path, {"event": kind, "E": self.E, "N": self.tree.N if self.tree else 0,
                                       "timestamp": datetime.now().isoformat(), **kw})

    def _write_tree(self):
        if not self._dump_path or not self.tree:
            return
        t = self.tree
        data = {"E": self.E, "N": t.N, "G": t.G, "b0": self.b0, "exhausted": t.exhausted, "counters": self.counters,
                "repair": {"valid": t.rep[0], "tried": t.rep[1], "enabled": t.repair_enabled()},
                "haz": t.haz, "clo": t.clo,
                "framings": [{"id": f.id, "card": f.card, "n": t.n_framing(f),
                              "attempts": [{"id": a.id, "plan": a.plan, "best": a.best, "n": a.n, "valid": a.valid,
                                            "age": a.age, "s": a.s, "E": a.E, "s0": a.s0, "probe_done": a.probe_done,
                                            "seed_valid": a.seed_valid, "programs": len(a.data.get("programs", []))}
                                           for a in f.attempts]} for f in t.framings]}
        tmp = self._dump_path + ".tmp"
        try:
            with open(tmp, "w") as fh:
                json.dump(data, fh, indent=1, default=str)
            os.replace(tmp, self._dump_path)
        except OSError as e:
            logger.warning("credtree: tree dump failed: %s", e)

    # ------------------------------------------------------------ time
    def _elapsed(self) -> float:
        return time.time() - self.t_start

    def _out_of_time(self) -> bool:
        return self._elapsed() > WALL_FRAC * self.T

    def _stop(self) -> bool:
        return self.E >= EVAL_CAP or self.shutdown_event.is_set() or self._out_of_time()

    # ------------------------------------------------------------ main loop
    async def run_discovery(self, start_iteration: int, max_iterations: int, checkpoint_callback=None):
        self._setup_logs()
        self.iteration = start_iteration - 1
        self._ckpt = checkpoint_callback
        seed = self.database.get_best_program()
        if seed is None:
            raise RuntimeError("CredTree needs the evaluated initial program in the database")
        self.initial_solution = seed.solution
        self.required = required_names(self._evaluator_src, self.initial_solution)
        self.contract = contract(self._evaluator_src, self.initial_solution)
        self.banned = banned_imports(self.system)
        self.has_markers = _split_block(self.initial_solution) is not None
        self.b0 = self.g_score = _score(seed.metrics)
        self.g_id = seed.id
        self.E = 1                                   # the runner's evaluation of the initial program
        self.cache.add(_norm_hash(seed.solution))
        self.tree = core.Tree(self.b0, rng=self.rng)
        ev_in_system = bool(self._evaluator_src.strip()) and self._evaluator_src.strip()[:200] in self.system
        logger.warning(
            "credtree: b0=%.6g required=%s contract=%s banned_imports=%s evolve_markers=%s T=%.0fs (LIMIT=%s) "
            "cap=%d regen=%d review_rounds=%d evaluator_source_in_system_prompt=%s evalsrc=%s(%d chars) graft=%s "
            "diag=%s constants=%s prompts=%s",
            self.b0, self.required, self.contract or "{}", sorted(self.banned), self.has_markers, self.T,
            os.environ.get("LIMIT"), EVAL_CAP, REGEN, REVIEW_ROUNDS, ev_in_system, EVALSRC, len(self._eval_view),
            GRAFT, DIAG,
            json.dumps(core.constants(), default=str), _prompt_dir())
        self._event("start", b0=self.b0, constants=core.constants(), cap=EVAL_CAP, T=self.T)

        # framing 0 is the initial program's own framing; its attempt 0 is the initial program (7)
        card0 = await self._describe_initial(seed)
        f0 = self.tree.add_framing(card0)
        a0 = self.tree.add_attempt(f0, "Refine the initial program as it is, keeping its approach.", initial=True)
        a0.data = {"programs": [{"id": seed.id, "score": self.b0, "valid": True}], "history": []}
        seed.metadata = dict(seed.metadata or {}, framing_id=0, attempt_id=0, role="initial")
        logger.warning("credtree: framing 0 = %s", card0.get("problem_class"))
        self._event("framing", framing=0, card=card0, source="initial")
        if DIAG:
            try:
                await self._setup_diag(seed)
            except Exception as e:                   # without a diagnoser the v3 prompts fall back to feedback
                logger.exception("credtree: diagnoser setup failed: %s", e)
                self.diag = None

        for _ in range(core.N_START):                # two more framings, deduplicated, probed first (7)
            if self._stop():
                break
            await self._grow_framing()

        empty, no_new = 0, False
        while not self._stop():
            e0 = self.E
            try:
                t = self.tree.next_target(no_new=no_new)
                no_new = False
                if t.kind == "new_framing":
                    if not await self._grow_framing():
                        no_new = True
                elif t.kind == "new_attempt":
                    if not await self._grow_attempt(t.framing):
                        no_new = True
                else:
                    self._cur = {"how": t.how, "d": t.d, "d_framing": t.d_framing}
                    await self._run_block(t.attempt)
            except Exception as e:                   # one bad step must not end the run
                logger.exception("credtree step failed: %s", e)
                no_new = True
            self._write_tree()
            # A step that evaluates nothing costs no budget, but an unreachable or rejecting LLM
            # server would otherwise be hammered in a tight loop.
            empty = 0 if self.E > e0 else empty + 1
            if empty >= 3:
                await asyncio.sleep(min(60.0, 2.0 ** min(empty - 3, 6)))
            if empty >= 400:
                logger.error("credtree: %d consecutive steps without an evaluation, giving up", empty)
                break
        for task in list(self._bg):                  # verifications still in flight do not outlive the run
            task.cancel()
        if self.diag:
            self.counters.update(diag_version=self.diag.version, diag_rebuilds=self.diag.rebuilds,
                                 diag_seconds=round(self.diag.seconds), diag_trusted=self.diag.trusted)
        self._write_tree()
        t = self.tree
        logger.warning("credtree: finished E=%d N=%d framings=%d attempts=%d best=%.6g exhausted=%s "
                       "elapsed=%.0fs counters=%s", self.E, t.N, len(t.framings), len(t.attempts), self.g_score,
                       t.exhausted, self._elapsed(), json.dumps(self.counters))
        self._event("finish", best=self.g_score, counters=self.counters)
        return self.database.get_best_program()

    # ------------------------------------------------------------ LLM
    async def _llm(self, user: str) -> Tuple[Optional[str], float]:
        if len(user) > PROMPT_CHAR_BUDGET:
            logger.warning("credtree: prompt %d chars over the %d budget", len(user), PROMPT_CHAR_BUDGET)
        t0 = time.time()
        try:
            r = await self._call_llm(self.system, user)
            return r.text, time.time() - t0
        except Exception as e:
            self.counters["llm_failures"] += 1
            logger.warning("credtree: LLM call failed: %s", e)
            return None, time.time() - t0

    @staticmethod
    def _view(code: str) -> str:
        return code if len(code) <= VIEW_CHARS else code[:VIEW_CHARS] + "\n# ... (program truncated here)\n"

    def _program(self, pid: Optional[str]) -> Optional[Program]:
        return self.database.get(pid) if pid else None

    # ------------------------------------------------------------ framings (7)
    def _cards(self, exclude_ids=()) -> str:
        rows = []
        for f in self.tree.framings:
            rows.append(f"[framing {f.id}] best score reached {self.tree.best_attempt(f).best:.6g}, "
                        f"{self.tree.n_framing(f)} evaluations spent\n" + _fmt_card(f.card))
        text = "\n\n".join(rows)
        return text if len(text) <= CARDS_CHARS else text[-CARDS_CHARS:]

    def _evaluator_section(self, audit: bool = True) -> str:
        if not self._eval_view:
            return ""
        return ("\n## Evaluator source (how every program is scored)\n```python\n" + self._eval_view +
                "\n```\nRead what it actually validates and how it computes the score." +
                (" The evaluator_audit field must say what it checks and what it does not check.\n" if audit else "\n"))

    async def _describe_initial(self, seed: Program) -> Dict[str, str]:
        user = _fill("framing_describe", score=f"{self.b0:.6g}", program=self._view(seed.solution),
                     evaluator=self._evaluator_section(), audit_field=AUDIT_FIELD)
        for _ in range(3):
            text, _ = await self._llm(user)
            card = _parse_card(text)
            if card:
                return card
        return {"problem_class": "the formulation of the initial program", "decision_variables": "as in the initial program",
                "objective_restatement": "as in the task statement", "assumptions_relaxed": "unknown",
                "why_it_could_beat_current_best": "n/a (this is the formulation of the current program)"}

    async def _judge(self, card: Dict[str, str]) -> Tuple[bool, str, Optional[int]]:
        """Free dedup judge: is `card` a duplicate of an existing framing on problem class + decision variables?"""
        short = "\n\n".join(f"[framing {f.id}]\nProblem class: {f.card.get('problem_class', '')}\n"
                            f"Decision variables: {f.card.get('decision_variables', '')}" for f in self.tree.framings)
        cand = (f"Problem class: {card['problem_class']}\nDecision variables: {card['decision_variables']}")
        self.counters["dedup_calls"] += 1
        text, _ = await self._llm(_fill("framing_judge", candidate=cand, cards=short))
        d = _parse_json(text)
        if not d or "duplicate" not in d:
            return False, "", None                    # an unreadable verdict does not block a framing
        dup = d["duplicate"] is True or str(d["duplicate"]).lower() == "true"
        m = d.get("matches")
        return dup, str(d.get("reason", ""))[:300], (m if isinstance(m, int) else None)

    async def _propose_framing(self) -> Optional[Dict[str, str]]:
        """Generator + judge, up to core.DEDUP_RETRIES tries (7). None means the root is exhausted."""
        gbest = self._program(self.g_id)
        rejection = ""
        for k in range(core.DEDUP_RETRIES):
            user = _fill("framing_new", score=f"{self.g_score:.6g}", program=self._view(gbest.solution),
                         cards=self._cards(), rejection=rejection, evaluator=self._evaluator_section(),
                         audit_field=AUDIT_FIELD)
            text, _ = await self._llm(user)
            card = _parse_card(text)
            if card is None:
                rejection = (f"\n## Your previous answer was rejected\nIt was not a JSON object with the {len(CARD_FIELDS)} required "
                             "string fields; answer with exactly that object.\n")
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
                         f"variables from every framing above, and from the rejected one.\n")
        return None

    async def _make_plan(self, card: Dict[str, str], f: Optional[core.Framing]) -> Optional[str]:
        gbest = self._program(self.g_id)
        sib = "(none yet)"
        if f is not None and f.attempts:
            sib = "\n".join(f"- ({a.best:.6g}) {a.plan[:500]}" for a in f.attempts)
        user = _fill("plan", framing=_fmt_card(card), score=f"{self.g_score:.6g}", program=self._view(gbest.solution),
                     siblings=sib)
        for _ in range(3):
            text, _ = await self._llm(user)
            if not text:
                continue
            m = re.search(r"PLAN:\s*(.+)", text, re.S)
            plan = (m.group(1) if m else text).strip().strip("`").strip()
            plan = re.sub(r"\s+", " ", plan)[:1200]
            if len(plan) >= 40:
                return plan
        return None

    async def _grow_framing(self) -> bool:
        card = await self._propose_framing()
        if card is None:
            self.tree.mark_exhausted()
            logger.warning("credtree: root exhausted after %d duplicate framings in a row at E=%d (reopens after %d "
                           "evaluations without a record)", core.DEDUP_RETRIES, self.E, core.REOPEN_AFTER)
            self._event("exhausted")
            return False
        plan = await self._make_plan(card, None)
        if plan is None:
            logger.warning("credtree: no plan could be written for the new framing; dropped")
            return False
        f = self.tree.add_framing(card)
        logger.warning("credtree: framing %d = %s | %s", f.id, card["problem_class"], card["decision_variables"])
        self._event("framing", framing=f.id, card=card, source="generated")
        await self._start_attempt(f, plan)
        return True

    async def _grow_attempt(self, f: core.Framing) -> bool:
        plan = await self._make_plan(f.card, f)
        if plan is None:
            logger.warning("credtree: no plan could be written under framing %d; skipped", f.id)
            return False
        await self._start_attempt(f, plan)
        return True

    # ------------------------------------------------------------ attempts and seeds (6)
    async def _start_attempt(self, f: core.Framing, plan: str):
        a = self.tree.add_attempt(f, plan)
        a.data = {"programs": [], "history": []}
        logger.warning("credtree: attempt %d under framing %d (prior rho=%.2f) at E=%d: %s", a.id, f.id,
                       a.s0 / a.E0, self.E, plan[:160])
        self._event("attempt", attempt=a.id, framing=f.id, plan=plan, prior=a.s0 / a.E0)
        self._cur = {"how": "new", "d": None, "d_framing": None}
        await self._seed(a, f)

    async def _seed(self, a: core.Attempt, f: core.Framing):
        """Full rewrite of the CURRENT GLOBAL BEST under plan + framing; 1 evaluation; one repair if it crashes."""
        parent = self._program(self.g_id)

        def build(error: str, lean: bool) -> str:
            return _fill("seed_graft" if GRAFT else "seed", framing=_fmt_card(f.card), plan=a.plan,
                         score=f"{self.g_score:.6g}",
                         program=parent.solution, required=", ".join(self.required) or "(see system message)",
                         notes="(omitted)" if lean else self._notes(), rejected=error)

        self.counters["seed_tries"] += 1
        cand = await self._candidate(build, parent, "seed", a)
        if cand is None:
            self.counters["empty_slots"] += 1
            logger.warning("credtree: attempt %d: no seed candidate could be written; it starts from the global best", a.id)
        else:
            res = (await self._evaluate_many([cand]))[0]
            out = self._commit(cand, res, a, f, "seed")
            if out.valid:
                self.counters["seed_valid"] += 1
            else:
                self.counters["seed_crashed"] += 1
                if self.tree.repair_enabled() and self.E < EVAL_CAP and not self._stop():
                    await self._repair(a, f, cand, res)
        self.tree.seed_done(a)
        if a.seed_valid is False:                     # the plan survives; the programs are a copy of the global best
            a.data["programs"].append({"id": self.g_id, "score": self.g_score, "valid": True})
            self._event("seed_fallback", attempt=a.id, copy_of=self.g_id)

    async def _repair(self, a: core.Attempt, f: core.Framing, bad: Dict[str, Any], res):
        """One repair of a crashed seed, with the error message (6.4). Leaf edits are never repaired."""
        err = _error_text(dict(res.metrics or {}), res.artifacts or {})
        fb = ((res.artifacts or {}).get("feedback") or "")[:FEEDBACK_CHARS] or "(none)"
        bad_prog = self._program(bad["id"])

        def build(error: str, lean: bool) -> str:
            return _fill("repair", framing=_fmt_card(f.card), plan=a.plan, program=bad_prog.solution,
                         feedback=fb, notes="(omitted)" if lean else self._notes(), rejected=error)

        self.counters["repair_tried"] += 1
        cand = await self._candidate(build, bad_prog, "repair", a)
        if cand is None:
            self.counters["empty_slots"] += 1
            return
        res2 = (await self._evaluate_many([cand]))[0]
        out = self._commit(cand, res2, a, f, "repair", first_error=err)
        self.counters["repair_valid"] += int(out.valid)
        t = self.tree
        if t.rep[1] == t.c["repair_min_tried"] and not t.repair_enabled():
            logger.warning("credtree: seed repair switched OFF for the rest of the run (%d of %d repairs valid)",
                           t.rep[0], t.rep[1])
            self._event("repair_off", valid=t.rep[0], tried=t.rep[1])

    # ------------------------------------------------------------ diagnosis, hypotheses, lessons (14, v3)
    async def _setup_diag(self, seed: Program):
        """Build the run's diagnoser. With a zero-scoring initial program there is nothing to reproduce yet: the
        module is installed after the static check only and stays untrusted until it reproduces valid programs."""
        self._diag_sem = asyncio.Semaphore(1)       # one diagnosis run at a time, beside the real evaluations
        wd = os.path.join(self.output_dir or ".", "diag")
        os.makedirs(wd, exist_ok=True)

        async def llm(user: str) -> Optional[str]:
            text, _ = await self._llm(user)
            return text

        def build_prompt(sources: str, previous: str, failure: str) -> str:
            fail = (f"\n## Your previous module was rejected\n```python\n{previous[:6000]}\n```\nWhy: {failure}\n"
                    "Fix exactly this problem.\n") if previous else ""
            return _fill("diagnoser", sources=sources, program=self._view(self.initial_solution), failure=fail)

        self.diag = diagmod.Diagnoser(self.evaluation_file, wd, llm, build_prompt, timeout=DIAG_TIMEOUT_FRAC * self.T,
                                      log=lambda kind, **kw: self._event(kind, **kw))
        refs = [(seed.id, seed.solution, (seed.artifacts or {}).get("feedback") or "", self.b0)] if self.b0 > 0 else []
        ok = await self.diag.build(refs)
        logger.warning("credtree: diagnoser %s (v%d, %d reference programs, %.0fs of diagnosis runs)",
                       "installed" if ok else f"NOT built: {self.diag.last_failure[:200]}", self.diag.version,
                       len(refs), self.diag.seconds)

    async def _diagnosis(self, prog: Optional[Program]) -> Optional[str]:
        """Verified diagnosis of an evaluated program, or None (untrusted diagnoser, a program it does not
        reproduce, a non-deterministic program, or the diagnosis time share used up)."""
        d = self.diag
        if (d is not None and d.path is None and not self._rebuilding and d.rebuilds < DIAG_REBUILDS
                and len({round(_score(p.metrics), 9) for p in self.database.programs.values()
                         if _score(p.metrics) > 0}) >= diagmod.TRUST_MIN * (d.rebuilds + 1)):
            self._spawn(self._rebuild_diag())        # the first build failed: retry once more programs exist
        if d is None or prog is None or d.path is None:
            return None
        if prog.id not in d.cache and d.seconds > DIAG_FRAC * self.T:
            return None
        lock = self._diag_locks.setdefault(prog.id, asyncio.Lock())
        async with lock:
            async with self._diag_sem:
                text = await asyncio.to_thread(d.diagnose, prog.id, prog.solution,
                                               (prog.artifacts or {}).get("feedback") or "", _score(prog.metrics))
        if d.needs_rebuild() and not self._rebuilding:
            self._spawn(self._rebuild_diag())
        return text

    async def _rebuild_diag(self):
        d = self.diag
        self._rebuilding = True
        try:
            if d.rebuilds >= DIAG_REBUILDS:
                d.misses = []
                return
            d.rebuilds += 1
            good = [(pid, c) for pid, c in d.cache.items() if c["version"] == d.version and c["ok"]]
            bad = [(pid, c) for pid, c in d.cache.items() if c["version"] == d.version and not c["ok"] and not c["nondet"]]
            if d.path is None:                       # no module yet: reference programs from the database
                valid = sorted((p for p in self.database.programs.values() if _score(p.metrics) > 0),
                               key=lambda p: -_score(p.metrics))
                good = [(p.id, None) for p in valid]
            refs, seen = [], set()
            for pid, _ in bad[-1:] + good[::-1]:
                p = self._program(pid)
                s = round(_score(p.metrics), 9) if p else None
                if p is None or s in seen:
                    continue
                seen.add(s)
                refs.append((p.id, p.solution, (p.artifacts or {}).get("feedback") or "", _score(p.metrics)))
                if len(refs) > diagmod.TRUST_MIN:
                    break
            prev = open(d.path).read() if d.path else ""
            ok = await d.build(refs, previous=prev, failure=d.miss_evidence())
            d.misses = []
            logger.warning("credtree: diagnoser rebuild %d %s (v%d)", d.rebuilds, "ok" if ok else "failed", d.version)
        finally:
            self._rebuilding = False

    def _spawn(self, coro):
        task = asyncio.ensure_future(coro)
        self._bg.add(task)
        task.add_done_callback(self._bg.discard)

    def _lessons(self, f: core.Framing) -> str:
        rows = self.lessons.get(f.id, [])[-LESSONS_SHOWN:]
        return "\n".join(f"- {r}" for r in rows) or "(none yet)"

    @staticmethod
    def _fallback(prog: Optional[Program]) -> str:
        return "(no verified diagnosis for this program; the evaluator's per-instance feedback)\n" + _feedback(prog)

    async def _hypothesis(self, a, f, parent: Program, outcomes: str, diagnosis: str) -> Dict[str, str]:
        user = _fill("micro_idea_diag", framing=_fmt_card(f.card), plan=a.plan, score=f"{_score(parent.metrics):.6g}",
                     program=self._view(parent.solution), diagnosis=diagnosis, lessons=self._lessons(f),
                     outcomes=outcomes)
        text, _ = await self._llm(user)
        hyp = {}
        for key in ("CAUSE", "CHANGE", "PREDICT"):
            m = re.search(rf"{key}:\s*(.+)", text or "")
            hyp[key.lower()] = re.sub(r"\s+", " ", m.group(1)).strip()[:300] if m else ""
        if not hyp["change"]:
            hyp["change"] = re.sub(r"\s+", " ", next((l for l in (text or "").splitlines() if l.strip()), ""))[:300]
        self.counters["hypotheses"] += int(bool(hyp["cause"] and hyp["predict"]))
        return hyp

    async def _verify(self, c: Dict[str, Any], a: core.Attempt, f: core.Framing):
        """Free: diagnose the evaluated child, ask whether the prediction held, keep the lesson."""
        try:
            child, parent, hyp = self._program(c["id"]), c["parent"], c["hypothesis"]
            if child is None:
                return
            cs, ps = _score(child.metrics), _score(parent.metrics)
            if cs > 0:
                child_text = await self._diagnosis(child) or self._fallback(child)
            else:
                child_text = "the program scored 0: " + (_error_text(child.metrics, child.artifacts or {}) or "(no message)")
            user = _fill("verify", cause=hyp.get("cause") or "(not stated)", idea=hyp.get("change") or "(not stated)",
                         predict=hyp.get("predict") or "(not stated)", parent_score=f"{ps:.6g}", child_score=f"{cs:.6g}",
                         parent_diagnosis=c.get("parent_diag") or self._fallback(parent), child_diagnosis=child_text)
            text, _ = await self._llm(user)
            mv = re.search(r"VERDICT:\s*(CONFIRMED|REFUTED|UNCLEAR)", text or "", re.I)
            ml = re.search(r"LESSON:\s*(.+)", text or "")
            if not ml:
                return
            verdict = mv.group(1).upper() if mv else "UNCLEAR"
            lesson = re.sub(r"\s+", " ", ml.group(1)).strip()[:240]
            self.lessons.setdefault(f.id, []).append(f"[{verdict.lower()}, score {ps:.6g} -> {cs:.6g}] {lesson}")
            self.counters["lessons"] += 1
            self.counters["confirmed"] += int(verdict == "CONFIRMED")
            self.counters["refuted"] += int(verdict == "REFUTED")
            self._event("lesson", framing=f.id, attempt=a.id, program=c["id"], verdict=verdict, lesson=lesson,
                        hypothesis=hyp, parent_score=ps, child_score=cs, child_diagnosed=not child_text.startswith("("))
        except Exception as e:
            logger.warning("credtree: verify failed: %s", e)

    # ------------------------------------------------------------ blocks and leaf edits (5, 6)
    async def _run_block(self, a: core.Attempt):
        f = self.tree.framings[a.fid]
        b = self.tree.begin_block(a)
        while b.pending() and not self._stop():
            n = min(b.pending(), EVAL_CAP - self.E)
            cands = await asyncio.gather(*(self._make_edit(a, f) for _ in range(n)))
            got, seen = [], set()
            for c in cands:
                if c is None:
                    self.counters["empty_slots"] += 1
                    self.tree.note_empty(a)
                elif c["hash"] in seen:               # two slots wrote the same program: evaluate one
                    continue
                else:
                    seen.add(c["hash"])
                    got.append(c)
            if not got:
                b.abort()
                break
            results = await self._evaluate_many(got)
            for c, res in zip(got, results):
                out = self._commit(c, res, a, f, "edit")
                b.after_edit(out)
                if c.get("hypothesis"):
                    if len(self._bg) < DIAG_BACKGROUND:
                        self._spawn(self._verify(c, a, f))
                    else:
                        self.counters["verify_skipped"] += 1

    async def _make_edit(self, a: core.Attempt, f: core.Framing) -> Optional[Dict[str, Any]]:
        try:
            members = [(p["id"], p["score"], p["valid"]) for p in a.data["programs"]]
            parent = self._program(core.pick_parent(members, self.tree.G, self.b0, self.rng))
            outcomes = "\n".join(f"- {h}" for h in a.data["history"][-HISTORY:]) or "(no edits evaluated yet)"
            if DIAG:
                return await self._make_edit_diag(a, f, parent, outcomes)
            idea = await self._micro_idea(a, f, parent, outcomes)

            def build(error: str, lean: bool) -> str:
                return _fill("edit", framing=_fmt_card(f.card), plan=a.plan, idea=idea or "(none: use your judgement)",
                             outcomes=outcomes, score=f"{_score(parent.metrics):.6g}", program=parent.solution,
                             feedback="(omitted)" if lean else _feedback(parent),
                             notes="(omitted)" if lean else self._notes(), rejected=error,
                             required=", ".join(self.required) or "(see system message)")

            cand = await self._candidate(build, parent, "edit", a)
            if cand is not None:
                cand["micro_idea"] = idea
            return cand
        except Exception as e:
            logger.exception("credtree: edit on attempt %d failed: %s", a.id, e)
            return None

    async def _make_edit_diag(self, a: core.Attempt, f: core.Framing, parent: Program,
                              outcomes: str) -> Optional[Dict[str, Any]]:
        """v3 leaf edit: verified diagnosis of the parent -> CAUSE / CHANGE / PREDICT -> the edit."""
        diag = await self._diagnosis(parent)
        self.counters["diag_shown" if diag else "diag_missing"] += 1
        shown = diag or self._fallback(parent)
        hyp = await self._hypothesis(a, f, parent, outcomes, shown)

        def build(error: str, lean: bool) -> str:
            return _fill("edit_diag", framing=_fmt_card(f.card), plan=a.plan, cause=hyp.get("cause") or "(not stated)",
                         idea=hyp.get("change") or "(none: use your judgement)",
                         predict=hyp.get("predict") or "(not stated)", diagnosis="(omitted)" if lean else shown,
                         lessons=self._lessons(f), outcomes=outcomes, score=f"{_score(parent.metrics):.6g}",
                         program=parent.solution, notes="(omitted)" if lean else self._notes(), rejected=error,
                         required=", ".join(self.required) or "(see system message)")

        cand = await self._candidate(build, parent, "edit", a)
        if cand is not None:
            cand["micro_idea"] = hyp.get("change")
            cand["hypothesis"] = hyp
            cand["parent_diag"] = diag
        return cand

    async def _micro_idea(self, a, f, parent: Program, outcomes: str) -> str:
        user = _fill("micro_idea", framing=_fmt_card(f.card), plan=a.plan, score=f"{_score(parent.metrics):.6g}",
                     program=self._view(parent.solution), feedback=_feedback(parent), outcomes=outcomes)
        text, _ = await self._llm(user)
        if not text:
            return ""
        m = re.search(r"IDEA:\s*(.+)", text)
        line = (m.group(1) if m else next((l for l in text.splitlines() if l.strip()), "")).strip()
        return re.sub(r"\s+", " ", line)[:300]

    # ------------------------------------------------------------ candidate pipeline (6.1, 6.2)
    def _parse(self, text: str, parent_code: str) -> Tuple[Optional[str], Optional[str], Optional[str]]:
        """(code, changes, error): SEARCH/REPLACE edits on the parent, else a fenced full program."""
        if extract_diffs(text):
            code, err = _strict_apply(parent_code, text)
            return (code, format_diff_summary(extract_diffs(text)), None) if code else (None, None, err)
        code = _fenced(text)
        if code is None:
            return None, None, ("no SEARCH/REPLACE edits and no ```python fenced program found; answer in the "
                                "required format")
        # a full rewrite keeps the frozen code outside the EVOLVE block exactly as the parent has it
        pp, cp = _split_block(parent_code), _split_block(code)
        if pp and cp:
            code = pp[0] + cp[1] + pp[2]
        return code, "Full rewrite", None

    def _static(self, code: str, parent_code: str) -> Optional[str]:
        """Static pre-check (6.1): an error message, or None. Costs no evaluation."""
        if len(code) > MAX_LEN:
            return f"program is {len(code)} characters, over the {MAX_LEN} limit"
        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            return f"the program does not parse: {e.msg} at line {e.lineno}"
        if self.has_markers and _split_block(code) is None:
            return f"the {_BLOCK_START} and {_BLOCK_END} marker comments must stay in the program, in that order"
        missing = [n for n in self.required if n not in _top_level_names(tree)]
        if missing:
            return f"the program no longer defines {', '.join(missing)}, which the evaluator calls"
        for cls, methods in self.contract.items():
            for m in methods:
                if not _provides(tree, m):
                    return (f"the program no longer provides {cls}.{m}, which the evaluator calls on the object "
                            f"your function returns; keep it defined")
        bad = sorted(_unguarded_imports(tree) & self.banned)
        if bad:
            return f"the program imports {', '.join(bad)}, which is not installed; remove the import"
        if code.strip() == (parent_code or "").strip():
            return "the edit did not change the program"
        if _norm_hash(code) in self.cache:
            return "this exact program has already been evaluated; make a real change"
        return None

    async def _candidate(self, build: Callable[[str, bool], str], parent: Program, role: str,
                         a: core.Attempt) -> Optional[Dict[str, Any]]:
        """Generate -> static check (regenerate <= REGEN) -> self-review (<= REVIEW_ROUNDS revise) -> candidate."""
        error, llm_time, rejections = "", 0.0, []
        code = changes = last_text = last_code = prompt = None
        for k in range(1 + REGEN):
            rej = (f"\n## Your previous answer was rejected before evaluation\n{error}\n" if error else "")
            user = build(rej, False)
            if len(user) > PROMPT_CHAR_BUDGET:
                user = build(rej, True)
            prompt = {"system": self.system, "user": user}
            text, dt = await self._llm(user)
            llm_time += dt
            if not text:
                error = "empty response"
                rejections.append(error)
                continue
            c, ch, err = self._parse(text, parent.solution)
            if err is None:
                err = self._static(c, parent.solution)
            if err is None:
                code, changes, last_text = c, ch, text
                break
            rejections.append(err)
            self.counters["prechecks"] += 1
            error = err
            if c is not None:
                last_code, last_text = c, text
            if k < REGEN:
                self.counters["regens"] += 1
        if code is None and last_code is not None:
            # still failing the static check: one free self-review with the problem spelled out
            fixed, dt = await self._review(last_code, rejections[-1:])
            llm_time += dt
            if fixed and self._static(fixed, parent.solution) is None:
                code, changes = fixed, (changes or "self-review fix")
                self.counters["revised"] += 1
        elif code is not None:
            for _ in range(REVIEW_ROUNDS):
                fixed, dt = await self._review(code, [])
                llm_time += dt
                if fixed and self._static(fixed, parent.solution) is None:
                    code, changes = fixed, (changes or "") + " + review"
                    self.counters["revised"] += 1
        if code is None:
            self._event("no_candidate", role=role, attempt=a.id, rejections=rejections)
            return None
        return {"id": str(uuid.uuid4()), "code": code, "hash": _norm_hash(code), "role": role, "parent": parent,
                "llm_time": llm_time, "prompt": prompt, "response": last_text, "changes": changes,
                "rejections": rejections}

    async def _review(self, code: str, errors: List[str]) -> Tuple[Optional[str], float]:
        """Free LLM self-review against the written checks. Returns revised code or None (no change)."""
        user = _fill("review", program=code, errors="\n".join(f"- {e}" for e in errors) or "(none)",
                     evaluator=self._evaluator_section(audit=False))
        if len(user) > PROMPT_CHAR_BUDGET:
            return None, 0.0
        self.counters["reviews"] += 1
        text, dt = await self._llm(user)
        if not text or not re.search(r"VERDICT:\s*FIX", text, re.I):
            return None, dt
        new, _ = _strict_apply(code, text)
        return (new if new and new.strip() != code.strip() else None), dt

    # ------------------------------------------------------------ evaluation and recording
    async def _evaluate_many(self, cands: List[Dict[str, Any]]):
        async def one(c):
            t0 = time.time()
            try:
                res = await self.evaluator.evaluate_program(c["code"], c["id"])
            except Exception as e:                    # the call was made, so it counts as an evaluation
                logger.warning("credtree: evaluator raised: %s", e)
                res = type("R", (), {"metrics": {"combined_score": 0.0, "error": f"evaluator raised: {e}"},
                                     "artifacts": {}})()
            c["eval_time"] = time.time() - t0
            return res
        return await asyncio.gather(*(one(c) for c in cands))

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
        return "\n".join(f"- {n}x: {ex}" for n, ex in rows)[:NOTES_CHARS]

    def _commit(self, c: Dict[str, Any], res, a: core.Attempt, f: core.Framing, role: str,
                first_error: str = "") -> core.Outcome:
        """Record one evaluated candidate: database, core credibility, the three logs. Returns the outcome."""
        metrics = dict(res.metrics or {})
        score = _score(metrics)
        valid = score > 0 and math.isfinite(score)
        self.E += 1
        self.iteration += 1
        self.cache.add(c["hash"])
        parent = c["parent"]
        prev_best, prev_G = a.best, self.tree.G
        out = self.tree.record(a, valid, score, role)
        prog = Program(
            id=c["id"], solution=c["code"], language=self.config.language, metrics=metrics,
            iteration_found=self.iteration, parent_id=parent.id if parent else None, other_context_ids=[],
            generation=(parent.generation + 1) if parent else 0,
            metadata={"changes": c.get("changes"), "parent_metrics": parent.metrics if parent else None,
                      "framing_id": f.id, "attempt_id": a.id, "role": role, "micro_idea": c.get("micro_idea"),
                      "success": out.success, "global_record": out.record},
            artifacts=res.artifacts or {})
        if valid:
            a.data["programs"].append({"id": prog.id, "score": score, "valid": True})
        if valid and score > self.g_score:
            self.g_id, self.g_score = prog.id, score
        pscore = _score(parent.metrics) if parent else 0.0
        err = "" if valid else _error_text(metrics, res.artifacts or {})
        if role == "edit":
            idea = (c.get("micro_idea") or "")[:100]
            if valid:
                msg = (f"{'improved the attempt best' if out.success else 'no gain'}: score {score:.6g} "
                       f"({score - pscore:+.4g} vs its parent {pscore:.6g})")
            else:
                msg = f"invalid, scored 0: {err}"
            a.data["history"].append(f"[{idea}] {msg}" if idea else msg)
        self._note_feedback(prog)
        result = SerializableResult(
            child_program_dict=prog.to_dict(), parent_id=prog.parent_id, other_context_ids=[],
            iteration_time=c["llm_time"] + c.get("eval_time", 0.0), llm_generation_time=c["llm_time"],
            eval_time=c.get("eval_time", 0.0), prompt=c.get("prompt"), llm_response=c.get("response"),
            iteration=self.iteration)
        self._process_iteration_result(result, self.iteration, self._ckpt)
        now = datetime.now().isoformat()
        ns = self.tree.node_stats(a)
        cur = self._cur if role == "edit" else {"how": "new", "d": None, "d_framing": None}
        self._append(self._tree_path, {
            "eval": self.E, "iteration": self.iteration, "timestamp": now, "framing": f.id, "attempt": a.id,
            "role": role, "how": cur.get("how"), "parent": prog.parent_id, "program": prog.id, "score": score,
            "valid": valid, "success": out.success, "global_record": out.record, "G": self.tree.G,
            "prev_best": prev_best, "prev_G": prev_G, "age": out.age, "h": out.h, "gap_bucket": out.gap_bucket,
            "d": cur.get("d"), "d_framing": cur.get("d_framing"), "micro_idea": c.get("micro_idea"),
            "prechecks": c.get("rejections"), "error": err or first_error or None, "nodes": ns,
            **({"hypothesis": c.get("hypothesis"), "diagnosed": bool(c.get("parent_diag"))} if DIAG else {})})
        self._append(self._stats_path, {
            "iteration": self.iteration, "timestamp": now, "evaluations": self.E,
            "global": {"global_best_score": self.tree.G},
            "iteration_result": {"success": True, "child_program": {
                "id": prog.id, "metrics": prog.metrics, "parent_id": prog.parent_id}},
            "credtree": {"framing": f.id, "attempt": a.id, "role": role, "how": cur.get("how"),
                         "valid": valid, "success": out.success, "global_record": out.record,
                         "framings": len(self.tree.framings), "attempts": len(self.tree.attempts),
                         "N": self.tree.N, "exhausted": self.tree.exhausted}})
        logger.warning("credtree: E=%d F%d/A%d %s%s score=%.6g (parent %.6g) best_A=%.6g G=%.6g%s%s", self.E, f.id,
                       a.id, role, f" age={out.age}" if role == "edit" else "", score, pscore,
                       a.best if a.best is not None else float("nan"), self.tree.G,
                       " SUCCESS" if out.success else "", " RECORD" if out.record else "")
        return out


def install():
    """Register search type `credtree`. Called from run_capped.py when CREDTREE=1."""
    for name in TEMPLATES:                            # fail loudly on a missing prompt, not at the first call
        _tpl(name)
    _DB_CONFIG_BY_TYPE["credtree"] = AdaEvolveDatabaseConfig     # archive/intensity defaults, unused
    register_database("credtree", CredTreeDatabase)
    register_controller("credtree", CredTreeController)
    from skydiscover.optimize.search.route import _CONTROLLER_REGISTRY
    assert _CONTROLLER_REGISTRY.get("credtree") is CredTreeController, "credtree controller not registered"
    logger.warning("credtree installed: prompts=%s cap=%d constants=%s", _prompt_dir(), EVAL_CAP,
                   json.dumps(core.constants(), default=str))
