"""Arm 3: two-step strategy-specification generation, layered on AdaEvolve.

Every program carries a strategy specification: prose describing the approach its
code implements. Each iteration either inherits the parent's specification or, when
the search has stalled, spends one extra LLM call revising it first. The code call
then implements whatever specification it was handed.

The motivation is that a 30B generator is poor at inferring a strategy from code,
mutating that strategy, and re-expressing it as a diff, all inside one response. The
two steps separate reasoning from translation and give each its own output budget --
the protocol fixes max_tokens at 8000 per call, not per iteration, and Section 2.4
states that extra LLM calls for reflection do not count against the 500 evaluations.

What this does NOT change: parent selection, islands, migration, the archive, the
budget, the evaluator. AdaEvolve's population management is untouched, so arm 3 minus
arm 2 isolates the prompting change.

Stagnation is "no improvement above STRATEGY_DELTA within the last STRATEGY_K
evaluations". Measured on the fixed-arm runs, new bests arrive every 4-42 evaluations
and the median gain is 1.3-2.8%, so k=10 / delta=0.02 fires during most dry spells
while leaving a productive streak alone. A revised specification is protected for
STRATEGY_MIN_TRIALS children before it can itself be declared stagnant, otherwise
aggressive triggering degenerates into random restart.
"""
import collections
import hashlib
import logging
import os
import re

logger = logging.getLogger("strategy_spec")

K = int(os.environ.get("STRATEGY_K", "10"))
DELTA = float(os.environ.get("STRATEGY_DELTA", "0.02"))
MIN_TRIALS = int(os.environ.get("STRATEGY_MIN_TRIALS", "5"))
PROMPTS_DIR = os.environ.get("PROMPTS_DIR", "")
LANG = os.environ.get("STRATEGY_LANG", "en")

_SPEC_RE = re.compile(r"<<<STRATEGY>>>(.*?)<<<END STRATEGY>>>", re.S)
_SKILL_RE = re.compile(r"<<<SKILL>>>(.*?)<<<END SKILL>>>", re.S)
SKILL_MAX = int(os.environ.get("STRATEGY_SKILL_MAX", "1800"))
# Set here rather than in run_task.sh: that file is bash, and bash reads a
# script incrementally, so editing it desynchronises every job already
# running it. Python is imported once at start, so this is safe to change.
NUM_CONTEXT = int(os.environ.get("NUM_CONTEXT", "2"))
# Two axes, chosen separately: the most-tried strategies show what has been
# exhausted, the best-scoring show what is worth building on. A single
# ranking hides one or the other.
PORTFOLIO_TRIED = int(os.environ.get("STRATEGY_PORTFOLIO_TRIED", "3"))
PORTFOLIO_BEST = int(os.environ.get("STRATEGY_PORTFOLIO_BEST", "3"))
PORTFOLIO_CHARS = int(os.environ.get("STRATEGY_PORTFOLIO_CHARS", "2600"))


def _load(name):
    with open(os.path.join(PROMPTS_DIR, name)) as fh:
        return fh.read()


def _parse_skill(text):
    """Pull the run-level skill notes out of a step-1 response, or None."""
    if not text:
        return None
    m = _SKILL_RE.search(text)
    if not m:
        return None
    sk = m.group(1).strip()
    return sk[:SKILL_MAX] if sk else None


def _parse_spec(text):
    """Pull the specification out of a step0/step1 response, or None if absent."""
    if not text:
        return None
    m = _SPEC_RE.search(text)
    if not m:
        return None
    spec = m.group(1).strip()
    return spec or None


class _State:
    """Per-run bookkeeping: specifications, their age, and the stagnation window."""

    def __init__(self):
        self.spec_of = {}          # program id -> specification text
        self.trials = {}           # specification key -> children evaluated under it
        self.best = None
        self.since_improvement = 0
        self.revisions = 0
        self.seeded = False
        # Run-level, not per-program: what earlier attempts learned about what the
        # evaluator actually enforces and which approaches are already ruled out.
        self.skill = "(nothing recorded yet)"
        # Per-strategy ledger. Specs are attached to programs and parents are drawn
        # from a pruned archive, so a strategy survives about two iterations before
        # the sampler moves to another lineage: one prism run alternated between two
        # texts for 29 programs while three genuine rewrites died after one child
        # each. The revision call cannot avoid re-proposing what has already failed
        # unless it is shown the whole portfolio.
        self.ledger = collections.OrderedDict()   # key -> stats

    @staticmethod
    def _key(spec):
        return hashlib.md5((spec or "").encode()).hexdigest()[:8]

    def record(self, score):
        """Advance the stagnation window. A gain must exceed DELTA to reset it."""
        if score is None:
            return
        if self.best is None:
            self.best = score
            self.since_improvement = 0
            return
        gain = (score - self.best) / abs(self.best) if self.best else (1.0 if score > 0 else 0.0)
        if score > self.best:
            self.best = score
        if gain > DELTA:
            self.since_improvement = 0
        else:
            self.since_improvement += 1

    def should_revise(self, spec):
        """Revise only when the search has stalled AND this spec has had a fair run."""
        if self.since_improvement < K:
            return False
        return self.trials.get(self._key(spec), 0) >= MIN_TRIALS

    def bump(self, spec):
        k = self._key(spec)
        self.trials[k] = self.trials.get(k, 0) + 1

    def note(self, spec, score, feedback):
        """Record one evaluated child against the strategy that produced it."""
        e = self.ledger.setdefault(self._key(spec), {
            "text": spec or "", "n": 0, "best": None, "zero": 0,
            "errors": collections.Counter(), "first_seen": len(self.ledger) + 1})
        e["n"] += 1
        if score:
            e["best"] = score if e["best"] is None else max(e["best"], score)
        else:
            e["zero"] += 1
            # Real evaluator feedback writes the class inside parentheses
            # ("error, score=0 (ValueError: ...)"), but not always; match
            # either, and treat a scoreless run with no exception as the
            # distinct case it is -- the program ran and was rejected.
            m = re.search(r"(\w*Error)\b", feedback or "")
            e["errors"][m.group(1) if m else "invalid output (ran, rejected)"] += 1

    def portfolio(self, k_tried=None, k_best=None, budget=None):
        """The strategies tried so far: the k1 most-tried and the k2 best-scoring.

        Capped because the window is 16k tokens and program sources already claim
        most of it. Labelled by which list each entry came from, so the revision
        can tell "this has been hammered and is exhausted" from "this looks
        promising and is under-explored".
        """
        k_tried = PORTFOLIO_TRIED if k_tried is None else k_tried
        k_best = PORTFOLIO_BEST if k_best is None else k_best
        budget = PORTFOLIO_CHARS if budget is None else budget
        if not self.ledger:
            return "(no strategy has been evaluated yet)"
        rows = list(self.ledger.values())
        most = sorted(rows, key=lambda e: -e["n"])[:k_tried]
        scored = [e for e in rows if e["best"] is not None]
        top = [e for e in sorted(scored, key=lambda e: -e["best"]) if e not in most][:k_best]
        tags = {id(e): "most tried" for e in most}
        for e in top:
            tags[id(e)] = "best scoring"
        picked = sorted(most + top, key=lambda e: e["first_seen"])
        out, used, shown = [], 0, 0
        for e in picked:
            label = "nothing admissible" if e["best"] is None else f"best {e['best']:.4g}"
            fails = ", ".join(f"{k} x{v}" for k, v in e["errors"].most_common(3)) or "none"
            head = " ".join(e["text"].split())[:200]
            blk = (f"- Strategy {e['first_seen']} ({tags[id(e)]}): tried on {e['n']} "
                   f"program(s), {label}, {e['zero']} of {e['n']} scored zero.\n"
                   f"  Failures seen: {fails}\n"
                   f"  It said: \"{head}...\"")
            if used + len(blk) > budget:
                break
            out.append(blk)
            used += len(blk)
            shown += 1
        hidden = len(self.ledger) - shown
        if hidden > 0:
            out.append(f"- ... {hidden} further strategies tried and not shown here.")
        return "\n".join(out)


STATE = _State()


def _feedback_of(program):
    try:
        return (program.artifacts or {}).get("feedback", "") or "(no feedback recorded)"
    except Exception:
        return "(no feedback recorded)"


def _score_of(program):
    try:
        return (program.metrics or {}).get("combined_score")
    except Exception:
        return None


def _history(database, limit=6):
    """Recent programs and their scores, newest last, as plain prose for step 1."""
    try:
        progs = sorted(
            database.programs.values(),
            key=lambda p: (getattr(p, "iteration_found", 0) or 0),
        )[-limit:]
    except Exception:
        return "(no history available)"
    lines = []
    for p in progs:
        s = _score_of(p)
        lines.append(f"- iteration {getattr(p, 'iteration_found', '?')}: score "
                     f"{'n/a' if s is None else round(s, 6)}")
    return "\n".join(lines) or "(no history available)"


def _patch_sample(db_cls):
    """Let the wrapper pre-sample a parent and have the inner call reuse it.

    _run_iteration samples its own parent, so without this the specification would be
    prepared for one program and the diff generated against another.
    """
    if getattr(db_cls, "_arm3_patched", False):
        return
    orig = db_cls.sample

    def sample(self, *args, **kwargs):
        pending = getattr(self, "_arm3_pending", None)
        if pending is not None:
            self._arm3_pending = None
            return pending
        return orig(self, *args, **kwargs)

    db_cls.sample = sample
    db_cls._arm3_patched = True


def install():
    """Patch AdaEvolve's generation path.

    An earlier version patched DiscoveryController._run_iteration and _build_prompt.
    Both are dead ends under AdaEvolve: AdaEvolveController overrides _run_iteration
    with a different signature, and its _generate_child calls
    self.context_builder.build_prompt(...) directly rather than self._build_prompt.
    The patch installed cleanly, logged that it had installed, and did nothing --
    a debug pilot produced 7 programs, none carrying a specification. Hook
    _generate_child and the context builder instead, and fail loudly if the
    controller is not AdaEvolve rather than repeating that silence.
    """
    from skydiscover.optimize.search.adaevolve.controller import AdaEvolveController
    from skydiscover.optimize.context_builder.adaevolve import AdaEvolveContextBuilder

    step0 = _load("step0_seed_spec.txt")
    step1 = _load("step1_strategy.txt")
    inject = _load("inject_step2.txt")
    mode_rewrite = _load("_mode_rewrite.txt")
    mode_refine = _load("_mode_refine.txt")
    clause = _load("_language_clause_zh.txt" if LANG == "zh" else "_language_clause_en.txt")

    orig_generate_child = AdaEvolveController._generate_child
    orig_build_prompt = AdaEvolveContextBuilder.build_prompt

    def _system(obj):
        try:
            return obj._get_system_message()
        except Exception:
            return ""

    async def _seed_spec(self, parent):
        text = (await self._call_llm(_system(self.context_builder), step0.format(
            language_clause=clause,
            current_program=parent.solution,
            score=_score_of(parent),
            feedback=_feedback_of(parent),
        ))).text
        spec = _parse_spec(text)
        if spec is None:
            logger.warning("seed specification had no <<<STRATEGY>>> block; using a stub")
            spec = "The initial program's approach, unchanged. No specification was recovered."
        return spec

    async def _revise_spec(self, parent, spec):
        reason = (f"No improvement above {DELTA:.0%} in the last {STATE.since_improvement} "
                  f"evaluations (threshold {K}).")
        text = (await self._call_llm(_system(self.context_builder), step1.format(
            language_clause=clause,
            skill=STATE.skill,
            portfolio=STATE.portfolio(),
            current_spec=spec,
            score=_score_of(parent),
            metrics=str(parent.metrics or {}),
            feedback=_feedback_of(parent),
            history=_history(self.database),
            trigger_reason=reason,
        ))).text
        sk = _parse_skill(text)
        if sk:
            STATE.skill = sk
            logger.warning("skill notes updated (%d chars)", len(sk))
        revised = _parse_spec(text)
        if revised is None:
            logger.warning("revision had no <<<STRATEGY>>> block; keeping the current spec")
            return spec, False
        STATE.revisions += 1
        STATE.since_improvement = 0
        logger.warning("strategy revised (#%d) after %d stagnant evaluations",
                       STATE.revisions, K)
        return revised, True

    async def _generate_child(self, iteration, error_context=None, force_exploration=False):
        # Pre-sample so the specification is prepared for the program the diff will
        # actually be generated against. _generate_child samples again internally;
        # the cached value is handed back to it once.
        # Buy prompt space for the skill notes and the specification.
        self.num_context_programs = NUM_CONTEXT
        _patch_sample(type(self.database))
        pre = self.database.sample(num_context_programs=self.num_context_programs)
        self.database._arm3_pending = pre
        raw_parent = pre[0]
        parent = (list(raw_parent.values())[0] if isinstance(raw_parent, dict) else raw_parent)

        spec = STATE.spec_of.get(parent.id)
        if spec is None:
            spec = await _seed_spec(self, parent)
            STATE.spec_of[parent.id] = spec
            STATE.seeded = True
        rewrote = False
        if STATE.should_revise(spec):
            spec, rewrote = await _revise_spec(self, parent, spec)

        self.context_builder._arm3_spec = spec
        self.context_builder._arm3_mode = mode_rewrite if rewrote else mode_refine
        self.context_builder._arm3_timeout = getattr(
            getattr(self.config, "evaluator", None), "timeout", 360)
        # Step 1 is told to design from scratch when the search stalls. Asking step 2
        # for a minimal diff in the same breath is a contradiction: the strategy gets
        # rewritten and the program never follows. On a rewrite, switch the framework
        # to its full-rewrite template -- the parser reads the same flag, so prompt
        # and parse stay consistent -- and restore diff mode afterwards.
        prev_diff = getattr(self.config, "diff_based_generation", True)
        if rewrote:
            self.config.diff_based_generation = False
            logger.warning("strategy rewritten: generating a full program this iteration")
        try:
            result = await orig_generate_child(self, iteration, error_context, force_exploration)
        finally:
            self.config.diff_based_generation = prev_diff
            self.context_builder._arm3_spec = None
            self.context_builder._arm3_mode = None

        child = getattr(result, "child_program_dict", None)
        if child:
            md = child.get("metadata")
            if not isinstance(md, dict):
                md = {}
                child["metadata"] = md
            md["strategy_spec"] = spec
            cid = child.get("id")
            if cid:
                STATE.spec_of[cid] = spec
            STATE.bump(spec)
            sc = (child.get("metrics") or {}).get("combined_score")
            STATE.note(spec, sc, ((child.get("artifacts") or {}).get("feedback") or ""))
            STATE.record(sc)
        return result

    def _build_prompt(self, current_program, context=None, **kwargs):
        spec = getattr(self, "_arm3_spec", None)
        prompt = orig_build_prompt(self, current_program, context, **kwargs)
        if not spec:
            return prompt
        # Augment, do not replace. An earlier version built its own lean template
        # and silently dropped the context programs and attempt history, taking the
        # median prompt from 49,902 to 8,206 characters -- that measured "the spec
        # minus all context", not "the spec".
        block = inject.format(skill=STATE.skill, spec=spec,
                              mode=getattr(self, "_arm3_mode", "") or "")
        user = prompt.get("user", "")
        marker = "\n# Task"
        i = user.find(marker)
        prompt["user"] = (user[:i] + block + user[i:]) if i >= 0 else (user + block)
        return prompt

    AdaEvolveController._generate_child = _generate_child
    AdaEvolveContextBuilder.build_prompt = _build_prompt
    logger.warning("strategy-spec arm installed on AdaEvolve: k=%d delta=%.3f "
                   "min_trials=%d lang=%s", K, DELTA, MIN_TRIALS, LANG)
