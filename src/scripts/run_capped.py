"""Run skydiscover with the protocol's budget: a cap on EVALUATIONS, not LLM calls.

  EVAL_CAP=500 python run_capped.py optimize <program> <evaluator> -c ... -i 3000

Section 2.4 caps evaluations at 500 per task and says in as many words that
"additional LLM calls for reflection, critique, multi-agent approaches, or other
purposes are allowed and do not count toward the evaluation budget". EvoX's own
`-i` budget cannot express that: it advances by `attempts_used`
(evox/controller.py:208), so a diff the generator malforms spends budget without
ever reaching the evaluator.

This wrapper separates the two. `-i` is set high enough never to bind; the real
stop is a counter on Evaluator.evaluate_program, the single chokepoint every
task evaluation passes through. On reaching the cap it sets the controller's
shutdown_event, which the search loop already checks at the top of each
iteration (evox/controller.py:184), so the run ends the same way a Ctrl-C ends
it: current iteration finishes, checkpoint is written, best program returned.

What is deliberately NOT counted:
  - mode="test", the single re-score of the winner after the loop (runner.py:221);
  - meta-evolution, which scores *database implementations* rather than task
    programs (evox/database/search_strategy_evaluator.py) and never calls the
    task evaluator;
  - retried LLM calls, which is the entire point.

The search algorithm is untouched -- this only decides when to stop it.
"""
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from skydiscover.optimize.evaluation.evaluator import Evaluator
from skydiscover.optimize.search.default_discovery_controller import (
    DiscoveryController,
)
from skydiscover.optimize.context_builder.default.builder import (
    DefaultContextBuilder,
)
import skydiscover.main as skydiscover_main

logger = logging.getLogger("eval_cap")

CAP = int(os.environ.get("EVAL_CAP", "500"))

# EVAL_DONE: evaluations a resumed run already spent before it died (jobs/22_resume.pbs); 0 for a fresh run.
_count = int(os.environ.get("EVAL_DONE", "0"))
_controllers = []

_orig_init = DiscoveryController.__init__


def _init(self, *args, **kwargs):
    # CoEvolutionController.__init__ calls super(), so this catches it and every
    # other search type -- the cap does not depend on which search runs.
    _orig_init(self, *args, **kwargs)
    _controllers.append(self)


_orig_evaluate = Evaluator.evaluate_program


async def _evaluate(self, program_solution, program_id="", mode="train"):
    global _count
    if mode != "test":
        _count += 1
        if _count >= CAP:
            # Set before awaiting, so this call is the CAP-th evaluation and the
            # loop stops rather than starting a CAP+1-th.
            for c in _controllers:
                c.shutdown_event.set()
            logger.warning("evaluation budget reached: %d/%d — stopping search", _count, CAP)
    return await _orig_evaluate(self, program_solution, program_id=program_id, mode=mode)


DiscoveryController.__init__ = _init
Evaluator.evaluate_program = _evaluate


# ---------------------------------------------------------------- prompt size
# Nothing in the prompt assembly budgets tokens: the parent and every context
# program go in whole (context_builder/default/builder.py), and
# `max_solution_length` is a monitor display setting, not a cap on generated
# code. With num_context_programs=4 and a 13k-character program, the user
# message passes 16384 tokens and vLLM rejects the call outright --
# 585 times in one signal_processing run, each retried 3x and charged to the
# budget. Rebuild without the context programs rather than truncating them:
# half a program is worse context than none.
PROMPT_CHAR_BUDGET = int(os.environ.get("PROMPT_CHAR_BUDGET", "45000"))

_orig_build_prompt = DefaultContextBuilder.build_prompt


def _build_prompt(self, *args, **kwargs):
    prompt = _orig_build_prompt(self, *args, **kwargs)
    user = prompt.get("user", "")
    # AdaEvolve calls build_prompt(parent_dict, context) positionally while the
    # default controller uses keywords. Reading kwargs alone meant this guard
    # never fired for AdaEvolve: cloudcast sent 2045 prompts at a median of
    # 84,444 chars and had 2034 of them rejected outright.
    ctx = kwargs.get("context")
    if ctx is None and len(args) >= 2 and isinstance(args[1], dict):
        ctx = args[1]
    if len(user) <= PROMPT_CHAR_BUDGET or not isinstance(ctx, dict):
        return prompt

    ocp = ctx.get("other_context_programs")
    if isinstance(ocp, dict):
        items = list(ocp.items())
        rebuild = lambda n: dict(items[:n])
    elif isinstance(ocp, list):
        items = list(ocp)
        rebuild = lambda n: items[:n]
    else:
        return prompt

    # Drop one context program at a time instead of all of them, so a task
    # whose programs are small keeps the full set and a task whose programs
    # are huge keeps as many as still fit.
    for n in range(len(items) - 1, -1, -1):
        shrunk_ctx = dict(ctx)
        shrunk_ctx["other_context_programs"] = rebuild(n)
        if "context" in kwargs:
            retry_args, retry_kwargs = args, dict(kwargs, context=shrunk_ctx)
        else:
            retry_args = list(args); retry_args[1] = shrunk_ctx
            retry_args, retry_kwargs = tuple(retry_args), dict(kwargs)
        try:
            cand = _orig_build_prompt(self, *retry_args, **retry_kwargs)
        except Exception:
            return prompt
        size = len(cand.get("user", ""))
        if size <= PROMPT_CHAR_BUDGET:
            logger.warning(
                "prompt %d chars > budget %d; kept %d/%d context programs -> %d chars",
                len(user), PROMPT_CHAR_BUDGET, n, len(items), size)
            return cand
        if n == 0:
            # Nothing left to shed. Returning it anyway costs three retries on a
            # call the server will certainly reject, so mark it and let the
            # caller fail the iteration immediately.
            logger.warning("prompt still %d chars with no context programs; "
                           "iteration cannot fit the window", size)
            cand["oversize"] = True
            return cand
    return prompt


def _guard(cls):
    """Bound the prompt this builder emits, whatever it adds after super()."""
    orig = cls.build_prompt

    def guarded(self, *args, **kwargs):
        global _orig_build_prompt
        prev, _orig_build_prompt = _orig_build_prompt, orig
        try:
            return _build_prompt(self, *args, **kwargs)
        finally:
            _orig_build_prompt = prev

    cls.build_prompt = guarded


# AdaEvolveContextBuilder subclasses DefaultContextBuilder and appends its own
# paradigm text after calling super(), so guarding only the parent would bound the
# inner prompt and let the final one exceed the window again. Guard both, outermost
# last: the subclass check sees the fully assembled message.
_guard(DefaultContextBuilder)
try:
    from skydiscover.optimize.context_builder.adaevolve import AdaEvolveContextBuilder
    _guard(AdaEvolveContextBuilder)
except Exception as exc:                      # search type not installed in this venv
    logger.debug("adaevolve context builder not guarded: %s", exc)


# ------------------------------------------------------- attempt-history label
# _determine_outcome defaults a missing parent score to 0 (builder.py:453), and
# every task score is positive, so a program with no parent is reported to the
# model as "Improvement in combined_score". It fires once per run, on the
# initial program. Report the truth instead.
_orig_outcome = DefaultContextBuilder._determine_outcome


def _determine_outcome(program_metrics, parent_metrics):
    if parent_metrics is None or "combined_score" not in (parent_metrics or {}):
        return "No parent to compare against"
    return _orig_outcome(program_metrics, parent_metrics)


DefaultContextBuilder._determine_outcome = staticmethod(_determine_outcome)

# ------------------------------------------------------------------ arm 3
# Opt-in: the two-step strategy-specification mechanism. Off by default, so the
# baseline arms run through this wrapper with only the bug fixes applied.
if os.environ.get("STRATEGY_SPEC") == "1":
    import strategy_spec
    strategy_spec.install()

# ------------------------------------------------------------------ arm 4
# Opt-in: SPROUT (scripts/sprout.py), registered as search type "sprout". It must be
# installed before skydiscover parses the config, which maps search.type to a
# database-config class.
if os.environ.get("SPROUT") == "1":
    import sprout
    sprout.install()

# ------------------------------------------------------------------ CredTree
# Opt-in: the framing -> attempt credibility tree (scripts/credtree.py), registered as search
# type "credtree". Same constraint as SPROUT: installed before skydiscover parses the config.
if os.environ.get("CREDTREE") == "1":
    import credtree
    credtree.install()

# ------------------------------------------------------------------ BANYAN
# Opt-in: the incumbent-rooted credibility tree (scripts/banyan.py), registered as search type "banyan".
# Builds on the CredTree modules, which it imports but does not change; CREDTREE and BANYAN are separate switches.
if os.environ.get("BANYAN") == "1":
    import banyan
    banyan.install()


if __name__ == "__main__":
    logging.getLogger("eval_cap").setLevel(logging.INFO)
    sys.exit(skydiscover_main.main())
