# Brief: write prompts/v5

Produce the best system prompts we can defend, one per task, plus a written defence.

## Deliverables (all under `prompts/v5/`)

1. `system_<task>.txt` for all five tasks: prism, cloudcast, signal_processing,
   heilbronn_triangle, circle_packing_rect.
2. `RATIONALE.md`: the defence. Write it as if you are presenting these prompts and
   must justify every choice to someone who will push back. For each task: what the
   evaluator actually enforces, which failure modes you are targeting, what you
   deliberately did NOT say and why, and which failures prompting cannot fix.
3. `FAILURES.md`: the catalogue. Every common failure mode we have observed, per task,
   with how often it occurs, and a verdict on each: solvable by prompting, partly
   solvable, or not solvable by prompting (say what would be needed instead).
4. Extend `scripts/verify_prompt_contracts.py` with cases for anything new you assert,
   and report the pass count. Do not weaken existing cases.

## Hard constraints (COL8385_Project.pdf, in the project root -- read section 2.3 and 5)

- Prompts MAY be task-specific. The search algorithm's code and configuration must be
  identical across tasks, so anything task-specific belongs in the prompt or nowhere.
- Prompts MUST NOT contain task-specific program code written or obtained outside the
  run. No reference solutions, no known constructions, no literature coordinates.
  Describing *what the evaluator checks* is fine; handing over an answer is not.
- Our own additional rule: **no statistics measured in our previous runs** may appear in
  a prompt ("in prior runs 45% of candidates..."). It is indefensible to a grader. State
  what the evaluator rejects, not what our runs did.
- Do not prescribe a method. State the output contract and the constraints; let the
  search find the algorithm. We have direct evidence this matters: a v1 line telling the
  model to "handle the first and last window-lengths explicitly" pushed it off a working
  one-liner and collapsed signal_processing from 89% admissible to 0.4%.
- Every factual claim about the evaluator must be true. A wrong claim is worse than
  silence: v1-v4 told the model the signal output must be the same length as the input,
  when the evaluator requires len(x) - window_size + 1. Admissibility went to 1/70.
  After correcting it: 96%.

## Search-algorithm agnostic

These prompts will be used by several arms (EvoX, AdaEvolve, SPROUT, and ablations).
So the system prompt must not assume any particular search loop: no references to
lineages, strategies, islands, parents, iterations, or "your previous attempt". It is
the task contract plus the objective, nothing about how the search works.

## What to read

| what | where |
|---|---|
| the task evaluators -- THE source of truth | `extracted/col8385_project/problems/<task>/evaluator.py`, plus `data.py`, `metrics.py`, `sim/` for cloudcast |
| the shipped prompts (the baseline) | `extracted/col8385_project/problems/<task>/prompt.txt` |
| our prompt history | `prompts/v1` .. `prompts/v4`; v4 is current and was verified against evaluator code |
| the contract verifier (21/21 passing on v4) | `scripts/verify_prompt_contracts.py`, run via `jobs/43_verify_prompts.pbs` |
| patched evaluators (a second mode we run) | `extracted/col8385_project_patched/problems/{cloudcast,signal_processing}/evaluator.py` |
| stored programs with scores and evaluator feedback | `runs_final/*/*/sprout_*/checkpoints/checkpoint_*/programs/*.json` and the same under `runs_r2/`, `runs_v1/` |
| per-evaluation stream | `<run>/sprout_iteration_stats_*.jsonl`, `<run>/run.log` |
| arm-to-arm differences and known traps | `INCONSISTENCIES.md`, `.claude/skills/new-arm/SKILL.md` |

## Findings you should build on (verify any you rely on)

1. **The math prompts have been telling the model to be fast, on tasks where time buys
   score.** The shipped heilbronn prompt lists `eval_time` as "EFFICIENCY - minimize";
   circle says "keep reasonable, prefer accuracy over speed". v1-v3 carried both. v4
   removed them but mentions the 360 s limit only as a failure threshold.
   Measured, pooled over our runs: on circle the fastest quartile of admissible programs
   averages 0.6734 and the slowest 0.8082; the 25 best programs have a median runtime of
   52.7 s against 0.92 s for all admissible. On heilbronn: 0.3104 vs 0.5164, and the 25
   best run 138.7 s against 1.97 s. The median program uses 0.2-0.3% of its 360 s limit.
   A run uses only 22-44% of its 10 h wall clock, so there is room to spend.
   **Decide how to say this without prescribing an algorithm**, and note in RATIONALE.md
   that per-program time trades against the number of evaluations a run can afford.
2. `eval_time` and `sum_radii` are named as metrics in the shipped prompts but the
   evaluators never report them. Only `combined_score`, `n_instances`, `n_failed`,
   `n_timeout` exist.
3. Libraries: numpy, scipy, pandas, networkx are installed. pywt/PyWavelets, PyEMD,
   sklearn, skimage, statsmodels, torch, numba are NOT; importing one scores zero. The
   shipped signal prompt suggests wavelets, EMD and neural networks anyway.
4. cloudcast: the evaluator calls `bc.set_num_partitions(k)` on the returned object, so a
   program that drops that method scores zero -- it happened in 190 and 183 of 500
   programs in two runs, every one of them zero.
5. signal_processing: output length must be `len(x) - window_size + 1`, `output[i]` may
   use input samples `0 .. i + window_size - 1` only, and correlation with the clean
   signal below 0.1 scores zero with no error message.
6. heilbronn: the score is the MINIMUM triangle area over all triples, so one collinear
   or coincident pair makes the whole arrangement worth zero, silently.
7. Containment and overlap checks carry a 1e-6 tolerance, not exact.

## Method

Read the evaluators first and treat them as the only authority; our prompt history is
evidence of what went wrong, not of what is true. Measure the failure modes yourself
from the stored programs and feedback rather than trusting the numbers above. Then write
prompts that state the contract exactly, name the failures that a program can avoid by
knowing about them, and stay silent about method.

Be honest in FAILURES.md: several failure modes are properties of the search, not of the
prompt, and saying so is more useful than claiming prompt coverage we do not have.
