# Failure catalogue (for prompts/v5)

Source: every stored program under `runs_final/`, `runs_r2/`, `runs_v1/`, `runs_v3/`,
`runs_v4/` (latest checkpoint of each run; 33,088 programs, 91 runs), classified from the
evaluator feedback stored with each program, plus evaluation wall time from `run.log`.
Two caveats. (1) SPROUT stores every evaluated program; EvoX and AdaEvolve store a pruned
archive (40-480 of ~500), so pooled rates over-weight SPROUT runs and under-count the
baselines' worst programs. (2) "v1-3" and "v4" denote the prompt version the program was
generated under (detected from the stored system message); v4 cohorts are much larger.
Percentages are of programs unless stated. Frequencies are *observed under v1-v4*; the
verdict says what v5 does about each.

Verdicts: **P** = prompting can remove it (the program only needs to know the rule);
**p** = partly (prompting reduces it, the residue is generation/search quality);
**S** = search-side (not a prompt problem; says what would be needed).

## Cross-task

| failure | how often | verdict | notes |
|---|---|---|---|
| Generation produces no evaluable program (unparsable or non-applying SEARCH/REPLACE diff, no fenced code, syntax error) | EvoX heilbronn: 1,022 LLM iterations for 489 evaluations, i.e. ~52% of attempts; SyntaxError/IndentationError still reaches the evaluator in 0.2-0.6% of stored programs per task | **S** | A system-prompt line about diff format would be search-specific. SPROUT's strict-diff/AST pre-filter is the right place; EvoX/AdaEvolve pay for it in budget. |
| Program drops a name the evaluator calls (`run_signal_processing`, `set_num_partitions`, `BroadCastTopology`) | signal 51 programs; cloudcast 369 (365 in two `runs_r2` runs); cloudcast 50 lack the class | **p** | v5 states every called name in every prompt. The residue is full rewrites that ignore the interface; SPROUT's required-name check removes it on that arm. |
| Misleading time limit in the user message (`Time limit: ... 360/600/400 seconds`, from `evaluator.timeout`) | prism 52 programs killed at 12 s (1,847 instances); cloudcast 100 killed at 90 s; math 34 + 72 killed at 370 s | **p** | v5 states the true per-call limit and that the larger figure is the evaluation-wide timeout. Cannot stop a program whose fixed iteration count is simply too large on the evaluator's hardware. |
| Import of an uninstalled package (`pywt`, `sklearn`, `statsmodels`, `skimage`, `pulp`, `matplotlib`, `cvxpy`) | signal 91 (all v1-3; 0 under v4); prism 1; cloudcast 6; heilbronn 5; circle 5 | **P** | Listed on every task in v5 (v4 listed on signal only). Signal shows the line works. |
| Generic runtime exceptions in new code (NameError, UnboundLocalError, ZeroDivisionError, broadcasting ValueError, TypeError on new helpers) | 1-5% per task; heilbronn `UnboundLocalError 'v1'` alone 128 programs | **S** | Code quality of a 30B generator at temperature 0.99. Needs repair calls or a syntax/semantic pre-check, not prompt text. |
| Program raises on an input it finds hard (its own `raise`/`assert`) | prism "Unable to place" family 5; heilbronn AssertionError 90; circle 21 | **p** | The preamble says never to raise. The asserts are usually wrong invariants (e.g. asserting the point is inside when it is not), which is a bug, not a policy. |
| Non-reproducible result (unseeded RNG, clock pacing) | 0 programs pace by the clock; unseeded `default_rng()` not measured | **P** | v5 states the seeding and the re-run. Low observed frequency; included because the leaderboard re-runs the program. |

## prism (6,683 programs, 17 runs; 98.0% admissible; v4 cohort 98.2%)

| failure | how often | verdict | notes |
|---|---|---|---|
| `TypeError: unhashable type: 'Model'` (Model used as set member / dict key) | 52 programs (0.8%) | **P** | v5: "Model instances are not hashable ... and do not support `<`". |
| `'<' not supported between instances of 'Model'` | 6 | **P** | same line |
| Timeout: killed at the 12 s hard stop | 52 programs, 1,847 instances | **p** | true limit now stated; user message says 360 s |
| `ZeroDivisionError` | 32 programs | **S** | division by free memory 0 or by an empty group; the prompt states the 80 GB = 1e6 rule, the bug is in new code |
| `GPU memory constraint violated` | 15 programs, 529 instances | **p** | stated; residue is bugs in capacity bookkeeping |
| Copies / re-created Model objects | 4 programs | **P** | stated explicitly ("a copy or re-created Model ... fail") |
| Keyword-argument rename (`gpu_num`, `models`) | 3 | **P** | stated |
| `list.remove(x): x not in list`, `max() arg is an empty sequence` | 11 | **S** | code bugs |
| Partial failure (score > 0, some instances failed) | 90 programs (1.3%) | **S** | edge-case bugs; note the mean-over-instances makes these cheap but non-zero |
| Not a failure but a loophole: GPU ids are not checked against `gpu_num` | 0 observed | -- | deliberately not mentioned (RATIONALE §4) |

Prism is not a prompt problem: 98% admissible, and the score spread (median 22.7, best
26.1) is search quality.

## cloudcast (8,633 programs, 22 runs; 88.2% admissible; v1-3 38.6%, v4 91.4%)

| failure | how often | verdict | notes |
|---|---|---|---|
| Returned object lacks `set_num_partitions` | 369 programs (4.3%); 365 in `runs_r2/*/arm4` seed 0 | **p** | v5 states the call. The concentration in two runs says it was a full-rewrite habit of one arm at one time; SPROUT's required-name check is the structural fix. |
| `Invalid paths` -- "path does not reach destination" | 138 programs | **P** | v5: each partition holds one contiguous path ending at the destination |
| `Invalid paths` -- "edge format invalid" (2-element hops) | 30 | **P** | stated |
| `Invalid paths` -- "path discontinuity" | 13 | **P** | stated |
| `Empty partitions` | 147 (1.7%; v1-3 18.7%, v4 0.6%) | **P** | v4 line kept |
| `Missing partitions` (incl. integer keys) | 39 | **P** | v5: keys are strings, the class converts |
| `KeyError` (edge/node lookups, `paths[dst][partition]`) | 84 (v1-3 15.1%, v4 0.1%) | **P** | fixed by the v4 contract statement; kept |
| `NetworkXNotImplemented: not implemented for directed type` | 27 | **P** | v5 says G is a DiGraph |
| Program calls its own `make_nx_graph` | 8 (0 admissible) | **P** | v5 says why it raises |
| Timeout: killed at 90 s | 100 (1.2%) | **S** | No code pattern separates them (39 use `all_simple_paths`, but 90 users of it finish). Generated search too slow on a complete 71-node graph. |
| `unhashable type: 'list'`, `'float' object is not subscriptable`, dijkstra kwargs | 68 | **S** | code bugs |
| `pulp` import | 6 | **P** | listed |
| Exploit: fake `cost`/`throughput` in hop data (stock evaluator prices from it) | 23 hacked programs > 0.012; top 0.0566 vs patched max 0.0085 | **S** | Prompt states the contract (`G[u][v]`), which is all it can do without describing the hole. Only the patched evaluator removes it; the leaderboard uses the stock one. |

## signal_processing (8,196 programs, 21 runs; 87.7%; v1-3 33.2%, v4 96.7%; patched mode 91.7%)

| failure | how often | verdict | notes |
|---|---|---|---|
| `IndexError: index N is out of bounds` (output longer than n-19; mostly same-length outputs) | 499 programs (6.1%); v1-3 42.7%, v4 0.2% | **P** | the v4 correction (len = n - W + 1) removed it; kept |
| Non-causal filter (patched evaluator only) | 125 of 1,500 patched programs (8.3%); 329 hacked programs carry `mode='same'`/`savgol`/`filtfilt`/`medfilt` | **p** | v5 defines causality operationally and says look-ahead is not a solution even where undetected. Under the stock evaluator it is rewarded, so the search will keep finding it. |
| Uninstalled package import | 91 (v1-3 6.4%, v4 0%) | **P** | fixed by v4, kept |
| Broadcasting / "setting an array element with a sequence" / "truth value ambiguous" | 96 (1.2%) | **S** | code bugs in new filters |
| `module 'program' has no attribute 'run_signal_processing'` | 51 | **P** | v5: the program must define that function with those names |
| `AttributeError`/`UnboundLocalError`/`NameError` | ~120 | **S** | |
| Empty/NaN output | 9 | **P** | stated |
| Not a dict / missing key | 2 | **P** | stated |
| Silent zero (constant or corr < 0.1) | 16 (0.2%) | **p** | stated; residue is filters that genuinely destroy the signal |
| Loophole: outputs shorter than N are scored on the prefix; a 2-sample output scores ~0.9 | 0 observed | -- | deliberately not mentioned |

## heilbronn_triangle (4,631 programs, 14 runs; 45.8%; v1-3 56.2%, v4 43.9%)

| failure | how often | verdict | notes |
|---|---|---|---|
| **Silent zero: a collinear or coincident triple** | 1,686 programs (36.4%); v1-3 25.6%, v4 38.3%; 25-45% in every SPROUT run, 31% in EvoX | **p** | The single largest loss in the project. v4's statement of the rule did not reduce it. Sampled v4 cases are hand-placed symmetric configurations (vertices + edge points + rows), 34/40 full rewrites, not `np.zeros` fallbacks (35/1,691). v5 names the patterns that are collinear (edge triples, vertex-centroid-midpoint, grid rows). Expect a reduction, not removal: the residue is geometric reasoning. The reliable fix -- compute all 165 areas before returning -- is a method the prompt may not prescribe; a search-side pre-evaluation check (cheap: 165 cross products) would remove it entirely. |
| Point outside the triangle | 469 (10.1%); 386 of 403 parsed points lie inside the unit square but outside the triangle; median violation 0.16, only 2 within 1e-4 | **p** | v5 gives the three inequalities and a concrete outside point. The residue is programs that optimise in the square and never project. |
| `UnboundLocalError` (`'v1'` etc.) | 135 (2.9%) | **S** | code bugs |
| `AssertionError` (program's own) | 90 (1.9%) | **S** | wrong self-asserted invariants |
| Timeout at 370 s | 34 (0.7%); none pace by the clock | **p** | v5 time section; residue is oversized fixed iteration counts |
| `expected shape (11, 2)` | 12 | **P** | stated |
| NameError/TypeError/ZeroDivision etc. | ~50 | **S** | |
| Not a failure: time left unused | median admissible program uses 1.1 s of 360 s; score rises monotonically with time (0.21 below 1 s, 0.59 at 180-360 s) | **p** | v5 states the budget and the trade-off (RATIONALE §3). Whether a prior about "fast code" moves on a statement of fact is the experiment. |

## circle_packing_rect (4,945 programs, 17 runs; 77.0%; v1-3 63.9%, v4 78.6%)

| failure | how often | verdict | notes |
|---|---|---|---|
| **Overlap** (`circles i and j overlap`) | 783 (15.8%); v1-3 21.8%, v4 15.6%; 10-27% per run; 609/783 had an admissible parent | **p** | All 60 sampled v4 failures mention overlap, contain `1e-6` and have a pairwise loop; 54/60 are full rewrites. The model knows the rule and still returns an infeasible packing -- the failures are in the implementation of its own check (tolerance equal to the evaluator's, check before a final move, wrong index). v5 adds the one true, non-method sentence (the evaluator recomputes distances; tight-to-1e-6 fails on rounding). Beyond that: search-side feasibility check or repair. |
| Perimeter (`w + h > 2`) | 61 (1.2%); median excess 0.22, 9/63 within 1e-4 | **p** | gross errors (a 2x2 or 2x1 box) are a wrong container concept; v5 restates the bounding-box definition |
| Timeout at 370 s | 72 (1.5%); none pace by the clock | **p** | as heilbronn |
| `NameError` | 48 (1.0%; v1-3 6.2%, v4 0.4%) | **S** | |
| `expected shape (21, 3)` | 14 | **P** | stated |
| NaN / negative radius | 4 | **P** | stated |
| Silent zero (all radii zero) | 38 (0.8%); 25 of them under the shipped prompt | **p** | the initial program's `np.zeros` kept as a fallback; v5 says zero radius "contributes nothing" |
| Not a failure: time left unused | median admissible program uses 0.95 s; mean score 0.72 below 1 s, 0.85 at 180-360 s; the 25 best programs have median runtime 52.7 s | **p** | RATIONALE §3 |

## What prompting cannot fix, in one list

1. Unparsable or non-applying diffs and syntax errors (~half of EvoX attempts).
2. Runtime exceptions in freshly generated code (1-5% per task).
3. Programs whose fixed iteration counts exceed the limit on the evaluator's hardware
   (0.7-1.5% on the math tasks, 1.2% cloudcast).
4. Most circle overlaps: the rule is known and restated by the failing programs
   themselves; the error is in executing it.
5. Most heilbronn silent zeros: a geometric reasoning failure about the model's own
   construction; the prompt can name the patterns, not check the 165 areas.
6. The cloudcast pricing exploit and the signal look-ahead reward under the stock
   evaluators: evaluator properties; the prompt can only state the contract.
7. Variance between seeds: single-seed admissibility differs by up to 20 points between
   runs of the same arm and prompt (heilbronn 37.6-51.8%, circle 72-85%); no prompt
   change under ~10 points is measurable at n = 1.

Where a verdict says **p**, the honest expectation for v5 is a reduction in the named
failure of a few points, measurable only with three seeds per task.
