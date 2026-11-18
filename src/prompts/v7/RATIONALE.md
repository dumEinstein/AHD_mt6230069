# prompts/v5 -- rationale and defence

Written 2026-11-06 against the evaluators in `extracted/col8385_project/problems/`
(and the patched pair in `extracted/col8385_project_patched/`). Every factual sentence
in the five `system_<task>.txt` files was checked against evaluator source and, where a
check can be executed, is covered by a case in `scripts/verify_prompt_contracts.py`.
The failure frequencies quoted here come from a census of every stored program under
`runs_final/`, `runs_r2/`, `runs_v1/`, `runs_v3/`, `runs_v4/` (33,088 programs); the
census is for this document and `FAILURES.md` only. None of it appears in a prompt.

## 1. Rules the prompts obey

| rule | source | how v5 honours it |
|---|---|---|
| prompts may be task-specific; search code and config identical across tasks | handout §2.3 | all task knowledge is in `system_<task>.txt`; nothing else differs per task |
| no task-specific program code written or obtained outside the run; no known constructions | handout §2.3 | the prompts contain no code, no coordinates, no packings; the only constants are the evaluator's own normalisers (0.0365298..., 2.3658321...), which the shipped prompts and `data.py` already expose |
| no statistics measured in our own runs | project rule (INCONSISTENCIES.md §5) | zero run-derived numbers; every number in a prompt is a constant of the evaluator, the handout or the shipped data |
| search-algorithm agnostic | brief | no lineage, strategy, island, parent, iteration or "previous attempt" wording; the one mention of "the run" is the handout's wall-clock budget, which binds every arm identically |
| state the contract, never the method | brief, and the v1 signal collapse (89% -> 0.4% admissible after "handle the first and last window-lengths explicitly") | the prompts say what is checked, how the score is computed and what the resources are; they never say how to construct anything |
| every claim true | brief, and the v1-v4 "same length as the input" error (1/70 admissible) | 64 executable cases plus 9 re-derivation/environment checks in the verifier; see §6 |

## 2. What v5 changes for every task

**The time limit the model sees is wrong, and v5 corrects it.** SkyDiscover's default
context builder appends to every user message
`- Time limit: Programs should complete execution within {evaluator.timeout} seconds`
(`skydiscover/optimize/context_builder/default/builder.py:119-121`). `evaluator.timeout`
is the timeout of the *whole* `evaluate()` call, set in each task's shipped `config.yaml`:
360 for prism and signal_processing (true per-call limit 10 s), 600 for cloudcast (true
60 s), 400 for heilbronn and circle (true 360 s). The stored prompts confirm the line is
sent on every task. A program that believes it has 360 s on prism loses every instance it
runs past 10 s on; 52 stored prism programs (0.8%) were killed at the 12 s hard stop, on
up to 50 instances each. This is config, not prompt, and the config is the shipped one,
so the system prompt is the only place to state the true limit. v5 states the per-call
limit, the grace period, that import time is excluded from the measurement but not from
the kill, and adds one sentence saying that a larger figure elsewhere in the context is
the evaluation-wide timeout. That sentence is search-agnostic (it is true under every
SkyDiscover search type) and verified (`evaluator.py` TIME_LIMIT/GRACE, `evalkit._loop`).

**Environment.** numpy 2.x, scipy, pandas and networkx are installed; the thirteen
packages the prompts name as absent were each checked with `importlib` in the evaluation
venv. v4 listed libraries only on signal_processing; v5 lists them everywhere because
`pulp` (prism, cloudcast), `sklearn`, `matplotlib` and `cvxpy` (math tasks) also appear in
stored programs, each scoring zero.

**Metrics.** Every prompt says the evaluator reports exactly `combined_score`,
`n_instances`, `n_failed`, `n_timeout` plus one feedback line per instance, in the
format `evalkit.summarize` produces. The shipped prompts name `eval_time`,
`min_area_normalized` and `sum_radii`, none of which exists.

**Objective stated as the evaluator computes it.** cloudcast now carries the simulator's
cost model and signal_processing the per-signal formula; both are re-derived from scratch
in the verifier and match the evaluators to 1e-9 / 1e-12. The shipped cloudcast prompt
says nothing about how cost is computed; the shipped signal prompt lists four components
without the weights, the clipping, or the fact that lag and tracking error are measured
against the noisy input rather than the clean signal. A model cannot optimise an objective
it cannot see; stating the objective is not stating a method.

**Removed from the shipped text.** `eval_time ... EFFICIENCY - minimize` and `keep
reasonable, prefer accuracy over speed` (phantom metric; also wrong incentive, §3); `The
generated program should be as simple as possible` (unsupported by the evaluator, and
SkyDiscover's own template already pushes shortness with "Solution is long (>500 chars);
consider simplifying"); `Use the BroadCastTopology class and make_nx_graph function`
(the program's `make_nx_graph` reads `profiles/*.csv` relative to the program file and
raises `FileNotFoundError` where the evaluator runs it; 8 stored programs called it, none
admissible); the signal technique list `Kalman filters, particle filters, wavelets, EMD,
polynomial fitting, neural networks` (a method prescription, and two of the named
techniques need packages that are not installed: 91 programs imported `pywt`,
`sklearn`, `statsmodels` or `skimage` and scored zero).

**Kept.** The v4 admissibility preamble, shortened; the v4 contract bullets, with the
quoted error messages, which the census shows worked (signal IndexError 42.7% -> 0.2%,
cloudcast KeyError 15.1% -> 0.1%, Empty partitions 18.7% -> 0.6%, missing modules
6.4% -> 0% between the v1-3 and v4 cohorts).

## 3. The decision on time for the two mathematical tasks

### The problem

The evaluator gives each call 360 s and scores only the returned construction. The
shipped prompts told the model to minimise `eval_time`; v4 mentioned 360 s only as a
failure threshold; SkyDiscover's template adds "consider simplifying". The pooled census
(admissible programs, evaluator wall time from `run.log`, which includes ~0.1 s of
overhead):

| task | time bucket | n | mean score | best |
|---|---|---|---|---|
| heilbronn | < 1 s | 984 | 0.206 | 0.666 |
| | 1-10 s | 492 | 0.359 | 0.807 |
| | 10-60 s | 324 | 0.399 | 0.822 |
| | 60-180 s | 206 | 0.534 | 0.851 |
| | 180-360 s | 60 | 0.586 | 0.900 |
| circle | < 1 s | 1696 | 0.722 | 0.949 |
| | 1-10 s | 834 | 0.750 | 0.994 |
| | 10-60 s | 456 | 0.805 | 0.975 |
| | 60-180 s | 243 | 0.823 | 0.998 |
| | 180-360 s | 56 | 0.848 | 0.977 |

The median admissible program uses 1.1 s (heilbronn) and 0.95 s (circle), i.e. 0.3% of
its allowance. Nothing in any stored program paces by the clock (`time` is not imported
by a single heilbronn program), so this is a prior about "efficient code", not a
budgeting choice. Under v4 the heilbronn median rose from 0.02 s (v1-3) to 1.8 s and the
mean admissible score from 0.09 to 0.36 -- removing "minimise eval_time" already moved
both. v5 goes one step further.

### The trade-off that has to be stated honestly

Per-program runtime is not free. Handout §2.4 gives a run 10 h for at most 500
evaluations: 72 s per evaluation *including* generation. Our final runs used 2.2-2.6 h
(heilbronn) and 2.9-4.3 h (circle) of those 10 h, so there is room for roughly a 2-3x
rise in mean evaluation time before the wall clock, not the evaluation cap, ends the run.
But a run in which every program used 300 s would evaluate ~110 programs, not 500. And
the leaderboard re-runs the submitted program on other hardware: a program that works
until 350 s have elapsed will produce a different construction there, and one that works
until 359 s may time out.

### Options considered

1. *Say nothing beyond the limit (v4).* Rejected: the measured prior leaves 99.7% of the
   allowance unused and the score-time relation above is monotone.
2. *Prescribe "use most of the 360 s" or "run an optimiser until the time is up".*
   Rejected: this prescribes a method class (iterative refinement under a clock), which
   the brief forbids and which we have measured to collapse admissibility when done on
   signal; it also invites exactly the timeouts (1.5% of circle programs already die at
   370 s) and the hardware-dependence described above.
3. *State the facts and the arithmetic, name the envelope, prescribe nothing.* Chosen.

### What the prompt says (identical on both tasks)

- the limit (360 s on the call, import excluded, process killed at ~370 s) -- verified;
- runtime is not part of the score; a 300 s program and a 0.3 s program are scored only
  on what they return; finishing early gains nothing -- verified (no time metric exists);
- caution 1: the leaderboard re-runs the program on other hardware, so keep a quarter of
  the limit as margin and let the result depend on work done (loop counts, convergence tests), not elapsed time;
- caution 2: the run has 10 h for at most 500 evaluations (~70 s each including
  generation), so runtime is paid for in evaluations; "tens of seconds per call is always
  affordable, the full allowance only for a few candidates";
- the misleading-figure sentence from §2 (the user message says 400 s here).

"Tens of seconds is always affordable" is arithmetic, not a run statistic: 500 x 30 s of
evaluation plus 500 x ~25 s of generation is 7.6 h, inside the 10 h. It is an *envelope*,
not an algorithm: the prompt does not say what to do with the time, and a program that
returns a closed-form construction in 0.1 s is as welcome as before. Caution 1 is a
reproducibility requirement that follows from how submissions are scored (handout §5),
not a design hint.

### Pushback I expect

*"You are telling the model to be slow."* No: I am telling it that slow is not penalised
and what the actual budget is, after four prompt versions and a template that told it the
opposite. The one directive is a ceiling, not a floor.

*"A 70-second average is search-specific."* It is handout arithmetic that holds for every
arm (10 h / 500). Arms that evaluate in parallel can afford more; none can afford less.
I chose the conservative figure.

*"Why not let the search discover that time helps?"* It had ~15,000 evaluations across
our runs to do so and did not: the search samples from the model's prior, and the prior
is "fast". Stating a resource is contract, not method.

## 4. Per task

### prism

*Enforced* (`prism/evaluator.py`): return a dict; every value a list; the lists contain
exactly the input `Model` objects (identity check by `id()`); per-GPU `model_size` sum
<= 80; 10 s per call (+2 s grace); score 1 / max KVPR with a full GPU at 1e6; mean over
instances. Called with keyword arguments `gpu_num=`, `models=`.

*Targeted failures* (census, 6,683 programs, 98.0% admissible): `TypeError: unhashable
type: 'Model'` (52 programs) and `'<' not supported between instances of 'Model'` (6) --
the prompt now says Model is an unhashable, unordered dataclass; keyword-argument
renames (3); timeouts at the 12 s kill (52 programs, 1,847 instances) -- addressed by the
true time limit and the misleading-figure sentence; copies of Model objects (4); memory
violations (15); exceptions raised on "impossible" inputs (the initial program's own
`raise ValueError("Unable to place ...")` appears in the census) -- the preamble says
never to raise.

*Deliberately not said.* The evaluator does not check that dict keys lie in
`0 .. gpu_num-1`, nor that the number of GPUs used is `gpu_num` (`prism/evaluator.py`
lines 235-243 look only at `placement.values()`). A program could place every model
on its own imaginary GPU and score ~1/(r/s) per model. The prompt states the task's
contract (ids 0 .. gpu_num-1) and stays silent about the gap; exploiting it would be a
leaderboard submission we could not defend. Also not said: anything about which models
to pair, sorting orders, or local search -- the initial program already is a greedy
placement and the search's job is to improve on it.

*Prompting cannot fix:* ZeroDivisionError and generic exceptions from new code (0.5%),
and the 1.3% of programs that fail on a handful of instances while scoring on the rest
(bugs on edge cases of the generated code); these are generation-quality failures, not
misunderstandings of the contract.

### cloudcast

*Enforced* (`cloudcast/evaluator.py`, `sim/validate.py`, `sim/simulator.py`):
`bc.set_num_partitions(k)` is called on the returned object before anything else; `.src`
equal, `.dsts` set-equal; every (destination, str(partition)) key present with a non-empty
list of hops; each hop `[u, v, data]` with `len >= 3`, edge in G, contiguous, ending at the
destination; the topology is `json.dump`ed; the simulator reads `cost` and `throughput`
from the hop data; 60 s per configuration (+30 s grace, and the simulator runs inside the
same process after the call). The cost model in the prompt is step-for-step the
simulator's (`__construct_g`, `__transfer_time`, `__total_cost`) and is reproduced by
`check_cloudcast_cost_model` on all five training configurations to a relative 1e-9.

*Targeted failures* (8,633 programs, 88.2% admissible; 91.4% under v4): missing
`set_num_partitions` (369 programs; 365 of them in the two `runs_r2` seed-0 runs where
full rewrites dropped the method) -- one sentence in THE OUTPUT; integer partition keys
(a share of the 39 "Missing partitions"); "path does not reach destination" (138 of 210
Invalid-paths programs) -- the contract now says each partition holds *one* contiguous
path ending at the destination; `edge format invalid` (30) -- hops have three elements;
`make_nx_graph` from the program (8, all zero) -- stated; `NetworkXNotImplemented: not
implemented for directed type` (27) -- the prompt says G is a DiGraph; `pulp` (6).

*Deliberately not said.* (a) The shipped simulator prices each hop from the data the
*program* attaches, so a program can route over real edges while declaring them free.
23 of 6,633 hacked-mode programs score above 0.012 (the patched-mode maximum is 0.0085;
the top exploit scores 0.0566). The prompt states the contract -- the third element is
`G[u][v]` and the simulator reads cost and throughput from it -- which is true in both
modes, and does not mention that the stock evaluator fails to check it. (b) The graph is
complete (71 regions, 4,970 directed edges, 71 x 70), so "edge not in G" can only come
from a wrong node name; stating completeness would invite dropping the check and adds
nothing to admissibility. (c) Shipped configurations all use 10 partitions, 300 GB and
6-7 destinations; the hidden set may not, so the prompt says these come from the
configuration and nothing more. (d) Nothing about multicast trees, relay regions,
splitting partitions across paths or provider costs -- the cost model lets the model
derive these; naming them would be method.

*Prompting cannot fix:* the 100 programs (1.2%) killed at 90 s. Their code does not
cluster on any pattern (39 call `nx.all_simple_paths`, but 90 programs that call it
finish in time), so this is the generated search being too slow on the complete graph,
which the true limit statement can only partly deter. The exploit itself is an evaluator
property: only the patched evaluator removes it.

### signal_processing

*Enforced* (`signal_processing/evaluator.py`, `metrics.py`): `run_signal_processing
(noisy_signal=x, window_size=20)` by keyword; dict with `filtered_signal`; finite,
non-empty; `m = min(len(y), len(clean) - 19) >= 2`; `calculate_lag_error` indexes
`noisy[19 + len(y) - 1]`, so an output longer than `n - 19` raises IndexError; corr < 0.1
or constant output -> 0 silently; the formula in the prompt is `score_signal` plus
`calculate_composite_score`, reproduced by `check_signal_score_formula` to 1e-12.

*Targeted failures* (8,196 programs, 87.7%; 96.7% under v4): IndexError from
over-long output (499 programs, 42.7% of the v1-3 cohort, 0.2% of v4 -- the v4 fix
worked and is kept verbatim in substance); `module 'program' has no attribute
'run_signal_processing'` (51 programs that rewrote the file without the wrapper) -- the
prompt now says the program *must define* that function with those names; broadcasting
and "setting an array element with a sequence" ValueErrors (96) -- the output must be a
one-dimensional sequence; `pywt`/`sklearn`/`statsmodels` imports (91, all v1-3); look-
ahead: 125 of 1,500 patched-mode programs (8.3%) were rejected as non-causal, and 329
hacked-mode programs carry `mode='same'`, `savgol`, `filtfilt` or `medfilt`.

*On causality.* The stock evaluator does not check it; the patched evaluator re-runs the
program on 45/70/90% prefixes. The prompt must be true under both, so it defines causal
behaviour operationally ("run on any prefix of x the program must reproduce the leading
samples of its output on the whole of x") and says that a look-ahead filter "is not a
solution to [real-time filtering] even where a scorer does not detect the look-ahead".
That is honest about the stock evaluator without naming it, correct for the patched one,
and consistent with the task as the initial program and the shipped prompt define it
("Real-Time", "real-time adaptive filtering"). Both directions are verified: the stock
kit accepts a centred window aligned to the convention (score > 0) and the patched kit
rejects the same program with "non-causal filter".

*Deliberately not said.* (a) Outputs shorter than `n - 19` are accepted and scored on
the samples returned, with `S`, `R` and the alignment shrinking accordingly; a 2-sample
output would trivially score ~0.9. The prompt states the output length as the contract
(exactly N) and says only that *longer* fails. (b) Nothing about moving averages,
exponential smoothing, Kalman gains or window handling -- the v1 line "handle the first
and last window-lengths explicitly" is the one documented case of a method hint
destroying admissibility. (c) The weights in the formula make `S` appear twice (in `J`
and in `0.2/(1+S/20)`); the prompt states the formula and leaves the inference to the
model.

*Prompting cannot fix:* 0.6% generic exceptions and the 37 partial failures; a 0.2%
silent-zero rate (constant or anti-correlated outputs) that already carries a prompt line.

### heilbronn_triangle

*Enforced* (`heilbronn_triangle/evaluator.py`): `np.asarray(points, float).shape ==
(11, 2)`; each point satisfies `y >= -1e-6`, `sqrt(3) x <= sqrt(3) - y + 1e-6`,
`y <= sqrt(3) x + 1e-6`; score = min triangle area over `itertools.combinations(points,
3)` / (sqrt(3)/4) / 0.03653; 360 s (+10 s); `random.seed(0)`, `np.random.seed(0)` before
the call.

*Targeted failures* (4,631 programs, 45.8% admissible; 43.9% under v4):

1. *Silent zero, 36.4% of all programs and 38.3% of the v4 cohort* -- the dominant
   failure of the whole project, and v4's sentence ("if any three points are collinear
   or any two coincide ... zero") did not reduce it. Reading 40 silent-zero v4 programs:
   34 are full rewrites that hand-place a symmetric configuration -- the three vertices
   plus points on the edges, or points in horizontal rows -- and 6 are small edits; only
   35 of 1,691 fall back to `np.zeros`. The model knows the rule and does not connect it
   to its own constructions. v5 therefore spells out where collinear triples come from in
   terms the model uses: three points on one edge (two vertices plus a point between
   them count), a vertex with the centroid and the opposite midpoint, three points on a
   row, column or diagonal of a grid. These are facts about the objective, verified as
   exact silent zeros in the verifier. Whether naming the patterns changes the rate is
   the open question; see FAILURES.md for the verdict "partly".
2. *Outside the triangle, 10.1%.* Of 403 rejected points, 386 lie inside the unit square
   but outside the triangle (e.g. (0.1, 0.4)), median violation 0.16 -- not precision,
   wrong region. v4 gave only the vertices; v5 gives the three inequalities the
   evaluator tests and one concrete outside point.
3. `UnboundLocalError ... 'v1'` (128 programs, 2.8% of v4) and `AssertionError` from the
   program's own asserts (90): generation bugs, see below.

*Deliberately not said.* No coordinates, no lattice, no symmetry group, no "perturb and
polish", no "use scipy.optimize" (239 programs import scipy already). No comparison with
the reference value beyond the evaluator's own normaliser.

*Prompting cannot fix:* the UnboundLocal/Assertion/NameError family (~5%) -- generated
code that references a variable before assignment or asserts its own wrong invariant;
the 0.7% that time out without pacing by the clock (fixed iteration counts chosen too
large) -- the time section can only make the limit salient; and most of the silent-zero
mass, which is a reasoning failure about geometry, not a contract misunderstanding. The
only mechanism that reliably removes silent zeros is one the prompt may not prescribe:
checking all 165 areas before returning.

### circle_packing_rect

*Enforced* (`circle_packing_rect/evaluator.py`): shape (21, 3); no NaN, no negative
radius (zero allowed); for every pair `hypot(dx, dy) >= r_i + r_j - 1e-6`; bounding box
`w + h <= 2 + 1e-6`; score = sum r / 2.3658321334167627; 360 s (+10 s); seeds as above.

*Targeted failures* (4,945 programs, 77.0%; 78.6% under v4): overlap 15.8% (v1-3 21.8%,
v4 15.6%). 609 of the 783 overlapping programs had an admissible parent, 54 of 60
sampled are full rewrites, and all 60 mention "overlap", contain the literal `1e-6` and
have a pairwise `i < j` loop: the model knows the rule, uses the evaluator's tolerance as
its own, and still returns an infeasible packing. v5 adds the one true sentence that
addresses this: the evaluator recomputes the distances itself, so a packing tight to
within 1e-6 by the program's arithmetic can fail by rounding in the evaluator's.
Perimeter 1.2%, median excess 0.22 (gross, 9 of 63 are precision); the bounding-box
definition is restated. Timeouts 1.5% (72 programs, all killed at 370 s; none paces by
the clock) -- the time section. NameError 1.0%, wrong shape 0.3%.

*Deliberately not said.* No aspect ratio, no hexagonal hint, no "equal radii", no
"optimise with SLSQP", no repair recipe ("shrink radii until feasible" was in v1-v3 and
is a method). Not said that a radius of exactly zero is a way to satisfy constraints
cheaply -- the prompt says zero is accepted and contributes nothing, which is the fact.

*Prompting cannot fix:* most overlap failures (they occur in programs that already state
the rule), timeouts from fixed oversized iteration counts, NameErrors. These need either
a repair step, a pre-evaluation feasibility check, or a search that keeps admissible
parents -- all search-side.

## 5. What was considered and rejected for all tasks

- *Quoting the initial program's score or "the best we have seen".* Run statistics; out.
- *Telling the model how many instances there are* (50 train, 5 configs, 5 signals).
  Shipped data, hence allowed, but it invites per-instance tuning and the test set is
  hidden; the prompts say only what varies.
- *"Return the initial program's answer on failure".* Reasonable engineering, but it is
  a method, and on the math tasks the initial program scores zero anyway.
- *A per-task list of "do not" phrases* (do not use grids, do not use Kalman filters).
  Negative method prescriptions are still method prescriptions.
- *Mentioning EVOLVE-BLOCK mechanics.* Search-specific (SPROUT splices the block, EvoX
  diffs the whole file); only the signal prompt mentions the block, as a fact about where
  `run_signal_processing` lives in the initial program.

## 6. Verification

`scripts/verify_prompt_contracts.py` keeps the 21 v4 cases byte-for-byte and adds 43
executable cases plus 9 re-derivation/environment checks (73 in total; 2 of the 43 are
the 360 s timeout cases, run only with `VERIFY_SLOW=1` because they cost ~13 minutes).
New executable cases, by claim:

- prism: copies rejected; keyword-argument call; exception message surfaces; `Model`
  unhashable; unavailable module; 10.5 s call -> "program took"; 8 s import -> accepted
  (import time not measured); 13 s import -> "no result within 10s" (but the kill).
- cloudcast: missing `set_num_partitions`; integer partition keys -> "Missing
  partitions"; destinations changed; path stops short; two-element hop; discontinuous
  path; `float32` hop data -> not JSON-serialisable; the program's `make_nx_graph` ->
  `FileNotFoundError`; exception message; module; and the full cost model reproduced.
- signal: output one too long; `inf`; one-sample output -> "too short"; non-dict result;
  wrapper deleted -> AttributeError; keyword names changed; `pywt`; aligned centred window
  accepted by the stock scorer and rejected by the patched one; a causal filter passes
  the patched scorer; the score formula reproduced; data lengths and types.
- heilbronn: point on the base edge accepted; (0.1, 0.4) outside; NaN outside; exactly
  collinear triple -> silent zero; two vertices plus a point between -> silent zero;
  exception message; RNG seeded with 0; list-of-lists accepted; 361 s -> timeout (slow).
- circle: zero radius accepted; NaN rejected; exception message; RNG seeded; `sklearn`;
  375 s import -> killed at the 370 s hard stop (slow).
- all: metric keys are exactly the four named, with the feedback labels; prism score
  formula with full/empty GPU; environment (py3.11, numpy 2.x, networkx 3.x, 13 named
  packages absent); seed-0 reference values.

Pass count: see the end of this file (filled in from `jobs/logs/43_verify_prompts*.out`).

## 7. Open risks

- The time section is the one place v5 goes beyond v4 in *telling* rather than
  *describing*. If a v5 run shows a timeout rate on the math tasks well above the current
  0.7-1.5%, the "quarter of the limit" margin is the sentence to strengthen, and the
  "tens of seconds" sentence the one to drop. Measure before changing either.
- The cloudcast cost model is long. If prompt length is a constraint for an arm (the
  signal prompt is ~4.7k characters, cloudcast ~4.8k), the cost-model paragraph is
  objective, not contract, and is the first candidate to compress; everything under WHAT
  THE EVALUATOR CHECKS should stay.
- The heilbronn collinearity patterns are facts, but they are also the first time a
  prompt names *configurations*. They name what is zero, not what is good; I judge that
  on the right side of the line. If a reviewer disagrees, the sentence can be reduced to
  the edge-and-grid clause without losing the verified content.

## Verifier result

Run 2026-11-06 on a debug-queue node (`jobs/43_verify_prompts.pbs`, submitted without a
GPU request because every GPU was allocated; logs in `jobs/logs/43_verify_prompts_nogpu*.out`).

- Fast suite (`VERIFY_SLOW` unset): **71 PASS, 0 FAIL, 2 SKIPPED** (the two 360 s cases).
  The 21 v4 cases all pass unchanged. Both re-derivations match the evaluators exactly
  (cloudcast cost model max relative difference 0.0 over the five training
  configurations; signal score formula max absolute difference 0.0 over the five
  training signals).
- Slow suite (`VERIFY_SLOW=1`): first run 72 PASS, 1 FAIL -- the failing case was a
  test-design error, not a prompt error: a 365 s import followed by a fast call finishes
  before the 370 s kill, and the evaluator correctly accepted it (program_time is measured
  on the call only, exactly as the prompt says). The case now sleeps 375 s. Second run:
  **73 PASS, 0 FAIL, 0 SKIPPED -- ALL PROMPT CLAIMS CONFIRMED.**
