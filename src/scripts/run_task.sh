#!/usr/bin/env bash
# One protocol-compliant run:  run_task.sh <task> <seed> [search] [extra skydiscover args...]
# Stops at ITERS evaluations or the task's wall-clock limit, whichever comes first.
# SkyDiscover checkpoints every 10 iterations, so a killed run still leaves its
# programs on disk for validation-set selection.
set -euo pipefail
TASK=$1; SEED=$2; SEARCH=${3:-evox}; shift $(( $# >= 3 ? 3 : $# ))
PROJ=${PROJ:-$(cd "$(dirname "$0")/.." && pwd)}       # src/
KIT=${KIT:-$(cd "$PROJ/.." && pwd)}                  # folder holding problems/<task> (the starter kit's layout)
SKY=${SKY:-$PROJ/skydiscover}                        # cloned SkyDiscover, uv-synced
API=${API:-http://localhost:8000/v1}
case $TASK in heilbronn_triangle|circle_packing_rect) DEF=10h ;; *) DEF=5h ;; esac
LIMIT=${LIMIT:-$DEF}                                 # override only for pilots, e.g. LIMIT=1h
# EvoX advances its counter by attempts_used, not by evaluations (see
# evox/controller.py: `iteration += attempts_used`), so a retried parse
# failure spends budget without ever reaching the evaluator. 500 therefore
# bounds evaluations from above -- a run can only undershoot the protocol
# cap, never exceed it.
ITERS=${ITERS:-500}
P=$KIT/problems/$TASK
OUT=${OUT:-$PROJ/runs}/${TASK}/${SEARCH}_seed${SEED}_$(date +%Y%m%d_%H%M%S)
mkdir -p "$OUT"
# Record where this run landed so the calling job can find it after the search
# (the directory name carries a timestamp the job does not know in advance).
echo "$OUT" > "${OUT%/*}/.last_run"
# Record which evaluator kit this run is scored by (hacked vs patched mode).
echo "kit=$KIT" > "$OUT/KIT.txt"
# Copy the task config and set only the search type and selection seed; the
# generator and its decoding parameters stay exactly as shipped.
TASK_NAME="$TASK" "$SKY/.venv/bin/python" - "$P/config.yaml" "$OUT/config.yaml" "$SEED" "$SEARCH" <<'PY'
import os, sys, yaml
src, dst, seed, search = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
c = yaml.safe_load(open(src))
c.setdefault("search", {})["type"] = search
c["search"].setdefault("database", {})["random_seed"] = seed
# BEST_OF_N set (with SEARCH=best_of_n): independent sampling. Every child is generated
# from the seed program with no other programs in context, as a no-search control.
if os.environ.get("BEST_OF_N"):
    c["search"]["database"]["best_of_n"] = int(os.environ["BEST_OF_N"])
    c["search"]["num_context_programs"] = 0
# EvoX derives its meta-evolution cadence from max_iterations when
# search.switch_interval is unset (evox/controller.py:165, 10% of the budget),
# so raising -i silently stops the search strategy ever evolving: 135 switches
# at -i 500 became 0 at -i 6000. Pin it so the cadence does not move with the
# budget. 50 is what the 10% rule gave at the protocol's 500.
c["search"]["switch_interval"] = int(os.environ.get("SWITCH_INTERVAL", "50"))
# PROMPTS_DIR overrides the shipped system message with prompts/<ver>/system_<task>.txt.
# Section 7 requires every arm to use the same system prompts, so this applies to the
# baseline arms exactly as it does to the method arm.
# Arm 3 replaces AdaEvolve's paradigm injection with its own strategy
# specifications. Both are natural-language strategy guidance, so running them
# together confounds the ablation; and an active paradigm makes _generate_child
# swap the sampled parent for the global best, which would attach a specification
# to one program and generate the diff against another.
if os.environ.get("STRATEGY_SPEC") == "1":
    c["search"].setdefault("database", {})["use_paradigm_breakthrough"] = False

pd = os.environ.get("PROMPTS_DIR")
if pd:
    sp = os.path.join(pd, "system_%s.txt" % os.environ["TASK_NAME"])
    with open(sp) as fh:
        c.setdefault("prompt", {})["system_message"] = fh.read()
    print("system prompt <- %s" % sp, file=sys.stderr)
yaml.safe_dump(c, open(dst, "w"), sort_keys=False)
PY
cd "$SKY"
# EVAL_CAP set: stop on evaluations instead of on LLM calls, and raise -i out of
# the way so it never binds first. Unset: the stock EvoX budget, unchanged.
if [ -n "${EVAL_CAP:-}" ]; then
  ENTRY=("$SKY/.venv/bin/python" "$PROJ/scripts/run_capped.py")
  ITERS=${ITERS_CAPPED:-6000}
  echo "evaluation-capped run: EVAL_CAP=$EVAL_CAP, -i $ITERS"
else
  ENTRY=("$SKY/.venv/bin/skydiscover")
fi
# Compute nodes have no internet, so call the synced venv directly, not `uv run`.
START=$(date +%s)
# LIMIT is exported so SPROUT can schedule against the same wall clock that stops it.
OPENAI_API_KEY=${OPENAI_API_KEY:-dummy} EVAL_CAP=${EVAL_CAP:-} LIMIT="$LIMIT" timeout --signal=INT "$LIMIT" \
  "${ENTRY[@]}" optimize "$P/initial_program.py" "$P/evaluator.py" \
    -c "$OUT/config.yaml" -i "$ITERS" -o "$OUT" --api-base "$API" "$@" 2>&1 | tee "$OUT/run.log" || true
echo "wall_clock_seconds=$(( $(date +%s) - START ))" | tee -a "$OUT/run.log"
