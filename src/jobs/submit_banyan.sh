#!/usr/bin/env bash
# BANYAN full runs (design/round7/BANYAN.md): 2 seeds x 5 tasks into runs_final/banyan.   NOT submitted yet.
#   bash jobs/submit_banyan.sh                 (from the project root; SEEDS="0 1 2" for a third seed)
# Flags: BANYAN=1 selects the arm (search type banyan); the diagnoser is on, lessons off, deadline min(0.6*L, 0.5*T/cap/waves) (defaults).
# Optional BANYAN knobs (env, all task-independent): BANYAN_DIAG=0|1, BANYAN_LESSONS=0|1, BANYAN_CLOCK_FRAC=0.6, BANYAN_BUDGET_FRAC=0.5,
# BANYAN_EVALSRC=0|1, BANYAN_RERUN_PARALLEL=4.
set -euo pipefail
cd "$(dirname "$0")/.."
wt() { case $1 in heilbronn_triangle|circle_packing_rect) echo 12:00:00 ;; *) echo 07:30:00 ;; esac; }
for SEED in ${SEEDS:-0 1}; do
  for T in prism cloudcast signal_processing heilbronn_triangle circle_packing_rect; do
    qsub -P neuro.symbolic.spons -N "BN${SEED}_${T:0:6}" -l walltime=$(wt "$T") \
      -o "jobs/logs/96_banyan_${T}_seed${SEED}.out" \
      -v "TASK=$T,SEED=$SEED,SEARCH=banyan,BANYAN=1,EVAL_CAP=500,MODE=final,PROMPTS_DIR=$PWD/prompts/v5,RUNS_ROOT=$PWD/runs_final/banyan" \
      jobs/20_run.pbs
  done
done
