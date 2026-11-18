GitHub repository: https://github.com/dumEinstein/AHD_mt6230069

# BANYAN: incumbent-rooted credibility trees for LLM heuristic discovery (COL8385)

Team: Abdul Ahmed (`mt6230069`), Reeshabh Kotecha (`cs1230018`), Vihaan Luhariwala (`cs1230151`).

| Path | Contents |
|---|---|
| `report.pdf` | Final report (NeurIPS format) |
| `leaderboard3/<task>/mt6230069_heuristic.py` | Submitted program per task: the best-on-validation program of our BANYAN runs, unedited |
| `src/` | Source code (below) |
| `problems/` | Not included. Put the course starter kit's `problems/` folder here (`problems/<task>/`, `problems/evalkit.py`) |

## Submitted programs

| Task | Run (report prompt version, seed) | Train | Val |
|---|---|---|---|
| prism | v1, seed 0 | 26.180 | 25.705 |
| cloudcast | v3, seed 1 | 0.009264 | 0.012399 |
| signal_processing | v2, seed 0 | 0.7598 | 0.7546 |
| heilbronn_triangle | v2, seed 0 | 0.9018 | 0.9018 |
| circle_packing_rect | v1, seed 0 | 0.9916 | 0.9916 |

## Source layout (`src/`)

- `run_banyan.sh`: one full BANYAN run on a GPU machine: serve the generator, search, select on validation.
- `scripts/banyan.py`, `scripts/banyan_core.py`: the BANYAN controller and scheduler (credibility, two-stage Thompson draw, widening, re-rooting).
- `scripts/banyan_diag.py`, `scripts/banyan_replay.py`, `scripts/diagnoser.py`, `scripts/diag_runner.py`: the verified diagnoser.
- `scripts/credtree.py`, `scripts/credtree_core.py`: the CredTree base classes that BANYAN subclasses.
- `scripts/run_capped.py`: SkyDiscover entry point that caps evaluations at 500 (free LLM calls are not counted) and registers the arms.
- `scripts/run_task.sh`: one protocol run (task config, system prompt, wall clock 5 h or 10 h).
- `scripts/select_by_val.py`: re-scores the top programs by train score on validation and writes `mt6230069_heuristic.py`.
- `scripts/serve_qwen.sh`: vLLM server for the fixed generator.
- `scripts/sprout.py`, `scripts/strategy_spec.py`: earlier arms, imported only when their flags are set.
- `prompts/v5`, `prompts/v7`, `prompts/v8`: system prompts, the report's prompt versions v1, v2 and v3. `prompts/banyan`, `prompts/credtree`: BANYAN's own prompts.
- `jobs/`: the PBS scripts we used on the IIT Delhi Pragya cluster (`20_run.pbs` runs one job, `submit_banyan.sh` submits all tasks, `22_resume.pbs` resumes after an outage, `40_select.pbs` redoes selection).
- `analysis/`: scripts that regenerate the report's tables and figures from the run directories.

## Environment setup

Requirements: Linux, one 80 GB GPU (A100/H100) for the BF16 generator, Python 3.11, [uv](https://docs.astral.sh/uv/).

```bash
cd AHD_mt6230069/src

# 1. SkyDiscover, the search framework (the commit we ran)
git clone https://github.com/skydiscover-ai/skydiscover skydiscover
git -C skydiscover checkout 0d932b690670a7e544388ad362e9876c8bd256a0
(cd skydiscover && uv sync)
skydiscover/.venv/bin/python -m pip install numpy scipy pyyaml   # if not already present

# 2. vLLM for the generator, in its own venv (we used vllm 0.32.0, torch 2.13)
python3.11 -m venv .venv-vllm
.venv-vllm/bin/pip install vllm==0.32.0

# 3. Generator weights
huggingface-cli download Qwen/Qwen3-Coder-30B-A3B-Instruct --local-dir models/Qwen3-Coder-30B-A3B-Instruct
```

The starter kit's own `requirements.txt` lists what the task evaluators need; install it into the SkyDiscover venv as well.

## Running

From `AHD_mt6230069/`, with the course data in `problems/`:

```bash
MODEL=src/models/Qwen3-Coder-30B-A3B-Instruct bash src/run_banyan.sh prism 0 v5
```

- Arguments: task (`prism`, `cloudcast`, `signal_processing`, `heilbronn_triangle`, `circle_packing_rect`), seed (we used 0, 1, 2), and prompt folder (`v5`, `v7`, `v8`).
- The search stops at 500 evaluations or the task's wall clock (5 h; 10 h on the two math tasks).
- The output lands in `src/runs/<task>/banyan_seed<seed>_<time>/`. It contains every program, the logs, `selection.json` and the selected `mt6230069_heuristic.py`.
- Seeds: the seed is SkyDiscover's database random seed. LLM sampling at temperature 0.99 is not seeded, so reruns vary (the report gives mean ± std over seeds).

Optional environment variables:
- Paths: `SKY`, `VLLM_VENV`, `RUNS_ROOT`, `PORT`, and `TP=2` for two smaller GPUs.
- Ablations, each with a task-independent default: `BANYAN_P2=0` (A1 standing tree), `BANYAN_ALLOC=random` (A3), `BANYAN_DIAG=0` (A4), `BANYAN_FLAT=1` (A5).

All BANYAN constants are module constants in `scripts/banyan_core.py` and `scripts/banyan.py`. They are identical on every task (report, App. D).

**Baselines.** To run EvoX or AdaEvolve under the same protocol, run `scripts/run_task.sh` the same way as in `run_banyan.sh`, with search `evox` or `adaevolve` and without `BANYAN=1`. Then run `select_by_val.py` on the result.

**Re-scoring a submitted program.** Run the starter kit's evaluator, for example `evaluate("leaderboard3/prism/mt6230069_heuristic.py", split="val")` from `problems/prism/evaluator.py`.

The ShinkaEvolve speedrun in Sec. 4 of the report was run by our teammates (cs1230018, cs1230151) from their own ShinkaEvolve-based code. It is not part of `src/`.
