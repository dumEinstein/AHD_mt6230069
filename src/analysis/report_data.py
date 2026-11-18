"""Per-run facts for the final report (REPORT_HANDOFF.md §9), run on the cluster from the project root.

  ~/flash/col8385/envs/vllm/bin/python analysis/report_data.py > analysis/report_runs.jsonl

One JSON row per run directory of every final-kit cohort (no de-duplication: duplicate seed folders are kept and
tagged with their timestamp so the report can state its rule). Fields:
  arm, cohort, task, seed, stamp, dir     identity
  scores                                  train score of every evaluation in order (initial program excluded)
  sel                                     selection.json "best" (train, val, program id)
  wall                                    run wall clock in s (run.log wall_clock_seconds, else first..last log line)
  llm_calls                               chat-completion requests in logs/*.log (every log of a resumed run)
  prompt_tok, completion_tok              token counts: prompts tokenised exactly on a sample of requests and scaled
                                          by characters; completions estimated from the response content-length
                                          (bytes minus a fixed JSON envelope, at the prompts' bytes-per-token)
  mech                                    BANYAN only: incumbent-parent share, trunk share, records, re-roots, F/A, diag
"""
import ast
import glob
import json
import os
import random
import re
import sys
from datetime import datetime

from tokenizers import Tokenizer

TOK = Tokenizer.from_file(os.path.expanduser("~/flash/col8385/models/Qwen3-Coder-30B-A3B-Instruct/tokenizer.json"))
TASKS = ["prism", "cloudcast", "signal_processing", "heilbronn_triangle", "circle_packing_rect"]
COHORTS = [  # (arm, cohort, root)
    ("banyan", "v5", "runs_final/banyan"), ("evox", "v5", "runs_final/base_v5/evox"),
    ("adaevolve", "v5", "runs_final/base_v5/adaevolve"), ("adaevolve", "v5fixed", "runs_final/base_v5_fixed/adaevolve"),
    ("banyan", "v7", "runs_final/v7/banyan"), ("evox", "v7", "runs_final/v7/evox"), ("adaevolve", "v7", "runs_final/v7/adaevolve"),
    ("banyan", "v8", "runs_final/v8/banyan"), ("evox", "v8", "runs_final/v8/evox"), ("adaevolve", "v8", "runs_final/v8/adaevolve"),
] + [("banyan", "abl_" + a, "runs_final/ablation/" + a) for a in ["ref", "p2off", "random", "nodiag", "flat"]]
SEED_RE = re.compile(r"_seed(\d+)_(\d{8}_\d{6})$")
TS_RE = re.compile(r"^(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d)")
LEN_RE = re.compile(r"'content-length': '(\d+)'")
ENVELOPE = 550  # bytes of a vLLM chat.completion JSON around the message content (id, usage, finish_reason, ...)
SAMPLE = 40     # requests per run tokenised exactly


def scores_of(d):
    if os.path.exists(d + "/tree_stats.jsonl"):
        return [float(json.loads(l)["score"] or 0.0) for l in open(d + "/tree_stats.jsonl") if l.strip()]
    st = glob.glob(d + "/*iteration_stats*.jsonl")
    if st:
        out = []
        for f in sorted(st):
            for l in open(f):
                try:
                    s = json.loads(l)
                except ValueError:
                    continue
                ir = s.get("iteration_result") or {}
                ch = ir.get("child_program")
                if ch and ir.get("success", True):
                    out.append(float((ch.get("metrics") or {}).get("combined_score") or 0.0))
        return out
    cks = sorted(glob.glob(d + "/checkpoints/checkpoint_*"), key=lambda c: int(c.rsplit("_", 1)[1]))
    progs = []
    for f in glob.glob(cks[-1] + "/programs/*.json") if cks else []:
        try:
            p = json.load(open(f))
        except ValueError:
            continue
        if p.get("iteration_found", 0) == 0 and not p.get("parent_id"):
            continue
        progs.append(p)
    progs.sort(key=lambda p: (p.get("iteration_found", 0), p.get("timestamp", 0)))
    return [float((p.get("metrics") or {}).get("combined_score") or 0.0) for p in progs]


def log_facts(d):
    calls, pchars, rbytes, first, last, sample = 0, 0, 0, None, None, []
    for f in sorted(glob.glob(d + "/logs/*.log")):
        for line in open(f, errors="replace"):
            m = TS_RE.match(line)
            if m:
                t = datetime.fromisoformat(m.group(1))
                first = first or t
                last = t
            if "Request options:" in line and "/chat/completions" in line:
                calls += 1
                try:
                    opts = ast.literal_eval(line.split("Request options: ", 1)[1].strip())
                    text = "".join(str(msg.get("content") or "") for msg in opts["json_data"]["messages"])
                except (ValueError, SyntaxError, KeyError, TypeError):
                    continue
                pchars += len(text)
                if len(sample) < SAMPLE:
                    sample.append(text)
                elif random.random() < SAMPLE / calls:
                    sample[random.randrange(SAMPLE)] = text
            elif "HTTP Response: POST" in line and "/chat/completions" in line:
                m = LEN_RE.search(line)
                if m:
                    rbytes += max(0, int(m.group(1)) - ENVELOPE)
    wall = None
    if os.path.exists(d + "/run.log"):
        for l in open(d + "/run.log", errors="replace"):
            if l.startswith("wall_clock_seconds="):
                wall = float(l.split("=", 1)[1])
    if wall is None and first and last:
        wall = (last - first).total_seconds()
    ptok = ctok = None
    if sample:
        chars = sum(len(s) for s in sample)
        toks = sum(len(e.ids) for e in TOK.encode_batch(sample))
        ptok = pchars * toks / chars
        # response bytes are JSON-escaped text: use the escaped length of the same sample for bytes per token
        ctok = rbytes * toks / sum(len(json.dumps(s)) - 2 for s in sample)
    return dict(llm_calls=calls, wall=wall, prompt_tok=ptok, completion_tok=ctok)


def mech_of(d):
    if not os.path.exists(d + "/tree_stats.jsonl"):
        return None
    rows = [json.loads(l) for l in open(d + "/tree_stats.jsonl") if l.strip()]
    if not rows:
        return None
    n = len(rows)
    edits = [r for r in rows if r["role"] == "edit"]
    return dict(inc=sum(r["parent_is_incumbent"] for r in rows) / n, trunk=sum(r["node_role"] == "trunk" for r in rows) / n,
                rec=sum(r["global_record"] for r in rows), reroots=max(r["reroots"] for r in rows),
                F=len({r["framing"] for r in rows}), A=len({r["attempt"] for r in rows}),
                valid_flag=sum(bool(r["valid"]) for r in rows) / n,
                diag=sum(bool(r.get("diag_shown")) for r in edits) / max(1, len(edits)))


random.seed(0)
for arm, cohort, root in COHORTS:
    for task in TASKS:
        for d in sorted(glob.glob(f"{root}/{task}/{arm}_seed*")):
            m = SEED_RE.search(d)
            if not m:
                continue
            sel = json.load(open(d + "/selection.json")).get("best", {}) if os.path.exists(d + "/selection.json") else {}
            row = dict(arm=arm, cohort=cohort, task=task, seed=int(m.group(1)), stamp=m.group(2), dir=d,
                       scores=scores_of(d), sel={k: sel.get(k) for k in ("train", "val", "id", "program_id")},
                       mech=mech_of(d), **log_facts(d))
            print(json.dumps(row), flush=True)
            print(f"{cohort:10s} {arm:9s} {task:20s} s{row['seed']} n={len(row['scores'])} calls={row['llm_calls']}",
                  file=sys.stderr, flush=True)
