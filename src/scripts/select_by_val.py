"""Pick the submission for one run: rank logged programs by train score, re-score the top K on val.

  python select_by_val.py <run_dir> <task_dir> [--top 20] [--name kerberos]

Reads every program from the run's latest checkpoint, evaluates the top-K (by train
combined_score) on the validation split, and writes <run_dir>/selection.json plus
<run_dir>/<name>_heuristic.py (the best on val, unmodified). For the two math tasks
train and val are the same instance, so this just picks the best train score.
"""
import argparse, glob, importlib.util, json, os, sys

ap = argparse.ArgumentParser()
ap.add_argument("run_dir"); ap.add_argument("task_dir")
ap.add_argument("--top", type=int, default=20); ap.add_argument("--name", default="kerberos")
a = ap.parse_args()

ckpts = sorted(glob.glob(os.path.join(a.run_dir, "checkpoints", "checkpoint_*")),
               key=lambda p: int(p.rsplit("_", 1)[1]))
if not ckpts:
    sys.exit(f"no checkpoints under {a.run_dir}")
progs = [json.load(open(f)) for f in glob.glob(os.path.join(ckpts[-1], "programs", "*.json"))]
progs = [p for p in progs if p.get("metrics", {}).get("combined_score") is not None]
progs.sort(key=lambda p: p["metrics"]["combined_score"], reverse=True)

sys.path.insert(0, os.path.dirname(os.path.abspath(a.task_dir)))
spec = importlib.util.spec_from_file_location("task_evaluator", os.path.join(a.task_dir, "evaluator.py"))
ev = importlib.util.module_from_spec(spec); spec.loader.exec_module(ev)

cand_dir = os.path.join(a.run_dir, "val_candidates"); os.makedirs(cand_dir, exist_ok=True)
rows = []
for p in progs[: a.top]:
    path = os.path.join(cand_dir, f"{p['id']}.py")
    open(path, "w").write(p["solution"])
    res = ev.evaluate(path, split="val")
    val = res["combined_score"] if isinstance(res, dict) else getattr(res, "metrics", {}).get("combined_score")
    rows.append({"id": p["id"], "iteration": p.get("iteration_found"),
                 "train": p["metrics"]["combined_score"], "val": val, "path": path})
    print(f"{p['id'][:8]} iter={rows[-1]['iteration']} train={rows[-1]['train']:.4f} val={val:.4f}")

best = max(rows, key=lambda r: r["val"])
out = os.path.join(a.run_dir, f"{a.name}_heuristic.py")
open(out, "w").write(open(best["path"]).read())
json.dump({"best": best, "candidates": rows}, open(os.path.join(a.run_dir, "selection.json"), "w"), indent=2)
print(f"selected {best['id']} (train {best['train']:.4f}, val {best['val']:.4f}) -> {out}")
