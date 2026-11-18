"""Subprocess side of scripts/diagnoser.py: run a generated diagnoser on one program.
  python diag_runner.py <diagnoser.py> <evaluator.py> <program.py> <out.json>

The evaluator is imported as module `evaluator` (its own sys.path setup, data, scoring helpers), with
data.load restricted to the training split. `diagkit` gives the diagnoser the evaluator's process isolation
with the payload kept (evalkit.run_instances keeps only the score)."""
import importlib.util
import json
import multiprocessing as mp
import multiprocessing.connection as mpc
import os
import signal
import sys
import time
import traceback
import types


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _make_diagkit(limit, grace, parallel):
    kit = types.ModuleType("diagkit")

    def run_each(fn, args_list, limit=limit, grace=grace, parallel=parallel):
        """fn(*args) for each args, each in its own forked process, at most `parallel` at once, killed after
        limit + grace seconds. Returns, in order, ("ok", value) | ("error", "Type: message") | ("timeout", None)."""
        ctx = mp.get_context("fork")
        out, pending, running = [None] * len(args_list), list(enumerate(args_list)), {}

        def target(w, a):
            os.setpgrp()
            try:
                w.send(("ok", fn(*a)))
            except BaseException as e:  # noqa: BLE001
                w.send(("error", f"{type(e).__name__}: {e}"))
            finally:
                w.close()

        def kill(p):
            if p.is_alive():
                try:
                    os.killpg(p.pid, signal.SIGKILL)
                except (ProcessLookupError, PermissionError):
                    p.kill()
            p.join(5)

        try:
            while pending or running:
                while pending and len(running) < max(1, parallel):
                    i, a = pending.pop(0)
                    r, w = ctx.Pipe(duplex=False)
                    p = ctx.Process(target=target, args=(w, a), daemon=True)
                    p.start()
                    w.close()
                    running[r] = (i, p, time.monotonic())
                dl = min(t0 + limit + grace for (_, _, t0) in running.values())
                for r in mpc.wait(list(running), timeout=max(0.0, dl - time.monotonic())):
                    i, p, _ = running.pop(r)
                    try:
                        out[i] = r.recv()
                    except EOFError:
                        out[i] = ("error", "process died before reporting")
                    r.close()
                    kill(p)
                now = time.monotonic()
                for r, (i, p, t0) in list(running.items()):
                    if now - t0 >= limit + grace:
                        running.pop(r)
                        r.close()
                        kill(p)
                        out[i] = ("timeout", None)
        finally:
            for r, (_, p, _) in list(running.items()):
                r.close()
                kill(p)
        return out

    kit.run_each = run_each
    kit.TIME_LIMIT, kit.GRACE, kit.PARALLEL = limit, grace, parallel
    return kit


def main():
    diag_path, ev_path, prog_path, out_path = sys.argv[1:5]
    ev = _load(ev_path, "evaluator")
    data = getattr(ev, "data", None)
    if data is not None and hasattr(data, "load"):
        orig = data.load

        def train_only(split="train", *a, **k):
            if split != "train":
                raise PermissionError(f"the diagnoser may use the training split only, not {split!r}")
            return orig(split, *a, **k)
        data.load = train_only
    limit = float(getattr(ev, "TIME_LIMIT", 60.0))
    grace = float(getattr(ev, "GRACE", 5.0))
    parallel = int(getattr(getattr(ev, "common", None), "PARALLEL", 4) or 4)
    sys.modules["diagkit"] = _make_diagkit(limit, grace, parallel)
    try:
        mod = _load(diag_path, "generated_diagnoser")
        rows = mod.diagnose(prog_path)
        out = []
        for r in rows:
            checks = []
            for c in r.get("checks") or []:
                checks.append({"name": str(c["name"])[:80], "value": float(c["value"]), "limit": float(c["limit"]),
                               "sense": str(c["sense"]).strip()})
            fc = r.get("failed_check")
            out.append({"label": str(r.get("label", "")), "score": float(r.get("score")),
                        "report": str(r.get("report", "")), "checks": checks,
                        "failed_check": str(fc) if fc else None})
        rows = out
    except BaseException:  # noqa: BLE001
        sys.stderr.write(traceback.format_exc(limit=4)[-1500:])
        sys.exit(2)
    with open(out_path, "w") as fh:
        json.dump(rows, fh)


if __name__ == "__main__":
    main()
