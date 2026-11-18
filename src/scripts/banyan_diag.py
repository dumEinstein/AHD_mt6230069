"""BANYAN's diagnoser: the v3 Diagnoser with the spec-section-9 same-output gate (design/round7/BANYAN.md).

diagnoser.Diagnoser re-runs a program and compares with the LOGGED score, which a clock-paced (non-deterministic)
program can never reproduce, so v3 would give it no diagnosis. ReplayDiagnoser instead (scripts/banyan_replay.py):
  1. re-runs the evaluated program once under the task evaluator with its outputs recorded;
  2. scores a replay copy that returns those stored outputs, with the same evaluator;
  3. accepts the replay only if that score equals the live re-run's score;
  4. runs the generated diagnoser on the replay copy, where v3's gate (every per-instance score, the combined
     score, every check) is exact because the replay is deterministic.
Every re-run is of an already-evaluated program on the training split and creates no candidate; none goes through
the run's evaluation counter. The re-run's score also yields a noise sample |re-run - logged| for the record margin.
The path each program took ("replay" or why not) is logged, so a task where the replay cannot work is visible.
"""
import asyncio
import os
from typing import Any, Callable, Dict, List, Optional, Tuple

import banyan_replay as br
import diagnoser as diagmod

FAITHFUL_TOL = 1e-6        # relative: the evaluator's score of the replay vs of the live re-run it was recorded from


class ReplayDiagnoser(diagmod.Diagnoser):
    def __init__(self, evaluator_path: str, workdir: str, llm, build_prompt, timeout: float,
                 log: Callable[..., None] = lambda *a, **k: None, parallel: int = 4,
                 on_sample: Optional[Callable[..., None]] = None):
        super().__init__(evaluator_path, workdir, llm, build_prompt, timeout, log)
        self.entries = br.entry_names(open(evaluator_path).read())
        self.parallel = parallel
        self.on_sample = on_sample             # (program_id, |live - logged|, live, logged)
        self.replays: Dict[str, Dict[str, Any]] = {}
        self.paths: Dict[str, int] = {}

    # ------------------------------------------------------------ record + replay of one program
    def replay_for(self, pid: str, code: str, logged: float) -> Dict[str, Any]:
        """Re-run, record, replay and compare (cached per program). r["path"] == "replay" means it can be diagnosed."""
        r = self.replays.get(pid)
        if r is not None:
            return r
        root = os.path.join(self.workdir, "replay", pid[:16])
        store, rec_dir, rep_dir = (os.path.join(root, d) for d in ("store", "rec", "rep"))
        for d in (rec_dir, rep_dir):
            os.makedirs(d, exist_ok=True)
        r = {"path": "", "logged": logged, "live": None, "replay": None, "code": None, "feedback": None, "detail": ""}
        if logged <= 0:                       # a program that scored 0 has nothing to diagnose and no noise to measure
            r["path"] = "logged_zero"
            self.replays[pid] = r
            self.paths[r["path"]] = self.paths.get(r["path"], 0) + 1
            return r
        rec_py = os.path.join(rec_dir, "program.py")
        with open(rec_py, "w") as fh:
            fh.write(br.wrapped_source(code, self.entries, store, "record"))
        rec = br.run_eval(self.evaluator_path, rec_py, self.timeout, self.parallel)
        self.seconds += rec.get("seconds", 0.0)
        if not rec.get("ok"):
            r.update(path="rerun_failed", detail=str(rec.get("error"))[:300])
        else:
            r["live"] = float(rec["combined"])
            if self.on_sample:
                self.on_sample(pid, abs(r["live"] - logged), r["live"], logged)
            files = sorted(os.listdir(store)) if os.path.isdir(store) else []
            pk = [f for f in files if f.endswith(".pkl")]
            fails = [f for f in files if f.endswith(".fail")]
            if r["live"] <= 0:
                r["path"] = "rerun_invalid"
            elif fails:
                r.update(path="payload_unpicklable", detail=open(os.path.join(store, fails[0])).read()[:200])
            elif not pk:
                r.update(path="no_payload", detail=f"entry points {self.entries} were never called")
            else:
                rep_py = os.path.join(rep_dir, "program.py")
                with open(rep_py, "w") as fh:
                    fh.write(br.wrapped_source(code, self.entries, store, "replay"))
                rep = br.run_eval(self.evaluator_path, rep_py, self.timeout, self.parallel)
                self.seconds += rep.get("seconds", 0.0)
                if not rep.get("ok"):
                    r.update(path="replay_failed", detail=str(rep.get("error"))[:300])
                else:
                    r["replay"] = float(rep["combined"])
                    if abs(r["replay"] - r["live"]) > FAITHFUL_TOL * max(1.0, abs(r["live"])):
                        r.update(path="replay_unfaithful",
                                 detail=f"evaluator scored the replay {r['replay']:.10g}, the live re-run {r['live']:.10g}")
                    else:
                        with open(rep_py) as fh:
                            r.update(path="replay", code=fh.read(), feedback=rep.get("feedback") or "")
        self.replays[pid] = r
        self.paths[r["path"]] = self.paths.get(r["path"], 0) + 1
        self.log("diag_path", program=pid, path=r["path"], live=r["live"], replay=r["replay"], logged=logged,
                 detail=r["detail"], seconds=round(self.seconds))
        return r

    def noise_run(self, pid: str, code: str, logged: float) -> Optional[float]:
        """One more live re-run of an evaluated program, only for the noise sample |re-run - logged| (no replay)."""
        k = self._noise_runs = getattr(self, "_noise_runs", 0) + 1
        root = os.path.join(self.workdir, "replay", f"noise_{pid[:12]}_{k}")
        store = os.path.join(root, "store")
        os.makedirs(root, exist_ok=True)
        py = os.path.join(root, "program.py")
        with open(py, "w") as fh:
            fh.write(br.wrapped_source(code, self.entries, store, "record"))
        rec = br.run_eval(self.evaluator_path, py, self.timeout, self.parallel)
        self.seconds += rec.get("seconds", 0.0)
        if not rec.get("ok"):
            self.log("noise_failed", program=pid, error=str(rec.get("error"))[:300])
            return None
        live = float(rec["combined"])
        if self.on_sample:
            self.on_sample(pid, abs(live - logged), live, logged)
        return live

    # ------------------------------------------------------------ overrides
    async def build(self, refs: List[Tuple[str, str, str, float]], previous: str = "", failure: str = "") -> bool:
        """Reference programs are first turned into replays; one with no faithful replay cannot gate the module."""
        conv = []
        for pid, code, fb, comb in refs:
            r = await asyncio.to_thread(self.replay_for, pid, code, comb)
            if r["path"] == "replay":
                conv.append((pid, r["code"], r["feedback"], r["replay"]))
        return await super().build(conv, previous=previous, failure=failure)

    def diagnose(self, pid: str, code: str, feedback: str, combined: float) -> Optional[str]:
        """Run (cached) and verify on the replay. Returns the rendered diagnosis if verified AND trusted."""
        if self.path is None:
            return None
        c = self.cache.get(pid)
        if c is None or c["version"] != self.version:
            r = self.replay_for(pid, code, combined)
            if r["path"] == "replay":
                rows, err, nondet, dt = diagmod.check(self.path, self.evaluator_path, r["code"], r["feedback"],
                                                      r["replay"], self.timeout)
                self.seconds += dt
                c = {"version": self.version, "rows": rows, "ok": err is None, "error": err, "nondet": nondet,
                     "seconds": dt, "path": "replay", "live": r["live"], "logged": combined}
                if err is None and r["replay"] > 0:
                    self.verified.add(round(r["replay"], 9))
                if not nondet:
                    self.misses = (self.misses + [err is not None])[-5:]
            else:                                  # no faithful replay: nothing to blame the diagnoser for
                c = {"version": self.version, "rows": None, "ok": False, "error": r["path"], "nondet": True,
                     "seconds": 0.0, "path": r["path"], "live": r["live"], "logged": combined}
            self.cache[pid] = c
            self.log("diag_run", program=pid, version=self.version, ok=c["ok"], nondet=c["nondet"], path=c["path"],
                     seconds=round(c["seconds"], 1), error=(c["error"] or "")[:300], trusted=self.trusted)
        if not (c["ok"] and self.trusted):
            return None
        head = (f"(measured on a re-run of this program that scored {c['live']:.6g}; its logged score is "
                f"{c['logged']:.6g}. The program is clock-paced, so its score differs a little from run to run.)\n")
        return head + diagmod.render(c["rows"])
