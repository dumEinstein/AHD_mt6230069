"""Same-output replay for BANYAN's diagnoser (design/round7/BANYAN.md section 9).

A clock-paced program is non-deterministic, so the v3 gate (re-run the program, compare with the logged score) can
never pass. Instead, per diagnosed program:
  1. RECORD  the evaluator scores a recording copy of the program (the program plus a wrapper around the entry
             points the evaluator calls, found as `prog.<name>(` in the evaluator source). Every call's return value
             is pickled next to a fingerprint of its arguments. The evaluator's score of this live re-run is the
             noise sample |re-run - logged| (and the program is never scored as a new candidate).
  2. REPLAY  a replay copy of the program (same source, entry points replaced by stubs that return the stored
             payload for the same arguments) is scored by the SAME evaluator. If its score is not the live re-run's
             score, the replay is not faithful and the program gets no diagnosis.
  3. The generated diagnoser then runs on the replay copy: deterministic by construction, and checked by the v3
             gate against the evaluator's score of THAT payload (diagnoser.check).
Nothing here is task-specific: entry points come from the evaluator source, arguments are fingerprinted
structurally, and results that alias the call's arguments (prism returns the input Model objects) keep their
identity through pickle persistent ids.

  python banyan_replay.py eval <evaluator.py> <program.py> <out.json>     # subprocess side: evaluate on TRAIN
  python banyan_replay.py selftest <evaluator.py> <program.py> <workdir>  # record + replay one program, print result
"""
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from typing import Any, Dict, List, Optional

_HERE = os.path.dirname(os.path.abspath(__file__))
SELF = os.path.abspath(__file__)

# Shared by both tails: the fingerprint of a call, and the registry of objects reachable from its arguments.
_COMMON = r'''
def __banyan_install(_G, _entries, _dir, _mode):
    import functools, hashlib, inspect, io, os, pickle, sys

    def _atom(o):
        return o is None or isinstance(o, (bool, int, float, complex, str, bytes))

    def _fp(o, depth=0):
        if o is None or isinstance(o, (bool, int, float, str, bytes)):
            return repr(o)[:200]
        if hasattr(o, "dtype") and hasattr(o, "tobytes") and hasattr(o, "shape"):
            try:
                return "arr%s:%s" % (tuple(o.shape), hashlib.sha1(o.tobytes()).hexdigest())
            except Exception:
                return "arr"
        if depth > 3:
            return type(o).__name__
        if isinstance(o, (list, tuple)):
            return "[" + ",".join(_fp(x, depth + 1) for x in list(o)[:64]) + "]%d" % len(o)
        if isinstance(o, dict):
            return "{" + ",".join(_fp(k, depth + 1) + ":" + _fp(v, depth + 1) for k, v in list(o.items())[:64]) + "}%d" % len(o)
        d = getattr(o, "__dict__", None)
        if isinstance(d, dict) and not isinstance(o, type):
            return type(o).__name__ + "(" + ",".join("%s=%s" % (k, _fp(v, depth + 1)) for k, v in list(d.items())[:32]) + ")"
        return type(o).__name__

    def _children(o):
        if isinstance(o, (list, tuple)):
            return list(o)
        if isinstance(o, dict):
            return [x for kv in o.items() for x in kv]
        d = getattr(o, "__dict__", None)
        if isinstance(d, dict) and not isinstance(o, type):
            return list(d.values())
        return []

    def _registry(vals):
        reg, order = {}, []

        def visit(o, depth):
            if _atom(o) or depth > 6 or len(order) > 200000 or id(o) in reg:
                return
            reg[id(o)] = len(order)
            order.append(o)
            for c in _children(o):
                visit(c, depth + 1)
        for v in vals:
            visit(v, 0)
        return reg, order

    def _bind(sig, a, k):
        if sig is not None:
            try:
                b = sig.bind(*a, **k)
                b.apply_defaults()
                return list(b.arguments.values())
            except Exception:
                pass
        return list(a) + [v for _, v in sorted(k.items())]

    def _key(name, vals):
        return hashlib.sha1((name + "|" + "|".join(_fp(v) for v in vals)).encode()).hexdigest()[:20]

    class _P(pickle.Pickler):
        def __init__(self, f, reg):
            super().__init__(f, protocol=4)
            self._reg = reg

        def persistent_id(self, obj):
            i = self._reg.get(id(obj)) if not _atom(obj) else None
            return ("arg", i) if i is not None else None

    class _U(pickle.Unpickler):
        def __init__(self, f, order):
            super().__init__(f)
            self._order = order

        def persistent_load(self, pid):
            return self._order[pid[1]]

    if _mode == "record":
        os.makedirs(_dir, exist_ok=True)
        try:
            with open(os.path.join(_dir, "meta.json"), "w") as fh:
                fh.write('{"module": "%s"}' % _G.get("__name__", "program"))
        except Exception:
            pass

        def _save(name, key, payload, reg):
            tmp = os.path.join(_dir, "%s_%s.tmp%d" % (name, key, os.getpid()))
            try:
                buf = io.BytesIO()
                _P(buf, reg).dump(payload)
                with open(tmp, "wb") as fh:
                    fh.write(buf.getvalue())
                os.replace(tmp, os.path.join(_dir, "%s_%s.pkl" % (name, key)))
            except BaseException as e:
                try:
                    with open(os.path.join(_dir, "%s_%s.fail" % (name, key)), "w") as fh:
                        fh.write("%s: %s" % (type(e).__name__, e))
                except Exception:
                    pass

        def _make(name, fn):
            try:
                sig = inspect.signature(fn)
            except Exception:
                sig = None

            @functools.wraps(fn)
            def wrapper(*a, **k):
                try:
                    vals = _bind(sig, a, k)
                    key, (reg, _) = _key(name, vals), _registry(vals)
                except BaseException:
                    return fn(*a, **k)
                try:
                    res = fn(*a, **k)
                except BaseException as e:
                    _save(name, key, ("raise", "%s: %s" % (type(e).__name__, e)), reg)
                    raise
                _save(name, key, ("ok", res), reg)
                return res
            return wrapper
    else:
        try:
            _mod = __import__("json").load(open(os.path.join(_dir, "meta.json"))).get("module")
            if _mod and _mod not in sys.modules:       # payload classes were defined in the module of the recording run
                sys.modules[_mod] = sys.modules[_G.get("__name__")]
        except Exception:
            pass

        def _make(name, fn):
            try:
                sig = inspect.signature(fn)
            except Exception:
                sig = None

            def stub(*a, **k):
                vals = _bind(sig, a, k)
                key, (reg, order) = _key(name, vals), _registry(vals)
                path = os.path.join(_dir, "%s_%s.pkl" % (name, key))
                if not os.path.exists(path):
                    have = [f for f in os.listdir(_dir) if f.startswith(name + "_") and f.endswith(".pkl")]
                    if len(have) != 1:
                        raise RuntimeError("banyan replay: no recorded payload for this call (%d recorded)" % len(have))
                    path = os.path.join(_dir, have[0])
                with open(path, "rb") as fh:
                    tag, val = _U(fh, order).load()
                if tag == "raise":
                    raise RuntimeError("replayed exception: " + val)
                return val
            stub.__name__ = getattr(fn, "__name__", name)
            return stub

    for _n in _entries:
        _f = _G.get(_n)
        if callable(_f) and not isinstance(_f, type):
            _G[_n] = _make(_n, _f)


__banyan_install(globals(), __ENTRIES__, __DIR__, __MODE__)
'''


def entry_names(evaluator_src: str) -> List[str]:
    """Functions the evaluator calls on the program module: `prog.<name>(`."""
    return sorted(set(re.findall(r"\bprog\.(\w+)\s*\(", evaluator_src or "")))


def _tail(entries: List[str], store_dir: str, mode: str) -> str:
    return (_COMMON.replace("__ENTRIES__", repr(list(entries))).replace("__DIR__", repr(store_dir))
            .replace("__MODE__", repr(mode)))


def wrapped_source(program_code: str, entries: List[str], store_dir: str, mode: str) -> str:
    """The program followed by the record ("record") or replay ("replay") wrapper."""
    assert mode in ("record", "replay")
    return program_code.rstrip("\n") + "\n\n# ===== banyan %s wrapper (appended by the controller; not part of the candidate)\n" % mode + \
        _tail(entries, store_dir, mode)


def run_eval(evaluator_path: str, program_path: str, timeout: float, parallel: int = 4,
             repeat: int = 1) -> Dict[str, Any]:
    """Score a program file with the task evaluator (train split) in a fresh subprocess. Never goes through the
    run's counted Evaluator, so it spends no evaluation. Returns {"ok", "combined", "feedback", "seconds", "error"}."""
    t0 = time.time()
    with tempfile.TemporaryDirectory(prefix="banyan_eval_") as td:
        out = os.path.join(td, "out.json")
        env = dict(os.environ, EVAL_PARALLEL=str(parallel))
        try:
            p = subprocess.run([sys.executable, SELF, "eval", evaluator_path, program_path, out, str(repeat)], env=env,
                               capture_output=True, text=True, timeout=timeout, start_new_session=True)
        except subprocess.TimeoutExpired:
            return {"ok": False, "error": f"evaluation did not finish within {timeout:.0f}s", "seconds": time.time() - t0}
        if p.returncode != 0 or not os.path.exists(out):
            tail = (p.stderr or p.stdout or "").strip()[-600:]
            return {"ok": False, "error": tail or f"exit code {p.returncode}", "seconds": time.time() - t0}
        try:
            r = json.load(open(out))
        except Exception as e:
            return {"ok": False, "error": f"output unreadable: {e}", "seconds": time.time() - t0}
    return {"ok": True, **r, "seconds": time.time() - t0}


def _main_eval(ev_path: str, prog_path: str, out_path: str, repeat: str = "1"):
    sys.path.insert(0, _HERE)
    import importlib.util

    spec = importlib.util.spec_from_file_location("evaluator", ev_path)
    ev = importlib.util.module_from_spec(spec)
    sys.modules["evaluator"] = ev
    spec.loader.exec_module(ev)
    data = getattr(ev, "data", None)
    if data is not None and hasattr(data, "load"):
        orig = data.load

        def train_only(split="train", *a, **k):
            if split != "train":
                raise PermissionError(f"BANYAN re-runs use the training split only, not {split!r}")
            return orig(split, *a, **k)
        data.load = train_only
    for _ in range(max(1, int(repeat))):          # repeat > 1: the last evaluation is warm (caches, imports)
        t0 = time.time()
        res = ev.evaluate(prog_path, split="train")
        eval_seconds = time.time() - t0
    if hasattr(res, "metrics"):
        metrics, feedback = dict(res.metrics or {}), (getattr(res, "artifacts", None) or {}).get("feedback", "")
    else:
        metrics, feedback = {k: v for k, v in dict(res).items() if k != "feedback"}, dict(res).get("feedback", "")
    with open(out_path, "w") as fh:
        json.dump({"combined": float(metrics.get("combined_score", 0.0)), "feedback": feedback or "",
                   "n_instances": metrics.get("n_instances"), "eval_seconds": eval_seconds}, fh)


def selftest(ev_path: str, prog_path: str, workdir: str, parallel: int = 2):
    """Record + replay one program file; print what the controller would conclude."""
    code = open(prog_path).read()
    entries = entry_names(open(ev_path).read())
    store = os.path.join(workdir, "store")
    os.makedirs(workdir, exist_ok=True)
    rec_dir, rep_dir = os.path.join(workdir, "rec"), os.path.join(workdir, "rep")
    for d in (rec_dir, rep_dir):
        os.makedirs(d, exist_ok=True)
    open(os.path.join(rec_dir, "program.py"), "w").write(wrapped_source(code, entries, store, "record"))
    rec = run_eval(ev_path, os.path.join(rec_dir, "program.py"), 1800, parallel)
    files = sorted(os.listdir(store)) if os.path.isdir(store) else []
    open(os.path.join(rep_dir, "program.py"), "w").write(wrapped_source(code, entries, store, "replay"))
    rep = run_eval(ev_path, os.path.join(rep_dir, "program.py"), 1800, parallel)
    print(json.dumps({"entries": entries, "record": {k: rec.get(k) for k in ("ok", "combined", "seconds", "error")},
                      "payload_files": len(files), "fail_files": [f for f in files if f.endswith(".fail")],
                      "replay": {k: rep.get(k) for k in ("ok", "combined", "seconds", "error")},
                      "faithful": bool(rec.get("ok") and rep.get("ok") and
                                       abs(rec["combined"] - rep["combined"]) <= 1e-9 * max(1.0, abs(rec["combined"])))},
                     indent=1))


if __name__ == "__main__":
    if sys.argv[1] == "eval":
        _main_eval(*sys.argv[2:6])
    elif sys.argv[1] == "selftest":
        selftest(*sys.argv[2:5])
