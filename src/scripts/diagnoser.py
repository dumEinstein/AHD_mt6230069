"""Logic-level diagnosis of evaluated programs, written by the generator itself (CredTree v3, design/round6 §14).

The evaluator returns one score per instance. To say *why* a program scores what it does (which GPU binds,
which triangle is smallest, which penalty term dominates), the generator reads the evaluator source once per
run and writes a diagnoser module:

    def diagnose(program_path) -> [{"label": str, "score": float, "report": str}, ...]   # one per TRAIN instance

Nothing here is task-specific: the same code builds, checks and runs the diagnoser on all five tasks (handout
2.3); only the generated module differs, and it is generated during the run.

Correctness gate. A diagnoser is trusted only after it reproduces the evaluator's official result -- every
per-instance score in the feedback and the full-precision combined_score -- on TRUST_MIN valid programs with
distinct nonzero scores (a zero-scoring initial program proves nothing). After that every diagnosis is checked
the same way before it is shown; a program it does not reproduce gets no diagnosis, and repeated misses trigger
a rebuild with the evidence.

Budget. diagnose() re-runs a program that has ALREADY been evaluated, on the training split only (data.load is
wrapped to refuse any other split): no new candidate is scored, so no evaluation is spent.

No skydiscover import: the caller passes an async llm(user) -> text.
"""
import ast
import asyncio
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from typing import Any, Awaitable, Callable, Dict, List, Optional, Tuple

_HERE = os.path.dirname(os.path.abspath(__file__))
RUNNER = os.path.join(_HERE, "diag_runner.py")

SRC_CHARS = 16000          # evaluator + helper sources in the build prompt (16k-token context, 8k output)
REPORT_CHARS = 450         # per instance (the checks line is added after)
SHOW = 5                   # instances shown in a prompt when there are more
DIAG_CHARS = 3200           # whole diagnosis section
TRUST_MIN = int(os.environ.get("DIAG_TRUST_MIN", "3"))   # distinct nonzero-scored programs reproduced before any diagnosis is shown
REL_TOL = 1e-5             # feedback prints scores with 6 significant digits
MEAN_TOL = 1e-6            # combined_score is full precision
BUILD_TRIES = 6
PARALLEL = int(os.environ.get("DIAG_PARALLEL", "2"))   # instances at once inside a diagnosis run
ERR_RE = re.compile(r"Traceback|\b[A-Z]\w*(Error|Exception)\b|\berror:|\bfailed:|not iterable|has no attribute")
SPLIT_RE = re.compile(r"""['"](val|valid|validation|test)['"]""")


# ---------------------------------------------------------------- sources for the build prompt
def _signatures(src: str) -> str:
    """Top-level defs/classes with their docstrings and constants: what a module offers, not how."""
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return src[:1500]
    out = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            seg = ast.get_source_segment(src, node) or ""
            head = seg.split(":", 1)[0] if isinstance(node, ast.ClassDef) else seg.split(")", 1)[0] + ")"
            doc = ast.get_docstring(node)
            out.append(head.splitlines()[0] + ": ..." + (f'  # "{doc.splitlines()[0][:150]}"' if doc else ""))
            if isinstance(node, ast.ClassDef):
                for sub in node.body:
                    if isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        s = ast.get_source_segment(src, sub) or ""
                        out.append("    " + s.split(")", 1)[0].splitlines()[0] + "): ...")
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            seg = ast.get_source_segment(src, node) or ""
            if len(seg) < 200:
                out.append(seg)
    return "\n".join(out)


def module_names(src: str) -> str:
    """Names a module defines at top level (imports, assignments, defs), each with what it is."""
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return "(unparsable)"
    out = []
    for n in tree.body:
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
            out.append(f"{n.name} (function)")
        elif isinstance(n, ast.ClassDef):
            out.append(f"{n.name} (class)")
        elif isinstance(n, (ast.Import, ast.ImportFrom)):
            out += [f"{a.asname or a.name.split('.')[0]} (imported)" for a in n.names]
        elif isinstance(n, (ast.Assign, ast.AnnAssign)):
            targets = n.targets if isinstance(n, ast.Assign) else [n.target]
            for t in targets:
                for e in (t.elts if isinstance(t, ast.Tuple) else [t]):
                    if isinstance(e, ast.Name):
                        v = ast.get_source_segment(src, n.value) or ""
                        out.append(f"{e.id} (= {v[:60]})" if "load_program" in v else e.id)
    return ", ".join(dict.fromkeys(out))


def gather_sources(evaluator_path: str) -> str:
    """The evaluator in full, then the local modules it loads: in full while the budget lasts, else signatures."""
    ev_dir = os.path.dirname(os.path.abspath(evaluator_path))
    ev_src = open(evaluator_path).read()
    names = re.findall(r"""['"]([A-Za-z_][\w]*)\.py['"]""", ev_src)
    names += re.findall(r"^\s*(?:from|import)\s+([A-Za-z_][\w]*)", ev_src, re.M)
    files, seen = [], set()
    for n in names:
        for cand in (os.path.join(ev_dir, f"{n}.py"), *(os.path.join(ev_dir, d, f"{n}.py") for d in sorted(os.listdir(ev_dir))
                                                       if os.path.isdir(os.path.join(ev_dir, d)))):
            if os.path.isfile(cand) and cand not in seen:
                seen.add(cand)
                files.append(cand)
    parts = [f"### {os.path.basename(evaluator_path)}\n```python\n{ev_src}\n```\n"
             f"Top-level names of the evaluator module, i.e. everything `E.<name>` can reach: {module_names(ev_src)}"]
    left = SRC_CHARS - len(parts[0])
    for f in sorted(files, key=os.path.getsize):
        src = open(f).read()
        rel = os.path.relpath(f, ev_dir)
        if len(src) + 40 <= left:
            parts.append(f"### {rel}\n```python\n{src}\n```")
        else:
            parts.append(f"### {rel} (signatures only)\n```python\n{_signatures(src)}\n```")
        left -= len(parts[-1])
    return "\n\n".join(parts)


# ---------------------------------------------------------------- checks
def extract_code(text: Optional[str]) -> Optional[str]:
    blocks = re.findall(r"```(?:python)?\s*\n(.*?)```", text or "", re.S)
    return max(blocks, key=len) if blocks else None


def static_check(code: str) -> Optional[str]:
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return f"the module does not parse: {e.msg} at line {e.lineno}"
    if not any(isinstance(n, ast.FunctionDef) and n.name == "diagnose" for n in tree.body):
        return "the module must define a top-level function diagnose(program_path)"
    m = SPLIT_RE.search(code)
    if m:
        return f"the module names the split {m.group(0)}; it may use the training instances only (data.load('train'))"
    return None


def official(feedback: str) -> List[Tuple[str, str, float]]:
    """(label, status, score) per instance from the evaluator's 'label: status, score=x' feedback lines."""
    out = []
    for line in (feedback or "").splitlines():
        m = re.match(r"\s*(.+?):\s*(ok|error|timeout),\s*score=([-+0-9.eE]+|nan|inf)", line)
        if m:
            out.append((m.group(1).strip(), m.group(2), float(m.group(3))))
    return out


CHECK_TOL = 1e-5           # relative, on a check's limit (the evaluators' own tolerances are 1e-6)


def _passes(c: Dict[str, Any]) -> bool:
    v, lim = float(c["value"]), float(c["limit"])
    tol = CHECK_TOL * max(1.0, abs(lim))
    return v <= lim + tol if c["sense"] == "<=" else v >= lim - tol


def _slack(c: Dict[str, Any]) -> float:
    return float(c["limit"]) - float(c["value"]) if c["sense"] == "<=" else float(c["value"]) - float(c["limit"])


def compare(rows: List[Dict[str, Any]], feedback: str, combined: float) -> Optional[str]:
    """None if the diagnoser reproduced the official result, else what differs (shown to the generator on rebuild)."""
    ref = official(feedback)
    if not ref:
        return "no per-instance lines in the evaluator feedback to compare with"
    if len(rows) != len(ref):
        return f"diagnose() returned {len(rows)} instances, the evaluator scored {len(ref)}"
    bad = []
    for i, (r, (lab, st, sc)) in enumerate(zip(rows, ref)):
        d = r.get("score")
        if not isinstance(d, (int, float)) or d != d:
            bad.append(f"instance {i} ({lab}): score {d!r} is not a number")
        elif abs(d - sc) > REL_TOL * max(1.0, abs(sc)) + 1e-12:
            rep = re.sub(r"\s+", " ", str(r.get("report") or ""))[:220]
            bad.append(f"instance {i} ({lab}): diagnose() gave {d:.6g}, the evaluator gave {sc:.6g} ({st}); "
                       f"your report for it: {rep!r}")
    if bad:
        return f"{len(bad)} of {len(ref)} per-instance scores differ from the evaluator's:\n" + "\n".join(bad[:6])
    # checks: every comparison must pass where the evaluator accepted the output; where it rejected one and the
    # diagnoser names the comparison that failed (failed_check), that comparison must fail. A rejection that is no
    # comparison (missing items, wrong type, a crash) has failed_check null.
    for i, (r, (lab, st, sc)) in enumerate(zip(rows, ref)):
        checks = r.get("checks") or []
        for c in checks:
            if c.get("sense") not in ("<=", ">="):
                bad.append(f"instance {i} ({lab}): check {c.get('name')!r} has sense {c.get('sense')!r}; use '<=' or '>='")
        if bad:
            break
        fails = [c for c in checks if not _passes(c)]
        accepted = st == "ok" and sc > 0                  # a comparison that zeroes the score also rejects
        if accepted and fails:
            bad.append(f"instance {i} ({lab}): the evaluator ACCEPTED this output (score {sc:.6g}), but your checks "
                       "say it fails: " +
                       "; ".join(f"{c['name']}: value {c['value']:.9g} {c['sense']} limit {c['limit']:.9g} is false"
                                 for c in fails[:3]) + " -- the quantity or the limit is computed wrongly")
        elif not accepted and r.get("failed_check"):
            # only a named check is a claim (failed_check itself is never shown); other labels are ignored
            named = [c for c in checks if c["name"] == r["failed_check"]]
            if named and _passes(named[0]):
                c = named[0]
                bad.append(f"instance {i} ({lab}): the evaluator REJECTED this output and you name {c['name']!r} as "
                           f"the failed check, but value {c['value']:.9g} {c['sense']} limit {c['limit']:.9g} holds"
                           " -- the quantity or the limit is computed wrongly")
    if bad:
        return "the checks are inconsistent with the evaluator:\n" + "\n".join(bad[:6])
    mean = sum(float(r["score"]) for r in rows) / len(rows)
    if abs(mean - combined) > MEAN_TOL * max(1.0, abs(combined)):
        return f"mean of diagnose() scores {mean:.10g} differs from the evaluator's combined_score {combined:.10g}"
    if any(not str(r.get("report") or "").strip() for r in rows):
        return "some instances have an empty report"
    for i, (r, (lab, st, sc)) in enumerate(zip(rows, ref)):
        m = ERR_RE.search(str(r.get("report") or ""))
        if st == "ok" and sc > 0 and m:      # the score was right but the analysis after it crashed
            return (f"instance {i} ({lab}): the evaluator accepted this output, but your report contains an error "
                    f"({m.group(0)!r}): {str(r['report'])[:200]!r} -- the analysis code fails; fix it")
    return None


# ---------------------------------------------------------------- running
def run(diag_path: str, evaluator_path: str, program_code: str, timeout: float) -> Tuple[Optional[list], str, float]:
    """Run diagnose() on one program in a fresh subprocess. Returns (rows | None, error, seconds)."""
    t0 = time.time()
    with tempfile.TemporaryDirectory(prefix="diag_") as td:
        prog = os.path.join(td, "program.py")
        out = os.path.join(td, "out.json")
        with open(prog, "w") as fh:
            fh.write(program_code)
        env = dict(os.environ, EVAL_PARALLEL=str(PARALLEL))   # leave the CPUs to the real evaluations
        try:
            p = subprocess.run([sys.executable, RUNNER, diag_path, evaluator_path, prog, out], env=env,
                               capture_output=True, text=True, timeout=timeout, start_new_session=True)
        except subprocess.TimeoutExpired:
            return None, f"diagnose() did not finish within {timeout:.0f}s", time.time() - t0
        if p.returncode != 0 or not os.path.exists(out):
            tail = (p.stderr or p.stdout or "").strip()[-700:]
            return None, f"diagnose() failed: {tail or f'exit code {p.returncode}'}", time.time() - t0
        try:
            rows = json.load(open(out))
        except Exception as e:
            return None, f"diagnose() output unreadable: {e}", time.time() - t0
    if not isinstance(rows, list) or not all(isinstance(r, dict) for r in rows):
        return None, "diagnose() must return a list of dicts", time.time() - t0
    for r in rows:
        r["report"] = str(r.get("report") or "")[:REPORT_CHARS]
    return rows, "", time.time() - t0


def _same(a: Optional[list], b: Optional[list]) -> bool:
    if not a or not b or len(a) != len(b):
        return False
    return all(abs(float(x.get("score") or 0) - float(y.get("score") or 0)) <= REL_TOL * max(1.0, abs(float(x.get("score") or 0)))
               for x, y in zip(a, b))


def check(diag_path: str, evaluator_path: str, code: str, feedback: str, combined: float,
          timeout: float) -> Tuple[Optional[list], Optional[str], bool, float]:
    """(rows, error, nondeterministic, seconds). On a mismatch the program is run once more: if the two runs
    disagree with each other the program is non-deterministic (unseeded randomness, time-based search) and the
    mismatch says nothing about the diagnoser."""
    rows, err, dt = run(diag_path, evaluator_path, code, timeout)
    err = err or compare(rows, feedback, combined)
    if err is None or rows is None:
        return rows, err, False, dt
    rows2, err2, dt2 = run(diag_path, evaluator_path, code, timeout)
    return rows, err, not _same(rows, rows2), dt + dt2


def _checks_text(checks: list) -> str:
    """Slack lines written here, from verified check values -- never by the generated module."""
    if not checks:
        return ""
    parts = []
    for c in sorted(checks, key=lambda c: _slack(c) / max(1e-12, abs(float(c["limit"])) or 1.0))[:4]:
        s = _slack(c)
        state = "AT ITS LIMIT" if abs(s) <= CHECK_TOL * max(1.0, abs(float(c["limit"]))) else (
            f"slack {s:.4g}" if s > 0 else f"VIOLATED by {-s:.4g}")
        parts.append(f"{c['name']} = {float(c['value']):.6g} (must be {c['sense']} {float(c['limit']):.6g}: {state})")
    return " | checks: " + "; ".join(parts)


def render(rows: list) -> str:
    """The diagnosis section of a prompt: every instance when few, else the lowest-scoring SHOW."""
    if len(rows) <= SHOW + 1:
        pick = list(enumerate(rows))
        head = ""
    else:
        pick = sorted(enumerate(rows), key=lambda t: float(t[1].get("score") or 0.0))[:SHOW]
        pick.sort()
        mean = sum(float(r.get("score") or 0) for r in rows) / len(rows)
        head = f"{len(rows)} training instances, mean score {mean:.6g}; the {SHOW} lowest-scoring:\n"
    lines = [f"- {r.get('label') or f'instance {i}'} (score {float(r.get('score') or 0):.6g}): {r['report']}"
             + _checks_text(r.get("checks") or []) for i, r in pick]
    return (head + "\n".join(lines))[:DIAG_CHARS]


# ---------------------------------------------------------------- the diagnoser over a run
class Diagnoser:
    """Builds, verifies and runs the generated diagnoser. State lives here; the controller owns scheduling."""

    def __init__(self, evaluator_path: str, workdir: str, llm: Callable[[str], Awaitable[Optional[str]]],
                 build_prompt: Callable[..., str], timeout: float, log: Callable[..., None] = lambda *a, **k: None):
        self.evaluator_path = evaluator_path
        self.workdir = workdir
        self.llm = llm
        self.build_prompt = build_prompt       # (sources, previous_code, failure) -> user message
        self.timeout = timeout
        self.log = log
        self.sources = gather_sources(evaluator_path)
        self.path: Optional[str] = None        # current diagnoser module
        self.version = 0
        self.verified: set = set()             # distinct nonzero scores reproduced by the current version
        self.misses: List[bool] = []           # recent reproduce failures of the current version (True = miss)
        self.rebuilds = 0
        self.seconds = 0.0                     # total time spent running diagnoses
        self.cache: Dict[str, Dict[str, Any]] = {}   # program id -> {"version", "rows", "ok", "error"}
        self.last_failure = ""

    @property
    def trusted(self) -> bool:
        return self.path is not None and len(self.verified) >= TRUST_MIN

    async def build(self, refs: List[Tuple[str, str, str, float]], previous: str = "", failure: str = "") -> bool:
        """Generate a diagnoser; keep the first that passes the static check and reproduces every ref
        (program_id, code, feedback, combined_score). refs may be empty (then only the static check and a run
        on nothing). Returns True if a module was installed."""
        code_prev, why = previous, failure
        for k in range(BUILD_TRIES):
            text = await self.llm(self.build_prompt(self.sources, code_prev, why))
            code = extract_code(text)
            if code is None:
                why, code_prev = "no ```python fenced module found in the answer", code_prev
                continue
            err = static_check(code)
            if err is None:
                path = os.path.join(self.workdir, f"diagnoser_v{self.version + 1}.py")
                with open(path, "w") as fh:
                    fh.write(code)
                passed = set()
                for pid, pcode, fb, comb in refs:
                    rows, err, nondet, dt = await asyncio.to_thread(check, path, self.evaluator_path, pcode, fb, comb,
                                                                    self.timeout)
                    self.seconds += dt
                    if err and nondet:                # says nothing about the module: skip this reference
                        err = None
                        continue
                    if err:
                        break
                    if comb > 0:
                        passed.add(round(comb, 9))
                if err is None:
                    self.version += 1
                    self.path = path
                    self.verified = passed
                    self.misses = []
                    self.cache.clear()
                    self.log("diag_build", version=self.version, tries=k + 1, refs=len(refs), path=path)
                    return True
            self.log("diag_reject", try_=k + 1, error=err[:600])
            code_prev, why = code, err
        self.last_failure = why
        return False

    def diagnose(self, pid: str, code: str, feedback: str, combined: float) -> Optional[str]:
        """Run (cached) and verify. Returns the rendered diagnosis if verified AND the diagnoser is trusted."""
        if self.path is None:
            return None
        c = self.cache.get(pid)
        if c is None or c["version"] != self.version:
            rows, err, nondet, dt = check(self.path, self.evaluator_path, code, feedback, combined, self.timeout)
            self.seconds += dt
            c = {"version": self.version, "rows": rows, "ok": err is None, "error": err, "nondet": nondet,
                 "seconds": dt}
            self.cache[pid] = c
            if err is None and combined > 0:
                self.verified.add(round(combined, 9))
            if not nondet:
                self.misses = (self.misses + [err is not None])[-5:]
            self.log("diag_run", program=pid, version=self.version, ok=err is None, nondet=nondet,
                     seconds=round(dt, 1), error=(err or "")[:300], trusted=self.trusted)
        return render(c["rows"]) if c["ok"] and self.trusted else None

    def needs_rebuild(self) -> bool:
        return self.path is not None and sum(self.misses) >= 2

    def miss_evidence(self) -> str:
        bad = [c for c in self.cache.values() if c["version"] == self.version and not c["ok"] and not c["nondet"]]
        return bad[-1]["error"] if bad else ""
