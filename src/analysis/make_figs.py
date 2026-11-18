"""Tables and figures for the report from analysis/report_runs.jsonl (made on the cluster by analysis/report_data.py).

  python report/make_figs.py            # writes report/figs/*.pdf, report/tables/*.tex, prints a summary

Rules (stated in the report):
- prompt versions are named v1, v2, v3 in the report (repo folders prompts/v5, v7, v8); each task is reported on every
  version it was run on, and the headline is the average over versions (each version weighted equally);
- figure: per version the mean (std) curve over seeds, then averaged over versions; generalisation and compute
  tables: every run of every version pooled;
- ablations run on one version per task (ABLV, the version of the best BANYAN seed, chosen before the ablations);
- AdaEvolve v1 prism seeds 0 and 1 used the GPU-count hack before the course patch; their re-runs on the patched kit
  (cohort v5fixed) replace them;
- duplicate seed folders (EvoX v1 prism s1, cloudcast s0, signal s0): keep the run with more evaluations;
- runs without a selection.json (still running) are left out of every table and figure.
"""
import json
import os

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROWS = [json.loads(l) for l in open(os.path.join(HERE, "..", "analysis", "report_runs.jsonl")) if l.strip()]
os.makedirs(os.path.join(HERE, "figs"), exist_ok=True)
os.makedirs(os.path.join(HERE, "tables"), exist_ok=True)

TASKS = ["prism", "cloudcast", "signal_processing", "heilbronn_triangle", "circle_packing_rect"]
TNAME = {"prism": "prism", "cloudcast": "cloudcast", "signal_processing": "signal", "heilbronn_triangle": "heilbronn",
         "circle_packing_rect": "circle"}
PV = {"v5": "v1", "v7": "v2", "v8": "v3"}   # repo prompt folder -> name in the report
ABLV = {"prism": "v5", "cloudcast": "v8", "signal_processing": "v7", "heilbronn_triangle": "v7", "circle_packing_rect": "v5"}
B0 = {"prism": 21.75, "cloudcast": 0.00526, "signal_processing": 0.511, "heilbronn_triangle": 0.0, "circle_packing_rect": 0.0}
ARMS = ["banyan", "evox", "adaevolve"]
ANAME = {"banyan": "BANYAN", "evox": "EvoX", "adaevolve": "AdaEvolve"}
COLOR = {"banyan": "#2a78d6", "evox": "#eb6834", "adaevolve": "#1baf7a"}   # categorical slots 1-3 (dataviz palette)
ABL = [("abl_ref", "reference"), ("abl_p2off", "A1 standing tree"), ("abl_random", "A3 random alloc."),
       ("abl_nodiag", "A4 no diagnoser"), ("abl_flat", "A5 flat tree")]
# the ablation reference per task: BANYAN's runs on ABLV; on signal all 6 BANYAN v2 runs (the 3 of the main table,
# resumed after the 15 Nov outage within their 5 h search clock, and 3 fresh re-runs)
REF = {t: [c] for t, c in ABLV.items()}
REF["signal_processing"] = ["v7", "abl_ref"]


def done(r):
    return r["sel"].get("val") is not None


def runs(arm, cohort, task):
    """One run per seed under the rules above."""
    if arm == "adaevolve" and cohort == "v5":
        fixed = {r["seed"]: r for r in ROWS if r["arm"] == arm and r["cohort"] == "v5fixed" and r["task"] == task and done(r)}
    else:
        fixed = {}
    by = {}
    for r in ROWS:
        if r["arm"] == arm and r["cohort"] == cohort and r["task"] == task and done(r):
            if r["seed"] not in by or len(r["scores"]) > len(by[r["seed"]]["scores"]):
                by[r["seed"]] = r
    by.update(fixed)
    return [by[k] for k in sorted(by)]


# prompt versions each task was run on (by BANYAN; the baselines ran on the same ones)
VERS = {t: [c for c in PV if runs("banyan", c, t)] for t in TASKS}
# BANYAN v2 signal: the 3 runs resumed after the 15 Nov outage plus 3 fresh re-runs on the same prompts (6 seeds)
EXTRA = {("banyan", "v7", "signal_processing"): ["abl_ref"]}


def vruns(arm, cohort, task):
    """Every run reported for one arm under one prompt version."""
    return runs(arm, cohort, task) + [r for c in EXTRA.get((arm, cohort, task), []) for r in runs(arm, c, task)]


def ms(xs):
    xs = [x for x in xs if x is not None]
    if not xs:
        return None, None, 0
    return float(np.mean(xs)), float(np.std(xs, ddof=1)) if len(xs) > 1 else 0.0, len(xs)


def fnum(x, task):
    if x is None:
        return "--"
    if task == "prism":
        return f"{x:.2f}"
    if task == "cloudcast":
        return f"{x * 1000:.3f}"   # reported in units of 1e-3
    return f"{x:.3f}"


def cell(xs, task, bold=False):
    m, s, n = ms(xs)
    if m is None:
        return "--"
    body = f"{fnum(m, task)} $\\pm$ {fnum(s, task)}"
    return (f"\\textbf{{{fnum(m, task)}}} $\\pm$ {fnum(s, task)}" if bold else body) + (f"$^{{({n})}}$" if n != 3 else "")


def gain_txt(b, e):
    return f"{(b - e) / abs(e) * 100:+.1f}" + "\\%"


def write(name, text):
    open(os.path.join(HERE, "tables", name), "w").write(text)


# ---------------------------------------------------------------- Table: main comparison, every version + average
lines, avg = [], {}
for task in TASKS:
    vmeans = {a: [] for a in ARMS}
    for i, c in enumerate(VERS[task]):
        vals = {a: [r["sel"]["val"] for r in vruns(a, c, task)] for a in ARMS}
        means = {a: ms(v)[0] for a, v in vals.items()}
        for a in ARMS:
            vmeans[a].append(means[a])
        top = "banyan" if means["banyan"] >= means["evox"] else "evox"
        lines.append(f"{TNAME[task] if i == 0 else ''} & {PV[c]} & {cell(vals['banyan'], task, top == 'banyan')} & "
                     f"{cell(vals['evox'], task, top == 'evox')} & {gain_txt(means['banyan'], means['evox'])} & "
                     f"{cell(vals['adaevolve'], task)} \\\\")
    avg[task] = {a: float(np.mean(v)) for a, v in vmeans.items()}
    if len(VERS[task]) > 1:
        m = avg[task]
        b, e = (f"\\textbf{{{fnum(m[a], task)}}}" if m[a] == max(m["banyan"], m["evox"]) else fnum(m[a], task)
                for a in ("banyan", "evox"))
        lines.append(f"\\cmidrule(l){{2-6}} & average & {b} & {e} & {gain_txt(m['banyan'], m['evox'])} & "
                     f"{fnum(m['adaevolve'], task)} \\\\")
    lines.append("\\midrule")
write("main.tex", "\n".join(lines[:-1]) + "\n")

# ---------------------------------------------------------------- Figure: best-so-far train score vs evaluations
E = 500


def curve(r):
    s = np.maximum.accumulate(np.maximum(np.array(r["scores"][:E], float), B0[r["task"]]))
    return np.concatenate([s, np.full(E - len(s), s[-1])]) if len(s) < E else s   # carry forward


fig, axs = plt.subplots(2, 3, figsize=(7.0, 3.3))
axes = axs.ravel()
for ax, task in zip(axes, TASKS):
    for arm in ARMS:
        mcs, scs, allr = [], [], []
        for c in VERS[task]:
            rs = vruns(arm, c, task)
            if not rs:
                continue
            cs = np.array([curve(r) for r in rs])
            mcs.append(cs.mean(0))
            scs.append(cs.std(0, ddof=1) if len(cs) > 1 else np.zeros(E))
            allr += rs
        m, sd = np.mean(mcs, 0), np.mean(scs, 0)   # each prompt version weighted equally
        x = np.arange(1, E + 1)
        ax.plot(x, m, color=COLOR[arm], lw=1.6, label=ANAME[arm])
        lo, hi = m - sd, m + sd
        if task in ("heilbronn_triangle", "circle_packing_rect"):
            lo, hi = np.clip(lo, 0, 1), np.clip(hi, 0, 1)
        ax.fill_between(x, lo, hi, color=COLOR[arm], alpha=0.15, lw=0)
        for r in allr:   # where each run stopped
            n = min(len(r["scores"]), E)
            ax.plot([n], [m[n - 1]], marker="|", color=COLOR[arm], ms=7, mew=1.4)
    ax.set_title(TNAME[task], fontsize=9)
    ax.set_xlabel("evaluations", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.grid(alpha=0.25, lw=0.5)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    if task == "prism":
        ax.set_ylim(21.5, 26.5)
        ax.axhline(26.1802, color="#52514e", lw=0.8, ls=":")
    if task == "signal_processing":
        ax.set_ylim(0.45, 0.85)
for ax in (axes[0], axes[3]):
    ax.set_ylabel("best train score", fontsize=8)
for ax in axes[:2]:
    ax.set_xlabel("")
axes[5].axis("off")
axes[5].legend(*axes[0].get_legend_handles_labels(), fontsize=9, frameon=False, loc="center")
fig.tight_layout(w_pad=0.8, h_pad=0.6)
fig.savefig(os.path.join(HERE, "figs", "curves.pdf"))
plt.close(fig)

# ---------------------------------------------------------------- Table: train vs val, valid rate, compute (all runs)
gen, comp = [], []
f2 = lambda xs, p=2: f"{ms(xs)[0]:.{p}f} $\\pm$ {ms(xs)[1]:.{p}f}"
for task in TASKS:
    for arm in ARMS:
        rs = [r for c in VERS[task] for r in vruns(arm, c, task)]
        if not rs:
            continue
        tr = [r["sel"]["train"] for r in rs]
        va = [r["sel"]["val"] for r in rs]
        valid = [np.mean(np.array(r["scores"]) > 0) for r in rs]
        n = [len(r["scores"]) for r in rs]
        name = TNAME[task] if arm == "banyan" else ""
        aname = ANAME[arm] + ("$^\\dagger$" if arm == "banyan" and task == "signal_processing" else "")
        mtr, str_, k = ms(tr)
        mva, sva, _ = ms(va)
        gen.append(f"{name} & {ANAME[arm]} & {k} & {fnum(mtr, task)} $\\pm$ {fnum(str_, task)} & "
                   f"{fnum(mva, task)} $\\pm$ {fnum(sva, task)} & {f2(valid)} & {f2(n, 0)} \\\\")
        # the v2 BANYAN signal runs were resumed after the 15 Nov outage (two jobs each); their compute is not a
        # run's compute, so this table uses the fresh BANYAN signal seeds on the same prompts (ablation reference)
        if arm == "banyan" and task == "signal_processing":
            rs = runs(arm, "v5", task) + runs(arm, "abl_ref", task)
            n = [len(r["scores"]) for r in rs]
        wall = [r["wall"] / 3600 for r in rs if r["wall"]]
        calls = [r["llm_calls"] for r in rs]
        pt = [r["prompt_tok"] / 1e6 for r in rs if r["prompt_tok"]]
        ct = [r["completion_tok"] / 1e6 for r in rs if r["completion_tok"]]
        comp.append(f"{name} & {aname} & {f2(wall, 1)} & {f2(n, 0)} & {f2(calls, 0)} & {f2(pt, 1)} & {f2(ct, 2)} \\\\")
    gen.append("\\midrule")
    comp.append("\\midrule")
write("gen.tex", "\n".join(gen[:-1]) + "\n")
write("compute.tex", "\n".join(comp[:-1]) + "\n")

# ---------------------------------------------------------------- Table: ablations (val) + mechanism
def groups(cohort, task):
    """Groups of seeds (each a list of runs) for an ablation column; the reference may pool several groups."""
    return [runs("banyan", c, task) for c in (REF[task] if cohort == "abl_ref" else [cohort])]


def best_of_3(gs):
    """Best val of a group of 3 seeds; with several groups, the average of their bests (an estimate of best-of-3)."""
    return float(np.mean([max(r["sel"]["val"] for r in g) for g in gs if g]))


abl, mech = [], []
for task in TASKS:
    mrow, brow = [f"{TNAME[task]} ({PV[ABLV[task]]})", "mean"], ["", "best"]
    for cohort, _ in ABL:
        gs = groups(cohort, task)
        vals = [r["sel"]["val"] for g in gs for r in g]
        mrow.append(cell(vals, task) if vals else "--")
        mark = "$^\\ddagger$" if len([g for g in gs if g]) > 1 else ""
        brow.append(fnum(best_of_3(gs), task) + mark if vals else "--")
    abl.append(" & ".join(mrow) + " \\\\")
    abl.append(" & ".join(brow) + " \\\\")
    abl.append("\\midrule")
    for cohort, label in ABL:
        rs = [r for g in groups(cohort, task) for r in g if r["mech"]]
        if not rs:
            continue
        g = lambda k: f"{np.mean([r['mech'][k] for r in rs]):.2f}"
        mech.append(f"{TNAME[task] if cohort == 'abl_ref' else ''} & {label} & {len(rs)} & {g('inc')} & {g('trunk')} & "
                    f"{np.mean([r['mech']['reroots'] for r in rs]):.1f} & {np.mean([r['mech']['rec'] for r in rs]):.1f} & "
                    f"{np.mean([r['mech']['F'] for r in rs]):.1f} & {g('diag')} \\\\")
    mech.append("\\midrule")
write("abl.tex", "\n".join(abl[:-1]) + "\n")
write("mech.tex", "\n".join(mech[:-1]) + "\n")

# ---------------------------------------------------------------- summary to stdout
for task in TASKS:
    print(f"{TNAME[task]:9s} versions={[PV[c] for c in VERS[task]]}  average: " +
          "  ".join(f"{a}={avg[task][a]:.6g}" for a in ARMS) +
          f"  gain={(avg[task]['banyan'] - avg[task]['evox']) / abs(avg[task]['evox']) * 100:+.1f}%")
print()
for task in TASKS:
    ref = [r["sel"]["val"] for g in groups("abl_ref", task) for r in g]
    for cohort, label in ABL:
        gs = groups(cohort, task)
        v = [r["sel"]["val"] for g in gs for r in g]
        if not v:
            continue
        wins = sum(a > b for a in ref for b in v) if cohort != "abl_ref" else "-"
        print(f"{TNAME[task]:9s} {label:18s} n={len(v)} mean={np.mean(v):.6g} best3={best_of_3(gs):.6g} "
              f"ref-wins={wins}/{len(ref) * len(v)} val={[round(x, 5) for x in v]}")
