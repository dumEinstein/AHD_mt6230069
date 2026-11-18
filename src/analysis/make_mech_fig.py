"""Schematic of BANYAN's mechanism (a static version of the dashboard's Theory figure) -> figs/mechanism.pdf.

(a) is an illustrative trajectory, not run data: the incumbent's score is the bar every evaluation must beat; the
trunk edits the incumbent, branches are grafted at the bar, and a branch that crosses the bar becomes the trunk.
(b) is the Thompson-draw example of Sec. 3.2: three Gamma posteriors and one draw from each.
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from math import lgamma

HERE = os.path.dirname(os.path.abspath(__file__))
plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
                     "mathtext.fontset": "stix", "font.size": 8, "axes.linewidth": 0.6,
                     "xtick.major.width": 0.6, "ytick.major.width": 0.6})
TRUNK, B1, B2, B3, FROZEN, BAR = "#2a78d6", "#eb6834", "#8c5bd6", "#1baf7a", "#b9b7b0", "#2b2a27"


def best_so_far(graft, pts):
    """Step line of a branch's best score: it starts at the bar where it was grafted, then follows its own records."""
    xs, ys, best = [graft[0]], [graft[1]], None
    for t, s in pts:
        if best is None or s > best:
            xs += [t, t] if best is not None else [t]
            ys += [best, s] if best is not None else [s]
            best = s
    return xs, ys


fig, (ax, bx) = plt.subplots(1, 2, figsize=(7.0, 1.85), gridspec_kw={"width_ratios": [1.45, 1]})

# ---- (a) trunk, branches and a re-root -------------------------------------------------------------------------
trunk_pts = [(0, .42), (4, .47), (9, .52), (15, .55), (22, .58), (30, .60), (38, .60), (50, .60), (66, .60)]
lp_pts = [(41, .46), (47, .53), (55, .575), (66, .63)]          # branch that sets a record at t=66 -> re-root
an_pts = [(52, .49), (58, .535), (64, .55)]                      # open branch, frozen at the re-root
new_trunk = [(66, .63), (76, .645), (90, .655), (104, .66), (130, .66)]
dp_pts = [(112, .57), (119, .61), (126, .64)]                    # grafted on the new incumbent

bar_t = [0, 4, 9, 15, 22, 30, 66, 76, 90, 104, 140]
bar_s = [.42, .47, .52, .55, .58, .60, .63, .645, .655, .66, .66]
ax.step(bar_t, bar_s, where="post", color=BAR, lw=2.4, alpha=.18, zorder=1)
ax.text(1, .675, "bar to beat: the incumbent's score $G$", fontsize=7, color="#52514e")

x, y = best_so_far((0, .42), trunk_pts)
ax.plot(x[:2] + x[2:], y, color=TRUNK, lw=1.5, zorder=3)
ax.plot([66, 140], [.60, .60], color=FROZEN, lw=1.2, ls="--", zorder=2)
ax.text(13, .50, "trunk: edits of $I$", color=TRUNK, fontsize=7)

x, y = best_so_far((40, .60), lp_pts)
ax.plot(x, y, color=B1, lw=1.3, zorder=3)
ax.text(39, .425, "branch: LP relaxation", color=B1, fontsize=7)
ax.text(58.5, .472, "anneal", color=B2, fontsize=7)
x, y = best_so_far((51, .60), an_pts)
ax.plot(x, y, color=B2, lw=1.1, zorder=3)
ax.plot([64, 140], [.55, .55], color=FROZEN, lw=1.0, ls="--", zorder=2)
ax.text(139, .532, "frozen at the re-root", color="#8a8882", fontsize=7, ha="right")

ax.plot([t for t, _ in new_trunk], [s for _, s in new_trunk], color=TRUNK, lw=1.5, zorder=3, drawstyle="steps-post")
ax.text(80, .668, "new trunk", color=TRUNK, fontsize=7)
x, y = best_so_far((111, .66), dp_pts)
ax.plot(x, y, color=B3, lw=1.1, zorder=3)
ax.text(127.5, .625, "new\ngraft", color=B3, fontsize=7, va="center")

for g in [(40, .60), (51, .60), (111, .66)]:
    ax.plot(*g, marker="v", ms=4, color=BAR, zorder=4)
ax.plot(66, .63, marker="o", ms=5.5, mfc="white", mec=B1, mew=1.4, zorder=5)
ax.annotate("re-root: the branch\nbeats $G$ and\nbecomes the trunk", xy=(66, .63), xytext=(80, .525), fontsize=7,
            arrowprops=dict(arrowstyle="-", lw=.6, color="#52514e"), va="top")
ax.text(41, .612, "graft", fontsize=6.5, color="#52514e", ha="center")
ax.set_xlim(0, 140); ax.set_ylim(.40, .69)
ax.set_xlabel("evaluations"); ax.set_ylabel("best score so far")
ax.set_title("(a) incumbent-rooted tree (illustration)", fontsize=8)
ax.spines[["top", "right"]].set_visible(False)

# ---- (b) Thompson draw over Gamma posteriors -------------------------------------------------------------------
def gamma_pdf(x, k, rate):
    return np.exp(k * np.log(rate) + (k - 1) * np.log(x) - rate * x - lgamma(k))

xs = np.linspace(1e-3, 4, 400)
rows = [("productive: Gamma(6, 3)", 6, 3, TRUNK, 2.35, "wins 84%"),
        ("untried: Gamma(2, 2)", 2, 2, "#52514e", 1.55, "wins 16%"),
        ("exhausted: Gamma(2, 6)", 2, 6, B1, 0.22, "wins 0.4%")]
for i, (lab, k, rate, col, draw, win) in enumerate(rows):
    base = 2 - i
    d = gamma_pdf(xs, k, rate)
    d = d / d.max() * .78
    bx.fill_between(xs, base, base + d, color=col, alpha=.18 if i else .3, lw=0)
    bx.plot(xs, base + d, color=col, lw=1.0)
    bx.plot([0, 4], [base, base], color="#d6d4cd", lw=.6)
    bx.plot([draw, draw], [base, base + .9], color=col, lw=1.1)
    bx.plot(draw, base + .9, "o", ms=4.5 if i else 5.5, color=col)
    bx.text(4.0, base + .55, lab, fontsize=6.8, ha="right", color=col)
    bx.text(4.0, base + .28, win, fontsize=6.5, ha="right", color="#52514e")
bx.axvline(1, color="#52514e", lw=.6, ls=":")
bx.text(1.04, 2.93, r"$\rho=1$", fontsize=6.5, color="#52514e")
bx.text(2.52, 2.93, "highest draw: takes the block", fontsize=6.5, color=TRUNK)
bx.set_xlim(0, 4); bx.set_ylim(-.05, 3.05)
bx.set_yticks([])
bx.set_xlabel(r"credibility $\rho$ (successes / expected)")
bx.set_title(r"(b) one Thompson draw per option", fontsize=8)
bx.spines[["top", "right", "left"]].set_visible(False)

fig.tight_layout(pad=.3, w_pad=1.2)
out = os.path.join(HERE, "figs", "mechanism.pdf")
fig.savefig(out)
fig.savefig(out.replace(".pdf", ".png"), dpi=200)
print("wrote", out)
