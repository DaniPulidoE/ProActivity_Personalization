"""Stated preference vs revealed selections -- the two-panel scatter.

Reproduces the figure in data_analysis/analysis_study2.qmd section 4: each
driver is drawn as their own id rather than a marker, so the tie structure and
any outlier are both readable from twelve points. Palette and theme constants
are copied from the analysis file so the figure matches the rest of the study.
"""
import numpy as np, pandas as pd, pathlib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SERIES_BLUE, SURFACE = "#2a78d6", "#fcfcfb"
INK, INK_2, INK_MUTED, GRID = "#0b0b0b", "#52514e", "#8f8e88", "#e6e5e1"

DEMOG = "data/study2_data/demographics_questionnaire_study2.csv"
LABELS = "data/study1_data/user_loa_labels.csv"
PHONE_FN = "Respond to a phone call"

d = pd.read_csv(DEMOG, dtype=str).iloc[2:]        # drop question text + ImportId rows
demog = pd.DataFrame({
    "participantid": d.Q1.str.strip().str.zfill(3),
    "pref_loa": d.Q2.str.extract(r"Level (\d)")[0].astype(float) - 1,
    "context_dep": d.Q5_1.astype(float),
})

lab = pd.read_csv(LABELS, dtype={"participantid": str})
lab = lab[lab.functionname == PHONE_FN].copy()
lab["win_loa"] = lab.user_selected_loa.astype(str).map(
    lambda s: np.mean([float(x) for x in s.split(";") if x.strip() != ""]))
beh = lab.groupby("participantid").win_loa.agg(
    loa_mean="mean", loa_var="var").reset_index()

P = demog.merge(beh, on="participantid").sort_values("participantid")
assert len(P) == 12, f"expected 12 drivers, got {len(P)}"

PANELS = [
    ("pref_loa", "loa_mean",
     "Stated preferred level of proactivity", "Mean level selected in study 1"),
    ("context_dep", "loa_var",
     "Agreement that proactivity should depend on context",
     "Variance of levels selected in study 1"),
]

plt.rcParams.update({"pdf.fonttype": 42, "font.size": 9})
fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.7))
fig.patch.set_facecolor(SURFACE)

for ax, (xc, yc, xlab, ylab) in zip(axes, PANELS):
    ax.set_facecolor(SURFACE)
    x, y = P[xc].to_numpy(float), P[yc].to_numpy(float)
    # Least-squares line, drawn only to indicate direction -- the association is
    # reported as Kendall's tau_b, not as a linear fit.
    b, a = np.polyfit(x, y, 1)
    xs = np.linspace(x.min(), x.max(), 2)
    ax.plot(xs, a + b * xs, ls=(0, (4, 2)), lw=0.8, color=INK_MUTED, zorder=1)
    # The questionnaire items take few distinct values, so several drivers share
    # an x. Where two of them are also close in y their labels would overlap and
    # neither would be readable, so such pairs are pushed to opposite sides. The
    # nudge is horizontal only and never crosses to another x value, so no point
    # can be misread as belonging to a different answer.
    xr, yr = (x.max() - x.min()) or 1.0, (y.max() - y.min()) or 1.0
    xd = x.copy()
    for xv in np.unique(x):
        col = np.where(x == xv)[0]
        col = col[np.argsort(y[col])]
        i = 0
        while i < len(col) - 1:
            a, b = col[i], col[i + 1]
            if abs(y[b] - y[a]) < 0.06 * yr:
                xd[a] -= 0.035 * xr
                xd[b] += 0.035 * xr
                i += 2
            else:
                i += 1

    for xi, yi, pid in zip(xd, y, P.participantid):
        ax.text(xi, yi, pid.lstrip("0"), ha="center", va="center",
                fontsize=8, color=SERIES_BLUE, zorder=2)
    pad_x = 0.08 * (x.max() - x.min() or 1)
    pad_y = 0.12 * (y.max() - y.min() or 1)
    ax.set_xlim(x.min() - pad_x, x.max() + pad_x)
    ax.set_ylim(y.min() - pad_y, y.max() + pad_y)
    ax.set_xlabel(xlab, color=INK_2)
    ax.set_ylabel(ylab, color=INK_2)
    ax.set_xticks(sorted(set(x)))
    ax.grid(axis="y", color=GRID, lw=0.4)
    ax.set_axisbelow(True)
    ax.tick_params(length=0, colors=INK_2)
    for s in ("top", "right", "left", "bottom"):
        ax.spines[s].set_visible(False)

fig.tight_layout()
out = pathlib.Path("latex/figures/stated_vs_revealed.pdf")
out.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out, format="pdf", bbox_inches="tight", facecolor=SURFACE)
print(f"[fig] {out}")
print(f"[check] {len(P)} drivers; "
      f"preferred level takes values {sorted(set(P.pref_loa.astype(int)))}, "
      f"modal answer given by {P.pref_loa.value_counts().max():.0f} of {len(P)}")
