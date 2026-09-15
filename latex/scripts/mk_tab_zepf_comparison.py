"""Per-function autonomy levels: this cohort against Zepf et al.'s online survey.

Five of the fourteen functions in Zepf et al. are also in this study's prompt
pool, so the two elicitations can be put side by side function by function.

The Zepf values are READ OFF Fig. 2 of the paper (boxplots of user autonomy
level ratings per function) -- that figure is the only per-function breakdown
they publish, so medians and quartiles are the finest grain available, and they
are exact only to the resolution of the printed axis. They are transcribed here
rather than recomputed, and are marked as such in the caption.

This study's side is recomputed from the raw label file. A driver may mark a
SET of acceptable levels; each marked level enters as one observation, which is
the reading that matches "which levels would be acceptable here". The set-mean
alternative is checked below and moves no median.
"""
import numpy as np, pandas as pd, pathlib

LABELS = "data/study1_data/user_loa_labels.csv"

# Zepf et al., Fig. 2 -- (Q1, median, Q3), read from the printed boxplots.
# Keyed by their function label; mapped to this study's prompt names.
ZEPF = {
    "Phone call":            (0.0, 1.0, 2.0),
    "Send text message":     (0.0, 1.0, 2.0),
    "Change song":           (0.0, 1.0, 2.0),
    "Provide weather update":(1.5, 4.0, 4.0),
    "Switch to traffic news":(1.0, 2.0, 3.0),
}
# this study's functionname  ->  (Zepf's label, display name)
SHARED = [
    ("Respond to a phone call",   "Phone call",             "Phone call"),
    ("Respond to a text message", "Send text message",      "Text message"),
    ("Change song",               "Change song",            "Change song"),
    ("Provide weather update",    "Provide weather update", "Weather update"),
    ("Provide traffic news",      "Switch to traffic news", "Traffic news"),
]

d = pd.read_csv(LABELS, dtype={"participantid": str})
d = d[d.user_selected_loa.notna()].copy()


def levels(s):
    return [float(x) for x in str(s).split(";") if x.strip() != ""]


rows = []
for _, r in d.iterrows():
    for v in levels(r.user_selected_loa):
        rows.append((r.functionname, r.participantid, v))
E = pd.DataFrame(rows, columns=["fn", "pid", "loa"])

d["setmean"] = d.user_selected_loa.map(levels).map(np.mean)

out_rows = []
for fn, zkey, disp in SHARED:
    g = E[E.fn == fn]
    q1, med, q3 = np.percentile(g.loa, [25, 50, 75])
    # Sensitivity: collapsing each window's set to its mean must not move the
    # median, or the expansion is doing the work rather than the data.
    med_sm = np.median(d.loc[d.functionname == fn, "setmean"])
    assert abs(med_sm - med) < 1e-9, f"{fn}: median moves under set-mean ({med_sm} vs {med})"
    pm = g.groupby("pid").loa.mean()
    assert len(pm) == 12, f"{fn}: {len(pm)} drivers, expected 12"
    out_rows.append(dict(
        disp=disp, n=len(g),
        zq1=ZEPF[zkey][0], zmed=ZEPF[zkey][1], zq3=ZEPF[zkey][2],
        q1=q1, med=med, q3=q3,
        extreme=100 * g.loa.isin([0, 4]).mean(),
        dsd=pm.std(ddof=1), dlo=pm.min(), dhi=pm.max(),
    ))

DF = pd.DataFrame(out_rows)

# The headline claim the table has to support: the two information functions
# sit above the two communication functions in BOTH studies, while this
# cohort's dispersion is the wider one on every shared function.
comm = DF[DF.disp.isin(["Phone call", "Text message"])]
info = DF[DF.disp.isin(["Weather update", "Traffic news"])]
assert info.zmed.min() >= comm.zmed.max(), "Zepf: information not above communication"
assert info.med.min() >= comm.med.max(), "this cohort: information not above communication"


def n(v, spec="{:.1f}"):
    return spec.format(v).rstrip("0").rstrip(".") if v == int(v) else spec.format(v)


L = [
    r"\begin{table}[htbp]", r"\centering", r"\footnotesize",
    r"\setlength{\tabcolsep}{5pt}",
    r"\caption{Autonomy levels selected for the five functions shared between "
    r"this study's prompt pool and the fourteen functions surveyed by "
    r"\citet{zepf}. Both columns give the median with the interquartile range "
    r"beneath it. The values for \citet{zepf} are read from their Fig.~2, the "
    r"only per-function breakdown they report, and are therefore exact only to "
    r"the resolution of that figure. This study's values are computed over all "
    r"label rows for each function, with every level in a multi-label set "
    r"entering as one observation; collapsing each window to its set mean "
    r"leaves every median unchanged. \emph{Extreme} is the share of selections "
    r"at level 0 or level 4, and the final column is the standard deviation of "
    r"the twelve per-driver means, with their range beneath it.}",
    r"\label{tab:zepf-comparison}",
    r"\begin{tabular}{l cc c cc}", r"\toprule",
    r"& \multicolumn{1}{c}{\citet{zepf}} & \multicolumn{1}{c}{This study} & & "
    r"\multicolumn{2}{c}{This study} \\",
    r"\cmidrule(lr){2-2}\cmidrule(lr){3-3}\cmidrule(lr){5-6}",
    r"Function & Median [IQR] & Median [IQR] & $n$ & Extreme & "
    r"Driver SD [range] \\",
    r"\midrule",
]
for _, r in DF.iterrows():
    L.append(" & ".join([
        r.disp,
        r"\shortstack{" + n(r.zmed) + r" \\[3pt] {\scriptsize [" + n(r.zq1) + ", " + n(r.zq3) + r"]}}",
        r"\shortstack{" + n(r.med) + r" \\[3pt] {\scriptsize [" + n(r.q1) + ", " + n(r.q3) + r"]}}",
        f"{int(r.n)}",
        f"{r.extreme:.0f}" + r"\,\%",
        r"\shortstack{" + f"{r.dsd:.2f}" + r" \\[3pt] {\scriptsize ["
        + f"{r.dlo:.2f}, {r.dhi:.2f}" + r"]}}",
    ]) + r" \\")
    L.append(r"\addlinespace[3pt]")
L = L[:-1]
L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]

out = "\n".join(L)
pathlib.Path("latex/tab_zepf_comparison.tex").write_text(out, encoding="utf-8")
print(out)
print("\n[check]")
print(DF.round(2).to_string(index=False))
