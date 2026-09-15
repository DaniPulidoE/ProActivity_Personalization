"""Paired Cliff's delta between K conditions, one table.

Mirrors data_analysis/analysis_study2.qmd section 1.11 exactly: the same paired
estimator, the same pair ordering from combn(K_ORDER, 2), and the same
ALL_SCORES list -- which does NOT include vdl_overall_mean, since that column is
derived later in section 2.
"""
import numpy as np, pandas as pd, pathlib

Q = pd.read_csv("data/study2_data/study_2_questionnaire.csv",
                dtype={"participantid": str})
Q["k"] = Q.k_condition.astype(int)

K_ORDER = [0, 1, 2]
KLAB = {0: 0, 1: 4, 2: 8}                       # condition index -> K
PAIRS = [(a, b) for i, a in enumerate(K_ORDER) for b in K_ORDER[i + 1:]]
SCALES = [
    ("proactivity_mean",    "Proactivity"),
    ("vdl_usefulness_mean", "VDL pragmatic"),
    ("vdl_satisfying_mean", "VDL hedonic"),
    ("intui_magical_mean",  "INTUI magical"),
]


def cliffs_delta_paired(x, y):
    d = np.asarray(x, float) - np.asarray(y, float)
    return (np.sum(d > 0) - np.sum(d < 0)) / len(d)


def magnitude(d):
    a = abs(d)
    return ("negligible" if a < 0.147 else
            "small" if a < 0.33 else
            "medium" if a < 0.474 else "large")


rows, ns = [], set()
for col, label in SCALES:
    wide = Q.pivot(index="participantid", columns="k", values=col)
    r = {"label": label}
    for k1, k2 in PAIRS:
        both = wide[[k1, k2]].dropna()
        ns.add(len(both))
        d = cliffs_delta_paired(both[k1], both[k2])
        r[(k1, k2)] = (d, magnitude(d))
    rows.append(r)
assert len(ns) == 1, f"unequal pair counts {ns} -- check for missing blocks"
n = ns.pop()

L = [
    r"\begin{table}[htbp]", r"\centering", r"\footnotesize",
    r"\setlength{\tabcolsep}{8pt}",
    r"\caption{Paired Cliff's $\delta$ between $K$ conditions for each rating "
    r"scale ($n=" + str(n) + r"$ participants per comparison). The paired form "
    r"compares each participant against themselves, matching the within-subject "
    r"design; the independent-samples form would fold between-driver variation "
    r"into the estimate. Each column is ordered low~$K$ against high~$K$, so a "
    r"\emph{negative} value indicates that the larger $K$ was rated more "
    r"favourably. Magnitude labels follow the conventional thresholds "
    r"\citep{romano2006}, which were derived for the independent-samples "
    r"statistic and are given here as descriptive labels only.}",
    r"\label{tab:cliffs-delta}",
    r"\begin{tabular}{l ccc}", r"\toprule",
    r"Scale & " + " & ".join(f"$K={KLAB[a]}$ vs $K={KLAB[b]}$" for a, b in PAIRS)
    + r" \\",
    r"\midrule",
]
for r in rows:
    cells = [f"{r[p][0]:+.3f}" + r" {\scriptsize (" + r[p][1] + r")}" for p in PAIRS]
    L.append(" & ".join([r["label"]] + cells) + r" \\")
L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]

out = "\n".join(L)
pathlib.Path("latex/tab_cliffs_delta.tex").write_text(out, encoding="utf-8")
print(out)
print("\n[check]")
for r in rows:
    print(f"  {r['label']:15s} " + "  ".join(
        f"{KLAB[a]}v{KLAB[b]}: {r[(a,b)][0]:+.3f} {r[(a,b)][1]:<10s}" for a, b in PAIRS))
