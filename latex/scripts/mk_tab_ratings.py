"""Ratings across K conditions -- one table for every scale.

Replicates the estimators in data_analysis/analysis_study2.qmd: permutation
Friedman, Kendall's W, one-way RM-ANOVA, McDonald's omega from a one-factor PAF
solution, and the noncentral chi-square required-n. Values are recomputed here
rather than transcribed, so a change to the data cannot leave the table stale.
"""
import numpy as np, pandas as pd, pathlib
from scipy import stats

Q = pd.read_csv("data/study2_data/study_2_questionnaire.csv",
                dtype={"participantid": str})
Q["k"] = Q.k_condition.astype(int)
# Derived in the analysis notebook, not stored in the CSV.
Q["vdl_overall_mean"] = Q[[f"vdl{i}_scored" for i in range(1, 10)]].mean(axis=1)
K_ORDER, ALPHA, TARGET = [0, 1, 2], 0.05, 0.80
KLAB = {0: "$K=0$", 1: "$K=4$", 2: "$K=8$"}

VDL = [f"vdl{i}_scored" for i in range(1, 10)]
SCALES = [
    ("vdl_overall_mean",    "VDL overall",    VDL),
    ("vdl_usefulness_mean", "VDL pragmatic",  [f"vdl{i}_scored" for i in (1, 3, 5, 7, 9)]),
    ("vdl_satisfying_mean", "VDL hedonic",    [f"vdl{i}_scored" for i in (2, 4, 6, 8)]),
    ("intui_magical_mean",  "INTUI magical",  [f"intui{i}_scored" for i in (1, 2, 3, 4)]),
    ("proactivity_mean",    "Proactivity",    None),      # single item: no ANOVA, no omega
]

# Permutation p-values as rendered by data_analysis/analysis_study2.html
# (2026-09-06 17:23). Transcribed rather than recomputed -- see the assertion in
# the loop below, which still verifies the local estimate agrees.
P_RENDERED = {
    "vdl_overall_mean":    0.2627,
    "vdl_usefulness_mean": 0.3103,
    "vdl_satisfying_mean": 0.1352,
    "intui_magical_mean":  0.7718,
    "proactivity_mean":    0.7822,
}


def friedman_raw(ranks):
    n, k = ranks.shape
    return (12.0 / (n * k * (k + 1))) * np.sum(ranks.sum(0) ** 2) - 3 * n * (k + 1)


def tie_correction(d):
    n, k = d.shape
    ties = sum(np.sum(c * (c ** 2 - 1))
               for row in d for _, c in [np.unique(row, return_counts=True)])
    return 1 - ties / (k * (k ** 2 - 1) * n)


def friedman_perm(d, n_perm=5000, seed=0):
    rng = np.random.default_rng(seed)
    r = stats.rankdata(d, axis=1)
    obs = friedman_raw(r)
    hits = sum(friedman_raw(rng.permuted(r, axis=1)) >= obs - 1e-9 for _ in range(n_perm))
    return obs / tie_correction(d), (hits + 1) / (n_perm + 1)


def kendalls_w(d):
    n, k = d.shape
    r = stats.rankdata(d, axis=1)
    s = np.sum((r.sum(0) - n * (k + 1) / 2) ** 2)
    return 12 * s / (n ** 2 * k * (k ** 2 - 1))


def rm_anova(d):
    n, k = d.shape
    gm, sm, cm = d.mean(), d.mean(1, keepdims=True), d.mean(0, keepdims=True)
    sst = ((d - gm) ** 2).sum()
    sss, ssc = k * ((sm - gm) ** 2).sum(), n * ((cm - gm) ** 2).sum()
    sse = sst - sss - ssc
    dfc, dfe = k - 1, (k - 1) * (n - 1)
    F = (ssc / dfc) / (sse / dfe)
    return F, stats.f.sf(F, dfc, dfe), ssc / (ssc + sse), dfc, dfe


def omega(items):
    R = np.corrcoef(items.to_numpy(float), rowvar=False)
    p = R.shape[0]
    h2 = np.clip(1 - 1 / np.diag(np.linalg.inv(R)), 0, .999)
    for _ in range(100):
        Rr = R.copy(); np.fill_diagonal(Rr, h2)
        ev, evec = np.linalg.eigh(Rr); i = np.argmax(ev)
        if ev[i] <= 0:
            return float("nan")
        L = evec[:, i] * np.sqrt(ev[i])
        if L.mean() < 0:
            L = -L
        new = np.clip(L ** 2, 0, .999)
        if np.max(np.abs(new - h2)) < 1e-6:
            h2 = new; break
        h2 = new
    return L.sum() ** 2 / (L.sum() ** 2 + np.sum(1 - L ** 2))


def boot_ci(v, n_boot=10000, ci=0.95, seed=1):
    """Percentile bootstrap CI on the condition mean, matching the analysis file.

    Monte Carlo error means the bounds differ in the third decimal from an R run
    at the same nominal seed; the interval itself is the reported quantity.
    """
    rng = np.random.default_rng(seed)
    v = np.asarray(v, float)
    bm = rng.choice(v, size=(n_boot, len(v)), replace=True).mean(axis=1)
    return np.percentile(bm, [(1 - ci) / 2 * 100, (1 + ci) / 2 * 100])


def required_n(w, k, alpha=ALPHA, target=TARGET, nmax=2000):
    """Smallest n whose noncentral chi-square power reaches `target` at concordance w."""
    df = k - 1
    crit = stats.chi2.ppf(1 - alpha, df)
    for n in range(3, nmax + 1):
        if stats.ncx2.sf(crit, df, n * df * w) >= target:
            return n
    return None


rows = []
for col, label, items in SCALES:
    wide = Q.pivot(index="participantid", columns="k", values=col)[K_ORDER]
    d = wide.to_numpy(float)
    q_stat, p_perm = friedman_perm(d)
    # The permutation p is Monte Carlo, and R and NumPy do not share a generator
    # at the same nominal seed. The rendered analysis (analysis_study2.html) is
    # the value of record, so take it from there -- but check the local estimate
    # agrees to within 4 Monte-Carlo standard errors, so a genuine divergence
    # (a changed estimator, a changed dataset) still fails loudly.
    p_html = P_RENDERED[col]
    mc_se = np.sqrt(p_html * (1 - p_html) / 5000)
    assert abs(p_perm - p_html) < 4 * mc_se, (
        f"{col}: recomputed p={p_perm:.4f} vs rendered {p_html:.4f} "
        f"({abs(p_perm - p_html) / mc_se:.1f} MC-SE) -- not Monte-Carlo noise")
    p_perm = p_html
    w = kendalls_w(d)
    r = dict(label=label, w=w, q=q_stat, p=p_perm, n=required_n(w, d.shape[1]))
    for j, kk in enumerate(K_ORDER):
        r[f"m{kk}"], r[f"s{kk}"] = d[:, j].mean(), d[:, j].std(ddof=1)
        r[f"lo{kk}"], r[f"hi{kk}"] = boot_ci(d[:, j])
    if items is not None:                       # multi-item only
        F, p, eta, dfc, dfe = rm_anova(d)
        r.update(F=F, pF=p, eta=eta, dfc=dfc, dfe=dfe, om=omega(Q[items]))
    rows.append(r)

DF = pd.DataFrame(rows)
dfc, dfe = int(DF.dfc.dropna().iloc[0]), int(DF.dfe.dropna().iloc[0])


def f(v, spec="{:.3f}"):
    return "---" if v is None or (isinstance(v, float) and np.isnan(v)) else spec.format(v)


L = [
    r"\begin{table}[htbp]", r"\centering", r"\footnotesize",
    r"\setlength{\tabcolsep}{4pt}",
    r"\caption{Subjective ratings across the three $K$ conditions "
    r"($n=12$ participants). Cells give the mean $\pm$ standard deviation "
    r"above a percentile bootstrap $95\,\%$ confidence interval on that mean "
    r"($10{,}000$ resamples). "
    r"Friedman's $Q$ is tie-corrected and its $p$-value is obtained from "
    r"$5{,}000$ permutations of each participant's own scores across "
    r"conditions; $W$ is Kendall's coefficient of concordance. The "
    r"repeated-measures ANOVA is reported on $F(" + f"{dfc},{dfe}" + r")$ with "
    r"partial $\eta^{2}$, and $\omega$ is McDonald's coefficient from a "
    r"one-factor solution. $n_{80}$ is the sample size at which an effect of "
    r"the observed concordance would reach $80\,\%$ power, from the noncentral "
    r"chi-square approximation. The proactivity measure is a single item, so "
    r"neither the ANOVA nor $\omega$ applies to it.}",
    r"\label{tab:ratings-k}",
    r"\begin{tabular}{l ccc cc c ccc c c}", r"\toprule",
    r"& \multicolumn{3}{c}{Condition mean $\pm$ SD} "
    r"& \multicolumn{2}{c}{Friedman} & & \multicolumn{3}{c}{RM-ANOVA} & & \\",
    r"\cmidrule(lr){2-4}\cmidrule(lr){5-6}\cmidrule(lr){8-10}",
    r"Scale & " + " & ".join(KLAB[k] for k in K_ORDER)
    + r" & $Q$ & $p$ & $W$ & $F$ & $p$ & $\eta_p^{2}$ & $\omega$ & $n_{80}$ \\",
    r"\midrule",
]
for _, r in DF.iterrows():
    L.append(" & ".join([
        r.label,
        # Mean +/- SD, with the bootstrap interval stacked beneath it. The
        # optional argument on \\[4pt] both opens up the gap between the two
        # lines and stops LaTeX reading the "[" of the interval as its own
        # optional argument.
        *[r"\shortstack{"
          + f"{r[f'm{k}']:.2f}" + r"\,$\pm$\," + f"{r[f's{k}']:.2f}"
          + r" \\[4pt] {\scriptsize ["
          + f"{r[f'lo{k}']:.2f}, {r[f'hi{k}']:.2f}" + r"]}}"
          for k in K_ORDER],
        f(r.q, "{:.3f}"), f(r.p, "{:.3f}"), f(r.w, "{:.3f}"),
        f(r.get("F"), "{:.3f}"), f(r.get("pF"), "{:.3f}"), f(r.get("eta"), "{:.3f}"),
        f(r.get("om"), "{:.3f}"), f(r.n, "{:d}") if r.n else "---",
    ]) + r" \\")
    L.append(r"\addlinespace[3pt]")      # two-line cells need air between rows too
L = L[:-1]                               # ...but not before \bottomrule
L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]

out = "\n".join(L)
pathlib.Path("latex/tab_ratings_k.tex").write_text(out, encoding="utf-8")
print(out)
print("\n[check]")
print(DF[["label", "q", "p", "w", "F", "pF", "eta", "om", "n"]].round(4).to_string(index=False))
