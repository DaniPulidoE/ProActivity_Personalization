"""Stated phone-call autonomy preference against revealed study-1 selections.

Mirrors data_analysis/analysis_study2.qmd section 4. Every value is recomputed
from the raw files, then checked against the rendered analysis
(analysis_study2.html, 2026-09-06) before the table is written -- so a change to
either input fails loudly rather than producing a stale table.
"""
import numpy as np, pandas as pd, pathlib, re
from scipy import stats

DEMOG = "data/study2_data/demographics_questionnaire_study2.csv"
LABELS = "data/study1_data/user_loa_labels.csv"
PHONE_FN = "Respond to a phone call"
N_PERM, SEED = 20000, 0

# Qualtrics export: row 0 is the question text, row 1 the ImportId JSON.
d = pd.read_csv(DEMOG, dtype=str).iloc[2:]
demog = pd.DataFrame({
    "participantid": d.Q1.str.strip().str.zfill(3),
    # "Level N: ..." is the UI's 1-based label; -1 puts it on the 0-4 code.
    "pref_loa": d.Q2.str.extract(r"Level (\d)")[0].astype(float) - 1,
    "context_dep": d.Q5_1.astype(float),
})

lab = pd.read_csv(LABELS, dtype={"participantid": str})
lab = lab[lab.functionname == PHONE_FN].copy()
# A driver may mark a SET of acceptable levels; collapse to the set mean.
lab["win_loa"] = lab.user_selected_loa.astype(str).map(
    lambda s: np.mean([float(x) for x in s.split(";") if x.strip() != ""]))
beh = lab.groupby("participantid").win_loa.agg(
    n_calls="size", loa_mean="mean", loa_median="median", loa_var="var").reset_index()

P = demog.merge(beh, on="participantid").sort_values("participantid")
assert len(P) == 12, f"expected 12 drivers, got {len(P)}"

# Spot-checks against the rendered per-driver table.
for pid, pref, ctx, n, mean, med, var in [
        ("001", 3, 7, 24, 2.65, 2.5, 0.684),
        ("010", 2, 4, 24, 2.00, 2.0, 0.000),
        ("012", 3, 7, 19, 1.79, 2.0, 0.287)]:
    r = P[P.participantid == pid].iloc[0]
    assert (r.pref_loa, r.context_dep, r.n_calls) == (pref, ctx, n), f"{pid} items"
    assert abs(r.loa_mean - mean) < 5e-3 and abs(r.loa_var - var) < 5e-3, f"{pid} behaviour"


def kendall_perm(x, y, n_perm=N_PERM, seed=SEED):
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    obs = stats.kendalltau(x, y, variant="b")
    rng = np.random.default_rng(seed)
    hits = sum(abs(stats.kendalltau(x, rng.permutation(y), variant="b").statistic)
               >= abs(obs.statistic) - 1e-12 for _ in range(n_perm))
    return obs.statistic, (hits + 1) / (n_perm + 1), obs.pvalue, len(x)


ITEM = {"pref_loa": "Preferred level", "context_dep": "Context dependence"}
BEH = {"loa_mean": "mean selected", "loa_median": "median selected",
       "loa_var": "variance of selections"}
# Only the pre-specified associations are tabulated. The three exploratory
# pairings are computed and checked below, so a change to them still trips an
# assertion, but they are not reported as results.
PAIRS = [("pref_loa", "loa_mean", "pre-specified"),
         ("pref_loa", "loa_median", "pre-specified"),
         ("context_dep", "loa_var", "pre-specified"),
         ("pref_loa", "loa_var", "exploratory"),
         ("context_dep", "loa_mean", "exploratory"),
         ("context_dep", "loa_median", "exploratory")]
REPORTED = "pre-specified"

# tau_b and the asymptotic p as rendered; the permutation p is Monte Carlo and
# is taken from the render, with the local estimate checked against it below.
RENDERED = {("pref_loa", "loa_mean"):      (0.410, 0.116, 0.097),
            ("pref_loa", "loa_median"):    (0.363, 0.170, 0.154),
            ("context_dep", "loa_var"):    (0.259, 0.295, 0.262),
            ("pref_loa", "loa_var"):       (0.328, 0.225, 0.184),
            ("context_dep", "loa_mean"):   (0.194, 0.453, 0.400),
            ("context_dep", "loa_median"): (0.252, 0.320, 0.287)}

rows = []
for item, behav, hyp in PAIRS:
    tau, p_perm, p_asy, n = kendall_perm(P[item], P[behav])
    r_tau, r_perm, r_asy = RENDERED[(item, behav)]
    assert abs(tau - r_tau) < 5e-4, f"{item}~{behav}: tau {tau:.4f} vs {r_tau}"
    assert abs(p_asy - r_asy) < 5e-4, f"{item}~{behav}: asymptotic p differs"
    se = np.sqrt(r_perm * (1 - r_perm) / N_PERM)
    assert abs(p_perm - r_perm) < 4 * se, (
        f"{item}~{behav}: permutation p {p_perm:.4f} vs {r_perm} -- not MC noise")
    rows.append((ITEM[item], BEH[behav], hyp, r_tau, r_perm, r_asy, n))

n_all = {r[6] for r in rows}
assert len(n_all) == 1
n = n_all.pop()

L = [
    r"\begin{table}[htbp]", r"\centering", r"\footnotesize",
    r"\setlength{\tabcolsep}{6pt}",
    r"\caption{Stated phone-call autonomy preference against the levels the same "
    r"drivers selected during the population collection ($n=" + str(n) + r"$). "
    r"Association is Kendall's $\tau_b$, which is tie-corrected; $p_{\text{perm}}$ "
    r"comes from $" + f"{N_PERM:,}".replace(",", "{,}") + r"$ shuffles of the "
    r"questionnaire item across drivers and $p_{\text{asym}}$ is the asymptotic "
    r"value, shown as a cross-check. Only the pre-specified associations are "
    r"reported.}",
    r"\label{tab:stated-revealed}",
    r"\begin{tabular}{ll ccc}", r"\toprule",
    r"Questionnaire item & Compared with & $\tau_b$ & $p_{\text{perm}}$ "
    r"& $p_{\text{asym}}$ \\",
    r"\midrule",
]
for item, behav, hyp, tau, pp, pa, _ in rows:
    if hyp != REPORTED:
        continue
    L.append(f"{item} & {behav} & {tau:.3f} & {pp:.3f} & {pa:.3f}" + r" \\")
L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]

out = "\n".join(L)
pathlib.Path("latex/tab_stated_revealed.tex").write_text(out, encoding="utf-8")
print(out)
print(f"\n[check] all 6 tau_b and asymptotic p reproduced; "
      f"permutation p within Monte-Carlo error")
print(f"[check] Q2 tie structure: "
      f"{P.pref_loa.value_counts().max():.0f} of {len(P)} drivers gave the same answer")
