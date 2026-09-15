"""Both soft-CORN vs cross-entropy comparisons, as two LaTeX tables.

Table 1  results/pop_pipeline/{corn_w10, ce_w10}      -- unadapted metric, 6 configs
Table 2  results/pop_adapt_full/{corn_w10, ce_w20}    -- both metrics, 8 shared configs
         (the ce_w20 directory is misnamed: it holds window_seconds = 10)
"""
import pandas as pd, numpy as np, pathlib
from scipy import stats

UN = "smoothed_best_set_mae"
AD = "smoothed_best_adapt_mae"


def paired(j, metric, key):
    """CE - CORN, averaged within configuration then tested across them."""
    d = j[metric + "_ce"] - j[metric + "_co"]          # + favours soft-CORN
    g = j.assign(_d=d).groupby(key)["_d"].mean()
    t = stats.ttest_1samp(g, 0)
    return g, g.mean(), g.std(ddof=1) / np.sqrt(len(g)), t.pvalue, int((g > 0).sum())


def ms(d, c):
    return f"{d[c].mean():.3f}\\,$\\pm$\\,{d[c].std(ddof=1):.3f}"


out = []

# ---------------------------------------------------------------- table 1
co = pd.read_csv("results/pop_pipeline/corn_w10/sweep_results.csv")
ce = pd.read_csv("results/pop_pipeline/ce_w10/sweep_results.csv")
K1 = ["dropout", "lr"]
j1 = ce.merge(co, on=K1 + ["fold", "seed"], suffixes=("_ce", "_co"))
g1, m1, se1, p1, w1 = paired(j1, UN, K1)
floor = float(co["const_set_mae"].mean())

L = [
    r"\begin{table}[htbp]", r"\centering", r"\footnotesize",
    r"\caption{Soft-CORN versus cross-entropy on the unadapted validation "
    r"set-MAE, the sweep on which the loss was chosen. Entries are the mean "
    r"$\pm$ standard deviation over 30 runs (6 folds $\times$ 5 seeds); lower "
    r"is better. $\Delta$ is CE minus soft-CORN, so a positive value favours "
    r"soft-CORN. The grid searched only two learning rates and both losses "
    r"attain their minimum at the lower one, so neither best configuration is "
    r"interior; the comparison is nevertheless paired within configuration, so "
    r"both losses are affected equally. No configuration of either loss beats "
    r"the constant baseline of " + f"{floor:.3f}" + r".}",
    r"\label{tab:loss-unadapted}",
    r"\begin{tabular}{cc ccc}", r"\toprule",
    r"Dropout & LR & soft-CORN & CE & $\Delta$ \\", r"\midrule",
]
for (d_, lr_), gg in j1.groupby(K1):
    L.append(f"{d_:.2f} & {lr_:g} & {ms(gg, UN+'_co')} & {ms(gg, UN+'_ce')} & "
             f"{g1.loc[(d_, lr_)]:+.3f}" + r" \\")
L += [
    r"\midrule",
    r"\multicolumn{2}{l}{\emph{Paired over configurations}} & "
    + f"{j1[UN+'_co'].mean():.3f} & {j1[UN+'_ce'].mean():.3f} & "
    + f"\\textbf{{{m1:+.3f}\\,$\\pm$\\,{se1:.3f}}}" + r" \\",
    r"\multicolumn{4}{l}{\emph{soft-CORN better in " + f"{w1}/{len(g1)}"
    + r" configurations,\ $p = " + f"{p1:.3f}" + r"$}} & \\",
    r"\bottomrule", r"\end{tabular}", r"\end{table}",
]
out.append("\n".join(L))

# ---------------------------------------------------------------- table 2
co2 = pd.read_csv("results/pop_adapt_full/corn_w10/sweep_results.csv")
ce2 = pd.read_csv("results/pop_adapt_full/ce_w20/sweep_results.csv")
assert set(ce2.window_seconds.unique()) == {10.0}
K2 = ["dropout", "lr", "embed_fcd"]
j2 = ce2.merge(co2, on=K2 + ["fold", "seed"], suffixes=("_ce", "_co"))
gA, mA, seA, pA, wA = paired(j2, AD, K2)
gU, mU, seU, pU, wU = paired(j2, UN, K2)
sub = j2[j2.lr < 0.01]
gX, mX, seX, pX, wX = paired(sub, AD, K2)

L = [
    r"\begin{table}[htbp]", r"\centering", r"\scriptsize",
    r"\setlength{\tabcolsep}{4pt}",
    r"\caption{Soft-CORN versus cross-entropy on the eight configurations the "
    r"two grids share, scored both before and after per-driver adaptation. "
    r"Entries are the mean $\pm$ standard deviation over 12 runs (6 folds "
    r"$\times$ 2 seeds); $\Delta$ is CE minus soft-CORN, so positive favours "
    r"soft-CORN. Both metrics come from the same runs, so the change of sign "
    r"between the two panels is a change of metric, not of data. The "
    r"cross-entropy grid was never extended beyond these eight cells, and its "
    r"best configuration sits at $\text{lr} = 0.010$, the top of its range.}",
    r"\label{tab:loss-adapted}",
    r"\begin{tabular}{ccc ccc ccc}", r"\toprule",
    r"& & & \multicolumn{3}{c}{Post-adaptation} "
    r"& \multicolumn{3}{c}{Unadapted} \\",
    r"\cmidrule(lr){4-6}\cmidrule(lr){7-9}",
    r"Drop. & LR & FCD & soft-CORN & CE & $\Delta$ "
    r"& soft-CORN & CE & $\Delta$ \\", r"\midrule",
]
for (d_, lr_, f_), gg in j2.groupby(K2):
    L.append(f"{d_:.2f} & {lr_:g} & {int(f_)} & "
             f"{ms(gg, AD+'_co')} & {ms(gg, AD+'_ce')} & {gA.loc[(d_, lr_, f_)]:+.3f} & "
             f"{ms(gg, UN+'_co')} & {ms(gg, UN+'_ce')} & {gU.loc[(d_, lr_, f_)]:+.3f}"
             + r" \\")
L += [
    r"\midrule",
    r"\multicolumn{3}{l}{\emph{Paired over configurations}} & "
    + f"{j2[AD+'_co'].mean():.3f} & {j2[AD+'_ce'].mean():.3f} & "
    + f"\\textbf{{{mA:+.3f}\\,$\\pm$\\,{seA:.3f}}} & "
    + f"{j2[UN+'_co'].mean():.3f} & {j2[UN+'_ce'].mean():.3f} & "
    + f"\\textbf{{{mU:+.3f}\\,$\\pm$\\,{seU:.3f}}}" + r" \\",
    r"\multicolumn{9}{l}{\emph{soft-CORN better in " + f"{wA}/{len(gA)}"
    + r" configurations post-adaptation ($p = " + f"{pA:.2f}" + r"$), "
    + f"{wU}/{len(gU)}" + r" unadapted ($p = " + f"{pU:.2f}" + r"$)}} \\",
    r"\multicolumn{9}{l}{\emph{Excluding $\text{lr} = 0.010$: post-adaptation "
    r"$\Delta = " + f"{mX:+.3f} \\pm {seX:.3f}$, $p = {pX:.2f}$" + r"}} \\",
    r"\bottomrule", r"\end{tabular}", r"\end{table}",
]
out.append("\n".join(L))

txt = "\n\n".join(out)
pathlib.Path("latex/tab_loss_comparison.tex").write_text(txt, encoding="utf-8")
print(txt)
print(f"\n[check] table1 floor={floor:.4f}  n_cells={len(j1)}")
print(f"[check] table2 cells={len(j2)}  configs={len(gA)}")
