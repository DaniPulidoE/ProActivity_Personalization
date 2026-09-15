"""Prior-precision (tau) sweep table.

Source: results/l2sp_sweep/l2sp_tau_sweep.csv -- the sweep run against the
deployed LODO checkpoints (embed_fcd=1, head_in=76).  NOT
results/l2sp_sweep_fcd_extend, which used superseded checkpoints (its per-driver
unadapted floors differ) and lacks the embed_fcd/head_in columns.
"""
import pandas as pd, numpy as np, pathlib

K_CAP = 60
SEL = 1.0

d = pd.read_csv("results/l2sp_sweep/l2sp_tau_sweep.csv")
s = d[d.k <= K_CAP]

# driver x tau matrix of within-driver means, so every statistic below is
# computed on drivers as the unit rather than on individual (driver, K) cells.
M = s.groupby(["tau", "pid"])["set_mae"].mean().unstack("pid")
mean = M.mean(axis=1)
sd = M.std(axis=1, ddof=1)                      # between-driver spread
cen = M.sub(M.mean(axis=0), axis=1)             # remove the driver main effect
se_blocked = cen.std(axis=1, ddof=1) / np.sqrt(M.shape[1])
conv = 100 * s.groupby("tau").apply(
    lambda g: (g.grad_norm.abs() <= 1e-3).mean(), include_groups=False)

floor = s.base_set_mae.mean()
best = mean.min()
band = best + se_blocked[SEL]
within = mean[mean <= band].index.tolist()
assert max(within) == SEL, f"1-SE band selects {max(within)}, not {SEL}"

L = [
    r"\begin{table}[htbp]", r"\centering", r"\footnotesize",
    r"\caption{Prior-precision sweep. Set-MAE is averaged within driver and then "
    r"across the twelve leave-one-driver-out folds over $K \le " + str(K_CAP) +
    r"$, against an unadapted population floor of $" + f"{floor:.3f}" + r"$. The "
    r"standard deviation is between drivers and is an order of magnitude larger "
    r"than any difference across the grid; the driver-blocked standard error, "
    r"which removes the driver main effect, is the appropriate yardstick for a "
    r"within-driver comparison. ``Converged'' is the proportion of adaptations "
    r"reaching $\lVert\nabla\rVert \le 10^{-3}$: at low $\tau$ the fixed step "
    r"budget does not reach the optimum, so those rows partly measure the "
    r"optimiser rather than the objective. The selected value is in bold.}",
    r"\label{tab:tau-sweep}",
    r"\begin{tabular}{lccc}", r"\toprule",
    r"$\tau$ & set-MAE & blocked SE & Converged \\", r"\midrule",
]
for t in mean.index:
    cells = [f"{t:g}", f"{mean[t]:.3f}\\,$\\pm$\\,{sd[t]:.3f}",
             f"{se_blocked[t]:.3f}", f"{conv[t]:.1f}\\,\\%"]
    if t == SEL:
        cells = [r"\textbf{" + c + "}" for c in cells]
    L.append(" & ".join(cells) + r" \\")
L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]

out = "\n".join(L)
pathlib.Path("latex/tab_tau_sweep.tex").write_text(out, encoding="utf-8")
print(out)
print(f"\n[check] min {best:.4f} at tau={mean.idxmin():g}; "
      f"1-SE band <= {band:.4f}; within: {[f'{x:g}' for x in within]}")
