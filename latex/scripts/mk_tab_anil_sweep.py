"""ANIL (iMAML) meta-hyperparameter sweep table.

Source: results/anil_sweep_v2/anil_sweep_results.csv.
tau is frozen at the L2-SP value, so the grid is outer_lr x augmentation only.
"""
import pandas as pd, numpy as np, pathlib, json

d = pd.read_csv("results/anil_sweep_v2/anil_sweep_results.csv")
sel = json.loads(pathlib.Path("results/anil_sweep_v2/selected_anil.json")
                 .read_text(encoding="utf-8"))
M = "smoothed_best_val_mae"
d["aug"] = np.where((d.crop_frac > 0) | (d.jitter_std > 0), 1, 0)
CFG = ["outer_lr", "aug"]
d["block"] = d.fold.astype(str) + "_" + d.seed.astype(str)

W = d.groupby(CFG + ["block"])[M].mean().unstack("block")   # config x block
mean, sd = W.mean(axis=1), W.std(axis=1, ddof=1)
cen = W.sub(W.mean(axis=0), axis=1)                          # remove block effect
se = cen.std(axis=1, ddof=1) / np.sqrt(W.shape[1])
mstar = d.groupby(CFG)["best_epoch_1se"].median()

SEL = (sel["outer_lr"], 1 if (sel["crop_frac"] or sel["jitter_std"]) else 0)
# The rule is 1-SE, not argmin: within one blocked SE of the best, take the
# SMALLEST outer LR (the meta-init closest to the warm start).
band = mean.min() + se[mean.idxmin()]
in_band = mean[mean <= band].index.tolist()
assert SEL in in_band, f"{SEL} outside the 1-SE band {in_band}"
assert SEL[0] == min(x[0] for x in in_band), "selected is not the smallest LR in band"
edge = SEL[0] in (min(d.outer_lr), max(d.outer_lr))

L = [
    r"\begin{table}[htbp]", r"\centering", r"\footnotesize",
    r"\caption{ANIL meta-hyperparameter sweep. The prior precision $\tau$ is "
    r"frozen at the value selected for the L2-SP arm, so that the two arms run "
    r"the identical adaptation procedure and differ only in $\theta_{\text{init}}$; "
    r"the grid is therefore the outer learning rate crossed with task "
    r"augmentation (window crops and feature jitter, applied in meta-training "
    r"only). Entries are the smoothed meta-validation set-MAE, mean $\pm$ "
    r"standard deviation over 12 blocks (6 folds $\times$ 2 seeds), with a "
    r"fold-blocked standard error. $M^{*}$ is the median across runs of the "
    r"earliest meta-epoch within one standard error of the smoothed minimum. "
    r"The selected configuration is in bold; note that its outer learning rate "
    r"lies on the lower boundary of the searched range, which was not "
    r"extended.}",
    r"\label{tab:anil-sweep}",
    r"\begin{tabular}{lcccc}", r"\toprule",
    r"Outer LR & Augment. & meta-val set-MAE & blocked SE & $M^{*}$ \\",
    r"\midrule",
]
for idx in sorted(mean.index):
    lr, a = idx
    cells = [f"{lr:g}", "yes" if a else "no",
             f"{mean[idx]:.3f}\\,$\\pm$\\,{sd[idx]:.3f}",
             f"{se[idx]:.3f}", f"{mstar[idx]:.0f}"]
    if idx == SEL:
        cells = [r"\textbf{" + c + "}" for c in cells]
    L.append(" & ".join(cells) + r" \\")
L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]

out = "\n".join(L)
pathlib.Path("latex/tab_anil_sweep.tex").write_text(out, encoding="utf-8")
print(out)
print(f"\n[check] selected {SEL}  on grid edge: {edge}  "
      f"(json says {sel['outer_lr_on_grid_edge']})")
# augmentation main effect, paired within (outer_lr, block)
p = d.pivot_table(index=["outer_lr", "block"], columns="aug", values=M)
diff = (p[0] - p[1]).dropna()
print(f"[check] augmentation effect (off - on): {diff.mean():+.4f} "
      f"+/- {diff.std(ddof=1)/np.sqrt(len(diff)):.4f}  n={len(diff)}, helps in "
      f"{(diff > 0).sum()}/{len(diff)}")
