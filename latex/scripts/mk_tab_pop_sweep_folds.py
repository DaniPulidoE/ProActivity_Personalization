import pandas as pd, numpy as np, pathlib

a = pd.read_csv("results/pop_adapt_full/corn_w20/sweep_results.csv")
b = pd.read_csv("results/pop_adapt_full_ext/corn_w20/sweep_results.csv")
m = pd.concat([a, b], ignore_index=True).drop_duplicates(
    subset=["dropout", "lr", "embed_fcd", "fold", "seed"])

cfg = ["dropout", "lr", "embed_fcd"]
A = "smoothed_best_adapt_mae"
PAIR = {0: "001/005", 1: "002/006", 2: "003/007",
        3: "004/008", 4: "009/011", 5: "010/012"}

mean_f = m.groupby(cfg + ["fold"])[A].mean().unstack("fold")
sd_f = m.groupby(cfg + ["fold"])[A].std(ddof=1).unstack("fold")
overall = m.groupby(cfg)[A].agg(["mean", "std"])
glob = overall["mean"].idxmin()

nested = {f: m[m.fold != f].groupby(cfg)[A].mean().idxmin() for f in range(6)}
assert set(nested.values()) == {glob}, "nested selection is NOT unanimous"
assert (m.groupby(cfg + ["fold"]).size() == 2).all(), "not 2 seeds per cell"

order = mean_f.sort_index(level=["embed_fcd","dropout","lr"]).index

def bf(s):
    return r"\textbf{" + s + "}"

def cell(mu, sd, bold):
    s = f"{mu:.3f}\\,$\\pm$\\,{sd:.3f}"
    return bf(s) if bold else s

L = [
    r"\begin{table}[htbp]",
    r"\centering",
    r"\scriptsize",
    r"\setlength{\tabcolsep}{3pt}",
    r"\caption{Post-adaptation validation set-MAE for every hyperparameter "
    r"combination, broken down by held-out driver fold. Lower is better; the "
    r"selected configuration is shown in bold. Each fold cell is the mean "
    r"$\pm$ standard deviation over "
    r"the two seeds run for that cell, so its dispersion reflects seed "
    r"variation alone and rests on $n=2$. The mean over all twelve runs of "
    r"each row is given in Table~\ref{tab:pop-sweep}. FCD indicates whether "
    r"the task vector is appended "
    r"to the head input. The grid is not fully crossed: it merges an initial "
    r"grid (dropout $0.10$--$0.20$, learning rate $0.001$--$0.002$, both FCD "
    r"settings) with an extension (dropout $0.20$--$0.30$, learning rate "
    r"$0.002$--$0.010$, FCD appended) run after the first optimum landed on a "
    r"boundary.}",
    r"\label{tab:pop-sweep-folds}",
    r"\begin{tabular}{ccc cccccc}",
    r"\toprule",
    r"& & & \multicolumn{6}{c}{Held-out driver pair} \\",
    r"\cmidrule(lr){4-9}",
    r"Drop. & LR & FCD & " + " & ".join(PAIR[f] for f in range(6))
    + r" \\",
    r"\midrule",
]

prev = None
for idx in order:
    d, lr, fc = idx
    if prev is not None and fc != prev:
        L.append(r"\midrule")
    prev = fc
    best = (idx == glob)
    lab = [f"{d:.2f}", f"{lr:g}", str(int(fc))]
    if best:
        lab = [bf(x) for x in lab]
    cells = [cell(mean_f.loc[idx, f], sd_f.loc[idx, f], best) for f in range(6)]
    L.append(" & ".join(lab + cells) + r" \\")

L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]

out = "\n".join(L)
pathlib.Path("latex/tab_pop_sweep_folds.tex").write_text(out, encoding="utf-8")
print(out)
print("\n[width] longest body row:",
      max(len(x) for x in L if x.endswith(r"\\")), "chars")
