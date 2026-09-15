import pandas as pd, numpy as np, pathlib

a = pd.read_csv("results/pop_adapt_full/corn_w20/sweep_results.csv")
b = pd.read_csv("results/pop_adapt_full_ext/corn_w20/sweep_results.csv")
m = pd.concat([a, b], ignore_index=True).drop_duplicates(
    subset=["dropout", "lr", "embed_fcd", "fold", "seed"])

cfg = ["dropout", "lr", "embed_fcd"]
A = "smoothed_best_adapt_mae"
PAIR = {0: "001/005", 1: "002/006", 2: "003/007",
        3: "004/008", 4: "009/011", 5: "010/012"}

glob = m.groupby(cfg)[A].mean().idxmin()

def name(c):
    return f"({c[0]:g}, {c[1]:g}, {int(c[2])})"

rows = []
for f in range(6):
    sub = m[m.fold != f]
    stats = sub.groupby(cfg)[A].agg(["mean", "std"]).sort_values("mean")
    w, r = stats.index[0], stats.index[1]
    # Paired difference, blocked on (fold, seed): the selection is a ranking of
    # configurations measured on the SAME runs, so the between-fold spread in
    # the `std` columns is common to both and cancels here.
    wv = sub[(sub.dropout == w[0]) & (sub.lr == w[1]) & (sub.embed_fcd == w[2])] \
        .set_index(["fold", "seed"])[A]
    rv = sub[(sub.dropout == r[0]) & (sub.lr == r[1]) & (sub.embed_fcd == r[2])] \
        .set_index(["fold", "seed"])[A].reindex(wv.index)
    d = (rv - wv).values
    rows.append(dict(
        fold=f, w=w, wm=stats.loc[w, "mean"], wsd=stats.loc[w, "std"],
        r=r, rm=stats.loc[r, "mean"], rsd=stats.loc[r, "std"],
        marg=d.mean(), se=d.std(ddof=1) / np.sqrt(len(d)), n=len(d)))

assert {x["w"] for x in rows} == {glob}, "nested selection is NOT unanimous"

L = [
    r"\begin{table}[htbp]",
    r"\centering",
    r"\footnotesize",
    r"\setlength{\tabcolsep}{5pt}",
    r"\caption{Nested hyperparameter selection. Each row withholds one driver "
    r"pair and re-runs the selection on the five folds that remain, i.e.\ it "
    r"reports what the selection rule would have seen had that pair been "
    r"unavailable. The same configuration is selected every time, so the "
    r"hyperparameters applied to a driver's model would have been chosen "
    r"identically without that driver's data. Set-MAE columns are the mean "
    r"$\pm$ standard deviation over the ten runs behind each entry; that "
    r"dispersion is dominated by differences in fold difficulty, is common to "
    r"both configurations, and is not what separates them. The claim here is "
    r"about which configuration is selected, not about a significant gap "
    r"between configurations. Note also that the six selections are not "
    r"independent -- any two share four of their five folds -- so the "
    r"unanimity is weaker evidence of robust superiority than six independent "
    r"replications would be; what it does establish is the "
    r"leave-one-out counterfactual the objection concerns. Configurations are "
    r"written (dropout, learning rate, FCD appended).}",
    r"\label{tab:pop-sweep-nested}",
    r"\begin{tabular}{l l c l c}",
    r"\toprule",
    r"Pair & \multicolumn{2}{c}{Selected} "
    r"& \multicolumn{2}{c}{Runner-up} \\",
    r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}",
    r"withheld & config & set-MAE & config & set-MAE \\",
    r"\midrule",
]

for x in rows:
    L.append(" & ".join([
        PAIR[x["fold"]],
        name(x["w"]), f"{x['wm']:.3f}\\,$\\pm$\\,{x['wsd']:.3f}",
        name(x["r"]), f"{x['rm']:.3f}\\,$\\pm$\\,{x['rsd']:.3f}",
    ]) + r" \\")

L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]

out = "\n".join(L)
pathlib.Path("latex/tab_pop_sweep_nested.tex").write_text(out, encoding="utf-8")
print(out)
