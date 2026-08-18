r"""Lookup baselines — personalization with no model at all.

Emits the SAME K-curve schema the learned arms produce (``pid, k, set_mae,
set_acc, ...``), so ``compare_arms_k_curve --extra-arm`` reads it as a third arm
with no special handling.

WHY THIS EXISTS
---------------
Measured on the collected cohort, on the within-driver temporal split the xLSTM
is scored on, a constant per (driver, function) cell fitted on the driver's own
past reaches **set-MAE 0.260**. The trained model on the identical split reaches
**0.956**. The lookup wins from K=5 onward (0.738 vs 0.956) using five labels, no
driver state, and no parameters.

The reason is visible in the data: each driver saw exactly 5 distinct functions
(~24 labels each), and within a (driver, function) cell their answers are highly
consistent — mean set-MAE 0.357 to that cell's own best constant. The label is
close to a deterministic function of *who x which task*. The xLSTM is given the
task (12 FCD dims of 33) and driver state (21 dims) but NOT driver identity, so
it is missing one of the two variables that matter and must infer it from
PERCLOS, gaze and steering — which a probe recovers only ~68 % of the time.

A baseline that beats the model is not a reason to hide the baseline. Reported
as a third arm it makes the K-curve interpretable and gives research question (b)
— how many labels before a driver is served well — an answer that does not
depend on any of the modelling holding up.

THE VARIANT THAT DECIDES SOMETHING
----------------------------------
``driver_function`` cannot predict a function the driver has never labelled, and
generalizing across tasks through the 12-dim FCD vector is the entire
justification for the FCD design. ``driver_function_fcd`` closes that gap with a
nearest-neighbour in FCD space. If it also covers unseen functions, the model's
last structural advantage is gone; if it does not, that regime is where the model
earns its place — and that is the contribution.

PROTOCOL — identical to the arms, deliberately
----------------------------------------------
Per driver: support = the first K labels (true session prefix), query =
everything after. Same ``set_mae``/``set_accuracy``, same one-row-per-(pid, K)
output, same fixed-tail second protocol. LODO is trivially satisfied: no variant
ever reads another driver's data, so there is nothing to hold out.

Chronology comes from (session start, window_idx, prompt_in_window), which is the
order ``labeled_data.jsonl`` was written in and therefore the order the cache's
``seg_order`` gives the arms. With ``--cache`` the segment sets are asserted equal
— if the baseline scored different segments than the arms, the comparison would
be invalid in a way no downstream table would reveal.

Usage::

    python -m ProVoice.training_scripts.baseline_lookup \
        --labels data/user_loa_labels.csv \
        --cache data/cache/segments_w10_hz10.npz \
        --outdir results/baseline_lookup
"""
from __future__ import annotations

import argparse
import csv
import pathlib
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from ProVoice.fcd_config import FCD_NAMES, get_fcd_for_function
from ProVoice.models.train_XLSTM import set_accuracy, set_mae

VARIANTS = ("driver", "driver_function", "driver_function_fcd",
            "driver_function_oracle")
# NOT DEPLOYABLE, and not a baseline in the usual sense. The oracle fits each
# (driver, function) cell's constant on the QUERY ITSELF instead of on the
# driver's earlier labels, so it is the argmin over the (driver x function)
# hypothesis class evaluated on the data it was fitted to. Nothing keyed only on
# who-is-driving and which-task can beat it.
#
# It exists to DECOMPOSE the deployable lookup's error:
#
#     deployable lookup at K=60 : 0.476
#     oracle (structural limit) : 0.167
#     ------------------------------------
#     estimation error (closes with K) : 0.309
#
# which is what says the lookup is NOT near its ceiling -- most of its remaining
# error is finite-sample, not structural. That matters for reading the arm
# comparison: the "the lookup will saturate and the model will overtake it"
# argument is not available on this cohort.
#
# It bounds the LOOKUP, not the personalized model. The model sees driver state
# and so belongs to a strictly richer class; it could in principle explain
# within-cell variation that no (driver x function) rule can.
ORACLE_VARIANT = "driver_function_oracle"
DEFAULT_KS: Tuple[int, ...] = (5, 10, 20, 30, 40, 50, 60)
MIN_QUERY = 20              # matches train_XLSTM._ADAPT_MIN_QUERY
DEFAULT_EVAL_TAIL = 30      # matches sweep_tau_quick.DEFAULT_EVAL_TAIL

# `pid, k, set_mae, set_acc` are what compare_arms_k_curve.load_arm reads; the
# rest are diagnostics. pid is written zero-padded and must be READ as a string
# downstream or '001' becomes 1 and the arm join comes back empty.
RESULTS_COLUMNS = [
    "pid", "k", "variant", "fit_on", "set_mae", "set_acc", "n_val", "n_support",
    "n_functions_seen", "n_fallback", "mean_cell_rows",
    "set_mae_tail", "set_acc_tail", "n_val_tail",
]


def marks_vector(cell: str, n_classes: int = 5) -> np.ndarray:
    """'0,3' -> multi-hot. The driver may mark SEVERAL acceptable LoAs."""
    v = np.zeros(n_classes, dtype=float)
    for tok in str(cell).replace(";", ",").split(","):
        tok = tok.strip()
        if tok.isdigit() and 0 <= int(tok) < n_classes:
            v[int(tok)] = 1.0
    return v


def best_constant(levels: np.ndarray, n_classes: int = 5) -> Optional[int]:
    """The set-MAE-optimal single LoA for a set of multi-hot labels.

    Same rule as ``train_XLSTM.constant_baseline``, so "constant" means one thing
    across every floor in the study. Returns None for an empty cell, which is
    what triggers the caller's fallback.
    """
    if levels.size == 0:
        return None
    n = levels.shape[0]
    return int(np.argmin([set_mae(levels, np.full(n, c, dtype=int))
                          for c in range(n_classes)]))


def load_labels(labels_csv: pathlib.Path) -> pd.DataFrame:
    """One row per labelled segment, in CHRONOLOGICAL order within each driver.

    Ordering is (session start, window_idx, prompt_in_window). Not a sort of
    segment_id: ids are ``<session_uuid>|win<n>p<k>``, so sorting them orders a
    driver's two sessions by UUID rather than by time, and the "prefix" would not
    be a prefix.
    """
    df = pd.read_csv(labels_csv, dtype={"participantid": str})
    need = {"participantid", "functionname", "user_selected_loa",
            "session_id", "window_idx", "prompt_in_window"}
    missing = need - set(df.columns)
    if missing:
        raise SystemExit(f"{labels_csv} is missing column(s): {sorted(missing)}")
    df["t0"] = pd.to_datetime(df.get("window_start_timestamp"), errors="coerce")
    df["marks"] = df["user_selected_loa"].map(marks_vector)
    # Rows where the driver marked nothing carry no label and cannot be scored.
    keep = df["marks"].map(lambda v: bool(v.sum()))
    if (~keep).any():
        print(f"[labels] dropping {int((~keep).sum())} row(s) with no marked LoA")
    df = df[keep]
    df["segment_id"] = (df["session_id"].astype(str) + "|win"
                        + df["window_idx"].astype(int).map("{:03d}".format) + "p"
                        + df["prompt_in_window"].astype(int).astype(str))
    return df.sort_values(["participantid", "t0", "window_idx", "prompt_in_window"]
                          ).reset_index(drop=True)


def check_against_cache(df: pd.DataFrame, cache_path: str) -> None:
    """Assert the baseline scores the SAME segments as the learned arms.

    The one check that makes the three-arm table trustworthy. A mismatch means
    the baseline and the arms were measured on different data, which no
    downstream table would reveal — the numbers would simply be wrong together.
    """
    cache = np.load(cache_path, allow_pickle=False)
    have = {str(s) for s in cache["segment_id"]}
    mine = set(df["segment_id"])
    only_cache, only_mine = have - mine, mine - have
    if only_cache or only_mine:
        raise SystemExit(
            f"segment mismatch against {pathlib.Path(cache_path).name}: "
            f"{len(only_cache)} in the cache but not the labels "
            f"(e.g. {sorted(only_cache)[:3]}), {len(only_mine)} the other way "
            f"(e.g. {sorted(only_mine)[:3]}). The baseline and the arms would be "
            f"scored on different segments.")
    print(f"[check] segment sets match the cache exactly ({len(mine)} segments)")


def _fcd_vec(name: str) -> np.ndarray:
    d = get_fcd_for_function(name)
    return np.asarray([float(d[k]) for k in FCD_NAMES])


def build_table(support: pd.DataFrame, min_cell: int) -> Tuple[Dict[str, int], Optional[int]]:
    """(function -> constant, driver constant) from the support labels alone."""
    driver_c = best_constant(np.stack(support["marks"].to_numpy()))
    table: Dict[str, int] = {}
    for fn, g in support.groupby("functionname"):
        if len(g) < min_cell:
            continue                      # too thin to trust; falls back
        c = best_constant(np.stack(g["marks"].to_numpy()))
        if c is not None:
            table[str(fn)] = c
    return table, driver_c


def predict(query: pd.DataFrame, table: Dict[str, int], driver_c: int,
            variant: str) -> Tuple[np.ndarray, int]:
    """Predicted LoA per query row, plus how many rows needed a fallback."""
    fcd_seen = ({fn: _fcd_vec(fn) for fn in table}
                if variant == "driver_function_fcd" else {})
    preds, n_fallback = [], 0
    for fn in query["functionname"].astype(str):
        if variant == "driver":
            preds.append(driver_c)
            continue
        if fn in table:
            preds.append(table[fn])
            continue
        n_fallback += 1
        if variant == "driver_function_fcd" and fcd_seen:
            # NEAREST SEEN FUNCTION in FCD space. This is the whole point of the
            # 12-dim descriptor — it is what lets ANY method generalize to a task
            # the driver never labelled — so a baseline that ignores it would
            # concede the model's one structural advantage for free.
            v = _fcd_vec(fn)
            nearest = min(fcd_seen, key=lambda s: float(np.linalg.norm(fcd_seen[s] - v)))
            preds.append(table[nearest])
        else:
            preds.append(driver_c)
    return np.asarray(preds, dtype=int), n_fallback


def curve_for_driver(g: pd.DataFrame, pid: str, ks: Sequence[int], variant: str,
                     min_cell: int, eval_tail: int) -> List[dict]:
    rows = []
    n = len(g)
    # The oracle fits on the SCORED set, every other variant on the support. That
    # one line is the whole difference, and it is why the oracle is a ceiling
    # rather than a method: it is handed the answers.
    is_oracle = variant == ORACLE_VARIANT
    rule = "driver_function" if is_oracle else variant
    for K in ks:
        if n < K + MIN_QUERY:
            continue
        support, query = g.iloc[:K], g.iloc[K:]
        table, driver_c = build_table(query if is_oracle else support, min_cell)
        if driver_c is None:
            continue
        pred, n_fb = predict(query, table, driver_c, rule)
        q_lv = np.stack(query["marks"].to_numpy())
        # Rows per (driver, function) cell in the SET THE TABLE WAS FITTED ON.
        # For the oracle this is the honest caveat: a function appearing once or
        # twice in the query is fitted essentially perfectly, so the oracle is
        # optimistically biased and the true structural limit is somewhat above it.
        fit_set = query if is_oracle else support
        cell_n = fit_set.groupby("functionname").size()
        r = {"pid": pid, "k": int(K), "variant": variant,
             "fit_on": "query (ORACLE)" if is_oracle else "support",
             "set_mae": float(set_mae(q_lv, pred)),
             "set_acc": float(set_accuracy(q_lv, pred)),
             "n_val": int(len(query)), "n_support": int(K),
             "n_functions_seen": int(len(table)), "n_fallback": int(n_fb),
             "mean_cell_rows": float(cell_n.mean()) if len(cell_n) else float("nan"),
             "set_mae_tail": float("nan"), "set_acc_tail": float("nan"),
             "n_val_tail": 0}
        # Fixed tail, disjoint from the support. The suffix query moves with K,
        # so its curve confounds "more support" with "different test set"; the
        # tail holds the query still so K is the only thing varying.
        if eval_tail > 0 and K + eval_tail <= n:
            tail = g.iloc[-eval_tail:]
            # The oracle needs its own table here: it is defined by fitting on
            # whatever is being scored, and the tail is a different set from the
            # suffix query.
            t_table, t_dc = (build_table(tail, min_cell) if is_oracle
                             else (table, driver_c))
            if t_dc is None:
                t_table, t_dc = table, driver_c
            t_pred, _ = predict(tail, t_table, t_dc, rule)
            t_lv = np.stack(tail["marks"].to_numpy())
            r["set_mae_tail"] = float(set_mae(t_lv, t_pred))
            r["set_acc_tail"] = float(set_accuracy(t_lv, t_pred))
            r["n_val_tail"] = int(len(tail))
        rows.append(r)
    return rows


def _metric_table(df: pd.DataFrame, col: str, ks: Sequence[int],
                  title: str, lower_is_better: bool) -> None:
    """One variant x K table for ``col``.

    Both metrics are printed because they disagree in a way that matters here.
    set-MAE is the design's selection metric — LoA is ordinal, so off-by-1 is not
    off-by-4 — while set-accuracy is the fraction of predictions the driver
    actually marked acceptable, which is the quantity the live study's
    satisfaction rating is closest to. A lookup that is 0.5 levels out on average
    but lands inside the accepted set half the time reads very differently under
    the two.
    """
    print(f"\n{title}   ({'lower' if lower_is_better else 'HIGHER'} is better;"
          f" *_oracle is fitted on the scored set and is NOT deployable)")
    print(f"{'variant':<22}" + "".join(f"{'K=' + str(k):>9}" for k in ks) + f"{'mean':>9}")
    for variant in VARIANTS:
        sub = df[df["variant"] == variant]
        if sub.empty or col not in sub:
            continue
        # At a fixed K there is exactly one row per driver, so the per-K cell is
        # already a per-driver mean. The `mean` column averages over K WITHIN
        # each driver first and only then across drivers — drivers contribute
        # unequal segment counts, so a flat mean over cells would weight the
        # long sessions.
        cells = [sub[sub["k"] == k][col].mean() for k in ks]
        per_driver = sub.groupby("pid")[col].mean()
        print(f"{variant:<22}" + "".join(f"{c:>9.3f}" for c in cells)
              + f"{per_driver.mean():>9.3f}")


def summarize(rows: List[dict], ks: Sequence[int]) -> None:
    df = pd.DataFrame(rows)
    _metric_table(df, "set_mae", ks, "set-MAE", lower_is_better=True)
    _metric_table(df, "set_acc", ks, "set-ACCURACY", lower_is_better=False)
    if ORACLE_VARIANT in set(df["variant"]):
        # The number the oracle exists to produce.
        dep = df[df["variant"] == "driver_function"].groupby("pid")["set_mae"].mean().mean()
        orc = df[df["variant"] == ORACLE_VARIANT].groupby("pid")["set_mae"].mean().mean()
        cells = df[df["variant"] == ORACLE_VARIANT]["mean_cell_rows"].mean()
        print(f"\nERROR DECOMPOSITION for the (driver x function) rule")
        print(f"  deployable lookup, fitted on the support : {dep:.3f}")
        print(f"  ORACLE, fitted on the scored set         : {orc:.3f}  <- structural limit")
        print(f"  estimation error, closes with more K     : {dep - orc:.3f}")
        print(f"  The lookup is {'NOT ' if dep - orc > 0.1 else ''}near its ceiling"
              f"{'; most of its remaining error is finite-sample.' if dep - orc > 0.1 else '.'}")
        print(f"  CAVEAT: oracle cells hold {cells:.1f} rows on average — small cells are "
              f"fitted almost perfectly, so the true limit is somewhat ABOVE this.")
        print(f"  The oracle bounds the LOOKUP, not a state-conditioned model, which "
              f"belongs to a richer class.")

    fb = df[df["variant"] == "driver_function"]["n_fallback"].sum()
    tot = df[df["variant"] == "driver_function"]["n_val"].sum()
    print(f"\nunseen-function rate for driver_function: {fb}/{tot} "
          f"({100 * fb / max(tot, 1):.1f}% of query rows fell back to the driver constant)")
    print("Compare against the learned arms' set_mae at the same K. Levels in the "
          "*_tail columns are NOT comparable with these — different test set.")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--labels", default="data/user_loa_labels.csv")
    ap.add_argument("--cache", default="",
                    help="Segment cache (.npz). Optional but RECOMMENDED: it asserts the "
                         "baseline scores the same segments as the learned arms.")
    ap.add_argument("--outdir", default="results/baseline_lookup")
    ap.add_argument("--ks", default=",".join(str(k) for k in DEFAULT_KS))
    ap.add_argument("--variants", default=",".join(VARIANTS))
    ap.add_argument("--eval-tail", dest="eval_tail", type=int, default=DEFAULT_EVAL_TAIL,
                    help="Also score on each driver's last N segments, a query that does "
                         "not move with K (0 = suffix only).")
    ap.add_argument("--min-cell", dest="min_cell", type=int, default=1,
                    help="Minimum support labels before a (driver, function) cell is "
                         "trusted; below it the driver constant is used instead. 1 = no "
                         "shrinkage, which is the honest default — raising it is a tuned "
                         "hyperparameter and must be disclosed as one.")
    args = ap.parse_args()

    ks = sorted({int(k) for k in args.ks.split(",") if k.strip()})
    variants = [v.strip() for v in args.variants.split(",") if v.strip()]
    bad = [v for v in variants if v not in VARIANTS]
    if bad:
        raise SystemExit(f"unknown variant(s) {bad}; choose from {list(VARIANTS)}")

    df = load_labels(pathlib.Path(args.labels))
    if args.cache:
        check_against_cache(df, args.cache)
    print(f"[data] {len(df)} labelled segments, {df['participantid'].nunique()} drivers, "
          f"{df['functionname'].nunique()} distinct functions "
          f"({df.groupby('participantid')['functionname'].nunique().mean():.1f} per driver)")

    rows: List[dict] = []
    for variant in variants:
        for pid, g in df.groupby("participantid"):
            rows += curve_for_driver(g.reset_index(drop=True), str(pid), ks,
                                     variant, args.min_cell, args.eval_tail)
    if not rows:
        raise SystemExit("no (driver, K) cells produced — is the K grid larger than "
                         "the drivers' label counts?")

    outdir = pathlib.Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    out_csv = outdir / "lookup_k_curve.csv"
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=RESULTS_COLUMNS)
        w.writeheader()
        w.writerows([{c: r.get(c, "") for c in RESULTS_COLUMNS} for r in rows])
    summarize(rows, ks)
    print(f"\n[OK] -> {out_csv}")
    print("Add it to the arm comparison with:\n"
          f"    --extra-arm {out_csv}:lookup")


if __name__ == "__main__":
    main()
