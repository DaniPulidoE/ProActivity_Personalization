r"""The K curve for ONE function — the live study's actual deployment condition.

The live follow-up study fires exactly one interactive event, ``Respond to a
phone call``, and serves a head personalized for it. Every offline K curve
produced so far pools all five functions, so none of them measures the thing the
study deploys. This script does: it personalizes on **that function's labels
only** and evaluates on **that function's temporal tail only**.

WHAT THIS IS NOT
----------------
It selects NOTHING. tau, the population configuration, the ANIL configuration
and the LODO checkpoints are all frozen inputs, read from the artifacts the
earlier stages already wrote. There is no grid, no ranking and no
``selected_*.json`` output. Adding one would re-open the selection-validity
question (``docs/selection_validity.md``) on a query tail far too small to
support it -- see the warning below.

WHAT CHANGES RELATIVE TO ``compare_arms_k_curve``
-------------------------------------------------
Only the segment filter, and two things that follow from it:

  * **A dense K grid.** ``sweep_train_frac.K_GRID_BASE`` runs to 60 and is spaced
    for pools of 66-99 segments. Restricted to one function a driver's pool is
    13-22 segments, so the fixed grid would return ~10 usable points with gaps
    exactly where the curve moves. Every integer K is used instead, which is
    what "all values for K until the maximum possible" asks for and costs less
    compute than the pooled sweep did.
  * **Ragged K.** Pools differ per driver (13-22), so above ``k_common =
    min(n_pool)`` the cohort mean is taken over a SHRINKING and non-random
    subset of drivers. Driver difficulty spans ~0.2 to ~1.9 set-MAE here, an
    order of magnitude more than any arm effect, so a mean over a changing
    driver set is not comparable to the one beside it. The headline statistics
    are therefore computed over ``K <= k_common`` only, and everything above it
    is plotted dashed and labelled with its driver count.

READ THE NOISE WARNING BEFORE READING THE CURVE
-----------------------------------------------
At ``--val-frac 0.3`` the query tail is **5-10 segments per driver** (median 8),
against 28-41 in the pooled sweeps. Per-cell noise is therefore roughly 2x the
~0.10 already measured there, and the per-driver curves will be visibly ragged.
Two further properties of this slice are structural, not fixable by more
compute, and are printed at run time:

  * driver 010 marked LoA 2 on all 24 of their phone-call windows, so a constant
    is the Bayes-optimal predictor for them and set-MAE 0 is attainable by the
    lookup;
  * 003, 007 and 009 used only two distinct levels.

The cohort-level paired comparison over 12 drivers is still meaningful -- it is
blocked on driver, so per-driver difficulty cancels exactly -- but a single
driver's curve is not.

Usage::

    python -m ProVoice.training_scripts.phone_call_k_curve \
        --l2sp-ckpt-dir trained_models/lodo \
        --anil-ckpt-dir trained_models/lodo_anil \
        --embed-fcd --steps 6000 \
        --outdir results/phone_call_k_curve
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import os
import pathlib
import subprocess
import sys
import tempfile
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ProVoice.models.xlstm_model import load_checkpoint
from ProVoice.models.head_adapt import (
    adapt_head, install_fcd_head, DEFAULT_ADAPT_LR, DEFAULT_ADAPT_STEPS,
)
from ProVoice.models.train_XLSTM import iter_jsonl, normalize_row, set_mae, set_accuracy
from ProVoice.training_scripts.baseline_lookup import best_constant
from ProVoice.training_scripts.folds import ALL_PIDS

# Imported, not re-implemented: these curves are read beside the pooled ones, so
# the segment construction, the backbone pass and the metrics must be the SAME
# code, not an equivalent copy.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "..", "..", "scripts"))
from sweep_train_frac import (  # noqa: E402
    build_segments, embed_segments, evaluate,
)

DEFAULT_FUNCTION = "Respond to a phone call"
GRAD_NORM_WARN = 1e-3
ARMS = ("l2sp", "anil")


def read_json(p: pathlib.Path) -> Optional[Dict]:
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


def load_driver_rows(src: pathlib.Path, pid: str, want_key: str) -> pd.DataFrame:
    """One driver's rows, restricted to ONE function.

    Matching goes through ``fcd_config.resolve_function_key`` rather than string
    equality, because that is the same canonicalization the FCD vector itself is
    looked up with -- so "the segments this filter keeps" and "the segments that
    carry this function's FCD vector" cannot come apart. It also picks up the
    legacy spellings ('Start a phone call'), which a literal comparison would
    silently drop.

    MEMORY: ``iter_jsonl`` streams. The full file parses to a measured ~4 GB as
    raw dicts; one driver's phone-call rows are roughly 1/60th of that, and the
    filter runs BEFORE ``normalize_row`` so the discarded 98 % is never copied.
    """
    from ProVoice.fcd_config import resolve_function_key
    rows = []
    for r in iter_jsonl(src):
        if str(r.get("participantid", "")) != pid:
            continue
        if resolve_function_key(str(r.get("functionname", "") or "")) != want_key:
            continue
        rows.append(normalize_row(r))
    return pd.DataFrame(rows)


def lookup_curve(vs: Sequence[np.ndarray], n_val: int, ks: Sequence[int]) -> List[dict]:
    """Model-free reference: the set-MAE-optimal CONSTANT from the first K labels.

    Within a single function the ``(driver x function)`` lookup of
    ``baseline_lookup`` collapses to exactly one number per driver, so this is
    that baseline's within-function form rather than a new one -- ``best_constant``
    is imported from it, not reimplemented.

    It is scored on the SAME ``vs`` array the model is scored on, not re-derived
    from the labels CSV. That matters: ``build_segments`` drops segments with
    missing or all-zero ``Level_*``, so a baseline read independently from the CSV
    could be evaluated on a slightly different set and the comparison would be
    off by whatever those segments contain.

    K=0 is omitted deliberately -- with no labels there is no constant to pick,
    and inventing one (a cohort mode) would give the baseline population
    information the K=0 model row does not have.
    """
    V = np.stack(vs, axis=0)
    pool, val = V[: len(V) - n_val], V[len(V) - n_val:]
    out = []
    for k in ks:
        if k < 1 or k > len(pool):
            continue
        c = best_constant(pool[:k])
        if c is None:
            continue
        pred = np.full(len(val), c, dtype=int)
        out.append({"k": k, "set_mae": set_mae(val, pred),
                    "set_acc": set_accuracy(val, pred), "constant": int(c)})
    return out


def curve_for_arm(ckpt: pathlib.Path, df: pd.DataFrame, tau: float, val_frac: float,
                  steps: int, lr: float, embed_fcd: bool, device: str,
                  k_step: int) -> Tuple[List[dict], dict]:
    """Every integer K from 0 to the pool size, for ONE driver and ONE arm.

    The backbone runs once. Everything after ``embed_segments`` is a ~260-parameter
    convex fit on a (K x 76) tensor, which is why the dense grid is affordable.
    """
    model, arch = load_checkpoint(str(ckpt))
    install_fcd_head(model, embed_fcd)
    # ORDER MATTERS, and BOTH halves do. install_fcd_head builds a new Linear on
    # the head's CURRENT device, so it has to run before the move, not after.
    #
    # .to(device): embed_segments does `model.in_proj(xb.to(device))` and never
    # moves the model itself -- load_checkpoint returns it on CPU, so without this
    # the first forward pass dies with mat1 on cuda:0 and the weights on cpu.
    #
    # .eval(): the population config carries dropout 0.2, which is ACTIVE in a
    # freshly-loaded module. Omitting this raises nothing -- it silently randomizes
    # every embedding, so the curve would be noise no one could trace back here.
    model.to(device).eval()
    head_type = arch.get("head_type", "softmax")
    gids, Xs, vs = build_segments(df, window_seconds=arch.get("window_seconds"),
                                  resample_hz=arch.get("resample_hz"))
    n_seg = len(gids)
    if n_seg < 4:
        return [], {"gids": gids, "n_seg": n_seg, "skip": "fewer than 4 segments"}

    Z = embed_segments(model, Xs, vs, arch["context_length"], device)
    V = torch.from_numpy(np.stack(vs, axis=0))
    pop_head = model.head.to(Z.device)

    # The tail is the chronologically LAST val_frac of this function's segments --
    # not of the driver's session. Those are different sets: phone-call windows are
    # interleaved with the other four functions, so this tail spans most of the
    # session in wall-clock terms while containing only phone-call labels. That is
    # the correct query set for a study that deploys on this function alone.
    n_val = max(1, round(val_frac * n_seg))
    n_pool = n_seg - n_val
    Zpool, Vpool = Z[:n_pool], V[:n_pool]
    Zval, Vval = Z[n_pool:], V[n_pool:]

    base = evaluate(pop_head, Zval, Vval, head_type)
    rows = [{"k": 0, "set_mae": base["mae"], "set_acc": base["acc"],
             "set_qwk": base["qwk"], "set_macro_f1": base["f1"],
             "grad_norm": 0.0, "l2sp": 0.0, "adapt_steps": 0}]
    for k in range(1, n_pool + 1, max(1, k_step)):
        head, info = adapt_head(pop_head, Zpool[:k], Vpool[:k], tau=tau,
                                head_type=head_type, steps=steps, lr=lr)
        m = evaluate(head, Zval, Vval, head_type)
        rows.append({"k": k, "set_mae": m["mae"], "set_acc": m["acc"],
                     "set_qwk": m["qwk"], "set_macro_f1": m["f1"],
                     "grad_norm": info["grad_norm"], "l2sp": info["l2sp"],
                     "adapt_steps": int(info["steps"])})
    meta = {"gids": gids, "n_seg": n_seg, "n_pool": n_pool, "n_val": n_val,
            "base_set_mae": base["mae"], "base_set_acc": base["acc"],
            "head_in": int(pop_head.in_features),
            "embed_fcd": int(pop_head.in_features > model.embedding_dim),
            "head_type": head_type, "vs": vs, "skip": None}
    return rows, meta


def sweep_drivers(pids: Sequence[str], src: pathlib.Path, want_key: str,
                  ckpt_dirs: Dict[str, pathlib.Path], prefixes: Dict[str, str],
                  tau: float, args, device: str) -> pd.DataFrame:
    """All arms for all drivers. The driver's rows are read ONCE and shared.

    Reading per driver rather than per (driver, arm) halves the passes over a 4 GB
    file. The rows are arm-independent by construction -- only the backbone that
    embeds them differs -- so sharing them is not a shortcut, it is the same data
    reaching both arms, which is what makes the comparison paired at the segment
    level rather than merely at the driver level.
    """
    out: List[dict] = []
    for pid in pids:
        df = load_driver_rows(src, pid, want_key)
        if df.empty:
            print(f"[{pid}] no rows for {want_key!r} — SKIPPED", flush=True)
            continue

        seen_gids, seen_vs, per_arm = None, None, {}
        for arm in ARMS:
            ck = ckpt_dirs[arm] / f"{prefixes[arm]}{pid}.pt"
            if not ck.exists():
                print(f"[{pid}][{arm}] no checkpoint at {ck} — SKIPPED", flush=True)
                continue
            rows, meta = curve_for_arm(ck, df, tau, args.val_frac, args.steps,
                                       args.lr, args.embed_fcd, device, args.k_step)
            if meta.get("skip"):
                print(f"[{pid}][{arm}] {meta['skip']} — SKIPPED", flush=True)
                continue
            # BOTH ARMS MUST SEE THE SAME SEGMENTS. build_segments reads
            # window_seconds/resample_hz out of each checkpoint's own arch, so two
            # arms built under different data contracts would silently produce
            # different segment sets and the "paired" difference below would be
            # comparing different query tails. They agree by construction today
            # (ANIL warm-starts from the population checkpoint), which is exactly
            # the kind of invariant that breaks quietly when someone rebuilds one
            # arm alone.
            if seen_gids is None:
                seen_gids, seen_vs = meta["gids"], meta["vs"]
            elif meta["gids"] != seen_gids:
                raise SystemExit(
                    f"[{pid}] the arms disagree on the segment set: "
                    f"{len(seen_gids)} vs {len(meta['gids'])} segments. Their "
                    f"checkpoints carry different window_seconds/resample_hz, so "
                    f"they cannot be compared on a common tail.")
            per_arm[arm] = meta
            for r in rows:
                out.append({"pid": pid, "arm": arm, "function": want_key,
                            "n_seg": meta["n_seg"], "n_pool": meta["n_pool"],
                            "n_val": meta["n_val"],
                            "base_set_mae": meta["base_set_mae"],
                            "head_in": meta["head_in"],
                            "embed_fcd": meta["embed_fcd"], "tau": tau, **r})
            print(f"[{pid}][{arm}] n_seg={meta['n_seg']} pool={meta['n_pool']} "
                  f"tail={meta['n_val']} floor={meta['base_set_mae']:.3f} "
                  f"K=1..{meta['n_pool']}", flush=True)

        if args.with_lookup and seen_gids is not None:
            m0 = next(iter(per_arm.values()))
            for r in lookup_curve(seen_vs, m0["n_val"], range(1, m0["n_pool"] + 1,
                                                              max(1, args.k_step))):
                out.append({"pid": pid, "arm": "lookup", "function": want_key,
                            "n_seg": m0["n_seg"], "n_pool": m0["n_pool"],
                            "n_val": m0["n_val"],
                            "base_set_mae": m0["base_set_mae"],
                            "head_in": -1, "embed_fcd": -1, "tau": float("nan"),
                            "set_qwk": float("nan"), "set_macro_f1": float("nan"),
                            "grad_norm": 0.0, "l2sp": 0.0, "adapt_steps": 0, **r})
    return pd.DataFrame(out)


def paired_over_k(a: pd.DataFrame, b: pd.DataFrame, k_max: int) -> pd.DataFrame:
    """Per-driver mean set-MAE difference (b - a) over 1 <= K <= k_max.

    K=0 is excluded: it is the unadapted floor, identical for every K condition
    within an arm, and averaging it into a personalization curve dilutes the
    effect toward zero by however many points the grid happens to contain.
    """
    def per_driver(d):
        d = d[(d["k"] >= 1) & (d["k"] <= k_max)]
        return d.groupby("pid")[["set_mae", "set_acc"]].mean()
    x, y = per_driver(a), per_driver(b)
    common = x.index.intersection(y.index)
    out = pd.DataFrame({
        "pid": common,
        "a_mae": x.loc[common, "set_mae"].to_numpy(),
        "b_mae": y.loc[common, "set_mae"].to_numpy(),
        "a_acc": x.loc[common, "set_acc"].to_numpy(),
        "b_acc": y.loc[common, "set_acc"].to_numpy(),
    })
    out["mae_delta"] = out["b_mae"] - out["a_mae"]
    out["acc_delta"] = out["b_acc"] - out["a_acc"]
    return out.round(4)


def report(df: pd.DataFrame, outdir: pathlib.Path, tau: float, args,
           want_key: str) -> Dict[str, object]:
    arms = [a for a in ARMS if a in set(df["arm"])]
    extras = [a for a in sorted(set(df["arm"])) if a not in ARMS]
    if not arms:
        raise SystemExit("no learned arm produced any rows")

    meta = (df[df["arm"] == arms[0]]
            .groupby("pid")[["n_seg", "n_pool", "n_val", "base_set_mae"]].first())
    print(f"\n=== {want_key} — per driver ===")
    print(f"{'pid':>5} {'labels':>7} {'pool':>6} {'tail':>6} "
          + " ".join(f"{'floor ' + a:>12}" for a in arms))
    for pid, r in meta.iterrows():
        floors = []
        for a in arms:
            s = df[(df["arm"] == a) & (df["pid"] == pid)]["base_set_mae"]
            floors.append(f"{s.iloc[0]:12.3f}" if len(s) else f"{'-':>12}")
        print(f"{pid:>5} {int(r['n_seg']):7d} {int(r['n_pool']):6d} "
              f"{int(r['n_val']):6d} " + " ".join(floors))

    k_common = int(meta["n_pool"].min())
    k_max_any = int(meta["n_pool"].max())
    tails = meta["n_val"]
    print(f"\n[pool] K=1..{k_common} has all {len(meta)} drivers; "
          f"the longest driver reaches K={k_max_any}.")
    print(f"[tail] query tail {int(tails.min())}-{int(tails.max())} segments "
          f"(median {int(tails.median())}).")
    if tails.median() < 15:
        print(f"[tail][WARN] the pooled sweeps evaluate on 28-41 segments; this tail is "
              f"~{tails.median() / 34:.2f}x that, so per-(driver, K) noise is roughly "
              f"{np.sqrt(34 / max(tails.median(), 1)):.1f}x larger. Read the COHORT "
              f"curve and the paired statistic; a single driver's curve is not "
              f"resolvable at this tail size.")

    # Convergence. lambda = tau/(2K), so the anchor weakens as K grows and the
    # largest K in each driver's grid is where the optimizer is most stretched.
    # A residual difference BETWEEN arms at the same K biases the comparison, so
    # this is reported per arm rather than pooled.
    print()
    for a in arms:
        g = df[(df["arm"] == a) & (df["k"] >= 1)]["grad_norm"]
        bad = int((g > GRAD_NORM_WARN).sum())
        print(f"[converge][{a}] max |grad| = {g.max():.2e}, "
              f"{len(g) - bad}/{len(g)} cells below {GRAD_NORM_WARN:g}"
              + ("" if bad == 0 else "  <-- HIGH, raise --steps"))

    print(f"\n=== cohort curve (mean over drivers at each K) ===")
    print(f"{'K':>4} {'n':>4} " + " ".join(f"{a:>9}" for a in arms + extras))
    for k in sorted(df["k"].unique()):
        cells, n_at_k = [], 0
        for a in arms + extras:
            s = df[(df["arm"] == a) & (df["k"] == k)]["set_mae"]
            n_at_k = max(n_at_k, len(s))
            cells.append(f"{s.mean():9.3f}" if len(s) else f"{'-':>9}")
        flag = "" if k <= k_common else "   (partial cohort)"
        print(f"{int(k):>4} {n_at_k:>4} " + " ".join(cells) + flag)

    summary: Dict[str, object] = {
        "function": want_key, "tau": tau, "val_frac": args.val_frac,
        "steps": args.steps, "embed_fcd": int(args.embed_fcd),
        "n_drivers": int(len(meta)), "k_common": k_common, "k_max_any": k_max_any,
        "tail_min": int(tails.min()), "tail_median": float(tails.median()),
        "tail_max": int(tails.max()),
        "floors": {a: float(df[df["arm"] == a].groupby("pid")["base_set_mae"]
                            .first().mean()) for a in arms},
    }

    # EVERY headline statistic is confined to K <= k_common. Above it the cohort
    # changes composition with K, and driver difficulty here spans ~0.2 to ~1.9
    # set-MAE -- so a mean taken over a shrinking driver set moves for reasons
    # that have nothing to do with K.
    print(f"\n=== paired, K=1..{k_common}, all {len(meta)} drivers ===")
    l2 = df[df["arm"] == "l2sp"]
    for other in [a for a in arms if a != "l2sp"] + extras:
        pr = paired_over_k(l2, df[df["arm"] == other], k_common)
        if pr.empty:
            continue
        pr = pr.rename(columns={"a_mae": "l2sp_mae", "b_mae": f"{other}_mae",
                                "a_acc": "l2sp_acc", "b_acc": f"{other}_acc"})
        pr.to_csv(outdir / f"paired_l2sp_vs_{other}.csv", index=False)
        d = pr["mae_delta"].to_numpy()
        n = len(d)
        mean_d, se_d = float(d.mean()), float(d.std(ddof=1) / np.sqrt(n))
        try:
            from scipy.stats import wilcoxon
            p = float(wilcoxon(pr[f"{other}_mae"], pr["l2sp_mae"]).pvalue)
        except Exception:
            p = float("nan")
        verdict = (("%s better" % other if mean_d < 0 else "l2sp better")
                   if abs(mean_d) > se_d else "indistinguishable (|delta| < 1 SE)")
        print(f"\n  {other} - l2sp = {mean_d:+.4f} +/- {se_d:.4f} SE over {n} drivers"
              + (f"; Wilcoxon p = {p:.4f}" if p == p else ""))
        print(f"  {other} better on {int((d < 0).sum())}/{n} drivers -> {verdict}")
        print(f"  {'pid':>5} {'l2sp':>8} {other:>8} {'delta':>8}")
        for _, r in pr.iterrows():
            print(f"  {r['pid']:>5} {r['l2sp_mae']:8.3f} {r[f'{other}_mae']:8.3f} "
                  f"{r['mae_delta']:+8.3f}")
        summary[f"{other}_vs_l2sp"] = {
            "mean_mae_delta": mean_d, "se": se_d, "wilcoxon_p": p,
            "better_on": int((d < 0).sum()), "n": n, "verdict": verdict}
    return summary


_COLORS = {"l2sp": "#4C72B0", "anil": "#DD8452", "lookup": "#55A868"}
_LABELS = {"l2sp": "L2-SP", "anil": "ANIL", "lookup": "lookup (best constant)"}


def _style(arm: str) -> Tuple[str, str]:
    return _COLORS.get(arm, "#937860"), _LABELS.get(arm, arm)


def plot_cohort(df: pd.DataFrame, outdir: pathlib.Path, k_common: int,
                tau: float, want_key: str, n_drivers: int) -> None:
    fig, ax = plt.subplots(figsize=(7.8, 4.8))
    for arm in sorted(set(df["arm"])):
        c = df[(df["arm"] == arm) & (df["k"] >= 1)].groupby("k")["set_mae"].mean().sort_index()
        col, lab = _style(arm)
        # Split at k_common: the solid part is a mean over ALL drivers, the dashed
        # part over whichever drivers happen to have a pool that long. Drawing them
        # as one line would invite reading a composition change as a trend.
        lo = c[c.index <= k_common]
        hi = c[c.index >= k_common]
        ax.plot(lo.index, lo.values, marker="o", ms=3, lw=1.8, color=col, label=lab)
        if len(hi) > 1:
            ax.plot(hi.index, hi.values, ls="--", lw=1.3, color=col, alpha=0.75)
    for arm in [a for a in ARMS if a in set(df["arm"])]:
        f = df[df["arm"] == arm].groupby("pid")["base_set_mae"].first().mean()
        ax.axhline(f, ls=":", lw=1.1, color=_COLORS.get(arm, "0.4"), alpha=0.8)
    ax.axvline(k_common, ls="-", lw=0.9, color="0.6")
    ax.annotate(f"all {n_drivers} drivers\n<- | ->\npartial cohort",
                xy=(k_common, ax.get_ylim()[1]), xytext=(2, -4),
                textcoords="offset points", va="top", fontsize=7, color="0.35")
    ax.set_xlabel(f"personalization labels K for '{want_key}'  (20 s windows)")
    ax.set_ylabel("set-MAE on this function's tail (lower is better)")
    ax.set_title(f"'{want_key}' only — {n_drivers} drivers (LODO), tau={tau:g}\n"
                 f"dotted = unadapted population floor per arm", fontsize=10)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(outdir / "phone_call_k_curve.png", dpi=150)
    print(f"[OK] plot  -> {outdir / 'phone_call_k_curve.png'}")


def plot_per_driver(df: pd.DataFrame, outdir: pathlib.Path, want_key: str) -> None:
    """One panel per driver. Ragged and noisy BY CONSTRUCTION at a 5-10 segment
    tail -- included because the cohort mean hides which drivers are lookup-shaped
    (a flat near-zero lookup line) and which have signal a model can reach."""
    pids = sorted(set(df["pid"]))
    ncol = 4
    nrow = int(np.ceil(len(pids) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.3 * ncol, 2.5 * nrow),
                             sharex=False, sharey=True)
    axes = np.atleast_1d(axes).ravel()
    for ax, pid in zip(axes, pids):
        d = df[df["pid"] == pid]
        for arm in sorted(set(d["arm"])):
            c = d[(d["arm"] == arm) & (d["k"] >= 1)].set_index("k")["set_mae"].sort_index()
            col, lab = _style(arm)
            ax.plot(c.index, c.values, marker="o", ms=2.5, lw=1.4, color=col, label=lab)
        for arm in [a for a in ARMS if a in set(d["arm"])]:
            f = d[d["arm"] == arm]["base_set_mae"].iloc[0]
            ax.axhline(f, ls=":", lw=1.0, color=_COLORS.get(arm, "0.4"), alpha=0.8)
        tail = int(d["n_val"].iloc[0])
        ax.set_title(f"{pid}  (tail n={tail})", fontsize=9)
        ax.grid(alpha=0.3)
    for ax in axes[len(pids):]:
        ax.axis("off")
    axes[0].legend(fontsize=7)
    fig.supxlabel("K", fontsize=9)
    fig.supylabel("set-MAE", fontsize=9)
    fig.suptitle(f"'{want_key}' only — per driver", fontsize=11)
    fig.tight_layout()
    fig.savefig(outdir / "phone_call_per_driver.png", dpi=150)
    print(f"[OK] plot  -> {outdir / 'phone_call_per_driver.png'}")


def head_width(d: pathlib.Path, prefix: str, embed_fcd: bool) -> Tuple:
    """(stored width, width AS ADAPTED) — same guard as compare_arms_k_curve.

    Comparing STORED widths is wrong: --embed-fcd has asymmetric on-disk
    consequences (the L2-SP head is widened in memory at load, the ANIL head is a
    meta-parameter and is persisted wide), so a correct pair stores 64 and 76 and
    adapts 76 and 76.
    """
    for f in sorted(d.glob(f"{prefix}*.pt")):
        try:
            model, _ = load_checkpoint(str(f))
            stored = int(model.head.in_features)
            install_fcd_head(model, embed_fcd)
            return stored, int(model.head.in_features)
        except Exception:
            continue
    return None, None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in", dest="in_jsonl", default="data/labeled_data.jsonl")
    ap.add_argument("--function", default=DEFAULT_FUNCTION,
                    help="The ONE function to personalize and evaluate on. Matched "
                         "through fcd_config.resolve_function_key, so legacy spellings "
                         "resolve; a name that falls back to UNKNOWN_FUNCTION_KEY is "
                         "refused rather than silently matching every unrecognised name.")
    ap.add_argument("--l2sp-ckpt-dir", dest="l2sp_ckpt_dir", default="trained_models/lodo")
    ap.add_argument("--anil-ckpt-dir", dest="anil_ckpt_dir", default="trained_models/lodo_anil")
    ap.add_argument("--l2sp-prefix", dest="l2sp_prefix", default="pop_heldout_")
    ap.add_argument("--anil-prefix", dest="anil_prefix", default="anil_heldout_")
    ap.add_argument("--outdir", default="results/phone_call_k_curve")
    ap.add_argument("--pids", default="", help="Comma-separated subset (default: all 12).")
    ap.add_argument("--selected-tau", dest="selected_tau",
                    default="results/l2sp_sweep/selected_tau.json")
    ap.add_argument("--tau", type=float, default=None,
                    help="Override the frozen tau. NOT swept here -- tau was chosen once, "
                         "on the pooled L2-SP sweep, and both arms run that value. "
                         "Re-selecting it on this slice would tune the shared adaptation "
                         "procedure on a 5-10 segment tail.")
    ap.add_argument("--val-frac", dest="val_frac", type=float, default=0.3,
                    help="Chronologically-last fraction of THIS FUNCTION's segments held "
                         "out as the query tail. 0.3 matches compare_arms_k_curve so the "
                         "two are read on the same protocol -- but note it buys only 5-10 "
                         "segments here against 28-41 there.")
    ap.add_argument("--steps", type=int, default=6000,
                    help="Full-batch adaptation steps per cell. 6000 matches the pooled "
                         "sweeps. K is small here so the objective is easy, but keep it "
                         "identical rather than tuned down: a residual difference between "
                         "the arms at the same K biases the comparison.")
    ap.add_argument("--lr", type=float, default=DEFAULT_ADAPT_LR)
    ap.add_argument("--k-step", dest="k_step", type=int, default=1,
                    help="Stride of the dense K grid (default 1 = every integer K). The "
                         "pools are 13-22 segments, so every K is affordable and the "
                         "fixed K_GRID_BASE would leave gaps where the curve moves.")
    ap.add_argument("--embed-fcd", dest="embed_fcd", action="store_true",
                    help="Adapt an FCD-augmented head in BOTH arms, matching how the "
                         "pooled comparison was run. NOTE: restricted to one function the "
                         "FCD block is CONSTANT across every segment, so it acts as a "
                         "second bias rather than as information. Pass it anyway when the "
                         "ANIL checkpoints were meta-trained with it -- otherwise the "
                         "widened head is a different object from its own anchor.")
    ap.add_argument("--no-lookup", dest="with_lookup", action="store_false",
                    help="Omit the model-free best-constant reference.")
    ap.add_argument("--jobs", type=int, default=1,
                    help="Shard drivers across subprocesses (wall-clock only; every "
                         "statistic is computed once, in the parent, over the merged "
                         "table).")
    ap.add_argument("--threads-per-job", dest="threads_per_job", type=int, default=0,
                    help="Torch intra-op threads per shard, set in the child's ENVIRONMENT "
                         "since torch sizes its pool at import.")
    args = ap.parse_args()

    from ProVoice.fcd_config import resolve_function_key, UNKNOWN_FUNCTION_KEY
    want_key = resolve_function_key(args.function)
    if want_key == UNKNOWN_FUNCTION_KEY:
        raise SystemExit(
            f"--function {args.function!r} does not resolve to a known function; it would "
            f"match every unrecognised name at once. Known keys are listed in "
            f"src/ProVoice/fcd_config.py.")
    pids = [p.strip() for p in args.pids.split(",") if p.strip()] or list(ALL_PIDS)
    outdir = pathlib.Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    csv_path = outdir / "phone_call_k_curve.csv"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tau = args.tau
    if tau is None:
        sel = read_json(pathlib.Path(args.selected_tau))
        if not sel or "tau" not in sel:
            raise SystemExit(f"No tau (expected {args.selected_tau}); pass --tau.")
        tau = float(sel["tau"])
    print(f"[function] {want_key!r}")
    print(f"[tau] {tau:g} - frozen, the SAME value for both arms; nothing is selected here")

    ckpt_dirs = {"l2sp": pathlib.Path(args.l2sp_ckpt_dir),
                 "anil": pathlib.Path(args.anil_ckpt_dir)}
    prefixes = {"l2sp": args.l2sp_prefix, "anil": args.anil_prefix}

    _sl, _wl = head_width(ckpt_dirs["l2sp"], prefixes["l2sp"], args.embed_fcd)
    _sa, _wa = head_width(ckpt_dirs["anil"], prefixes["anil"], args.embed_fcd)
    if _wl is not None and _wa is not None and _wl != _wa:
        raise SystemExit(
            f"ARM MISMATCH: with --embed-fcd={int(args.embed_fcd)} the L2-SP arm adapts a "
            f"{_wl}-input head (stored {_sl}) and the ANIL arm a {_wa}-input one "
            f"(stored {_sa}). They adapt different objects.")
    if _wl is not None:
        print(f"[arms] adapting a {_wl}-input head on both arms "
              f"[stored: l2sp {_sl}, anil {_sa}]")
        if args.embed_fcd and _sa is not None and _sa != _wa:
            print("[arms][WARN] the ANIL checkpoints were meta-trained WITHOUT --embed-fcd "
                  "and are widened here with a zero block.")

    # Shard over drivers. Mirrors sweep_l2sp_tau: the children do the per-driver
    # work and write partial tables; the PARENT merges and does every statistic
    # once, over the full driver set. A child must never summarise -- k_common and
    # the paired test are defined over all 12 drivers, and a shard would compute
    # both on its own subset.
    if args.jobs > 1 and not os.environ.get("_PCK_SHARD"):
        shards = [sh for sh in (pids[i::args.jobs] for i in range(args.jobs)) if sh]
        print(f"[jobs] {len(pids)} driver(s) over {len(shards)} shard(s): "
              + " | ".join(",".join(sh) for sh in shards), flush=True)
        env = dict(os.environ, _PCK_SHARD="1")
        if args.threads_per_job > 0:
            for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                      "NUMEXPR_NUM_THREADS"):
                env[v] = str(args.threads_per_job)
        base = list(sys.argv[1:])

        def strip(flags):
            out, skip = [], False
            for a in base:
                if skip:
                    skip = False
                    continue
                if a in flags:
                    skip = True
                    continue
                if any(a.startswith(f + "=") for f in flags):
                    continue
                out.append(a)
            return out

        with tempfile.TemporaryDirectory() as td:
            def run(i_sh):
                i, sh = i_sh
                od = pathlib.Path(td) / f"shard{i}"
                cmd = ([sys.executable, "-m", "ProVoice.training_scripts.phone_call_k_curve"]
                       + strip({"--pids", "--outdir", "--jobs", "--threads-per-job"})
                       + ["--pids", ",".join(sh), "--outdir", str(od), "--jobs", "1"])
                r = subprocess.run(cmd, env=env)
                return i, od / "phone_call_k_curve.csv", r.returncode

            frames, failed = [], 0
            with cf.ThreadPoolExecutor(max_workers=len(shards)) as ex:
                for i, cp, rc in ex.map(run, list(enumerate(shards))):
                    if rc != 0 or not cp.exists():
                        print(f"[jobs] shard {i} FAILED (exit {rc})")
                        failed += 1
                        continue
                    frames.append(pd.read_csv(cp, dtype={"pid": str}))
            if failed or not frames:
                raise SystemExit(f"{failed} shard(s) failed; not writing a partial table")
            df = pd.concat(frames, ignore_index=True)
    else:
        df = sweep_drivers(pids, pathlib.Path(args.in_jsonl), want_key,
                           ckpt_dirs, prefixes, tau, args, device)

    if df.empty:
        raise SystemExit(f"no rows produced for {want_key!r}")
    df = df.sort_values(["pid", "arm", "k"]).reset_index(drop=True)
    df.to_csv(csv_path, index=False)
    print(f"\n[OK] table -> {csv_path}  ({len(df)} rows)")

    # A shard writes its partial table and stops. Everything below is defined over
    # the full driver set.
    if os.environ.get("_PCK_SHARD"):
        return

    summary = report(df, outdir, tau, args, want_key)
    k_common = int(summary["k_common"])
    plot_cohort(df, outdir, k_common, tau, want_key, int(summary["n_drivers"]))
    plot_per_driver(df, outdir, want_key)
    (outdir / "phone_call_k_curve.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[OK] -> {outdir / 'phone_call_k_curve.json'}")


if __name__ == "__main__":
    main()
