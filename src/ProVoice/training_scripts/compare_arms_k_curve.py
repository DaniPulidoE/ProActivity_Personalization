r"""Stage C — the K curve for BOTH arms, and the paired comparison between them.

This is the study's primary offline result: personalization quality against the
number of driver labels K, for the L2-SP arm and the ANIL arm, and the
**paired per-driver difference** between them.

WHAT MAKES IT A COMPARISON
--------------------------
The arms differ in exactly ONE thing: theta_init. Everything else is held
identical, and most of it by construction rather than by convention —

  * the same ``head_adapt.adapt_head``, the same 2000 full-batch steps,
  * the same tau (frozen from the L2-SP sweep, used by ANIL's inner problem too),
  * the same K grid, the same temporal tail, the same 12 LODO folds,
  * per driver, the ANIL init warm-started from that fold's OWN L2-SP population
    checkpoint, so neither arm has seen the held-out driver.

Both curves therefore come out of ``sweep_l2sp_tau`` with only ``--ckpt-dir``
changed, which is the point of having unified the adaptation.

WHY PAIRED
----------
Driver difficulty dominates everything here: per-driver constant floors range
~0.47 to ~1.58, an order of magnitude more spread than any plausible arm effect.
Comparing arm MEANS across drivers throws that pairing away and buries the
effect in between-driver variance. Both arms are evaluated on the SAME driver,
the SAME tail and the SAME K, so differencing within a (driver, K) cell cancels
driver difficulty exactly and leaves the initialization effect.

The headline statistic is the per-driver mean difference over the deployable K
range, tested with a **Wilcoxon signed-rank** over the 12 drivers — paired,
distribution-free, and appropriate at n=12 where normality is untestable.

READING A NULL
--------------
A null (ANIL ~ L2-SP) is an informative finding at ~10 meta-training drivers and
was anticipated by the design. It is only interpretable if the ANIL arm was
genuinely tuned, so this script reports the stage-A caveats — grid-edge outer_lr,
whether meta-training's query loss ever dropped — alongside the result rather
than leaving them in a different file.

Usage::

    python -m ProVoice.training_scripts.compare_arms_k_curve \
        --l2sp-ckpt-dir trained_models/lodo \
        --anil-ckpt-dir trained_models/lodo_anil \
        --outdir results/arm_comparison
"""
from __future__ import annotations

import argparse
import json
import pathlib
import subprocess
import sys
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ARMS = ("l2sp", "anil")


def read_json(p: pathlib.Path) -> Optional[Dict]:
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


def ensure_curve(arm: str, ckpt_dir: pathlib.Path, prefix: str, outdir: pathlib.Path,
                 tau: float, args) -> Optional[pathlib.Path]:
    """Run sweep_l2sp_tau for one arm at the SINGLE frozen tau, or reuse its CSV.

    tau is not swept here. It was chosen once, on the L2-SP arm, and both arms
    run it — re-sweeping per arm would let each arm tune the shared adaptation
    procedure, which is exactly the confound the design forbids.
    """
    arm_dir = outdir / arm
    csv_path = arm_dir / "l2sp_tau_sweep.csv"
    if csv_path.exists() and not args.force:
        print(f"[{arm}] reusing {csv_path}")
        return csv_path
    if not ckpt_dir.exists():
        print(f"[{arm}] SKIPPED — no checkpoints at {ckpt_dir}")
        return None
    print(f"[{arm}] sweeping K from {ckpt_dir} at tau={tau:g}", flush=True)
    cmd = [sys.executable, "-m", "ProVoice.training_scripts.sweep_l2sp_tau",
           "--in", args.in_jsonl, "--ckpt-dir", str(ckpt_dir),
           "--outdir", str(arm_dir), "--taus", f"{tau:g}",
           "--ckpt-prefix", prefix,
           "--val-frac", str(args.val_frac), "--k-cap", str(args.k_cap),
           "--max-points", str(args.max_points)]
    if args.embed_fcd:
        # ONE flag, BOTH arms. Passing it here rather than expecting two earlier
        # commands to agree is what makes arm symmetry structural instead of a
        # convention someone has to remember; the head-width guard above is then
        # a backstop for checkpoints built elsewhere, not the only defence.
        cmd += ["--embed-fcd"]
    if subprocess.run(cmd).returncode != 0 or not csv_path.exists():
        print(f"[{arm}] FAILED")
        return None
    return csv_path


def load_arm(path: pathlib.Path, arm: str) -> pd.DataFrame:
    # pid as STRING: participant ids are zero-padded ('001'), and pandas would
    # otherwise infer int64 and render them as 1, 2, 3. Cosmetic in the printed
    # table, but the two arms are joined on this column — if the arms' CSVs ever
    # parsed it differently (one int, one str) the intersection would come back
    # empty and the paired comparison would silently report zero drivers.
    df = pd.read_csv(path, dtype={"pid": str})
    df["arm"] = arm
    return df


def paired_table(l2: pd.DataFrame, an: pd.DataFrame, k_cap: int) -> pd.DataFrame:
    """Per-driver mean difference over K <= k_cap. Positive = ANIL worse."""
    def per_driver(df):
        d = df[df["k"] <= k_cap]
        return d.groupby("pid")[["set_mae", "set_acc"]].mean()
    a, b = per_driver(l2), per_driver(an)
    common = a.index.intersection(b.index)
    out = pd.DataFrame({
        "pid": common,
        "l2sp_mae": a.loc[common, "set_mae"].to_numpy(),
        "anil_mae": b.loc[common, "set_mae"].to_numpy(),
        "l2sp_acc": a.loc[common, "set_acc"].to_numpy(),
        "anil_acc": b.loc[common, "set_acc"].to_numpy(),
    })
    out["mae_delta"] = out["anil_mae"] - out["l2sp_mae"]
    out["acc_delta"] = out["anil_acc"] - out["l2sp_acc"]
    out["anil_better"] = out["mae_delta"] < 0
    return out.round(4)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in", dest="in_jsonl", default="data/labeled_data.jsonl")
    ap.add_argument("--l2sp-ckpt-dir", dest="l2sp_ckpt_dir", default="trained_models/lodo")
    ap.add_argument("--anil-ckpt-dir", dest="anil_ckpt_dir", default="trained_models/lodo_anil")
    ap.add_argument("--l2sp-prefix", dest="l2sp_prefix", default="pop_heldout_")
    ap.add_argument("--anil-prefix", dest="anil_prefix", default="anil_heldout_")
    ap.add_argument("--outdir", default="results/arm_comparison")
    ap.add_argument("--selected-tau", dest="selected_tau",
                    default="results/l2sp_sweep/selected_tau.json")
    ap.add_argument("--selected-anil", dest="selected_anil",
                    default="results/anil_sweep/selected_anil.json")
    ap.add_argument("--tau", type=float, default=None, help="Override the frozen tau.")
    ap.add_argument("--val-frac", dest="val_frac", type=float, default=0.3,
                    help="Chronologically-last fraction of each driver's segments held out "
                         "as the query tail, passed to BOTH arms so they cannot disagree. "
                         "Raised 0.2 -> 0.3 on 2026-08-18: at 0.2 the tail was only 19-27 "
                         "segments per driver, below the 20-segment minimum the "
                         "--adapt-eval path enforces, so each (driver, K) point rested on "
                         "a query set too small to be stable. At 0.3 it is 28-41 (median "
                         "36) and every driver still reaches K=60, the shortest pool being "
                         "66. MUST match run_lodo_population, sweep_l2sp_tau and "
                         "probe_embeddings, or the K=0 floor and these curves are measured "
                         "on different segments.")
    ap.add_argument("--k-cap", dest="k_cap", type=int, default=60)
    ap.add_argument("--max-points", dest="max_points", type=int, default=20)
    ap.add_argument("--embed-fcd", dest="embed_fcd", action="store_true",
                    help="Adapt an FCD-augmented head ([z_64 | FCD_12]) in BOTH arms. "
                         "Applied here rather than per arm so the two cannot disagree: "
                         "the comparison isolates theta_init, so anything else that "
                         "differs between the arms confounds it. Requires nothing of the "
                         "checkpoints -- the population heads stay narrow and are widened "
                         "with a zero block at adaptation time.")
    ap.add_argument("--force", action="store_true", help="Recompute curves even if present.")
    ap.add_argument("--extra-arm", dest="extra_arms", action="append", default=[],
                    metavar="PATH:NAME",
                    help="Additional arm to report, as <csv>:<name>, repeatable. The CSV "
                         "needs only `pid, k, set_mae, set_acc` — the same columns the "
                         "learned arms emit — so a model-free reference drops in without "
                         "special handling. Built for ProVoice.training_scripts."
                         "baseline_lookup, whose per-(driver, function) constant beats the "
                         "trained model on this cohort from K=5 onward; a baseline that "
                         "strong belongs in the headline table, not a footnote. Extra arms "
                         "are summarised and paired against L2-SP, but never define the "
                         "common-driver set (see below) — a reference with different "
                         "coverage must not silently shrink the arms' comparison.")
    args = ap.parse_args()

    tau = args.tau
    if tau is None:
        sel = read_json(pathlib.Path(args.selected_tau))
        if not sel or "tau" not in sel:
            raise SystemExit(f"No tau (expected {args.selected_tau}); pass --tau to override.")
        tau = float(sel["tau"])
    print(f"[tau] {tau:g} — the SAME value for both arms, by design")

    # ARM SYMMETRY: both arms must adapt the SAME object. --embed-fcd widens the
    # head, and an L2-SP arm run with it against an ANIL arm run without it would
    # differ by more than their initialization -- which is the one thing this
    # comparison exists to isolate. Checked from the checkpoints themselves,
    # because the flag lives in two separate earlier commands and nothing else
    # would catch the mismatch.
    def _head_width(d: pathlib.Path, prefix: str) -> tuple:
        """(stored width, width AS ADAPTED) for the first readable checkpoint.

        The two differ, and comparing the STORED widths is wrong: --embed-fcd has
        deliberately asymmetric on-disk consequences. For the L2-SP arm the head is
        widened in memory at load time, so pop_heldout_*.pt stays narrow; for the
        ANIL arm the head is a META-PARAMETER, so meta-training persists the widened
        one. A correct pair of arms therefore stores 64 and 76 and adapts 76 and 76.
        The earlier version of this guard rejected exactly that configuration.

        The effective width is obtained by running the checkpoint through the same
        two functions the sweep runs it through, rather than re-deriving the rule --
        install_fcd_head is idempotent, so an already-wide head is returned untouched.
        """
        from ProVoice.models.head_adapt import install_fcd_head
        from ProVoice.models.xlstm_model import load_checkpoint
        for f in sorted(d.glob(f"{prefix}*.pt")):
            try:
                model, _arch = load_checkpoint(str(f))
                stored = int(model.head.in_features)
                install_fcd_head(model, args.embed_fcd)
                return stored, int(model.head.in_features)
            except Exception:
                continue
        return None, None

    _sl, _wl = _head_width(pathlib.Path(args.l2sp_ckpt_dir), args.l2sp_prefix)
    _sa, _wa = _head_width(pathlib.Path(args.anil_ckpt_dir), args.anil_prefix)
    if _wl is not None and _wa is not None and _wl != _wa:
        raise SystemExit(
            f"ARM MISMATCH: after --embed-fcd={int(args.embed_fcd)} is applied the L2-SP "
            f"arm adapts a {_wl}-input head (stored {_sl}) and the ANIL arm a {_wa}-input "
            f"one (stored {_sa}). They adapt different objects, so the comparison would "
            f"measure more than the initialization. Rebuild one arm to match.")
    if _wl is not None:
        print(f"[arms] adapting a {_wl}-input head on both arms"
              + (" (FCD-augmented)" if _wl != _sl or (_sa is not None and _wl != _sa)
                 else "")
              + f"  [stored: l2sp {_sl}, anil {_sa}]")
        # Symmetric in the OBJECT but not in what meta-training saw: a plain ANIL
        # checkpoint widened here carries a zero FCD block its meta-training never
        # used, so its anchor is uninformed about the task input the L2-SP arm's
        # adaptation will exploit. Not fatal, but it is not the intended arm either.
        # "_sa != _wa" means the ANIL checkpoint had to be widened HERE, which can
        # only happen if meta-training ran without --embed-fcd. No constant needed.
        if args.embed_fcd and _sa is not None and _sa != _wa:
            print("[arms][WARN] the ANIL checkpoints were meta-trained WITHOUT --embed-fcd "
                  "and are being widened here with a zero block. Both arms adapt the same "
                  "object, but ANIL's meta-learned anchor never saw the FCD input. "
                  "Re-run run_lodo_anil with --embed-fcd for the intended comparison.")

    outdir = pathlib.Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    paths = {
        "l2sp": ensure_curve("l2sp", pathlib.Path(args.l2sp_ckpt_dir),
                             args.l2sp_prefix, outdir, tau, args),
        "anil": ensure_curve("anil", pathlib.Path(args.anil_ckpt_dir),
                             args.anil_prefix, outdir, tau, args),
    }
    if not paths["l2sp"]:
        raise SystemExit("the L2-SP curve is required")
    l2 = load_arm(paths["l2sp"], "l2sp")
    if not paths["anil"]:
        print("\n[warn] no ANIL curve — reporting the L2-SP arm alone.")
        an = None
    else:
        an = load_arm(paths["anil"], "anil")

    # RESTRICT BOTH ARMS TO THE DRIVERS THEY SHARE before averaging. Folds can go
    # missing independently (a meta-training run that failed, a checkpoint not yet
    # built), and an arm mean taken over a different driver set than the other's
    # is not comparable to it — driver difficulty spans ~0.47 to ~1.58, so one
    # extra easy driver moves an arm mean further than any plausible arm effect.
    # Printing them side by side without this invites exactly the wrong reading.
    common = sorted(set(l2["pid"]) & set(an["pid"])) if an is not None else sorted(set(l2["pid"]))
    dropped = (sorted((set(l2["pid"]) | set(an["pid"])) - set(common))
               if an is not None else [])
    if dropped:
        print(f"\n[warn] {len(dropped)} driver(s) present for only one arm and EXCLUDED "
              f"from every comparison below: {dropped}. Both arms are summarised over the "
              f"same {len(common)} drivers, or the means would not be comparable.")
    l2c = l2[l2["pid"].isin(common)]
    anc = an[an["pid"].isin(common)] if an is not None else None

    # Extra arms are loaded AFTER `common` is fixed by the learned arms, and are
    # restricted to it rather than intersected into it. A model-free reference
    # can cover drivers the learned arms lack (it needs no checkpoint), and
    # letting it widen or narrow the set the arms are compared on would change
    # the primary result as a side effect of adding a baseline.
    extras: Dict[str, pd.DataFrame] = {}
    for spec in args.extra_arms:
        path, _, name = spec.rpartition(":")
        if not path or not name:
            raise SystemExit(f"--extra-arm expects <csv>:<name>, got {spec!r}")
        p = pathlib.Path(path)
        if not p.exists():
            print(f"[warn] --extra-arm {name}: {p} not found, skipping")
            continue
        df = load_arm(p, name)
        miss = [c for c in ("pid", "k", "set_mae", "set_acc") if c not in df.columns]
        if miss:
            print(f"[warn] --extra-arm {name}: missing column(s) {miss}, skipping")
            continue
        if "variant" in df.columns and df["variant"].nunique() > 1:
            # One CSV, several baselines: split them so each is its own arm rather
            # than being averaged into a meaningless blend.
            for v, g in df.groupby("variant"):
                extras[f"{name}:{v}"] = g[g["pid"].isin(common)]
        else:
            extras[name] = df[df["pid"].isin(common)]
        absent = sorted(set(common) - set(df["pid"]))
        if absent:
            print(f"[warn] --extra-arm {name}: no rows for driver(s) {absent}")

    floor = l2c.groupby("pid")["base_set_mae"].first().mean()
    print(f"\n{'arm':>6} {'drivers':>8} {'mean set-MAE':>13} {'se':>7} {'vs floor':>9}"
          f"   (K <= {args.k_cap}, averaged within driver first)")
    for arm, df in ([("l2sp", l2c), ("anil", anc)] + sorted(extras.items())):
        if df is None or df.empty:
            continue
        pd_mean = df[df["k"] <= args.k_cap].groupby("pid")["set_mae"].mean()
        se = pd_mean.std(ddof=1) / np.sqrt(len(pd_mean)) if len(pd_mean) > 1 else float("nan")
        print(f"{arm:>6} {len(pd_mean):8d} {pd_mean.mean():13.3f} {se:7.3f} "
              f"{pd_mean.mean() - floor:+9.3f}")
    print(f"{'floor':>6} {'':>8} {floor:13.3f}    (unadapted population model, K=0)")
    if extras:
        print("  NOTE extra arms are references, not study arms. If one beats both "
              "learned arms, that is the result to report, not an anomaly to "
              "explain away.")
    l2, an = l2c, anc

    summary: Dict[str, object] = {"tau": tau, "k_cap": args.k_cap,
                                  "unadapted_floor_set_mae": float(floor)}

    for name, df in sorted(extras.items()):
        # Paired against L2-SP on the SAME drivers, same K cap, same statistic as
        # the arm-vs-arm comparison — so a baseline is held to the study's own
        # standard of evidence rather than a looser one.
        pr = paired_table(l2c, df, args.k_cap)
        if pr.empty:
            continue
        safe = name.replace(":", "_")
        pr.to_csv(outdir / f"paired_l2sp_vs_{safe}.csv", index=False)
        d = pr["mae_delta"]
        print(f"\n[extra] {name} vs l2sp (paired over {len(pr)} drivers, K <= "
              f"{args.k_cap}): mean set-MAE difference {d.mean():+.3f} "
              f"({'the baseline is BETTER' if d.mean() < 0 else 'l2sp is better'})")

    if an is not None:
        pair = paired_table(l2, an, args.k_cap)
        pair.to_csv(outdir / "paired_per_driver.csv", index=False)
        d = pair["mae_delta"].to_numpy()
        n = len(d)
        mean_d = float(d.mean())
        se_d = float(d.std(ddof=1) / np.sqrt(n)) if n > 1 else float("nan")
        try:
            from scipy.stats import wilcoxon
            stat, p = wilcoxon(pair["anil_mae"], pair["l2sp_mae"])
            p = float(p)
        except Exception:
            stat, p = float("nan"), float("nan")

        print(f"\n{'driver':>7} {'L2-SP':>8} {'ANIL':>8} {'delta':>8}  better")
        for _, r in pair.iterrows():
            print(f"{r['pid']:>7} {r['l2sp_mae']:8.3f} {r['anil_mae']:8.3f} "
                  f"{r['mae_delta']:+8.3f}  {'ANIL' if r['anil_better'] else 'L2-SP'}")
        print(f"\n[paired] mean delta (ANIL - L2-SP) = {mean_d:+.4f} "
              f"+/- {se_d:.4f} SE over {n} drivers; "
              f"ANIL better on {int(pair['anil_better'].sum())}/{n}")
        if p == p:
            print(f"[paired] Wilcoxon signed-rank p = {p:.4f}")
        # The signed-rank test cannot reach p < 0.05 below n = 6 REGARDLESS of the
        # data — with n pairs the smallest attainable two-sided p is 2/2^n. Saying
        # so beats letting a structurally impossible p be read as "no effect".
        if n < 6:
            print(f"[paired] NOTE n={n}: the smallest two-sided p the signed-rank test can "
                  f"produce here is {2 / (2 ** n):.3f}, so significance is unreachable no "
                  f"matter what the data show. Read the per-driver deltas, not the p-value.")
        # Direction is what matters, and it is easy to invert: set-MAE is a LOSS,
        # so a NEGATIVE delta means ANIL is better.
        verdict = ("ANIL better" if mean_d < 0 else "L2-SP better") if abs(mean_d) > se_d \
            else "indistinguishable (|delta| < 1 SE)"
        print(f"[paired] verdict: {verdict}")
        summary.update({"n_drivers": n, "mean_mae_delta_anil_minus_l2sp": mean_d,
                        "se": se_d, "wilcoxon_p": p,
                        "anil_better_on": int(pair["anil_better"].sum()),
                        "verdict": verdict})

        # A null is only interpretable if the ANIL arm was genuinely tuned, so the
        # stage-A caveats travel WITH the result rather than living in another file.
        sa = read_json(pathlib.Path(args.selected_anil)) or {}
        caveats = []
        if sa.get("outer_lr_on_grid_edge"):
            caveats.append("stage A's winning outer_lr sat on a GRID EDGE — the ANIL arm "
                           "may simply be under-tuned")
        if sa.get("mean_query_loss_drop", 1.0) <= 0:
            caveats.append("stage A's meta-training query loss did not drop — the "
                           "meta-objective was not being optimized")
        if sa.get("worst_inner_residual", 0.0) and sa["worst_inner_residual"] > 1e-3:
            caveats.append(f"stage A's worst iMAML inner residual was "
                           f"{sa['worst_inner_residual']:.1e} — the implicit meta-gradient "
                           f"was biased")
        if caveats and verdict != "ANIL better":
            print("\n[CAVEAT] do not report this as 'meta-learning does not help':")
            for c in caveats:
                print(f"  - {c}")
        summary["caveats"] = caveats

        fig, ax = plt.subplots(figsize=(7.6, 4.8))
        for arm, df, col in (("L2-SP", l2, "#4C72B0"), ("ANIL", an, "#DD8452")):
            c = df.groupby("k")["set_mae"].mean().sort_index()
            ax.plot(c.index, c.values, marker="o", ms=3, lw=1.8, color=col, label=arm)
        ax.axhline(floor, ls="--", c="0.35", lw=1.2, label="unadapted population (K=0)")
        ax.axvline(args.k_cap, ls=":", c="0.6", lw=1.0)
        ax.set_xlabel("personalization segments K  (20 s each)")
        ax.set_ylabel("set-MAE (lower is better)")
        ax.set_title(f"Personalization vs K, {n} drivers (LODO), tau={tau:g}")
        ax.grid(alpha=0.3); ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(outdir / "arm_comparison.png", dpi=150)
        print(f"[OK] plot  -> {outdir / 'arm_comparison.png'}")
        print(f"[OK] table -> {outdir / 'paired_per_driver.csv'}")

    (outdir / "arm_comparison.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[OK] -> {outdir / 'arm_comparison.json'}")


if __name__ == "__main__":
    main()
