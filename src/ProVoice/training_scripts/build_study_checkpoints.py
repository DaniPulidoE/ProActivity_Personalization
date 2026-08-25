r"""Mint the live study's served checkpoints: one per (participant, K condition).

    python -m ProVoice.training_scripts.build_study_checkpoints \
        --in-data data/labeled_data.jsonl \
        --ckpt-dir trained_models/lodo \
        --outdir trained_models/user_study \
        --verify-against results/phone_call_k_curve/phone_call_k_curve.csv

Writes ``trained_models/user_study/xlstm_p<pid>_k<condition>.pt`` — the exact
path ``start_experiment.study_model_path`` will look for — for every participant
and every condition:

    condition 0  ->  K=0   the driver's own LODO population model, unadapted
    condition 1  ->  K=4   adapted on their first 4 phone-call labels
    condition 2  ->  K=8   adapted on their first 8

WHY L2-SP AND NOT ANIL
======================
Decided on ``results/phone_call_k_curve`` — the only offline curve that measures
the deployment condition (personalize on ``Respond to a phone call`` labels only,
score on that function's temporal tail only). Paired over 12 drivers:

    K   L2-SP   ANIL    anil-l2sp        ANIL better on
    0   0.948   0.818   -0.130 (1.15 SE)      5/12
    4   0.708   0.686   -0.022 (0.49 SE)      4/12
    8   0.514   0.501   -0.013 (0.19 SE)      3/12

The arms are indistinguishable at both study K values (the run's own headline:
Wilcoxon p=0.73, "indistinguishable"), so accuracy does not choose. What chooses
is the K=0 row.

**The K=0 confound is the whole argument.** The study serves
``pop_heldout_<pid>.pt`` as its unpersonalized condition, and the design requires
all three conditions to share one backbone so that no part of the K effect can be
"the backbone had seen your data". ANIL's floor is 0.130 better than L2-SP's
because its backbone is meta-trained. Serve ANIL at K>0 with the LODO model at
K=0 and **0.130 of the 0.240 K=0->K=4 movement is the backbone changing, not the
labels** — more than half the effect, confounded. Serve the ANIL meta-init at
K=0 instead and the condition stops being "the population model": a meta-init is
trained specifically to be adapted, so labelling it "no personalization" in the
write-up would be false.

Two lesser reasons point the same way. L2-SP moves FURTHER from its own floor,
which is what the live study is trying to detect (K=4: -0.240 vs -0.133; K=8:
-0.434 vs -0.318, improving 9/12 drivers vs 8/12) — a bigger, more consistent
per-label movement is a better chance of a detectable satisfaction difference.
And it is the simpler artifact to defend.

NOT a reason: "LODO wins at few labels and ANIL at many". The measured pattern is
the reverse at the low end (ANIL's mean is nominally better at K=1-4 and L2-SP's
at K=5-6), and nothing anywhere on the curve clears 1 SE except K=6. At K=4 the
mean favours ANIL while only 4 of 12 drivers do — the mean is carried by a couple
of large wins, so even the sign is unreliable. Do not report a K-dependent
crossover; there is no evidence for one.

THE SUPPORT SET IS THE DRIVER'S FIRST K PHONE-CALL LABELS
=========================================================
Chronological, from their population session, with no gap and no selection:
``pool_idx[:K]`` where the pool is every phone-call segment recorded before the
evaluation tail. That is bit-identical to the support ``phone_call_k_curve`` used
at the same K, which is what makes the deployed checkpoint the same estimator as
the curve the K values were read off — the property CLAUDE.md requires and the
one ``--verify-against`` actually checks, cell by cell, against that run's CSV.

Everything downstream of ``embed_segments`` is a ~260-parameter convex fit on a
(K x 76) tensor, so the whole cohort costs one backbone pass per driver.

TAU COMES FROM ``results/committed_tau.json``, NOT FROM A FLAG
=============================================================
tau=0.05, 2000 steps, lr=5e-3 — the values committed on 2026-08-18 for BOTH
arms. Note the K curve above was produced at tau=1.0 / 6000 steps, so the
checkpoints this script writes are NOT at the curve's tau by default. That is
deliberate: the committed file is the decision, and its own analysis shows tau is
a tenth-order effect (the whole grid below tau=1 spans 0.028 against a
between-driver sd of 0.343), so it cannot move the arm choice. It does move the
absolute numbers slightly, which is why ``--verify-against`` tolerances are on by
default rather than exact — pass ``--tau 1.0 --steps 6000`` to reproduce the
curve exactly and expect agreement to ~1e-6.

WHAT PROVENANCE GETS WRITTEN, AND WHY IN ``arch``
=================================================
``arch['study']`` carries participant_id, held_out_pid, arm, condition, K, tau,
the support segment ids, and the base checkpoint's sha256. It rides in ``arch``
because that is already the checkpoint's data contract, it survives a rename, and
the serving loader needs it to assert the held-out pid equals ``--participantid``
before a session starts. **That assert does not exist yet** — this script only
makes it possible. Serving 004's head to 003 is silent and unrecoverable after
the fact, so wire it before the first participant.
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import pathlib
import sys
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
import torch

from ProVoice.models.xlstm_model import load_checkpoint, save_checkpoint
from ProVoice.models.head_adapt import (
    adapt_head, install_fcd_head, DEFAULT_ADAPT_LR, DEFAULT_ADAPT_STEPS,
)
from ProVoice.training_scripts.folds import ALL_PIDS
# The function filter is IMPORTED, not re-implemented: it resolves through
# fcd_config.resolve_function_key, so "the segments this keeps" and "the segments
# carrying this function's FCD vector" cannot come apart, and the legacy spelling
# ('Start a phone call') is picked up rather than silently dropped.
from ProVoice.training_scripts.phone_call_k_curve import (
    load_driver_rows, DEFAULT_FUNCTION,
)

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3] / "scripts"))
from sweep_train_frac import build_segments, embed_segments, evaluate  # noqa: E402


# condition -> K. The study's independent variable arrives as a FILENAME, so this
# mapping is the only place the two vocabularies meet. Condition 0 is K=0 by
# definition, not by choice.
DEFAULT_K_MAP = {0: 0, 1: 4, 2: 8}

# Above this the adapted head is not at the stationary point the Laplace layer
# expands about, and the value is a symptom of steps/lr being wrong rather than
# of the driver being unusual.
GRAD_NORM_WARN = 1e-3

# Tolerance for --verify-against when tau/steps differ from the curve's. Loose
# enough to pass a tau change (a tenth-order effect), tight enough that a
# different SUPPORT SET -- the failure this check exists to catch -- cannot slip
# through: per-driver set-MAE differences from a wrong support are ~0.1-0.5.
VERIFY_ATOL = 0.05


def sha256_of(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_committed(path: pathlib.Path) -> Dict:
    """The committed tau and adaptation budget, or a loud failure.

    Read rather than defaulted: ``committed_tau.json`` states that it applies to
    both arms and that the per-sweep ``selected_tau.json`` files are unusable
    (their 1-SE tie-break returns the grid maximum, an artifact of where each
    grid happened to end). Hardcoding a number here would be a fourth opinion.
    """
    if not path.exists():
        raise SystemExit(
            f"{path} not found. It is the record of which tau the study "
            f"deploys; pass --tau/--steps/--lr explicitly to override, or point "
            f"--committed at the right file.")
    return json.loads(path.read_text(encoding="utf-8"))


def support_and_tail(df: pd.DataFrame, arch: Dict, model, want_key: str,
                     val_frac: float, device: str):
    """Embed one driver's phone-call segments and split support / evaluation tail.

    Identical construction to ``phone_call_k_curve.curve_for_arm`` under
    ``--support-scope function``: the tail is the chronologically-last
    ``val_frac`` of the driver's target-function segments, the pool is every
    target-function segment recorded BEFORE the first tail segment, and both are
    positions in the driver's full chronological ordering.

    The tail is not used to fit anything. It exists so this script can report the
    number the curve reports and let --verify-against compare them.
    """
    gids, Xs, vs = build_segments(df, window_seconds=arch.get("window_seconds"),
                                  resample_hz=arch.get("resample_hz"))
    if len(gids) < 4:
        return None, f"only {len(gids)} segment(s) after build_segments"

    # By segment_id, never by position: build_segments SKIPS segments whose
    # Level_* labels are missing or all-zero, so its output is not index-aligned
    # with a groupby of the input and a positional mask would silently shift.
    from ProVoice.fcd_config import resolve_function_key
    fn_by_gid = df.groupby("segment_id", sort=False)["functionname"].first()
    is_eval = np.array([resolve_function_key(str(fn_by_gid.get(g, "") or "")) == want_key
                        for g in gids])
    eval_idx = np.flatnonzero(is_eval)
    if len(eval_idx) < 3:
        return None, f"only {len(eval_idx)} segment(s) of {want_key!r}"

    Z = embed_segments(model, Xs, vs, arch["context_length"], device)
    V = torch.from_numpy(np.stack(vs, axis=0))

    n_val = max(1, round(val_frac * len(eval_idx)))
    val_idx = eval_idx[len(eval_idx) - n_val:]
    cut = int(val_idx[0])
    pool_idx = np.flatnonzero((np.arange(len(gids)) < cut) & is_eval)
    if len(pool_idx) < 1:
        return None, "empty support pool"
    return {
        "Z": Z, "V": V, "gids": gids,
        "pool_idx": pool_idx, "val_idx": val_idx,
        "n_seg": len(gids), "n_eval_seg": int(len(eval_idx)),
    }, None


def build_for_driver(pid: str, args, cfg: Dict, k_map: Dict[int, int],
                     device: str, verify: Optional[pd.DataFrame]) -> List[dict]:
    ckpt = pathlib.Path(args.ckpt_dir) / f"{args.prefix}{pid}.pt"
    if not ckpt.exists():
        print(f"[{pid}] MISSING {ckpt} -- skipped.")
        return []

    model, arch = load_checkpoint(str(ckpt))
    # ORDER MATTERS. install_fcd_head builds a new Linear on the head's CURRENT
    # device, so it must run before .to(device). And .eval() is load-bearing: the
    # population config carries dropout, which is ACTIVE in a freshly-loaded
    # module -- omitting it raises nothing and silently randomizes every
    # embedding.
    install_fcd_head(model, args.embed_fcd)
    model.to(device).eval()
    head_type = arch.get("head_type", "softmax")
    pop_head = model.head

    df = load_driver_rows(pathlib.Path(args.in_data), pid, args.function)
    if df.empty:
        print(f"[{pid}] no rows for {args.function!r} -- skipped.")
        return []
    packed, why = support_and_tail(df, arch, model, args.function,
                                   args.val_frac, device)
    if packed is None:
        print(f"[{pid}] {why} -- skipped.")
        return []

    Z, V = packed["Z"], packed["V"]
    pool_idx, val_idx = packed["pool_idx"], packed["val_idx"]
    Zpool, Vpool = Z[pool_idx], V[pool_idx]
    Zval, Vval = Z[val_idx], V[val_idx]
    base_sha = sha256_of(ckpt)
    outdir = pathlib.Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    out = []
    for cond in sorted(k_map):
        k = k_map[cond]
        if k > len(pool_idx):
            # Not silently clamped to the pool size: a checkpoint labelled k8 that
            # actually saw 6 labels would make the study's independent variable a
            # lie, and nothing downstream could detect it.
            print(f"[{pid}] condition {cond} wants K={k} but the support pool "
                  f"holds only {len(pool_idx)} -- SKIPPED, no file written.")
            continue

        if k == 0:
            # The population head, untouched. Re-saved rather than copied so the
            # provenance block is present on all three conditions and the serving
            # loader's held-out assert can be uniform. Widening to the FCD head is
            # identity-preserving (expand_head_for_fcd appends ZEROS and anchors
            # them there), so this is numerically the checkpoint that stage 2
            # wrote -- it is not "personalized with K=0", it IS the LODO model.
            head, info = pop_head, {"grad_norm": 0.0, "l2sp": 0.0, "steps": 0}
            support = []
        else:
            head, info = adapt_head(pop_head, Zpool[:k], Vpool[:k], tau=cfg["tau"],
                                    head_type=head_type, steps=cfg["steps"],
                                    lr=cfg["lr"])
            support = [str(packed["gids"][i]) for i in pool_idx[:k]]
            if info["grad_norm"] > GRAD_NORM_WARN:
                print(f"[{pid}] condition {cond} K={k}: |grad| = "
                      f"{info['grad_norm']:.2e} > {GRAD_NORM_WARN:g}. The head is "
                      f"NOT at the stationary point; steps/lr are wrong.")

        model.head = head
        m = evaluate(head, Zval, Vval, head_type)

        arch_out = dict(arch)
        arch_out["study"] = {
            "participant_id": pid,
            # The LODO fold this backbone excluded. The serving loader must
            # assert this equals --participantid: serving 004's head to 003 is
            # silent and unrecoverable once the session has run.
            "held_out_pid": pid,
            "arm": args.arm,
            "condition": int(cond),
            "k": int(k),
            "tau": float(cfg["tau"]),
            "adapt_steps": int(info["steps"]),
            "adapt_lr": float(cfg["lr"]),
            "l2sp": float(info["l2sp"]),
            "grad_norm": float(info["grad_norm"]),
            "function": args.function,
            "embed_fcd": int(args.embed_fcd),
            "head_in": int(head.in_features),
            "base_checkpoint": ckpt.name,
            "base_sha256": base_sha,
            "support_segment_ids": support,
            "n_pool": int(len(pool_idx)),
            "n_tail": int(len(val_idx)),
            # Reported, NOT selected on. The tail never touched the fit; it is
            # here so a checkpoint can be traced to the curve point that justified
            # deploying it.
            "tail_set_mae": float(m["mae"]),
            "tail_set_qwk": float(m["qwk"]),
            "tail_set_acc": float(m["acc"]),
            "built_at": datetime.datetime.now().isoformat(timespec="seconds"),
            "built_by": "ProVoice.training_scripts.build_study_checkpoints",
        }
        path = outdir / f"xlstm_p{pid}_k{cond}.pt"
        if path.exists() and not args.overwrite:
            print(f"[{pid}] {path.name} exists -- pass --overwrite to replace.")
        elif not args.dry_run:
            save_checkpoint(model, str(path), arch_out)

        row = {"pid": pid, "condition": cond, "k": k, "file": path.name,
               "set_mae": m["mae"], "set_qwk": m["qwk"], "set_acc": m["acc"],
               "grad_norm": info["grad_norm"], "n_pool": len(pool_idx),
               "n_tail": len(val_idx), "verify": ""}
        if verify is not None:
            row["verify"] = check_against_curve(verify, pid, k, m["mae"], args)
        out.append(row)
        print("  [%s] cond %d  K=%-3d set-MAE %.4f  QWK %+.3f  |grad| %.1e  %s  %s"
              % (pid, cond, k, m["mae"], m["qwk"], info["grad_norm"],
                 path.name, row["verify"]))

    model.head = pop_head          # leave the loaded model as we found it
    return out


def check_against_curve(curve: pd.DataFrame, pid: str, k: int, mae: float,
                        args) -> str:
    """Compare this cell to ``phone_call_k_curve``'s, and say so out loud.

    THE POINT IS THE SUPPORT SET, not the metric. If the deployed checkpoint were
    fitted on a different set of labels than the curve point the K value was
    chosen from, nothing else in the pipeline would notice -- the file would load,
    serve, and be wrong. A per-driver set-MAE that reproduces the curve to within
    a tau change is strong evidence the two saw the same K labels; a mismatch of
    0.1+ means they did not.
    """
    sel = curve[(curve["pid"].astype(str).str.zfill(3) == pid)
                & (curve["arm"] == args.arm) & (curve["k"] == k)]
    if sel.empty:
        return "(no curve cell)"
    ref = float(sel.iloc[0]["set_mae"])
    d = mae - ref
    return ("MATCH  d=%+.4f" % d) if abs(d) <= VERIFY_ATOL else \
        ("MISMATCH d=%+.4f vs curve %.4f -- DIFFERENT SUPPORT?" % (d, ref))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in-data", dest="in_data", default="data/labeled_data.jsonl")
    ap.add_argument("--ckpt-dir", dest="ckpt_dir", default="trained_models/lodo",
                    help="Directory of LODO population checkpoints "
                         "(default: %(default)s).")
    ap.add_argument("--prefix", default="pop_heldout_",
                    help="Filename prefix inside --ckpt-dir (default: %(default)s).")
    ap.add_argument("--outdir", default="trained_models/user_study",
                    help="Where start_experiment.py looks (default: %(default)s).")
    ap.add_argument("--arm", default="l2sp", choices=("l2sp", "anil"),
                    help="Recorded in the provenance and used by "
                         "--verify-against. l2sp is the deployed arm -- see the "
                         "module docstring for why. Point --ckpt-dir at the ANIL "
                         "checkpoints if you change this.")
    ap.add_argument("--function", default=DEFAULT_FUNCTION,
                    help="The one function the study stages (default: %(default)s). "
                         "Matched through fcd_config.resolve_function_key.")
    ap.add_argument("--k-map", dest="k_map", default="0:0,1:4,2:8",
                    help="condition:K pairs (default: %(default)s). Condition 0 "
                         "must stay K=0 -- it is the unpersonalized reference.")
    ap.add_argument("--pids", default=",".join(ALL_PIDS))
    ap.add_argument("--val-frac", dest="val_frac", type=float, default=0.3,
                    help="Evaluation tail, for REPORTING only -- nothing is "
                         "fitted or selected on it (default: %(default)s). Must "
                         "match the curve's to compare against it.")
    ap.add_argument("--committed", default="results/committed_tau.json")
    ap.add_argument("--tau", type=float, default=None,
                    help="Override the committed tau. Only for reproducing a "
                         "specific curve run.")
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--no-embed-fcd", dest="embed_fcd", action="store_false",
                    help="Do NOT append the 12 FCD dims to the head input. The "
                         "committed configuration uses them; this exists for an "
                         "ablation, not for a study build.")
    ap.add_argument("--verify-against", dest="verify_against", default=None,
                    help="phone_call_k_curve.csv to check each cell against. Do "
                         "this on the first build: it is the only check that the "
                         "deployed head saw the same K labels as the curve the K "
                         "values were read off.")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--dry-run", dest="dry_run", action="store_true",
                    help="Adapt and report, write nothing.")
    ap.add_argument("--summary-csv", dest="summary_csv",
                    default="results/user_study_checkpoints.csv")
    args = ap.parse_args()

    try:
        k_map = {int(a): int(b) for a, b in
                 (p.split(":") for p in args.k_map.split(","))}
    except ValueError:
        raise SystemExit(f"--k-map must be 'cond:K,cond:K,...'; got {args.k_map!r}")
    if k_map.get(0, 0) != 0:
        raise SystemExit(
            "--k-map assigns K=%d to condition 0. Condition 0 is the "
            "UNPERSONALIZED reference and the whole K contrast is read against "
            "it; a non-zero K there would make the study's baseline a "
            "personalized model." % k_map[0])

    committed = load_committed(pathlib.Path(args.committed))
    cfg = {
        "tau": args.tau if args.tau is not None else float(committed["tau"]),
        "steps": args.steps if args.steps is not None
                 else int(committed.get("adapt_steps", DEFAULT_ADAPT_STEPS)),
        "lr": args.lr if args.lr is not None
              else float(committed.get("adapt_lr", DEFAULT_ADAPT_LR)),
    }
    verify = pd.read_csv(args.verify_against) if args.verify_against else None

    print("arm=%s  function=%r  tau=%g  steps=%d  lr=%g  embed_fcd=%d  device=%s"
          % (args.arm, args.function, cfg["tau"], cfg["steps"], cfg["lr"],
             int(args.embed_fcd), args.device))
    print("conditions: " + "  ".join("%d->K=%d" % (c, k_map[c]) for c in sorted(k_map)))
    if verify is not None and (cfg["tau"] != 1.0 or cfg["steps"] != 6000):
        print("NOTE verifying against a curve produced at a different tau/steps; "
              "cells are compared to +-%.2f set-MAE, which catches a wrong "
              "SUPPORT SET but not a small tau shift." % VERIFY_ATOL)
    if args.dry_run:
        print("DRY RUN -- no files will be written.")
    print()

    rows: List[dict] = []
    for pid in [p.strip() for p in args.pids.split(",") if p.strip()]:
        rows.extend(build_for_driver(pid, args, cfg, k_map, args.device, verify))

    if not rows:
        raise SystemExit("Nothing was built.")
    df = pd.DataFrame(rows)
    if args.summary_csv and not args.dry_run:
        pathlib.Path(args.summary_csv).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(args.summary_csv, index=False)

    print("\n%d checkpoint(s) over %d driver(s)." % (len(df), df["pid"].nunique()))
    print("\nmean set-MAE on the held-out tail, by condition:")
    for cond, g in df.groupby("condition"):
        print("  cond %d (K=%-3d) %.4f   QWK %+.3f   n=%d"
              % (cond, g["k"].iloc[0], g["set_mae"].mean(), g["set_qwk"].mean(), len(g)))

    bad = df[df["grad_norm"] > GRAD_NORM_WARN]
    if len(bad):
        print("\n%d cell(s) above the |grad| threshold -- those heads are not at "
              "the MAP and their Laplace posteriors would be invalid:" % len(bad))
        for _, r in bad.iterrows():
            print("  %s cond %d  |grad| %.2e" % (r["pid"], r["condition"], r["grad_norm"]))

    miss = df[df["verify"].astype(str).str.startswith("MISMATCH")]
    if len(miss):
        print("\n%d cell(s) DISAGREE with the curve -- do not run the study on "
              "these until it is understood:" % len(miss))
        for _, r in miss.iterrows():
            print("  %s cond %d  %s" % (r["pid"], r["condition"], r["verify"]))

    expected = len(args.pids.split(",")) * len(k_map)
    if len(df) != expected:
        print("\n%d of %d expected files were produced. A participant missing a "
              "condition CANNOT be run: the counterbalancing needs all three."
              % (len(df), expected))
    if args.summary_csv and not args.dry_run:
        print("\nSummary -> %s" % args.summary_csv)


if __name__ == "__main__":
    main()
