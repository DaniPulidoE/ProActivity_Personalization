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
Wilcoxon p=0.73, "indistinguishable"), and note the sign counts: L2-SP is better
for 8 of 12 drivers at K=4 and 9 of 12 at K=8, while ANIL's nominally lower MEAN
is carried by two or three large wins. By the median driver L2-SP *is* the better
model, so "pick the best one" does not select ANIL either.

**The argument that decides it is definitional, not statistical.** The K=0
condition has to mean *no personalization*. The ANIL meta-init is a model whose
entire training objective was "be easy to personalize" -- it IS personalization
machinery, and labelling it the no-personalization control is a category error
regardless of its MAE. L2-SP's K=0 is ``pop_heldout_<pid>.pt``, a population
model and nothing else, which is exactly the claim the condition makes.

That also settles the mixed design, which is the one arrangement that is
outright broken: LODO at K=0 with ANIL at K>0 puts a different backbone under the
baseline than under the treatments, so part of the K effect is the backbone
changing. (Treat the size of that part as unknown rather than as the 0.130 floor
gap -- that gap is itself only 1.15 SE and favours L2-SP on 7 of 12 drivers.)

One measurement consequence, worth knowing before anyone argues for all-ANIL on
"use the best models" grounds: a better floor COMPRESSES the effect the study
exists to measure. All-ANIL runs 0.818 -> 0.686 -> 0.501 (K span -0.317);
all-L2-SP runs 0.948 -> 0.708 -> 0.514 (-0.434), and the K=0->K=4 step nearly
halves, -0.240 to -0.132. With 12 participants and a satisfaction DV that is
power there is none to spare.

WHERE "USE THE BEST MODELS" ACTUALLY BITES -- AND IT IS NOT THE ARM
==================================================================
Two things in this file already answer it, and both are worth defending out loud
rather than leaving implicit.

**Phone-call-only support is ~7x more label-efficient than the alternative.**
Scored on the identical phone-call tail, support drawn from ALL functions
(``phone_call_k_curve_all``) needs K~60 to reach what this function's own labels
reach at K=6:

    K            4       6       8      40      60
    function  0.708   0.467   0.514     --      --
    all       0.827   0.876   0.833   0.648   0.430

**And K=8 is already the plateau for this function**, so the supervisor's premise
-- more data in the real world -- does not imply a better condition-2 model here.
K=13 reaches 0.492 against K=8's 0.514: K=8 captures ~95% of the total achievable
gain, and the remaining 0.022 is noise at a 5-10 segment tail.

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

TAU=1.0, ADOPTED FROM THE CURVE -- NOT ``committed_tau.json``
=============================================================
``results/committed_tau.json`` commits tau=0.05 for both arms. **It was never
applied.** Confirmed by the operator on 2026-08-25 and corroborated on disk: the
phone-call curve records tau=1.0, ``selected_anil.json`` records tau=1.0, and the
only runs at 0.05 (``results/arm_comparison``, 2026-08-19) predate the ANIL sweep
and use a superseded ANIL with a different unadapted floor.

So this script adopts tau, steps, val_frac and embed_fcd from
``--match-curve``'s own JSON. The reason is not deference to history: the ladder
0.948 / 0.708 / 0.514 that justified K=4 and K=8 is a **tau=1.0 ladder**, and K
was read off it. Deploy at 0.05 and the served head corresponds to no measured
curve point -- ``--verify-against`` would be comparing two different estimators,
and the study's independent variable would have been chosen on a curve the
deployment does not sit on.

What that costs, stated rather than left to be discovered: **tau=0.05 is
measurably better at these K.** Mean set-MAE over K in 1..12 on
``results/l2sp_sweep_fcd_extend``, monotone in tau: 0.005 -> 1.3135,
0.05 -> 1.3390, 0.5 -> 1.3576, **1.0 -> 1.3653**. A weaker anchor wins at every
low K, which is the opposite of the usual intuition. Adopting it properly means
re-running the phone-call curve at 0.05 and re-reading K off the new curve --
and, for the ANIL arm, re-meta-training, since iMAML's inner loop is
tau-dependent. Until then, matching the curve is worth more than 0.026 of
set-MAE against a between-driver sd of 0.343.

**Do not re-select tau on the phone-call slice** in any case: 5-10 query segments
per driver is exactly the selection-validity problem ``phone_call_k_curve``
refuses to open.

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
    DEFAULT_TAU as DEFAULT_TAU_FALLBACK,
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

# --verify-against tolerances. TWO, because they answer different questions.
#
# When the adaptation settings were ADOPTED from the curve (the default), this
# script runs the identical estimator on the identical support, so the two
# numbers must agree to float noise. Anything above the exact bound is a real
# difference -- a different support set, a different split, a rebuilt LODO
# checkpoint -- and should stop the build.
#
# When tau or steps were overridden the estimators genuinely differ, and only the
# SUPPORT SET is still being checked. A wrong support moves per-driver set-MAE by
# ~0.1-0.5 on this slice, so the loose bound still catches it while tolerating a
# tau change.
VERIFY_ATOL_EXACT = 1e-4
VERIFY_ATOL_LOOSE = 0.05


def resolve_key(name: str) -> str:
    """Display name -> the canonical key every downstream comparison uses.

    ``load_driver_rows`` and the ``is_eval`` mask both test
    ``resolve_function_key(row.functionname) == want_key``, so ``want_key`` has to
    be the RESOLVED key ('startaphonecall'), never the display name. Passing
    'Respond to a phone call' straight through compares a key against a label,
    matches nothing, and reports "no rows" for every driver -- which is what it
    did, and which no amount of staring at the data explains.

    Idempotent, so either spelling works as input. ``phone_call_k_curve.json``
    records the resolved form, which is why this is also what goes in the
    provenance.
    """
    from ProVoice.fcd_config import resolve_function_key, UNKNOWN_FUNCTION_KEY
    key = resolve_function_key(name)
    if key == UNKNOWN_FUNCTION_KEY:
        raise SystemExit(
            f"--function {name!r} does not resolve to a known function; it "
            f"lands on {UNKNOWN_FUNCTION_KEY!r}, whose FCD vector is neutral "
            f"all-3s. Every segment would be scored with no task context. "
            f"Check the spelling against fcd_config.FUNCTIONS.")
    return key


def report_available_functions(src: pathlib.Path, pid: str, limit: int = 12) -> None:
    """Say WHICH functions this driver actually has, after an empty filter.

    A bare "no rows" sends you to look at the data file; the useful question is
    whether the filter or the data is wrong, and that is answerable in one pass.
    """
    from ProVoice.fcd_config import resolve_function_key
    from ProVoice.models.train_XLSTM import iter_jsonl
    import collections
    seen: collections.Counter = collections.Counter()
    rows = 0
    for r in iter_jsonl(src):
        if str(r.get("participantid", "")) != pid:
            continue
        rows += 1
        seen[resolve_function_key(str(r.get("functionname", "") or ""))] += 1
    if not rows:
        print(f"       ...and NO rows at all for participantid {pid!r}. Check "
              f"the id spelling in {src} (zero-padded '001', not '1'?).")
        return
    print(f"       {rows} row(s) for {pid!r}, resolving to: "
          + ", ".join(f"{k!r}x{v}" for k, v in seen.most_common(limit)))


def sha256_of(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def resolve_adapt_cfg(args) -> Dict:
    """The adaptation settings, taken FROM THE CURVE THAT CHOSE THE K VALUES.

    Not from ``committed_tau.json``. That file records a decision (tau=0.05) that
    was never applied: the LODO checkpoints, the ANIL meta-training and every
    published curve were all run at **tau=1.0**, confirmed by the operator on
    2026-08-25 and by ``phone_call_k_curve.json`` and ``selected_anil.json``
    agreeing on 1.0. Deploying at 0.05 would make the served head correspond to
    no measured curve point at all -- the ladder 0.948 / 0.708 / 0.514 that
    justified K=4 and K=8 is a tau=1.0 ladder, and K was read off it.

    So the curve's own JSON is the source of truth here, which also means this
    script cannot drift from it silently: change the curve, rerun this, and the
    settings follow. ``committed_tau.json`` is still read, but only to say out
    loud that the two disagree.

    Note what is being given up, so it can be stated rather than discovered:
    tau=0.05 is measurably BETTER at these K (pooled over K<=12, monotone in tau:
    0.05 -> 1.3390 against 1.0 -> 1.3653). Deploying the better tau would mean
    re-running the phone-call curve at 0.05 and re-reading K off it -- and, for
    the ANIL arm, re-meta-training, since its inner loop is tau-dependent. Until
    that happens, matching the curve is worth more than 0.026 of set-MAE.
    """
    cfg: Dict = {"lr": DEFAULT_ADAPT_LR}
    src = pathlib.Path(args.match_curve) if args.match_curve else None
    if src is not None and src.exists():
        j = json.loads(src.read_text(encoding="utf-8"))
        cfg["tau"] = float(j["tau"])
        cfg["steps"] = int(j["steps"])
        cfg["source"] = str(src)
        # val_frac and embed_fcd are inherited too, for the same reason: they
        # define the split and the head width the curve was measured on, and a
        # mismatch in either would silently score a different estimator.
        cfg["val_frac"] = float(j.get("val_frac", args.val_frac))
        cfg["embed_fcd"] = bool(j.get("embed_fcd", 1))
        cfg["curve_function"] = j.get("function")
    else:
        if src is not None:
            print(f"[WARN] {src} not found; falling back to head_adapt defaults.")
        cfg.update(tau=DEFAULT_TAU_FALLBACK, steps=DEFAULT_ADAPT_STEPS,
                   source="head_adapt defaults", val_frac=args.val_frac,
                   embed_fcd=True)

    for k, v in (("tau", args.tau), ("steps", args.steps), ("lr", args.lr)):
        if v is not None:
            cfg[k] = v
            cfg["source"] = "CLI override"

    committed = pathlib.Path(args.committed)
    if committed.exists():
        ct = float(json.loads(committed.read_text(encoding="utf-8")).get("tau", float("nan")))
        if ct == ct and abs(ct - cfg["tau"]) > 1e-12:
            print("[NOTE] %s commits tau=%g, but this build uses tau=%g -- the "
                  "value everything was actually RUN at. The committed file was "
                  "never applied; keeping the pipeline self-consistent beats "
                  "adopting it for the served checkpoints alone. Say which was "
                  "used in the write-up." % (committed, ct, cfg["tau"]))
    return cfg


def support_and_tail(df: pd.DataFrame, arch: Dict, model, want_key: str,
                     val_frac: float, device: str):
    # `want_key` is the CANONICAL key from resolve_function_key ('startaphonecall'),
    # never the display name -- see resolve_key().
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

    df = load_driver_rows(pathlib.Path(args.in_data), pid, args.function_key)
    if df.empty:
        print(f"[{pid}] no rows for {args.function!r} "
              f"(key {args.function_key!r}) -- skipped.")
        report_available_functions(pathlib.Path(args.in_data), pid)
        return []
    packed, why = support_and_tail(df, arch, model, args.function_key,
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
            # The canonical key, matching what phone_call_k_curve.json records,
            # plus the display name it came from.
            "function": args.function_key,
            "function_name": args.function,
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
    atol = VERIFY_ATOL_EXACT if args._matches_curve else VERIFY_ATOL_LOOSE
    if abs(d) <= atol:
        return "MATCH  d=%+.1e" % d
    hint = ("DIFFERENT SUPPORT?" if args._matches_curve
            else "expected under a tau/steps override -- check the SIZE")
    return "MISMATCH d=%+.4f vs curve %.4f -- %s" % (d, ref, hint)


def check_anil_tau(args, cfg: Dict) -> None:
    """Refuse to serve an ANIL init at a tau it was not meta-trained for.

    THIS IS NOT A STYLE CHECK. The ANIL arm is iMAML, and iMAML was chosen over
    path-differentiated ANIL for exactly one reason (CLAUDE.md): implicit
    differentiation needs only the inner problem's SOLUTION, so the meta-training
    inner loop can be the identical objective that gets deployed. That objective
    is soft-CORN plus ``lam * ||theta - theta_pop||^2`` with ``lam = tau / (2K)``
    -- it is tau-dependent. Meta-train at tau=1.0 and serve at tau=0.05 and the
    deployed problem is NOT the one that was differentiated through, which voids
    the single guarantee the variant was selected for. Nothing would raise; the
    checkpoint would simply be an init optimized for a different fixed point.

    ``results/anil_sweep*/selected_anil.json`` currently records tau=1.0 with the
    note "tau is FROZEN from the L2-SP sweep", but it was written 2026-08-21,
    three days AFTER results/committed_tau.json committed tau=0.05 for both arms
    and said sweep_anil_hparams must be pointed at that file. So the two records
    disagree, and an ANIL build inherits the disagreement silently. Resolve it by
    re-meta-training at the committed tau -- not by passing --tau here, which only
    moves the mismatch from meta-training to serving.
    """
    if args.arm != "anil":
        return                     # L2-SP has no meta-trained tau to match

    # ONE named file, not a glob over every sweep on disk. The draft runs used
    # different taus (anil_sweep_draft is at 0.05, the finals at 1.0), so a scan
    # can only ever contradict itself -- and which sweep produced the checkpoints
    # in --ckpt-dir is something the operator knows and this script does not.
    p = pathlib.Path(args.anil_selected) if args.anil_selected else None
    if p is None:
        cands = [q for q in pathlib.Path("results").glob("anil_sweep*/selected_anil.json")
                 if "draft" not in q.parent.name]
        p = max(cands, key=lambda q: q.stat().st_mtime) if cands else None
    if p is None or not p.exists():
        print("[WARN] --arm anil but no selected_anil.json was found, so the "
              "meta-training tau cannot be checked against tau=%g. Verify by "
              "hand before serving -- see check_anil_tau." % cfg["tau"])
        return

    j = json.loads(p.read_text(encoding="utf-8"))
    t = float(j.get("tau", float("nan")))
    if t == t and abs(t - cfg["tau"]) > 1e-12:
        raise SystemExit(
            "REFUSING to build ANIL checkpoints at tau=%g: %s records the "
            "meta-training inner-loop tau as %g.\n\n"
            "iMAML's inner objective depends on tau (lam = tau/2K), so an init "
            "meta-trained at %g is optimized for a fixed point that tau=%g does "
            "not have. Serving it here would void the one property iMAML was "
            "chosen for over path-differentiated ANIL.\n\n"
            "Options, in order of preference:\n"
            "  1. Re-meta-train ANIL with inner tau=%g (the committed value).\n"
            "  2. Pass --tau %g to match the meta-init, and state in the "
            "write-up that the ANIL arm is served at a tau the L2-SP sweep did "
            "not select.\n"
            "  3. Point --anil-selected at the sweep that actually produced "
            "%s.\n"
            % (cfg["tau"], p, t, t, cfg["tau"], cfg["tau"], t, args.ckpt_dir))
    print("[ok] ANIL meta-training tau (%s) matches the adaptation tau: %g."
          % (p, cfg["tau"]))


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
    ap.add_argument("--match-curve", dest="match_curve",
                    default="results/phone_call_k_curve/phone_call_k_curve.json",
                    help="The K curve this study reads its K values off. Its "
                         "tau, steps, val_frac and embed_fcd are ADOPTED, so the "
                         "served head is the same estimator as the curve point "
                         "that justified deploying it (default: %(default)s).")
    ap.add_argument("--committed", default="results/committed_tau.json",
                    help="Cross-check only. Records a tau that was never "
                         "applied; a disagreement with --match-curve is printed, "
                         "not acted on.")
    ap.add_argument("--anil-selected", dest="anil_selected", default=None,
                    help="selected_anil.json for the sweep that produced the "
                         "--ckpt-dir checkpoints. Used ONLY with --arm anil, to "
                         "refuse a build whose adaptation tau differs from the "
                         "meta-training inner-loop tau (see check_anil_tau -- "
                         "the mismatch voids iMAML's train/serve identity). "
                         "Defaults to the newest non-draft results/anil_sweep*/.")
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

    # Resolve ONCE, here. Every downstream comparison is key-vs-key.
    args.function_key = resolve_key(args.function)
    cfg = resolve_adapt_cfg(args)
    check_anil_tau(args, cfg)
    # Inherited from the curve so the estimator cannot drift; an explicit CLI
    # value still wins, since --val-frac has a real default.
    if args.val_frac == 0.3:
        args.val_frac = cfg.get("val_frac", args.val_frac)
    if args.embed_fcd and not cfg.get("embed_fcd", True):
        args.embed_fcd = False
    # Did we actually run the curve's estimator? Decides which tolerance
    # --verify-against applies, so it is settled once rather than re-derived.
    args._matches_curve = str(cfg.get("source", "")).endswith(".json")
    verify = pd.read_csv(args.verify_against) if args.verify_against else None

    print("arm=%s  function=%r (key %r)  tau=%g  steps=%d  lr=%g  embed_fcd=%d  "
          "device=%s" % (args.arm, args.function, args.function_key, cfg["tau"],
                         cfg["steps"], cfg["lr"], int(args.embed_fcd), args.device))
    # The curve records the key it was computed on; if this build is filtering to
    # a different function, every --verify-against cell would silently miss.
    curve_fn = cfg.get("curve_function")
    if curve_fn and curve_fn != args.function_key:
        print("[WARN] --match-curve was computed on function %r but this build "
              "filters to %r. The K values were read off a curve for a "
              "DIFFERENT function." % (curve_fn, args.function_key))
    print("conditions: " + "  ".join("%d->K=%d" % (c, k_map[c]) for c in sorted(k_map)))
    if verify is not None:
        print("verify tolerance: %s (%.0e) -- %s"
              % (("EXACT", VERIFY_ATOL_EXACT,
                  "settings adopted from the curve, so the estimators are "
                  "identical and only the support set can differ")
                 if args._matches_curve else
                 ("LOOSE", VERIFY_ATOL_LOOSE,
                  "tau/steps overridden, so only a wrong SUPPORT SET is being "
                  "checked, not the estimator")))
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
