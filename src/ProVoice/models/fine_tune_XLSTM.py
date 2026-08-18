import argparse, pathlib

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from ProVoice.fcd_config import FCD_NAMES, get_fcd_for_function
from ProVoice.models.xlstm_model import (
    save_checkpoint,
    load_checkpoint,
    logits_to_label,
)
from ProVoice.models.xlstm_model import _as01
from ProVoice.models.head_adapt import (
    adapt_head,
    DEFAULT_ADAPT_LR,
    DEFAULT_ADAPT_STEPS,
    DEFAULT_TAU,
)
from ProVoice.models.laplace_head import LaplacePosterior, attach_laplace_to_checkpoint
from ProVoice.models.head_adapt import (
    assert_zero_block_identity, augment_z, install_fcd_head,
)

from ProVoice.models.train_XLSTM import (
    set_seed,
    read_jsonl,
    iter_jsonl,
    normalize_row,
    SeqDataset,
    make_collate,
    set_accuracy,
    set_macro_f1,
    set_mae,
    set_qwk,
)

LEVELS = [f"Level_{i}" for i in range(1, 6)]

@torch.no_grad()
def embed_all(model, dl, device, embed_fcd: bool = False):
    # function that precomputes the embeddings for all sequences in a dataloader, using the frozen in_proj and backbone of the model. Returns pooled embeddings and labels.
    """Run the frozen in_proj+backbone once; return pooled embeddings and labels."""
    zs, vs = [], []
    for xb, lb, vb in dl:
        h = model.backbone(model.in_proj(xb.to(device).to(torch.float32)))
        # Batches are RIGHT-padded: read the hidden state at the last REAL
        # frame (index length-1), same readout as model.forward.
        idx = (lb.to(h.device).long() - 1).clamp(min=0)
        z = h[torch.arange(h.size(0), device=h.device), idx]
        # Augmented from the SAME batch the embedding came from, so a segment's
        # FCD can never be paired with another segment's z.
        zs.append(augment_z(z, xb.to(device).to(torch.float32), embed_fcd).cpu())
        vs.append(vb)   # multi-hot Level_* — the training target AND the metric target
    return torch.cat(zs), torch.cat(vs)


def main():
    ap = argparse.ArgumentParser(description="Fine-tune official xLSTM (single-label 5-class).")
    ap.add_argument("--in-data",        dest="in_jsonl", required=True)
    ap.add_argument("--in-model",        dest="in_model", required=True)
    ap.add_argument("--out",       dest="out_pt",   default="trained_models/state_xlstm_finetune.pt")
    ap.add_argument("--log",       dest="log_path", default="",
                    help="Optional path for a JSONL log of the exact features fed to the "
                         "xLSTM (one line per frame). Off by default — pass a path to enable "
                         "for debugging.")
    #ap.add_argument("--label-map", dest="label_map", default=None, help="CSV with columns: segment_id, Level_1..Level_5")
    # Adaptation is full-batch with a FIXED, K-independent step budget and no
    # best-epoch selection — see ProVoice.models.head_adapt for why all three
    # matter. --epochs/--batch are gone with the mini-batch loop they described.
    ap.add_argument("--steps", type=int, default=DEFAULT_ADAPT_STEPS,
                    help="Full-batch gradient steps. Fixed a priori (tune it once on the "
                         "development drivers, not per run): selecting the best epoch on "
                         "the validation tail would select and report on the same "
                         "segments, spend part of the driver's labels on selection, and "
                         "bias the low-K end of the learning curve more than the high-K "
                         "end.")
    ap.add_argument("--embed-batch", dest="embed_batch", type=int, default=32,
                    help="How many segments go through the frozen backbone at once. "
                         "Throughput/VRAM only — it does not affect the adapted head.")
    ap.add_argument("--seed",   type=int, default=42)
    ap.add_argument("--lr",     type=float, default=DEFAULT_ADAPT_LR)
    ap.add_argument("--tau",    type=float, default=DEFAULT_TAU,
                    help="PRIOR PRECISION of the L2-SP anchor N(θ_pop, 1/τ). The penalty "
                         "strength λ is derived as τ/(2K) for the K support segments, "
                         "because the objective is a batch MEAN and so a fixed λ realises "
                         "τ = 2Kλ — an anchor that strengthens as data accumulates and "
                         "vanishes exactly where the design wants graceful degradation "
                         "(K→0 ⇒ τ→0). Larger τ keeps the personalized model closer to the "
                         "population model; τ→∞ recovers it exactly. Must match the τ "
                         "scripts/sweep_train_frac.py drew its learning curve with.")
    ap.add_argument("--val-frac", dest="val_frac", type=float, default=0.3,
                    help="Fraction of segments reserved as the chronologically-LAST validation "
                         "tail. Keep this fixed across runs so learning curves share one "
                         "measuring stick.")
    ap.add_argument("--train-frac", dest="train_frac", type=float, default=1.0,
                    help="Fraction of the non-validation segments (earliest first) used for "
                         "training. Sweep this (e.g. 0.2, 0.5, 1.0) to measure personalization "
                         "vs. data collection time against the fixed validation tail.")
    ap.add_argument("--embed-fcd", dest="embed_fcd", action="store_true",
                    help="Give the adapted head direct access to the task: it sees "
                         "[z_64 | FCD_12] (308 parameters instead of 260). The backbone is "
                         "untouched, so no retraining is implied; the appended block is "
                         "initialized AND L2-SP-anchored at zero, so K=0 reproduces the "
                         "population head exactly. The saved checkpoint carries the wider "
                         "head, and both forward() and the decision engine detect it from "
                         "its shape -- there is no serving flag to forget.")
    ap.add_argument("--laplace", action="store_true",
                    help="After training, fit a Laplace posterior over the adapted soft-CORN "
                         "head (exact per-unit Hessian on the training embeddings, prior "
                         "precision from --l2sp) and store it inside the output checkpoint. "
                         "Requires a CORN population checkpoint and --l2sp > 0.")
    args = ap.parse_args()
    if not (0.0 < args.val_frac < 1.0):
        raise ValueError(f"--val-frac must be in (0, 1), got {args.val_frac}")
    if not (0.0 < args.train_frac <= 1.0):
        raise ValueError(f"--train-frac must be in (0, 1], got {args.train_frac}")
    if args.laplace and args.tau <= 0.0:
        raise ValueError("--laplace requires --tau > 0 (the L2-SP anchor defines the prior).")
    
    # seed and cuda
    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # build df
    rows = [normalize_row(r) for r in iter_jsonl(pathlib.Path(args.in_jsonl))]
    if not rows:
        raise ValueError("JSONL is empty or contains no valid rows.")
    df = pd.DataFrame(rows)
    # check for errors
    lv = df[LEVELS].astype(float)
    bad = df.loc[lv.isna().any(axis=1) | (lv.sum(axis=1) <= 0), 'segment_id'].unique()
    if len(bad): 
        raise ValueError(f"Segments with missing/empty Level_* labels: {list(bad)[:5]} ...")

    
    # copy and paste label mapping logic if needed (probably won't be needed, given how the experiment is set up)
    
    # fill void points in dataset - technically already done in normalize_row and encode_frame
    if 'segment_id' not in df.columns or df['segment_id'].eq("").all():
        raise ValueError("Missing segment_id; cannot build sequences.")
    # fill missing values for categorical and numerical features (already done in encode_data and normalize_row) - add if actually needed

    # Temporal train-validation split by segments.
    gids = df['segment_id'].drop_duplicates().values
    n_seg = len(gids)
    if n_seg < 2:
        # not enough segments to split into train and validation
        raise ValueError(f"Need at least 2 segments for a train/val split, got {n_seg}.")
    n_val = max(1, round(args.val_frac * n_seg))
    if n_val >= n_seg:
        # not enough segments left for training after reserving validation
        raise ValueError(f"--val-frac {args.val_frac} leaves no training segments (total={n_seg}).")
    head = gids[:n_seg - n_val]  # training segments
    n_tr = max(1, round(args.train_frac * len(head))) # by default 1, this is to measure the effect of training data size on personalization
    tr_ids, te_ids = set(head[:n_tr]), set(gids[n_seg - n_val:])
    tr_df = df[df['segment_id'].isin(tr_ids)].reset_index(drop=True)
    te_df = df[df['segment_id'].isin(te_ids)].reset_index(drop=True)
    print(f"[split] temporal: train={n_tr}/{len(head)} earliest segments, val={n_val} latest segments (total={n_seg})")
    
    # logging
    log_fh = open(args.log_path, "w", encoding="utf-8") if args.log_path else None
    if log_fh:
        print(f"[log] writing feature log → {args.log_path}")
    
    # load model from checkpoint
    model, arch = load_checkpoint(args.in_model)
    context_length = arch["context_length"] # use same context length to build dataset than was used to train model
    # The head type (softmax+CE vs. soft-CORN ordinal) is NOT a CLI choice: it
    # is fixed by the population checkpoint (a CORN head has K-1 logits, so CE
    # would not even be shape-compatible). Old checkpoints lack the key -> softmax.
    head_type = arch.get("head_type", "softmax")
    # Same principle as head_type: the time window segments were cut to at
    # population training is a checkpoint property, not a CLI choice here.
    window_seconds = arch.get("window_seconds")
    # Same again for the resampling grid: serving or fine-tuning on a different
    # grid than the population model was trained on is exactly the mismatch the
    # contract exists to prevent. Pre-resampling checkpoints lack the key -> None.
    resample_hz = arch.get("resample_hz")
    print(f"[model] head_type={head_type} window_seconds={window_seconds} "
          f"resample_hz={resample_hz} (from checkpoint)")
    if args.laplace and head_type != "corn":
        raise ValueError(
            f"--laplace requires a CORN population checkpoint, got head_type={head_type!r}."
        )
    
    # create dataset
    try:
      train_ds = SeqDataset(tr_df, context_length=context_length, split="train", log_fh=log_fh,
                            window_seconds=window_seconds, resample_hz=resample_hz)
      test_ds  = SeqDataset(te_df, context_length=context_length, split="val",   log_fh=log_fh,
                            window_seconds=window_seconds, resample_hz=resample_hz)
    finally:
        # if error, close log
        if log_fh:
          log_fh.close()

    if len(train_ds) == 0 or len(test_ds) == 0:
        raise ValueError(f"Insufficient segments: train={len(train_ds)}, val={len(test_ds)}. Ensure Level_* labels exist.")

    # collate function for DataLoader to handle variable-length sequences.
    # These loaders now feed the frozen backbone ONCE (embed_all) and nothing
    # else — adaptation is full-batch on the cached embeddings — so --embed-batch
    # is a throughput/memory knob with no effect on the result: every sequence is
    # padded to context_length regardless of who it shares a batch with, and the
    # readout is taken at its own true length. shuffle=False keeps the cached
    # rows in dataset order, so Ztr[:k] means "the first k segments" for anyone
    # slicing a support prefix out of them (which is exactly what the sweep does).
    collate = make_collate(context_length)
    train_dl = DataLoader(train_ds, batch_size=args.embed_batch, shuffle=False, collate_fn=collate)
    test_dl  = DataLoader(test_ds,  batch_size=args.embed_batch, shuffle=False, collate_fn=collate)
    
    # set up model
    model.to(device)
    model.requires_grad_(False) # freeze all parameters
    model.head.requires_grad_(True) # only fine-tune head

    outp = pathlib.Path(args.out_pt); outp.parent.mkdir(parents=True, exist_ok=True)

    # precompute embeddings for all sequences in train and test datasets (more efficient - back-bone is frozen)
    # Widen the head BEFORE embedding, so `adapt_head` below anchors on the
    # zero-padded population head and the embeddings match its width. The gate
    # proves the padding is inert before anything is fitted.
    if args.embed_fcd:
        xb0, lb0, _ = next(iter(test_dl))
        assert_zero_block_identity(model, xb0.to(device), lb0.to(device))
        install_fcd_head(model, True)
        print(f"[embed-fcd] head widened to {model.head.in_features} inputs "
              f"({model.head.weight.numel() + model.head.bias.numel()} parameters)")
    Ztr, Vtr = embed_all(model, train_dl, device, args.embed_fcd)   # (n, 64) or (n, 76)
    Zte, Vte = embed_all(model, test_dl, device, args.embed_fcd)
    Ztr, Vtr = Ztr.to(device), Vtr.to(device)
    Zte, Vte = Zte.to(device), Vte.to(device)

    multi = int((Vtr.sum(dim=-1) > 1).sum())
    if multi:
        print(f"[info] {multi}/{len(Vtr)} fine-tuning segment(s) mark several acceptable LoAs.")

    # The population head is the anchor AND the initialization; adapt_head copies
    # it, so model.head still holds theta_pop until we install the result.
    head, info = adapt_head(model.head, Ztr, Vtr, tau=args.tau, head_type=head_type,
                            steps=args.steps, lr=args.lr)
    print(f"[adapt] K={info['n']} tau={info['tau']:.4g} -> lambda={info['l2sp']:.4g} | "
          f"{info['steps']} full-batch steps @ lr={info['lr']:g} | "
          f"final loss={info['final_loss']:.4f} |grad|={info['grad_norm']:.2e}")
    if info["grad_norm"] > 1e-3:
        # Not cosmetic: the Laplace layer expands about the MAP, so a head that
        # has not reached a stationary point invalidates the posterior's
        # exactness argument rather than merely being slightly undertrained.
        print(f"[adapt][warn] |grad| = {info['grad_norm']:.2e} — the head has NOT converged "
              f"to the MAP. Raise --steps or --lr; --laplace results are unreliable "
              f"until this is small.")
    model.head = head

    # ONE evaluation of the shipped head, on the temporal tail. The tail is for
    # REPORTING only now: it no longer picks an epoch, so it is not selected on
    # and reported from at the same time.
    with torch.no_grad():
        Yp = logits_to_label(model.head(Zte), head_type).cpu().numpy()
    Vl = Vte.cpu().numpy()
    # Set-aware throughout: a multi-label window is scored against the marked
    # level nearest the prediction, not against its lowest.
    sacc = set_accuracy(Vl, Yp)
    mf1 = set_macro_f1(Vl, Yp, 5)
    err = set_mae(Vl, Yp)
    kappa = set_qwk(Vl, Yp, 5)
    print(f"[val] set-acc={sacc:.3f} macro-F1={mf1:.3f} "
          f"set-MAE={err:.3f} QWK={kappa:.3f} (val_n={len(Yp)})")

    save_checkpoint(model, str(outp), arch=arch)
    print(f"[OK] saved -> {outp} (set-MAE={err:.3f})")

    if args.laplace:
        # Fit on the head we just shipped — no reload, because there is no
        # longer a "best" checkpoint that differs from the in-memory model.
        # l2sp comes from adapt_head, so the anchor the head was trained under
        # and the prior the posterior assumes are the same number by
        # construction (LaplacePosterior.fit re-derives tau = 2*N*l2sp from it).
        posterior = LaplacePosterior.fit(
            model.head.cpu(), Ztr.cpu(), Vtr.cpu(),
            l2sp=info["l2sp"], n_classes=model.n_classes,
        )
        model.head.to(device)
        attach_laplace_to_checkpoint(str(outp), posterior)
        print(f"[laplace] posterior over adapted soft-CORN head attached to {outp} "
              f"(n={posterior.n_examples}, soft_n_cond={posterior.n_cond_examples:.1f}, "
              f"tau={posterior.prior_precision:.4g})")

if __name__ == "__main__":
    main()
    

