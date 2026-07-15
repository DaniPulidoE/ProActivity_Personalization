import argparse, pathlib

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ProVoice.fcd_config import FCD_NAMES, get_fcd_for_function
from ProVoice.models.xlstm_model import (
    save_checkpoint,
    load_checkpoint,
    logits_to_probs,
)
from ProVoice.models.xlstm_model import _as01
from ProVoice.models.laplace_head import LaplacePosterior, attach_laplace_to_checkpoint
from coral_pytorch.losses import corn_loss

from ProVoice.models.train_XLSTM import (
    set_seed,
    read_jsonl,
    normalize_row,
    SeqDataset,
    make_collate,
    accuracy,
    macro_f1,
    mae,
    qwk,
)

LEVELS = [f"Level_{i}" for i in range(1, 6)]

@torch.no_grad()
def embed_all(model, dl, device):
    # function that precomputes the embeddings for all sequences in a dataloader, using the frozen in_proj and backbone of the model. Returns pooled embeddings and labels.
    """Run the frozen in_proj+backbone once; return pooled embeddings and labels."""
    zs, ys = [], []
    for xb, yb, lb in dl:
        h = model.backbone(model.in_proj(xb.to(device).to(torch.float32)))
        # Batches are RIGHT-padded: read the hidden state at the last REAL
        # frame (index length-1), same readout as model.forward.
        idx = (lb.to(h.device).long() - 1).clamp(min=0)
        zs.append(h[torch.arange(h.size(0), device=h.device), idx].cpu())
        ys.append(yb)
    return torch.cat(zs), torch.cat(ys)
    
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
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch",  type=int, default=16)
    ap.add_argument("--seed",   type=int, default=42)
    ap.add_argument("--lr",     type=float, default=2e-3)
    ap.add_argument("--l2sp",   type=float, default=0.01,
                    help="Strength λ of the L2-SP penalty λ·||θ_head − θ_pop||² anchoring the "
                         "fine-tuned head to the population head. 0 disables; larger values "
                         "keep the personalized model closer to the base model.")
    ap.add_argument("--val-frac", dest="val_frac", type=float, default=0.2,
                    help="Fraction of segments reserved as the chronologically-LAST validation "
                         "tail. Keep this fixed across runs so learning curves share one "
                         "measuring stick.")
    ap.add_argument("--train-frac", dest="train_frac", type=float, default=1.0,
                    help="Fraction of the non-validation segments (earliest first) used for "
                         "training. Sweep this (e.g. 0.2, 0.5, 1.0) to measure personalization "
                         "vs. data collection time against the fixed validation tail.")
    ap.add_argument("--laplace", action="store_true",
                    help="After training, fit a Laplace posterior over the adapted CORN head "
                         "(exact per-unit Hessian on the training embeddings, prior precision "
                         "from --l2sp) and store it inside the output checkpoint. Requires a "
                         "CORN population checkpoint and --l2sp > 0.")
    args = ap.parse_args()
    if not (0.0 < args.val_frac < 1.0):
        raise ValueError(f"--val-frac must be in (0, 1), got {args.val_frac}")
    if not (0.0 < args.train_frac <= 1.0):
        raise ValueError(f"--train-frac must be in (0, 1], got {args.train_frac}")
    if args.laplace and args.l2sp <= 0.0:
        raise ValueError("--laplace requires --l2sp > 0 (the L2-SP anchor defines the prior).")
    
    # seed and cuda
    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # build df
    rows = [normalize_row(r) for r in read_jsonl(pathlib.Path(args.in_jsonl))]
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
    # The head type (softmax+CE vs. CORN ordinal) is NOT a CLI choice: it is
    # fixed by the population checkpoint (a CORN head has K-1 logits, so CE
    # would not even be shape-compatible). Old checkpoints lack the key -> softmax.
    head_type = arch.get("head_type", "softmax")
    # Same principle as head_type: the time window segments were cut to at
    # population training is a checkpoint property, not a CLI choice here.
    window_seconds = arch.get("window_seconds")
    print(f"[model] head_type={head_type} window_seconds={window_seconds} (from checkpoint)")
    if args.laplace and head_type != "corn":
        raise ValueError(
            f"--laplace requires a CORN population checkpoint, got head_type={head_type!r}."
        )
    
    # create dataset
    try:
      train_ds = SeqDataset(tr_df, context_length=context_length, split="train", log_fh=log_fh,
                            window_seconds=window_seconds)
      test_ds  = SeqDataset(te_df, context_length=context_length, split="val",   log_fh=log_fh,
                            window_seconds=window_seconds)
    finally:
        # if error, close log
        if log_fh:
          log_fh.close()

    if len(train_ds) == 0 or len(test_ds) == 0:
        raise ValueError(f"Insufficient segments: train={len(train_ds)}, val={len(test_ds)}. Ensure Level_* labels exist.")

    # collate function for DataLoader to handle variable-length sequences
    collate = make_collate(context_length)
    train_dl = DataLoader(train_ds, batch_size=args.batch, shuffle=True,  collate_fn=collate)
    test_dl  = DataLoader(test_ds,  batch_size=max(8, args.batch), shuffle=False, collate_fn=collate)
    
    # set up model
    model.to(device)
    model.requires_grad_(False) # freeze all parameters
    model.head.requires_grad_(True) # only fine-tune head
    theta_pop = {n: p.detach().clone() for n, p in model.head.named_parameters()} # anchor weights (population model)
    opt = torch.optim.AdamW(model.head.parameters(), lr=args.lr, weight_decay=0.0) # no weight decay (penalty on distance to population model weights)
    if head_type == "corn":
        loss_fn = lambda logits, target: corn_loss(logits, target, num_classes=5)
    else:
        loss_fn = nn.CrossEntropyLoss()

    # model saving setup
    best = -1.0  # ensures the first epoch always saves, so a checkpoint always exists
    outp = pathlib.Path(args.out_pt); outp.parent.mkdir(parents=True, exist_ok=True)
    
    # precompute embeddings for all sequences in train and test datasets (more efficient - back-bone is frozen)
    Ztr, ytr = embed_all(model, train_dl, device)   # (n_train, 64), (n_train,), Z represents the embedding
    Zte, yte = embed_all(model, test_dl, device)
    Ztr, ytr = Ztr.to(device), ytr.to(device)
    Zte, yte = Zte.to(device), yte.to(device)

    
    n = Ztr.shape[0] # number of training samples
    for ep in range(args.epochs):
        # train
        perm = torch.randperm(n, device=device) # replaces DataLoader shuffling
        for i in range(0, n, args.batch):
            idx = perm[i:i + args.batch]
            logits = model.head(Ztr[idx])
            l2sp = sum(((p - theta_pop[n2]) ** 2).sum()
                    for n2, p in model.head.named_parameters())
            loss = loss_fn(logits, ytr[idx]) + args.l2sp * l2sp
            opt.zero_grad()
            loss.backward()
            opt.step()

        # evaluation: one matrix multiply, no batching needed
        with torch.no_grad():
            Yp = logits_to_probs(model.head(Zte), head_type).argmax(dim=-1).cpu().numpy()
        Yt = yte.cpu().numpy()
        acc = accuracy(Yt, Yp)
        mf1 = macro_f1(Yt, Yp, 5)
        err = mae(Yt, Yp)
        kappa = qwk(Yt, Yp, 5)
        print(f"[epoch {ep:02d}] acc={acc:.3f} macro-F1={mf1:.3f} MAE={err:.3f} QWK={kappa:.3f} (val_n={len(Yt)})")

        # save model if best so far (or always, for now)
        if acc > best:
            best = acc
            # DECIDE IN HOW TO MANAGE THIS LOGIC AFTERWARDS
            save_checkpoint(model, str(outp), arch=arch)
        print(f"[OK] saved-> {outp}")
        
    print(f"[BEST] acc={best:.3f}")

    if args.laplace:
        # The Laplace expansion is only valid at the head being shipped: the
        # in-memory model holds the LAST epoch, but the checkpoint holds the
        # best one — reload it and fit on the same adaptation embeddings.
        best_model, _ = load_checkpoint(str(outp))
        posterior = LaplacePosterior.fit(
            best_model.head, Ztr.cpu(), ytr.cpu(),
            l2sp=args.l2sp, n_classes=best_model.n_classes,
        )
        attach_laplace_to_checkpoint(str(outp), posterior)
        print(f"[laplace] posterior over adapted CORN head attached to {outp} "
              f"(n_cond={posterior.n_cond_examples}, tau={posterior.prior_precision:.4g})")

if __name__ == "__main__":
    main()
    

