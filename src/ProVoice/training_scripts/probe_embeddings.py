r"""Embedding informativeness — the probe ladder of ``docs/embedding_informativeness.md``.

WHAT THIS ANSWERS
-----------------
Both personalization arms freeze ``in_proj`` + backbone and adapt a
``Linear(64 -> 4)`` CORN head on top of the 64-dim embedding ``z``. Whatever a
linear head cannot extract from ``z``, no amount of L2-SP tuning or meta-learned
initialization can recover, so ``z`` is a hard ceiling on the whole
personalization layer. This script measures that ceiling.

Two experiments, one script, one split:

1. **The ladder** (§2) — run the SAME probe (linear CORN head, ``adapt_head``,
   same tau, same split) on inputs that each isolate one factor:

   ===  ======================================  ========================================
   row  probe input                             isolates
   ===  ======================================  ========================================
     0  nothing (constant baselines)            the floor
     1  the driver's previous label             within-session autocorrelation
     2  the 12 static FCD dims                  what the FUNCTION alone explains
     3  per-segment mean+std of the 33 features what the BACKBONE adds over raw signal
     4  ``z`` from an UNTRAINED backbone        what TRAINING the backbone adds
     5  ``z`` from ``pop_heldout_<pid>.pt``     the deployed representation
   ===  ======================================  ========================================

2. **The bias-only screen** (§4) — the decisive test. Adapt only the CORN head's
   4 BIASES (a per-driver level offset) versus all 260 parameters (offset plus a
   re-weighting of the embedding directions), at several K. If bias-only matches
   the full head, personalization on this representation reduces to learning each
   driver's preferred level, both arms are tied by construction, and that is a
   statable finding rather than the outcome of a ~10 h/arm meta-training run.

INFORMATIVENESS IS MEASURED IN NATS, AGAINST A REFERENCE (§1)
-------------------------------------------------------------
The quantity is V-information (Xu et al., ICLR 2020) — predictive information
relative to a restricted function family, here the linear CORN head that
actually ships::

    I_V(z -> Y) = H(Y) - min_theta E[ -log p_theta(Y | z) ]

The minimizer is ``head_adapt.adapt_head_tensors`` run to convergence and the
objective is ``xlstm_model.soft_corn_loss``, which is EXACTLY the cross-entropy
between the driver's label distribution and the CORN-decoded PMF (see
``docs/soft_corn_and_oldl.md`` §1) — a genuine categorical NLL, so the
subtraction is well-posed and the units are nats. Both references of §1 rule 2
are reported: the pooled train marginal (``iv_vs_marginal``) and the per-driver
train marginal (``iv_vs_pdconst``). The second is the binding one — a probe that
beats the pooled marginal but not the per-driver one has learned WHO is driving,
not WHEN they want autonomy.

Every number is held out (§1 rule 1): heads are fitted on the train side and
scored on the val side. In-sample V-information at n ~ 1150 with 260 parameters
is biased upward by a lot.

kNN mutual-information estimators are deliberately not used (§1 rule 3):
Kraskov-style estimators at d = 64 with n ~ 1450 are severely biased, and the
probe NLL is both cheaper and honest about what it measures.

PROTOCOL (§6) — the rules that keep the ladder comparable
---------------------------------------------------------
* **One split for every probe**: ``train_XLSTM.within_driver_temporal_split`` at
  ``--val-frac``. Per driver, the earliest 1-val_frac of their segments train and
  the latest tail evaluates. At ``--val-frac 0.2`` that tail is the SAME tail
  ``run_lodo_population`` reports as ``tail_*``, so the K=0 floor it prints and
  the numbers here are measured on identical segments.
* **Every per-driver statistic comes from the train prefix only** — centring
  means, constants, marginals, the K support sets. Fitting any of them on the
  val tail rebuilds exactly the leak this analysis exists to quantify.
* **Both regimes are reported**, side by side, for every row:
    - ``cross``  — the probe head is fitted on the OTHER 11 drivers' train
      prefixes and scored on the held-out driver's val tail. Identity-free.
    - ``within`` — the head is fitted on ALL 12 drivers' train prefixes,
      including the evaluated driver's own. Subject-dependent.
  The PAIR is the result; either alone is misleading.
* **>= 5 seeds** for the random backbone, with the spread reported. A single
  random init is a lottery ticket, not a control.
* **One tau across every probe.** A probe with a different anchor strength is a
  different estimator.

TWO CHOICES WORTH KNOWING ABOUT
-------------------------------
**The probe's anchor is zero, not a population head.** Rows 1-3 have no
population head to anchor to (their inputs are not 64-dim), so every ladder probe
is anchored at ``w=0, b=0`` and the L2-SP term degenerates to plain ridge at
precision tau. That makes one estimator serve all six rows. It also means the
ladder does NOT measure the deployed adaptation — experiment 2 does, and it
anchors at the real population head from the checkpoint.

**Ladder inputs are standardized; the bias screen's are not.** With a fixed tau,
an unstandardized input makes the ridge mean something different on every row —
``lead_distance_m`` reaches the model in raw metres, ~100x the dynamic range of
everything else (see CLAUDE.md). Standardizing with TRAIN-side statistics makes
tau comparable across the ladder, which is the ladder's entire purpose. The bias
screen instead runs on raw ``z`` against the checkpoint's own head, because there
the question is about the procedure that ships, not about a comparable estimator.
``--no-standardize`` turns the ladder's version off.

WHICH BACKBONE (§9)
-------------------
Row 5 and the bias screen read ``trained_models/lodo/pop_heldout_<pid>.pt`` —
written by ``run_lodo_population`` (stage 2), one per driver, each trained on the
other 11. Driver ``pid`` is embedded by the checkpoint that never saw them, so
these are the deployment-honest numbers that belong in the write-up. §4 describes
a cheaper one-way screen on a ``--split-mode within-driver`` backbone; that
exists to get an answer BEFORE stage 2, and is unnecessary once stage 2 has run.

Rows 0-4 need no trained backbone at all and run without ``--ckpt-dir``, which is
§9's recommended step 1. Missing checkpoints skip row 5 and the bias screen with
a warning rather than failing.

OUTPUTS
-------
  ``<outdir>/ladder.csv``         one row per (ladder row, regime, driver, seed)
  ``<outdir>/ladder_summary.csv`` means over drivers, per (row, regime)
  ``<outdir>/bias_vs_full.csv``   one row per (driver, K, variant)
  ``<outdir>/bias_summary.csv``   means over drivers, per (K, variant)

Usage::

    # step 1 of §9 — no model needed
    python -m ProVoice.training_scripts.probe_embeddings --rows 0,1,2,3,4

    # step 4 of §9 — after run_lodo_population has written the checkpoints
    python -m ProVoice.training_scripts.probe_embeddings
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import pathlib
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from ProVoice.models.head_adapt import (
    DEFAULT_ADAPT_LR, DEFAULT_ADAPT_STEPS, DEFAULT_TAU, adapt_head_tensors,
)
from ProVoice.models.train_XLSTM import (
    LEVELS, SEGMENT_CACHE_VERSION, SeqDataset, cache_name, constant_baseline,
    load_segment_cache, make_collate, per_driver_constant_baseline,
    set_accuracy, set_mae, set_qwk, within_driver_temporal_split,
)
from ProVoice.models.head_adapt import augment_z
from ProVoice.models.xlstm_model import (
    D_IN, DEFAULT_RESAMPLE_HZ, FCD_NAMES, FEATURE_NAMES, XLSTMSequenceClassifier,
    load_checkpoint, logits_to_probs, probs_to_label, soft_corn_loss,
)
from ProVoice.training_scripts.folds import ALL_PIDS

N_CLASSES = len(LEVELS)
N_CORN_UNITS = N_CLASSES - 1
# Mixed into every plug-in PMF before its cross-entropy is taken. A label
# distribution is exactly 0 outside the marked levels, so an unsmoothed plug-in
# scores -inf the first time the next window disagrees -- a number about one
# segment rather than about the predictor. Only the nll column of the plug-in
# persistence rule is affected; every fitted probe produces a strictly positive
# PMF and never touches this.
PLUGIN_EPS = 1e-2


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
@dataclass
class Segments:
    """The cached segments, plus the two orderings everything downstream needs."""
    X: List[np.ndarray]      # per segment, (T, D_IN) -- true T, never padded
    V: np.ndarray            # (N, 5) multi-hot marks, float32
    pid: np.ndarray          # (N,) participantid
    sid: np.ndarray          # (N,) segment_id
    chrono: np.ndarray       # (N,) rank in FIRST-APPEARANCE order (see below)
    ctx: int                 # context_length the cache was built for
    window_seconds: float
    resample_hz: float


def load_segments(cache_path: pathlib.Path, window_seconds: float,
                  resample_hz: float) -> Segments:
    """Read a segment cache built by ``scripts/build_segment_cache.py``.

    The staleness check covers the ENCODING contract (schema hash, ``D_IN``, the
    window and the grid) but deliberately not the source file's mtime: this
    script only reads embeddings and labels, so a re-touched
    ``labeled_data.jsonl`` should not block an analysis, whereas a schema change
    must.

    ``cache['segment_id']`` is in ``groupby`` (sorted) order while ``seg_order``
    is first-appearance order in the source file. Only the latter is chronology
    -- ids are ``<session_uuid>|win<i>p<prompt>``, so sorting them orders a
    driver's two sessions by UUID. ``chrono`` carries that ordering so the
    temporal split, the K prefixes and the persistence row all agree.
    """
    expect = {
        "version": SEGMENT_CACHE_VERSION,
        "window_seconds": float(window_seconds),
        "resample_hz": float(resample_hz),
        "d_in": int(D_IN),
        "features_sha": hashlib.sha256(",".join(FEATURE_NAMES).encode()).hexdigest()[:16],
    }
    cache = load_segment_cache(cache_path, expect, strict=True)
    x_flat, off = cache["x_flat"], cache["offsets"]
    sid = np.asarray([str(s) for s in cache["segment_id"]])
    pid = np.asarray([str(p) for p in cache["participantid"]])
    X = [x_flat[off[i]:off[i + 1]] for i in range(len(sid))]
    V = np.asarray(cache["levels"], dtype=np.float32)

    rank = {s: i for i, s in enumerate(cache["seg_order"])}
    missing = [s for s in sid if s not in rank]
    if missing:
        raise SystemExit(f"{len(missing)} cached segment(s) absent from seg_order "
                         f"(e.g. {missing[:3]}) — the cache is inconsistent; rebuild it.")
    chrono = np.asarray([rank[s] for s in sid], dtype=np.int64)

    meta = cache["meta"]
    ctx = int(round(float(meta["window_seconds"]) * float(meta["resample_hz"])))
    print(f"[data] {cache_path.name}: {len(sid)} segments, {len(np.unique(pid))} drivers, "
          f"window={meta['window_seconds']:g}s grid={meta['resample_hz']:g}Hz ctx={ctx}")
    return Segments(X=X, V=V, pid=pid, sid=sid, chrono=chrono, ctx=ctx,
                    window_seconds=float(meta["window_seconds"]),
                    resample_hz=float(meta["resample_hz"]))


def make_split(segs: Segments, val_frac: float) -> np.ndarray:
    """Boolean ``is_train`` mask from the ONE split every probe shares (§6 rule 1)."""
    order = np.argsort(segs.chrono)
    tr_ids, va_ids = within_driver_temporal_split(
        [segs.sid[i] for i in order], [segs.pid[i] for i in order], val_frac)
    is_train = np.asarray([s in tr_ids for s in segs.sid])
    unassigned = [s for s in segs.sid if s not in tr_ids and s not in va_ids]
    assert not unassigned, f"{len(unassigned)} segment(s) landed in neither half"
    return is_train


def driver_chrono_index(segs: Segments, pid: str, mask: np.ndarray) -> np.ndarray:
    """Indices of ``pid``'s segments under ``mask``, in chronological order."""
    idx = np.where((segs.pid == pid) & mask)[0]
    return idx[np.argsort(segs.chrono[idx])]


# --------------------------------------------------------------------------- #
# Likelihood bookkeeping (§1)
# --------------------------------------------------------------------------- #
def label_pmf(V: np.ndarray) -> np.ndarray:
    """Multi-hot marks -> per-row label distribution, uniform over marked levels."""
    return V / V.sum(axis=1, keepdims=True)


def xent(P: np.ndarray, V: np.ndarray) -> float:
    """Mean cross-entropy in NATS between the label distribution and a PMF ``P``.

    Same quantity ``soft_corn_loss`` returns for a CORN head, so a plug-in
    baseline and a fitted probe land on one scale and their difference is
    V-information.
    """
    if len(V) == 0:
        return float("nan")
    Q = label_pmf(V)
    P = np.broadcast_to(np.asarray(P, dtype=np.float64), Q.shape)
    return float(-(Q * np.log(np.clip(P, 1e-12, None))).sum(axis=1).mean())


def smooth(P: np.ndarray, eps: float = PLUGIN_EPS) -> np.ndarray:
    return (1.0 - eps) * P + eps / N_CLASSES


# --------------------------------------------------------------------------- #
# The probe
# --------------------------------------------------------------------------- #
@dataclass
class ProbeFit:
    w: torch.Tensor
    b: torch.Tensor
    mu: np.ndarray
    sd: np.ndarray
    n_fit: int
    d: int
    grad_norm: float
    final_loss: float


def fit_probe(Ftr: np.ndarray, Vtr: np.ndarray, *, tau: float, steps: int,
              lr: float, standardize: bool) -> ProbeFit:
    """Fit ``Linear(d -> K-1)`` CORN logits with the SAME optimizer that serves.

    Anchored at zero (see the module docstring), so the L2-SP penalty degenerates
    to ridge at precision tau and is identical across every ladder row.
    """
    Ftr = np.asarray(Ftr, dtype=np.float64)
    if standardize:
        mu = Ftr.mean(axis=0, keepdims=True)
        sd = Ftr.std(axis=0, keepdims=True)
        # A constant column carries no information; leaving sd at 0 would turn it
        # into inf/NaN rather than into the harmless zero it should be.
        sd = np.where(sd < 1e-8, 1.0, sd)
    else:
        mu = np.zeros((1, Ftr.shape[1]))
        sd = np.ones((1, Ftr.shape[1]))
    Z = torch.from_numpy(((Ftr - mu) / sd).astype(np.float32))
    V = torch.from_numpy(np.asarray(Vtr, dtype=np.float32))
    d = int(Z.shape[1])
    w0 = torch.zeros(N_CORN_UNITS, d)
    b0 = torch.zeros(N_CORN_UNITS)
    w, b, info = adapt_head_tensors(Z, V, w0, b0, tau=tau, head_type="corn",
                                    steps=steps, lr=lr)
    return ProbeFit(w=w, b=b, mu=mu, sd=sd, n_fit=int(Z.shape[0]), d=d,
                    grad_norm=info["grad_norm"], final_loss=info["final_loss"])


@torch.no_grad()
def eval_logits(logits: torch.Tensor, V_t: torch.Tensor,
                V_np: np.ndarray) -> Dict[str, float]:
    """Held-out NLL (nats) plus the set-aware metrics, from CORN logits."""
    probs = logits_to_probs(logits, "corn")
    yp = probs_to_label(probs, "corn").numpy()
    return {
        "nll": float(soft_corn_loss(logits, V_t)),
        "set_mae": float(set_mae(V_np, yp)),
        "set_acc": float(set_accuracy(V_np, yp)),
        "set_qwk": float(set_qwk(V_np, yp, N_CLASSES)),
    }


@torch.no_grad()
def eval_probe(fit: ProbeFit, Fva: np.ndarray, Vva: np.ndarray) -> Dict[str, float]:
    Fva = np.asarray(Fva, dtype=np.float64)
    Z = torch.from_numpy(((Fva - fit.mu) / fit.sd).astype(np.float32))
    V = torch.from_numpy(np.asarray(Vva, dtype=np.float32))
    return eval_logits(F.linear(Z, fit.w, fit.b), V, Vva)


# --------------------------------------------------------------------------- #
# Ladder feature builders (§2)
# --------------------------------------------------------------------------- #
def feats_fcd(segs: Segments) -> np.ndarray:
    """Row 2 — the 12 static FCD dims. Constant within a segment, so row 0 of it."""
    return np.stack([x[0, :len(FCD_NAMES)] for x in segs.X]).astype(np.float64)


def feats_raw_stats(segs: Segments) -> np.ndarray:
    """Row 3 — per-segment mean and std of all 33 encoded features (66 dims).

    Strictly contains row 2 (the FCD means ARE the FCD values, and their stds are
    identically 0), which is what makes the two a ladder: row 3 minus row 2 is
    what the raw driver-state and CARLA channels add before any sequence model
    touches them, and row 5 minus row 3 is what the backbone adds over that.
    """
    mean = np.stack([x.mean(axis=0) for x in segs.X])
    std = np.stack([x.std(axis=0) for x in segs.X])
    return np.concatenate([mean, std], axis=1).astype(np.float64)


def feats_prev_label(segs: Segments) -> np.ndarray:
    """Row 1 — the driver's PREVIOUS window's label, as a 5-dim PMF + a has-prev flag.

    Adjacent 20 s windows are heavily autocorrelated, so this is the honest floor
    for a temporal split: a strong predictor that uses no features at all. If the
    model only matches it, the within-driver result is autocorrelation rather
    than learning.

    TWO THINGS THIS IS NOT. It is not deployable — the served system does not see
    the previous label, so matching persistence demonstrates nothing either arm
    can use. And it is an UPPER control on the val tail specifically: from the
    second val segment on, its input is another val label, i.e. information the
    real system would not have. Both are the point; it is a ceiling on
    autocorrelation, not a candidate model.

    Chronology is per driver and spans the split, so the first val segment reads
    the last train segment. A driver's very first segment has no predecessor and
    gets a uniform PMF with the flag off; it is always on the train side (the
    split guarantees a non-empty prefix), so it never enters an evaluation.
    """
    N = len(segs.sid)
    out = np.zeros((N, N_CLASSES + 1), dtype=np.float64)
    P = label_pmf(segs.V)
    for pid in np.unique(segs.pid):
        idx = np.where(segs.pid == pid)[0]
        idx = idx[np.argsort(segs.chrono[idx])]
        for j, i in enumerate(idx):
            if j == 0:
                out[i, :N_CLASSES] = 1.0 / N_CLASSES
            else:
                out[i, :N_CLASSES] = P[idx[j - 1]]
                out[i, N_CLASSES] = 1.0
    return out


def persistence_plugin(segs: Segments) -> Tuple[np.ndarray, np.ndarray]:
    """The un-fitted persistence RULE: repeat the previous window's label.

    Returns ``(y_pred, pmf)``. The point prediction collapses a marked SET to the
    rounded mean of its levels — a fixed, prediction-independent rule, unlike
    ``resolve_targets``, so it cannot flatter itself on multi-label rows. The PMF
    is the previous label's distribution, smoothed by ``PLUGIN_EPS`` so its
    cross-entropy is finite (see that constant's note).
    """
    N = len(segs.sid)
    yp = np.zeros(N, dtype=int)
    pmf = np.full((N, N_CLASSES), 1.0 / N_CLASSES, dtype=np.float64)
    P = label_pmf(segs.V)
    lv = np.arange(N_CLASSES)
    for pid in np.unique(segs.pid):
        idx = np.where(segs.pid == pid)[0]
        idx = idx[np.argsort(segs.chrono[idx])]
        for j, i in enumerate(idx):
            if j == 0:
                yp[i] = int(round(float((P[i] * lv).sum())))   # train-side only
                continue
            prev = P[idx[j - 1]]
            pmf[i] = prev
            yp[i] = int(round(float((prev * lv).sum())))
    return yp, smooth(pmf)


# --------------------------------------------------------------------------- #
# Backbones (rows 4 and 5)
# --------------------------------------------------------------------------- #
@torch.no_grad()
def embed_all_segments(model: XLSTMSequenceClassifier, segs: Segments,
                       device: str, batch: int) -> np.ndarray:
    """Run the frozen ``in_proj`` + backbone over every segment; return (N, 64).

    Same readout as ``fine_tune_XLSTM.embed_all`` and ``model.forward``: the
    hidden state at the last REAL frame, since batches are right-padded and the
    stack is causal, so padding is exactly neutral rather than merely ignored.
    """
    ds = SeqDataset.__new__(SeqDataset)
    ds.context_length = segs.ctx
    ds.groups = [(segs.X[i], segs.V[i]) for i in range(len(segs.sid))]
    ds.pids = list(segs.pid)
    ds.segment_ids = list(segs.sid)
    dl = DataLoader(ds, batch_size=batch, shuffle=False,
                    collate_fn=make_collate(segs.ctx))
    model.eval().to(device)
    zs = []
    for xb, lb, _ in dl:
        x = xb.to(device).to(torch.float32)
        h = model.backbone(model.in_proj(x))
        idx = (lb.to(h.device).long() - 1).clamp(min=0)
        z = h[torch.arange(h.size(0), device=h.device), idx]
        # Inferred from the head's width, the same rule every other embedding
        # site uses. A plain population checkpoint is unaffected; an FCD-widened
        # one is probed at the width its head actually consumes instead of
        # raising a shape error several frames later.
        zs.append(augment_z(z, x, model.head_uses_fcd()).cpu())
    return torch.cat(zs).numpy().astype(np.float64)


def random_backbone(ctor_kwargs: Dict, seed: int) -> XLSTMSequenceClassifier:
    """An UNTRAINED backbone at the same arch — the reservoir control of §2 row 4.

    An untrained recurrent net is a strong random-feature reservoir, and the
    population sweep already shows trained CORN scoring worse than its own
    initialization at every window, which is precisely the signature of a
    backbone contributing nothing. Built from the checkpoint's own constructor
    kwargs so rows 4 and 5 differ ONLY by training, and averaged over >= 5 seeds
    because a single random init is a lottery ticket, not a control.
    """
    torch.manual_seed(seed)
    model = XLSTMSequenceClassifier(**ctor_kwargs)
    model.eval()
    return model


def checkpoint_path(ckpt_dir: pathlib.Path, pid: str) -> pathlib.Path:
    return ckpt_dir / f"pop_heldout_{pid}.pt"


def ctor_kwargs_from_arch(arch: Dict) -> Dict:
    """Model kwargs only — ``window_seconds``/``resample_hz`` are data contracts."""
    return {k: v for k, v in arch.items() if k not in ("window_seconds", "resample_hz")}


# --------------------------------------------------------------------------- #
# Experiment 1 — the ladder
# --------------------------------------------------------------------------- #
def run_ladder(segs: Segments, is_train: np.ndarray, drivers: List[str],
               rows: Sequence[int], args,
               embed_cache: Dict[str, np.ndarray]) -> List[dict]:
    """Every requested ladder row, in both regimes, per driver.

    A row whose features do not depend on the fold (1-4) has ONE ``within`` fit —
    its fit set is all 12 drivers' train prefixes regardless of which driver is
    being evaluated — so it is computed once and reused. That is an exact saving,
    not an approximation. ``cross`` always refits (its fit set drops the evaluated
    driver), and row 5 always refits because its embedding space changes with the
    backbone.
    """
    out: List[dict] = []
    V = segs.V
    tr_all = np.where(is_train)[0]

    # ---- row 0: the floor. No probe -- these ARE the references (§1 rule 2).
    if 0 in rows:
        va_all = np.where(~is_train)[0]
        glob = constant_baseline(V[tr_all], N_CLASSES)
        pdc = per_driver_constant_baseline(V[tr_all], segs.pid[tr_all],
                                           V[va_all], segs.pid[va_all], N_CLASSES)
        print(f"[row 0] best global train constant = LoA {glob['const_loa_mae']}; "
              f"per-driver constant set-MAE {pdc['pdconst_set_mae']:.3f} "
              f"(oracle {pdc['pdconst_oracle_set_mae']:.3f})")

    P_marginal = label_pmf(V[tr_all]).mean(axis=0)
    persist_yp, persist_pmf = persistence_plugin(segs)

    for pid in drivers:
        va = driver_chrono_index(segs, pid, ~is_train)
        tr_d = driver_chrono_index(segs, pid, is_train)
        if len(va) == 0 or len(tr_d) == 0:
            print(f"[warn] driver {pid} has an empty half; skipped")
            continue
        # Both references, fitted on the train side only (§6 rule 2).
        ref_marg = xent(P_marginal, V[va])
        ref_pd = xent(label_pmf(V[tr_d]).mean(axis=0), V[va])
        base = {"pid": pid, "n_eval": len(va), "n_train_seg": len(tr_d),
                "nll_marginal": ref_marg, "nll_pdconst": ref_pd}

        if 0 in rows:
            c = int(np.argmin([set_mae(V[tr_d], np.full(len(tr_d), k, dtype=int))
                               for k in range(N_CLASSES)]))
            yp = np.full(len(va), c, dtype=int)
            out.append({**base, "row": 0, "probe": "per-driver constant",
                        "regime": "cross", "seed": -1, "d": 0, "n_fit": len(tr_d),
                        "grad_norm": 0.0, "nll": ref_pd,
                        "set_mae": float(set_mae(V[va], yp)),
                        "set_acc": float(set_accuracy(V[va], yp)),
                        "set_qwk": float(set_qwk(V[va], yp, N_CLASSES)),
                        "iv_vs_marginal": ref_marg - ref_pd, "iv_vs_pdconst": 0.0})
            # The plug-in persistence RULE, reported next to the fitted row-1
            # probe because it is the rule §2 actually describes in prose.
            nll_p = xent(persist_pmf[va], V[va])
            out.append({**base, "row": 1, "probe": "persistence (plug-in)",
                        "regime": "cross", "seed": -1, "d": 0, "n_fit": 0,
                        "grad_norm": 0.0, "nll": nll_p,
                        "set_mae": float(set_mae(V[va], persist_yp[va])),
                        "set_acc": float(set_accuracy(V[va], persist_yp[va])),
                        "set_qwk": float(set_qwk(V[va], persist_yp[va], N_CLASSES)),
                        "iv_vs_marginal": ref_marg - nll_p,
                        "iv_vs_pdconst": ref_pd - nll_p})

    # ---- rows 1-5: fitted probes. (row, label, features(pid, seed), fold_dep, seeds)
    specs: List[Tuple[int, str, Callable[[str, int], np.ndarray], bool, List[int]]] = []
    if 1 in rows:
        Fp = feats_prev_label(segs)
        specs.append((1, "previous label", lambda pid, s: Fp, False, [0]))
    if 2 in rows:
        Ff = feats_fcd(segs)
        specs.append((2, "FCD only", lambda pid, s: Ff, False, [0]))
    if 3 in rows:
        Fr = feats_raw_stats(segs)
        specs.append((3, "raw mean+std", lambda pid, s: Fr, False, [0]))
    if 4 in rows:
        specs.append((4, "z (untrained backbone)",
                      lambda pid, s: embed_cache[f"rand{s}"], False,
                      list(range(args.random_seeds))))
    if 5 in rows:
        specs.append((5, "z (pop_heldout)", lambda pid, s: embed_cache[pid], True, [0]))

    for row, name, feats, fold_dep, seeds in specs:
        for seed in seeds:
            within_fit: Optional[ProbeFit] = None
            for pid in drivers:
                if row == 5 and pid not in embed_cache:
                    continue
                va = driver_chrono_index(segs, pid, ~is_train)
                tr_d = driver_chrono_index(segs, pid, is_train)
                if len(va) == 0 or len(tr_d) == 0:
                    continue
                Fmat = feats(pid, seed)
                ref_marg = xent(P_marginal, V[va])
                ref_pd = xent(label_pmf(V[tr_d]).mean(axis=0), V[va])
                base = {"pid": pid, "row": row, "probe": name, "seed": seed,
                        "n_eval": len(va), "n_train_seg": len(tr_d),
                        "nll_marginal": ref_marg, "nll_pdconst": ref_pd}

                # cross: fit on the OTHER 11 drivers' train prefixes.
                tr_x = tr_all[segs.pid[tr_all] != pid]
                fit_x = fit_probe(Fmat[tr_x], V[tr_x], tau=args.tau, steps=args.steps,
                                  lr=args.lr, standardize=not args.no_standardize)
                m = eval_probe(fit_x, Fmat[va], V[va])
                out.append({**base, "regime": "cross", "d": fit_x.d,
                            "n_fit": fit_x.n_fit, "grad_norm": fit_x.grad_norm, **m,
                            "iv_vs_marginal": ref_marg - m["nll"],
                            "iv_vs_pdconst": ref_pd - m["nll"]})

                # within: fit on ALL 12 drivers' train prefixes.
                if fold_dep or within_fit is None:
                    within_fit = fit_probe(Fmat[tr_all], V[tr_all], tau=args.tau,
                                           steps=args.steps, lr=args.lr,
                                           standardize=not args.no_standardize)
                m = eval_probe(within_fit, Fmat[va], V[va])
                out.append({**base, "regime": "within", "d": within_fit.d,
                            "n_fit": within_fit.n_fit,
                            "grad_norm": within_fit.grad_norm, **m,
                            "iv_vs_marginal": ref_marg - m["nll"],
                            "iv_vs_pdconst": ref_pd - m["nll"]})
            print(f"[row {row}] {name} (seed {seed}) done", flush=True)
    return out


# --------------------------------------------------------------------------- #
# Experiment 2 — the bias-only screen (§4)
# --------------------------------------------------------------------------- #
def run_bias_screen(segs: Segments, is_train: np.ndarray, drivers: List[str],
                    embed_cache: Dict[str, np.ndarray],
                    heads: Dict[str, Tuple[torch.Tensor, torch.Tensor]],
                    k_grid: List[int], args) -> List[dict]:
    """Bias-only (4 params) vs full-head (260) adaptation, per driver, per K.

    This is the DEPLOYED procedure, not a ladder probe: raw ``z``, the
    checkpoint's own head as both anchor and initialization, and the same
    ``head_adapt`` optimizer the fine-tuner and the sweep call. ``tau``, ``steps``
    and ``lr`` are identical across the two variants — otherwise the comparison
    measures optimizer budget rather than expressiveness (§4).

    Support is the driver's first K train-prefix segments in chronological order,
    which is where every serving path starts. ``K=0`` in the output is the
    UNADAPTED population head: the floor both variants have to beat, and the same
    quantity ``run_lodo_population`` prints as ``tail_set_mae`` when ``--val-frac``
    agrees.

    Direction of the result. The backbone here is the one that never saw this
    driver, so this is not §4's one-way screen — it is the deployment-honest
    comparison directly. If bias-only matches full-head here, the arms are tied on
    the representation that actually ships.
    """
    out: List[dict] = []
    V = segs.V
    for pid in drivers:
        if pid not in embed_cache or pid not in heads:
            continue
        Z = embed_cache[pid]
        w_pop, b_pop = heads[pid]
        va = driver_chrono_index(segs, pid, ~is_train)
        tr_d = driver_chrono_index(segs, pid, is_train)
        if len(va) == 0 or len(tr_d) == 0:
            continue
        Zva = torch.from_numpy(Z[va].astype(np.float32))
        Vva_t = torch.from_numpy(V[va])
        base = {"pid": pid, "n_eval": len(va), "n_train_seg": len(tr_d)}

        with torch.no_grad():
            m = eval_logits(F.linear(Zva, w_pop, b_pop), Vva_t, V[va])
        out.append({**base, "k_req": 0, "K": 0, "variant": "population",
                    "n_adapted": 0, "grad_norm": 0.0, "final_loss": float("nan"), **m})

        # k_req is the GRID value and K the realised support size. They differ
        # whenever a driver's prefix is shorter than the requested K, and always
        # for k_req=-1 ("all"), where K is 94-136 depending on the driver.
        # Summaries group on k_req, so a cell means "at this budget" across all
        # drivers instead of splitting into 12 groups of one.
        seen = set()
        for K in k_grid:
            k = len(tr_d) if K <= 0 else min(K, len(tr_d))
            k_req = -1 if K <= 0 else K
            if k in seen:      # a K past this driver's prefix collapses onto "all"
                continue
            seen.add(k)
            sup = tr_d[:k]
            Zs = torch.from_numpy(Z[sup].astype(np.float32))
            Vs = torch.from_numpy(V[sup])
            for variant, ap in (("full", "all"), ("bias", "bias")):
                w, b, info = adapt_head_tensors(
                    Zs, Vs, w_pop, b_pop, tau=args.tau, head_type="corn",
                    steps=args.steps, lr=args.lr, adapt_params=ap)
                with torch.no_grad():
                    m = eval_logits(F.linear(Zva, w, b), Vva_t, V[va])
                out.append({**base, "k_req": k_req, "K": k, "variant": variant,
                            "n_adapted": info["n_adapted"],
                            "grad_norm": info["grad_norm"],
                            "final_loss": info["final_loss"], **m})
        print(f"[bias] {pid}: {len(seen)} K x 2 variants done", flush=True)
    return out


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #
def write_csv(rows: List[dict], path: pathlib.Path) -> None:
    if not rows:
        return
    keys: List[str] = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in keys})
    print(f"[OK] -> {path}")


def summarize(rows: List[dict], by: Sequence[str],
              metrics: Sequence[str]) -> List[dict]:
    """Mean over DRIVERS, with the standard error.

    Never over segments: drivers contribute 94-136 segments each, so a
    segment-weighted mean would quietly weight the long sessions — the same
    convention ``run_lodo_population`` uses.
    """
    groups: Dict[tuple, List[dict]] = {}
    for r in rows:
        groups.setdefault(tuple(r[k] for k in by), []).append(r)
    out = []
    for key, rs in groups.items():
        rec = dict(zip(by, key))
        rec["n_rows"] = len(rs)
        rec["n_drivers"] = len({r["pid"] for r in rs})
        for m in metrics:
            v = np.asarray([r[m] for r in rs if r.get(m) is not None], dtype=float)
            v = v[np.isfinite(v)]
            rec[f"{m}_mean"] = float(v.mean()) if v.size else float("nan")
            rec[f"{m}_se"] = (float(v.std(ddof=1) / np.sqrt(v.size))
                              if v.size > 1 else float("nan"))
        out.append(rec)
    return out


def print_ladder(summary: List[dict]) -> None:
    print("\n=== LADDER — mean over drivers (I_V in nats; positive = the probe "
          "beats that reference) ===")
    hdr = (f"{'row':>3} {'probe':<24} {'regime':>7} {'d':>4} "
           f"{'NLL':>7} {'I_V|marg':>9} {'I_V|pdc':>8} {'setMAE':>7} {'setQWK':>7}")
    print(hdr)
    print("-" * len(hdr))
    for r in sorted(summary, key=lambda x: (x["row"], x["probe"], x["regime"])):
        print(f"{r['row']:>3} {r['probe'][:24]:<24} {r['regime']:>7} "
              f"{int(r.get('d_mean') or 0):>4} "
              f"{r['nll_mean']:>7.3f} {r['iv_vs_marginal_mean']:>9.3f} "
              f"{r['iv_vs_pdconst_mean']:>8.3f} {r['set_mae_mean']:>7.3f} "
              f"{r['set_qwk_mean']:>7.3f}")
    print("\nI_V|pdc <= 0 means the probe carries nothing beyond the driver's own "
          "constant.\nRead row 5 vs row 4 (does TRAINING the backbone add anything?), "
          "row 5 vs row 3\n(does the SEQUENCE MODEL add anything over per-segment "
          "aggregates?) and row 1\n(is the within-driver result just "
          "autocorrelation?) — §7 of the design doc.")


def print_bias(summary: List[dict]) -> None:
    print("\n=== BIAS-ONLY vs FULL HEAD — mean over drivers ===")
    # k_req = -1 means "the driver's whole train prefix", which belongs at the
    # END of the budget axis, not before K=0. Sorting maps it to +inf; the CSV
    # keeps -1, which reads better there than a magic large number would.
    def order(k: int) -> float:
        return float("inf") if k < 0 else float(k)

    def label(r) -> str:
        return "all" if r["k_req"] < 0 else str(r["k_req"])

    hdr = (f"{'K':>5} {'meanK':>6} {'variant':>11} {'params':>7} {'NLL':>7} "
           f"{'setMAE':>8} {'setACC':>8} {'setQWK':>8}")
    print(hdr)
    print("-" * len(hdr))
    for r in sorted(summary, key=lambda x: (order(x["k_req"]), x["variant"])):
        print(f"{label(r):>5} {r['K_mean']:>6.0f} {r['variant']:>11} "
              f"{int(r.get('n_adapted_mean') or 0):>7} "
              f"{r['nll_mean']:>7.3f} {r['set_mae_mean']:>8.3f} "
              f"{r['set_acc_mean']:>8.3f} {r['set_qwk_mean']:>8.3f}")
    by_k: Dict[int, Dict[str, float]] = {}
    for r in summary:
        by_k.setdefault(r["k_req"], {})[r["variant"]] = r["set_mae_mean"]
    print("\nfull-minus-bias set-MAE (negative = the 256 weights buy something):")
    for K in sorted(by_k, key=order):
        d = by_k[K]
        if "full" in d and "bias" in d:
            print(f"  K={('all' if K < 0 else K):<4} {d['full'] - d['bias']:+.4f}")
    print("If that column is ~0 at every K, personalization on this representation "
          "reduces to\nlearning each driver's preferred level: both study arms are "
          "tied by construction,\nand the contribution pivots to the K-vs-satisfaction "
          "question the live study answers.")


# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache", default="",
                    help="Segment cache. Default: <--cache-dir>/segments_w<win>_hz<hz>.npz, "
                         "with the window and grid taken from the checkpoints when present.")
    ap.add_argument("--cache-dir", dest="cache_dir", default="data/cache")
    ap.add_argument("--window-seconds", dest="window_seconds", type=float, default=10.0,
                    help="Only used when there is no checkpoint to read it from. With "
                         "checkpoints their arch wins — a probe on a different window "
                         "than the backbone was trained for is meaningless.")
    ap.add_argument("--resample-hz", dest="resample_hz", type=float,
                    default=DEFAULT_RESAMPLE_HZ)
    ap.add_argument("--ckpt-dir", dest="ckpt_dir", default="trained_models/lodo",
                    help="Stage 2's output. Absent => rows 0-4 only, no bias screen.")
    ap.add_argument("--outdir", default="results/embedding_probes")
    ap.add_argument("--val-frac", dest="val_frac", type=float, default=0.3,
                    help="MUST match run_lodo_population's --val-frac, or its floor and "
                         "these numbers are measured on different segments.")
    ap.add_argument("--tau", type=float, default=DEFAULT_TAU,
                    help="Prior precision, ONE value for every probe (§6 rule 5). "
                         "lambda = tau/(2n) is derived inside head_adapt.")
    ap.add_argument("--steps", type=int, default=DEFAULT_ADAPT_STEPS)
    ap.add_argument("--lr", type=float, default=DEFAULT_ADAPT_LR)
    ap.add_argument("--rows", default="0,1,2,3,4,5",
                    help="Ladder rows to run. Rows 0-4 need no trained backbone.")
    ap.add_argument("--random-seeds", dest="random_seeds", type=int, default=5,
                    help="Untrained backbones for row 4. >= 5 (§6 rule 4).")
    ap.add_argument("--k-grid", dest="k_grid", default="5,10,20,30,50,0",
                    help="Support sizes for the bias screen; 0 = the driver's whole "
                         "train prefix. Each K is clipped to what the driver has, and "
                         "duplicates after clipping are dropped. The CSV keeps both the "
                         "grid value (k_req) and the realised size (K).")
    ap.add_argument("--pids", default="", help="Comma-separated subset of drivers.")
    ap.add_argument("--embed-batch", dest="embed_batch", type=int, default=32)
    ap.add_argument("--device", default="")
    ap.add_argument("--no-standardize", dest="no_standardize", action="store_true",
                    help="Feed the ladder probes raw features. Off by default — see the "
                         "module docstring on why tau needs a common scale.")
    ap.add_argument("--skip-ladder", dest="skip_ladder", action="store_true")
    ap.add_argument("--skip-bias", dest="skip_bias", action="store_true")
    args = ap.parse_args()

    rows = sorted({int(r) for r in args.rows.split(",") if r.strip()})
    k_grid = [int(k) for k in args.k_grid.split(",") if k.strip()]
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    outdir = pathlib.Path(args.outdir)
    ckpt_dir = pathlib.Path(args.ckpt_dir)

    # --- Which checkpoints exist. Their arch decides the cache, because a probe
    # on a different window than the backbone was trained for is meaningless.
    ckpts = {p: checkpoint_path(ckpt_dir, p) for p in ALL_PIDS
             if checkpoint_path(ckpt_dir, p).exists()}
    ctor_kwargs: Optional[Dict] = None
    window_seconds, resample_hz = args.window_seconds, args.resample_hz
    if ckpts:
        _m, arch0 = load_checkpoint(str(ckpts[sorted(ckpts)[0]]))
        ctor_kwargs = ctor_kwargs_from_arch(arch0)
        window_seconds = float(arch0.get("window_seconds") or args.window_seconds)
        resample_hz = float(arch0.get("resample_hz") or args.resample_hz)
        print(f"[ckpt] {len(ckpts)}/{len(ALL_PIDS)} LODO checkpoints in {ckpt_dir} "
              f"(head={arch0.get('head_type')} window={window_seconds:g}s "
              f"grid={resample_hz:g}Hz)")
        if arch0.get("head_type") != "corn":
            raise SystemExit(
                f"checkpoints carry head_type={arch0.get('head_type')!r}. Every probe "
                "here is a CORN head and the bias screen adapts the checkpoint's own "
                "head, so a softmax population model cannot be put on this scale. "
                "Retrain stage 2 with --loss corn (the Laplace layer needs it anyway).")
        if len(ckpts) < len(ALL_PIDS):
            print(f"[warn] missing: {[p for p in ALL_PIDS if p not in ckpts]}. Row 5 and "
                  f"the bias screen cover only the drivers that have a checkpoint.")
    else:
        print(f"[ckpt] none in {ckpt_dir} — rows 0-4 only, no bias screen. That is §9 "
              f"step 1; run ProVoice.training_scripts.run_lodo_population for the rest.")
        rows = [r for r in rows if r != 5]
        args.skip_bias = True

    cache_path = (pathlib.Path(args.cache) if args.cache else
                  pathlib.Path(args.cache_dir) / cache_name(window_seconds, resample_hz))
    segs = load_segments(cache_path, window_seconds, resample_hz)
    if ctor_kwargs is None:
        ctor_kwargs = {"d_in": D_IN, "n_classes": N_CLASSES,
                       "context_length": segs.ctx, "head_type": "corn"}

    is_train = make_split(segs, args.val_frac)
    want = {p.strip() for p in args.pids.split(",") if p.strip()}
    drivers = [p for p in ALL_PIDS if p in set(segs.pid) and (not want or p in want)]
    if not drivers:
        raise SystemExit("no drivers to evaluate")

    # --- Embeddings. Row 5 and the bias screen share them, so each backbone makes
    # exactly one forward pass over the corpus. A random backbone has seen nobody,
    # so one per SEED serves all 12 folds; a LODO backbone is per fold.
    embed_cache: Dict[str, np.ndarray] = {}
    heads: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}
    if 4 in rows and not args.skip_ladder:
        if args.random_seeds < 5:
            print(f"[warn] --random-seeds {args.random_seeds} < 5: a single random init "
                  f"is a lottery ticket, not a control (§6 rule 4).")
        for s in range(args.random_seeds):
            embed_cache[f"rand{s}"] = embed_all_segments(
                random_backbone(ctor_kwargs, s), segs, device, args.embed_batch)
            print(f"[embed] untrained backbone seed {s}", flush=True)
    if (5 in rows and not args.skip_ladder) or not args.skip_bias:
        for pid in drivers:
            if pid not in ckpts:
                continue
            model, _arch = load_checkpoint(str(ckpts[pid]))
            embed_cache[pid] = embed_all_segments(model, segs, device, args.embed_batch)
            heads[pid] = (model.head.weight.detach().cpu().clone(),
                          model.head.bias.detach().cpu().clone())
            print(f"[embed] pop_heldout_{pid}", flush=True)

    metrics = ("nll", "iv_vs_marginal", "iv_vs_pdconst", "set_mae", "set_acc",
               "set_qwk", "d")
    if not args.skip_ladder:
        ladder = run_ladder(segs, is_train, drivers, rows, args, embed_cache)
        write_csv(ladder, outdir / "ladder.csv")
        summary = summarize(ladder, ("row", "probe", "regime"), metrics)
        write_csv(summary, outdir / "ladder_summary.csv")
        print_ladder(summary)

    if not args.skip_bias:
        bias = run_bias_screen(segs, is_train, drivers, embed_cache, heads, k_grid, args)
        write_csv(bias, outdir / "bias_vs_full.csv")
        bsum = summarize(bias, ("k_req", "variant"),
                         ("nll", "set_mae", "set_acc", "set_qwk", "n_adapted", "K"))
        write_csv(bsum, outdir / "bias_summary.csv")
        print_bias(bsum)


if __name__ == "__main__":
    main()
