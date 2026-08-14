"""Numerical-stability stress test for soft_corn_loss.

Run:  uv run python -m scripts.stress_soft_corn
Results and interpretation: docs/soft_corn_and_oldl.md §1.7.
"""
import torch
import torch.nn.functional as F
from ProVoice.models.xlstm_model import (
    soft_corn_loss, levels_to_distribution, levels_to_cumulative, levels_to_subset_weights,
)

K = 5
def lv_of(*marks):
    v = torch.zeros(1, K)
    for m in marks:
        v[0, m] = 1.0
    return v

print("=" * 70)
print("A. EXTREME LOGITS — does the loss stay finite?")
print("=" * 70)
lv = lv_of(2)                       # y=2 -> q=[1,1,0,0], p_k=[0,0,1,0]
for mag in [10, 50, 100, 1e3, 1e4, 1e8, 1e30, 3e38, float("inf")]:
    for sign in (+1, -1):
        z = torch.full((1, K - 1), float(sign * mag), dtype=torch.float32, requires_grad=True)
        loss = soft_corn_loss(z, lv)
        g = torch.autograd.grad(loss, z, retain_graph=True)[0] if torch.isfinite(loss) else None
        gs = "n/a" if g is None else f"max|g|={g.abs().max():.4g} finite={bool(torch.isfinite(g).all())}"
        print(f"  logits={sign*mag:>12.4g}  loss={loss.item():>14.6g}  "
              f"finite={bool(torch.isfinite(loss))}  {gs}")

print()
print("=" * 70)
print("B. THE 0 * inf TRAP — q_k=0 paired with logit -inf")
print("=" * 70)
# q_3 = 0 for y=2. Drive ONLY that logit to -inf.
z = torch.tensor([[0.0, 0.0, 0.0, -float("inf")]], requires_grad=True)
loss = soft_corn_loss(z, lv)
print(f"  q = {levels_to_cumulative(lv).tolist()[0]}  (q_3 = 0)")
print(f"  loss with logits[3] = -inf : {loss.item()}  -> NaN? {bool(torch.isnan(loss))}")
z2 = torch.tensor([[0.0, 0.0, 0.0, -1e38]], requires_grad=True)
loss2 = soft_corn_loss(z2, lv)
print(f"  loss with logits[3] = -1e38: {loss2.item():.6f}  -> NaN? {bool(torch.isnan(loss2))}")
print("  (float32 saturates at ~3.4e38; true inf requires the head to have already blown up)")

print()
print("=" * 70)
print("C. GRADIENT BOUND — analytic dL/dz_k = q_{k-1}*sigma(z_k) - q_k, so |g| <= 1")
print("=" * 70)
torch.manual_seed(0)
worst = 0.0
for trial in range(3000):
    B = 8
    z = (torch.randn(B, K - 1) * 10 ** torch.randint(0, 5, (1,)).item()).requires_grad_(True)
    y = torch.randint(0, K, (B,))
    v = F.one_hot(y, K).float()
    if trial % 3 == 0:                                  # inject multi-label rows
        v[torch.arange(B), torch.randint(0, K, (B,))] = 1.0
    loss = soft_corn_loss(z, v)
    g = torch.autograd.grad(loss, z)[0]
    assert torch.isfinite(loss) and torch.isfinite(g).all(), f"non-finite at trial {trial}"
    worst = max(worst, (g * B).abs().max().item())      # undo the batch mean
print(f"  3000 random trials, logit scales up to 1e4, incl. multi-label")
print(f"  worst per-sample |dL/dz| = {worst:.6f}   (analytic bound = 1.0)")
assert worst <= 1.0 + 1e-5, "gradient exceeded its analytic bound!"
print("  OK: gradient is bounded by construction -> the loss cannot explode gradients")

print()
print("=" * 70)
print("D. vs NAIVE log(sigmoid(x)) — the implementation this avoids")
print("=" * 70)
def naive(logits, levels):
    p = levels_to_distribution(levels); q = levels_to_cumulative(levels)
    return -(q * torch.log(torch.sigmoid(logits))
             + p[..., :-1] * torch.log(1 - torch.sigmoid(logits))).sum(-1).mean()
for mag in [20, 40, 80, 100]:
    z = torch.full((1, K - 1), -float(mag), dtype=torch.float32, requires_grad=True)
    a, b = soft_corn_loss(z, lv), naive(z, lv)
    ga = torch.autograd.grad(a, z, retain_graph=True)[0]
    z2 = torch.full((1, K - 1), -float(mag), dtype=torch.float32, requires_grad=True)
    gb = torch.autograd.grad(naive(z2, lv), z2)[0]
    print(f"  logit=-{mag:<4} logsigmoid={a.item():>12.4g} (g ok={bool(torch.isfinite(ga).all())})  "
          f"naive={b.item():>12.4g} (g ok={bool(torch.isfinite(gb).all())})")

print()
print("=" * 70)
print("E. DEGENERATE / EDGE-CASE LABELS")
print("=" * 70)
z = torch.randn(1, K - 1, requires_grad=True)
cases = {
    "all levels marked (max ambiguity)":            torch.ones(1, K),
    "gapped {0,4}":                                  lv_of(0, 4),
    "boundary y=0":                                  lv_of(0),
    "boundary y=K-1":                                lv_of(K - 1),
    "{0,1,2,3,4} minus middle -> {0,1,3,4}":         lv_of(0, 1, 3, 4),
}
for name, v in cases.items():
    loss = soft_corn_loss(z, v)
    g = torch.autograd.grad(loss, z, retain_graph=True)[0]
    print(f"  {name:<46} loss={loss.item():>10.6f}  finite={bool(torch.isfinite(loss))}  "
          f"g finite={bool(torch.isfinite(g).all())}")

# Rows carrying no usable label are now REJECTED rather than absorbed. They used
# to be inert in the loss (p = q = 0) but levels_to_subset_weights still gave
# them a hard 1.0 at threshold 0, so an unmarked row contributed a full
# observation of data curvature to the Laplace posterior while contributing
# nothing to the fit. Negative entries were worse: clamp(min=1e-8) on a
# non-positive row sum produced probabilities of +-1e8 and silently voided the
# |dL/dz| <= 1 bound that section C verifies.
print()
rejected = {
    "all-zero levels (no LoA marked)": torch.zeros(1, K),
    "zero row inside a valid batch":   torch.cat([lv_of(0), torch.zeros(1, K)]),
    "negative entry":                  torch.tensor([[1.0, -1.0, 0.0, 0.0, 0.0]]),
}
for name, v in rejected.items():
    try:
        soft_corn_loss(torch.zeros(v.shape[0], K - 1), v)
        print(f"  {name:<46} NOT REJECTED  <-- regression")
    except ValueError:
        print(f"  {name:<46} rejected (ValueError)  OK")

print()
print("=" * 70)
print("F. DTYPE — float64 reference, float32, float16")
print("=" * 70)
torch.manual_seed(1)
z64 = torch.randn(64, K - 1, dtype=torch.float64)
v64 = F.one_hot(torch.randint(0, K, (64,)), K).double()
v64[0, 0] = 1.0; v64[0, 4] = 1.0        # one gapped row
ref = soft_corn_loss(z64, v64)
f32 = soft_corn_loss(z64.float(), v64.float())
print(f"  float64 = {ref.item():.12f}")
print(f"  float32 = {f32.item():.12f}   rel.err = {abs(f32.item()-ref.item())/abs(ref.item()):.3e}")
try:
    f16 = soft_corn_loss(z64.half(), v64.half())
    print(f"  float16 = {f16.item():.6f}   rel.err = {abs(f16.item()-ref.item())/abs(ref.item()):.3e}")
except Exception as e:
    print(f"  float16 raised: {e}")
# The old clamp(min=1e-8) underflowed to 0 in fp16 (min subnormal ~6e-8), so a
# degenerate row became 0/0 = NaN there while merely being inert in fp32 — the
# guard was dtype-dependent. levels_to_distribution now rejects such rows in
# every dtype, and soft_corn_loss still promotes targets to >= fp32 so the PMF
# and its CDF are not quantized before being used as regression weights.
print(f"  NOTE degenerate rows now raise in every dtype (was: fp32 inert, fp16 NaN)")

print()
print("=" * 70)
print("G. LAPLACE HESSIAN CONDITIONING")
print("=" * 70)
import torch.nn as nn
from ProVoice.models.laplace_head import LaplacePosterior
torch.manual_seed(2)
head = nn.Linear(64, K - 1)
for name, (N, scale, l2sp) in {
    "typical K=30":                 (30, 1.0, 0.01),
    "tiny K=3 (cold start)":        (3,  1.0, 0.01),
    "K=1 (single label)":           (1,  1.0, 0.01),
    "large embeddings (scale 100)": (30, 100.0, 0.01),
    "tiny l2sp=1e-6":               (30, 1.0, 1e-6),
}.items():
    Z = torch.randn(N, 64) * scale
    LV = F.one_hot(torch.randint(0, K, (N,)), K).float()
    try:
        post = LaplacePosterior.fit(head, Z, LV, l2sp=l2sp, n_classes=K)
        H = post.chol_prec @ post.chol_prec.transpose(-1, -2)
        cond = torch.linalg.cond(H)
        pm = post.predictive_pmf(Z, n_samples=8)
        print(f"  {name:<30} tau={post.prior_precision:<10.4g} "
              f"max cond(H)={cond.max():.3e}  PMF finite={bool(torch.isfinite(pm).all())}")
    except Exception as e:
        print(f"  {name:<30} RAISED: {type(e).__name__}: {e}")

# thresholds a driver's labels never reach -> pure prior block
Z = torch.randn(10, 64)
LV = F.one_hot(torch.zeros(10, dtype=torch.long), K).float()   # every label is class 0
post = LaplacePosterior.fit(head, Z, LV, l2sp=0.01, n_classes=K)
Wt = levels_to_subset_weights(LV)
print(f"  all-labels-class-0: subset weights per unit = {Wt.sum(0).tolist()} "
      f"(units 1..3 see NO data -> pure prior block)")
H = post.chol_prec @ post.chol_prec.transpose(-1, -2)
print(f"    unit 3 Hessian == tau*I ? "
      f"{bool(torch.allclose(H[3], post.prior_precision*torch.eye(65, dtype=torch.float64)))}")

print("\nSTRESS TEST COMPLETE")
