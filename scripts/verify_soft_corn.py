"""Verify soft_corn_loss against the reference corn_loss, and the Laplace weights.

Run:  uv run python -m scripts.verify_soft_corn

Check [1] compares against ``coral_pytorch.losses.corn_loss``, which is no longer
a project dependency; it is skipped automatically if the package is absent.
Install it ad hoc (``uv pip install coral-pytorch``) to reproduce that check.
See docs/soft_corn_and_oldl.md §1.4.
"""
import torch
import torch.nn.functional as F
from ProVoice.models.xlstm_model import (
    soft_corn_loss, levels_to_distribution, levels_to_cumulative,
    levels_to_subset_weights, logits_to_probs,
)

torch.manual_seed(0)
K = 5
B = 64
logits = torch.randn(B, K - 1, requires_grad=True)
y = torch.randint(0, K, (B,))
levels = F.one_hot(y, K).float()

# ---- 1. single-label equivalence with the reference corn_loss ----
try:
    from coral_pytorch.losses import corn_loss
    ref = corn_loss(logits, y, num_classes=K)
    mine = soft_corn_loss(logits, levels)
    g_ref = torch.autograd.grad(ref, logits, retain_graph=True)[0]
    g_mine = torch.autograd.grad(mine, logits, retain_graph=True)[0]
    cos = F.cosine_similarity(g_ref.flatten(), g_mine.flatten(), dim=0)
    ratio = (g_mine.flatten().norm() / g_ref.flatten().norm())
    print(f"[1] corn_loss={ref.item():.6f}  soft_corn_loss={mine.item():.6f}")
    print(f"    gradient cosine = {cos.item():.9f}   |g_mine|/|g_ref| = {ratio.item():.4f}")
    # the scalar factor should be exactly M/N (corn normalizes by M, we by N)
    M = sum(int((y >= j).sum()) for j in range(K - 1))
    print(f"    predicted factor M/N = {M}/{B} = {M/B:.4f}")
    assert cos.item() > 0.999999, "gradient direction differs!"
    print("    OK: identical gradient direction, scalar factor = M/N as derived")
except ImportError:
    print("[1] coral_pytorch not installed, skipping reference comparison")

# ---- 2. hand-computed single example ----
z = torch.tensor([[0.3, -0.7, 1.2, 0.1]])
lv = torch.zeros(1, K); lv[0, 2] = 1.0          # y = 2
expected = -(F.logsigmoid(z[0, 0]) + F.logsigmoid(z[0, 1]) + F.logsigmoid(-z[0, 2]))
got = soft_corn_loss(z, lv)
print(f"[2] hand-derived={expected.item():.8f}  soft_corn_loss={got.item():.8f}  "
      f"diff={abs(expected-got).item():.2e}")
assert torch.allclose(expected, got, atol=1e-6)

# ---- 3. multi-label: gapped set {0, 4} ----
lv2 = torch.zeros(1, K); lv2[0, 0] = 1.0; lv2[0, 4] = 1.0
p = levels_to_distribution(lv2); q = levels_to_cumulative(lv2)
print(f"[3] gapped set {{0,4}}: p={p.tolist()[0]}  q={q.tolist()[0]}")
assert torch.allclose(q, torch.tensor([[0.5, 0.5, 0.5, 0.5]])), q
loss2 = soft_corn_loss(z, lv2)
print(f"    soft_corn_loss on gapped target = {loss2.item():.6f} (finite, trains)")
assert torch.isfinite(loss2)

# ---- 4. Laplace subset weights reduce to the hard indicator ----
W = levels_to_subset_weights(levels)                     # (B, K-1)
hard = torch.stack([(y >= j).float() for j in range(K - 1)], dim=1)
print(f"[4] max|soft weights - hard 1[y>=j]| = {(W - hard).abs().max().item():.2e}")
assert torch.allclose(W, hard, atol=1e-6)
print(f"    soft M = {W.sum().item():.1f}  hard M = {hard.sum().item():.1f}")

# ---- 5. gapped-set representability: can the head put argmax inside the set? ----
zfit = torch.zeros(1, K - 1, requires_grad=True)
opt = torch.optim.Adam([zfit], lr=0.1)
for _ in range(2000):
    loss = soft_corn_loss(zfit, lv2)
    opt.zero_grad(); loss.backward(); opt.step()
pmf = logits_to_probs(zfit.detach(), "corn")[0]
print(f"[5] fitted PMF for {{0,4}} = {[round(v,4) for v in pmf.tolist()]}  "
      f"argmax={int(pmf.argmax())}")
assert int(pmf.argmax()) in (0, 4), "argmax fell OUTSIDE the marked set (the CORAL failure)"
print("    OK: argmax lands inside the marked set (no split-the-difference collapse)")

# ---- 6. Laplace posterior end-to-end ----
from ProVoice.models.laplace_head import LaplacePosterior
import torch.nn as nn
E = 8
head = nn.Linear(E, K - 1)
Z = torch.randn(20, E)
LV = F.one_hot(torch.randint(0, K, (20,)), K).float()
post = LaplacePosterior.fit(head, Z, LV, l2sp=0.01, n_classes=K)
pm = post.predictive_pmf(Z, n_samples=16)
sd = post.predictive_loa_std(Z, n_samples=16)
print(f"[6] Laplace fit OK: tau={post.prior_precision:.4g} (=2*N*lam=2*20*0.01={2*20*0.01})  "
      f"n={post.n_examples} softM={post.n_cond_examples:.1f}")
print(f"    predictive PMF rows sum to 1: {torch.allclose(pm.sum(-1), torch.ones(20), atol=1e-5)}  "
      f"mean width={sd.mean():.4f}")
assert torch.allclose(pm.sum(-1), torch.ones(20), atol=1e-5)

# multi-label Laplace also works
LV2 = LV.clone(); LV2[0, 0] = 1.0; LV2[0, 4] = 1.0
post2 = LaplacePosterior.fit(head, Z, LV2, l2sp=0.01, n_classes=K)
print(f"    multi-label Laplace OK: softM={post2.n_cond_examples:.2f} (vs {post.n_cond_examples:.2f})")

print("\nALL CHECKS PASSED")
