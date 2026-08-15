"""CPU vs CUDA step time for the ACTUAL population model. Run on the GPU box.

    uv run --no-sync python bench_gpu.py

No data needed — it feeds random tensors of the exact training shape, so it
isolates the step cost from data loading.
"""
import time, torch
from ProVoice.models.xlstm_model import XLSTMSequenceClassifier, D_IN, soft_corn_loss

CTX, BATCH, WARMUP, ITERS = 100, 16, 10, 50


def bench(device: str) -> float:
    torch.manual_seed(0)
    m = XLSTMSequenceClassifier(d_in=D_IN, n_classes=5, embedding_dim=64,
                                num_blocks=2, num_heads=4, context_length=CTX,
                                head_type="corn", dropout=0.15).to(device).train()
    opt = torch.optim.AdamW(m.parameters(), lr=2e-3)
    xb = torch.randn(BATCH, CTX, D_IN, device=device)
    lb = torch.full((BATCH,), CTX, device=device, dtype=torch.long)
    vb = torch.zeros(BATCH, 5, device=device); vb[:, 2] = 1.0

    for i in range(WARMUP + ITERS):
        if i == WARMUP:
            if device == "cuda": torch.cuda.synchronize()
            t0 = time.time()
        loss = soft_corn_loss(m(xb, lengths=lb), vb)
        opt.zero_grad(); loss.backward(); opt.step()
    if device == "cuda": torch.cuda.synchronize()
    return (time.time() - t0) / ITERS


print(f"torch {torch.__version__}  cuda_available={torch.cuda.is_available()}")
cpu = bench("cpu")
print(f"CPU  : {cpu*1000:7.1f} ms/step")
if torch.cuda.is_available():
    print(f"GPU  : {torch.cuda.get_device_name(0)}")
    gpu = bench("cuda")
    print(f"CUDA : {gpu*1000:7.1f} ms/step   -> {cpu/gpu:.1f}x vs CPU")
    # ~77 steps/epoch, ~60 epochs average once patience=20 fires. FIXED is the
    # per-subprocess cost: ~4.7 s measured WITH a segment cache (python + torch
    # import + cache load). Without --cache it is ~29 s, because every run
    # re-parses the 971 MB JSONL -- which is why the pre-cache estimate was
    # dominated by startup and barely moved when the step time did.
    FIXED_S, STEPS, EPOCHS = 4.7, 77, 60
    print(f"\n  projection @ {STEPS} steps x {EPOCHS} epochs + {FIXED_S:.1f} s fixed per run:")
    print(f"  {'':<10}{'420 runs':>12}{'720 runs':>12}   (--window-grid inherit / full)")
    for name, s in (("CPU", cpu), ("CUDA", gpu)):
        per_run = s * STEPS * EPOCHS + FIXED_S
        print(f"  {name:<10}{420*per_run/3600:>10.1f} h{720*per_run/3600:>10.1f} h"
              f"   ({per_run:.0f} s/run)")
    print("  Divide by the --jobs speedup you measure (2.31x at 4x4 on CPU; "
          "re-measure on GPU, where concurrent CUDA contexts contend differently).")
else:
    print("CUDA NOT AVAILABLE — see setup_cuda_torch.py; launch with `uv run --no-sync`")
