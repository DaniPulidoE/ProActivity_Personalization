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
    # 420 runs across the 4 stages; ~60 epochs average once patience=20 fires.
    for name, s in (("CPU", cpu), ("CUDA", gpu)):
        per_run = s * 77 * 60 + 24        # +24 s fixed data cost per subprocess
        print(f"  full pipeline (420 runs) on {name:4}: {420*per_run/3600:6.1f} h")
else:
    print("CUDA NOT AVAILABLE — see setup_cuda_torch.py; launch with `uv run --no-sync`")
