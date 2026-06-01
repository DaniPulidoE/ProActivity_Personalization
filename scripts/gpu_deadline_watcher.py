"""Deadline watcher: free the GPU if training hasn't started in time.

The series orchestrator waits politely for the GPU to go idle. The user
authorised a hard deadline: if training has NOT started within
``--deadline`` seconds, kill the other GPU jobs so the orchestrator can
proceed.

Targeting: every process this project launches runs from the project
``.venv`` or the uv-managed CPython. The user's competing GPU jobs run
from the system ``Python312`` interpreter. So we target python processes
whose executable path contains ``Python312`` — a clean discriminator
that can never hit our own training.

If training has already started by the deadline, nothing is killed.

Run (background)::

    uv run --no-sync python scripts/gpu_deadline_watcher.py \\
        --deadline 3600 --log runs/yolo26_series.log
"""
from __future__ import annotations

import argparse
import time
from datetime import datetime
from pathlib import Path


def log(msg: str) -> None:
    print(f"[watcher {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


def training_started(log_path: Path, marker: str) -> bool:
    try:
        return marker in log_path.read_text(encoding="utf-8", errors="ignore")
    except FileNotFoundError:
        return False


def gpu_job_processes(interpreter_tag: str):
    import psutil
    targets = []
    for p in psutil.process_iter(["pid", "name", "exe", "cmdline"]):
        try:
            name = (p.info.get("name") or "").lower()
            exe = p.info.get("exe") or ""
            if name == "python.exe" and interpreter_tag.lower() in exe.lower():
                targets.append(p)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return targets


def kill_gpu_jobs(interpreter_tag: str) -> int:
    import psutil
    targets = gpu_job_processes(interpreter_tag)
    if not targets:
        log(f"no '{interpreter_tag}' processes found to kill")
        return 0
    for p in targets:
        try:
            cl = " ".join(p.cmdline())
        except Exception:
            cl = "<cmdline unavailable>"
        log(f"  terminating PID {p.pid}: {cl[:110]}")
        try:
            p.terminate()
        except (psutil.NoSuchProcess, psutil.AccessDenied) as exc:
            log(f"    terminate failed: {exc}")
    gone, alive = psutil.wait_procs(targets, timeout=10)
    for p in alive:
        try:
            p.kill()
            log(f"  force-killed PID {p.pid}")
        except Exception as exc:
            log(f"  force-kill failed PID {p.pid}: {exc}")
    return len(targets)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--deadline", type=int, default=3600,
                    help="Seconds to wait before killing GPU jobs (default 3600 = 1h).")
    ap.add_argument("--log", default="runs/yolo26_series.log")
    ap.add_argument("--marker", default="=== training",
                    help="Substring in the orchestrator log meaning training has begun.")
    ap.add_argument("--interpreter-tag", default="Python312",
                    help="Kill python.exe whose exe path contains this (the user's GPU env).")
    ap.add_argument("--poll", type=int, default=60)
    args = ap.parse_args()

    log_path = Path(args.log).resolve()
    start = time.time()
    log(f"armed. deadline={args.deadline}s, log={log_path}, "
        f"marker='{args.marker}', target='{args.interpreter_tag}'")

    while True:
        if training_started(log_path, args.marker):
            log("training started on its own — no kill needed. Exiting.")
            return
        elapsed = time.time() - start
        if elapsed >= args.deadline:
            log(f"DEADLINE reached ({int(elapsed)}s) and training has NOT started.")
            log(f"killing competing GPU jobs ('{args.interpreter_tag}' interpreter)...")
            n = kill_gpu_jobs(args.interpreter_tag)
            log(f"killed {n} process(es). Orchestrator should auto-start within ~90s.")
            return
        remaining = int(args.deadline - elapsed)
        log(f"waiting... {remaining}s until deadline (training not yet started)")
        time.sleep(args.poll)


if __name__ == "__main__":
    main()
