r"""Pre-encode the labelled segments once so training runs do not re-parse the JSONL.

WHY
---
``train_XLSTM`` parses ``data/labeled_data.jsonl`` (971 MB, 508,282 rows), builds
a DataFrame, and resamples every segment onto the fixed grid before a single
optimizer step runs — **24 s of single-threaded CPU, measured**. The population
pipeline launches 420 such processes, so that is ~2.8 h of repeatedly turning the
same bytes into the same ~1,470 arrays.

The encoding depends only on ``(source file, label map, window_seconds,
resample_hz)``. It does NOT depend on the train/val split, the seed, dropout, lr
or the loss — so ONE cache serves every run of a sweep stage.

The second effect matters more than the first: per-process peak RSS drops from
~0.6 GB to a few tens of MB, which is what makes ``--jobs`` in
``sweep_population_hparams`` affordable. Twelve concurrent trainers each parsing
971 MB would thrash; twelve each mmap-ing a 19 MB npz will not.

STALENESS
---------
The cache stores a fingerprint of the source file (name/size/mtime), the label
map, the window, the grid, ``D_IN`` and a hash of ``FEATURE_NAMES``.
``load_segment_cache`` refuses anything that does not match, and the trainer
treats a mismatch as a hard error rather than falling back to a re-parse. This is
deliberate: ``D_IN`` went 35 -> 33 on 2026-08-14 and invalidated every checkpoint
by failing at the first matmul; a name-keyed cache would reproduce that failure
silently and as *training data*, which is far worse.

Rebuild after: regenerating ``labeled_data.jsonl``, changing the feature schema,
or changing ``window_seconds`` / ``resample_hz``.

Usage::

    # one cache per window the pipeline sweeps
    python -m scripts.build_segment_cache --in data/labeled_data.jsonl \
        --outdir data/cache --windows 5,10,20
"""
from __future__ import annotations

import argparse
import pathlib
import time

import pandas as pd

from ProVoice.models.train_XLSTM import (
    LEVELS, SPLIT_VARIABLE, SeqDataset, cache_meta, cache_name, iter_jsonl,
    normalize_row, prepare_frame, save_segment_cache,
)
from ProVoice.models.xlstm_model import DEFAULT_RESAMPLE_HZ


def build_one(df: pd.DataFrame, window_seconds: float, resample_hz: float,
              context_length: int, out: pathlib.Path, meta: dict) -> None:
    t0 = time.time()
    # SeqDataset is the ONE encoder. Building through it rather than reimplementing
    # the loop is what guarantees a cached run and a direct run see identical
    # arrays — including the skip rule for segments with missing Level_* labels.
    ds = SeqDataset(df, context_length=context_length, split="cache",
                    window_seconds=window_seconds, resample_hz=resample_hz)
    # Recover the per-segment ids in the SAME order SeqDataset emitted them
    # (groupby('segment_id') sorts, and it drops unlabelled segments), so the
    # arrays line up with `groups` element for element.
    keep, pid_of = [], {}
    for gid, g in df.groupby('segment_id'):
        if not all(k in g.columns for k in LEVELS):
            continue                     # mirrors SeqDataset's own guard
        lv = g[LEVELS].iloc[0].astype(float).values
        if pd.isna(lv).any() or lv.sum() <= 0:
            continue
        keep.append(str(gid))
        pid_of[str(gid)] = str(g[SPLIT_VARIABLE].iloc[0])
    assert len(keep) == len(ds.groups), (
        f"segment id/group mismatch: {len(keep)} ids vs {len(ds.groups)} encoded groups")

    save_segment_cache(
        out, ds.groups, keep, [pid_of[s] for s in keep],
        pid_order=[str(p) for p in df[SPLIT_VARIABLE].drop_duplicates()],
        seg_order=[str(g) for g in df['segment_id'].drop_duplicates()],
        meta=meta)
    mb = out.stat().st_size / 1e6
    print(f"[OK] {out.name}: {len(ds.groups)} segments, {mb:.1f} MB, {time.time()-t0:.1f} s")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in", dest="in_jsonl", default="data/labeled_data.jsonl")
    ap.add_argument("--outdir", default="data/cache")
    ap.add_argument("--label-map", dest="label_map", default=None)
    ap.add_argument("--windows", default="5,10,20",
                    help="Comma-separated window_seconds to build a cache for. The "
                         "population pipeline sweeps 10, 5 and 20.")
    ap.add_argument("--resample-hz", dest="resample_hz", type=float,
                    default=DEFAULT_RESAMPLE_HZ)
    args = ap.parse_args()

    windows = [float(w) for w in args.windows.split(",") if w.strip()]
    outdir = pathlib.Path(args.outdir)

    # Parse ONCE for every window: the truncation to the last k seconds happens
    # per-segment inside SeqDataset, so the 24 s parse is shared across all three.
    t0 = time.time()
    rows = [normalize_row(r) for r in iter_jsonl(pathlib.Path(args.in_jsonl))]
    if not rows:
        raise SystemExit(f"{args.in_jsonl} is empty or has no valid rows")
    df = prepare_frame(pd.DataFrame(rows), args.label_map)
    del rows
    print(f"[parse] {len(df)} rows in {time.time()-t0:.1f} s -> building {len(windows)} cache(s)")

    for w in windows:
        ctx = int(round(w * args.resample_hz)) if args.resample_hz else None
        if ctx is None:
            raise SystemExit("--resample-hz 0 is not supported by the cache: without a "
                             "fixed grid, context_length is not derivable from the window.")
        build_one(df, w, args.resample_hz, ctx,
                  outdir / cache_name(w, args.resample_hz),
                  cache_meta(args.in_jsonl, args.label_map, w, args.resample_hz))

    print(f"\nUse with:  --cache {outdir / cache_name(windows[0], args.resample_hz)}")


if __name__ == "__main__":
    main()
