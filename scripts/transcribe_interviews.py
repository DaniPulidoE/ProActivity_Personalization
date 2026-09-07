"""Transcribe interviews/Interview-<pid>.mp4 with faster-whisper.

Downloads a medium-sized Whisper model (default: "medium", ~1.5 GB, auto-cached
by Hugging Face Hub on first run) and writes one Markdown transcript per
interview to interviews/transcripts/interview_transcript_p<pid>.md.

faster-whisper is not a project dependency (it would drag ctranslate2/onnxruntime
into the main lockfile and risks uv reverting the imperatively-installed GPU
torch on the next `uv sync`, see scripts/setup_cuda_torch.py). Run this script
as an ephemeral overlay instead:

    uv run --with faster-whisper python scripts/transcribe_interviews.py

Usage:
    uv run --with faster-whisper python scripts/transcribe_interviews.py [options]

Options:
    --model-size {tiny,base,small,medium,large-v2,large-v3}  default: medium
    --device {auto,cpu,cuda}                                  default: auto
    --compute-type TEXT   ctranslate2 compute type; default depends on device
    --language CODE       force a language (e.g. "en"); default: auto-detect
    --overwrite            re-transcribe even if the output file exists
    --interviews-dir PATH  default: interviews
    --output-dir PATH      default: interviews/transcripts
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

PID_PATTERN = re.compile(r"Interview-(\d+)", re.IGNORECASE)


def format_timestamp(seconds: float) -> str:
    total_ms = round(seconds * 1000)
    hours, rem_ms = divmod(total_ms, 3_600_000)
    minutes, rem_ms = divmod(rem_ms, 60_000)
    secs, ms = divmod(rem_ms, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}.{ms:03d}"


def pick_device_and_compute_type(device_arg: str, compute_type_arg: str | None) -> tuple[str, str]:
    if device_arg == "auto":
        try:
            import torch

            device = "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            device = "cpu"
    else:
        device = device_arg

    if compute_type_arg:
        return device, compute_type_arg
    return device, ("float16" if device == "cuda" else "int8")


def find_interviews(interviews_dir: Path) -> list[tuple[str, Path]]:
    found = []
    for path in sorted(interviews_dir.glob("Interview-*.mp4")):
        match = PID_PATTERN.search(path.stem)
        if not match:
            print(f"[skip] {path.name}: filename does not match 'Interview-<pid>.mp4'")
            continue
        found.append((match.group(1), path))
    return found


def transcribe_file(model, media_path: Path, language: str | None) -> str:
    segments, info = model.transcribe(
        str(media_path),
        beam_size=5,
        language=language,
        vad_filter=True,
    )

    lines = [
        f"# Interview Transcript — Participant {media_path.stem.split('-')[-1]}",
        "",
        f"- Source: `{media_path.name}`",
        f"- Model: faster-whisper",
        f"- Detected language: {info.language} (confidence {info.language_probability:.2f})",
        f"- Duration: {format_timestamp(info.duration)}",
        "",
        "## Transcript",
        "",
    ]

    n_segments = 0
    for segment in segments:
        text = segment.text.strip()
        lines.append(f"[{format_timestamp(segment.start)} → {format_timestamp(segment.end)}] {text}")
        n_segments += 1
        if n_segments % 10 == 0:
            print(f"    ... {format_timestamp(segment.end)} transcribed")

    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model-size", default="large-v3")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--compute-type", default=None)
    parser.add_argument("--language", default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--interviews-dir", type=Path, default=Path("interviews"))
    parser.add_argument("--output-dir", type=Path, default=Path("interviews") / "transcripts")
    args = parser.parse_args()

    try:
        from faster_whisper import WhisperModel
    except ImportError:
        print(
            "faster-whisper is not installed. Run this script via:\n"
            "  uv run --with faster-whisper python scripts/transcribe_interviews.py",
            file=sys.stderr,
        )
        return 1

    if not args.interviews_dir.is_dir():
        print(f"Interviews directory not found: {args.interviews_dir}", file=sys.stderr)
        return 1

    interviews = find_interviews(args.interviews_dir)
    if not interviews:
        print(f"No 'Interview-*.mp4' files found in {args.interviews_dir}")
        return 0

    args.output_dir.mkdir(parents=True, exist_ok=True)

    device, compute_type = pick_device_and_compute_type(args.device, args.compute_type)
    print(f"Loading faster-whisper model '{args.model_size}' (device={device}, compute_type={compute_type})...")
    # use_auth_token=False: these are public models; without this, huggingface_hub
    # implicitly picks up any locally cached token, and a stale/invalid one turns
    # into a confusing 401 on a public repo.
    model = WhisperModel(args.model_size, device=device, compute_type=compute_type, use_auth_token=False)

    for pid, media_path in interviews:
        out_path = args.output_dir / f"interview_transcript_p{pid}.md"
        if out_path.exists() and not args.overwrite:
            print(f"[skip] {out_path.name} already exists (use --overwrite to redo)")
            continue

        print(f"[transcribe] {media_path.name} -> {out_path.name}")
        markdown = transcribe_file(model, media_path, args.language)
        out_path.write_text(markdown, encoding="utf-8")
        print(f"    wrote {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
