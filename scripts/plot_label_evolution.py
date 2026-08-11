"""Per-driver LoA label evolution: one panel per session, one line per function.

Answers "did this driver's preference for each function move over the session?" --
the question raised by the first experimental run, where two of five functions
stepped from LoA 4 to LoA 0 partway through and never recovered, while the other
three stayed flat. A pooled mean hides that completely: it reads as a uniform
downward drift when in fact most functions were stable.

    python scripts/plot_label_evolution.py
    python scripts/plot_label_evolution.py --labels data/user_loa_labels.csv --out-dir results

Writes, per participant:
    results/label_evolution_p<pid>.png
    results/label_evolution_p<pid>.csv   -- the same numbers as a table

The CSV is not an extra: three of the five line colours sit below 3:1 contrast on
a light surface, and the palette rule for that case requires a table view (or
direct labels) as relief. It doubles as the data behind the figure.

A window where the driver marked SEVERAL acceptable levels is plotted at the mean
of the marked set, as a thin vertical bar spanning min..max with the marker at the
mean -- so an averaged multi-mark is never mistaken for a confident single one.
"""
from __future__ import annotations

import argparse
import collections
import csv
import datetime as dt
import pathlib
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# --------------------------------------------------------------------------- #
# Palette. Slots 1-5 of the validated categorical theme, IN ORDER (never cycled).
# Verified with the skill's validator on the adjacent pairlist -- the documented
# one for line charts -- in both modes:
#   light: CVD dE 9.1, normal-vision dE 19.6  -> all checks pass
#   dark : CVD dE 8.4, normal-vision dE 19.3  -> all checks pass
# Light mode WARNs on contrast for aqua/yellow/magenta, hence the companion CSV.
# Marker shape and dash pattern carry identity too, so nothing rests on hue alone
# (one pair sits at tritan dE 5.8, which is only legal with that secondary
# encoding). Adding a 6th+ function means re-validating -- do not extend by
# guessing a hue.
# --------------------------------------------------------------------------- #
SERIES = [
    ("#2a78d6", "o", (0, ())),                 # blue    -- solid
    ("#eb6834", "s", (0, (6, 2))),             # orange  -- dashed
    ("#1baf7a", "^", (0, (1, 1.6))),           # aqua    -- dotted
    ("#eda100", "D", (0, (7, 2, 1.5, 2))),     # yellow  -- dash-dot
    ("#e87ba4", "v", (0, (3, 1.6))),           # magenta -- short dash
]
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
INK_MUTED = "#8f8e88"
GRID = "#e6e5e1"

# LoA -> action, from decision_engine._LOA_POLICY. Shown on the y axis because
# "4" means nothing to a reader and "auto" means everything.
LOA_ACTIONS = {0: "none", 1: "suggest", 2: "ask approval", 3: "auto w/ veto", 4: "auto"}


def parse_iso(ts: str):
    try:
        return dt.datetime.fromisoformat(str(ts).strip())
    except Exception:
        return None


def load(path: pathlib.Path):
    """rows -> [(pid, session, window_idx, function, marks[])], sorted."""
    out = []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f):
            sid = (r.get("session_id") or "").strip()
            fn = (r.get("functionname") or "").strip()
            raw = (r.get("user_selected_loa") or "").strip()
            if not sid or not fn or not raw:
                continue
            try:
                w = int(float(r.get("window_idx") or ""))
            except ValueError:
                continue
            marks = []
            for part in raw.replace(",", ";").split(";"):
                part = part.strip()
                if not part:
                    continue
                try:
                    v = int(float(part))
                except ValueError:
                    print(f"[warn] un-parseable LoA {part!r} in {raw!r} -- ignored")
                    continue
                if not 0 <= v <= 4:
                    print(f"[warn] LoA {v} outside 0..4 in {raw!r} -- clamped")
                marks.append(max(0, min(4, v)))
            if not marks:
                continue
            out.append({
                "pid": (r.get("participantid") or "unknown").strip() or "unknown",
                "session": sid,
                "window": w,
                "function": fn,
                "marks": sorted(set(marks)),
                "start": parse_iso(r.get("window_start_timestamp") or ""),
            })
    return out


def panel(ax, rows, functions, colours, max_window):
    """One session: window_idx on x, marked LoA on y, one line per function."""
    for fn in functions:
        pts = sorted((r["window"], r["marks"]) for r in rows if r["function"] == fn)
        if not pts:
            continue
        colour, marker, dashes = colours[fn]
        xs = [w for w, _ in pts]
        ys = [sum(m) / len(m) for _, m in pts]
        # multi-mark range bars first, so the line and marker sit on top
        for w, m in pts:
            if len(m) > 1:
                ax.plot([w, w], [min(m), max(m)], color=colour, lw=4,
                        alpha=0.28, solid_capstyle="round", zorder=2)
        ax.plot(xs, ys, color=colour, lw=2, linestyle=dashes, marker=marker,
                markersize=7, markerfacecolor=colour, markeredgecolor=SURFACE,
                markeredgewidth=1.4, zorder=3, clip_on=False)

    ax.set_ylim(-0.35, 4.35)
    ax.set_yticks(range(5))
    ax.set_yticklabels([f"{k} · {LOA_ACTIONS[k]}" for k in range(5)],
                       fontsize=9, color=INK_2)
    ax.set_xlim(0.4, max_window + 0.6)
    step = 1 if max_window <= 20 else (2 if max_window <= 40 else 5)
    ax.set_xticks([w for w in range(1, max_window + 1) if w % step == 0 or w == 1])
    ax.tick_params(axis="x", labelsize=9, colors=INK_2, length=0)
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="y", color=GRID, lw=1, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.set_facecolor(SURFACE)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--labels", default="data/user_loa_labels.csv")
    ap.add_argument("--out-dir", dest="out_dir", default="results")
    args = ap.parse_args()

    path = pathlib.Path(args.labels)
    if not path.exists():
        raise SystemExit(f"labels not found: {path}")
    rows = load(path)
    if not rows:
        raise SystemExit("no usable label rows")

    # Colour is assigned from the functions present in the WHOLE file, so the same
    # function keeps the same colour across participants even when one of them
    # never saw it. Fixed order, never cycled -- past 8 the palette has no more
    # validated slots and inventing one is the documented failure mode.
    functions = sorted({r["function"] for r in rows})
    if len(functions) > len(SERIES):
        raise SystemExit(
            f"{len(functions)} distinct functions but only {len(SERIES)} validated "
            f"palette slots. Extend SERIES with the next slot(s) of the categorical "
            f"theme and RE-RUN the palette validator before trusting the figure.\n"
            f"  functions: {functions}")
    colours = {fn: SERIES[i] for i, fn in enumerate(functions)}

    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    by_pid = collections.defaultdict(list)
    for r in rows:
        by_pid[r["pid"]].append(r)

    for pid, prs in sorted(by_pid.items()):
        # Sessions in CHRONOLOGICAL order (earliest window), not by id string --
        # a session id is a uuid and sorts arbitrarily.
        starts = {}
        for r in prs:
            s = r["session"]
            if r["start"] is not None and (s not in starts or r["start"] < starts[s]):
                starts[s] = r["start"]
        sessions = sorted({r["session"] for r in prs},
                          key=lambda s: (starts.get(s) or dt.datetime.max, s))

        ncols = min(len(sessions), 2)
        nrows = (len(sessions) + ncols - 1) // ncols
        max_window = max(r["window"] for r in prs)
        # ~0.19 in per window keeps a panel near 3:2. Wider than that flattens the
        # vertical axis until a 4 -> 0 step reads as a gentle slope, which is the
        # one thing this figure exists to show.
        panel_w = max(4.5, min(0.19 * max_window, 7.5))
        fig, axes = plt.subplots(
            nrows, ncols, squeeze=False, sharey=True,
            figsize=(1.6 + panel_w * ncols, 4.1 * nrows + 1.7))
        fig.patch.set_facecolor(SURFACE)

        for k, sid in enumerate(sessions):
            ax = axes[k // ncols][k % ncols]
            srows = [r for r in prs if r["session"] == sid]
            panel(ax, srows, functions, colours, max_window)
            nw = len({r["window"] for r in srows})
            ax.set_title(f"session {sid[:8]}   ·   {len(srows)} labels over {nw} windows",
                         fontsize=10, color=INK, pad=10, loc="left")
            ax.set_xlabel("window number in session", fontsize=9, color=INK_2)
        for k in range(len(sessions), nrows * ncols):
            axes[k // ncols][k % ncols].axis("off")

        handles = [Line2D([], [], color=colours[fn][0], marker=colours[fn][1],
                          linestyle=colours[fn][2], lw=2, markersize=7,
                          markeredgecolor=SURFACE, markeredgewidth=1.4, label=fn)
                   for fn in functions]
        fig.legend(handles=handles, loc="lower center", ncol=min(len(functions), 3),
                   frameon=False, fontsize=9, labelcolor=INK_2,
                   bbox_to_anchor=(0.5, 0.0))
        fig.suptitle(f"Driver {pid} — LoA marked per function, over the session",
                     fontsize=12.5, color=INK, x=0.012, ha="left", y=0.985)
        fig.text(0.012, 0.925,
                 "a point is one label; a vertical bar spans a multi-mark, plotted at its mean",
                 fontsize=9, color=INK_MUTED, ha="left")
        fig.tight_layout(rect=(0, 0.09, 1, 0.90))

        png = out_dir / f"label_evolution_p{pid}.png"
        fig.savefig(png, dpi=160, facecolor=SURFACE)
        plt.close(fig)

        csv_path = out_dir / f"label_evolution_p{pid}.csv"
        with csv_path.open("w", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            w.writerow(["participantid", "session_id", "session_order", "window_idx",
                        "functionname", "marked_loa", "n_marks", "mean_loa"])
            for order, sid in enumerate(sessions, 1):
                for r in sorted((x for x in prs if x["session"] == sid),
                                key=lambda x: (x["window"], x["function"])):
                    w.writerow([pid, sid, order, r["window"], r["function"],
                                ";".join(map(str, r["marks"])), len(r["marks"]),
                                round(sum(r["marks"]) / len(r["marks"]), 3)])
        print(f"[p{pid}] {len(sessions)} session(s), {len(prs)} labels -> {png}")
        print(f"        table -> {csv_path}")


if __name__ == "__main__":
    main()
