#!/usr/bin/env python

"""Post-block questionnaire: Van der Laan acceptance + INTUI Magical Experience.

Run once after each driving block, before the next one is set up::

    uv run python -m src.drive.questionnaire \
        --participantid 001 --condition 1 --block-idx 2

Writes ONE row per block to ``data/study_1_questionnaire.csv`` and, the first
time it runs, a codebook beside it (see WHY A CODEBOOK below).

MOUSE AND KEYBOARD ONLY -- THE WHEEL IS DELIBERATELY DEAF HERE
==============================================================
``pygame.init()`` initialises the joystick module too, and a G25 sitting on a
desk streams JOYAXISMOTION continuously from wheel and pedal jitter. That would
flood the event queue behind a form the participant is reading, and a stray
paddle press could land on a radio button. So every joystick event type is
blocked outright at start-up rather than merely ignored in the loop: blocked
events are dropped by SDL before they reach the queue, which is the difference
between "we don't act on it" and "it isn't there".

The wheel therefore stays plugged in between blocks with no effect on this form.

SEVEN POINTS, NOT FIVE -- A DEVIATION WORTH STATING
===================================================
Van der Laan (1997) is published as a **five**-point semantic differential and
its norms, means and cut-offs are all on that scale. This runs it on **seven**
points, matching INTUI, so a participant meets one response format all evening
instead of two. That is a defensible choice and a real one: subscale scores are
NOT directly comparable with published Van der Laan values, and the write-up has
to say so rather than quoting norms. ``--scale-points 5`` restores the original.

POLARITY IS THE THING THAT GETS SILENTLY WRONG
==============================================
Both instruments ALTERNATE the side the positive pole sits on -- that is what
makes them resistant to a participant who just clicks down one column. Van der
Laan items 3 (Bad-Good), 6 (Irritating-Likeable) and 8 (Undesirable-Desirable)
are positive on the RIGHT; the other six are positive on the LEFT. INTUI items 2
and 3 are positive on the right, 1 and 4 on the left.

Position 1 is the LEFTMOST box, so the items needing a flip are the ones with
the positive pole on the LEFT -- six of nine in Van der Laan, two of four in
INTUI. Getting that backwards inverts every subscale and leaves a CSV that looks
completely normal, which is why ``score()`` carries a property test rather than
an argument.

So a raw click position means opposite things on adjacent rows, and averaging
raw values across a subscale produces a number that looks fine and is garbage.
Every item is therefore logged TWICE:

  ``*_raw``     the position clicked, 1..N left to right, exactly as displayed
  ``*_scored``  polarity-corrected, so higher ALWAYS means more positive

with the subscale means computed from ``_scored``. The raw column is kept
because it is the only record of what the participant actually saw, and because
a scoring bug is recoverable from it while the reverse is not.

WHY A CODEBOOK FILE
===================
``vdl3_raw = 2`` is meaningless a month later without knowing that item 3 was
"Bad vs Good" and reverse-coded. The codebook writes that mapping next to the
data on first run, so the CSV can be read by someone who does not have this
source in front of them -- including the version of you writing chapter 5.

SUBSCALES
=========
Van der Laan splits into two, and they are reported separately (never pooled):

  Usefulness  items 1, 3, 5, 7, 9   (useful / good / effective / assisting /
                                     raising alertness)
  Satisfying  items 2, 4, 6, 8      (pleasant / nice / likeable / desirable)

INTUI's four items here are one subscale, Magical Experience, from the full
INTUI (Ullrich & Diefenbach 2010) -- the other components (Gut Feeling, Verbal-
ization, Effortlessness) are not administered, so report it as a subscale and
not as "INTUI".

Both are also reported CENTRED (score minus the scale midpoint), because Van der
Laan is conventionally read on a symmetric scale where 0 is indifference and the
sign carries the meaning.
"""

import argparse
import csv
import datetime
import os
import sys
import time

import pygame


# --- The instruments ---------------------------------------------------------
#
# THE FLAG IS `positive_right`, not "reverse". It records WHICH SIDE the positive
# pole sits on, and the scoring follows from that -- an item whose positive pole
# is on the right already increases with positivity and needs NO flip, while one
# with the positive pole on the LEFT is the one that has to be reversed.
#
# The earlier name was `reverse` and it was read the other way round, which
# inverted every subscale while leaving the CSV looking entirely plausible: an
# enthusiastic participant scored 1.0 and a hostile one 7.0. Caught only because
# an all-positive fixture must yield a constant 7. Keep that test.
#
# Transcribed from the printed forms; check any edit against them, because a
# wrong flag is invisible in the data.
#
# The left/right strings are what the participant reads and are reproduced
# verbatim from the published instruments -- do not "improve" the wording, that
# is what makes it the validated instrument rather than a questionnaire of ours.

VDL_ITEMS = (
    # (left pole, right pole, positive_right, subscale)
    ("Useful",            "Useless",       False, "usefulness"),
    ("Pleasant",          "Unpleasant",    False, "satisfying"),
    ("Bad",               "Good",          True,  "usefulness"),
    ("Nice",              "Annoying",      False, "satisfying"),
    ("Effective",         "Superfluous",   False, "usefulness"),
    ("Irritating",        "Likeable",      True,  "satisfying"),
    ("Assisting",         "Worthless",     False, "usefulness"),
    ("Undesirable",       "Desirable",     True,  "satisfying"),
    ("Raising Alertness", "Sleep-inducing", False, "usefulness"),
)

# INTUI's Magical Experience component. Every item shares one stem, which is
# printed once above the block rather than repeated on each row.
INTUI_STEM = "Using the product..."
INTUI_ITEMS = (
    ("...was inspiring",      "...was insignificant",      False, "magical"),
    ("...was nothing special", "...was a magical experience", True, "magical"),
    ("...was trivial",        "...carried me away",        True,  "magical"),
    ("...was fascinating",    "...was dull",               False, "magical"),
)

# Van der Laan's own instruction, which every item completes: "I find such a
# system... Useful / Useless". Printed once above the block rather than repeated
# on each row, the same way INTUI's stem is.
#
# It says SYSTEM, not "assistant", because that is the published wording and the
# generic noun is what makes the instrument comparable across the studies that
# report it. --vdl-stem overrides it if the brief names the assistant instead;
# that is a wording change to record, not a formatting choice.
VDL_STEM = "I find such a system..."

PAGES = (
    ("vdl",   "Acceptance of the assistant", VDL_ITEMS, VDL_STEM),
    ("intui", "Your experience",             INTUI_ITEMS, INTUI_STEM),
)

DEFAULT_CSV = os.path.join("data", "study_1_questionnaire.csv")


def build_columns(scale_points: int):
    """The CSV header. Fixed and explicit, in the style of call_events.csv."""
    cols = ["logged_ts", "session_id", "participantid", "block_idx",
            "k_condition", "scale_points", "duration_s"]
    for i in range(len(VDL_ITEMS)):
        cols.append("vdl%d_raw" % (i + 1))
    for i in range(len(VDL_ITEMS)):
        cols.append("vdl%d_scored" % (i + 1))
    cols += ["vdl_usefulness_mean", "vdl_satisfying_mean",
             "vdl_usefulness_centered", "vdl_satisfying_centered"]
    for i in range(len(INTUI_ITEMS)):
        cols.append("intui%d_raw" % (i + 1))
    for i in range(len(INTUI_ITEMS)):
        cols.append("intui%d_scored" % (i + 1))
    cols += ["intui_magical_mean", "intui_magical_centered", "notes"]
    return tuple(cols)


def score(raw: int, positive_right: bool, scale_points: int) -> int:
    """Raw position -> polarity-corrected score. Higher is ALWAYS more positive.

    Position 1 is the leftmost box. So an item whose positive pole is on the
    right ("Bad | Good") already runs the right way and passes through; one whose
    positive pole is on the left ("Useful | Useless") has position 1 as the MOST
    positive answer and must be flipped.

    Verified by the property that motivates the whole scheme: a participant who
    picks the positive pole on every row scores a constant `scale_points`, no
    matter which side that pole was on.
    """
    return raw if positive_right else (scale_points + 1 - raw)


def write_codebook(path: str, scale_points: int) -> None:
    """One row per item: what it said, which way round, which subscale.

    Written once, beside the data. Without it the CSV is unreadable by anyone
    who does not have this module open -- and the reverse-coding is exactly the
    detail that gets reconstructed wrongly from memory.
    """
    if os.path.exists(path):
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["column", "instrument", "item_no", "left_pole", "right_pole",
                    "positive_pole", "reverse_coded", "subscale", "scale_points",
                    "scoring"])
        for tag, items in (("vdl", VDL_ITEMS), ("intui", INTUI_ITEMS)):
            for i, (lo, hi, pos_right, sub) in enumerate(items, start=1):
                w.writerow(["%s%d" % (tag, i), tag, i, lo, hi,
                            "right" if pos_right else "left",
                            int(not pos_right), sub, scale_points,
                            "scored = raw" if pos_right
                            else "scored = %d - raw" % (scale_points + 1)])
    print("[questionnaire] wrote codebook -> %s" % path)


# --- Appearance --------------------------------------------------------------
COL_BG = (28, 31, 36)
COL_CARD = (40, 44, 51)
COL_TEXT = (238, 240, 244)
COL_DIM = (150, 158, 170)
COL_ACCENT = (120, 200, 255)
COL_OK = (130, 220, 150)
COL_WARN = (255, 170, 90)
COL_FOCUS = (70, 78, 92)

DOT_R = 15
ROW_H = 62
MARGIN = 44


class Questionnaire(object):
    """Two pages of semantic differentials, mouse or keyboard."""

    def __init__(self, screen, scale_points: int, stems: dict = None):
        self.screen = screen
        self.n = scale_points
        # Pages are built ONCE, here, with the stems substituted -- rather than
        # patching the module-level PAGES, which would make the instrument
        # depend on import order and leak between two instances.
        stems = stems or {}
        self.pages = tuple(
            (k, t, items, stems.get(k, stem))
            for k, t, items, stem in PAGES)
        self.quit_requested = False
        w, h = screen.get_size()
        self.dim = (w, h)
        base = max(14, int(h * 0.024))
        f = pygame.font.get_default_font()
        self.f_title = pygame.font.Font(f, int(base * 1.55))
        self.f_item = pygame.font.Font(f, base)
        self.f_small = pygame.font.Font(f, int(base * 0.82))
        self.page = 0
        # answers[page][item] = raw position, or None
        self.answers = [[None] * len(p[2]) for p in self.pages]
        self.focus = 0
        self.started = time.time()
        self.done = False
        self._rows = []          # (rect, page, item, value) rebuilt each frame
        self._nav = {}

    # -- geometry ------------------------------------------------------------

    def _layout(self):
        """Row rects for the current page. Recomputed per frame; 13 rows is free."""
        w, h = self.dim
        _, _, items, stem = self.pages[self.page]
        top = MARGIN + self.f_title.get_height() + 18
        if stem:
            top += self.f_item.get_height() + 14
        # Dots occupy the middle third; labels take the outer thirds.
        dot_lo, dot_hi = int(w * 0.40), int(w * 0.66)
        step = (dot_hi - dot_lo) / float(self.n - 1)
        rows = []
        for i in range(len(items)):
            y = top + i * ROW_H
            centers = [(int(dot_lo + k * step), y + ROW_H // 2)
                       for k in range(self.n)]
            rows.append((y, centers))
        return top, rows, dot_lo, dot_hi

    # -- input ---------------------------------------------------------------

    def handle(self, ev):
        if ev.type == pygame.KEYDOWN:
            self._key(ev)
        elif ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1:
            self._click(ev.pos)

    def _key(self, ev):
        _, _, items, _ = self.pages[self.page]
        if ev.key == pygame.K_ESCAPE:
            self.quit_requested = True
        elif ev.key in (pygame.K_DOWN, pygame.K_TAB):
            self.focus = (self.focus + 1) % len(items)
        elif ev.key == pygame.K_UP:
            self.focus = (self.focus - 1) % len(items)
        elif ev.key in (pygame.K_LEFT, pygame.K_RIGHT):
            cur = self.answers[self.page][self.focus] or 0
            d = -1 if ev.key == pygame.K_LEFT else 1
            self.answers[self.page][self.focus] = max(1, min(self.n, cur + d))
        elif pygame.K_1 <= ev.key <= pygame.K_9:
            v = ev.key - pygame.K_0
            if v <= self.n:
                self.answers[self.page][self.focus] = v
                # Advance, so a participant using the number row can answer the
                # page without touching anything else. Stops at the last row
                # rather than wrapping, which would silently overwrite item 1.
                self.focus = min(self.focus + 1, len(items) - 1)
        elif ev.key in (pygame.K_RETURN, pygame.K_KP_ENTER):
            self._advance()
        elif ev.key == pygame.K_BACKSPACE:
            self.answers[self.page][self.focus] = None

    def _click(self, pos):
        for rect, page, item, val in self._rows:
            if rect.collidepoint(pos):
                self.answers[page][item] = val
                self.focus = item
                return
        for name, rect in self._nav.items():
            if rect.collidepoint(pos):
                if name == "next":
                    self._advance()
                elif name == "back" and self.page > 0:
                    self.page -= 1
                    self.focus = 0
                return

    def _advance(self):
        if self.missing():
            return
        if self.page < len(self.pages) - 1:
            self.page += 1
            self.focus = 0
        else:
            self.done = True

    def missing(self):
        return [i + 1 for i, v in enumerate(self.answers[self.page]) if v is None]

    # -- drawing -------------------------------------------------------------

    def render(self):
        s = self.screen
        w, h = self.dim
        s.fill(COL_BG)
        key, title, items, stem = self.pages[self.page]
        top, rows, dot_lo, dot_hi = self._layout()

        s.blit(self.f_title.render(title, True, COL_TEXT), (MARGIN, MARGIN))
        prog = "Page %d of %d" % (self.page + 1, len(self.pages))
        pr = self.f_small.render(prog, True, COL_DIM)
        s.blit(pr, (w - MARGIN - pr.get_width(), MARGIN + 8))
        if stem:
            s.blit(self.f_item.render(stem, True, COL_DIM),
                   (MARGIN, MARGIN + self.f_title.get_height() + 10))

        self._rows = []
        for i, (lo, hi, _pos_right, _sub) in enumerate(items):
            y, centers = rows[i]
            if i == self.focus:
                pygame.draw.rect(s, COL_FOCUS,
                                 pygame.Rect(MARGIN - 12, y, w - 2 * (MARGIN - 12),
                                             ROW_H - 8), border_radius=8)
            num = self.f_small.render("%d" % (i + 1), True, COL_DIM)
            s.blit(num, (MARGIN - 4, y + ROW_H // 2 - num.get_height() // 2))

            lt = self.f_item.render(lo, True, COL_TEXT)
            s.blit(lt, (dot_lo - 26 - lt.get_width(),
                        y + ROW_H // 2 - lt.get_height() // 2))
            rt = self.f_item.render(hi, True, COL_TEXT)
            s.blit(rt, (dot_hi + 26, y + ROW_H // 2 - rt.get_height() // 2))

            chosen = self.answers[self.page][i]
            for k, (cx, cy) in enumerate(centers):
                val = k + 1
                rect = pygame.Rect(cx - DOT_R - 4, cy - DOT_R - 4,
                                   2 * (DOT_R + 4), 2 * (DOT_R + 4))
                self._rows.append((rect, self.page, i, val))
                on = chosen == val
                pygame.draw.circle(s, COL_ACCENT if on else COL_DIM, (cx, cy),
                                   DOT_R, 0 if on else 2)
                lab = self.f_small.render(str(val), True,
                                          COL_BG if on else COL_DIM)
                s.blit(lab, (cx - lab.get_width() // 2, cy - lab.get_height() // 2))

        self._nav = {}
        miss = self.missing()
        bar_y = h - MARGIN - 46
        if miss:
            msg = "Answer every row to continue - missing: %s" % ", ".join(
                str(m) for m in miss)
            colour = COL_WARN
        else:
            msg = ("Continue" if self.page < len(self.pages) - 1 else "Finish") \
                + "  (Enter)"
            colour = COL_OK
        btn = pygame.Rect(w - MARGIN - 260, bar_y, 260, 46)
        pygame.draw.rect(s, COL_OK if not miss else COL_FOCUS, btn,
                         0 if not miss else 2, border_radius=10)
        bt = self.f_item.render("Continue" if self.page < len(self.pages) - 1
                                else "Finish", True,
                                COL_BG if not miss else COL_DIM)
        s.blit(bt, (btn.centerx - bt.get_width() // 2,
                    btn.centery - bt.get_height() // 2))
        if not miss:
            self._nav["next"] = btn
        if self.page > 0:
            bb = pygame.Rect(MARGIN, bar_y, 130, 46)
            pygame.draw.rect(s, COL_DIM, bb, 2, border_radius=10)
            lt = self.f_item.render("Back", True, COL_DIM)
            s.blit(lt, (bb.centerx - lt.get_width() // 2,
                        bb.centery - lt.get_height() // 2))
            self._nav["back"] = bb

        s.blit(self.f_small.render(msg, True, colour), (MARGIN, bar_y + 56))
        hint = ("Click a circle, or use the number keys 1-%d.  "
                "Up/Down or Tab moves between rows.  Enter continues." % self.n)
        s.blit(self.f_small.render(hint, True, COL_DIM), (MARGIN, bar_y + 12))


def collect_row(q: Questionnaire, args, duration_s: float):
    """Everything that goes in the CSV, derived once, here."""
    n = q.n
    mid = (n + 1) / 2.0
    row = {
        "logged_ts": datetime.datetime.now().isoformat(timespec="seconds"),
        "session_id": args.session_id or "",
        "participantid": args.participantid,
        "block_idx": args.block_idx,
        "k_condition": args.condition,
        "scale_points": n,
        "duration_s": round(duration_s, 1),
        "notes": args.notes or "",
    }
    bucket = {"usefulness": [], "satisfying": [], "magical": []}
    for pi, (tag, _t, items, _s) in enumerate(PAGES):
        for i, (_lo, _hi, pos_right, sub) in enumerate(items):
            raw = q.answers[pi][i]
            sc = score(raw, pos_right, n)
            row["%s%d_raw" % (tag, i + 1)] = raw
            row["%s%d_scored" % (tag, i + 1)] = sc
            bucket[sub].append(sc)

    def mean(v):
        return round(sum(v) / float(len(v)), 4) if v else ""
    for sub, prefix in (("usefulness", "vdl_usefulness"),
                        ("satisfying", "vdl_satisfying"),
                        ("magical", "intui_magical")):
        m = mean(bucket[sub])
        row[prefix + "_mean"] = m
        # Centred on the scale midpoint: Van der Laan is conventionally read
        # symmetrically, where 0 is indifference and the SIGN is the finding.
        row[prefix + "_centered"] = round(m - mid, 4) if m != "" else ""
    return row


def append_row(path: str, row: dict, columns) -> None:
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)
    new = not os.path.exists(path) or os.path.getsize(path) == 0
    full = {c: "" for c in columns}
    full.update({k: v for k, v in row.items() if k in columns})
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=columns)
        if new:
            w.writeheader()
        w.writerow(full)


def already_logged(path: str, args) -> bool:
    if not os.path.exists(path):
        return False
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if (r.get("participantid") == args.participantid
                    and str(r.get("block_idx")) == str(args.block_idx)):
                return True
    return False


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--participantid", required=True)
    ap.add_argument("--condition", required=True,
                    help="K condition this block served (0, 1 or 2). Recorded "
                         "verbatim; this module never resolves it to a K.")
    ap.add_argument("--block-idx", dest="block_idx", required=True,
                    help="Position of the block in the participant's sequence "
                         "(1, 2 or 3). Needed to model order effects.")
    ap.add_argument("--session-id", dest="session_id", default="",
                    help="Drive session id, to join this row to call_events.csv.")
    ap.add_argument("--out", default=DEFAULT_CSV)
    ap.add_argument("--scale-points", dest="scale_points", type=int, default=7,
                    help="Points per item (default: %(default)s). Van der Laan "
                         "is PUBLISHED as 5-point; 7 matches INTUI so the "
                         "participant meets one format, at the cost of "
                         "comparability with published norms. Keep it FIXED "
                         "across every participant and block.")
    ap.add_argument("--vdl-stem", dest="vdl_stem", default=VDL_STEM,
                    help="Shared instruction above the Van der Laan items "
                         "(default: %(default)r, the published wording). Note "
                         "it says 'system' while the page title says "
                         "'assistant'; change both together if the brief calls "
                         "it something else, and keep it FIXED across every "
                         "participant and block.")
    ap.add_argument("--intui-stem", dest="intui_stem", default=INTUI_STEM,
                    help="Shared stem for the INTUI items (default: %(default)r, "
                         "the published wording). Changing it to name the "
                         "assistant is a wording change to record in the "
                         "write-up, not a formatting choice.")
    ap.add_argument("--notes", default="", help="Free text stored with the row.")
    ap.add_argument("--fullscreen", action="store_true")
    ap.add_argument("--width", type=int, default=1400)
    ap.add_argument("--height", type=int, default=900)
    ap.add_argument("--allow-duplicate", dest="allow_duplicate",
                    action="store_true",
                    help="Log even though this participant already has a row "
                         "for this block. Off by default: a second row is "
                         "almost always a re-launch, and silently having two "
                         "answers for one block is worse than being stopped.")
    args = ap.parse_args()

    if not 3 <= args.scale_points <= 9:
        ap.error("--scale-points must be between 3 and 9 (the number keys are "
                 "the input path).")
    if already_logged(args.out, args) and not args.allow_duplicate:
        ap.error("participant %s already has a row for block %s in %s. Pass "
                 "--allow-duplicate if that is intended, or edit the file."
                 % (args.participantid, args.block_idx, args.out))

    columns = build_columns(args.scale_points)
    write_codebook(os.path.splitext(args.out)[0] + "_codebook.csv",
                   args.scale_points)

    pygame.init()
    # THE WHEEL IS BLOCKED, NOT IGNORED. pygame.init() brings up the joystick
    # module, and a G25 at rest streams axis motion from spring jitter; blocking
    # the types keeps them out of the queue entirely, so nothing here can be
    # driven by a paddle or a pedal. See the module docstring.
    for et in (pygame.JOYAXISMOTION, pygame.JOYBALLMOTION, pygame.JOYHATMOTION,
               pygame.JOYBUTTONDOWN, pygame.JOYBUTTONUP,
               pygame.JOYDEVICEADDED, pygame.JOYDEVICEREMOVED):
        pygame.event.set_blocked(et)
    pygame.mouse.set_visible(True)

    flags = pygame.FULLSCREEN if args.fullscreen else 0
    if args.fullscreen:
        info = pygame.display.Info()
        dim = (info.current_w, info.current_h)
    else:
        dim = (args.width, args.height)
    screen = pygame.display.set_mode(dim, flags)
    pygame.display.set_caption("Questionnaire - participant %s, block %s"
                               % (args.participantid, args.block_idx))

    q = Questionnaire(screen, args.scale_points,
                      {"vdl": args.vdl_stem, "intui": args.intui_stem})

    clock = pygame.time.Clock()
    started = time.time()
    while not q.done and not q.quit_requested:
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                q.quit_requested = True
            else:
                q.handle(ev)
        q.render()
        pygame.display.flip()
        clock.tick(60)

    pygame.quit()
    if q.quit_requested:
        # NOTHING is written on an abort. A partially-answered instrument scored
        # as if complete is worse than a missing block, because it is
        # indistinguishable from a real one afterwards.
        print("[questionnaire] ABORTED - nothing was written.")
        sys.exit(1)

    row = collect_row(q, args, time.time() - started)
    append_row(args.out, row, columns)
    print("[questionnaire] participant %s, block %s, condition %s -> %s"
          % (args.participantid, args.block_idx, args.condition, args.out))
    print("  Van der Laan  usefulness %.2f  satisfying %.2f   (centred %+.2f / %+.2f)"
          % (row["vdl_usefulness_mean"], row["vdl_satisfying_mean"],
             row["vdl_usefulness_centered"], row["vdl_satisfying_centered"]))
    print("  INTUI magical experience %.2f   (centred %+.2f)"
          % (row["intui_magical_mean"], row["intui_magical_centered"]))
    print("  completed in %.0f s" % row["duration_s"])


if __name__ == "__main__":
    main()
