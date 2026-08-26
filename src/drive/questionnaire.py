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

VAN DER LAAN IS 5 POINTS, UNLABELLED, AND NOW RENDERED THAT WAY
=================================================================
Verified against the printed form: nine rows of ``pole  |_|_|_|_|_|  pole`` --
five boxes, no digits in them. Van der Laan (1997) is a five-point semantic
differential, full stop; its norms, means and cut-offs are all on that scale.
An earlier version of this module ran it on seven numbered points to match
INTUI, on the reasoning that meeting one response format all evening beats two.
That reasoning is dropped, not merely overridden: matching the validated
instrument's actual form -- five unlabelled stops -- outranks response-format
consistency, so VDL_SCALE_POINTS is fixed at 5 and is NOT the ``--scale-points``
flag's business any more (that flag now sizes INTUI alone, still 7 by default).
Consequence worth stating plainly: the two pages ARE two different formats
again, dots-with-numbers for INTUI and an unlabelled slider for VDL. That is
the cost of being correct about VDL rather than convenient about it.

Rendered as a SLIDER -- a track with 5 stops and no printed numbers, dragged or
clicked -- rather than 5 numbered dots, because "5 boxes to tick" is what the
printed form shows and a slider is the closest on-screen analogue that keeps
the "no numbers" property; five numbered dots at N=5 would have been closer to
correct than N=7 but still shows digits the paper form does not.

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

# FIXED, not a CLI knob. The published instrument has 5 points; this is a fact
# about Van der Laan, not a formatting preference to sweep. See the module
# docstring. INTUI keeps its own, adjustable, --scale-points (default 7).
VDL_SCALE_POINTS = 5

# --- The proactivity item -----------------------------------------------------
#
# NOT part of Van der Laan or INTUI -- a single study-specific item, added
# after both because neither instrument asks about the thing this study
# actually manipulates: how MUCH the assistant did on its own. Rendered as a
# SLIDER (see SLIDER_PAGES below), the same unlabelled-track widget VDL uses,
# rather than INTUI's numbered dots -- an analogue judgement rather than a
# discrete pick from 5 labelled options suits a single global item better.
# The agree/disagree poles are still printed (they are not published-form
# boxes that must stay bare the way VDL's are); only the 5 stops between them
# go unlabelled.
#
# Wording was chosen from three candidates the operator was asked to pick
# between, on 2026-08-26: this one trades the originally-requested
# not-happy/happy poles for agree/disagree, the more standard Likert
# convention, keeping "happy" and "proactive" in the statement itself.
#
# Implements HALF of docs/live_study_setup.md section 4A.2's per-block pair --
# item 1, "overall satisfaction", reworded to name the construct (proactivity)
# rather than ask generically "how the calls were handled", which is what that
# section itself warns a bare satisfaction question invites. Item 2, the
# just-about-right "far too little...far too much" item, is NOT implemented
# here -- do not read this as that section being complete.
PROACTIVITY_STEM = ("I was happy with how proactive the assistant was during "
                    "this block.")
PROACTIVITY_ITEM = (
    ("Strongly disagree", "Strongly agree", True, "proactivity"),
)
# FIXED at 5, matching the 1-5 scale asked for -- not tied to --scale-points
# (INTUI's knob) or VDL_SCALE_POINTS (a fact about a published instrument,
# not a choice). This one is fixed because it was specified that way, not
# because of an external instrument's form.
PROACTIVITY_SCALE_POINTS = 5

# Which pages render as the unlabelled slider rather than numbered dots. VDL is
# here because that is its published form (see the module docstring);
# proactivity is here by request -- an unlabelled slider reads as a single
# analogue judgement rather than a discrete pick from 5 numbered options,
# which suits a global "how did this feel" item better than a Likert grid
# does. Nothing about its scoring changes: it is still 5 stops, still
# positive-right, still `score()` unmodified -- only the widget differs, the
# same way it does for VDL.
SLIDER_PAGES = frozenset({"vdl", "proactivity"})

PAGES = (
    ("vdl",         "Acceptance of the assistant", VDL_ITEMS, VDL_STEM),
    ("intui",       "Your experience",             INTUI_ITEMS, INTUI_STEM),
    ("proactivity", "Overall",                     PROACTIVITY_ITEM, PROACTIVITY_STEM),
)

DEFAULT_CSV = os.path.join("data", "study_1_questionnaire.csv")


def build_columns(intui_scale_points: int):
    """The CSV header. Fixed and explicit, in the style of call_events.csv.

    THREE scale-point columns: VDL is fixed at VDL_SCALE_POINTS, the
    proactivity item at PROACTIVITY_SCALE_POINTS, and INTUI is whatever
    --scale-points was for this run. A single shared column stopped being
    able to describe the questionnaire once the pages could disagree.
    """
    cols = ["logged_ts", "session_id", "participantid", "block_idx",
            "k_condition", "vdl_scale_points", "intui_scale_points",
            "proactivity_scale_points", "duration_s"]
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
    cols += ["intui_magical_mean", "intui_magical_centered"]
    for i in range(len(PROACTIVITY_ITEM)):
        cols.append("proactivity%d_raw" % (i + 1))
    for i in range(len(PROACTIVITY_ITEM)):
        cols.append("proactivity%d_scored" % (i + 1))
    cols += ["proactivity_mean", "proactivity_centered", "notes"]
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


def write_codebook(path: str, intui_scale_points: int) -> None:
    """One row per item: what it said, which way round, which subscale.

    Written once, beside the data. Without it the CSV is unreadable by anyone
    who does not have this module open -- and the reverse-coding is exactly the
    detail that gets reconstructed wrongly from memory.
    """
    if os.path.exists(path):
        return
    points_for = {"vdl": VDL_SCALE_POINTS, "intui": intui_scale_points,
                 "proactivity": PROACTIVITY_SCALE_POINTS}
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["column", "instrument", "item_no", "left_pole", "right_pole",
                    "positive_pole", "reverse_coded", "subscale", "scale_points",
                    "widget", "scoring"])
        for tag, items in (("vdl", VDL_ITEMS), ("intui", INTUI_ITEMS),
                          ("proactivity", PROACTIVITY_ITEM)):
            sp = points_for[tag]
            for i, (lo, hi, pos_right, sub) in enumerate(items, start=1):
                w.writerow(["%s%d" % (tag, i), tag, i, lo, hi,
                            "right" if pos_right else "left",
                            int(not pos_right), sub, sp,
                            "slider (unlabelled)" if tag in SLIDER_PAGES else "dots (numbered)",
                            "scored = raw" if pos_right
                            else "scored = %d - raw" % (sp + 1)])
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
TICK_R = 10               # slider stop half-size (a small square, not a dot)
TRACK_H = 3
ROW_H = 62
MARGIN = 44


class Questionnaire(object):
    """Two pages of semantic differentials, mouse or keyboard.

    The two pages use DIFFERENT widgets and, since this change, different
    point counts -- VDL a 5-stop unlabelled slider, INTUI N numbered dots. Both
    facts live in ``self._n_by_page`` / the ``page_kind`` helper rather than a
    single ``self.n``, which is what let the two pages silently share a scale
    they should not have.
    """

    def __init__(self, screen, intui_scale_points: int, stems: dict = None):
        self.screen = screen
        # Pages are built ONCE, here, with the stems substituted -- rather than
        # patching the module-level PAGES, which would make the instrument
        # depend on import order and leak between two instances.
        stems = stems or {}
        self.pages = tuple(
            (k, t, items, stems.get(k, stem))
            for k, t, items, stem in PAGES)
        # Per-page point count. VDL is never intui_scale_points, no matter what
        # was passed -- see VDL_SCALE_POINTS.
        _fixed_n = {"vdl": VDL_SCALE_POINTS, "proactivity": PROACTIVITY_SCALE_POINTS}
        self._n_by_page = tuple(
            _fixed_n.get(key, intui_scale_points)
            for key, _t, _items, _stem in self.pages)
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
        self._rows = []          # (rect, page, item, value) -- dot pages
        self._sliders = []       # (track_rect, page, item) -- slider pages
        self._dragging = None    # item index mid-drag on the CURRENT page, or None
        self._nav = {}

    @property
    def n(self):
        """This page's point count. VDL_SCALE_POINTS on the VDL page, else
        whatever --scale-points gave INTUI."""
        return self._n_by_page[self.page]

    @property
    def is_slider_page(self):
        return self.pages[self.page][0] in SLIDER_PAGES

    # -- geometry ------------------------------------------------------------

    def _layout(self):
        """Row rects for the current page. Recomputed per frame; 13 rows is free."""
        w, h = self.dim
        _, _, items, stem = self.pages[self.page]
        top = MARGIN + self.f_title.get_height() + 18
        if stem:
            top += self.f_item.get_height() + 14
        # The control band occupies the middle third; labels take the outer
        # thirds -- true whether that band holds N dots or a 5-stop slider.
        dot_lo, dot_hi = int(w * 0.40), int(w * 0.66)
        step = (dot_hi - dot_lo) / float(self.n - 1)
        rows = []
        for i in range(len(items)):
            y = top + i * ROW_H
            centers = [(int(dot_lo + k * step), y + ROW_H // 2)
                       for k in range(self.n)]
            rows.append((y, centers))
        return top, rows, dot_lo, dot_hi

    @staticmethod
    def _nearest_stop(x, dot_lo, dot_hi, n):
        """Pixel x -> the closest of the n evenly-spaced stops, as 1..n.

        Shared by a click ANYWHERE on the track and by dragging: both just ask
        "which stop is closest to this x", clamped to the track's own ends so
        an overshoot past either edge still lands on stop 1 or n rather than
        being ignored.
        """
        step = (dot_hi - dot_lo) / float(n - 1)
        x = max(dot_lo, min(dot_hi, x))
        return int(round((x - dot_lo) / step)) + 1

    # -- input ---------------------------------------------------------------

    def handle(self, ev):
        if ev.type == pygame.KEYDOWN:
            self._key(ev)
        elif ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1:
            self._click(ev.pos)
        elif ev.type == pygame.MOUSEMOTION and self._dragging is not None:
            self._drag(ev.pos)
        elif ev.type == pygame.MOUSEBUTTONUP and ev.button == 1:
            self._dragging = None

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
        # Sliders hit-test against the WHOLE track (a click anywhere on the bar
        # jumps to the nearest stop, same as a drag would settle there), not
        # per-stop circles the way the dot pages do -- a slider with 5 tiny
        # click targets and dead space between them would not read as a slider.
        for rect, page, item, dot_lo, dot_hi, n in self._sliders:
            if rect.collidepoint(pos):
                self.answers[page][item] = self._nearest_stop(
                    pos[0], dot_lo, dot_hi, n)
                self.focus = item
                self._dragging = item
                return
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

    def _drag(self, pos):
        """Follow the mouse while a slider handle is held.

        Re-reads the CURRENT page's track bounds from ``_layout()`` rather than
        caching them at mousedown: cheap (13 rows), and correct even in the
        pathological case of the window resizing mid-drag.
        """
        item = self._dragging
        if item is None or not self.is_slider_page:
            return
        _top, _rows, dot_lo, dot_hi = self._layout()
        self.answers[self.page][item] = self._nearest_stop(
            pos[0], dot_lo, dot_hi, self.n)

    def _advance(self):
        if self.missing():
            return
        self._dragging = None       # belongs to the page being left, if any
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

        is_slider = self.is_slider_page
        self._rows = []
        self._sliders = []
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
            if is_slider:
                self._render_slider(s, i, y, chosen, dot_lo, dot_hi)
            else:
                self._render_dots(s, i, centers, chosen)

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
        hint = ("Click or drag the slider, or use the number keys 1-5."
                if is_slider else
                "Click a circle, or use the number keys 1-%d." % self.n)
        hint += "  Up/Down or Tab moves between rows.  Enter continues."
        # ABOVE the button row, not blit at bar_y + 12: that sat INSIDE the
        # Back button's own rect (MARGIN, bar_y, 130, 46) whenever page > 0,
        # so the two overlapped -- invisible in a page-1 screenshot (no Back
        # button yet to collide with) and only visible once a later page was
        # actually rendered.
        hint_h = self.f_small.get_height()
        s.blit(self.f_small.render(hint, True, COL_DIM),
              (MARGIN, bar_y - hint_h - 10))

    def _render_dots(self, s, item, centers, chosen):
        """INTUI's widget: N numbered circles, unchanged from before this
        change -- VDL is the one that moved."""
        for k, (cx, cy) in enumerate(centers):
            val = k + 1
            rect = pygame.Rect(cx - DOT_R - 4, cy - DOT_R - 4,
                               2 * (DOT_R + 4), 2 * (DOT_R + 4))
            self._rows.append((rect, self.page, item, val))
            on = chosen == val
            pygame.draw.circle(s, COL_ACCENT if on else COL_DIM, (cx, cy),
                               DOT_R, 0 if on else 2)
            lab = self.f_small.render(str(val), True, COL_BG if on else COL_DIM)
            s.blit(lab, (cx - lab.get_width() // 2, cy - lab.get_height() // 2))

    def _render_slider(self, s, item, y, chosen, dot_lo, dot_hi):
        """The slider widget -- SLIDER_PAGES (VDL, proactivity): a track with
        this page's N unlabelled stops.

        Deliberately NO number is ever drawn here -- that is the entire point
        of switching off the dot widget for these pages. A stop is a small
        square (echoing VDL's printed "|_|" boxes, which is where this widget
        started), open when unanswered and filled when it is the current
        answer; nothing else marks position, so a participant reading the row
        has no digit to anchor a numeric judgement to.
        """
        cy = y + ROW_H // 2
        pygame.draw.line(s, COL_DIM, (dot_lo, cy), (dot_hi, cy), TRACK_H)
        step = (dot_hi - dot_lo) / float(self.n - 1)
        # ONE hit region spans the whole track -- see _click -- so only the
        # visual ticks are per-stop; the click/drag target is the full bar.
        self._sliders.append((
            pygame.Rect(dot_lo - TICK_R - 4, cy - ROW_H // 2,
                        (dot_hi - dot_lo) + 2 * (TICK_R + 4), ROW_H),
            self.page, item, dot_lo, dot_hi, self.n))
        for k in range(self.n):
            val = k + 1
            cx = int(dot_lo + k * step)
            on = chosen == val
            r = TICK_R + 2 if on else TICK_R
            rect = pygame.Rect(cx - r, cy - r, 2 * r, 2 * r)
            if on:
                pygame.draw.rect(s, COL_ACCENT, rect, 0, border_radius=3)
            else:
                pygame.draw.rect(s, COL_DIM, rect, 2, border_radius=3)


def collect_row(q: Questionnaire, args, duration_s: float):
    """Everything that goes in the CSV, derived once, here.

    THREE point counts, not one: q._n_by_page[pi] is VDL_SCALE_POINTS on the
    vdl page, PROACTIVITY_SCALE_POINTS on the proactivity page, and
    args.scale_points on the intui page, and `score()`/the midpoint each use
    the count for the page the item actually came from. Using ONE n for all
    of them (the pre-slider behaviour) would score 5-point answers as if they
    were on a 7-point scale -- silently, since every raw value 1-5 is also a
    valid 1-7 raw value, just the wrong scale's.
    """
    row = {
        "logged_ts": datetime.datetime.now().isoformat(timespec="seconds"),
        "session_id": args.session_id or "",
        "participantid": args.participantid,
        "block_idx": args.block_idx,
        "k_condition": args.condition,
        "vdl_scale_points": VDL_SCALE_POINTS,
        "intui_scale_points": q._n_by_page[1],
        "proactivity_scale_points": PROACTIVITY_SCALE_POINTS,
        "duration_s": round(duration_s, 1),
        "notes": args.notes or "",
    }
    bucket = {"usefulness": [], "satisfying": [], "magical": [], "proactivity": []}
    mid_for = {"usefulness": (VDL_SCALE_POINTS + 1) / 2.0,
              "satisfying": (VDL_SCALE_POINTS + 1) / 2.0,
              "magical": (q._n_by_page[1] + 1) / 2.0,
              "proactivity": (PROACTIVITY_SCALE_POINTS + 1) / 2.0}
    for pi, (tag, _t, items, _s) in enumerate(PAGES):
        n = q._n_by_page[pi]
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
                        ("magical", "intui_magical"),
                        ("proactivity", "proactivity")):
        m = mean(bucket[sub])
        row[prefix + "_mean"] = m
        # Centred on the scale midpoint: Van der Laan is conventionally read
        # symmetrically, where 0 is indifference and the SIGN is the finding.
        # Applied to every subscale here, proactivity included, for the same
        # reading -- 0 = neither agree nor disagree.
        row[prefix + "_centered"] = round(m - mid_for[sub], 4) if m != "" else ""
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
                    help="Points per INTUI item (default: %(default)s). Does "
                         "NOT touch Van der Laan -- VDL is fixed at "
                         "VDL_SCALE_POINTS=" + str(VDL_SCALE_POINTS) +
                         " (its published form, an unlabelled 5-stop "
                         "slider) regardless of this flag. Keep it FIXED "
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
    ap.add_argument("--proactivity-stem", dest="proactivity_stem",
                    default=PROACTIVITY_STEM,
                    help="The final item's statement (default: %(default)r). "
                         "This is OUR wording, not a published instrument's, "
                         "so it is overridable -- but keep it FIXED across "
                         "every participant and block like the other two.")
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
                      {"vdl": args.vdl_stem, "intui": args.intui_stem,
                       "proactivity": args.proactivity_stem})

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
    print("  Proactivity happiness %.2f   (centred %+.2f)"
          % (row["proactivity_mean"], row["proactivity_centered"]))
    print("  completed in %.0f s" % row["duration_s"])


if __name__ == "__main__":
    main()
