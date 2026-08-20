#!/usr/bin/env python

"""Simulated incoming phone call, handled according to the predicted LoA.

The interactive event of the live follow-up study. Design record and rationale:
``docs/live_study_setup.md`` (§5 the ladder, §6 the UI spec). This module is the
implementation of that spec and nothing else -- it owns no study logic, reads no
files, and knows nothing about K conditions or checkpoints.

WHAT THE LoA CONTROLS
---------------------
The call is a WORLD EVENT, not a system action: the phone rings in every
condition, identically. What the LoA governs is how involved the assistant gets.

This matters and is not decoration. If LoA 0 produced no on-screen event, the
driver would perceive nothing and the satisfaction rating for that window would
have no referent -- and since unpersonalized heads plausibly predict LoA 0 more
often, the unratable windows would concentrate in one experimental condition.
That is missing data correlated with the independent variable.

    LoA  assistant                     affordance        no input ->
    ---  ---------------------------   ---------------   ------------
    0    silent                        ACCEPT / DECLINE  rings out
    1    "Call from X."  + badge       ACCEPT / DECLINE  rings out
    2    "...Want me to answer?"       YES / NO          NOT answered
    3    "...Answering in 3, 2, 1."    CANCEL            ANSWERED
    4    "...Answering now."           none              answered

The "no input" column is the formal content of the ladder: doing nothing yields
*nothing* for 0-2 and *the action* for 3-4, and that flip is what "veto" means.

WHY 0->1 IS ONE MARKER AND 1->2 IS THE WHOLE PANEL
--------------------------------------------------
For a binary, immediate action, "suggest" and "ask approval" collapse in natural
language -- any natural phrasing of a suggestion about answering a call ("want to
take it?") *is* asking approval. So the distinction cannot live in the wording,
and it does not: it lives in who owns the screen. At LoA 1 the driver is
operating their own phone card and the assistant merely annotates it with a
badge; at LoA 2 the card is replaced by an assistant panel and the driver is
answering the assistant, not the phone.

That asymmetry mirrors ``decision_engine._LOA_POLICY``, which groups the levels
low/low/medium/high/high -- the visual hinge sits exactly where the policy hinge
does. These two rungs are ~59% of the labels drivers gave for this function, so
they are the ones that have to be separable.

DEGRADES TO SILENCE, NEVER TO A GUESS
-------------------------------------
Missing audio files, no mixer, no audio device: the panel still renders and the
interaction still resolves, exactly as ``ambience.py`` degrades. What it will
NOT do is invent an LoA -- ``arm()`` refuses a ``None`` LoA and the caller logs a
skipped window, because a fabricated level is indistinguishable from a served one
in the results.
"""

import math
import os

import pygame
from pygame.locals import K_a, K_b


# --- Geometry ---------------------------------------------------------------
#
# The panel sits in the HUD band to the RIGHT of the speed readout, sharing its
# baseline (docs/live_study_setup.md §6.1). Bottom-aligned and growing UPWARD:
# the speed is at 0.80 h because that is the base of the windshield, and a panel
# centred on that line would reach ~0.87 h and overlap the rendered wheel.
#
# The box is MEASURED, once, from the widest string any of the five renderings
# can produce (``_measure``), not set to a guessed fraction of the window. A
# fraction has to be picked for the worst case and then looks empty in every
# other state -- which is exactly how the first version read. Measuring keeps
# the footprint identical across the five (the invariant that matters: a panel
# that shrank at LoA 4 would make size a salience cue riding along with the
# condition) while removing the dead space.
#
# The clamps are guard rails, not the layout: they stop a pathological font or a
# very long caller name from producing a panel that spans the windscreen.
PANEL_LEFT_FRAC = 0.35
PANEL_MIN_WIDTH_FRAC = 0.16
PANEL_MAX_WIDTH_FRAC = 0.34

# VERTICAL ANCHOR: the panel is CENTRED on the speed readout rather than sharing
# its baseline, so growing the top margin below opens the box symmetrically
# instead of pushing its head toward the horizon.
#
# drive_improved's HUD._render_speed puts the digits' BOTTOM edge at
# int(height * 0.8), so the readout's midline is that minus half a glyph. The
# glyph height is derived here from the same formula rather than hard-coded, so
# the two stay aligned at 720p and fullscreen alike -- see _speed_block_height.
SPEED_BASELINE_FRAC = 0.80
PANEL_CENTER_NUDGE_PX = 0        # tuning knob; + moves the panel down

# Inner margins. Top and left are deliberately larger than bottom and right:
# the content hangs from the top-left corner, so that is where the box reads as
# cramped, and the trailing edges already have the measured slack.
PANEL_PAD = 14                   # bottom and right
PANEL_PAD_TOP = 22
PANEL_PAD_LEFT = 22
PANEL_ICON_GAP = 10

# CHROME (default 'none'): 'panel' draws the translucent plate and border;
# 'none' puts the text straight onto the scene. Per-instance via the `chrome`
# kwarg, so --call-chrome can flip it without editing this file. Text-only occludes far less of the road and matches
# the speed readout, which has no plate either -- but it hands legibility over
# to whatever the driver is passing, and this panel's body text is 15-18 pt
# against the speed readout's 39 pt digits. So text-only is only viable WITH the
# outline below, which is what the plate was standing in for.
PANEL_CHROME = 'none'
# Dark halo behind every glyph and marker. ONLY in 'none' mode -- under the
# plate it is redundant and reads as a heavy black stroke around the text.
# 1 px is enough: the halo only has to break the glyph away from the scene, and
# 2 px starts to look like an outline typeface.
OUTLINE_PX = 1
COL_OUTLINE = (0, 0, 0)

PANEL_RADIUS = 10                # corner rounding
PANEL_BORDER_W = 1               # outline weight; 2 read as heavy at this size
PLATE_ALPHA = 110          # same value the speed readout's (disabled) plate used


# --- Palette ----------------------------------------------------------------
#
# Marker AND colour in every state, per the convention in drive_improved.py: the
# state has to survive a projector and a colour-blind participant.
# PANEL BACKGROUND. White text sets a hard floor on how light this can go: at
# relative luminance above ~0.18 the contrast ratio drops under 4.5:1 and the
# body text stops being legible over a moving road. (60,120,180) measures 4.9:1
# against white -- push it lighter than this and the text has to go dark.
COL_PANEL_BG = (60, 120, 180)
# SEMI-TRANSPARENT, so the road reads through the panel. The cost is that the
# effective background is now whatever is behind it, and white-text contrast
# moves with the scene. Measured for (60,120,180) over four backdrops:
#
#   alpha   dark tarmac   mid road   bright tarmac   pale sky
#     180      5.72:1       4.48:1       3.71:1       3.19:1
#     190      5.6 :1       4.50:1       3.8 :1       3.3 :1
#     235      4.89:1       4.59:1       4.37:1       4.18:1
#
# 190 keeps every backdrop above the 3:1 large/bold threshold and mid road at
# the 4.5:1 body threshold, while still letting a quarter of the scene through.
# Going much lower puts pale backdrops under 3:1 and the text starts to swim.
PLATE_ALPHA_BG = 190

COL_WHITE = (255, 255, 255)
# Borders no longer carry the driver/assistant distinction by HUE -- everything
# is on one blue panel now, so a blue border would vanish into it. The
# distinction is single vs DOUBLE border, plus the wording and input modality
# that docs/live_study_setup.md 5.3 makes load-bearing.
COL_PHONE = (232, 240, 250)
COL_ASSISTANT = (232, 240, 250)
# Accent for headers and markers -- the blue that carries the hierarchy now the
# font is back to the drive UI's own (already-bold) face, so weight cannot.
COL_ACCENT = (140, 200, 255)
COL_DIM = (205, 220, 238)          # lifted off (140,140,140): too dark on blue
# The "INCOMING CALL" header. Its own constant, not an alias of COL_ACCENT: the
# two mean different things (a card title vs "the assistant owns this"), so the
# LoA 0/1 vs 2 distinction can be sharpened up by changing this one alone.
COL_CALL = (140, 200, 255)
COL_CONNECTED = (120, 240, 150)
COL_COUNTDOWN = (255, 170, 90)


# --- Input ------------------------------------------------------------------
#
# MUST stay in sync with drive_improved.py's wheel mapping. Button 7 is QUIT,
# honoured everywhere in the drive UI -- never bind it here.
#
# [A] is ALWAYS the affirmative (Answer / Yes) and [B] ALWAYS the negative
# (Decline / No / Cancel), at every LoA. If ACCEPT and YES were different
# physical buttons, a motor confound would ride along with the condition.
CALL_WHEEL_BUTTON_AFFIRM = 6
CALL_WHEEL_BUTTON_NEGATIVE = 5

CALL_KEY_AFFIRM = K_a
CALL_KEY_NEGATIVE = K_b


# --- Timing -----------------------------------------------------------------
#
# Onset is FIXED relative to window start and identical across conditions; only
# the resolution time varies. That asymmetry is intrinsic (autonomy is partly a
# claim about interaction time) but the onset must not add to it.
DEFAULT_ONSET_OFFSET_S = 5.0   # from arm() to the phone ringing
DEFAULT_CAP_S = 8.0            # ringing -> forced timeout, for the input rungs
RING_LEAD_S = 1.2              # ring alone before the assistant speaks or acts
COUNTDOWN_S = 3.0              # LoA 3 only
CONNECTED_HOLD_S = 3.0         # caller's canned reply, then auto hang-up


# --- States -----------------------------------------------------------------
IDLE = 'idle'
ARMED = 'armed'
RINGING = 'ringing'
AWAIT_INPUT = 'await_input'
COUNTDOWN = 'countdown'
DONE = 'done'
CONNECTED = 'connected'

# Which middle state each LoA enters after the ring lead-in. This dict is the
# ONLY place the LoA is branched on -- everything else is shared, which is what
# makes the five renderings one component instead of five.
_LOA_ROUTE = {
    0: AWAIT_INPUT,
    1: AWAIT_INPUT,
    2: AWAIT_INPUT,
    3: COUNTDOWN,
    4: CONNECTED,
}

# Assistant utterance per LoA. LoA 0 is silent -- that is the whole content of
# "no assistive action is taken".
_ASSISTANT_LINE = {
    0: None,
    1: 'Call from %s.',
    2: 'Call from %s. Want me to answer?',
    # LoA 3's on-screen text is built per frame by _panel_line (it carries the
    # live digit); this entry documents what loa3_line.wav says.
    3: 'Call from %s. Answering in 3… 2… 1…',
    4: 'Call from %s. Answering now.',
}

# Bitmap glyph for the receiver, dropped in beside the clips with its licence
# recorded in assets/calls/manifest.json. Blitted AS DRAWN -- scaled and nothing
# else. Override the file with PROVOICE_CALL_ICON.
#
# Consequence to be aware of: the artwork carries its own colours, so the icon
# looks identical on the driver's phone card and on the assistant's panel. The
# driver / assistant distinction is still carried by the border weight and
# colour, the diamond marker, which widget is on screen and the button verbs --
# the icon simply stops being one of the cues.
ICON_FILE = 'phone-call.png'

_SOUND_FILES = {
    'ring': 'ring.wav',
    1: 'loa1_line.wav',
    2: 'loa2_line.wav',
    3: 'loa3_line.wav',
    4: 'loa4_line.wav',
    'reply': 'caller_reply.wav',
}


def _default_assets_dir():
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.abspath(os.path.join(here, '..', '..'))
    return os.environ.get('PROVOICE_CALL_ASSETS_DIR',
                          os.path.join(root, 'assets', 'calls'))


def _mono_face():
    """The face drive_improved's HUD uses, or None for pygame's default.

    Replicates HUD.__init__'s lookup exactly: prefer ``ubuntumono``, else the
    first installed family matching 'courier' (Windows) / 'mono'. This is the
    face the SPEED READOUT is drawn in, which is the whole point -- the call
    panel sits beside it and has to look like part of the same instrument.

    Copied rather than imported: importing drive_improved from here would be a
    cycle, so it has to be re-checked if that selection ever changes.
    """
    try:
        font_name = 'courier' if os.name == 'nt' else 'mono'
        fonts = [x for x in pygame.font.get_fonts() if font_name in x]
        mono = 'ubuntumono' if 'ubuntumono' in fonts else (fonts[0] if fonts else None)
        return pygame.font.match_font(mono) if mono else None
    except Exception:                                         # noqa: BLE001
        return None


def _speed_block_height(height):
    """Glyph height of drive_improved's speed readout, for the given window.

    Replicates HUD.__init__: ``speed_pt = max(28, int(height * 0.055))``.
    """
    try:
        return pygame.font.Font(_mono_face(),
                                max(28, int(height * 0.055))).get_height()
    except Exception:                                         # noqa: BLE001
        return int(height * 0.062)          # what the formula yields in practice


def _load_icon(path):
    """The icon surface, or None. Never raises: a missing file is not fatal."""
    if not path or not os.path.exists(path):
        return None
    try:
        return pygame.image.load(path).convert_alpha()
    except Exception as exc:                                  # noqa: BLE001
        print('[WARN] call_event: could not read icon %s (%s); '
              'falling back to the drawn receiver.' % (path, exc))
        return None


def _load_sound(path):
    """A Sound, or None. Never raises: a missing clip must not end a session."""
    if not path or not os.path.exists(path):
        return None
    try:
        if not pygame.mixer.get_init():
            return None
        return pygame.mixer.Sound(path)
    except Exception as exc:                                  # noqa: BLE001
        print('[WARN] call_event: could not load %s (%s).' % (path, exc))
        return None


class CallEvent(object):
    """One incoming-call event per window, rendered according to a served LoA.

    Lifecycle per window::

        arm(now_ms, loa, window_idx)     -> schedules it at the fixed offset
        update(now_ms)                   -> every frame; returns the outcome
                                            dict on the tick it resolves
        handle_event(pygame_event)       -> while `active`
        render(display)                  -> every frame; no-op when inactive

    Deliberately does NOT freeze the simulation. The label pop-up dims the whole
    screen and holds the clock; this does neither, because the driver is still
    driving and the event is meant to be handled in traffic.
    """

    def __init__(self, dim, assets_dir=None, caller_name='Mark',
                 onset_offset_s=DEFAULT_ONSET_OFFSET_S, cap_s=DEFAULT_CAP_S,
                 enabled=True, chrome=None):
        self.dim = dim
        self.chrome = chrome or PANEL_CHROME
        self.enabled = enabled
        self.caller_name = caller_name
        self.onset_offset_ms = float(onset_offset_s) * 1000.0
        self.cap_ms = float(cap_s) * 1000.0

        self.state = IDLE
        self.loa = None
        self.window_idx = None
        self._onset_ms = 0
        self._state_entered_ms = 0
        self._response = None
        self._response_ms = None
        self._answered = False
        self._pending_outcome = None

        if pygame.font.get_init():
            # THE SPEED READOUT'S FACE, not pygame's default. The panel sits
            # beside the speed and has to read as the same instrument; the
            # default (freesansbold) is what the LoA pop-up uses, but that is a
            # full-screen modal, not something shown next to the digits.
            #
            # Mono is ~25 % wider per character than the proportional default,
            # so the sizes are a step down from the 22/18/15 that face wanted --
            # otherwise the measured width hits PANEL_MAX_WIDTH_FRAC and the
            # assistant's line wraps.
            face = _mono_face()
            self._font_title = pygame.font.Font(face, 19)
            self._font_text = pygame.font.Font(face, 16)
            self._font_small = pygame.font.Font(face, 13)
            # BOLD everywhere. Courier's regular weight is a thin, low-contrast
            # stroke that reads as weak at 13-19 pt over a moving scene -- the
            # speed readout gets away with it only because it is 39 pt. Set
            # BEFORE _layout: emboldening widens every glyph, so _measure has to
            # see the bold metrics or the panel is sized too small and wraps.
            for _f in (self._font_title, self._font_text, self._font_small):
                _f.set_bold(True)
        else:                        # standalone import, no display yet
            self._font_title = self._font_text = self._font_small = None

        self.icon_size = (self._font_title.get_height()
                          if self._font_title else 22)
        self.rect = self._layout(dim)

        self._assets_dir = assets_dir or _default_assets_dir()
        self._sounds = {}
        self._ring_channel = None
        self._voice_channel = None
        self._load_sounds()
        name = os.environ.get('PROVOICE_CALL_ICON') or ICON_FILE
        self._glyph = _load_icon(
            name if os.path.isabs(name) else os.path.join(self._assets_dir, name))
        self.icon_source = os.path.basename(name) if self._glyph is not None else None
        self._icon_cache = {}

    # -- layout --------------------------------------------------------------

    def _measure(self):
        """Widest string any of the five renderings can produce, in pixels.

        Every candidate is listed explicitly rather than measured lazily at draw
        time, because the box must be identical in all five states -- sizing it
        to whatever happens to be on screen would make the panel breathe between
        conditions.
        """
        if self._font_text is None:
            return 0
        cands = [(self._font_title, 'ASSISTANT'),
                 (self._font_title, 'INCOMING CALL'),
                 (self._font_text, '[A] Answer         RECOMMENDED'),
                 (self._font_text, '[A] Yes        [B] No'),
                 (self._font_text, '[B] Decline'),
                 (self._font_text, '[B] Cancel'),
                 (self._font_text, self.caller_name),
                 (self._font_small, '%s — ringing, on hold' % self.caller_name),
                 (self._font_small, '%s — connected     00:00' % self.caller_name)]
        for loa, tmpl in _ASSISTANT_LINE.items():
            if not tmpl:
                continue
            if loa == 3:
                # What is DRAWN is the live countdown ("Answering in 3…"),
                # never the three-digit template -- measuring the template made
                # the panel ~25 %% wider than anything it can display.
                tmpl = 'Call from %s. Answering in 3…'
            cands.append((self._font_text, '"%s"' % (tmpl % self.caller_name)))
        return max(f.size(t)[0] for f, t in cands)

    def _layout(self, dim):
        """The panel rect: measured width, row-counted height, fixed anchor."""
        width, height = dim
        if self._font_text is None:
            return pygame.Rect(int(width * PANEL_LEFT_FRAC), 0,
                               int(width * PANEL_MIN_WIDTH_FRAC), 0)

        content = self._measure() + self.icon_size + PANEL_ICON_GAP
        w = int(min(max(content + PANEL_PAD_LEFT + PANEL_PAD,
                        width * PANEL_MIN_WIDTH_FRAC),
                    width * PANEL_MAX_WIDTH_FRAC))

        # Tallest rendering is LoA 3: header, one spoken line, the countdown
        # bar, the cancel row, then the separator and status strip.
        th, xh, sh = (self._font_title.get_height(), self._font_text.get_height(),
                      self._font_small.get_height())
        h = (PANEL_PAD_TOP + PANEL_PAD + (th + 4) + 2 * (xh + 4) + 16 + 6 + sh + 6)

        speed_mid = (height * SPEED_BASELINE_FRAC
                     - _speed_block_height(height) / 2.0)
        top = int(speed_mid - h / 2.0) + PANEL_CENTER_NUDGE_PX
        return pygame.Rect(int(width * PANEL_LEFT_FRAC), top, w, h)

    def _icon(self, size):
        """The icon scaled to `size`, cached. smoothscale resamples the full
        source every call, which does not belong in a frame."""
        key = int(size)
        hit = self._icon_cache.get(key)
        if hit is None:
            hit = pygame.transform.smoothscale(self._glyph, (key, key))
            self._icon_cache[key] = hit
        return hit

    def _draw_handset(self, display, cx, cy, size, colour, waves=False):
        """A telephone receiver: two bulbous ends joined by a bowed handle.

        The first version drew equal-width circles on a straight bar, which
        reads as a dumbbell rather than a phone. What makes the silhouette
        legible at ~22 px is the CONTRAST between fat ends and a thin handle,
        plus the bow -- a straight connector looks like a barbell at any size.

        Drawn rather than loaded from an asset: it costs no file, scales with
        the window like everything else here, and recolours with the panel
        border, so the icon carries the same driver's-phone / assistant's-panel
        distinction the rest of the layout does.
        """
        if self._glyph is not None:
            # `colour` and `waves` are ignored: the artwork is used as drawn, in
            # its own colours, the same in every state. The drawn fallback below
            # still honours both.
            icon = self._icon(size * 1.35)
            display.blit(icon, (int(cx - icon.get_width() / 2),
                                int(cy - icon.get_height() / 2)))
            return

        # The ends are bars PERPENDICULAR to the handle, not blobs along it.
        # That is what separates a telephone receiver from a barbell, and it is
        # the whole reason the first two attempts read as gym equipment.
        handle = max(2, int(size * 0.11))
        cap_w = max(3, int(size * 0.16))
        half = size * 0.34
        ang = math.radians(-45)
        ux, uy = math.cos(ang), math.sin(ang)            # along the handle
        px, py = -uy, ux                                 # across it
        p0 = (cx - ux * half, cy - uy * half)            # mouthpiece
        p1 = (cx + ux * half, cy + uy * half)            # earpiece

        # Handle bows away from the diagonal, the way a receiver's does.
        bow = size * 0.20
        ctrl = (cx + px * bow, cy + py * bow)
        pts = []
        for k in range(9):
            t = k / 8.0
            u = 1.0 - t
            pts.append((u * u * p0[0] + 2 * u * t * ctrl[0] + t * t * p1[0],
                        u * u * p0[1] + 2 * u * t * ctrl[1] + t * t * p1[1]))
        pygame.draw.lines(display, colour, False, pts, handle)

        cap = size * 0.21
        for end in (p0, p1):
            a = (end[0] - px * cap, end[1] - py * cap)
            b = (end[0] + px * cap, end[1] + py * cap)
            pygame.draw.line(display, colour, a, b, cap_w)
            for tip in (a, b):
                pygame.draw.circle(display, colour,
                                   (int(tip[0]), int(tip[1])), cap_w // 2)

        if waves:
            # Two arcs off the earpiece. Static, never animated: the icon is
            # identical in every condition and must stay that way.
            for k, rad in enumerate((size * 0.62, size * 0.84)):
                box = pygame.Rect(0, 0, int(rad * 2), int(rad * 2))
                box.center = (int(cx), int(cy))
                pygame.draw.arc(display, colour, box, math.radians(20),
                                math.radians(70), max(1, int(size * 0.10) - k))

    def _draw_diamond(self, display, cx, cy, size, colour):
        """The assistant marker. Was a U+25C6 glyph, which freesansbold.ttf
        does not have -- it rendered as an empty .notdef box. Same for the star
        below and the U+25A0 the status strip used to carry. Anything outside
        Latin-1 has to be DRAWN here, not typed.
        """
        r = size * 0.34
        if OUTLINE_PX and self.chrome == 'none':
            o = r + OUTLINE_PX
            pygame.draw.polygon(display, COL_OUTLINE, [
                (cx, cy - o), (cx + o, cy), (cx, cy + o), (cx - o, cy)])
        pygame.draw.polygon(display, colour, [
            (cx, cy - r), (cx + r, cy), (cx, cy + r), (cx - r, cy)])

    def _draw_star(self, display, cx, cy, size, colour):
        """The recommendation marker (was U+2605; see _draw_diamond)."""
        def _pts(grow):
            out = []
            for k in range(10):
                rad = (size * 0.50 if k % 2 == 0 else size * 0.22) + grow
                a = math.radians(-90 + k * 36)
                out.append((cx + rad * math.cos(a), cy + rad * math.sin(a)))
            return out
        if OUTLINE_PX and self.chrome == 'none':
            pygame.draw.polygon(display, COL_OUTLINE, _pts(OUTLINE_PX))
        pygame.draw.polygon(display, colour, _pts(0))

    # -- audio ---------------------------------------------------------------

    def _load_sounds(self):
        missing = []
        for key, name in _SOUND_FILES.items():
            snd = _load_sound(os.path.join(self._assets_dir, name))
            self._sounds[key] = snd
            if snd is None:
                missing.append(name)
        if missing:
            # Loud once, then silent: the study can be piloted without audio,
            # but nobody should discover afterwards that it ran without any.
            print('[WARN] call_event: %d clip(s) missing from %s (%s). The event '
                  'will run SILENTLY.' % (len(missing), self._assets_dir,
                                          ', '.join(missing)))

    def _play(self, key, loops=0):
        snd = self._sounds.get(key)
        if snd is None:
            return None
        try:
            return snd.play(loops=loops)
        except Exception:                                     # noqa: BLE001
            return None

    def _stop_ring(self):
        if self._ring_channel is not None:
            try:
                self._ring_channel.stop()
            except Exception:                                 # noqa: BLE001
                pass
            self._ring_channel = None

    # -- lifecycle -----------------------------------------------------------

    @property
    def active(self):
        """True while the panel is on screen and may consume input."""
        return self.state in (RINGING, AWAIT_INPUT, COUNTDOWN, CONNECTED)

    @property
    def wants_audio_duck(self):
        """Whether the caller should duck the ambience bed this frame."""
        return self.active

    def arm(self, now_ms, loa, window_idx=None):
        """Schedule this window's call. False if nothing was armed.

        ``loa`` of None means no decision was available -- the transport was
        stale, or ProVoice had not produced one yet. We refuse rather than
        substituting a level: a fabricated LoA is indistinguishable from a served
        one afterwards. The caller logs the skip.
        """
        if not self.enabled or loa is None:
            return False
        try:
            loa = int(loa)
        except (TypeError, ValueError):
            return False
        if loa not in _LOA_ROUTE:
            print('[WARN] call_event: LoA %r out of range 0-4; not arming.' % (loa,))
            return False

        self.loa = loa
        self.window_idx = window_idx
        self.state = ARMED
        self._onset_ms = now_ms + self.onset_offset_ms
        self._state_entered_ms = now_ms
        self._response = None
        self._response_ms = None
        self._answered = False
        self._pending_outcome = None
        return True

    def _enter(self, state, now_ms):
        self.state = state
        self._state_entered_ms = now_ms
        if state == CONNECTED:
            self._stop_ring()
            self._answered = True
            self._play('reply')

    def _elapsed(self, now_ms):
        return now_ms - self._state_entered_ms

    def update(self, now_ms):
        """Advance the machine. Returns the outcome dict once, on resolution.

        EVERY resolution leaves through here, including the ones the driver
        triggers from ``handle_event``. Returning the outcome from two different
        methods would make it the caller's job to remember both, and the one
        that got forgotten would be the decline -- silently unlogged.
        """
        # Rendering reads this instead of calling pygame.time.get_ticks()
        # itself. The two agree in the drive loop, but the countdown digit and
        # the call timer are computed from it -- and when they disagreed, the
        # preview showed "Answering in 5" and a call duration of -1:59.
        self._now_ms = now_ms
        pending = self._take_pending()
        if pending is not None:
            return pending
        if self.state in (IDLE, DONE):
            return None

        if self.state == ARMED:
            if now_ms >= self._onset_ms:
                self._enter(RINGING, now_ms)
                self._ring_channel = self._play('ring', loops=-1)
                # LoA 0 is silent by definition; the rest speak at the lead-in.
            return None

        if self.state == RINGING:
            if self._elapsed(now_ms) >= RING_LEAD_S * 1000.0:
                if self.loa in self._sounds and self._sounds.get(self.loa):
                    self._voice_channel = self._play(self.loa)
                self._enter(_LOA_ROUTE[self.loa], now_ms)
            return None

        if self.state == AWAIT_INPUT:
            # Cap measured from ONSET, not from state entry, so the driver's
            # window to act does not silently differ between the rungs by the
            # length of the assistant's utterance.
            if now_ms - self._onset_ms >= self.cap_ms:
                self._response = 'timeout'
                self._response_ms = now_ms
                self._resolve(now_ms, answered=False)
                return self._take_pending()
            return None

        if self.state == COUNTDOWN:
            if self._elapsed(now_ms) >= COUNTDOWN_S * 1000.0:
                # Nobody vetoed: the action happens. This is the whole content
                # of auto_with_veto and the reason the timeout means the
                # opposite of what it means at LoA 2.
                if self._response is None:
                    self._response = 'timeout'
                    self._response_ms = now_ms
                self._enter(CONNECTED, now_ms)
            return None

        if self.state == CONNECTED:
            if self._elapsed(now_ms) >= CONNECTED_HOLD_S * 1000.0:
                self._resolve(now_ms, answered=True)
                return self._take_pending()
            return None

        return None

    def _resolve(self, now_ms, answered):
        self._stop_ring()
        self.state = DONE
        self._answered = answered
        self._pending_outcome = self.outcome(now_ms)
        return self._pending_outcome

    def _take_pending(self):
        out, self._pending_outcome = self._pending_outcome, None
        return out

    def outcome(self, now_ms=None):
        """The row for ``call_events.csv`` (docs/live_study_setup.md §9)."""
        latency = None
        if self._response_ms is not None:
            latency = int(self._response_ms - self._onset_ms)
        return {
            'window_idx': self.window_idx,
            'served_loa': self.loa,
            'event_onset_ms': int(self._onset_ms),
            'driver_response': self._response or 'none',
            'response_latency_ms': latency,
            'outcome': 'answered' if self._answered else 'not_answered',
            'skipped_reason': '',
        }

    def stop(self):
        """Release audio. Mirrors ``Ambience.stop()``; safe to call any time."""
        self._stop_ring()
        if self._voice_channel is not None:
            try:
                self._voice_channel.stop()
            except Exception:                                 # noqa: BLE001
                pass
            self._voice_channel = None
        self.state = IDLE

    # -- input ---------------------------------------------------------------

    def handle_event(self, event, now_ms=None):
        """Consume one pygame event. Returns 'affirmative'/'negative' or None.

        Only AWAIT_INPUT and COUNTDOWN are live, and COUNTDOWN accepts the
        negative alone -- at LoA 3 there is no affirmative button, because doing
        nothing already IS the affirmative.
        """
        if self.state not in (AWAIT_INPUT, COUNTDOWN):
            return None

        affirm = negative = False
        if event.type == pygame.KEYDOWN:
            affirm = event.key == CALL_KEY_AFFIRM
            negative = event.key == CALL_KEY_NEGATIVE
        elif event.type == pygame.JOYBUTTONDOWN:
            affirm = event.button == CALL_WHEEL_BUTTON_AFFIRM
            negative = event.button == CALL_WHEEL_BUTTON_NEGATIVE
        else:
            return None

        # Same clock as update(). The drive loop passes pygame.time.get_ticks()
        # to both, so the default matches -- but taking it implicitly made the
        # response latency depend on a caller invariant that nothing enforced,
        # and it is the one field the whole event exists to measure.
        if now_ms is None:
            now_ms = pygame.time.get_ticks()

        if self.state == COUNTDOWN:
            if negative:
                self._response = 'cancel'
                self._response_ms = now_ms
                self._resolve(now_ms, answered=False)
                return 'negative'
            return None

        if affirm:
            self._response = 'accept' if self.loa in (0, 1) else 'yes'
            self._response_ms = now_ms
            self._enter(CONNECTED, now_ms)
            return 'affirmative'
        if negative:
            self._response = 'decline' if self.loa in (0, 1) else 'no'
            self._response_ms = now_ms
            self._resolve(now_ms, answered=False)
            return 'negative'
        return None

    # -- rendering -----------------------------------------------------------

    def render(self, display):
        if not self.active or self._font_text is None:
            return

        assistant_owns = self.loa >= 2
        border = COL_ASSISTANT if assistant_owns else COL_PHONE

        # Backing plate. The speed readout's own (disabled) plate carries the
        # reason in its comment: white text over a bright road surface is
        # unreadable exactly when the driver is looking for it. A five-row panel
        # needs it far more than three digits do.
        if self.chrome == 'panel':
            # SRCALPHA + a rounded draw, rather than fill + set_alpha: set_alpha
            # is a whole-surface value and would square the corners back off.
            plate = pygame.Surface(self.rect.size, pygame.SRCALPHA)
            pygame.draw.rect(plate, COL_PANEL_BG + (PLATE_ALPHA_BG,),
                             plate.get_rect(), border_radius=PANEL_RADIUS)
            display.blit(plate, self.rect.topleft)

            pygame.draw.rect(display, border, self.rect, PANEL_BORDER_W,
                             border_radius=PANEL_RADIUS)
            if assistant_owns:
                # Double border: the single strongest cue that this panel is not
                # the driver's phone any more.
                pygame.draw.rect(display, border, self.rect.inflate(-8, -8),
                                 PANEL_BORDER_W,
                                 border_radius=max(2, PANEL_RADIUS - 4))

        x = self.rect.left + PANEL_PAD_LEFT
        y = self.rect.top + PANEL_PAD_TOP

        if assistant_owns:
            y = self._render_assistant(display, x, y)
        else:
            y = self._render_phone_card(display, x, y)

        self._render_status_strip(display)

    def _blit(self, display, font, text, colour, x, y):
        if OUTLINE_PX and self.chrome == 'none':
            halo = font.render(text, True, COL_OUTLINE)
            r = OUTLINE_PX
            for dx, dy in ((-r, 0), (r, 0), (0, -r), (0, r),
                           (-r, -r), (r, -r), (-r, r), (r, r)):
                display.blit(halo, (x + dx, y + dy))
        display.blit(font.render(text, True, colour), (x, y))
        return y + font.get_height() + 4

    def _render_phone_card(self, display, x, y):
        """LoA 0 and 1: the driver's own phone. The assistant only annotates."""
        ring = self.state in (RINGING, AWAIT_INPUT)
        self._draw_handset(display, x + self.icon_size / 2,
                           y + self._font_title.get_height() / 2,
                           self.icon_size, COL_PHONE, waves=ring)
        tx = x + self.icon_size + PANEL_ICON_GAP
        header = 'INCOMING CALL'
        head_y = y
        y = self._blit(display, self._font_title, header, COL_CALL, tx, y)
        if self.loa == 1:
            # Assistant present but passive: it annotates the driver's card
            # rather than owning it.
            self._draw_diamond(display,
                               tx + self._font_title.size(header)[0]
                               + self.icon_size * 0.7,
                               head_y + self._font_title.get_height() / 2,
                               self.icon_size, COL_ACCENT)
        y = self._blit(display, self._font_text, self.caller_name, COL_WHITE, tx, y)
        y += 6

        if self.loa == 1:
            # The badge IS the suggestion. Marker (star) plus colour, never
            # colour alone -- and it sits on the DRIVER's control, which is what
            # distinguishes advising from taking over.
            # Text first, then the star into the gap the spaces reserve --
            # measuring the FULL string up to the badge is what keeps the glyph
            # off the R. Drawing it from a guessed offset put it on top.
            label = '[A] Answer         RECOMMENDED'
            gap_at = self._font_text.size('[A] Answer   ')[0]
            y_mid = y + self._font_text.get_height() / 2
            self._blit(display, self._font_text, label, COL_ACCENT, tx, y)
            self._draw_star(display, tx + gap_at + self._font_text.get_height() * 0.3,
                            y_mid, self._font_text.get_height() * 0.95,
                            COL_ACCENT)
            y += self._font_text.get_height() + 4
        else:
            y = self._blit(display, self._font_text, '[A] Answer', COL_WHITE, tx, y)
        return self._blit(display, self._font_text, '[B] Decline', COL_WHITE, tx, y)

    def _render_assistant(self, display, x, y):
        """LoA 2-4: the assistant's panel has replaced the phone card."""
        self._draw_handset(display, x + self.icon_size / 2,
                           y + self._font_title.get_height() / 2,
                           self.icon_size, COL_ASSISTANT,
                           waves=self.state in (RINGING, AWAIT_INPUT, COUNTDOWN))
        x = x + self.icon_size + PANEL_ICON_GAP
        self._draw_diamond(display, x + self.icon_size * 0.35,
                           y + self._font_title.get_height() / 2,
                           self.icon_size, COL_ACCENT)
        y = self._blit(display, self._font_title, 'ASSISTANT', COL_ACCENT,
                       x + self.icon_size * 1.05, y)

        spoken = self._panel_line()
        if spoken:
            chunks = self._wrap(spoken, self._font_text,
                                self.rect.width - PANEL_PAD_LEFT - PANEL_PAD)
            for i, chunk in enumerate(chunks):
                if i == 0:
                    chunk = '"' + chunk
                if i == len(chunks) - 1:
                    chunk = chunk + '"'
                y = self._blit(display, self._font_text, chunk, COL_WHITE, x, y)

        if self.state == COUNTDOWN:
            y = self._render_countdown_bar(display, x, y + 4)
            return self._blit(display, self._font_text, '[B] Cancel',
                              COL_COUNTDOWN, x, y)
        if self.loa == 2:
            return self._blit(display, self._font_text, '[A] Yes        [B] No',
                              COL_WHITE, x, y + 4)
        return y

    def _panel_line(self):
        """What the assistant panel says, given the LoA AND the current state.

        Keyed on state, not on LoA alone. Keying on LoA alone meant the LoA 3
        countdown text was drawn in every state, and since ``_countdown_remaining``
        measures from whatever state was last entered, it counted down during the
        ring lead-in, jumped back to 3 when COUNTDOWN was actually entered, and
        then counted down a third time from CONNECTED.
        """
        if self.state == RINGING:
            # The assistant has not spoken yet -- its clip plays on leaving this
            # state -- so the panel shows only who is calling.
            return 'Call from %s.' % self.caller_name
        if self.state == COUNTDOWN:
            return 'Call from %s. Answering in %d…' % (
                self.caller_name, self._countdown_remaining())
        if self.state == CONNECTED:
            return 'Call from %s. Answering now.' % self.caller_name
        line = _ASSISTANT_LINE.get(self.loa)
        return (line % self.caller_name) if line else None

    def _countdown_remaining(self):
        """3, 2, 1 -- one second each, from the start of COUNTDOWN.

        ceil, not int()+1: at exactly t=1.0 s the latter gives 3 again, so the
        first digit held for two seconds and the last for none.
        """
        left = COUNTDOWN_S - self._elapsed(self._now_ms) / 1000.0
        return max(1, min(int(COUNTDOWN_S), int(math.ceil(left))))

    def _render_countdown_bar(self, display, x, y):
        """A draining bar, and the ONLY place in the five renderings one appears."""
        frac = max(0.0, 1.0 - min(1.0, self._elapsed(self._now_ms)
                                  / (COUNTDOWN_S * 1000.0)))
        # (c) Span the PANEL's inner width, taken from the rect -- the caller's
        # x has already been shifted right past the icon, so using it here ran
        # the bar out through the border and over the road.
        bx = self.rect.left + PANEL_PAD_LEFT
        full = self.rect.width - PANEL_PAD_LEFT - PANEL_PAD
        if OUTLINE_PX and self.chrome == 'none':
            pygame.draw.rect(display, COL_OUTLINE,
                             (bx - 1, y - 1, full + 2, 10), 0)
        pygame.draw.rect(display, COL_DIM, (bx, y, full, 8), 1)
        if frac > 0:
            pygame.draw.rect(display, COL_COUNTDOWN,
                             (bx + 1, y + 1, max(0, int((full - 2) * frac)), 6))
        return y + 16

    def _render_status_strip(self, display):
        """The phone, demoted. Present only once the assistant owns the panel."""
        if self.loa < 2:
            return
        y = self.rect.bottom - PANEL_PAD - self._font_small.get_height()
        pygame.draw.line(display, COL_DIM,
                         (self.rect.left + PANEL_PAD_LEFT, y - 6),
                         (self.rect.right - PANEL_PAD, y - 6), 1)
        if self.state == CONNECTED:
            secs = max(0, int(self._elapsed(self._now_ms) / 1000.0))
            text = '%s — connected     %02d:%02d' % (
                self.caller_name, secs // 60, secs % 60)
            colour = COL_CONNECTED
        else:
            text = '%s — ringing, on hold' % self.caller_name
            colour = COL_DIM
        # No icon here. At the small font's ~13 px the receiver loses its
        # silhouette and reads as a smudge; the header already carries it, and
        # the strip's job is the caller's state, which is words.
        display.blit(self._font_small.render(text, True, colour),
                     (self.rect.left + PANEL_PAD_LEFT, y))

    @staticmethod
    def _wrap(text, font, max_w):
        words, lines, cur = text.split(), [], ''
        for word in words:
            trial = (cur + ' ' + word).strip()
            if font.size(trial)[0] <= max_w or not cur:
                cur = trial
            else:
                lines.append(cur)
                cur = word
        if cur:
            lines.append(cur)
        return lines


# --- Standalone harness ------------------------------------------------------
#
# Being a separate module is what lets the five renderings be exercised without
# CARLA, without a second machine, and without a trained checkpoint.
#
#     uv run python src/drive/call_event.py [--fullscreen] [--speed-x-frac 0.20]
#
# Press 0-4 to arm that LoA, A/B to answer, ESC to quit.

class _DemoSpeed(object):
    """Stand-in for ``HUD._render_speed``, reproducing it EXACTLY.

    The panel shares this readout's baseline (§6.1), so a stand-in that merely
    looks speed-ish is worse than none: it would have the placement judged
    against the wrong size and the wrong anchor. Every constant here is copied
    from drive_improved.py and must be re-copied if that file changes --

        speed_pt   = max(28, int(height * 0.055))     # 39 pt at 720p, 59 at 1080p
        unit_pt    = max(12, speed_pt // 3)
        x          = (width - block_w) // 3.5
        y          = int(height * 0.8) - block_h

    NOTE the three places that already disagree about where this lives:
    the ``--speed`` help says "bottom-right corner", the ``_render_speed``
    docstring says "Centered horizontally", and the code puts it ~29% from the
    LEFT. Resolve that when applying §6.1 rather than adding a fourth.
    """

    def __init__(self, dim, x_frac=None):
        width, height = dim
        self.dim = dim
        self.x_frac = x_frac
        font_name = 'courier' if os.name == 'nt' else 'mono'
        fonts = [x for x in pygame.font.get_fonts() if font_name in x]
        default_font = 'ubuntumono'
        mono = default_font if default_font in fonts else (fonts[0] if fonts else None)
        mono = pygame.font.match_font(mono) if mono else None
        speed_pt = max(28, int(height * 0.055))
        self._font = pygame.font.Font(mono, speed_pt)
        self._font_unit = pygame.font.Font(mono, max(12, speed_pt // 3))

    def render(self, display, speed_kmh):
        value = self._font.render('%d' % round(speed_kmh), True, (255, 255, 255))
        unit = self._font_unit.render('km/h', True, (220, 220, 220))
        pad = max(6, int(self.dim[0] * 0.006))
        block_w = value.get_width() + pad + unit.get_width()
        block_h = value.get_height()
        if self.x_frac is None:
            x = int((self.dim[0] - block_w) // 3.5)      # verbatim, drift and all
        else:
            x = int(self.dim[0] * self.x_frac)           # the §6.1 "moved left" trial
        y = int(self.dim[1] * 0.8) - block_h
        display.blit(value, (x, y))
        display.blit(unit, (x + value.get_width() + pad,
                            y + block_h - unit.get_height() - 2))
        return pygame.Rect(x, y, block_w, block_h)


def _demo():
    import argparse

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--fullscreen', action='store_true',
                    help='Run at the desktop resolution. The panel is sized in '
                         'FRACTIONS of the window, so this is the only way to '
                         'see what the participant will actually see.')
    ap.add_argument('--width', type=int, default=1280)
    ap.add_argument('--height', type=int, default=720)
    ap.add_argument('--speed-x-frac', dest='speed_x_frac', type=float, default=None,
                    help='Override the speed readout x as a fraction of width, to '
                         'try the "moved left" placement of docs section 6.1. '
                         'reproduces drive_improved.py verbatim (~0.29).')
    ap.add_argument('--no-speed', action='store_true',
                    help='Hide the speed stand-in. Worth doing once: --speed is '
                         'OFF by default in the drive UI, and if the study does '
                         'not enable it the panel has no baseline to share.')
    args = ap.parse_args()

    pygame.init()
    try:
        pygame.mixer.init()
    except Exception:                                         # noqa: BLE001
        print('[WARN] no audio device; running silently.')

    if args.fullscreen:
        info = pygame.display.Info()
        dim = (info.current_w, info.current_h)
        screen = pygame.display.set_mode(dim, pygame.FULLSCREEN)
    else:
        dim = (args.width, args.height)
        screen = pygame.display.set_mode(dim)
    pygame.display.set_caption('call_event demo -- 0-4 to arm, A / B to answer')

    print('[demo] %dx%d  panel=%s  speed=%s'
          % (dim[0], dim[1],
             'x'.join(str(v) for v in (int(dim[0] * PANEL_WIDTH_FRAC),
                                       int(dim[1] * PANEL_HEIGHT_FRAC))),
             'hidden' if args.no_speed else 'shown'))

    clock = pygame.time.Clock()
    call = CallEvent(dim, onset_offset_s=0.2)
    speedo = None if args.no_speed else _DemoSpeed(dim, args.speed_x_frac)
    hint = pygame.font.Font(pygame.font.get_default_font(), 16)

    running, last, speed = True, None, 120.0
    while running:
        now = pygame.time.get_ticks()
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                running = False
            elif ev.type == pygame.KEYDOWN and ev.key == pygame.K_ESCAPE:
                running = False
            elif (ev.type == pygame.KEYDOWN and ev.unicode in '01234'
                  and not call.active):
                call.arm(now, int(ev.unicode), window_idx=1)
            else:
                call.handle_event(ev, now_ms=now)

        done = call.update(now)
        if done:
            last = done
            print(done)

        # Digits cycle 8 -> 128 so the block-width drift noted above is visible:
        # the readout shifts as it crosses 10 and 100 km/h.
        speed = 8 + (now // 40) % 120

        screen.fill((30, 34, 40))
        if speedo is not None:
            speedo.render(screen, speed)
        call.render(screen)
        screen.blit(hint.render('0-4 arm   A/B answer   ESC quit', True,
                                (120, 120, 120)), (20, 20))
        if last:
            screen.blit(hint.render(str(last), True, (160, 160, 160)), (20, 44))
        pygame.display.flip()
        clock.tick(60)

    call.stop()
    pygame.quit()


if __name__ == '__main__':
    _demo()
