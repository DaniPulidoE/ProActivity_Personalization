"""
Welcome to CARLA manual control.

Control Modes:
    --control test (default) : Only basic driving controls available
    --control full          : All controls available

LoA Popups:
    (default)    : Periodic LoA selection popups; labels are written to
                   data/user_loa_labels.csv
    --no-popup   : Plain drive, no popups and no labels
    --test-popup : Practice mode. The first window opens immediately and they
                   repeat quickly, so the driver can learn the control. Each
                   window holds two consecutive prompts about two different
                   functions, exactly like a real one. Nothing is written to
                   data/user_loa_labels.csv

LoA Popup Function:
    (default)        : Every popup asks about --functionname, one popup per window
    --random-function: Each popup draws its function from the five study functions;
                       a window then holds two prompts about two different ones

LoA Popup Input (which interface answers the popup):
    (default)        : Keyboard, on every rig, wheel attached or not -- the driver
                       says the levels out loud and the experimenter types them
    --keyboard-input : The default, stated explicitly. Number keys 0-4 tick a
                       level, the same number again unticks it, ENTER confirms
    --wheel-input    : Override: paddles move the cursor, the front button ticks
                       the level under it, CONFIRM submits
    Either mode      : a prompt can be dismissed WITHOUT writing a label -- the
                       failsafe for a window the driver cannot answer honestly.
                       On the keyboard, N ticks the INVALID FRAME box and ENTER
                       commits it, exactly like ticking levels; on the wheel it
                       is the NO INPUT row (N there is still a one-press
                       failsafe, in case the rim buttons are unmapped). A missing
                       label is data the analysis simply does not have; a guessed
                       one is data it cannot tell from a real answer.

Basic Controls (available in both modes):
    W            : throttle
    S            : brake
    A/D          : steer left/right
    Q            : toggle reverse
    Space        : hand-brake

    F1           : toggle HUD
    H/?          : toggle help
    ESC          : quit

Additional Controls (only in --control full mode):
    P            : toggle autopilot
    M            : toggle manual transmission
    ,/.          : gear up/down
    CTRL + W     : toggle constant velocity mode at 60 km/h

    L            : toggle next light type
    SHIFT + L    : toggle high beam
    Z/X          : toggle right/left blinker
    I            : toggle interior light

    TAB          : change sensor position
    ` or N       : next sensor
    [1-9]        : change to sensor [1-9]
    G            : toggle radar visualization
    C            : change weather (Shift+C reverse)
    Backspace    : change vehicle

    O            : open/close all doors of vehicle
    T            : toggle vehicle's telemetry

    V            : Select next map layer (Shift+V reverse)
    B            : Load current selected map layer (Shift+B to unload)

    R            : toggle recording images to disk

    CTRL + R     : toggle recording of simulation (replacing any previous)
    CTRL + P     : start replaying last recorded simulation
    CTRL + +     : increments the start time of the replay by 1 second (+SHIFT = 10 seconds)
    CTRL + -     : decrements the start time of the replay by 1 second (+SHIFT = 10 seconds)
"""


# ==============================================================================
# -- imports -------------------------------------------------------------------
# ==============================================================================

import carla

from carla import ColorConverter as cc

import argparse
import collections
import csv
import datetime
import json
import time
import urllib.request
import logging
import math
import os
import random
import re
import sys
import uuid
import weakref

try:
    import pygame
    from pygame.locals import KMOD_CTRL
    from pygame.locals import KMOD_SHIFT
    from pygame.locals import K_0
    from pygame.locals import K_9
    from pygame.locals import K_BACKQUOTE
    from pygame.locals import K_BACKSPACE
    from pygame.locals import K_COMMA
    from pygame.locals import K_DOWN
    from pygame.locals import K_ESCAPE
    from pygame.locals import K_F1
    from pygame.locals import K_KP_ENTER
    from pygame.locals import K_LEFT
    from pygame.locals import K_PERIOD
    from pygame.locals import K_RETURN
    from pygame.locals import K_RIGHT
    from pygame.locals import K_SLASH
    from pygame.locals import K_SPACE
    from pygame.locals import K_TAB
    from pygame.locals import K_UP
    from pygame.locals import K_a
    from pygame.locals import K_b
    from pygame.locals import K_c
    from pygame.locals import K_d
    from pygame.locals import K_f
    from pygame.locals import K_g
    from pygame.locals import K_h
    from pygame.locals import K_i
    from pygame.locals import K_l
    from pygame.locals import K_m
    from pygame.locals import K_n
    from pygame.locals import K_o
    from pygame.locals import K_p
    from pygame.locals import K_q
    from pygame.locals import K_r
    from pygame.locals import K_s
    from pygame.locals import K_t
    from pygame.locals import K_v
    from pygame.locals import K_w
    from pygame.locals import K_x
    from pygame.locals import K_z
    from pygame.locals import K_MINUS
    from pygame.locals import K_EQUALS
except ImportError:
    raise RuntimeError('cannot import pygame, make sure pygame package is installed')

try:
    import numpy as np
except ImportError:
    raise RuntimeError('cannot import numpy, make sure numpy package is installed')

try:
    from .ambience import Ambience, configure_mixer, DEFAULT_AMBIENT_GAIN
    from .call_event import CallEvent
    from .study_session import StudySession, LoASource, FULL_TRIAL, SHORT_TRIAL
except ImportError:
    # Launched as a plain script path rather than -m src.drive.drive_improved,
    # so there is no package context for a relative import.
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from ambience import Ambience, configure_mixer, DEFAULT_AMBIENT_GAIN
    from call_event import CallEvent
    from study_session import StudySession, LoASource, FULL_TRIAL, SHORT_TRIAL

OBJECT_TO_COLOR = [
    (255, 255, 255),
    (128, 64, 128),
    (244, 35, 232),
    (70, 70, 70),
    (102, 102, 156),
    (190, 153, 153),
    (153, 153, 153),
    (250, 170, 30),
    (220, 220, 0),
    (107, 142,  35),
    (152, 251, 152),
    (70, 130, 180),
    (220, 20, 60),
    (255, 0, 0),
    (0, 0, 142),
    (0, 0, 70),
    (0,  60, 100),
    (0,  80, 100),
    (0, 0, 230),
    (119, 11, 32),
    (110, 190, 160),
    (170, 120, 50),
    (55, 90, 80),
    (45, 60, 150),
    (157, 234, 50),
    (81, 0, 81),
    (150, 100, 100),
    (230, 150, 140),
    (180, 165, 180),
]

DEFAULT_DECISION_COLUMNS = [
    'action',
    'level',
    'LoA',
    'message',
    'fcd',
    'probs',
    'profile',
    'fallback',
    'fallback_reason',
]

USER_LOA_LABEL_COLUMNS = [
    'session_id',
    'window_idx',
    # 1-based position of this prompt inside its window. Two prompts share a
    # window_idx and its timestamps, so this (with functionname) is what tells
    # them apart, and it is what any check for order effects between the first
    # and the second answer has to group on.
    'prompt_in_window',
    'window_start_ms',
    'window_end_ms',
    'window_start_timestamp',
    'window_end_timestamp',
    'selection_timestamp',
    'selection_frame',
    'selection_sim_time',
    'selection_speed_kmh',
    'participantid',
    'environment',
    'secondary_task',
    'functionname',
    'emotion',
    'modeltype',
    'state_model',
    'w_fcd',
    # What the participant HEARD during this window. ambient_gain is the gain
    # actually applied, not the one requested: a rig whose mixer failed to open
    # logs 0 and is therefore correctly grouped with the silent runs instead of
    # looking like a noise condition that never happened. Background noise is an
    # arousal manipulation whether or not it is intended as one, and hr_delta /
    # rr_delta are model inputs -- without these columns a mid-study volume
    # change is a confound that cannot even be detected after the fact.
    'ambient_gain',
    'ambient_seed',
    # Which sound: the clip set's short hash when recordings were used, 'synth'
    # when the synthesiser was, 'off' when there was none. Without it a clip
    # swapped halfway through the study is an undetectable change of stimulus --
    # and ambient_seed only reproduces the SYNTHESISED path, so for recordings
    # this is the only thing tying a row to what was actually heard.
    'ambient_source',
    'user_selected_loa',
    'system_action',
    'system_level',
    'system_loa',
    'system_message',
    'system_probs',
    'system_profile',
    'system_fallback',
    'system_fallback_reason',
    'system_fcd',
]

# ==============================================================================
# -- Global functions ----------------------------------------------------------
# ==============================================================================


def _read_csv_headers(path):
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return []
    with open(path, 'r', newline='') as f:
        reader = csv.reader(f)
        return next(reader, [])


def _read_last_csv_row(path):
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return {}
    try:
        with open(path, 'r', newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            last = None
            for row in reader:
                last = row
            return last or {}
    except Exception:
        return {}


def _ensure_csv_columns(path, source_headers, added_headers):
    existing_headers = _read_csv_headers(path)
    headers = existing_headers[:] if existing_headers else source_headers[:]
    for col in added_headers:
        if col not in headers:
            headers.append(col)

    if existing_headers == headers:
        return headers

    rows = []
    if os.path.exists(path) and os.path.getsize(path) > 0:
        with open(path, 'r', newline='') as f:
            rows = list(csv.DictReader(f))

    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, '') for k in headers})
    return headers


def _current_session_id(explicit_session_id=''):
    return (explicit_session_id or os.getenv('PV_SESSION_ID') or f"session_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}")


def _load_latest_system_decision_snapshot(session_id=''):
    """Latest decision ProVoice logged FOR THIS SESSION, or {} if there is none.

    decisions.csv is appended to across runs, so the last line of the file
    usually belongs to an earlier session. Falling back to it whenever this
    session has no decisions yet — which is every row of a --data-collection run,
    and the opening rows of any run — stamps another run's prediction onto these
    labels and invents a system answer that was never made. Rows with a blank
    session_id still count: they come from decisions.csv files written before the
    column existed.
    """
    cwd = os.getcwd()
    path = os.path.join(cwd, 'data', 'decisions.csv')
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return {}
    try:
        with open(path, 'r', newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            last_match = None
            last_any = None
            for row in reader:
                last_any = row
                if session_id and row.get('session_id') not in (None, '', session_id):
                    continue
                last_match = row
            # last_any only when nothing identifies this session in the first place.
            chosen = last_match if session_id else last_any
            if chosen:
                chosen['_source_path'] = path
                return chosen
    except Exception:
        return {}
    return {}

""" Alternative (more efficient - doesn't read the whole decisions.csv file each time). Need to test it before uncommenting it.
def _load_latest_system_decision_snapshot(session_id=''):
    global _decisions_file_offset, _decisions_last_match
    path = os.path.join(os.getcwd(), 'data', 'decisions.csv')

    if not os.path.exists(path):
        return _decisions_last_match

    # If the file shrank (deleted and recreated mid-session), reset.
    if os.path.getsize(path) < _decisions_file_offset:
        _decisions_file_offset = 0

    if os.path.getsize(path) == 0:
        return _decisions_last_match

    try:
        with open(path, 'r', newline='', encoding='utf-8') as f:
            # Always read the header from byte 0 — cheap (one line).
            header_reader = csv.DictReader(f)
            fieldnames = header_reader.fieldnames   # advances f to just after header
            if not fieldnames:
                return _decisions_last_match

            # Seek to where we stopped last time.
            # On first call _decisions_file_offset == 0, so f is already
            # positioned just after the header — the seek is skipped.
            if _decisions_file_offset > 0:
                f.seek(_decisions_file_offset)

            # Second DictReader reuses the open file but gets fieldnames
            # explicitly so it doesn't try to re-read the header line.
            for row in csv.DictReader(f, fieldnames=fieldnames):
                if session_id and row.get('session_id') not in (None, '', session_id):
                    continue
                _decisions_last_match = row   # keep updating — we want the last one

            _decisions_file_offset = f.tell()  # save position for next call

    except Exception:
        _decisions_file_offset = 0  # corrupted state — start over next call

    return _decisions_last_match
"""

def _normalize_csv_value(value):
    if isinstance(value, (dict, list, tuple)):
        try:
            return json.dumps(value, ensure_ascii=False)
        except Exception:
            return str(value)
    if value is None:
        return ''
    return value


# Set once at startup by --test-popup (see main()). Practice answers teach the
# control and are not data: a participant's first attempts at a UI they have
# never seen would otherwise enter the training set as if they were considered
# preferences, and nothing downstream could tell them apart -- the rows carry no
# marker for "this was practice", and by design should not gain one.
#
# The guard lives HERE, in the only function that writes the file, rather than
# only at the call site: "these answers are never logged" is the entire promise
# of the mode, and a promise enforced by every caller remembering to check is
# one restatement of the popup loop away from being broken.
_USER_LOA_LOGGING_DISABLED_REASON = None


def disable_user_loa_logging(reason):
    """Turn off ALL writes to data/user_loa_labels.csv for this process."""
    global _USER_LOA_LOGGING_DISABLED_REASON
    _USER_LOA_LOGGING_DISABLED_REASON = reason


def append_user_loa_selection(selection_row):
    if _USER_LOA_LOGGING_DISABLED_REASON:
        print('[INFO] LoA selection NOT written to data/user_loa_labels.csv (%s)'
              % _USER_LOA_LOGGING_DISABLED_REASON)
        return
    cwd = os.getcwd()
    labels_path = os.path.join(cwd, 'data', 'user_loa_labels.csv')
    source_headers = _read_csv_headers(labels_path)
    if not source_headers:
        source_headers = USER_LOA_LABEL_COLUMNS[:]

    headers = _ensure_csv_columns(labels_path, source_headers, USER_LOA_LABEL_COLUMNS)
    row = {k: '' for k in headers}
    for key, value in (selection_row or {}).items():
        if key in row:
            row[key] = _normalize_csv_value(value)

    with open(labels_path, 'a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writerow(row)


def _set_world_frozen(world, frozen):
    """Freeze/unfreeze the whole scene while the LoA popup is open.

    Two mechanisms, and the good one needs --sync.

    WITH --sync AND a clock-pause file: ask the clock owner
    (fixed_npc_traffic.py) to stop ticking. That is a true pause -- every vehicle
    holds its exact position, heading, speed and wheel angle, and resumes from
    there, because from the simulation's point of view no time passed. Drive
    cannot do this alone: it is not the clock owner, so it asks by touching a
    file the owner polls.

    OTHERWISE (async, or no pause file): hold each vehicle at zero velocity, as
    below. This is a poor substitute and the live run showed why -- traffic stops
    dead at every popup and pulls away from rest afterwards, which reads as
    obviously unnatural. It is kept because the free-running clock offers nothing
    better: there is no tick to withhold.

    We deliberately do NOT toggle ``set_simulate_physics`` here: disabling and
    re-enabling a vehicle's physics resets its drivetrain state, so the ego
    comes back gutless (sluggish acceleration, lower top speed). Instead we use
    CARLA's constant-velocity hold, which pins each vehicle at 0 m/s server-side
    every tick while leaving the engine, gearbox and wheels fully alive. On
    resume we release the hold and restore the EGO's saved velocity, so the
    participant keeps the speed they had before the popup and accelerates exactly
    as before. No-ops harmlessly if the world/actor query fails.

    Velocity is restored for the ego ONLY, and this asymmetry is deliberate --
    restoring it on the NPCs was making them crash. Two reasons:

    * set_target_velocity() injects a world-frame velocity vector directly into
      the rigid body. After a hold the wheels have spun down to rest, so a car
      handed back 12 m/s has tires at zero angular velocity: full slip, one
      frame, in whatever direction it was previously travelling. On a car that
      was mid-corner, that direction no longer matches its heading. The result
      looks exactly like a vehicle that entered a turn far too fast and lost the
      back end -- and it was happening to all eleven NPCs every 20 s, which is
      why the crashes seemed constant. The ego escapes this because the
      participant is steering it and it is under a wheel/pedal control loop, not
      a waypoint follower.
    * The CARLA docs warn outright that enabling a constant velocity on a
      traffic-manager vehicle "may cause conflicts", because it overrides the
      TM's own velocity changes. Releasing the NPCs at rest and letting the TM
      accelerate them normally keeps that override as brief as possible.

    Angular velocity is zeroed at freeze time for the same reason: the constant-
    velocity hold pins LINEAR velocity only, so a vehicle caught mid-turn keeps
    yawing on the spot for the whole popup and can be facing well out of its lane
    by the time it is released.
    """
    if world is None or getattr(world, 'world', None) is None:
        return

    pause_file = getattr(world, 'clock_pause_file', '') or ''
    if pause_file:
        # Works under BOTH clocks, which is why there is no --sync check here.
        # fixed_npc_traffic.py holds a synchronous world by not ticking it, and a
        # free-running one by switching it into synchronous mode and not ticking
        # it either. Drive only ever asks; it does not need to know which.
        _request_clock_pause(pause_file, frozen)
        return

    try:
        vehicles = world.world.get_actors().filter('vehicle.*')
    except Exception:
        return

    ego_id = getattr(getattr(world, 'player', None), 'id', None)
    zero = carla.Vector3D(0, 0, 0)

    if frozen:
        saved = {}
        for actor in vehicles:
            try:
                if actor.id == ego_id:
                    saved[actor.id] = (actor.get_velocity(),
                                       actor.get_angular_velocity())
                actor.enable_constant_velocity(zero)
                actor.set_target_angular_velocity(zero)
            except Exception:
                pass
        world._frozen_velocities = saved
    else:
        saved = getattr(world, '_frozen_velocities', {})
        for actor in vehicles:
            try:
                actor.disable_constant_velocity()
                vel, ang = saved.get(actor.id, (None, None))
                if vel is not None:
                    actor.set_target_velocity(vel)
                    actor.set_target_angular_velocity(ang)
            except Exception:
                pass
        world._frozen_velocities = {}


def _request_clock_pause(pause_file, paused):
    """Ask the --sync clock owner to hold (or release) the simulation.

    The whole protocol is "does this file exist", which is enough because the
    question has exactly two states and one writer. The file is REWRITTEN each
    time rather than merely created: the owner ignores a request older than a few
    minutes, so a Drive that dies mid-popup cannot leave the rig frozen, and the
    mtime is what that check reads. The contents are for a human reading the file
    during a debug session -- nothing parses them.
    """
    try:
        if paused:
            with open(pause_file, 'w') as f:
                f.write(datetime.datetime.now().isoformat())
        else:
            try:
                os.remove(pause_file)
            except FileNotFoundError:
                pass
    except OSError as e:
        # Falling back would mean silently reintroducing the stop-dead-and-pull-
        # away behaviour this replaces, so say so rather than degrade quietly.
        print("[WARN] Could not %s the clock-pause flag %s: %s. The scene will "
              "NOT freeze for this popup."
              % ('set' if paused else 'clear', pause_file, e))


def _drive_is_holding_clock(args, loa_popup, end_overlay):
    """True when Drive has asked the --sync clock owner to STOP ticking.

    Drive must not then wait for a tick: it is the reason there is no tick. That
    wait would block the loop for the full stall timeout on every frame, so the
    popup the pause exists to serve would become unresponsive.

    Only the clock-pause path counts. Without a pause file the freeze is done by
    holding vehicles at zero velocity and the clock keeps running normally, so
    Drive should keep pacing to it.
    """
    if not (getattr(args, 'sync', False)
            and getattr(args, 'clock_pause_file', '')):
        return False
    return end_overlay is not None or loa_popup.active


# How long to wait for the world clock before deciding it has stalled. Generous
# against a 0.05 s step (100 missed ticks) because the only thing that should
# ever trip it is the clock owner dying, not a slow frame.
SYNC_STALL_TIMEOUT_S = 5.0


def _await_world_tick(sim_world, timeout_s=SYNC_STALL_TIMEOUT_S):
    """Block until the world-clock owner advances the simulation. True if it did.

    Under --sync Drive does not tick -- src/drive/fixed_npc_traffic.py does, and
    that file's docstring explains why it has to be that process. Drive instead
    paces itself to the clock here, which is what makes one loop iteration equal
    one simulation step: exactly one control application per tick, and no frame
    rendered twice from an unchanged world.

    The wait is BOUNDED because the failure it guards against is total. If the
    clock owner dies, an unbounded wait leaves the pygame window frozen and
    unresponsive with a participant sitting in front of it, mid-study. On timeout
    we return False and let the caller keep looping: input still works, the HUD
    still draws, and the driver can end the session cleanly.
    """
    try:
        sim_world.wait_for_tick(timeout_s)
        return True
    except RuntimeError:
        return False


# --- LoA selection via the steering wheel -------------------------------------
# The paddle shifters move the highlight and a front button confirms. Button
# indices are device-specific, so run scripts/map_wheel_buttons.py once and paste
# its output here. A D-pad is also accepted if the wheel has one, but this rig's
# G25 has no hat on the rim, so the paddles are the real navigation.
#
# The highlight deliberately starts on NO option — a pre-selected default would
# anchor the driver's answer — and does not wrap around, so repeated presses
# cannot jump from 0 straight to 4.
LOA_WHEEL_BUTTON_CONFIRM = 6
LOA_WHEEL_BUTTON_PREV = 5
LOA_WHEEL_BUTTON_NEXT = 4

# The other front button closes the simulation, exactly like the window's X.
# Unlike the three above, this is honoured everywhere — start screen, driving,
# and during an LoA prompt — not only while a selection is open.
WHEEL_BUTTON_QUIT = 7

# WHILE DRIVING ONLY (KeyboardControl.parse_events — not the start screen or
# the LoA popup, which are single-press by design), button 7 needs a SECOND
# press within this window to actually end the session. A single stray hit
# next to the paddles used for steering/LoA input is now cheap; ending the
# whole study by accident is not. Confirmed elsewhere (start screen, LoA
# popup) stays a single press: those are deliberate, momentary screens, not
# the hours of ordinary driving where an accidental touch is the risk.
WHEEL_QUIT_CONFIRM_WINDOW_MS = 1000

# Index == the LoA value that gets logged.
LOA_LABELS = (
    '1: No assistive action is taken',
    '2: Give Suggestion',
    '3: Ask for user confirmation',
    '4: Execute automatically but user can veto',
    '5: Fully automatic',
)

# Cursor position of the trailing "confirm" row. The driver may mark SEVERAL
# acceptable LoAs, so the confirm button has to both toggle and submit; parking
# the cursor on a dedicated row separates the two without a second button.
CONFIRM_ROW = len(LOA_LABELS)

# ...and of the "no input" row below it: dismiss this prompt WITHOUT writing a
# label. The failsafe for a window the driver cannot honestly answer -- they
# were mid-manoeuvre when the scene froze, they missed what happened, the
# prompt caught them looking away. The alternative is not "no data": it is a
# guessed label, indistinguishable from a considered one in
# user_loa_labels.csv and quietly wrong in the training set. A dropped window
# costs ~20 s of one session; a fabricated label is in the data forever.
#
# LAST in the cursor order on purpose. It sits past CONFIRM, so reaching it is
# deliberate -- the driver who overshoots CONFIRM lands here and can paddle
# straight back, where an entry BEFORE the LoA rows would be one stray paddle
# press away from discarding a real answer.
NO_INPUT_ROW = CONFIRM_ROW + 1


# --- Which interface answers the popup ----------------------------------------
# The two are experiment conditions, not a preference: 'wheel' keeps the hands on
# the rim (cursor + confirm row, as above), 'keyboard' addresses the levels
# directly by number and needs no cursor at all.
#
# The keyboard is the DEFAULT for every run, whether or not a wheel is bound: the
# study asks the driver to say the levels out loud and the experimenter types
# them, so the interface must not change with the rig it happens to run on. Pass
# --wheel-input to override it.
POPUP_INPUT_WHEEL = 'wheel'
POPUP_INPUT_KEYBOARD = 'keyboard'
POPUP_INPUT_DEFAULT = POPUP_INPUT_KEYBOARD


# --- Function pool for --random-function ---------------------------------------
# The five functions this data-collection run covers, drawn per prompt instead of
# fixing one for the whole session with --functionname.
#
# The spellings are NOT free text: they have to match src/ProVoice/fcd_config.py
# exactly. A paraphrase like 'Send a message' or 'Change music' resolves to
# fcd_config.UNKNOWN_FUNCTION_KEY and is scored with a NEUTRAL FCD vector (all
# 3s) rather than that function's real one, so the segment carries no task
# context. It warns once per distinct bad name ('[fcd][warn] ...') and the
# 'Unknown function' string reaches the decisions.csv `profile` column, so this
# is now detectable after a run instead of silent — but the segment is still
# wasted. Check the console for [fcd][warn] before trusting a session.
RANDOM_FUNCTION_POOL = (
    'Respond to a text message',
    'Respond to a phone call',
    'Provide weather update',
    'Provide traffic news',
    'Change song',
)

# Prompts per window once the pool is in play: two labels for the same 20 s of
# driving instead of one, each asking about a different function. Drawing without
# replacement is what keeps the pair distinct.
PROMPTS_PER_WINDOW = 2


# --- Waiting for ProVoice before the first window -----------------------------
# ProVoice needs the better part of a minute to load its models and start
# logging, while Drive's popup clock would otherwise start the moment the driver
# presses START. Windows opened in that gap have no driver-state frames behind
# them, and scripts/build_loa_dataset.py drops their labels — in one measured run
# that was half the session's labels.
#
# So the clock starts at ProVoice's FIRST logged frame instead. The first popup
# then follows one full interval later, which means the 20 s the driver is asked
# about is covered by data end to end.
PROVOICE_RAW_LOG = os.path.join('data', 'raw_data.jsonl')
POPUP_WAIT_TIMEOUT_S = 180.0


def _last_logged_session_id(path, tail_bytes=65536):
    """session_id on the last complete line of a JSONL log, or None.

    Only the tail is read: the log grows to tens of MB over a session and this
    runs every second. ProVoice appends in order, so once it is writing for this
    session its rows are the newest ones and the last line is enough.
    """
    try:
        size = os.path.getsize(path)
        with open(path, 'rb') as f:
            if size > tail_bytes:
                f.seek(size - tail_bytes)
                f.readline()          # drop the partial line the seek landed in
            chunk = f.read()
    except OSError:
        return None
    for raw in reversed(chunk.splitlines()):
        raw = raw.strip()
        if not raw:
            continue
        try:
            record = json.loads(raw.decode('utf-8', 'replace'))
        except Exception:
            continue                  # half-written line; try the one before it
        return (record.get('session_id') or '').strip()
    return None


# --- The same two facts, when ProVoice runs on another machine ----------------
# raw_data.jsonl is written on the ProVoice machine, so neither "it started
# logging" nor "it finished" can be READ here. scripts/provoice_status_server.py
# receives both as signals and publishes them into this file; the two watchers
# below read it instead of the log. A file rather than an HTTP poll on purpose:
# this runs inside a 60 Hz render loop, where a blocking socket read is a
# dropped frame for the participant.
PROVOICE_STATUS_FILE = 'provoice_status.json'


def _read_status_file(path):
    """The published status record, or None while there is nothing to read.

    Every failure mode collapses to None -- missing (the server has not started
    yet), unparseable (a read that raced the atomic replace, which os.replace
    makes vanishingly unlikely but not impossible on Windows), or unreadable.
    None always means "no signal yet", never "signal lost", so a bad read costs
    one poll rather than the channel.
    """
    try:
        with open(path, 'r', encoding='utf-8') as f:
            record = json.load(f)
    except (OSError, ValueError):
        return None
    return record if isinstance(record, dict) else None


def _post_drive_ended(status_url, session_id, participantid, reason=""):
    """Tell ProVoice the block is over, so it stops with us.

    ONE request at the end of a block, so urllib is right here -- the
    connection-reuse machinery on the ProVoice side exists for its 4 Hz decision
    feed, not for this.

    Failure is logged and swallowed. Drive's job is the participant's block, and
    it must not hang on a shutdown courtesy; the visible cost is a ProVoice that
    has to be stopped by hand, which is what happened before this existed.
    """
    if not status_url:
        return False
    endpoint = status_url.rstrip("/") + "/event"
    body = json.dumps({
        "event": "drive_ended",
        "session_id": session_id or "",
        "participantid": participantid or "",
        "reason": reason or "study block complete",
        "ts": time.time(),
    }).encode("utf-8")
    try:
        req = urllib.request.Request(endpoint, data=body, method="POST",
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=3.0) as resp:
            resp.read()
            print("[study] told ProVoice the block is over (%s)." % endpoint)
            return True
    except Exception as exc:                                  # noqa: BLE001
        print("[study] could NOT tell ProVoice the block is over (%s: %s). "
              "Stop it by hand." % (type(exc).__name__, exc))
        return False


def _status_event_ts(path, session_id, key):
    """Timestamp of `key` in the status file, if it is for THIS session.

    The session check is what stops a status file left by the previous
    participant from ending this drive the moment it starts. The server scopes
    itself the same way; this is the second half of that guard, on the side
    that acts on it.
    """
    record = _read_status_file(path)
    if not record:
        return None
    if session_id and (record.get('session_id') or '').strip() != session_id:
        return None
    return record.get(key)


class ProVoiceReadyWatcher(object):
    """Polls until ProVoice starts recording for this session.

    Two sources, one meaning. Locally that is the first line of raw_data.jsonl;
    with ``status_path`` (a --remote run) it is the collection_started signal
    the ProVoice machine posted. Same poll cadence, same 'ready'/'timeout'
    contract, so the game loop does not know which one it is waiting on.
    """

    def __init__(self, session_id, timeout_s, poll_s=1.0, path=None,
                 status_path=None):
        self.session_id = (session_id or '').strip()
        self.timeout_ms = max(0.0, float(timeout_s)) * 1000.0
        self.poll_ms = poll_s * 1000.0
        self.status_path = status_path
        self.path = path or os.path.join(os.getcwd(), PROVOICE_RAW_LOG)
        self.started_ms = None
        self._next_poll_ms = 0

    def start(self, now_ms):
        self.started_ms = now_ms
        self._next_poll_ms = now_ms

    def waited_s(self, now_ms):
        return 0.0 if self.started_ms is None else (now_ms - self.started_ms) / 1000.0

    def _is_recording(self):
        if self.status_path:
            return _status_event_ts(self.status_path, self.session_id,
                                    'collection_started_ts') is not None
        return bool(self.session_id) and \
            _last_logged_session_id(self.path) == self.session_id

    def poll(self, now_ms):
        """'waiting' | 'ready' | 'timeout'. Cheap between polls."""
        if self.started_ms is None or now_ms < self._next_poll_ms:
            return 'waiting'
        self._next_poll_ms = now_ms + self.poll_ms
        if self._is_recording():
            return 'ready'
        if self.timeout_ms and (now_ms - self.started_ms) >= self.timeout_ms:
            return 'timeout'
        return 'waiting'


class ProVoiceEndWatcher(object):
    """Polls the status file for 'ProVoice has exited' (--remote runs only).

    Local runs need nothing like this: start_experiment.py sees ProVoice's own
    process end and stops the session. Across two machines that exit is
    invisible here, and the participant would otherwise keep driving -- and
    keep answering LoA popups -- with nothing recording the frames those labels
    describe.

    Polled the whole session, not just at the end: ProVoice ending EARLY (a
    crash, an operator stop) is exactly the case worth catching, and it is the
    one that looks normal from the driver's seat.
    """

    def __init__(self, session_id, status_path, poll_s=1.0):
        self.session_id = (session_id or '').strip()
        self.status_path = status_path
        self.poll_ms = poll_s * 1000.0
        self._next_poll_ms = 0
        self.ended = False

    def poll(self, now_ms):
        if self.ended or not self.status_path or now_ms < self._next_poll_ms:
            return self.ended
        self._next_poll_ms = now_ms + self.poll_ms
        self.ended = _status_event_ts(self.status_path, self.session_id,
                                      'ended_ts') is not None
        return self.ended

    def reason(self):
        record = _read_status_file(self.status_path) or {}
        return (record.get('ended_reason') or '').strip()


# --- Practice popups (--test-popup) -------------------------------------------
# Teaching the control is repetition, not a driving window: the first prompt
# opens the moment the session starts and the rest follow this much later, so a
# participant can run through several attempts in under a minute.
TEST_POPUP_INTERVAL_S = 8

# Extra hold applied ON TOP of the first LoA window's 20 s wait, remote runs
# only, once the remote ProVoice's collection_started signal is actually
# received (not on the --popup-wait-timeout fallback). The rPPG pipeline's
# first heart-rate reading isn't available until ~6 s into recording; the
# other 4 s is slack for missed/dropped frames. Without this hold the first
# window's early frames would have no hr_delta/rr_delta yet.
PROVOICE_READY_POPUP_DELAY_S = 10


class LoASelectionPopup(object):
    def __init__(self, width, height, interval_seconds=20, enabled=True,
                 open_immediately=False, input_mode=POPUP_INPUT_KEYBOARD,
                 function_name='', function_pool=()):
        self.enabled = enabled
        self.open_immediately = open_immediately
        self.interval_ms = int(interval_seconds * 1000)
        self.next_prompt_ms = 0
        self.active = False
        self.prompt_started_ms = 0
        self.session_started_ms = 0
        self.session_started_walltime = None
        self.window_idx = 0
        self.window_start_ms = 0
        self.window_end_ms = 0
        self.window_start_timestamp = ''
        self.window_end_timestamp = ''
        self.width = width
        self.height = height
        # Cursor row (None until the driver moves it, so nothing is anchored),
        # and the set of LoAs marked acceptable — more than one is allowed.
        self.selected = None
        self.chosen = set()
        # Keyboard mode's discard, ticked with N and committed with ENTER like
        # every other answer. Mutually exclusive with `chosen`: a window is
        # either answerable or it is not.
        self.invalid_frame = False
        self.input_mode = input_mode
        # The vehicle function the levels refer to. Shown under the title so the
        # driver answers about the right one; the line is dropped entirely when
        # no name is available. With a pool (--random-function) this holds the
        # function of the prompt currently on screen, redrawn every window;
        # without one it stays the fixed --functionname for the whole session.
        self.fixed_function_name = (function_name or '').strip()
        self.function_name = self.fixed_function_name
        self.function_pool = tuple(function_pool)
        # One prompt per window with a fixed function — a second would ask the
        # identical question about the identical 20 s. The pool is what makes a
        # second prompt worth showing, so it is also what turns it on.
        self.prompts_per_window = (min(PROMPTS_PER_WINDOW, len(self.function_pool))
                                   if self.function_pool else 1)
        self.prompt_in_window = 0
        self._window_functions = ()
        # Paddle/confirm indices are device-specific; without them the wheel
        # cannot drive the popup and we fall back to the keyboard.
        self.wheel_mapped = (LOA_WHEEL_BUTTON_CONFIRM is not None
                             and LOA_WHEEL_BUTTON_PREV is not None
                             and LOA_WHEEL_BUTTON_NEXT is not None)
        self._title_font = pygame.font.Font(pygame.font.get_default_font(), 30)
        self._text_font = pygame.font.Font(pygame.font.get_default_font(), 24)
        self._small_font = pygame.font.Font(pygame.font.get_default_font(), 20)

    def start(self):
        # With data collection off there is nothing to label. Leaving
        # next_prompt_ms at 0 keeps should_open() False for the rest of the
        # session, so the scene is never frozen and no rows are ever written.
        if not self.enabled:
            return
        self.session_started_ms = pygame.time.get_ticks()
        self.session_started_walltime = datetime.datetime.now()
        self.window_idx = 0
        # max(1, ...) because should_open() treats 0 as "never scheduled", which
        # open_immediately would otherwise hit if start() landed on tick 0.
        delay = 0 if self.open_immediately else self.interval_ms
        self.next_prompt_ms = max(1, self.session_started_ms + delay)

    def should_open(self, now_ms):
        return (not self.active) and self.next_prompt_ms and now_ms >= self.next_prompt_ms

    def open(self, now_ms):
        """Open the first prompt of a new window."""
        self.window_idx += 1
        self.prompt_in_window = 0
        self.window_start_ms = max(self.session_started_ms, now_ms - self.interval_ms)
        self.window_end_ms = now_ms
        if self.session_started_walltime is not None:
            elapsed = now_ms - self.session_started_ms
            start_wall = self.session_started_walltime + datetime.timedelta(milliseconds=self.window_start_ms - self.session_started_ms)
            end_wall = self.session_started_walltime + datetime.timedelta(milliseconds=self.window_end_ms - self.session_started_ms)
            self.window_start_timestamp = start_wall.isoformat()
            self.window_end_timestamp = end_wall.isoformat()
        if self.function_pool:
            # sample() draws WITHOUT replacement, which is what guarantees the
            # two prompts of a window never ask about the same function.
            self._window_functions = tuple(
                random.sample(self.function_pool, self.prompts_per_window))
        else:
            self._window_functions = (self.fixed_function_name,) * self.prompts_per_window
        self._open_prompt(now_ms)

    def advance(self, now_ms):
        """Open the next prompt of the SAME window; False when the window is done.

        The window fields are deliberately left untouched: both prompts label the
        same 20 s of driving and only differ in the function they ask about, so
        they belong to one window_idx and share its start/end timestamps.
        """
        if self.prompt_in_window >= self.prompts_per_window:
            return False
        self._open_prompt(now_ms)
        return True

    def _open_prompt(self, now_ms):
        self.active = True
        # Never carry a choice over from the previous prompt.
        self.selected = None
        self.chosen = set()
        self.invalid_frame = False
        self.prompt_started_ms = now_ms
        self.function_name = self._window_functions[self.prompt_in_window]
        self.prompt_in_window += 1

    def close(self, now_ms):
        self.active = False
        self.next_prompt_ms = now_ms + self.interval_ms

    def _move(self, direction):
        """Move the cursor, entering the list from whichever end was pressed.

        The cursor spans the five LoA rows plus the trailing CONFIRM and NO
        INPUT rows, because the wheel only has one free front button: it
        toggles the LoA under the cursor, and acts when the cursor is parked on
        one of the two trailing rows.
        """
        if self.selected is None:
            # Enter on an actual LoA row from either end. Landing on CONFIRM
            # first would put the cursor on a row that does nothing until
            # something is ticked, which reads as an unresponsive control —
            # and entering on NO INPUT would put the discard under the very
            # first press.
            self.selected = 0 if direction > 0 else len(LOA_LABELS) - 1
        else:
            self.selected = max(0, min(NO_INPUT_ROW, self.selected + direction))

    def _toggle(self, loa):
        if loa in self.chosen:
            self.chosen.discard(loa)
        else:
            self.chosen.add(loa)
            # A level and "invalid window" are contradictory answers about the
            # same 20 s, so the later press wins rather than leaving both ticked
            # and ENTER having to guess which one was meant.
            self.invalid_frame = False

    def _toggle_invalid(self):
        self.invalid_frame = not self.invalid_frame
        if self.invalid_frame:
            self.chosen.clear()

    def _activate(self):
        """Act on the cursor row.

        Returns the (action, payload) pair handle_event hands back, or None
        when the press only changed state (ticking a level, or a CONFIRM with
        nothing ticked).
        """
        if self.selected is None:
            return None
        if self.selected == NO_INPUT_ROW:
            # Discard: whatever was ticked goes with it, deliberately. Half an
            # answer from a window the driver could not follow is exactly the
            # thing this row exists to keep out of the data.
            return 'skip', None
        if self.selected == CONFIRM_ROW:
            # Submitting nothing would write an empty label, so it is a no-op.
            # NO INPUT is the way to answer nothing on purpose, and it says so.
            return ('select', sorted(self.chosen)) if self.chosen else None
        self._toggle(self.selected)
        return None

    def handle_event(self, event):
        if event.type == pygame.QUIT:
            return 'quit', None

        # The number keys stay live in wheel mode too: they are the fallback if
        # the rim buttons turn out to be unmapped or the wheel drops out
        # mid-session, which would otherwise leave the popup unanswerable and
        # the scene frozen for good.
        if event.type == pygame.KEYDOWN:
            if event.key == K_ESCAPE:
                return 'quit', None
            if event.unicode in ('1', '2', '3', '4', '5'):
                # Number keys toggle rather than submit — otherwise a second LoA
                # could never be added, and pressing the same number again is
                # how a level is taken back off the list.
                self._toggle(int(event.unicode)-1)
                return None, None
            if event.key in (K_RETURN, K_KP_ENTER):
                # One key commits whatever is ticked: the levels as a label, or
                # the invalid-window row as a discard. Nothing ticked is a no-op,
                # because submitting nothing would write an empty label.
                if self.invalid_frame:
                    return 'skip', None
                if self.chosen:
                    return 'select', sorted(self.chosen)
                return None, None
            if event.key == K_n:
                if self.input_mode != POPUP_INPUT_WHEEL:
                    # Ticks the row the driver can see and commits on ENTER, so
                    # the destructive answer takes the same two steps as a real
                    # one and can be taken back before it lands.
                    self._toggle_invalid()
                    return None, None
                # In wheel mode N stays a one-press failsafe: it exists for the
                # case where the rim buttons are unmapped or the wheel drops out
                # mid-session, and a failsafe that needs a second confirmation
                # on a screen showing no invalid-frame row is a worse one.
                return 'skip', None
            if self.input_mode == POPUP_INPUT_KEYBOARD:
                # Numbers and ENTER are the entire interface here; the cursor
                # exists only to give the wheel's single button something to
                # point at, so leave it parked on None.
                return None, None
            if event.key in (K_LEFT, K_a):
                self._move(-1)
            elif event.key in (K_RIGHT, K_d):
                self._move(1)
            elif event.key == K_SPACE:
                acted = self._activate()
                if acted:
                    return acted
            return None, None

        # Quitting from the rim works in both modes, exactly as it does on the
        # start screen and while driving.
        if (event.type == pygame.JOYBUTTONDOWN
                and WHEEL_BUTTON_QUIT is not None
                and event.button == WHEEL_BUTTON_QUIT):
            return 'quit', None

        if self.input_mode != POPUP_INPUT_WHEEL:
            return None, None

        if event.type == pygame.JOYHATMOTION:
            # value is (x, y); x returns to 0 on release, which we ignore.
            x = event.value[0]
            if x:
                self._move(1 if x > 0 else -1)
            return None, None

        if event.type == pygame.JOYBUTTONDOWN:
            if LOA_WHEEL_BUTTON_PREV is not None and event.button == LOA_WHEEL_BUTTON_PREV:
                self._move(-1)          # left paddle -> lower LoA
            elif LOA_WHEEL_BUTTON_NEXT is not None and event.button == LOA_WHEEL_BUTTON_NEXT:
                self._move(1)           # right paddle -> higher LoA
            elif (LOA_WHEEL_BUTTON_CONFIRM is not None
                    and event.button == LOA_WHEEL_BUTTON_CONFIRM):
                acted = self._activate()
                if acted:
                    return acted

        return None, None

    def render(self, display):
        overlay = pygame.Surface((self.width, self.height))
        overlay.set_alpha(200)
        overlay.fill((0, 0, 0))
        display.blit(overlay, (0, 0))

        # Keyboard mode shows no key hint at all: there the keys are the
        # experimenter's, not the driver's, and the CONFIRM and NO INPUT rows
        # below already name ENTER and N for whoever is typing.
        hint = ''
        if self.input_mode == POPUP_INPUT_WHEEL:
            if self.wheel_mapped:
                hint = ('Paddles move  -  front button ticks a level  -  '
                        'then pick CONFIRM (or NO INPUT)')
            else:
                # Unmapped buttons would make the whole wheel feel dead; say so
                # rather than leaving the driver pressing an inert control.
                hint = ('Wheel buttons not mapped - run scripts/map_wheel_buttons.py. '
                        'Use number keys 0-4 and ENTER, or N for no input.')

        # Keyboard mode is the spoken-answer setup: the driver answers out loud
        # and the experimenter types it. 'Read out' rather than 'say' on purpose
        # — it points at the numbered list below, so the driver utters the
        # numbers the experimenter has to type instead of paraphrasing ('the
        # veto one'), which is where transcription errors come from.
        instruction = ('Read out EVERY Level of Proactivity you would accept for '
                       'the last 20 seconds.'
                       if self.input_mode != POPUP_INPUT_WHEEL
                       else 'Mark EVERY Level of Proactivity you would accept '
                            'for the last 20 seconds.')

        header = [
            (self._title_font, 'Level of Proactivity Selection', (255, 255, 255)),
        ]
        if self.function_name:
            # Blue, not the yellow/green used for the cursor and the ticks: this
            # line is context, never something that can be selected.
            header.append(
                (self._text_font, 'Function: %s' % self.function_name.upper(), (140, 200, 255)))
        header.append((self._text_font, instruction, (255, 255, 255)))
        if hint:
            header.append((self._text_font, hint, (255, 255, 255)))

        # Each header line beyond the original three would push the confirm row
        # toward the bottom edge of a 720p window, so the block rises half a line
        # per extra one — and drops half a line per missing one — keeping the
        # footprint it had.
        y = int(self.height * 0.32) - 22 * (len(header) - 3)
        for font, text, colour in header:
            surface = font.render(text, True, colour)
            rect = surface.get_rect(center=(self.width // 2, y))
            display.blit(surface, rect)
            y += 45

        # separate header from LoA's
        text ="- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -"
        colour = (255, 170, 90) if self.invalid_frame else (140, 140, 140)
        surface = self._small_font.render(text, True, colour)
        rect = surface.get_rect(center=(self.width // 2, y))
        display.blit(surface, rect)
        y+=45

        # Markers as well as colour, so the state survives a projector or a
        # colour-blind participant: [x] is ticked, '>' is the cursor.
        for idx, label in enumerate(LOA_LABELS):
            under_cursor = (idx == self.selected)
            ticked = idx in self.chosen
            text = '%s %s' % ('[x]' if ticked else '[ ]', label)
            if under_cursor:
                text = '> %s <' % text
            if ticked:
                colour = (120, 240, 150)
            elif under_cursor:
                colour = (255, 220, 0)
            else:
                colour = (255, 255, 255)
            surface = self._small_font.render(text, True, colour)
            rect = surface.get_rect(center=(self.width // 2, y))
            display.blit(surface, rect)
            y += 45

        # The CONFIRM and NO INPUT rows exist for the wheel: they are the two
        # rows its single front button parks on, so without them the wheel could
        # neither submit nor discard. Keyboard mode addresses ENTER and N
        # directly and never draws a cursor, so the rows would be two lines of
        # instruction aimed at the experimenter on a screen the participant is
        # reading — it gets the invalid-frame tick box instead.
        if self.input_mode != POPUP_INPUT_WHEEL:
            # separator
            text ="- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -"
            colour = (255, 170, 90) if self.invalid_frame else (140, 140, 140)
            surface = self._small_font.render(text, True, colour)
            rect = surface.get_rect(center=(self.width // 2, y))
            display.blit(surface, rect)
            y+=45
            
            # no input with a [x] option for selection
            text ="%s Invalid window" % ('[x]' if self.invalid_frame else '[ ]')
            colour = (255, 170, 90) if self.invalid_frame else (140, 140, 140)
            surface = self._small_font.render(text, True, colour)
            rect = surface.get_rect(center=(self.width // 2, y))
            display.blit(surface, rect)
            return

        if self.chosen:
            confirm_text = 'CONFIRM (%s)' % ', '.join(str(v) for v in sorted(self.chosen))
            confirm_colour = (255, 220, 0) if self.selected == CONFIRM_ROW else (200, 200, 200)
        else:
            confirm_text = 'CONFIRM (tick at least one level first)'
            confirm_colour = (140, 140, 140)
        if self.selected == CONFIRM_ROW:
            confirm_text = '> %s <' % confirm_text
        surface = self._text_font.render(confirm_text, True, confirm_colour)
        rect = surface.get_rect(center=(self.width // 2, y + 12))
        display.blit(surface, rect)

        # The failsafe row. Muted grey unless the cursor is on it: it has to be
        # findable when it is needed and unremarkable the rest of the time,
        # because a discard that looks as inviting as CONFIRM is one a tired
        # participant will start reaching for.
        no_input_label = 'NO INPUT - discard this question, nothing is saved'
        if self.selected == NO_INPUT_ROW:
            no_input_label = '> %s <' % no_input_label
            no_input_colour = (255, 170, 90)
        else:
            no_input_colour = (140, 140, 140)
        surface = self._small_font.render(no_input_label, True, no_input_colour)
        rect = surface.get_rect(center=(self.width // 2, y + 52))
        display.blit(surface, rect)


class StartScreenOverlay(object):
    def __init__(self, width, height, has_wheel=False):
        self.width = width
        self.height = height
        # Only used to word the hint. The CONFIRM button is accepted either
        # way -- an unbound wheel simply never sends the event -- so this
        # decides what the screen PROMISES, and promising a control that is
        # not attached is worse than not mentioning it.
        self.has_wheel = has_wheel
        self._title_font = pygame.font.Font(pygame.font.get_default_font(), 40)
        self._text_font = pygame.font.Font(pygame.font.get_default_font(), 24)
        self._button_font = pygame.font.Font(pygame.font.get_default_font(), 28)

    def _button_rect(self):
        button_w = 220
        button_h = 70
        x = (self.width - button_w) // 2
        y = int(self.height * 0.58)
        return pygame.Rect(x, y, button_w, button_h)

    def handle_event(self, event):
        if event.type == pygame.QUIT:
            return 'quit'
        if event.type == pygame.JOYBUTTONDOWN:
            # QUIT first: the two are distinct buttons, but checking the
            # destructive one first means a future remap that collides can
            # only ever fail safe.
            if (WHEEL_BUTTON_QUIT is not None
                    and event.button == WHEEL_BUTTON_QUIT):
                return 'quit'
            # The same button that submits an LoA prompt also starts the drive.
            # One button for "confirm", wherever the participant meets it: they
            # are taught it in the practice phase and then use it every 20 s all
            # session, so requiring the mouse for the one screen that comes
            # first is the odd case, not this.
            if (LOA_WHEEL_BUTTON_CONFIRM is not None
                    and event.button == LOA_WHEEL_BUTTON_CONFIRM):
                return 'start'
        if event.type == pygame.KEYDOWN and event.key == K_ESCAPE:
            return 'quit'
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if self._button_rect().collidepoint(event.pos):
                return 'start'
        return None

    def render(self, display):
        overlay = pygame.Surface((self.width, self.height))
        overlay.set_alpha(170)
        overlay.fill((0, 0, 0))
        display.blit(overlay, (0, 0))

        title = self._title_font.render('ProActivity Experiment', True, (255, 255, 255))
        title_rect = title.get_rect(center=(self.width // 2, int(self.height * 0.35)))
        display.blit(title, title_rect)

        hint = self._text_font.render(
            'Press the CONFIRM button on the wheel, or click Start, to begin driving.'
            if self.has_wheel else 'Click Start to begin driving.',
            True, (220, 220, 220))
        hint_rect = hint.get_rect(center=(self.width // 2, int(self.height * 0.45)))
        display.blit(hint, hint_rect)

        button_rect = self._button_rect()
        pygame.draw.rect(display, (30, 144, 255), button_rect, border_radius=8)
        pygame.draw.rect(display, (255, 255, 255), button_rect, width=2, border_radius=8)
        label = self._button_font.render('START', True, (255, 255, 255))
        label_rect = label.get_rect(center=button_rect.center)
        display.blit(label, label_rect)


class SessionEndedOverlay(object):
    """Terminal screen: ProVoice has stopped recording, so the drive is over.

    ANY wheel button exits, not a specific one -- the participant has just been
    told the session is finished and should not have to hunt for the right
    control, and unlike the LoA popup there is no second action to confuse it
    with. Any key and a mouse click do the same, because a session can end on a
    machine with no wheel bound (a --keyboard-input run, or a wheel that failed
    to initialise), and an exit screen that cannot be dismissed would leave the
    participant staring at a frozen simulator.
    """

    def __init__(self, width, height, reason='', continues=False):
        self.width = width
        self.height = height
        self.reason = (reason or '').strip()
        # `continues` = something happens AFTER this screen, so do not promise
        # the participant they are finished. In a study block the button press
        # opens the questionnaire; in every other run it really is the end.
        # Telling someone the session is over and then handing them thirteen
        # more items is a worse experience than the extra word costs.
        self.continues = continues
        self._title_font = pygame.font.Font(pygame.font.get_default_font(), 44)
        self._text_font = pygame.font.Font(pygame.font.get_default_font(), 26)
        self._small_font = pygame.font.Font(pygame.font.get_default_font(), 20)

    def handle_event(self, event):
        if event.type == pygame.QUIT:
            return 'quit'
        if event.type in (pygame.JOYBUTTONDOWN, pygame.KEYDOWN,
                          pygame.MOUSEBUTTONDOWN):
            return 'quit'
        return None

    def render(self, display):
        overlay = pygame.Surface((self.width, self.height))
        overlay.set_alpha(200)
        overlay.fill((0, 0, 0))
        display.blit(overlay, (0, 0))

        title = self._title_font.render(
            'Driving block complete' if self.continues
            else 'Experiment session ended', True, (255, 255, 255))
        display.blit(title, title.get_rect(
            center=(self.width // 2, int(self.height * 0.38))))

        hint = self._text_font.render(
            'Press any button on the steering wheel to continue.'
            if self.continues else
            'Press any button on the steering wheel to exit.', True,
            (235, 235, 235))
        display.blit(hint, hint.get_rect(
            center=(self.width // 2, int(self.height * 0.50))))

        alt = self._small_font.render('(any key also works)', True, (170, 170, 170))
        display.blit(alt, alt.get_rect(
            center=(self.width // 2, int(self.height * 0.56))))

        if self.reason:
            why = self._small_font.render(self.reason, True, (140, 140, 140))
            display.blit(why, why.get_rect(
                center=(self.width // 2, int(self.height * 0.63))))


def find_weather_presets():
    rgx = re.compile('.+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)')
    name = lambda x: ' '.join(m.group(0) for m in rgx.finditer(x))
    presets = [x for x in dir(carla.WeatherParameters) if re.match('[A-Z].+', x)]
    return [(getattr(carla.WeatherParameters, x), name(x)) for x in presets]


def get_actor_display_name(actor, truncate=250):
    name = ' '.join(actor.type_id.replace('_', '.').title().split('.')[1:])
    return (name[:truncate - 1] + u'\u2026') if len(name) > truncate else name

def get_actor_blueprints(world, filter, generation):
    bps = world.get_blueprint_library().filter(filter)

    if generation.lower() == "all":
        return bps

    # If the filter returns only one bp, we assume that this one needed
    # and therefore, we ignore the generation
    if len(bps) == 1:
        return bps

    try:
        int_generation = int(generation)
        # Check if generation is in available generations
        if int_generation in [1, 2, 3, 4]:
            bps = [x for x in bps if int(x.get_attribute('generation')) == int_generation]
            return bps
        else:
            print("   Warning! Actor Generation is not valid. No actor will be spawned.")
            return []
    except:
        print("   Warning! Actor Generation is not valid. No actor will be spawned.")
        return []


# ==============================================================================
# -- World ---------------------------------------------------------------------
# ==============================================================================


# --- Ego spawn point ----------------------------------------------------------
# Spawning stays random by default. --fixed pins the ego to this index instead,
# so calibration runs all start from an identical position. 152 is deliberately
# ~80 m from every index fixed_npc_traffic.py uses (0, 5, 10 ... 50), which
# matters because those NPCs are spawned before Drive starts and would otherwise
# be able to block the point.
FIXED_SPAWN_POINT_INDEX = 152


# ==============================================================================
# -- Brake assist ---------------------------------------------------------------
# ==============================================================================

# CARLA 0.10 ships every vehicle with brake torque worth only ~0.5-0.7 g, and on
# this build the per-wheel fields of VehiclePhysicsControl are effectively
# read-only: apply_physics_control accepts the write, ticks, and leaves the
# wheels array unchanged (verified with scripts/check_braking.py probe). No
# blueprint in the fleet reaches a realistic 0.9 g -- the best sedan, the MKZ
# this file spawns, measures 6.6 m/s^2 -- so the shortfall cannot be fixed where
# it belongs, in the physics.
#
# Instead the deficit is topped up with a force opposing travel, scaled by the
# brake input, so full brake produces roughly what a real car does. The stock
# capability is measured from the spawned vehicle's own physics, so this adapts
# if the blueprint changes rather than hard-coding one car's shortfall.
#
# What this deliberately does NOT reproduce:
#   - Weight transfer. Actor.add_force takes no application point, so the force
#     acts at the centre of mass and the car does not pitch under braking. The
#     deceleration is right; the body attitude is not.
#   - Tyre behaviour. The top-up bypasses the tyre model, so it is capped at a
#     friction-derived ceiling that falls in rain rather than braking as hard
#     wet as dry. That ceiling approximates grip; it does not simulate it.
#   - ABS, on the topped-up portion.
#
# IMPORTANT: deterministic only when the drive loop runs one iteration per
# simulation tick, i.e. under --sync (where fixed_npc_traffic.py owns the
# clock). Without --sync the force lands at whatever rate the loop happens to
# run and braking becomes frame-rate dependent.
#
# Keep BRAKE_ASSIST_TARGET_DECEL fixed across every participant and both study
# arms. Braking strength changes driving behaviour, which feeds brake,
# speed_ratio_* and indirectly the physiological features -- varying it would
# confound the personalization comparison exactly as a varying --decision-hz
# would.
BRAKE_ASSIST_TARGET_DECEL = 8.5    # m/s^2 at full brake on dry asphalt (~0.9 g)
BRAKE_ASSIST_WET_CEILING = 0.65    # ceiling multiplier at 100% precipitation
BRAKE_ASSIST_MIN_SPEED = 1.0       # m/s; below this the stock brakes hold the car
BRAKE_ASSIST_WEATHER_PERIOD = 40   # ticks between weather refreshes (one RPC each)

# --condition-sun-rain / --condition-rain-sun: a scripted precipitation ramp,
# in CARLA's 0-100 percent scale. Timed off wall-clock (pygame.time.get_ticks),
# the same clock the 20 s LoA window interval uses, so it runs at the same
# rate whether or not --sync holds the world clock for popups.
CONDITION_PEAK_PRECIPITATION = 80.0    # percent
CONDITION_RAMP_MINUTES = 10.0          # minutes to cross the full 0-80 range
CONDITION_SUN_HOLD_MINUTES = 5.0       # rain-sun only: sunny minutes before the ramp starts
CONDITION_PRECIP_STEP = 0.5            # percent; smaller changes are not sent to CARLA


class World(object):
    def __init__(self, carla_world, hud, traffic_manager, args):
        self.world = carla_world
        # Drive never ticks in either mode (see _await_world_tick), so restart()
        # no longer branches on this -- but _set_world_frozen does: the clock can
        # only be genuinely paused when there is a fixed-step clock to withhold.
        self.sync = args.sync
        self.clock_pause_file = getattr(args, 'clock_pause_file', '')
        self.traffic_manager = traffic_manager
        self.actor_role_name = args.rolename
        self.control_mode = args.control  # Store control mode
        try:
            self.map = self.world.get_map()
        except RuntimeError as error:
            print('RuntimeError: {}'.format(error))
            print('  The server could not send the OpenDRIVE (.xodr) file:')
            print('  Make sure it exists, has the same name of your town, and is correct.')
            sys.exit(1)
        self.hud = hud
        self.player = None
        self.collision_sensor = None
        self.lane_invasion_sensor = None
        self.gnss_sensor = None
        self.imu_sensor = None
        self.radar_sensor = None
        self.camera_manager = None
        self._weather_presets = find_weather_presets()
        self._weather_index = 0
        # Scripted precipitation schedule (--condition-sun-rain /
        # --condition-rain-sun). None = no schedule, weather stays whatever
        # next_weather()/CARLA's default leaves it at.
        if getattr(args, 'condition_sun_rain', False):
            self._precip_mode = 'sun_rain'
        elif getattr(args, 'condition_rain_sun', False):
            self._precip_mode = 'rain_sun'
        else:
            self._precip_mode = None
        self._precip_start_ms = None
        self._precip_last_pct = None
        self._actor_filter = args.filter
        self._actor_generation = args.generation
        self._gamma = args.gamma
        self._render_scale = getattr(args, 'render_scale', 1.0)
        self._fixed_spawn = args.fixed
        # Brake assist. Measured per spawn in _cache_brake_assist, which runs
        # from modify_vehicle_physics inside restart() -- so these have to exist
        # before restart() is called.
        self.brake_assist_decel = getattr(args, 'brake_assist_decel',
                                          BRAKE_ASSIST_TARGET_DECEL)
        self._assist_mass = 0.0
        self._assist_stock_decel = 0.0
        self._assist_wet = 0.0
        self._assist_wet_age = 0
        self._assist_dt = self.world.get_settings().fixed_delta_seconds or 0.05
        self.restart()
        self.world.on_tick(hud.on_world_tick)
        self.recording_enabled = False
        self.recording_start = 0
        self.constant_velocity_enabled = False
        self.show_vehicle_telemetry = False
        self.doors_are_open = False
        self.current_map_layer = 0
        self.map_layer_names = [
            carla.MapLayer.NONE,
            carla.MapLayer.Buildings,
            carla.MapLayer.Decals,
            carla.MapLayer.Foliage,
            carla.MapLayer.Ground,
            carla.MapLayer.ParkedVehicles,
            carla.MapLayer.Particles,
            carla.MapLayer.Props,
            carla.MapLayer.StreetLights,
            carla.MapLayer.Walls,
            carla.MapLayer.All
        ]

    def _spawn_candidates(self, spawn_points):
        """Spawn points to try, in order.

        Randomly ordered, unless --fixed starts the walk at
        FIXED_SPAWN_POINT_INDEX so calibration runs always begin in the same
        place. Either way this returns an *order* rather than one point:
        try_spawn_actor returns None when a point is occupied, and the original
        code re-rolled a random point on each failure, which could retry the same
        blocked point forever. A full permutation bounds the retries — and under
        --fixed it keeps the fallback deterministic too, so a blocked point
        yields the same second choice on every run.
        """
        count = len(spawn_points)
        if not self._fixed_spawn:
            return random.sample(spawn_points, count)
        start = FIXED_SPAWN_POINT_INDEX % count
        return [spawn_points[(start + i) % count] for i in range(count)]

    def restart(self):
        self.player_max_speed = 1.589
        self.player_max_speed_fast = 3.713
        # Keep same camera config if the camera manager exists.
        cam_index = self.camera_manager.index if self.camera_manager is not None else 0
        cam_pos_index = self.camera_manager.transform_index if self.camera_manager is not None else 0
        # Get a random blueprint.
        blueprint_list = get_actor_blueprints(self.world, self._actor_filter, self._actor_generation)
        if not blueprint_list:
            raise ValueError("Couldn't find any blueprints with the specified filters")
        # blueprint = random.choice(blueprint_list)
        # vehicle selection
        #mkz = self.world.get_blueprint_library().filter('vehicle.lincoln.mkz*')
        mkz = self.world.get_blueprint_library().filter('vehicle.dodge.charger')
        if not mkz:
            raise ValueError("Couldn't find a 'vehicle.dodge.charger*' blueprint")
        blueprint = mkz[0]
        blueprint.set_attribute('role_name', self.actor_role_name)
        if blueprint.has_attribute('terramechanics'):
            blueprint.set_attribute('terramechanics', 'true')
        if blueprint.has_attribute('color'):
            color = random.choice(blueprint.get_attribute('color').recommended_values)
            blueprint.set_attribute('color', color)
        if blueprint.has_attribute('driver_id'):
            driver_id = random.choice(blueprint.get_attribute('driver_id').recommended_values)
            blueprint.set_attribute('driver_id', driver_id)
        if blueprint.has_attribute('is_invincible'):
            blueprint.set_attribute('is_invincible', 'true')
        # set the max speed
        if blueprint.has_attribute('speed'):
            self.player_max_speed = float(blueprint.get_attribute('speed').recommended_values[1])
            self.player_max_speed_fast = float(blueprint.get_attribute('speed').recommended_values[2])

        # Spawn the player.
        if self.player is not None:
            spawn_point = self.player.get_transform()
            spawn_point.location.z += 2.0
            spawn_point.rotation.roll = 0.0
            spawn_point.rotation.pitch = 0.0
            self.destroy()
            self.player = self.world.try_spawn_actor(blueprint, spawn_point)
            self.show_vehicle_telemetry = False
            self.modify_vehicle_physics(self.player)
        if self.player is None:
            spawn_points = self.map.get_spawn_points()
            if not spawn_points:
                print('There are no spawn points available in your map/town.')
                print('Please add some Vehicle Spawn Point to your UE5 scene.')
                sys.exit(1)
            for spawn_point in self._spawn_candidates(spawn_points):
                self.player = self.world.try_spawn_actor(blueprint, spawn_point)
                if self.player is not None:
                    break
            if self.player is None:
                raise RuntimeError('Could not spawn the ego vehicle at any of the '
                                   '%d spawn points' % len(spawn_points))
            self.show_vehicle_telemetry = False
            self.modify_vehicle_physics(self.player)

        # Set up the sensors.
        self.collision_sensor = CollisionSensor(self.player, self.hud, self.control_mode)
        self.lane_invasion_sensor = LaneInvasionSensor(self.player, self.hud, self.control_mode)
        self.gnss_sensor = GnssSensor(self.player)
        self.imu_sensor = IMUSensor(self.player)
        self.camera_manager = CameraManager(self.player, self.hud, self._gamma,
                                            render_scale=self._render_scale)
        self.camera_manager.transform_index = cam_pos_index
        self.camera_manager.set_sensor(cam_index, notify=False)
        actor_type = get_actor_display_name(self.player)
        if self.control_mode == 'full':
            self.hud.notification(actor_type)
        self.traffic_manager.update_vehicle_lights(self.player, True)

        # Wait for a tick either way -- under --sync the clock belongs to
        # fixed_npc_traffic.py, so Drive waits here rather than ticking. Failing
        # to get one is not fatal at this point (the sensors are attached and the
        # ego exists), but it means nothing is driving the clock, so say so
        # loudly instead of writing a vehicle id that ProVoice will act on.
        if not _await_world_tick(self.world):
            print("[WARN] No world tick within %.0fs while setting up the ego. "
                  "If this is --sync, the clock owner "
                  "(src/drive/fixed_npc_traffic.py --sync) is not running."
                  % SYNC_STALL_TIMEOUT_S)

        # Write vehicle id AFTER the world tick so ProVoice only connects
        # once Drive is fully initialised (prevents CARLA race condition).
        try:
            id_file = os.path.join(os.getcwd(), "vehicle_id.txt")
            tmp_file = id_file + ".tmp"
            with open(tmp_file, "w") as f:
                f.write(str(self.player.id))
            os.replace(tmp_file, id_file)
            print(f"[INFO] Written vehicle id to {id_file}")
        except Exception as e:
            print("[WARN] Failed to write vehicle id file:", e)

    def next_weather(self, reverse=False):
        self._weather_index += -1 if reverse else 1
        self._weather_index %= len(self._weather_presets)
        preset = self._weather_presets[self._weather_index]
        if self.control_mode == 'full':
            self.hud.notification('Weather: %s' % preset[1])
        self.player.get_world().set_weather(preset[0])

    def _apply_precipitation(self, pct):
        """Push a precipitation level to CARLA, skipping negligible changes.

        Precipitation alone renders rain particles under whatever sky the map
        started with, which -- if that base weather is a clear/sunny preset --
        makes the ramp barely visible. CARLA's own presets never move
        precipitation on its own: cloudiness and wind_intensity track it
        ~1:1 (MidRainyNoon: cloudiness=wind_intensity=precipitation=60;
        HardRainNoon: 100/100/100) and fog_density rises with it too
        (ClearNoon 2.0 -> HardRainNoon 7.0), while `wetness` stays 0.0 in
        every single built-in preset including the rain ones -- so it is
        dropped here rather than set to `pct` as before, which was doing
        nothing CARLA's own presets don't also leave at zero.

        One RPC (get + set) per call that actually moves the needle. A 0-80%
        ramp over CONDITION_RAMP_MINUTES only crosses CONDITION_PRECIP_STEP
        every few seconds, so this naturally throttles itself to about the
        same rate as the brake-assist weather refresh (BRAKE_ASSIST_WEATHER_PERIOD)
        without a separate tick counter.
        """
        pct = max(0.0, min(100.0, pct))
        if (self._precip_last_pct is not None
                and abs(pct - self._precip_last_pct) < CONDITION_PRECIP_STEP):
            return
        weather = self.world.get_weather()
        weather.precipitation = pct
        weather.precipitation_deposits = pct
        weather.cloudiness = pct
        weather.wind_intensity = pct
        weather.fog_density = 2.0 + (pct / 100.0) * 5.0
        self.world.set_weather(weather)
        self._precip_last_pct = pct

    def start_weather_schedule(self, now_ms):
        """Arm the --condition-sun-rain / --condition-rain-sun ramp.

        Called once, when the driver actually starts driving (not at world
        spawn, which can sit on the start screen for an arbitrary time) --
        so "start of session" means what a participant would expect it to.
        """
        if self._precip_mode is None:
            return
        self._precip_start_ms = now_ms
        initial = (CONDITION_PEAK_PRECIPITATION if self._precip_mode == 'sun_rain'
                  else 0.0)
        self._apply_precipitation(initial)
        print('[CONDITION] %s: precipitation schedule armed'
              % self._precip_mode)

    def update_weather_schedule(self, now_ms):
        """Advance the scripted precipitation ramp. No-op with no schedule armed.

        Timed off elapsed wall-clock minutes since start_weather_schedule(),
        not incremented per call, so calling it irregularly (e.g. skipped
        while a LoA popup freezes the scene) cannot drift it -- the next call
        just computes the correct value for however much time has actually
        passed.
        """
        if self._precip_mode is None or self._precip_start_ms is None:
            return
        elapsed_min = (now_ms - self._precip_start_ms) / 60000.0
        if self._precip_mode == 'sun_rain':
            # 80% -> 0% over CONDITION_RAMP_MINUTES, starting immediately.
            pct = CONDITION_PEAK_PRECIPITATION * max(0.0, 1.0 - elapsed_min / CONDITION_RAMP_MINUTES)
        else:
            # sunny for CONDITION_SUN_HOLD_MINUTES, then 0% -> 80% over
            # CONDITION_RAMP_MINUTES.
            ramp_elapsed = elapsed_min - CONDITION_SUN_HOLD_MINUTES
            pct = CONDITION_PEAK_PRECIPITATION * min(1.0, max(0.0, ramp_elapsed / CONDITION_RAMP_MINUTES))
        self._apply_precipitation(pct)

    def next_map_layer(self, reverse=False):
        self.current_map_layer += -1 if reverse else 1
        self.current_map_layer %= len(self.map_layer_names)
        selected = self.map_layer_names[self.current_map_layer]
        if self.control_mode == 'full':
            self.hud.notification('LayerMap selected: %s' % selected)

    def load_map_layer(self, unload=False):
        selected = self.map_layer_names[self.current_map_layer]
        if unload:
            if self.control_mode == 'full':
                self.hud.notification('Unloading map layer: %s' % selected)
            self.world.unload_map_layer(selected)
        else:
            if self.control_mode == 'full':
                self.hud.notification('Loading map layer: %s' % selected)
            self.world.load_map_layer(selected)

    def toggle_radar(self):
        if self.radar_sensor is None:
            self.radar_sensor = RadarSensor(self.player)
        elif self.radar_sensor.sensor is not None:
            self.radar_sensor.sensor.destroy()
            self.radar_sensor = None

    def modify_vehicle_physics(self, actor):
        #If actor is not a vehicle, we cannot use the physics control
        try:
            physics_control = actor.get_physics_control()
            physics_control.use_sweep_wheel_collision = True
            actor.apply_physics_control(physics_control)
        except Exception:
            pass
        # Deliberately outside the try above: a failure to measure the brake
        # assist is not the same failure as a vehicle with no physics control,
        # and swallowing it would silently disable the assist.
        self._cache_brake_assist(actor)

    def _cache_brake_assist(self, actor):
        """Measure how much braking the stock physics gives, once per spawn.

        Peak deceleration is the summed per-wheel braking force (torque over
        radius, for the wheels the brake acts on) divided by mass. Validated
        against measurement: this predicted 5.7 m/s^2 for the Dodge Charger and
        scripts/check_braking.py measured 6.0, the excess being engine braking.
        """
        self._assist_mass = 0.0
        self._assist_stock_decel = 0.0
        if self.brake_assist_decel <= 0.0:
            print("[INFO] Brake assist disabled (--brake-assist-decel 0); the car "
                  "brakes on stock CARLA physics.")
            return
        try:
            pc = actor.get_physics_control()
        except Exception as e:
            print("[WARN] Brake assist off: could not read physics control (%s)" % e)
            return
        force = 0.0
        for w in pc.wheels:
            radius_m = w.wheel_radius / 100.0  # centimetres in CARLA 0.10
            if radius_m > 0 and w.affected_by_brake:
                force += w.max_brake_torque / radius_m
        if pc.mass <= 0 or force <= 0:
            print("[WARN] Brake assist off: implausible physics (mass=%.1f, "
                  "force=%.1f)" % (pc.mass, force))
            return
        self._assist_mass = pc.mass
        self._assist_stock_decel = force / pc.mass
        share = max(0.0, self.brake_assist_decel - self._assist_stock_decel)
        print("[INFO] Brake assist: stock %.1f m/s^2, target %.1f m/s^2 -- assist "
              "supplies %.0f%% of peak braking."
              % (self._assist_stock_decel, self.brake_assist_decel,
                 100.0 * share / self.brake_assist_decel))
        if not self.sync:
            print("[WARN] Brake assist is frame-rate dependent without --sync. "
                  "Run the study with --sync so braking is reproducible.")

    def apply_brake_assist(self, control):
        """Top up the stock brakes toward BRAKE_ASSIST_TARGET_DECEL.

        Called once per drive-loop iteration, which under --sync is once per
        simulation tick. A no-op when not braking, when nearly stopped, or when
        the stock physics already meets the target.
        """
        if self._assist_mass <= 0.0 or control.brake <= 0.0:
            return
        velocity = self.player.get_velocity()
        # Horizontal only: on a slope the full 3D direction would add a vertical
        # component that changes wheel load, and a bouncing car would get a
        # force with no relation to braking.
        speed = math.sqrt(velocity.x * velocity.x + velocity.y * velocity.y)
        if speed < BRAKE_ASSIST_MIN_SPEED:
            return

        self._assist_wet_age -= 1
        if self._assist_wet_age <= 0:
            try:
                self._assist_wet = self.world.get_weather().precipitation / 100.0
            except Exception:
                self._assist_wet = 0.0
            self._assist_wet_age = BRAKE_ASSIST_WEATHER_PERIOD

        ceiling = self.brake_assist_decel * (
            1.0 - (1.0 - BRAKE_ASSIST_WET_CEILING) * self._assist_wet)
        accel = max(0.0, ceiling - self._assist_stock_decel) * control.brake
        # Never more than brings the car to rest this tick, so the assist can
        # never push it backwards.
        accel = min(accel, speed / max(self._assist_dt, 1e-3))
        if accel <= 0.0:
            return
        magnitude = self._assist_mass * accel
        self.player.add_force(carla.Vector3D(
            -velocity.x / speed * magnitude,
            -velocity.y / speed * magnitude,
            0.0))

    def tick(self, clock):
        self.hud.tick(self, clock)

    def render(self, display):
        self.camera_manager.render(display)
        self.hud.render(display)

    def destroy_sensors(self):
        self.camera_manager.sensor.destroy()
        self.camera_manager.sensor = None
        self.camera_manager.index = None

    def destroy(self):
        if self.radar_sensor is not None:
            self.toggle_radar()
        sensors = [
            self.camera_manager.sensor,
            self.collision_sensor.sensor,
            self.lane_invasion_sensor.sensor,
            self.gnss_sensor.sensor,
            self.imu_sensor.sensor]
        for sensor in sensors:
            if sensor is not None:
                sensor.stop()
                sensor.destroy()
        if self.player is not None:
            self.player.destroy()


# ==============================================================================
# -- KeyboardControl -----------------------------------------------------------
# ==============================================================================


# --- Steering wheel axis mapping ---------------------------------------------
# Two layouts exist depending on how the wheel enumerates:
#
#   NATIVE (Logitech Gaming Software running, G25 = USB PID 0xC299): four axes,
#   one per pedal. Pedals rest at +1.0 and travel to -1.0 when fully pressed.
#
#   COMPATIBILITY ("Driving Force", PID 0xC294 — what you get with no Logitech
#   driver installed): two axes only. Throttle and brake are SUMMED onto a
#   single axis resting at centre, so they cannot be pressed independently and
#   pressing both cancels out. 240 deg of steering travel instead of 900, no FFB.
#
# We pick the layout from the axis count at runtime, so the same build works
# either way. If throttle and brake come out swapped in compatibility mode, flip
# the sign of WHEEL_COMBINED_THROTTLE_SIGN — that is the only knob needed.
WHEEL_AXIS_STEER = 0
WHEEL_AXIS_THROTTLE = 1       # native only
WHEEL_AXIS_BRAKE = 2          # native only
WHEEL_AXIS_COMBINED = 1       # compatibility only
WHEEL_COMBINED_THROTTLE_SIGN = -1.0  # which direction of the combined axis is throttle
WHEEL_STEER_DEADZONE = 0.02
WHEEL_PEDAL_DEADZONE = 0.05
WHEEL_STEER_LIMIT = 1.0       # keyboard caps at 0.7; a wheel should use full lock


class KeyboardControl(object):
    """Class that handles keyboard input, and steering wheel input when present."""
    def __init__(self, world, start_in_autopilot, control_mode='test', use_wheel=True):
        self._autopilot_enabled = start_in_autopilot
        self._ackermann_enabled = False
        self._ackermann_reverse = 1
        self._control_mode = control_mode  # 'test' or 'full'
        self._wheel = None
        self._wheel_pedals_combined = False
        # First (unconfirmed) press of WHEEL_BUTTON_QUIT, in the parse_events
        # `now_ms` clock; None when no press is pending. See
        # WHEEL_QUIT_CONFIRM_WINDOW_MS.
        self._wheel_quit_armed_ms = None
        if use_wheel:
            self._init_wheel()
        if isinstance(world.player, carla.Vehicle):
            self._control = carla.VehicleControl()
            self._ackermann_control = carla.VehicleAckermannControl()
            self._lights = carla.VehicleLightState.NONE
            world.player.set_autopilot(self._autopilot_enabled)
            world.player.set_light_state(self._lights)
        elif isinstance(world.player, carla.Walker):
            self._control = carla.WalkerControl()
            self._autopilot_enabled = False
            self._rotation = world.player.get_transform().rotation
        else:
            raise NotImplementedError("Actor type not supported")
        self._steer_cache = 0.0
        if self._control_mode == 'full':
            world.hud.notification("Press 'H' or '?' for help.", seconds=4.0)
        if self._wheel is not None:
            world.hud.notification(
                "Steering wheel: %s (%s pedals)" % (
                    self._wheel.get_name(),
                    "combined" if self._wheel_pedals_combined else "separate"),
                seconds=4.0)

    @property
    def has_wheel(self):
        return self._wheel is not None

    def _init_wheel(self):
        """Bind the first attached wheel, or stay on keyboard if there is none."""
        try:
            pygame.joystick.init()
            if pygame.joystick.get_count() == 0:
                print("[INFO] No steering wheel detected — using keyboard control.")
                return
            js = pygame.joystick.Joystick(0)
            js.init()
            axes = js.get_numaxes()
            if axes < 2:
                print("[WARN] '%s' reports only %d axis/axes — too few to drive; "
                      "using keyboard control." % (js.get_name(), axes))
                return
            self._wheel = js
            # Fewer than three axes means the pedals share one axis.
            self._wheel_pedals_combined = axes < 3
            print("[INFO] Steering wheel: '%s' (%d axes, %d buttons)"
                  % (js.get_name(), axes, js.get_numbuttons()))
            if self._wheel_pedals_combined:
                print("[WARN] Wheel is in COMPATIBILITY mode: throttle and brake share "
                      "one axis (cannot be pressed independently), reduced steering "
                      "range, no force feedback. Install Logitech Gaming Software to "
                      "get native mode with separate pedal axes.")
        except Exception as e:
            print("[WARN] Steering wheel init failed (%s) — using keyboard control." % e)
            self._wheel = None

    def parse_events(self, client, world, clock, sync_mode, events=None,
                     suppress_wheel_quit=False, now_ms=None):
        """Consume one frame's events; True means the session should end.

        ``suppress_wheel_quit``: True while a call panel is on screen. Button 7
        (WHEEL_BUTTON_QUIT) sits next to the paddles a driver is reaching for to
        answer a call (CALL_WHEEL_BUTTON_AFFIRM/NEGATIVE in call_event.py), and
        an accidental hit there does not mean "end the session" the way it does
        the rest of the time -- it means a participant reached one button too
        far while under time pressure from an 8 s cap. Ending the whole study
        on that mistake is a far worse outcome than the mistake itself, so the
        button is dropped ENTIRELY for as long as the call is up: it neither
        arms nor confirms the double-press below, and any press pending from
        just before the call started is cleared rather than left to be
        completed once the call ends.

        Otherwise (ordinary driving) button 7 needs a SECOND press within
        ``WHEEL_QUIT_CONFIRM_WINDOW_MS`` to end the session -- see that
        constant. The first press only arms it and shows a HUD hint; nothing
        ends until the second press lands inside the window, and a press
        arriving after the window has elapsed re-arms rather than confirms, so
        two unrelated taps minutes apart can never combine into a quit.

        Only the wheel-quit branch has any of this gating: window-close
        (pygame.QUIT) and the keyboard quit shortcuts stay a single press
        throughout, since those are the experimenter's controls, not something
        a participant reaches for by accident.
        """
        if now_ms is None:
            now_ms = pygame.time.get_ticks()
        if isinstance(self._control, carla.VehicleControl):
            current_lights = self._lights
        event_list = events if events is not None else pygame.event.get()
        for event in event_list:
            if event.type == pygame.QUIT:
                return True
            elif (event.type == pygame.JOYBUTTONDOWN
                    and WHEEL_BUTTON_QUIT is not None
                    and event.button == WHEEL_BUTTON_QUIT):
                if suppress_wheel_quit:
                    # Fully inert during a call: does not arm, does not
                    # confirm, and drops any press armed just before the call
                    # started so it cannot be completed once the call ends.
                    self._wheel_quit_armed_ms = None
                elif (self._wheel_quit_armed_ms is not None
                        and now_ms - self._wheel_quit_armed_ms
                        <= WHEEL_QUIT_CONFIRM_WINDOW_MS):
                    # Second press, in time: same effect as closing the window.
                    self._wheel_quit_armed_ms = None
                    return True
                else:
                    # First press (or the window on a previous one elapsed):
                    # arm it and say so, rather than quitting silently on a
                    # press the driver may not even have meant as the first of
                    # two.
                    self._wheel_quit_armed_ms = now_ms
                    world.hud.notification(
                        "Press EXIT again within 1s to end the session",
                        seconds=WHEEL_QUIT_CONFIRM_WINDOW_MS / 1000.0)
            elif event.type == pygame.KEYUP:
                if self._is_quit_shortcut(event.key):
                    return True
                elif event.key == K_F1:
                    world.hud.toggle_info()
                elif event.key == K_h or (event.key == K_SLASH and pygame.key.get_mods() & KMOD_SHIFT):
                    world.hud.help.toggle()
                # Only allow additional controls in 'full' mode
                elif self._control_mode == 'full':
                    if event.key == K_BACKSPACE:
                        if self._autopilot_enabled:
                            world.player.set_autopilot(False)
                            world.restart()
                            world.player.set_autopilot(True)
                        else:
                            world.restart()
                    elif event.key == K_v and pygame.key.get_mods() & KMOD_SHIFT:
                        world.next_map_layer(reverse=True)
                    elif event.key == K_v:
                        world.next_map_layer()
                    elif event.key == K_b and pygame.key.get_mods() & KMOD_SHIFT:
                        world.load_map_layer(unload=True)
                    elif event.key == K_b:
                        world.load_map_layer()
                    elif event.key == K_TAB:
                        world.camera_manager.toggle_camera()
                    elif event.key == K_c and pygame.key.get_mods() & KMOD_SHIFT:
                        world.next_weather(reverse=True)
                    elif event.key == K_c:
                        world.next_weather()
                    elif event.key == K_g:
                        world.toggle_radar()
                    elif event.key == K_BACKQUOTE:
                        world.camera_manager.next_sensor()
                    elif event.key == K_n:
                        world.camera_manager.next_sensor()
                    elif event.key == K_w and (pygame.key.get_mods() & KMOD_CTRL):
                        if world.constant_velocity_enabled:
                            world.player.disable_constant_velocity()
                            world.constant_velocity_enabled = False
                            if self._control_mode == 'full':
                                world.hud.notification("Disabled Constant Velocity Mode")
                        else:
                            world.player.enable_constant_velocity(carla.Vector3D(17, 0, 0))
                            world.constant_velocity_enabled = True
                            if self._control_mode == 'full':
                                world.hud.notification("Enabled Constant Velocity Mode at 60 km/h")
                    elif event.key == K_o:
                        try:
                            if world.doors_are_open:
                                if self._control_mode == 'full':
                                    world.hud.notification("Closing Doors")
                                world.doors_are_open = False
                                world.player.close_door(carla.VehicleDoor.All)
                            else:
                                if self._control_mode == 'full':
                                    world.hud.notification("Opening doors")
                                world.doors_are_open = True
                                world.player.open_door(carla.VehicleDoor.All)
                        except Exception:
                            pass
                    elif event.key == K_t:
                        if world.show_vehicle_telemetry:
                            world.player.show_debug_telemetry(False)
                            world.show_vehicle_telemetry = False
                            if self._control_mode == 'full':
                                world.hud.notification("Disabled Vehicle Telemetry")
                        else:
                            try:
                                world.player.show_debug_telemetry(True)
                                world.show_vehicle_telemetry = True
                                if self._control_mode == 'full':
                                    world.hud.notification("Enabled Vehicle Telemetry")
                            except Exception:
                                pass
                    elif event.key > K_0 and event.key <= K_9:
                        index_ctrl = 0
                        if pygame.key.get_mods() & KMOD_CTRL:
                            index_ctrl = 9
                        world.camera_manager.set_sensor(event.key - 1 - K_0 + index_ctrl)
                    elif event.key == K_r and not (pygame.key.get_mods() & KMOD_CTRL):
                        world.camera_manager.toggle_recording()
                    elif event.key == K_r and (pygame.key.get_mods() & KMOD_CTRL):
                        if (world.recording_enabled):
                            client.stop_recorder()
                            world.recording_enabled = False
                            if self._control_mode == 'full':
                                world.hud.notification("Recorder is OFF")
                        else:
                            client.start_recorder("manual_recording.rec")
                            world.recording_enabled = True
                            if self._control_mode == 'full':
                                world.hud.notification("Recorder is ON")
                    elif event.key == K_p and (pygame.key.get_mods() & KMOD_CTRL):
                        # stop recorder
                        client.stop_recorder()
                        world.recording_enabled = False
                        # work around to fix camera at start of replaying
                        current_index = world.camera_manager.index
                        world.destroy_sensors()
                        # disable autopilot
                        self._autopilot_enabled = False
                        world.player.set_autopilot(self._autopilot_enabled)
                        if self._control_mode == 'full':
                            world.hud.notification("Replaying file 'manual_recording.rec'")
                        # replayer
                        client.replay_file("manual_recording.rec", world.recording_start, 0, 0)
                        world.camera_manager.set_sensor(current_index)
                    elif event.key == K_MINUS and (pygame.key.get_mods() & KMOD_CTRL):
                        if pygame.key.get_mods() & KMOD_SHIFT:
                            world.recording_start -= 10
                        else:
                            world.recording_start -= 1
                        if self._control_mode == 'full':
                            world.hud.notification("Recording start time is %d" % (world.recording_start))
                    elif event.key == K_EQUALS and (pygame.key.get_mods() & KMOD_CTRL):
                        if pygame.key.get_mods() & KMOD_SHIFT:
                            world.recording_start += 10
                        else:
                            world.recording_start += 1
                        if self._control_mode == 'full':
                            world.hud.notification("Recording start time is %d" % (world.recording_start))
                if isinstance(self._control, carla.VehicleControl):
                    # Q (reverse toggle) is always available in both modes
                    if event.key == K_q:
                        if not self._ackermann_enabled:
                            self._control.gear = 1 if self._control.reverse else -1
                        else:
                            self._ackermann_reverse *= -1
                            # Reset ackermann control
                            self._ackermann_control = carla.VehicleAckermannControl()
                    # Other vehicle controls only available in 'full' mode
                    elif self._control_mode == 'full':
                        if event.key == K_f:
                            # Toggle ackermann controller
                            self._ackermann_enabled = not self._ackermann_enabled
                            world.hud.show_ackermann_info(self._ackermann_enabled)
                            if self._control_mode == 'full':
                                world.hud.notification("Ackermann Controller %s" %
                                                       ("Enabled" if self._ackermann_enabled else "Disabled"))
                        elif event.key == K_m:
                            self._control.manual_gear_shift = not self._control.manual_gear_shift
                            self._control.gear = world.player.get_control().gear
                            if self._control_mode == 'full':
                                world.hud.notification('%s Transmission' %
                                                       ('Manual' if self._control.manual_gear_shift else 'Automatic'))
                        elif self._control.manual_gear_shift and event.key == K_COMMA:
                            self._control.gear = max(-1, self._control.gear - 1)
                        elif self._control.manual_gear_shift and event.key == K_PERIOD:
                            self._control.gear = self._control.gear + 1
                        elif event.key == K_p and not pygame.key.get_mods() & KMOD_CTRL:
                            if not self._autopilot_enabled and not sync_mode:
                                print("WARNING: You are currently in asynchronous mode and could "
                                      "experience some issues with the traffic simulation")
                            self._autopilot_enabled = not self._autopilot_enabled
                            world.player.set_autopilot(self._autopilot_enabled)
                            if self._control_mode == 'full':
                                world.hud.notification(
                                    'Autopilot %s' % ('On' if self._autopilot_enabled else 'Off'))
                        elif event.key == K_l and pygame.key.get_mods() & KMOD_CTRL:
                            current_lights ^= carla.VehicleLightState.Special1
                        elif event.key == K_l and pygame.key.get_mods() & KMOD_SHIFT:
                            current_lights ^= carla.VehicleLightState.HighBeam
                        elif event.key == K_l:
                            # Use 'L' key to switch between lights:
                            # closed -> position -> low beam -> fog
                            if not self._lights & carla.VehicleLightState.Position:
                                if self._control_mode == 'full':
                                    world.hud.notification("Position lights")
                                current_lights |= carla.VehicleLightState.Position
                            else:
                                if self._control_mode == 'full':
                                    world.hud.notification("Low beam lights")
                                current_lights |= carla.VehicleLightState.LowBeam
                            if self._lights & carla.VehicleLightState.LowBeam:
                                if self._control_mode == 'full':
                                    world.hud.notification("Fog lights")
                                current_lights |= carla.VehicleLightState.Fog
                            if self._lights & carla.VehicleLightState.Fog:
                                if self._control_mode == 'full':
                                    world.hud.notification("Lights off")
                                current_lights ^= carla.VehicleLightState.Position
                                current_lights ^= carla.VehicleLightState.LowBeam
                                current_lights ^= carla.VehicleLightState.Fog
                        elif event.key == K_i:
                            current_lights ^= carla.VehicleLightState.Interior
                        elif event.key == K_z:
                            current_lights ^= carla.VehicleLightState.LeftBlinker
                        elif event.key == K_x:
                            current_lights ^= carla.VehicleLightState.RightBlinker

        if not self._autopilot_enabled:
            if isinstance(self._control, carla.VehicleControl):
                if self._wheel is not None:
                    # Wheel owns steer/throttle/brake; the keyboard would fight it
                    # (it zeroes throttle on every frame no key is held).
                    self._parse_vehicle_wheel()
                else:
                    self._parse_vehicle_keys(pygame.key.get_pressed(), clock.get_time())
                self._control.reverse = self._control.gear < 0
                # Set automatic control-related vehicle lights
                if self._control.brake:
                    current_lights |= carla.VehicleLightState.Brake
                else: # Remove the Brake flag
                    current_lights &= ~carla.VehicleLightState.Brake
                if self._control.reverse:
                    current_lights |= carla.VehicleLightState.Reverse
                else: # Remove the Reverse flag
                    current_lights &= ~carla.VehicleLightState.Reverse
                if current_lights != self._lights: # Change the light state only if necessary
                    world.player.set_light_state(carla.VehicleLightState(current_lights))
                # Apply control
                if not self._ackermann_enabled:
                    world.player.apply_control(self._control)
                    # Top up the stock brakes, which CARLA 0.10 leaves well
                    # short of a real car and which cannot be raised through
                    # the physics API on this build. No-op when not braking.
                    world.apply_brake_assist(self._control)
                else:
                    world.player.apply_ackermann_control(self._ackermann_control)
                    # Update control to the last one applied by the ackermann controller.
                    self._control = world.player.get_control()
                    # Update hud with the newest ackermann control
                    world.hud.update_ackermann_control(self._ackermann_control)

            elif isinstance(self._control, carla.WalkerControl):
                self._parse_walker_keys(pygame.key.get_pressed(), clock.get_time(), world)
                world.player.apply_control(self._control)

        self._lights = current_lights

    def _parse_vehicle_wheel(self):
        """Map wheel axes onto the vehicle control.

        Steering is taken raw (no rounding) — the keyboard path quantises to 0.1,
        which on a wheel would feel like notched steering.
        """
        # The caller may already have drained the event queue; pump explicitly so
        # axis state is current regardless.
        pygame.event.pump()
        js = self._wheel

        steer = js.get_axis(WHEEL_AXIS_STEER)
        if abs(steer) < WHEEL_STEER_DEADZONE:
            steer = 0.0
        steer = max(-WHEEL_STEER_LIMIT, min(WHEEL_STEER_LIMIT, steer))

        if self._wheel_pedals_combined:
            # One axis, resting at centre: one direction is throttle, the other brake.
            combined = js.get_axis(WHEEL_AXIS_COMBINED) * WHEEL_COMBINED_THROTTLE_SIGN
            throttle = max(0.0, combined)
            brake = max(0.0, -combined)
        else:
            # Separate axes, resting at +1.0 and travelling to -1.0 when pressed.
            throttle = 1.0 - (js.get_axis(WHEEL_AXIS_THROTTLE) + 1.0) / 2.0
            brake = 1.0 - (js.get_axis(WHEEL_AXIS_BRAKE) + 1.0) / 2.0

        if throttle < WHEEL_PEDAL_DEADZONE:
            throttle = 0.0
        if brake < WHEEL_PEDAL_DEADZONE:
            brake = 0.0

        if not self._ackermann_enabled:
            self._control.steer = round(steer, 4)
            self._control.throttle = min(1.0, throttle)
            self._control.brake = min(1.0, brake)
            # Handbrake stays on the keyboard — no dedicated wheel button is mapped.
            self._control.hand_brake = pygame.key.get_pressed()[K_SPACE]
        else:
            self._ackermann_control.steer = round(steer, 4)
        self._steer_cache = steer

    def _parse_vehicle_keys(self, keys, milliseconds):
        if keys[K_UP] or keys[K_w]:
            if not self._ackermann_enabled:
                self._control.throttle = min(self._control.throttle + 0.1, 1.00)
            else:
                self._ackermann_control.speed += round(milliseconds * 0.005, 2) * self._ackermann_reverse
        else:
            if not self._ackermann_enabled:
                self._control.throttle = 0.0

        if keys[K_DOWN] or keys[K_s]:
            if not self._ackermann_enabled:
                self._control.brake = min(self._control.brake + 0.2, 1)
            else:
                self._ackermann_control.speed -= min(abs(self._ackermann_control.speed), round(milliseconds * 0.005, 2)) * self._ackermann_reverse
                self._ackermann_control.speed = max(0, abs(self._ackermann_control.speed)) * self._ackermann_reverse
        else:
            if not self._ackermann_enabled:
                self._control.brake = 0

        steer_increment = 5e-4 * milliseconds
        if keys[K_LEFT] or keys[K_a]:
            if self._steer_cache > 0:
                self._steer_cache = 0
            else:
                self._steer_cache -= steer_increment
        elif keys[K_RIGHT] or keys[K_d]:
            if self._steer_cache < 0:
                self._steer_cache = 0
            else:
                self._steer_cache += steer_increment
        else:
            self._steer_cache = 0.0
        self._steer_cache = min(0.7, max(-0.7, self._steer_cache))
        if not self._ackermann_enabled:
            self._control.steer = round(self._steer_cache, 1)
            self._control.hand_brake = keys[K_SPACE]
        else:
            self._ackermann_control.steer = round(self._steer_cache, 1)

    def _parse_walker_keys(self, keys, milliseconds, world):
        self._control.speed = 0.0
        if keys[K_DOWN] or keys[K_s]:
            self._control.speed = 0.0
        if keys[K_LEFT] or keys[K_a]:
            self._control.speed = .01
            self._rotation.yaw -= 0.08 * milliseconds
        if keys[K_RIGHT] or keys[K_d]:
            self._control.speed = .01
            self._rotation.yaw += 0.08 * milliseconds
        if keys[K_UP] or keys[K_w]:
            self._control.speed = world.player_max_speed_fast if pygame.key.get_mods() & KMOD_SHIFT else world.player_max_speed
        self._control.jump = keys[K_SPACE]
        self._rotation.yaw = round(self._rotation.yaw, 1)
        self._control.direction = self._rotation.get_forward_vector()

    @staticmethod
    def _is_quit_shortcut(key):
        return (key == K_ESCAPE) or (key == K_q and pygame.key.get_mods() & KMOD_CTRL)


# ==============================================================================
# -- HUD -----------------------------------------------------------------------
# ==============================================================================


class HUD(object):
    def __init__(self, width, height, show_speed=False):
        self.dim = (width, height)
        font = pygame.font.Font(pygame.font.get_default_font(), 20)
        font_name = 'courier' if os.name == 'nt' else 'mono'
        fonts = [x for x in pygame.font.get_fonts() if font_name in x]
        default_font = 'ubuntumono'
        mono = default_font if default_font in fonts else fonts[0]
        mono = pygame.font.match_font(mono)
        self._font_mono = pygame.font.Font(mono, 12 if os.name == 'nt' else 14)
        self._notifications = FadingText(font, (width, 40), (0, height - 40))
        self.help = HelpText(pygame.font.Font(mono, 16), width, height)
        self.server_fps = 0
        self.frame = 0
        self.simulation_time = 0
        self._show_info = False  # hide HUD initially
        self._info_text = []
        self._server_clock = pygame.time.Clock()

        # Speedometer, opt-in via --speed. Separate from the F1 debug panel above
        # on purpose: that panel shows twenty lines of diagnostics, which is not
        # something to put in front of a participant. This is the one number a
        # driver actually needs.
        #
        # Off by default because it is an addition to what participants have seen
        # so far: a speed readout is a real change to the driving task (it gives
        # them a precise instrument to regulate against), so turning it on should
        # be a decision, and one held constant across the whole study.
        #
        # Sized from the window rather than fixed, so it reads the same on the
        # 1280x720 development window and the fullscreen rig.
        self.speed_kmh = 0.0
        self.show_speed = show_speed
        speed_pt = max(28, int(height * 0.055))
        self._font_speed = pygame.font.Font(mono, speed_pt)
        self._font_speed_unit = pygame.font.Font(mono, max(12, speed_pt // 3))

        self._show_ackermann_info = False
        self._ackermann_control = carla.VehicleAckermannControl()

    def on_world_tick(self, timestamp):
        self._server_clock.tick()
        self.server_fps = self._server_clock.get_fps()
        self.frame = timestamp.frame
        self.simulation_time = timestamp.elapsed_seconds

    def tick(self, world, clock):
        self._notifications.tick(world, clock)

        # Speed is read BEFORE the early return below, because the speedometer
        # is always on while the debug panel (_show_info, F1) is off by default
        # and off entirely in --control test, which is what the study runs.
        try:
            vel = world.player.get_velocity()
            self.speed_kmh = 3.6 * math.sqrt(vel.x**2 + vel.y**2 + vel.z**2)
        except Exception:
            pass

        if not self._show_info:
            return
        t = world.player.get_transform()
        v = world.player.get_velocity()
        c = world.player.get_control()
        compass = world.imu_sensor.compass
        heading = 'N' if compass > 270.5 or compass < 89.5 else ''
        heading += 'S' if 90.5 < compass < 269.5 else ''
        heading += 'E' if 0.5 < compass < 179.5 else ''
        heading += 'W' if 180.5 < compass < 359.5 else ''
        colhist = world.collision_sensor.get_collision_history()
        collision = [colhist[x + self.frame - 200] for x in range(0, 200)]
        max_col = max(1.0, max(collision))
        collision = [x / max_col for x in collision]
        vehicles = world.world.get_actors().filter('vehicle.*')
        self._info_text = [
            'Server:  % 16.0f FPS' % self.server_fps,
            'Client:  % 16.0f FPS' % clock.get_fps(),
            '',
            'Vehicle: % 20s' % get_actor_display_name(world.player, truncate=20),
            'Map:     % 20s' % world.map.name.split('/')[-1],
            'Simulation time: % 12s' % datetime.timedelta(seconds=int(self.simulation_time)),
            '',
            'Speed:   % 15.0f km/h' % (3.6 * math.sqrt(v.x**2 + v.y**2 + v.z**2)),
            u'Compass:% 17.0f\N{DEGREE SIGN} % 2s' % (compass, heading),
            'Accelero: (%5.1f,%5.1f,%5.1f)' % (world.imu_sensor.accelerometer),
            'Gyroscop: (%5.1f,%5.1f,%5.1f)' % (world.imu_sensor.gyroscope),
            'Location:% 20s' % ('(% 5.1f, % 5.1f)' % (t.location.x, t.location.y)),
            'GNSS:% 24s' % ('(% 2.6f, % 3.6f)' % (world.gnss_sensor.lat, world.gnss_sensor.lon)),
            'Height:  % 18.0f m' % t.location.z,
            '']
        if isinstance(c, carla.VehicleControl):
            self._info_text += [
                ('Throttle:', c.throttle, 0.0, 1.0),
                ('Steer:', c.steer, -1.0, 1.0),
                ('Brake:', c.brake, 0.0, 1.0),
                ('Reverse:', c.reverse),
                ('Hand brake:', c.hand_brake),
                ('Manual:', c.manual_gear_shift),
                'Gear:        %s' % {-1: 'R', 0: 'N'}.get(c.gear, c.gear)]
            if self._show_ackermann_info:
                self._info_text += [
                    '',
                    'Ackermann Controller:',
                    '  Target speed: % 8.0f km/h' % (3.6*self._ackermann_control.speed),
                ]
        elif isinstance(c, carla.WalkerControl):
            self._info_text += [
                ('Speed:', c.speed, 0.0, 5.556),
                ('Jump:', c.jump)]
        self._info_text += [
            '',
            'Collision:',
            collision,
            '',
            'Number of vehicles: % 8d' % len(vehicles)]
        if len(vehicles) > 1:
            self._info_text += ['Nearby vehicles:']
            distance = lambda l: math.sqrt((l.x - t.location.x)**2 + (l.y - t.location.y)**2 + (l.z - t.location.z)**2)
            vehicles = [(distance(x.get_location()), x) for x in vehicles if x.id != world.player.id]
            for d, vehicle in sorted(vehicles, key=lambda vehicles: vehicles[0]):
                if d > 200.0:
                    break
                vehicle_type = get_actor_display_name(vehicle, truncate=22)
                self._info_text.append('% 4dm %s' % (d, vehicle_type))

    def show_ackermann_info(self, enabled):
        self._show_ackermann_info = enabled

    def update_ackermann_control(self, ackermann_control):
        self._ackermann_control = ackermann_control

    def toggle_info(self):
        self._show_info = not self._show_info

    def notification(self, text, seconds=2.0):
        self._notifications.set_text(text, seconds=seconds)

    def error(self, text):
        self._notifications.set_text('Error: %s' % text, (255, 0, 0))

    def render(self, display):
        if self._show_info:
            info_surface = pygame.Surface((220, self.dim[1]))
            info_surface.set_alpha(100)
            display.blit(info_surface, (0, 0))
            v_offset = 4
            bar_h_offset = 100
            bar_width = 106
            for item in self._info_text:
                if v_offset + 18 > self.dim[1]:
                    break
                if isinstance(item, list):
                    if len(item) > 1:
                        points = [(x + 8, v_offset + 8 + (1.0 - y) * 30) for x, y in enumerate(item)]
                        pygame.draw.lines(display, (255, 136, 0), False, points, 2)
                    item = None
                    v_offset += 18
                elif isinstance(item, tuple):
                    if isinstance(item[1], bool):
                        rect = pygame.Rect((bar_h_offset, v_offset + 8), (6, 6))
                        pygame.draw.rect(display, (255, 255, 255), rect, 0 if item[1] else 1)
                    else:
                        rect_border = pygame.Rect((bar_h_offset, v_offset + 8), (bar_width, 6))
                        pygame.draw.rect(display, (255, 255, 255), rect_border, 1)
                        f = (item[1] - item[2]) / (item[3] - item[2])
                        if item[2] < 0.0:
                            rect = pygame.Rect((bar_h_offset + f * (bar_width - 6), v_offset + 8), (6, 6))
                        else:
                            rect = pygame.Rect((bar_h_offset, v_offset + 8), (f * bar_width, 6))
                        pygame.draw.rect(display, (255, 255, 255), rect)
                    item = item[0]
                if item:  # At this point has to be a str.
                    surface = self._font_mono.render(item, True, (255, 255, 255))
                    display.blit(surface, (8, v_offset))
                v_offset += 18
        self._notifications.render(display)
        self._render_speed(display)
        self.help.render(display)

    def _render_speed(self, display):
        """Draw the speed readout as a HUD element over the steering wheel.

        VERTICAL: the block's BOTTOM edge sits at exactly 0.8 of window height
        (``y`` is that line minus the glyph height), anchoring it to the base of
        the windshield rather than the bottom screen edge — that's where the
        wheel sits in the driver-seat camera, and it mimics where a real HUD
        projects speed: just above the dash, not down on the hood where it is
        easy to miss.

        HORIZONTAL: ``(dim[0] - block_w) // 3.5`` puts it LEFT OF CENTRE, at
        roughly 26-27 % of the window width — measured 27.2 % at "8" and 26.1 %
        at "120", identically at 720p, 1080p and 1440p. It is *not* centred and
        *not* in the bottom-right corner, whatever this docstring and the
        ``--speed`` help text used to say (both corrected 2026-08-19; the
        placement had moved and neither followed).

        KNOWN WART: because ``x`` is derived from ``block_w``, the whole block
        DRIFTS LEFT as the number gains digits — about 1.1 % of window width
        between a one- and a three-digit speed, i.e. 14 px at 720p, visible as a
        sideways shift each time the driver crosses 10 or 100 km/h. Only the
        unit was stabilised against this (see the blit below); the block was
        not. Anything positioned RELATIVE to this readout inherits the drift, so
        anchor to absolute coordinates instead — see the call-event panel in
        ``docs/live_study_setup.md`` §6.1.
        """
        if not self.show_speed:
            return

        value = self._font_speed.render('%d' % round(self.speed_kmh), True,
                                        (255, 255, 255))
        unit = self._font_speed_unit.render('km/h', True, (220, 220, 220))

        pad = max(6, int(self.dim[0] * 0.006))
        block_w = value.get_width() + pad + unit.get_width()
        block_h = value.get_height()
        x = (self.dim[0] - block_w) // 4
        y = int(self.dim[1] * 0.8) - block_h

        # Dimmed plate behind the digits: the camera view is arbitrary and white
        # text over a bright road surface is unreadable exactly when the driver
        # is looking for it.
        """
        plate = pygame.Surface((block_w + 2 * pad, block_h + pad))
        plate.set_alpha(110)
        plate.fill((0, 0, 0))
        display.blit(plate, (x - pad, y - pad // 2))
        """

        display.blit(value, (x, y))
        # Unit sits on the digits' baseline rather than centred, so it does not
        # bounce as the number changes width between 9 and 10 km/h.
        display.blit(unit, (x + value.get_width() + pad,
                            y + block_h - unit.get_height() - 2))


# ==============================================================================
# -- FadingText ----------------------------------------------------------------
# ==============================================================================


class FadingText(object):
    def __init__(self, font, dim, pos):
        self.font = font
        self.dim = dim
        self.pos = pos
        self.seconds_left = 0
        self.surface = pygame.Surface(self.dim)

    def set_text(self, text, color=(255, 255, 255), seconds=2.0):
        text_texture = self.font.render(text, True, color)
        self.surface = pygame.Surface(self.dim)
        self.seconds_left = seconds
        self.surface.fill((0, 0, 0, 0))
        self.surface.blit(text_texture, (10, 11))

    def tick(self, _, clock):
        delta_seconds = 1e-3 * clock.get_time()
        self.seconds_left = max(0.0, self.seconds_left - delta_seconds)
        self.surface.set_alpha(500.0 * self.seconds_left)

    def render(self, display):
        display.blit(self.surface, self.pos)


# ==============================================================================
# -- HelpText ------------------------------------------------------------------
# ==============================================================================


class HelpText(object):
    """Helper class to handle text output using pygame"""
    def __init__(self, font, width, height):
        lines = __doc__.split('\n')
        self.font = font
        self.line_space = 18
        self.dim = (780, len(lines) * self.line_space + 12)
        self.pos = (0.5 * width - 0.5 * self.dim[0], 0.5 * height - 0.5 * self.dim[1])
        self.seconds_left = 0
        self.surface = pygame.Surface(self.dim)
        self.surface.fill((0, 0, 0, 0))
        for n, line in enumerate(lines):
            text_texture = self.font.render(line, True, (255, 255, 255))
            self.surface.blit(text_texture, (22, n * self.line_space))
            self._render = False
        self.surface.set_alpha(220)

    def toggle(self):
        self._render = not self._render

    def render(self, display):
        if self._render:
            display.blit(self.surface, self.pos)


# ==============================================================================
# -- CollisionSensor -----------------------------------------------------------
# ==============================================================================


class CollisionSensor(object):
    def __init__(self, parent_actor, hud, control_mode='test'):
        self.sensor = None
        self.history = []
        self._parent = parent_actor
        self.hud = hud
        self.control_mode = control_mode
        world = self._parent.get_world()
        bp = world.get_blueprint_library().find('sensor.other.collision')
        self.sensor = world.spawn_actor(bp, carla.Transform(), attach_to=self._parent)
        # We need to pass the lambda a weak reference to self to avoid circular
        # reference.
        weak_self = weakref.ref(self)
        self.sensor.listen(lambda event: CollisionSensor._on_collision(weak_self, event))

    def get_collision_history(self):
        history = collections.defaultdict(int)
        for frame, intensity in self.history:
            history[frame] += intensity
        return history

    @staticmethod
    def _on_collision(weak_self, event):
        self = weak_self()
        if not self:
            return
        actor_type = get_actor_display_name(event.other_actor)
        if self.control_mode == 'full':
            self.hud.notification('Collision with %r' % actor_type)
        impulse = event.normal_impulse
        intensity = math.sqrt(impulse.x**2 + impulse.y**2 + impulse.z**2)
        self.history.append((event.frame, intensity))
        if len(self.history) > 4000:
            self.history.pop(0)


# ==============================================================================
# -- LaneInvasionSensor --------------------------------------------------------
# ==============================================================================


class LaneInvasionSensor(object):
    def __init__(self, parent_actor, hud, control_mode='test'):
        self.sensor = None

        # If the spawn object is not a vehicle, we cannot use the Lane Invasion Sensor
        if parent_actor.type_id.startswith("vehicle."):
            self._parent = parent_actor
            self.hud = hud
            self.control_mode = control_mode
            world = self._parent.get_world()
            bp = world.get_blueprint_library().find('sensor.other.lane_invasion')
            self.sensor = world.spawn_actor(bp, carla.Transform(), attach_to=self._parent)
            # We need to pass the lambda a weak reference to self to avoid circular
            # reference.
            weak_self = weakref.ref(self)
            self.sensor.listen(lambda event: LaneInvasionSensor._on_invasion(weak_self, event))

    @staticmethod
    def _on_invasion(weak_self, event):
        self = weak_self()
        if not self:
            return
        lane_types = set(x.type for x in event.crossed_lane_markings)
        text = ['%r' % str(x).split()[-1] for x in lane_types]
        if self.control_mode == 'full':
            self.hud.notification('Crossed line %s' % ' and '.join(text))


# ==============================================================================
# -- GnssSensor ----------------------------------------------------------------
# ==============================================================================


class GnssSensor(object):
    def __init__(self, parent_actor):
        self.sensor = None
        self._parent = parent_actor
        self.lat = 0.0
        self.lon = 0.0
        world = self._parent.get_world()
        bp = world.get_blueprint_library().find('sensor.other.gnss')
        self.sensor = world.spawn_actor(bp, carla.Transform(carla.Location(x=1.0, z=2.8)), attach_to=self._parent)
        # We need to pass the lambda a weak reference to self to avoid circular
        # reference.
        weak_self = weakref.ref(self)
        self.sensor.listen(lambda event: GnssSensor._on_gnss_event(weak_self, event))

    @staticmethod
    def _on_gnss_event(weak_self, event):
        self = weak_self()
        if not self:
            return
        self.lat = event.latitude
        self.lon = event.longitude


# ==============================================================================
# -- IMUSensor -----------------------------------------------------------------
# ==============================================================================


class IMUSensor(object):
    def __init__(self, parent_actor):
        self.sensor = None
        self._parent = parent_actor
        self.accelerometer = (0.0, 0.0, 0.0)
        self.gyroscope = (0.0, 0.0, 0.0)
        self.compass = 0.0
        world = self._parent.get_world()
        bp = world.get_blueprint_library().find('sensor.other.imu')
        self.sensor = world.spawn_actor(
            bp, carla.Transform(), attach_to=self._parent)
        # We need to pass the lambda a weak reference to self to avoid circular
        # reference.
        weak_self = weakref.ref(self)
        self.sensor.listen(
            lambda sensor_data: IMUSensor._IMU_callback(weak_self, sensor_data))

    @staticmethod
    def _IMU_callback(weak_self, sensor_data):
        self = weak_self()
        if not self:
            return
        limits = (-99.9, 99.9)
        self.accelerometer = (
            max(limits[0], min(limits[1], sensor_data.accelerometer.x)),
            max(limits[0], min(limits[1], sensor_data.accelerometer.y)),
            max(limits[0], min(limits[1], sensor_data.accelerometer.z)))
        self.gyroscope = (
            max(limits[0], min(limits[1], math.degrees(sensor_data.gyroscope.x))),
            max(limits[0], min(limits[1], math.degrees(sensor_data.gyroscope.y))),
            max(limits[0], min(limits[1], math.degrees(sensor_data.gyroscope.z))))
        self.compass = math.degrees(sensor_data.compass)


# ==============================================================================
# -- RadarSensor ---------------------------------------------------------------
# ==============================================================================


class RadarSensor(object):
    def __init__(self, parent_actor):
        self.sensor = None
        self._parent = parent_actor
        bound_x = 0.5 + self._parent.bounding_box.extent.x
        bound_y = 0.5 + self._parent.bounding_box.extent.y
        bound_z = 0.5 + self._parent.bounding_box.extent.z

        self.velocity_range = 7.5 # m/s
        world = self._parent.get_world()
        self.debug = world.debug
        bp = world.get_blueprint_library().find('sensor.other.radar')
        bp.set_attribute('horizontal_fov', str(35))
        bp.set_attribute('vertical_fov', str(20))
        self.sensor = world.spawn_actor(
            bp,
            carla.Transform(
                carla.Location(x=bound_x + 0.05, z=bound_z+0.05),
                carla.Rotation(pitch=5)),
            attach_to=self._parent)
        # We need a weak reference to self to avoid circular reference.
        weak_self = weakref.ref(self)
        self.sensor.listen(
            lambda radar_data: RadarSensor._Radar_callback(weak_self, radar_data))

    @staticmethod
    def _Radar_callback(weak_self, radar_data):
        self = weak_self()
        if not self:
            return
        # To get a numpy [[vel, altitude, azimuth, depth],...[,,,]]:
        # points = np.frombuffer(radar_data.raw_data, dtype=np.dtype('f4'))
        # points = np.reshape(points, (len(radar_data), 4))

        current_rot = radar_data.transform.rotation
        for detect in radar_data:
            azi = math.degrees(detect.azimuth)
            alt = math.degrees(detect.altitude)
            # The 0.25 adjusts a bit the distance so the dots can
            # be properly seen
            fw_vec = carla.Vector3D(x=detect.depth - 0.25)
            carla.Transform(
                carla.Location(),
                carla.Rotation(
                    pitch=current_rot.pitch + alt,
                    yaw=current_rot.yaw + azi,
                    roll=current_rot.roll)).transform(fw_vec)

            def clamp(min_v, max_v, value):
                return max(min_v, min(value, max_v))

            norm_velocity = detect.velocity / self.velocity_range # range [-1, 1]
            r = int(clamp(0.0, 1.0, 1.0 - norm_velocity) * 255.0)
            g = int(clamp(0.0, 1.0, 1.0 - abs(norm_velocity)) * 255.0)
            b = int(abs(clamp(- 1.0, 0.0, - 1.0 - norm_velocity)) * 255.0)
            self.debug.draw_point(
                radar_data.transform.location + fw_vec,
                size=0.075,
                life_time=0.06,
                persistent_lines=False,
                color=carla.Color(r, g, b))

# ==============================================================================
# -- CameraManager -------------------------------------------------------------
# ==============================================================================


class CameraManager(object):
    def __init__(self, parent_actor, hud, gamma_correction, render_scale=1.0):
        self.sensor = None
        self.surface = None
        self._parent = parent_actor
        self.hud = hud
        self.recording = False

        # Camera resolution, decoupled from the window it is shown in.
        #
        # This is the main lever on how fast the SERVER can produce a frame, and
        # under --sync the server's frame rate is the ceiling on the whole
        # simulation: ask for ticks faster than it can render and simulated time
        # falls behind real time, which the participant sees as everything moving
        # in slow motion. At fullscreen the camera was rendering 1920x1080 twice
        # over -- once for the spectator window and once for this sensor -- and
        # then _parse_image copied that 6 MB buffer several times per frame in
        # numpy (reshape, channel reversal, swapaxes, make_surface). Both costs
        # fall roughly with the pixel count, so halving the scale is close to a
        # 4x saving on each.
        #
        # The image is scaled back up at blit time, so the view still fills the
        # window; it is softer, and on a driving task that is a far better trade
        # than slow motion. 1.0 keeps the previous behaviour exactly.
        self.render_scale = max(0.1, min(1.0, float(render_scale)))
        self.render_dim = (max(64, int(hud.dim[0] * self.render_scale)),
                           max(64, int(hud.dim[1] * self.render_scale)))
        # Reused across frames so the upscale does not allocate a new surface 20
        # times a second.
        self._scaled = None
        bound_x = 0.5 + self._parent.bounding_box.extent.x
        bound_y = 0.5 + self._parent.bounding_box.extent.y
        bound_z = 0.5 + self._parent.bounding_box.extent.z
        Attachment = carla.AttachmentType

        if not self._parent.type_id.startswith("walker.pedestrian"):
            # The drive-seat eye point sits just FORWARD of the rim (x=0.35
            # rather than the natural 0.25) so the car's own steering wheel is
            # out of frame. CARLA models the interior but never animates the
            # wheel — it stays dead straight through full lock, verified on
            # 0.10.0 — and the API cannot hide one part of a vehicle mesh, so
            # moving the camera past it is the only way to drop it. A rim that
            # ignores the driver's hands reads as more broken than no rim at
            # all, and on this rig the real one is in their hands anyway.
            # Raising z instead does not work: by 1.35 the camera is inside the
            # headliner and the roof lining eats the top of the frame.
            self._camera_transforms = [
                (carla.Transform(carla.Location(x=0.35, y=-0.33, z=1.21), carla.Rotation(pitch=-2.0)), Attachment.Rigid),  # drive-seat view
                (carla.Transform(carla.Location(x=-2.0*bound_x, y=+0.0*bound_y, z=2.0*bound_z), carla.Rotation(pitch=8.0)), Attachment.SpringArmGhost),
                (carla.Transform(carla.Location(x=+0.8*bound_x, y=+0.0*bound_y, z=1.3*bound_z)), Attachment.Rigid),
                (carla.Transform(carla.Location(x=+1.9*bound_x, y=+1.0*bound_y, z=1.2*bound_z)), Attachment.SpringArmGhost),
                (carla.Transform(carla.Location(x=-2.8*bound_x, y=+0.0*bound_y, z=4.6*bound_z), carla.Rotation(pitch=6.0)), Attachment.SpringArmGhost),
                (carla.Transform(carla.Location(x=-1.0, y=-1.0*bound_y, z=0.4*bound_z)), Attachment.Rigid)]
        else:
            self._camera_transforms = [
                (carla.Transform(carla.Location(x=-2.5, z=0.0), carla.Rotation(pitch=-8.0)), Attachment.SpringArmGhost),
                (carla.Transform(carla.Location(x=1.6, z=1.7)), Attachment.Rigid),
                (carla.Transform(carla.Location(x=2.5, y=0.5, z=0.0), carla.Rotation(pitch=-8.0)), Attachment.SpringArmGhost),
                (carla.Transform(carla.Location(x=-4.0, z=2.0), carla.Rotation(pitch=6.0)), Attachment.SpringArmGhost),
                (carla.Transform(carla.Location(x=0, y=-2.5, z=-0.0), carla.Rotation(yaw=90.0)), Attachment.Rigid)]

        self.transform_index = 1
        self.sensors = [
            ['sensor.camera.rgb', cc.Raw, 'Camera RGB', {}],
            ['sensor.camera.depth', cc.Raw, 'Camera Depth (Raw)', {}],
            ['sensor.camera.depth', cc.Depth, 'Camera Depth (Gray Scale)', {}],
            ['sensor.camera.depth', cc.LogarithmicDepth, 'Camera Depth (Logarithmic Gray Scale)', {}],
            ['sensor.camera.semantic_segmentation', cc.Raw, 'Camera Semantic Segmentation (Raw)', {}],
            ['sensor.camera.semantic_segmentation', cc.CityScapesPalette, 'Camera Semantic Segmentation (CityScapes Palette)', {}],
            ['sensor.camera.instance_segmentation', cc.Raw, 'Camera Instance Segmentation (Raw)', {}],
            ['sensor.lidar.ray_cast', None, 'Lidar (Ray-Cast)', {'range': '50'}],
            ['sensor.lidar.ray_cast_semantic', None, 'Semantic Lidar (Ray-Cast)', {'range': '50'}],
            ['sensor.camera.rgb', cc.Raw, 'Camera RGB Distorted',
                {'lens_circle_multiplier': '3.0',
                'lens_circle_falloff': '3.0',
                'chromatic_aberration_intensity': '0.5',
                'chromatic_aberration_offset': '0'}],
            ['sensor.camera.optical_flow', cc.Raw, 'Optical Flow', {}],
            ['sensor.camera.normals', cc.Raw, 'Camera Normals', {}],
        ]
        world = self._parent.get_world()
        bp_library = world.get_blueprint_library()
        for item in self.sensors:
            bp = bp_library.find(item[0])
            if item[0].startswith('sensor.camera'):
                bp.set_attribute('image_size_x', str(self.render_dim[0]))
                bp.set_attribute('image_size_y', str(self.render_dim[1]))
                if bp.has_attribute('gamma'):
                    bp.set_attribute('gamma', str(gamma_correction))
                for attr_name, attr_value in item[3].items():
                    bp.set_attribute(attr_name, attr_value)
            elif item[0].startswith('sensor.lidar'):
                self.lidar_range = 50

                for attr_name, attr_value in item[3].items():
                    bp.set_attribute(attr_name, attr_value)
                    if attr_name == 'range':
                        self.lidar_range = float(attr_value)

            item.append(bp)
        self.index = None

    def toggle_camera(self):
        self.transform_index = (self.transform_index + 1) % len(self._camera_transforms)
        self.set_sensor(self.index, notify=False, force_respawn=True)

    def set_sensor(self, index, notify=True, force_respawn=False):
        index = index % len(self.sensors)
        needs_respawn = True if self.index is None else \
            (force_respawn or (self.sensors[index][2] != self.sensors[self.index][2]))
        if needs_respawn:
            if self.sensor is not None:
                self.sensor.destroy()
                self.surface = None
            self.sensor = self._parent.get_world().spawn_actor(
                self.sensors[index][-1],
                self._camera_transforms[self.transform_index][0],
                attach_to=self._parent,
                attachment_type=self._camera_transforms[self.transform_index][1])
            # We need to pass the lambda a weak reference to self to avoid
            # circular reference.
            weak_self = weakref.ref(self)
            self.sensor.listen(lambda image: CameraManager._parse_image(weak_self, image))
        if notify:
            self.hud.notification(self.sensors[index][2])
        self.index = index

    def next_sensor(self):
        self.set_sensor(self.index + 1)

    def toggle_recording(self):
        self.recording = not self.recording
        self.hud.notification('Recording %s' % ('On' if self.recording else 'Off'))

    def render(self, display):
        if self.surface is None:
            return
        if self.surface.get_size() == display.get_size():
            display.blit(self.surface, (0, 0))
            return
        # Upscale a reduced-resolution camera frame to fill the window.
        # transform.scale is a C blit, unlike smoothscale, which is markedly
        # slower and whose softening buys nothing on a moving driving scene. The
        # destination surface is reused rather than reallocated each frame.
        if self._scaled is None or self._scaled.get_size() != display.get_size():
            self._scaled = pygame.Surface(display.get_size()).convert()
        pygame.transform.scale(self.surface, display.get_size(), self._scaled)
        display.blit(self._scaled, (0, 0))

    @staticmethod
    def _parse_image(weak_self, image):
        self = weak_self()
        if not self:
            return
        if self.sensors[self.index][0] == 'sensor.lidar.ray_cast':
            points = np.frombuffer(image.raw_data, dtype=np.dtype('f4'))
            points = np.reshape(points, (int(points.shape[0] / 4), 4))
            lidar_data = np.array(points[:, :2])
            lidar_data *= min(self.hud.dim) / (2.0 * self.lidar_range)
            lidar_data += (0.5 * self.hud.dim[0], 0.5 * self.hud.dim[1])
            lidar_data = np.fabs(lidar_data)  # pylint: disable=E1111
            lidar_data = lidar_data.astype(np.int32)
            lidar_data = np.reshape(lidar_data, (-1, 2))
            lidar_img_size = (self.hud.dim[0], self.hud.dim[1], 3)
            lidar_img = np.zeros((lidar_img_size), dtype=np.uint8)
            lidar_img[tuple(lidar_data.T)] = (255, 255, 255)
            self.surface = pygame.surfarray.make_surface(lidar_img)
        elif self.sensors[self.index][0] == 'sensor.lidar.ray_cast_semantic':
            points = np.frombuffer(image.raw_data, dtype=np.dtype('f4'))
            points = np.reshape(points, (int(points.shape[0] / 6), 6))
            lidar_data = np.array(points[:, :2])
            lidar_data *= min(self.hud.dim) / (2.0 * self.lidar_range)
            lidar_data += (0.5 * self.hud.dim[0], 0.5 * self.hud.dim[1])
            lidar_data = lidar_data.astype(np.int32)
            lidar_data = np.reshape(lidar_data, (-1, 2))
            lidar_img_size = (self.hud.dim[0], self.hud.dim[1], 3)
            lidar_img = np.zeros((lidar_img_size), dtype=np.uint8)
            for i in range(len(image)):
                point = lidar_data[i]
                lidar_tag = image[i].object_tag
                lidar_img[tuple(point.T)] = OBJECT_TO_COLOR[int(lidar_tag)]
            self.surface = pygame.surfarray.make_surface(lidar_img)
        elif self.sensors[self.index][0].startswith('sensor.camera.optical_flow'):
            image = image.get_color_coded_flow()
            array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
            array = np.reshape(array, (image.height, image.width, 4))
            array = array[:, :, :3]
            array = array[:, :, ::-1]
            self.surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))
        else:
            image.convert(self.sensors[self.index][1])
            array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
            array = np.reshape(array, (image.height, image.width, 4))
            array = array[:, :, :3]
            array = array[:, :, ::-1]
            self.surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))
        if self.recording:
            image.save_to_disk('_out/%08d' % image.frame)


# ==============================================================================
# -- game_loop() ---------------------------------------------------------------
# ==============================================================================


def _resolve_popup_input(args, has_wheel):
    """Decide which interface answers the LoA popup.

    No flag means the keyboard, on every rig — the presence of a wheel used to
    decide this, which made the labelling interface an accident of the machine
    rather than a property of the experiment. Asking for the wheel on a rig that
    has none would leave the participant staring at a popup they cannot answer
    with the scene frozen behind it, so that falls back loudly instead of
    failing.
    """
    mode = getattr(args, 'popup_input', None) or POPUP_INPUT_DEFAULT
    if mode == POPUP_INPUT_WHEEL and not has_wheel:
        print('[WARN] --wheel-input, but no steering wheel is bound — the LoA '
              'popup falls back to the keyboard (0-4 tick, ENTER confirms).')
        return POPUP_INPUT_KEYBOARD
    return mode


def _should_wait_for_provoice(args):
    """Whether the LoA windows hold until ProVoice's first frame.

    Only a run that actually records labels alongside ProVoice has anything to
    gain: --test-popup teaches the control with no ProVoice process at all and
    would wait forever, and --no-popup opens no windows to protect.

    A STUDY BLOCK waits too, despite forcing --no-popup. It has no labelling
    windows to protect, but it has something stronger: every call is served by
    ProVoice, so a block whose clock started first would burn its first minutes
    -- and its first calls -- against a model that is not answering yet, and
    log them as skipped. --random-loa / --test-calls are the exception, since
    those generate the LoA locally and need no ProVoice at all.
    """
    if getattr(args, "study", False) or getattr(args, "short_trial", False):
        if getattr(args, "random_loa", False) or getattr(args, "test_calls", False):
            return False
        return args.popup_wait_timeout > 0
    return (not args.no_popup and not args.test_popup
            and args.popup_wait_timeout > 0)


def _advance_or_close(loa_popup, world, now_ms):
    """Move to the next prompt of this window, or end the window.

    The scene stays frozen between the prompts of one window: they label the same
    20 s of driving, so letting the car roll on between them would put the second
    answer about a stretch the driver has already driven past.
    """
    if loa_popup.advance(now_ms):
        return
    loa_popup.close(now_ms)
    _set_world_frozen(world, False)


def game_loop(args):
     # SDL's default is to MINIMISE a fullscreen window whenever it loses
    # keyboard focus, and nothing here would ever restore it -- so any window
    # that flashes into the foreground for a few hundred milliseconds drops the
    # drive into the taskbar for the rest of the run, with no user input and
    # nothing left on screen to explain it.
    #
    # Measured 2026-07-30: two scheduled tasks belonging to OTHER projects on
    # this machine (afd_watchdog and MicrogestureExtractionWatchdog) fire every
    # 10 minutes and launch `powershell -WindowStyle Hidden`. On Windows 11 the
    # default console host is Windows Terminal, which -WindowStyle Hidden cannot
    # hide, so a real window is created and ACTIVATED before the script exits.
    # Because the trigger sits on a wall-clock grid unrelated to run start
    # (minute :05/:09/:15/:19/...), a participant run catches at most one and it
    # looks random. With this hint set the window keeps its pixels: focus is
    # still lost for ~100-270 ms, then automatically regained.
    #
    # setdefault, not assignment: an operator who deliberately exported the
    # variable keeps their choice.
    os.environ.setdefault('SDL_VIDEO_MINIMIZE_ON_FOCUS_LOSS', '0')
    # BEFORE pygame.init(), which is what opens the audio device: pre_init only
    # supplies the parameters that init will use, so afterwards it is a no-op and
    # the mixer buffer -- the one parameter that cannot be changed later -- keeps
    # SDL's default. Harmless when --ambient-gain is 0; the device is only
    # actually opened if Ambience is asked for sound.
    configure_mixer()
    pygame.init()
    pygame.font.init()
    world = None
    ambience = None
    # Pre-declared for the same reason as ambience: the finally block below
    # touches it, and an exception before the try body assigns it would turn a
    # real error into a NameError that hides it.
    call_preview = None
    study = None
    study_ended = False
    original_settings = None

    try:
        client = carla.Client(args.host, args.port)
        client.set_timeout(20.0)

        sim_world = client.get_world()
        traffic_manager = client.get_trafficmanager(args.tm_port)

        # --sync means "the world is synchronous and ANOTHER client owns the
        # clock", not "I tick". Drive neither applies world settings nor calls
        # world.tick() nor synchronises a traffic manager -- all three belong to
        # src/drive/fixed_npc_traffic.py --sync, which is the process that owns
        # the NPCs' traffic manager and therefore the only process CARLA lets
        # drive it (that file's docstring has the mechanism).
        #
        # This is the inversion of what this block used to do. It used to set the
        # world synchronous and synchronise the traffic manager on port 8000,
        # which has no NPC registered to it, and then tick -- leaving the port
        # 9000 traffic manager that drives the eleven cars unticked, so they all
        # stopped dead.
        if args.sync:
            if not sim_world.get_settings().synchronous_mode:
                print("[WARN] --sync was passed but the world is NOT in "
                      "synchronous mode. Drive does not set it: launch "
                      "src/drive/fixed_npc_traffic.py --sync (start_experiment.py "
                      "--sync does both). Continuing, but this loop will wait on "
                      "a clock nobody is driving and time out.")
            else:
                print("[INFO] --sync: pacing to the external world clock "
                      "(fixed step %.3f s), TM port %d."
                      % (sim_world.get_settings().fixed_delta_seconds or 0.0,
                         args.tm_port))

        if args.autopilot and not sim_world.get_settings().synchronous_mode:
            print("WARNING: You are currently in asynchronous mode and could "
                  "experience some issues with the traffic simulation")

        flags = pygame.HWSURFACE | pygame.DOUBLEBUF
        if args.fullscreen:
            # Adopt the desktop resolution BEFORE anything else reads
            # args.width/height — the HUD, the LoA popup, the start overlay and
            # the CARLA camera sensor are all sized from them just below, so a
            # stale 1280x720 here would letterbox the view and mis-centre the
            # popup text.
            info = pygame.display.Info()
            args.width, args.height = info.current_w, info.current_h
            flags |= pygame.FULLSCREEN
            print("[INFO] Fullscreen at %dx%d" % (args.width, args.height))
        display = pygame.display.set_mode((args.width, args.height), flags)
        display.fill((0,0,0))
        pygame.display.flip()

        hud = HUD(args.width, args.height, show_speed=args.speed)
        # Started here, before the start overlay and therefore before ProVoice's
        # 60 s calibration, and left running for the whole session. Noise onset
        # is an arousal event: starting it partway through would move the HR/RR
        # baseline that hr_delta and rr_delta are normalised against, for that
        # participant only.
        ambience = Ambience(gain=args.ambient_gain, seed=args.ambient_seed,
                            assets_dir=args.ambient_dir)
        # Trial mode only (--call-preview). Sized from args.width/height, which
        # fullscreen has already overwritten with the desktop resolution above,
        # so the panel lands where it will in the study rather than where a
        # stale 1280x720 would put it.
        study_on = bool(args.study or args.short_trial)
        if call_preview is not None:
            print('[call-preview] press 0-4 to stage a call at that LoA, '
                  'A / B to answer. Chrome=%s, panel %dx%d at (%d, %d). '
                  'Nothing is logged.'
                  % (call_preview.chrome, call_preview.rect.w,
                     call_preview.rect.h, call_preview.rect.x,
                     call_preview.rect.y))
        world = World(sim_world, hud, traffic_manager, args)
        controller = KeyboardControl(world, args.autopilot, args.control,
                                     use_wheel=not args.no_wheel)
        # Only advertise wheel controls in the popup if a wheel is actually bound.
        popup_input = _resolve_popup_input(args, controller.has_wheel)
        # Same wheel-vs-keyboard resolution the label pop-up uses, so the call
        # panel names the control the driver is actually holding.
        call_preview = (CallEvent((args.width, args.height),
                                  chrome=args.call_chrome,
                                  input_mode=('wheel'
                                              if popup_input == POPUP_INPUT_WHEEL
                                              else 'keyboard'))
                        if (args.call_preview or study_on) else None)
        loa_popup = LoASelectionPopup(
            args.width, args.height,
            interval_seconds=TEST_POPUP_INTERVAL_S if args.test_popup else 20,
            enabled=not args.no_popup,
            # --popup-immediate skips only the first window's wait, keeping the
            # real 20 s interval and everything else about a normal run. That is
            # what separates it from --test-popup, which also shortens the
            # interval and is a different mode entirely.
            open_immediately=args.test_popup or args.popup_immediate,
            input_mode=popup_input,
            function_name=getattr(args, 'functionname', ''),
            # --test-popup gets the pool too, so practice reproduces the shape of
            # a real window: TWO consecutive prompts about two different
            # functions, scene frozen across both. Practising a single prompt
            # would leave the participant meeting the second one for the first
            # time during data collection, which is the one moment the control
            # has to be automatic. The pool costs nothing here — practice
            # answers are never logged (see disable_user_loa_logging), so the
            # function they were about is not recorded either way.
            function_pool=(RANDOM_FUNCTION_POOL
                           if (args.random_function or args.test_popup) else ()))
        start_overlay = StartScreenOverlay(args.width, args.height,
                                           has_wheel=controller.has_wheel)
        started = False
        wait_for_provoice = _should_wait_for_provoice(args)
        popups_armed = False
        # Set once the remote collection_started signal is seen; the extra
        # PROVOICE_READY_POPUP_DELAY_S hold runs from provoice_ready_ms, after
        # which loa_popup.start() actually fires. Local runs and the
        # --popup-wait-timeout fallback never set this — they arm immediately,
        # as before.
        provoice_ready_pending = False
        provoice_ready_ms = 0
        session_id = _current_session_id(getattr(args, 'session_id', ''))
        # Same id the labels are written under, which is exactly what has to be
        # matched in ProVoice's log — a second _current_session_id() call would
        # mint a different one whenever no id was passed in.
        status_path = getattr(args, 'provoice_status_file', None) or None
        if status_path:
            status_path = os.path.abspath(status_path)
        # AFTER session_id and status_path: the LoA source is scoped to this
        # session and reads that file, so building it any earlier -- next to
        # call_preview, where it started -- raised UnboundLocalError on both.
        if study_on:
            cfg = dict(SHORT_TRIAL if args.short_trial else FULL_TRIAL)
            study = StudySession(
                LoASource(('sequence' if args.test_calls else
                           'random' if args.random_loa else 'bridge'),
                          status_path=status_path, session_id=session_id,
                          seed=args.study_seed,
                          max_age_ms=(args.loa_max_age * 1000.0
                                      if args.loa_max_age > 0 else None)),
                log_path=os.path.join(os.getcwd(), 'data', 'call_events.csv'),
                seed=args.study_seed, session_id=session_id,
                participantid=getattr(args, 'participantid', ''),
                block_idx=args.block_idx, k_condition=args.k_condition,
                spam_call_idx=(-1 if args.test_spam else args.spam_call),
                **cfg)
            print('[study] block %s condition %r: %d calls over %.0f min, '
                  '~%.0f s apart (jitter +/-%.0f s), LoA from %s.'
                  % (args.block_idx or '?', args.k_condition or '?',
                     cfg['n_calls'], cfg['duration_s'] / 60.0,
                     cfg['interval_s'], cfg['jitter_s'], study.source.mode))
            # Printed, but the console is not where it is RECORDED -- the index
            # rides in every call_events.csv row, so the analysis never has to
            # trust a scrollback buffer for which trial was the inverted one.
            print('[study] spam call: %s'
                  % ('EVERY call (--test-spam; not a study configuration)'
                     if study.spam_call_idx == -1 else
                     'none in this block' if study.spam_call_idx <= 0 else
                     'call %d of %d%s' % (study.spam_call_idx, cfg['n_calls'],
                                          ' (pinned)' if args.spam_call
                                          is not None else '')))
            if not args.random_loa:
                print('[study] a served LoA older than %.1f s will be refused '
                      '(--loa-max-age).' % args.loa_max_age
                      if args.loa_max_age > 0 else
                      '[study] WARNING staleness check DISABLED '
                      '(--loa-max-age 0): a stalled ProVoice will be served '
                      'silently.')
            if args.random_loa or args.test_calls:
                print('[study] WARNING the LoA is generated LOCALLY (%s), not '
                      'served by a model. Not a study configuration.'
                      % study.source.mode)

        provoice_watcher = ProVoiceReadyWatcher(session_id, args.popup_wait_timeout,
                                                status_path=status_path)
        end_watcher = ProVoiceEndWatcher(session_id, status_path)
        end_overlay = None
        # Set the moment 'provoice_ended' is read while a LoA prompt is on
        # screen: the window it belongs to still gets to finish (both prompts
        # answered or skipped) before the exit screen appears, so ending the
        # session never discards a label the driver was already answering.
        provoice_end_pending = False
        pending_end_reason = ''
        skipped_prompts = 0
        label_context = {
            'session_id': session_id,
            'participantid': getattr(args, 'participantid', ''),
            'environment': getattr(args, 'environment', ''),
            'secondary_task': getattr(args, 'secondary_task', ''),
            'functionname': getattr(args, 'functionname', 'Adjust seat positioning'),
            'emotion': getattr(args, 'emotion', ''),
            'modeltype': getattr(args, 'modeltype', ''),
            'state_model': getattr(args, 'state_model', ''),
            'w_fcd': getattr(args, 'w_fcd', ''),
            # effective_gain and source, not the requested values — see
            # USER_LOA_LABEL_COLUMNS. Both report what was actually played.
            'ambient_gain': ambience.effective_gain,
            'ambient_seed': args.ambient_seed,
            'ambient_source': ambience.source,
        }
        print(f"[INFO] Drive session_id={session_id}")
        if world._precip_mode == 'sun_rain':
            print("[INFO] --condition-sun-rain: precipitation starts at %.0f%% and "
                  "ramps down to 0%% over %.0f min, from the moment driving starts."
                  % (CONDITION_PEAK_PRECIPITATION, CONDITION_RAMP_MINUTES))
        elif world._precip_mode == 'rain_sun':
            print("[INFO] --condition-rain-sun: sunny for %.0f min, then "
                  "precipitation ramps up to %.0f%% over the following %.0f min."
                  % (CONDITION_SUN_HOLD_MINUTES, CONDITION_PEAK_PRECIPITATION,
                     CONDITION_RAMP_MINUTES))
        if args.no_popup:
            print("[INFO] Popups off: no LoA popups, no user labels written")
        elif args.test_popup:
            print(f"[INFO] Practice mode: popups every {TEST_POPUP_INTERVAL_S:.0f} s, "
                  f"selections are NOT logged")
        else:
            print("[INFO] User LoA labels will be written to data/user_loa_labels.csv")
        if status_path:
            print("[INFO] Remote ProVoice signals read from %s: the LoA windows "
                  "start %.0f s after collection_started, and the drive ends on "
                  "provoice_ended." % (status_path, PROVOICE_READY_POPUP_DELAY_S))
        if not args.no_popup:
            print("[INFO] LoA popup input: %s" % (
                'steering wheel (paddles move, front button ticks, CONFIRM submits)'
                if popup_input == POPUP_INPUT_WHEEL
                else 'keyboard (0-4 tick, same key again unticks, ENTER confirms)'))
            if loa_popup.prompts_per_window > 1:
                print("[INFO] %d prompts per window, each about a different function "
                      "drawn from: %s"
                      % (loa_popup.prompts_per_window, ', '.join(RANDOM_FUNCTION_POOL)))
            else:
                print("[INFO] 1 prompt per window about '%s'" % loa_popup.function_name)
            if wait_for_provoice and status_path:
                print("[INFO] The first window waits for the remote ProVoice's "
                      "collection_started signal via %s (giving up after %.0f s)"
                      % (status_path, args.popup_wait_timeout))
            elif wait_for_provoice:
                print("[INFO] The first window waits for ProVoice's first logged frame "
                      "(giving up after %.0f s)" % args.popup_wait_timeout)
            elif not args.test_popup:
                print("[WARN] Not waiting for ProVoice (--popup-wait-timeout 0): windows "
                      "opened before it logs have no driver-state data behind them")

        _await_world_tick(sim_world)

        clock = pygame.time.Clock()
        # True while the --sync world clock is not advancing, so the warning is
        # printed on the transition instead of once per frame.
        sync_stalled = False

        # Loop rate cap. Under --sync the world clock paces this loop, so the cap
        # only has to stay ABOVE the tick rate: at or below it, pygame would gate
        # the loop and the participant would get fewer frames than the simulation
        # is producing. It still matters while the clock is deliberately held for
        # a popup, where it is the only thing stopping a busy-spin.
        loop_cap = 60
        if args.sync:
            step = sim_world.get_settings().fixed_delta_seconds or 0.05
            loop_cap = max(60, int(round(1.0 / step)) + 10)
            print("[INFO] --sync: loop capped at %d Hz against a %.1f Hz world "
                  "clock." % (loop_cap, 1.0 / step))

        while True:
            # Wait for the tick BEFORE draining input, not after.
            #
            # This ordering is the difference between one step of control latency
            # and two. Input is read from the OS queue at the moment event.get()
            # is called; whatever is applied afterwards takes effect on the next
            # tick. With the wait placed after event.get() -- where the old
            # sim_world.tick() call sat -- the events were already up to a full
            # step old before the control derived from them was even applied, so
            # a steering input could take two steps to reach the wheels. At 20 Hz
            # that is 100 ms, which is squarely in the range a driver feels as
            # lag. Waiting first means input is always drained immediately after a
            # tick and lands on the very next one.
            if args.sync and not _drive_is_holding_clock(args, loa_popup,
                                                        end_overlay):
                if _await_world_tick(sim_world):
                    if sync_stalled:
                        sync_stalled = False
                        print("[INFO] World clock recovered.")
                elif not sync_stalled:
                    sync_stalled = True
                    print("[WARN] No world tick for %.0fs. The clock owner "
                          "(src/drive/fixed_npc_traffic.py --sync) has probably "
                          "died; the scene is frozen but Drive stays responsive "
                          "so the session can be ended cleanly."
                          % SYNC_STALL_TIMEOUT_S)
                    world.hud.notification('Simulation stalled - clock lost',
                                           seconds=8.0)

            clock.tick_busy_loop(loop_cap)
            now_ms = pygame.time.get_ticks()
            world.update_weather_schedule(now_ms)
            events = pygame.event.get()

            # ProVoice on the other machine has exited: from here the drive is
            # over. Checked BEFORE everything else in the tick so no popup
            # opens, no label is taken and no input is applied for a stretch of
            # driving that nothing is recording. Once entered, this state is
            # terminal -- the only way out is quitting.
            if end_overlay is None and not provoice_end_pending and end_watcher.poll(now_ms):
                provoice_end_pending = True
                pending_end_reason = end_watcher.reason()
                if not loa_popup.active:
                    print("[INFO] Remote ProVoice reported it has ended%s — "
                          "stopping the vehicle and closing the session."
                          % (" (%s)" % pending_end_reason if pending_end_reason else ""))
                    end_overlay = SessionEndedOverlay(args.width, args.height,
                                                      pending_end_reason)
                    _set_world_frozen(world, True)
                else:
                    # A prompt is on screen for the window: let the driver
                    # finish answering (or skipping) BOTH of its prompts
                    # normally, exactly as if ProVoice had not signalled yet.
                    # The scene is already frozen for the popup, so nothing
                    # drives on in the meantime; the exit screen appears the
                    # instant the popup itself closes, below.
                    print("[INFO] Remote ProVoice reported it has ended%s — a "
                          "LoA prompt is on screen; finishing it before closing "
                          "the session."
                          % (" (%s)" % pending_end_reason if pending_end_reason else ""))
            elif end_overlay is None and provoice_end_pending and not loa_popup.active:
                print("[INFO] Remote ProVoice ended while a LoA prompt was open; "
                      "the prompt is answered — closing the session now%s."
                      % (" (%s)" % pending_end_reason if pending_end_reason else ""))
                end_overlay = SessionEndedOverlay(args.width, args.height,
                                                  pending_end_reason)
                _set_world_frozen(world, True)

            if end_overlay is not None:
                if isinstance(world.player, carla.Vehicle):
                    stop_control = carla.VehicleControl()
                    stop_control.throttle = 0.0
                    stop_control.brake = 1.0
                    stop_control.hand_brake = True
                    world.player.apply_control(stop_control)
                for event in events:
                    if end_overlay.handle_event(event) == 'quit':
                        return
                # Same reason as the popup branch: no world.tick here, and the
                # session is over -- the bed settles to idle rather than holding
                # the speed the car had when ProVoice stopped. Not ducked (even
                # if a popup closed straight into this overlay) -- an ended
                # session should sound like the car stopping, not going mute.
                ambience.set_ducked(False)
                ambience.update(0.0)
                world.render(display)
                end_overlay.render(display)
                pygame.display.flip()
                continue

            if not started:
                if isinstance(world.player, carla.Vehicle):
                    pause_control = carla.VehicleControl()
                    pause_control.throttle = 0.0
                    pause_control.brake = 1.0
                    pause_control.hand_brake = True
                    world.player.apply_control(pause_control)
                for event in events:
                    action = start_overlay.handle_event(event)
                    if action == 'quit':
                        return
                    if action == 'start':
                        started = True
                        world.start_weather_schedule(now_ms)
                        if wait_for_provoice:
                            provoice_watcher.start(now_ms)
                            print("[INFO] Holding the LoA windows until ProVoice logs "
                                  "its first frame for this session...")
                            world.hud.notification(
                                'Waiting for ProVoice to start logging...', seconds=6.0)
                        else:
                            loa_popup.start()
                            popups_armed = True
                        break
                world.render(display)
                start_overlay.render(display)
                pygame.display.flip()
                continue

            if not popups_armed:
                if provoice_ready_pending:
                    # Extra hold past the collection_started signal itself —
                    # see PROVOICE_READY_POPUP_DELAY_S.
                    if now_ms >= provoice_ready_ms + PROVOICE_READY_POPUP_DELAY_S * 1000.0:
                        loa_popup.start()
                        popups_armed = True
                        provoice_ready_pending = False
                        print("[INFO] %.0f s past ProVoice's collection_started signal; "
                              "first window starts now and its popup opens in %.0f s, "
                              "so the whole window has driver-state data."
                              % (PROVOICE_READY_POPUP_DELAY_S, loa_popup.interval_ms / 1000.0))
                        world.hud.notification('Recording started', seconds=4.0)
                else:
                    # Driving is already live — only the label windows wait, so the
                    # hold doubles as the driver's adaptation time.
                    state = provoice_watcher.poll(now_ms)
                    if state == 'ready' and status_path:
                        # Remote run and the signal itself (not the timeout
                        # fallback): hold PROVOICE_READY_POPUP_DELAY_S more
                        # before arming, on top of the usual 20 s window wait.
                        provoice_ready_pending = True
                        provoice_ready_ms = now_ms
                        waited = provoice_watcher.waited_s(now_ms)
                        print("[INFO] ProVoice is logging after %.1f s; holding %.0f s "
                              "more before the first window starts."
                              % (waited, PROVOICE_READY_POPUP_DELAY_S))
                        world.hud.notification('Recording started - stabilizing...',
                                               seconds=4.0)
                    elif state in ('ready', 'timeout'):
                        loa_popup.start()
                        popups_armed = True
                        waited = provoice_watcher.waited_s(now_ms)
                        if state == 'ready':
                            print("[INFO] ProVoice is logging after %.1f s; first window "
                                  "starts now and its popup opens in %.0f s, so the whole "
                                  "window has driver-state data."
                                  % (waited, loa_popup.interval_ms / 1000.0))
                            world.hud.notification('Recording started', seconds=4.0)
                        else:
                            print("[WARN] No ProVoice frame for this session after %.0f s "
                                  "— starting the LoA windows anyway. Early windows may "
                                  "have no driver-state data, and scripts/"
                                  "build_loa_dataset.py will drop those labels."
                                  % waited)
                            world.hud.notification(
                                'ProVoice not logging - labels may be unusable', seconds=8.0)

            if not provoice_end_pending and loa_popup.should_open(now_ms):
                loa_popup.open(now_ms)
                # Freeze the whole scene (ego + NPC traffic) so nothing moves
                # while the driver deliberates over the LoA for the last 20 s.
                _set_world_frozen(world, True)

            if loa_popup.active:
                for event in events:
                    action, selected_loa = loa_popup.handle_event(event)
                    if action == 'quit':
                        return
                    if action == 'skip':
                        # NO INPUT: the driver could not answer this window
                        # honestly. Nothing is appended -- that is the whole
                        # point -- so the window simply leaves no label, exactly
                        # as if the prompt had never opened. Counted and printed
                        # because a participant who starts skipping repeatedly
                        # is telling the experimenter something.
                        skipped_prompts += 1
                        print("[INFO] NO INPUT for window %d prompt %d ('%s') "
                              "- no label written (%d skipped so far)"
                              % (loa_popup.window_idx, loa_popup.prompt_in_window,
                                 loa_popup.function_name, skipped_prompts))
                        world.hud.notification('No input recorded for that question',
                                               seconds=3.0)
                        if loa_popup.prompt_in_window == 1 and loa_popup.prompts_per_window > 1:
                            # NO INPUT on the FIRST prompt means the driver could
                            # not honestly answer for this 20 s at all, which makes
                            # the whole window invalid -- the second prompt would
                            # just be asking about the same unusable stretch, so
                            # skip it too instead of opening it.
                            loa_popup.close(now_ms)
                            _set_world_frozen(world, False)
                        else:
                            _advance_or_close(loa_popup, world, now_ms)
                        break
                    if action == 'select':
                        if args.test_popup:
                            # Practice answers teach the control, they are not
                            # data — nothing is computed or appended for them.
                            print("[INFO] practice selection %s for '%s' (not logged)"
                                  % (sorted(selected_loa), loa_popup.function_name))
                            _advance_or_close(loa_popup, world, now_ms)
                            break
                        player_velocity = world.player.get_velocity() if world and world.player else carla.Vector3D()
                        speed_kmh = 3.6 * math.sqrt(
                            player_velocity.x ** 2 + player_velocity.y ** 2 + player_velocity.z ** 2)
                        system_snapshot = _load_latest_system_decision_snapshot(label_context['session_id'])
                        append_user_loa_selection({
                            **label_context,
                            # Overrides label_context: with --random-function the
                            # function is drawn per prompt, so the row has to
                            # record the one that was actually on screen.
                            'functionname': loa_popup.function_name,
                            'window_idx': loa_popup.window_idx,
                            'prompt_in_window': loa_popup.prompt_in_window,
                            'window_start_ms': loa_popup.window_start_ms,
                            'window_end_ms': loa_popup.window_end_ms,
                            'window_start_timestamp': loa_popup.window_start_timestamp,
                            'window_end_timestamp': loa_popup.window_end_timestamp,
                            'selection_timestamp': datetime.datetime.now().isoformat(timespec='milliseconds'),
                            'selection_frame': world.hud.frame if world else '',
                            'selection_sim_time': round(world.hud.simulation_time, 3) if world else '',
                            'selection_speed_kmh': round(speed_kmh, 2),
                            # Several LoAs may be marked acceptable; joined with
                            # ';' so a single value still reads as a bare int
                            # for anything expecting the old format.
                            'user_selected_loa': ';'.join(str(v) for v in selected_loa),
                            'system_action': system_snapshot.get('action', ''),
                            'system_level': system_snapshot.get('level', ''),
                            'system_loa': system_snapshot.get('LoA', system_snapshot.get('loa', '')),
                            'system_message': system_snapshot.get('message', ''),
                            'system_probs': system_snapshot.get('probs', ''),
                            'system_profile': system_snapshot.get('profile', ''),
                            'system_fallback': system_snapshot.get('fallback', ''),
                            'system_fallback_reason': system_snapshot.get('fallback_reason', ''),
                            'system_fcd': system_snapshot.get('fcd', system_snapshot.get('FCD', '')),
                        })
                        _advance_or_close(loa_popup, world, now_ms)
                        # Stop reading this batch: anything still queued was typed
                        # at the prompt just answered and must not leak into the
                        # next one.
                        break
                # Explicitly idle: this branch never reaches world.tick, so
                # hud.speed_kmh still holds the speed the car was doing when the
                # scene froze, and the bed would sit at motorway level over a
                # motionless picture for the whole deliberation. Ducked to
                # silence on top of that -- idle alone still leaves an idling
                # engine under the prompt, which the deliberation should not
                # have to compete with.
                ambience.set_ducked(True)
                ambience.update(0.0)
                world.render(display)
                loa_popup.render(display)
                pygame.display.flip()
                continue

            # No tick or wait here: the wait moved to the TOP of the loop so that
            # input is drained fresh off a tick and the control applied just below
            # lands on the very next one. See the comment there.
            #
            # suppress_wheel_quit=call_preview.active: while the call panel is
            # up, button 7 sits one paddle away from the ones the driver is
            # actually reaching for (CALL_WHEEL_BUTTON_AFFIRM/NEGATIVE), and a
            # mis-press there must not end the whole study out from under a
            # participant answering a phone call.
            if controller.parse_events(
                    client, world, clock, args.sync, events,
                    suppress_wheel_quit=(call_preview is not None
                                         and call_preview.active),
                    now_ms=now_ms):
                return
            world.tick(clock)
            # After world.tick, which is what refreshes hud.speed_kmh (it does so
            # regardless of --speed: the readout is drawn from it, not the other
            # way round). Throttle comes off the controller's OWN control object
            # -- a local field it just wrote, not an RPC to the server, so the
            # engine note follows the pedal for free on a loop this project
            # already tunes for frame rate.
            if call_preview is not None:
                for event in events:
                    # The 0-4 keys stage a call by hand. Preview only: in a
                    # study block the LoA comes from the model, and a
                    # participant who found this key could manufacture one.
                    if (args.call_preview and not study_on
                            and event.type == pygame.KEYDOWN
                            and pygame.K_0 <= event.key <= pygame.K_4
                            and not call_preview.active):
                        # SHIFT gives the SPAM rendering of the same rung, so
                        # both wordings can be checked against each other on the
                        # rig without restarting.
                        #
                        # Keyed on event.key, not event.unicode: with SHIFT held
                        # the character is layout-dependent ('"' on a UK board,
                        # '@' on a US one) and would not match a digit at all.
                        call_preview.arm(
                            now_ms, event.key - pygame.K_0, 0,
                            spam=bool(event.mod & pygame.KMOD_SHIFT))
                    else:
                        call_preview.handle_event(event, now_ms=now_ms)

                if (study is not None and study.started_ms is None
                        and popups_armed):
                    # Gated on `popups_armed`, which is really "the session
                    # clock has started" -- it is set by the ProVoice-ready path
                    # above even when there are no pop-ups to arm. So the 10
                    # minutes begin when ProVoice is logging, exactly as the
                    # label windows do in a collection run, rather than at the
                    # first driving tick. Without this the block would spend its
                    # opening minutes serving calls from a model that has not
                    # started answering, and log them as skipped.
                    study.start(now_ms)
                    print('[study] ProVoice is up; block clock starts now.')

                if study is not None and not study_ended:
                    due = study.update(now_ms, world.hud.speed_kmh,
                                       loa_popup.active, call_preview.active,
                                       _read_status_file)
                    if due is not None:
                        loa, call_idx, is_spam = due
                        call_preview.arm(now_ms, loa, call_idx, spam=is_spam)
                        print('[study] call %d/%d armed at LoA %d, %s (%.0f s in).'
                              % (call_idx, study.n_calls, loa,
                                 'SPAM' if is_spam else 'genuine',
                                 study.elapsed_s(now_ms)))

                finished = call_preview.update(now_ms)
                if finished:
                    if study is not None:
                        study.note_outcome(finished, now_ms)
                        print('[study] call %s (%s, proposed %s) -> %s (%s)'
                              % (finished.get('window_idx'),
                                 finished.get('call_kind'),
                                 finished.get('proposed_action'),
                                 finished.get('driver_response'),
                                 finished.get('outcome')))
                    else:
                        print('[call-preview] %s' % finished)

                if (study is not None and not study_ended
                        and study.finished(now_ms)):
                    study_ended = True
                    print('[study] block complete: %s' % study.summary())
                    # ProVoice has no clock of its own for this: DRIVE owns the
                    # call schedule and the block length, so the other machine
                    # has to be told. Two independent timers would drift and the
                    # first to fire would truncate the other's data.
                    _post_drive_ended(args.provoice_status_url, session_id,
                                      getattr(args, 'participantid', ''),
                                      study.summary())
                    end_overlay = SessionEndedOverlay(
                        args.width, args.height,
                        'block complete (%s)' % study.summary(),
                        # The questionnaire follows the button press, so this
                        # screen is a hand-off and not a goodbye.
                        continues=True)
                    _set_world_frozen(world, True)

            # NOT ducked for a call. set_ducked(True) fades the bed to SILENCE
            # (_duck_target = 0.0), which is right for the LoA popup because
            # that freezes the scene -- but a call happens while the car is
            # moving, so cutting the engine would be both unrealistic and a cue
            # that something is about to happen. The ring and the assistant play
            # OVER the bed; if they are hard to hear, raise RING_GAIN in
            # call_event.py or lower --ambient-gain, do not silence the car.
            ambience.set_ducked(False)
            ambience.update(world.hud.speed_kmh,
                            throttle=getattr(
                                getattr(controller, '_control', None),
                                'throttle', 0.0))
            world.render(display)
            if call_preview is not None:
                # AFTER world.render, which blits the camera frame and then the
                # HUD -- the panel belongs on top of both.
                call_preview.render(display)
            pygame.display.flip()

    finally:

        # Deliberately NOT restoring world settings. Drive no longer changes
        # them -- under --sync the clock owner (fixed_npc_traffic.py) sets
        # synchronous_mode and the fixed step, and restores both in its own
        # finally. Two processes each restoring a snapshot they took at different
        # moments is how a server ends up stuck in synchronous mode with nobody
        # ticking it, which blocks every client that connects afterwards.
        if original_settings:
            sim_world.apply_settings(original_settings)

        # Release the clock unconditionally. Quitting with a popup open would
        # otherwise leave the request file behind, and the clock owner outlives
        # Drive -- it would sit holding the simulation until its staleness
        # timeout. Cheap insurance against ending a session on a frozen rig.
        if getattr(args, 'clock_pause_file', ''):
            _request_clock_pause(args.clock_pause_file, False)

        if (world and world.recording_enabled):
            client.stop_recorder()

        if world is not None:
            world.destroy()

        if study is not None and not study_ended:
            print('[study] block ended early: %s' % study.summary())
        if call_preview is not None:
            call_preview.stop()
        if ambience is not None:
            ambience.stop()

        pygame.quit()


# ==============================================================================
# -- main() --------------------------------------------------------------------
# ==============================================================================


def main():
    argparser = argparse.ArgumentParser(description='CARLA Manual Control Client')
    argparser.add_argument(
        '-v', '--verbose', action='store_true', dest='debug',
        help='print debug information')
    argparser.add_argument(
        '--host', metavar='H', default='127.0.0.1',
        help='IP of the host server (default: 127.0.0.1)')
    argparser.add_argument(
        '-p', '--port', metavar='P', default=2000, type=int,
        help='TCP port to listen to (default: 2000)')
    argparser.add_argument(
        '-a', '--autopilot', action='store_true',
        help='enable autopilot')
    argparser.add_argument(
        '--res', metavar='WIDTHxHEIGHT', default='1280x720',
        help='window resolution (default: 1280x720)')
    argparser.add_argument(
        '--fullscreen', action='store_true',
        help='run fullscreen at the desktop resolution (overrides --res)')
    argparser.add_argument(
        '--filter', metavar='PATTERN', default='vehicle.*',
        help='actor filter (default: "vehicle.*")')
    argparser.add_argument(
        '--generation', metavar='G', default='All',
        help='restrict to certain actor generation (values: "2","3","All" - default: "All")')
    argparser.add_argument(
        '--rolename', metavar='NAME', default='hero',
        help='actor role name (default: "hero")')
    argparser.add_argument(
        '--gamma', default=1.0, type=float,
        help='Gamma correction of the camera (default: 1.0)')
    argparser.add_argument(
        '--brake-assist-decel', default=BRAKE_ASSIST_TARGET_DECEL, type=float,
        help='peak deceleration in m/s^2 at full brake (default: %.1f, about '
             '0.9g). CARLA 0.10 vehicles manage only ~0.6g and the physics API '
             'cannot raise it, so the shortfall is supplied by a force opposing '
             'travel. 0 disables the assist and leaves stock CARLA braking. '
             'KEEP THIS FIXED ACROSS ALL PARTICIPANTS AND BOTH STUDY ARMS.'
             % BRAKE_ASSIST_TARGET_DECEL)
    argparser.add_argument(
        '--participantid', default='',
        help='participant id for label logging')
    argparser.add_argument(
        '--environment', default='',
        help='environment for label logging')
    argparser.add_argument(
        '--secondary-task', dest='secondary_task', default='',
        help='secondary task for label logging')
    argparser.add_argument(
        '--functionname', default='Adjust seat positioning',
        help='function/task name for label logging')
    argparser.add_argument(
        '--emotion', default='',
        help='emotion label for label logging')
    argparser.add_argument(
        '--modeltype', default='',
        help='model type for label logging')
    argparser.add_argument(
        '--state-model', dest='state_model', default='',
        help='state model name for label logging')
    argparser.add_argument(
        '--w-fcd', dest='w_fcd', default=0.7, type=float,
        help='FCD weight for label logging')
    argparser.add_argument(
        '--session-id', dest='session_id', default='',
        help='shared session id for aligning logs')
    argparser.add_argument(
        '--sync', action='store_true',
        help='The world is in synchronous mode and ANOTHER client owns the '
             'clock: pace this loop to it with wait_for_tick instead of '
             'free-running. Drive does NOT set synchronous mode and does NOT '
             'tick -- src/drive/fixed_npc_traffic.py --sync does both, because '
             'CARLA only lets the process that owns the NPCs\' traffic manager '
             'drive it. Launch both together (start_experiment.py --sync).')
    argparser.add_argument(
        '--clock-pause-file', dest='clock_pause_file', default='',
        help='Path used to ask the --sync clock owner to hold the simulation '
             'while a LoA popup is open, giving a true freeze-and-resume instead '
             'of stopping every vehicle dead and making it pull away from rest. '
             'Must match the clock owner\'s --pause-file. Ignored without '
             '--sync; start_experiment.py --sync sets both.')
    argparser.add_argument(
        '--tm-port', dest='tm_port', default=8000, type=int,
        help='Traffic manager port (default: 8000). Under --sync this MUST '
             'match the clock owner\'s --tm-port (9000 in this project), or '
             'Drive holds a second traffic manager that nothing ticks.')
    argparser.add_argument(
        '--control',
        choices=['test', 'full'],
        default='test',
        help='Control mode: "test" for basic controls, "full" for all controls (default: test)')
    argparser.add_argument(
        '--no-wheel',
        action='store_true',
        help='Ignore any attached steering wheel and force keyboard control')
    argparser.add_argument(
        '--fixed', dest='fixed', action='store_true',
        help='Always spawn the ego at the same map spawn point (index %d) instead of '
             'a random one. Intended for calibration runs, which have to start from '
             'an identical position. If that point is occupied the next free index is '
             'used, deterministically.' % FIXED_SPAWN_POINT_INDEX)
    argparser.add_argument(
        '--test-popup', dest='test_popup', action='store_true',
        help='Practice mode for teaching the LoA control: the first window opens as '
             'soon as the session starts and they repeat every %d s. Each window '
             'holds the same %d consecutive prompts about %d different functions as a '
             'real one, so the pair is what gets practised. Selections are NOT '
             'written to data/user_loa_labels.csv.'
             % (TEST_POPUP_INTERVAL_S, PROMPTS_PER_WINDOW, PROMPTS_PER_WINDOW))
    argparser.add_argument(
        '--no-popup', dest='no_popup', action='store_true',
        help='Suppress the LoA selection popups for the whole session. By default '
             'the scene freezes every 20 s and a popup asks for the acceptable LoAs; '
             'with this flag there are no popups, no freezes, and nothing is appended '
             'to data/user_loa_labels.csv')
    argparser.add_argument(
        '--popup-wait-timeout', dest='popup_wait_timeout', type=float,
        default=POPUP_WAIT_TIMEOUT_S,
        help='Seconds to wait for ProVoice to log its first frame for this session '
             'before opening the first LoA window anyway (default: %(default)s). The '
             'wait is what keeps early windows from being labelled with no '
             'driver-state data behind them: the first window starts when ProVoice '
             'does, so the popup one interval later covers a fully logged 20 s. '
             'Driving is not held up, only the windows. 0 disables the wait; ignored '
             'with --test-popup, which runs without ProVoice.')
    argparser.add_argument(
        '--render-scale', dest='render_scale', default=1.0, type=float,
        help='Render the drive camera at this fraction of the window size and '
             'upscale it to fill the screen (default: 1.0 = unchanged). This is '
             'the main lever on the CARLA server frame rate, which under --sync '
             'is the ceiling on the whole simulation: if the server cannot render '
             'as fast as the clock ticks, simulated time falls behind and '
             'everything runs in slow motion. 0.7 renders about half the pixels, '
             '0.5 about a quarter. The view gets softer, not smaller. Check the '
             'effect in the [SYNC] ceiling line from fixed_npc_traffic.py.')
    argparser.add_argument(
        '--call-preview', dest='call_preview', action='store_true',
        help='Trial mode for the live study call pop-up: drive normally and '
             'press 0-4 to stage an incoming call at that Level of Autonomy, '
             'A / B to answer it. Shows the panel over the REAL camera view at '
             'the real resolution, alongside the real HUD, so placement and '
             'legibility can be judged before any of it is wired into the study '
             'loop. Nothing is logged and no decision is read -- the LoA comes '
             'from the key press, not from ProVoice. NOT for participant runs.')
    argparser.add_argument(
        '--study', action='store_true',
        help='Run ONE block of the live follow-up study: %d min of free '
             'driving with %d incoming calls at ~%d s spacing, each handled at '
             'the Level of Autonomy the model predicted. One invocation is one '
             'block, i.e. one K condition -- three blocks means three runs, '
             'with the served checkpoint swapped on the ProVoice side between '
             'them. Writes data/call_events.csv.'
             % (FULL_TRIAL['duration_s'] / 60, FULL_TRIAL['n_calls'],
                FULL_TRIAL['interval_s']))
    argparser.add_argument(
        '--short-trial', dest='short_trial', action='store_true',
        help='Same as --study but %d calls in %d min (~%d s apart), to walk the '
             'whole path -- arming, motion gate, rendering, input, logging -- '
             'in two minutes instead of ten. Implies --study. NOT a study '
             'configuration; every row is still stamped with its real spacing.'
             % (SHORT_TRIAL['n_calls'], SHORT_TRIAL['duration_s'] / 60,
                SHORT_TRIAL['interval_s']))
    argparser.add_argument(
        '--random-loa', dest='random_loa', action='store_true',
        help='Draw the LoA locally instead of reading it from the ProVoice '
             'bridge, so the drive half can be tested with nothing else '
             'running. Deals a SHUFFLED PERMUTATION of 0-4, so a five-call '
             'block exercises every rendering exactly once -- five independent '
             'draws would leave a rung untested about half the time. NOT a '
             'study configuration: every row it writes is stamped '
             "loa_source='random' so it cannot be mistaken for served data.")
    argparser.add_argument(
        '--test-calls', dest='test_calls', action='store_true',
        help='Like --random-loa but walks the five levels IN ORDER (0, 1, 2, 3, '
             '4) instead of shuffling them, so a reviewer knows which rung is '
             'coming and can watch for one specific thing. Pairs with '
             '--short-trial, which is exactly five calls. Implies --study; NOT '
             'a study configuration.')
    argparser.add_argument(
        '--spam-call', dest='spam_call', type=int, default=None,
        help='Pin WHICH call in the block is the suspected spam call, instead '
             'of drawing it. One call per block is spam and it is never the '
             'first (see study_session.py); by default the position is redrawn '
             'each block so a driver cannot learn it across their three. Pass '
             '0 to run a block with no spam call at all. The value is written '
             'to every row, so a pinned block is identifiable afterwards.')
    argparser.add_argument(
        '--test-spam', dest='test_spam', action='store_true',
        help='Make EVERY call a spam call. For walking the five spam '
             'renderings in one pass -- pair it with --test-calls, which deals '
             'the rungs 0-4 in order. Implies --study; NOT a study '
             'configuration, and every row still records call_kind.')
    argparser.add_argument(
        '--loa-max-age', dest='loa_max_age', type=float, default=3.0,
        help='Refuse a served LoA older than this many seconds and log the '
             'call as skipped (default: %(default)s). STALENESS IS THE SILENT '
             'FAILURE HERE: if ProVoice stalls, the camera drops or the bridge '
             'stops writing, nothing errors -- the status file simply stops '
             'changing and every call in the block gets served the same dead '
             'value, which is indistinguishable from a model that confidently '
             'predicts one level. A healthy decision is a few hundred ms old, '
             'so 3 s is loose enough for any reasonable push rate and tight '
             'enough to catch a stall. 0 disables the check.')
    argparser.add_argument(
        '--study-seed', dest='study_seed', type=int, default=None,
        help='Seed for the call jitter and, under --random-loa, the LoA deal. '
             'Fixing it makes a trial run reproducible; leave it unset for '
             'participants so the jitter differs between blocks.')
    argparser.add_argument(
        '--k-condition', dest='k_condition', default='',
        help='Which K condition this block serves (e.g. k000, k010, k030). '
             'Recorded in every call_events.csv row -- it is the independent '
             'variable, and nothing else in the drive process knows it.')
    argparser.add_argument(
        '--block-idx', dest='block_idx', default='',
        help='Position of this block in the participant sequence (1, 2 or 3). '
             'Recorded per row; needed to model order effects.')
    argparser.add_argument(
        '--provoice-status-url', dest='provoice_status_url', default='',
        help='Address of the reverse status bridge, used to tell ProVoice on '
             'the other machine that a --study block has finished so it stops '
             'too. Set by start_experiment.py under --remote; without it a '
             'remote ProVoice keeps recording after the drive ends and has to '
             'be stopped by hand.')
    argparser.add_argument(
        '--call-chrome', dest='call_chrome', choices=('none', 'panel'),
        default='none',
        help="How the call pop-up is drawn under --call-preview. 'none' "
             '(default) puts the text straight on the scene with a dark halo, '
             'occluding less of the road and matching the speed readout, which '
             "has no backing plate either. 'panel' draws a translucent blue "
             'plate and border behind it, which is more legible over bright '
             'tarmac but covers more of the view. Hold this FIXED across all '
             'participants and conditions.')
    argparser.add_argument(
        '--speed', action='store_true',
        help='Show the vehicle speed in km/h low on screen and left of centre '
             '(~27%% of window width, bottom edge at 80%% of height). Unlike '
             'the F1 debug panel this is a single large readout suitable to have '
             'in front of a participant. Off by default because it changes the '
             'driving task -- it gives the driver a precise instrument to regulate '
             'against -- so if it is used at all it should be used for EVERY '
             'participant and both study arms.')
    argparser.add_argument(
        '--ambient-gain', dest='ambient_gain', type=float,
        default=DEFAULT_AMBIENT_GAIN,
        help='Gain of the synthesised road/cabin noise, 0-1 (default: '
             '%(default)s). ON by default: CARLA has no audio of its own in any '
             'version, so without this the rig is silent. The level tracks '
             'vehicle speed and idles at a standstill. Pass 0 for silence. '
             'Sound is an arousal manipulation whether or not it is meant as '
             'one, and hr_delta / rr_delta are model inputs, so this must be '
             'IDENTICAL for every participant and both study arms -- gain and '
             'physical volume alike. The number is not a level: set the '
             'amplifier once, measure dB(A) at the driver\'s head, and report '
             'that. Logged per label row as ambient_gain.')
    argparser.add_argument(
        '--ambient-seed', dest='ambient_seed', type=int, default=0,
        help='Seed for the ambience noise loop (default: %(default)s). The bed '
             'is synthesised, so this seed plus the gain reproduce exactly what '
             'a participant heard; it is logged per label row. There is no '
             'reason to vary it across participants, and good reason not to.')
    argparser.add_argument(
        '--ambient-dir', dest='ambient_dir', default=None,
        help='Directory of recorded cabin clips (default: <repo>/assets/'
             'ambience, or $PROVOICE_AMBIENCE_DIR). Files named '
             'interior_<kmh>.wav are crossfaded by speed; a single interior.wav '
             'also works. Clips are loop-repaired on load, so they can be '
             'dropped in untouched. With no usable clips the synthesiser is '
             'used instead and ambient_source is logged as "synth".')
    argparser.add_argument(
        '--popup-immediate', dest='popup_immediate', action='store_true',
        help='Open the FIRST LoA window straight away instead of one interval in. '
             'For rehearsing the popup flow on the real rig: unlike --test-popup '
             'this keeps the genuine 20 s interval, the NPC traffic and everything '
             'else about a normal run, so only the first window moves. Pair with '
             '--popup-wait-timeout 0 to also skip the wait for ProVoice. Labels '
             'from a run using this are NOT participant data -- the first window '
             'covers driving that nothing was recording.')
    argparser.add_argument(
        '--provoice-status-file', dest='provoice_status_file', default=None,
        help='File published by scripts/provoice_status_server.py, for runs where '
             'ProVoice is on ANOTHER machine (default: %s under --remote). It '
             'carries the two facts this process cannot observe from here: when '
             'the remote ProVoice started recording (the LoA windows wait for it, '
             'exactly as they wait for the first line of raw_data.jsonl locally) '
             'and when it ended (the vehicle is stopped and the end-of-session '
             'screen is shown). Unset = single-machine run, both read from the '
             'log and from the launcher.' % PROVOICE_STATUS_FILE)
    argparser.add_argument(
        '--random-function', dest='random_function', action='store_true',
        help='Draw the function each popup asks about at random from the %d study '
             'functions (%s) instead of using --functionname for the whole session. '
             'Each 20 s window then holds %d prompts about %d DIFFERENT functions '
             '(drawn without replacement), and each row records the function it was '
             'actually shown.'
             % (len(RANDOM_FUNCTION_POOL), ', '.join(RANDOM_FUNCTION_POOL),
                PROMPTS_PER_WINDOW, PROMPTS_PER_WINDOW))
    popup_input_group = argparser.add_mutually_exclusive_group()
    popup_input_group.add_argument(
        '--wheel-input', dest='popup_input', action='store_const',
        const=POPUP_INPUT_WHEEL,
        help='Answer the LoA popups from the steering wheel instead of the keyboard: '
             'the paddles move the cursor, the front button ticks the level under it, '
             'and the CONFIRM row submits. The row below it, NO INPUT, dismisses the '
             'prompt without writing a label. The number keys (and N for no input) '
             'stay live as a fallback in case the rim buttons are unmapped.')
    popup_input_group.add_argument(
        '--keyboard-input', dest='popup_input', action='store_const',
        const=POPUP_INPUT_KEYBOARD,
        help='Answer the LoA popups from the keyboard: number keys 0-4 tick a level, '
             'pressing the same number again unticks it, N ticks INVALID FRAME '
             'instead, and ENTER commits whatever is ticked (an invalid frame writes '
             'no label). The popup ignores the wheel in this mode (it still steers). '
             'THE DEFAULT - the flag only states it explicitly.')
    argparser.set_defaults(popup_input=None)
    condition_group = argparser.add_mutually_exclusive_group()
    condition_group.add_argument(
        '--condition-sun-rain', dest='condition_sun_rain', action='store_true',
        help='Scripted weather condition: precipitation starts at %.0f%%%% and '
             'ramps linearly down to 0%%%% over %.0f minutes, from the moment '
             'driving starts (the start-screen "start" action, not process '
             'launch). Mutually exclusive with --condition-rain-sun.'
             % (CONDITION_PEAK_PRECIPITATION, CONDITION_RAMP_MINUTES))
    condition_group.add_argument(
        '--condition-rain-sun', dest='condition_rain_sun', action='store_true',
        help='Scripted weather condition: sunny (0%%%% precipitation) for the '
             'first %.0f minutes of driving, then ramps linearly up to %.0f%%%% '
             'over the following %.0f minutes. Mutually exclusive with '
             '--condition-sun-rain.'
             % (CONDITION_SUN_HOLD_MINUTES, CONDITION_PEAK_PRECIPITATION,
                CONDITION_RAMP_MINUTES))
    args = argparser.parse_args()

    # --- study mode normalisation ------------------------------------------
    #
    # A study block measures the CALLS. The 20 s Level-of-Proactivity pop-up is
    # the data-collection instrument and has no place here: it would freeze the
    # scene every 20 seconds, interrupt calls mid-interaction, and ask the
    # driver to label windows nobody is going to use. Forced off rather than
    # left to the experimenter to remember, because forgetting it does not fail
    # -- it quietly produces a block that measures something else.
    if args.short_trial or args.test_calls or args.test_spam:
        args.study = True
    if args.test_spam and args.spam_call is not None:
        argparser.error('--test-spam makes every call spam, so --spam-call has '
                        'nothing left to pin. Pick one.')
    if args.spam_call is not None and args.spam_call == 1:
        argparser.error('--spam-call 1 is not allowed: the first call is where '
                        'the driver learns what the interface does, and meeting '
                        'the inverted proposal there would make the exception '
                        'the reference point for the other four. Use 0 to '
                        'disable, or 2 and up.')
    if args.test_calls and args.random_loa:
        argparser.error('--test-calls and --random-loa both replace the served '
                        'LoA and contradict each other: one walks 0-4 in order, '
                        'the other shuffles. Pick one.')
    if args.study:
        if args.test_popup:
            argparser.error('--test-popup and --study contradict each other: '
                            'one is nothing but labelling pop-ups, the other '
                            'suppresses them to measure calls instead.')
        if args.random_function:
            argparser.error('--random-function only affects the labelling '
                            'pop-up, which --study turns off. Drop it.')
        if not args.no_popup:
            args.no_popup = True
            print('[study] labelling pop-ups disabled for this block '
                  '(--no-popup implied): the block measures calls, and a '
                  'pop-up every 20 s would freeze the scene on top of them.')
    if (args.random_loa or args.test_calls) and not args.study:
        argparser.error('--random-loa / --test-calls only have an effect on a '
                        'study block. Add --study (or --short-trial), or drop '
                        'them.')
    if args.spam_call is not None and not args.study:
        argparser.error('--spam-call only has an effect on a study block. Add '
                        '--study (or --short-trial), or drop it.')
    # A real participant block, as opposed to the dev/test configurations that
    # also set args.study (--short-trial, --test-calls, --test-spam,
    # --random-loa) and explicitly are NOT study data. Those are exempt so a
    # quick two-minute walkthrough doesn't need meaningless K/order values;
    # a real block does, because nothing else in the drive process knows
    # them, and a forgotten flag would silently write blank k_condition /
    # block_idx into every call_events.csv row for the block.
    if args.study and not (args.short_trial or args.test_calls
                            or args.test_spam or args.random_loa):
        if not args.k_condition:
            argparser.error('--k-condition is required for a real --study '
                            'block: it is the independent variable and '
                            'nothing else in the drive process knows it. '
                            'Pass e.g. --k-condition k010, or use '
                            '--short-trial/--test-calls/--test-spam/'
                            '--random-loa for a dev run that does not need it.')
        if not args.block_idx:
            argparser.error('--block-idx is required for a real --study '
                            'block: it is needed to model order effects and '
                            'nothing else in the drive process knows it. '
                            'Pass e.g. --block-idx 1, or use --short-trial/'
                            '--test-calls/--test-spam/--random-loa for a dev '
                            'run that does not need it.')

    if args.test_popup and args.no_popup:
        argparser.error('--test-popup and --no-popup contradict each other: one is '
                        'nothing but popups, the other suppresses them.')
    if args.test_popup:
        # Before anything can run, so no ordering inside game_loop can leave a
        # window where a practice answer reaches the file.
        disable_user_loa_logging('--test-popup practice mode')
        print('[INFO] --test-popup: user LoA label logging is DISABLED for this '
              'process; data/user_loa_labels.csv is not written or created.')
    if args.popup_input == POPUP_INPUT_WHEEL and args.no_wheel:
        argparser.error('--wheel-input needs the steering wheel that --no-wheel '
                        'disables. Drop one of the two.')
    if args.popup_input is not None and args.no_popup:
        # Not fatal, but it means the run was configured for an interface it will
        # never show — worth saying rather than silently ignoring the flag.
        print('[WARN] --%s-input has no effect with --no-popup: there are no popups '
              'to answer.' % args.popup_input)
    if args.random_function and args.no_popup:
        print('[WARN] --random-function has no effect with --no-popup: the function '
              'is only ever asked about in a popup.')

    args.width, args.height = [int(x) for x in args.res.split('x')]

    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(format='%(levelname)s: %(message)s', level=log_level)

    logging.info('listening to server %s:%s', args.host, args.port)

    print(__doc__)

    try:

        game_loop(args)

    except KeyboardInterrupt:
        print('\nCancelled by user. Bye!')


if __name__ == '__main__':

    main()
