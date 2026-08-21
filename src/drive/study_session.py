#!/usr/bin/env python

"""Driving block for the live follow-up study: timing, LoA source, logging.

One process invocation == ONE block == one K condition. Three blocks means three
runs, with the served checkpoint swapped on the ProVoice side between them. Full
design record: ``docs/live_study_setup.md`` (section 4 the session, 4.2 the
arming rules, 9 the log schema).

WHAT THIS OWNS AND WHAT IT DOES NOT
-----------------------------------
This module decides WHEN a call fires and WHICH LoA it carries, and it writes
the outcome. ``call_event.CallEvent`` owns the interaction itself and knows
nothing about schedules, blocks or conditions. Keeping the split means the five
renderings can be exercised standalone while the schedule stays testable on its
own clock.

THE DRIVE PROCESS DOES NOT LOAD A MODEL
---------------------------------------
It never has and it must not start: the model lives on the ProVoice machine and
its prediction arrives over the bridge (``docs/live_study_setup.md`` section 7).
Two sources are supported:

``bridge``  read ``latest_loa`` from the ProVoice status file, session-scoped.
            This is the study configuration.
``random``  draw locally, for testing the drive half without ProVoice running
            at all. NOT a study configuration -- every row it writes is stamped
            ``loa_source='random'`` so it can never be mistaken for served data.

The random source deals a SHUFFLED PERMUTATION of 0..4 rather than sampling
independently, so a five-call block exercises every rendering exactly once. Five
iid draws would leave a rung untested about half the time, which is the opposite
of what a trial run is for.
"""

import csv
import datetime
import os
import random


# THREE CLOCKS, and they are not interchangeable -- hence three columns rather
# than one ambiguous `timestamp`:
#
#   loa_frame_ts    PROVOICE machine. The frame the decision was computed FROM,
#                   exactly as decisions.csv stamps it. This is the one that
#                   identifies the ~10 s window of driver state behind the
#                   prediction, so it is what joins a call to raw_data.jsonl.
#   call_onset_ts   DRIVE machine. Wall clock when the phone actually rang.
#                   Locates the call in the session and in any recording.
#   logged_ts       DRIVE machine. When the row was written, i.e. when the
#                   interaction RESOLVED. Useful only for ordering rows.
#
# The two machines have separate wall clocks, so do NOT difference loa_frame_ts
# against call_onset_ts to get staleness -- skew would show up as age. That is
# what loa_age_ms is for: computed on the sender, in its own clock.
CALL_EVENT_COLUMNS = (
    'logged_ts', 'call_onset_ts', 'loa_frame_ts', 'loa_age_ms',
    'session_id', 'participantid', 'block_idx', 'k_condition',
    'call_idx', 'loa_source', 'served_loa', 'checkpoint_id',
    'event_onset_ms', 'speed_kmh_at_onset', 'driver_response', 'input_mode',
    'response_latency_ms', 'outcome', 'skipped_reason',
)

# Presets. --short-trial exists so the whole path -- arming, gating, rendering,
# input, logging -- can be walked in two minutes instead of ten.
FULL_TRIAL = dict(duration_s=600.0, n_calls=5, interval_s=120.0, jitter_s=20.0)
SHORT_TRIAL = dict(duration_s=120.0, n_calls=5, interval_s=24.0, jitter_s=4.0)

# How long a due call waits for the motion gate before it is written off. A
# driver stopped this long is not driving, and the call would land on a
# stationary car -- which measures something other than what the study asks.
MAX_DEFER_S = 45.0
DEFER_RETRY_S = 1.0

# The block ends when every call is ACCOUNTED FOR, not when the nominal duration
# elapses. Spacing is relative to the previous call, so any deferral -- a driver
# stopped at a light, a call held for the rating pop-up -- pushes the rest back,
# and a hard cut at `duration_s` silently drops the last call. Losing a fifth of
# a condition's data to a red light is far worse than a block running long.
# `duration_s` therefore sets the NOMINAL length (and the schedule that fills
# it); this is the runaway guard on top.
OVERRUN_FACTOR = 1.5

MIN_SPEED_KMH = 5.0        # "moving" for the gate
MIN_MOVING_S = 3.0         # ...and for at least this long


class LoASource(object):
    """Where the LoA for the next call comes from. Never invents one."""

    def __init__(self, mode, status_path=None, session_id='', seed=None,
                 max_age_ms=None):
        self.mode = mode
        self.status_path = status_path
        self.session_id = session_id
        self.max_age_ms = max_age_ms
        self._rng = random.Random(seed)
        self._deck = []
        self.last_age_ms = None
        self.last_frame_ts = ''
        self.last_checkpoint = ''

    def _deal(self):
        if not self._deck:
            self._deck = [0, 1, 2, 3, 4]
            self._rng.shuffle(self._deck)
        return self._deck.pop()

    def next_loa(self, read_status):
        """(loa, reason). `loa` None means do not arm; `reason` says why."""
        self.last_age_ms = None
        self.last_frame_ts = ''
        self.last_checkpoint = ''
        if self.mode == 'random':
            return self._deal(), ''

        record = read_status(self.status_path)
        if not record:
            return None, 'no_status_file'
        if self.session_id and (record.get('session_id') or '').strip() != self.session_id:
            return None, 'status_other_session'
        raw = record.get('latest_loa')
        if raw is None or raw == '':
            return None, 'no_decision_yet'
        try:
            loa = int(raw)
        except (TypeError, ValueError):
            return None, 'bad_decision_value'
        if loa < 0 or loa > 4:
            return None, 'loa_out_of_range'

        age = record.get('latest_loa_age_ms')
        try:
            self.last_age_ms = int(age) if age is not None else None
        except (TypeError, ValueError):
            self.last_age_ms = None
        # Freshness, not just presence: a ProVoice stall would otherwise serve a
        # decision computed from driver state minutes old, with nothing to show
        # for it in the data.
        if (self.max_age_ms is not None and self.last_age_ms is not None
                and self.last_age_ms > self.max_age_ms):
            return None, 'decision_stale'
        # The frame the decision was computed from -- the whole point of
        # logging it is being able to recover the window that produced the
        # prediction. Absent is not fatal (the call still fires), but it costs
        # that recovery, so it is worth noticing in the data.
        self.last_frame_ts = str(record.get('latest_loa_frame_ts') or '')
        self.last_checkpoint = str(record.get('checkpoint_id') or '')
        return loa, ''


class StudySession(object):
    """One block: schedules N calls over a fixed duration and logs each one."""

    def __init__(self, source, duration_s, n_calls, interval_s, jitter_s,
                 warmup_s=None, seed=None, log_path=None, session_id='',
                 participantid='', block_idx='', k_condition=''):
        self.source = source
        self.duration_ms = float(duration_s) * 1000.0
        self.n_calls = int(n_calls)
        self.interval_ms = float(interval_s) * 1000.0
        self.jitter_ms = float(jitter_s) * 1000.0
        # Half an interval, so the first call is not on top of the start and the
        # last is not on top of the end: at 5x120 s over 600 s that lands them
        # at 60, 180, 300, 420, 540 s.
        self.warmup_ms = (float(warmup_s) * 1000.0 if warmup_s is not None
                          else self.interval_ms / 2.0)
        self._rng = random.Random(seed)
        self.log_path = log_path
        self.session_id = session_id
        self.participantid = participantid
        self.block_idx = block_idx
        self.k_condition = k_condition

        self.started_ms = None
        self.calls_fired = 0
        self.calls_skipped = 0
        self.call_idx = 0
        self._due_ms = None
        self._due_since_ms = None
        self._moving_since_ms = None
        self._pending = None
        self.overran = False

    # -- schedule ------------------------------------------------------------

    def start(self, now_ms):
        self.started_ms = now_ms
        self._schedule(now_ms, first=True)

    def _schedule(self, now_ms, first=False):
        if self.call_idx >= self.n_calls:
            self._due_ms = None
            return
        base = self.warmup_ms if first else self.interval_ms
        # Jitter is uniform and symmetric: exactly every 120 s is predictable
        # and drivers begin anticipating it, which changes what is measured.
        self._due_ms = now_ms + base + self._rng.uniform(-self.jitter_ms,
                                                         self.jitter_ms)
        self._due_since_ms = None

    def elapsed_s(self, now_ms):
        return 0.0 if self.started_ms is None else (now_ms - self.started_ms) / 1000.0

    def finished(self, now_ms):
        """Every call accounted for, or the overrun guard has tripped."""
        if self.started_ms is None:
            return False
        if (self.calls_fired + self.calls_skipped >= self.n_calls
                and self._pending is None):
            return True
        if (now_ms - self.started_ms) >= self.duration_ms * OVERRUN_FACTOR:
            self.overran = True
            return True
        return False

    # -- arming --------------------------------------------------------------

    def _gate(self, now_ms, speed_kmh, popup_active, call_active):
        """None if clear to fire, else the reason it is being held."""
        if call_active or self._pending is not None:
            return 'call_in_progress'
        if popup_active:
            return 'popup_open'
        if speed_kmh is None or speed_kmh < MIN_SPEED_KMH:
            self._moving_since_ms = None
            return 'stationary'
        if self._moving_since_ms is None:
            self._moving_since_ms = now_ms
        if (now_ms - self._moving_since_ms) < MIN_MOVING_S * 1000.0:
            return 'just_started_moving'
        return None

    def update(self, now_ms, speed_kmh, popup_active, call_active, read_status):
        """Returns (loa, call_idx) when a call should be armed, else None."""
        if self.started_ms is None or self._due_ms is None:
            return None
        if now_ms < self._due_ms:
            return None

        if self._due_since_ms is None:
            self._due_since_ms = now_ms

        held = self._gate(now_ms, speed_kmh, popup_active, call_active)
        if held is not None:
            if (now_ms - self._due_since_ms) >= MAX_DEFER_S * 1000.0:
                self._write_skip(now_ms, held, speed_kmh)
                self.call_idx += 1
                self.calls_skipped += 1
                self._schedule(now_ms)
            else:
                self._due_ms = now_ms + DEFER_RETRY_S * 1000.0
            return None

        loa, reason = self.source.next_loa(read_status)
        if loa is None:
            # No decision is NOT a reason to invent one. The window is written
            # off with its reason, which is recoverable in analysis; a
            # fabricated LoA would be indistinguishable from a served one.
            self._write_skip(now_ms, reason or 'no_decision', speed_kmh)
            self.call_idx += 1
            self.calls_skipped += 1
            self._schedule(now_ms)
            return None

        self.call_idx += 1
        self.calls_fired += 1
        self._pending = {
            'call_idx': self.call_idx,
            'served_loa': loa,
            'loa_age_ms': self.source.last_age_ms,
            'loa_frame_ts': self.source.last_frame_ts,
            'checkpoint_id': self.source.last_checkpoint,
            'call_onset_ts': datetime.datetime.now().isoformat(
                timespec='milliseconds'),
            'speed_kmh_at_onset': round(speed_kmh, 1) if speed_kmh is not None else '',
        }
        self._schedule(now_ms)
        return loa, self.call_idx

    def note_outcome(self, outcome, now_ms, input_mode='wheel'):
        """Called with CallEvent's outcome dict when the interaction resolves."""
        row = dict(self._pending or {})
        self._pending = None
        row.update({
            'event_onset_ms': outcome.get('event_onset_ms', ''),
            'driver_response': outcome.get('driver_response', ''),
            'response_latency_ms': outcome.get('response_latency_ms', ''),
            'outcome': outcome.get('outcome', ''),
            'input_mode': input_mode,
            'skipped_reason': '',
        })
        self._write(row)

    # -- logging -------------------------------------------------------------

    def _write_skip(self, now_ms, reason, speed_kmh):
        self._write({
            'call_idx': self.call_idx + 1,
            'served_loa': '',
            'skipped_reason': reason,
            'speed_kmh_at_onset': round(speed_kmh, 1) if speed_kmh is not None else '',
        })
        print('[study] call %d SKIPPED (%s)' % (self.call_idx + 1, reason))

    def _write(self, row):
        if not self.log_path:
            return
        full = {k: '' for k in CALL_EVENT_COLUMNS}
        full.update({
            'logged_ts': datetime.datetime.now().isoformat(timespec='milliseconds'),
            'session_id': self.session_id,
            'participantid': self.participantid,
            'block_idx': self.block_idx,
            'k_condition': self.k_condition,
            'loa_source': self.source.mode,
        })
        for k, v in row.items():
            if k in full:
                full[k] = '' if v is None else v
        new = not os.path.exists(self.log_path) or os.path.getsize(self.log_path) == 0
        d = os.path.dirname(self.log_path)
        if d:
            os.makedirs(d, exist_ok=True)
        with open(self.log_path, 'a', newline='', encoding='utf-8') as f:
            w = csv.DictWriter(f, fieldnames=CALL_EVENT_COLUMNS)
            if new:
                w.writeheader()
            w.writerow(full)

    def summary(self):
        base = ('%d call(s) fired, %d skipped, of %d planned'
                % (self.calls_fired, self.calls_skipped, self.n_calls))
        if self.overran:
            base += (' -- ENDED ON THE OVERRUN GUARD at %.0f x the nominal '
                     'duration, so calls are MISSING; check how much of the '
                     'block the driver spent stationary' % OVERRUN_FACTOR)
        return base
