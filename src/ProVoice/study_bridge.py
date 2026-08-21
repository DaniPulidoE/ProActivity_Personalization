#!/usr/bin/env python

"""Publish each LoA decision to the CARLA machine, for the live study.

Enabled by ``--study-bridge``. The drive process owns the call schedule and
reads the latest published decision at the instant a call fires; see
``docs/live_study_setup.md`` section 7 and ``src/drive/study_session.py``.

WHY A SLOT AND A THREAD, NOT A DIRECT POST
------------------------------------------
The decision engine runs on its own thread at ``--decision-hz`` (4 by default),
which leaves 250 ms per cycle. A blocking POST on that thread hands the
experiment's inference cadence to the network: one stalled socket and decisions
stop being made at all, which is a far worse failure than the bridge going
quiet.

So ``publish`` only drops into a ONE-SLOT mailbox and returns. A daemon thread
does the sending. The slot OVERWRITES rather than queues, which is not a
shortcut -- only the newest decision is ever wanted, so dropping intermediate
ones under load is the correct behaviour rather than a lossy compromise. A queue
would deliver a backlog of decisions that were already stale when they left.

CONNECTION REUSE
----------------
One held ``http.client.HTTPConnection``, reconnected on error, rather than
``urllib.request`` per POST. urllib sends ``Connection: close``, and this feed
runs at 4 Hz for the whole session -- the same connection churn that
``docs/remote_setup.md`` and CLAUDE.md tie to the two machine bugchecks of
2026-07-28. The lifecycle signals can afford urllib because there are two of
them; this cannot.
"""

import http.client
import json
import threading
import time
import urllib.parse


EVENT_DECISION = "decision"

# How long a send may block before the connection is dropped and rebuilt. Short:
# the sender thread is not on the critical path, but a socket wedged for minutes
# would keep the bridge silent while looking alive.
SEND_TIMEOUT_S = 2.0
# After a failure, wait this long before trying again. Without it a dead peer
# means a reconnect attempt at the full decision rate.
RETRY_BACKOFF_S = 2.0


class DecisionBridge(object):
    """Fire-and-forget publisher for the served LoA."""

    def __init__(self, status_url, session_id="", participantid="",
                 checkpoint_id="", on_drive_ended=None):
        self.status_url = (status_url or "").rstrip("/")
        self.session_id = session_id or ""
        self.participantid = participantid or ""
        self.checkpoint_id = checkpoint_id or ""
        # Called ONCE, from the sender thread, when the CARLA machine reports
        # the block is over. The signal rides back in the response to our own
        # decision POSTs, so it costs no extra request and arrives within one
        # decision period.
        self.on_drive_ended = on_drive_ended
        self.drive_ended = False

        parts = urllib.parse.urlsplit(self.status_url)
        self._host = parts.hostname
        self._port = parts.port or (443 if parts.scheme == "https" else 80)
        self._path = (parts.path or "").rstrip("/") + "/event"

        self._slot = None
        self._cv = threading.Condition()
        self._running = False
        self._thread = None
        self._conn = None
        self._next_try = 0.0
        self.sent = 0
        self.failed = 0
        self.dropped = 0

    # -- lifecycle -----------------------------------------------------------

    def start(self):
        if not self.status_url or not self._host:
            print("[study-bridge] no status URL; decisions will NOT be published.")
            return False
        self._running = True
        self._thread = threading.Thread(target=self._run, name="study-bridge",
                                        daemon=True)
        self._thread.start()
        print("[study-bridge] publishing decisions to %s%s"
              % (self.status_url, self._path))
        return True

    def stop(self):
        self._running = False
        with self._cv:
            self._cv.notify_all()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        self._close()
        print("[study-bridge] %d sent, %d failed, %d superseded before sending."
              % (self.sent, self.failed, self.dropped))

    # -- producer side (decision thread) -------------------------------------

    def publish(self, loa, frame_ts=None):
        """Called from the decision thread. Never blocks, never raises."""
        if not self._running:
            return
        try:
            loa = int(loa)
        except (TypeError, ValueError):
            return
        if not (0 <= loa <= 4):
            return
        with self._cv:
            if self._slot is not None:
                self.dropped += 1
            self._slot = {
                "event": EVENT_DECISION,
                "session_id": self.session_id,
                "participantid": self.participantid,
                "loa": loa,
                "frame_ts": frame_ts or "",
                "checkpoint_id": self.checkpoint_id,
                "ts": time.time(),
            }
            self._cv.notify()

    # -- sender side ---------------------------------------------------------

    def _run(self):
        while self._running:
            with self._cv:
                while self._running and self._slot is None:
                    self._cv.wait(timeout=0.5)
                payload, self._slot = self._slot, None
            if payload is None:
                continue
            now = time.monotonic()
            if now < self._next_try:
                # Still backing off from a failure. Drop this one rather than
                # letting it age in the slot -- a newer decision is moments away.
                continue
            self._send(payload)

    def _connect(self):
        self._close()
        self._conn = http.client.HTTPConnection(self._host, self._port,
                                                timeout=SEND_TIMEOUT_S)

    def _close(self):
        if self._conn is not None:
            try:
                self._conn.close()
            except Exception:                                 # noqa: BLE001
                pass
            self._conn = None

    def _check_drive_ended(self, raw):
        """Notice the CARLA machine reporting the block over. Never raises."""
        if self.drive_ended:
            return
        try:
            state = (json.loads(raw) or {}).get("status") or {}
        except Exception:                                     # noqa: BLE001
            return
        if state.get("drive_ended_ts") is None:
            return
        self.drive_ended = True
        print("[study-bridge] Drive reports the block is over (%s). Shutting "
              "down." % (state.get("drive_ended_reason") or "no reason given"))
        if self.on_drive_ended is not None:
            try:
                self.on_drive_ended()
            except Exception as exc:                          # noqa: BLE001
                print("[study-bridge] shutdown callback failed: %s" % exc)

    def _send(self, payload):
        body = json.dumps(payload).encode("utf-8")
        for attempt in (1, 2):
            try:
                if self._conn is None:
                    self._connect()
                self._conn.request("POST", self._path, body=body, headers={
                    "Content-Type": "application/json",
                    "Content-Length": str(len(body)),
                    "Connection": "keep-alive",
                })
                resp = self._conn.getresponse()
                raw = resp.read()
                if resp.status == 200:
                    self.sent += 1
                    self._check_drive_ended(raw)
                    return True
                # A 409 means the server is scoped to another session. Retrying
                # cannot fix that and it matters: two sessions are live and one
                # is talking to the wrong machine.
                print("[study-bridge] decision REJECTED: HTTP %d" % resp.status)
                self.failed += 1
                self._next_try = time.monotonic() + RETRY_BACKOFF_S
                return False
            except Exception as exc:                          # noqa: BLE001
                self._close()
                if attempt == 2:
                    self.failed += 1
                    self._next_try = time.monotonic() + RETRY_BACKOFF_S
                    if self.failed in (1, 10, 100) or self.failed % 500 == 0:
                        # Loud on the first failure, then rate-limited: a peer
                        # that is down stays down, and 4 messages a second of
                        # identical traceback would bury everything else.
                        print("[study-bridge] send failed (%s): %s"
                              % (type(exc).__name__, exc))
        return False
