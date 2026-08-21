#!/usr/bin/env python3
"""REVERSE bridge: ProVoice -> the CARLA machine.

scripts/vehicle_state_server.py carries CARLA state one way (CARLA machine ->
ProVoice machine). This carries two lifecycle signals the other way, so a split
session behaves like a single-machine one:

  collection_started  ProVoice has logged its first frame for this session.
                      Drive holds its LoA windows until it sees this, exactly
                      as it holds them for the first line of raw_data.jsonl in
                      a local run -- windows opened before ProVoice logs have
                      no driver-state data behind them and their labels are
                      dropped by scripts/build_loa_dataset.py.
  provoice_ended      ProVoice has exited. Drive stops the car and shows the
                      end-of-session screen instead of leaving the participant
                      driving a session that is no longer being recorded.

    python scripts/provoice_status_server.py --port 8081 --session-id <id>

start_experiment.py --remote starts this for you and hands the URL to ProVoice
through the vehicle bridge's /session, so nothing is typed on either machine.

--------------------------------------------------------------------------
WHY A FILE, AND WHY A SEPARATE PROCESS
--------------------------------------------------------------------------
Drive does not speak HTTP and should not learn to: it runs a 60 Hz pygame loop
where a blocking socket read is a dropped frame for the participant. So events
are published into a small JSON file (--out) written atomically, and Drive
polls it once a second -- the same shape as its existing raw_data.jsonl watch,
and the same file-handoff the vehicle_id.txt and vehicle_state_file_bridge
paths already use.

The file is also why this is a separate process from the vehicle-state server
rather than two more routes on it. That server holds a carla.Client, and it is
built to EXIT when CARLA goes away so its supervisor can restart it with a
fresh one. Events living in that process would be lost on exactly the restart
that a mid-session CARLA hiccup causes, and the two signals here are one-shot
and unrecoverable: miss provoice_ended and the participant keeps driving a
dead session. Nothing here touches CARLA, so nothing here restarts.

--------------------------------------------------------------------------
SESSION SCOPING
--------------------------------------------------------------------------
With --session-id, an event carrying a different session is REJECTED (409) and
not written. The failure it guards against is real and silent: a ProVoice left
running from the previous participant, or restarted against a stale command
line, would otherwise end the current session's drive from across the network.
"""

import argparse
import datetime
import json
import os
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, ".."))
sys.path.insert(0, _HERE)

from vehicle_state_file_bridge import install_graceful_stop  # noqa: E402

# The two signals. Anything else is a 400: a typo'd event name that was quietly
# accepted would look exactly like a signal that never arrived.
EVENT_STARTED = "collection_started"
EVENT_ENDED = "provoice_ended"
# The live study's LoA feed. Unlike the two lifecycle signals this one REPEATS
# -- once per decision, at the engine's own rate -- and LAST wins rather than
# first. It carries the served level plus the frame timestamp it was computed
# from, which is what lets a call be joined back to the window of driver state
# behind the prediction.
EVENT_DECISION = "decision"
# Drive telling ProVoice the block is over. It travels back in the RESPONSE to
# ProVoice's own decision POSTs -- no new channel and no polling, because
# ProVoice is already talking to this server four times a second, so it learns
# within ~250 ms of the last call resolving.
EVENT_DRIVE_ENDED = "drive_ended"
KNOWN_EVENTS = (EVENT_STARTED, EVENT_ENDED, EVENT_DECISION, EVENT_DRIVE_ENDED)

# Body cap. These payloads are ~200 bytes; the limit stops a stray large POST
# from being read into memory before it can be rejected.
MAX_BODY_BYTES = 64 * 1024


def _now_iso():
    return datetime.datetime.now().isoformat(timespec="milliseconds")


class SessionStatus:
    """The published record. One writer at a time; readers get a whole file."""

    def __init__(self, out_path: str, session_id: "str | None") -> None:
        self.out_path = out_path
        self.session_id = session_id
        self._lock = threading.Lock()
        self._state = {
            "session_id": session_id,
            "participantid": None,
            "collection_started_ts": None,
            "collection_started_iso": None,
            "ended_ts": None,
            "ended_iso": None,
            "ended_reason": None,
            "updated_ts": None,
            "updated_iso": None,
            "events": 0,
            # --- live study LoA feed (EVENT_DECISION) ---------------------
            "latest_loa": None,
            "latest_loa_frame_ts": None,      # ProVoice's clock: the FRAME
            # This machine's clock, stamped on ARRIVAL. Drive runs here too, so
            # it can subtract this from its own time() to get a true age. The
            # alternative -- having ProVoice send an age -- measures only the
            # queueing delay, because the drive reads the value up to two
            # minutes later; and differencing latest_loa_frame_ts against a
            # local clock would read the machines' skew as staleness.
            "latest_loa_recv_ts": None,
            "checkpoint_id": None,
            "decisions": 0,
            # --- Drive -> ProVoice shutdown (EVENT_DRIVE_ENDED) ------------
            "drive_ended_ts": None,
            "drive_ended_iso": None,
            "drive_ended_reason": None,
        }

    def snapshot(self) -> dict:
        with self._lock:
            return dict(self._state)

    def record(self, event: str, payload: dict) -> dict:
        """Apply an event and republish. Returns the new state."""
        with self._lock:
            st = self._state
            st["events"] += 1
            if payload.get("participantid"):
                st["participantid"] = payload["participantid"]
            if payload.get("session_id"):
                st["session_id"] = payload["session_id"]
            now = time.time()
            if event == EVENT_STARTED:
                # FIRST one wins: the signal means "logging began", and a
                # ProVoice that re-sends it (a retry, a reconnect) has not
                # begun again. Overwriting would move the mark forward and
                # make the wait Drive already finished look unfinished.
                if st["collection_started_ts"] is None:
                    st["collection_started_ts"] = round(now, 3)
                    st["collection_started_iso"] = _now_iso()
            elif event == EVENT_DECISION:
                loa = payload.get("loa")
                try:
                    loa = int(loa)
                except (TypeError, ValueError):
                    loa = None
                if loa is not None and 0 <= loa <= 4:
                    st["latest_loa"] = loa
                    st["latest_loa_frame_ts"] = payload.get("frame_ts") or None
                    st["latest_loa_recv_ts"] = round(now, 3)
                    if payload.get("checkpoint_id"):
                        st["checkpoint_id"] = payload["checkpoint_id"]
                    st["decisions"] += 1
            elif event == EVENT_DRIVE_ENDED:
                # FIRST wins, like collection_started: the block ended once, and
                # a resend (a retry, an operator closing the window afterwards)
                # must not move the mark.
                if st["drive_ended_ts"] is None:
                    st["drive_ended_ts"] = round(now, 3)
                    st["drive_ended_iso"] = _now_iso()
                    st["drive_ended_reason"] = payload.get("reason") or ""
            elif event == EVENT_ENDED:
                if st["ended_ts"] is None:
                    st["ended_ts"] = round(now, 3)
                    st["ended_iso"] = _now_iso()
                    st["ended_reason"] = payload.get("reason") or ""
            st["updated_ts"] = round(now, 3)
            st["updated_iso"] = _now_iso()
            state = dict(st)
            self._write(state)
        return state

    def _write(self, state: dict) -> None:
        """Atomic republish: readers see the old file or the new one, never half.

        os.replace is atomic on Windows and POSIX alike. Drive polls this file
        from its render loop, so a partially written record is not a hypothetical
        -- it would be a JSON parse error one frame in a hundred, and the
        watcher would read it as "no signal yet".
        """
        tmp = self.out_path + ".tmp"
        try:
            os.makedirs(os.path.dirname(self.out_path) or ".", exist_ok=True)
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(state, f, indent=2)
            os.replace(tmp, self.out_path)
        except OSError as e:
            # Never fatal: the HTTP side still answers and /status still holds
            # the truth, so a transient file error costs one republish rather
            # than the channel.
            print(f"[status] FAILED to write {self.out_path}: {e}", flush=True)

    def publish_initial(self) -> None:
        """Adopt an existing record for THIS session, else publish an empty one.

        Two jobs. Normally it just makes the file exist while the server is up,
        so a reader can tell 'the channel is live, nothing has happened yet'
        from 'no channel'.

        The adoption matters on a RESTART. This server is supervised, and the
        naive version -- always write the empty record -- would erase a signal
        that had already arrived, turning a restart into an un-ending session:
        provoice_ended was delivered, ProVoice is gone, and nothing will ever
        send it again. Reloading keeps one-shot signals one-shot rather than
        one-shot-per-process.

        A record from a DIFFERENT session is not adopted: that is the stale
        file from the previous participant, and inheriting its ended_ts would
        end this drive on sight.
        """
        try:
            with open(self.out_path, "r", encoding="utf-8") as f:
                previous = json.load(f)
        except (OSError, ValueError):
            previous = None

        if isinstance(previous, dict) and self.session_id and \
                previous.get("session_id") == self.session_id:
            with self._lock:
                for key in self._state:
                    if key in previous:
                        self._state[key] = previous[key]
            print(f"[status] resumed {self.out_path} for session "
                  f"{self.session_id!r}: started="
                  f"{self._state['collection_started_iso']} "
                  f"ended={self._state['ended_iso']}", flush=True)
        self._write(self.snapshot())


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "ProVoiceStatusBridge/1"
    sys_version = ""

    # --- routes -----------------------------------------------------------

    def do_POST(self):
        route = self.path.split("?", 1)[0].rstrip("/")
        if route not in ("/event", "/events"):
            self._respond(404, {"error": "not found"})
            return
        try:
            length = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            length = 0
        if length > MAX_BODY_BYTES:
            self._respond(413, {"error": "body too large"})
            return
        try:
            payload = json.loads(self.rfile.read(length) or b"{}")
            if not isinstance(payload, dict):
                raise ValueError("expected a JSON object")
        except Exception as e:  # noqa: BLE001 — a bad body is the client's fault
            self._respond(400, {"error": f"unparseable body: {e}"})
            return

        event = (payload.get("event") or "").strip()
        if event not in KNOWN_EVENTS:
            self._respond(400, {"error": f"unknown event {event!r}",
                                "known": list(KNOWN_EVENTS)})
            return

        expected = self.server.status.session_id
        got = (payload.get("session_id") or "").strip()
        if expected and got and got != expected:
            print(f"[status] REJECTED {event} from session {got!r}: this server "
                  f"is scoped to {expected!r}. A ProVoice from another session "
                  f"is talking to this machine.", flush=True)
            self._respond(409, {"error": "session mismatch",
                                "expected": expected, "got": got})
            return

        state = self.server.status.record(event, payload)
        # Decisions arrive at the engine's rate (4 Hz) and would otherwise
        # produce four identical lines a second for the whole session. The
        # count is in /health and in the published record; the FIRST one is
        # worth announcing, because it is the proof the feed came up at all.
        if event != EVENT_DECISION or state.get("decisions") == 1:
            print(f"[status] {event}"
                  + (f" #{state['decisions']}" if event == EVENT_DECISION else "")
                  + (f" (reason: {payload['reason']})" if payload.get("reason") else "")
                  + f" -> {self.server.status.out_path}", flush=True)
        self._respond(200, {"ok": True, "event": event, "status": state})

    def do_GET(self):
        route = self.path.split("?", 1)[0].rstrip("/")
        if route in ("", "/status"):
            self._respond(200, self.server.status.snapshot())
        elif route in ("/health", "/healthz"):
            st = self.server.status.snapshot()
            self._respond(200, {
                "ok": True,
                "session_id": st["session_id"],
                "started": st["collection_started_ts"] is not None,
                "ended": st["ended_ts"] is not None,
                "events": st["events"],
                "out": self.server.status.out_path,
            })
        else:
            self._respond(404, {"error": "not found"})

    do_HEAD = do_GET

    # --- plumbing ---------------------------------------------------------

    def _respond(self, code: int, body: dict) -> None:
        raw = json.dumps(body).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(raw)

    def handle_one_request(self):
        # A client that vanishes mid-request is routine (ProVoice exits right
        # after posting the end event) and must not print a traceback.
        try:
            super().handle_one_request()
        except (ConnectionResetError, ConnectionAbortedError, BrokenPipeError):
            self.close_connection = True

    def log_message(self, *_args):
        pass  # the [status] lines above are the log

    def log_error(self, *_args):
        pass


class StatusServer(ThreadingHTTPServer):
    daemon_threads = True
    block_on_close = False
    allow_reuse_address = True

    def __init__(self, addr, handler_cls, status: SessionStatus):
        super().__init__(addr, handler_cls)
        self.status = status


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8081)
    parser.add_argument("--bind", default="0.0.0.0",
                        help="Interface to listen on. 0.0.0.0 accepts from the "
                             "LAN, which is the point of this bridge.")
    parser.add_argument("--out", default="provoice_status.json",
                        help="File the events are published into, polled by "
                             "Drive. Written atomically.")
    parser.add_argument("--session-id", dest="session_id", default=None,
                        help="Scope this server to one session: events carrying "
                             "any other session id are rejected. Without it, "
                             "whatever ProVoice reaches this port can end the "
                             "drive.")
    parser.add_argument("--idle-timeout", type=float, default=30.0,
                        help="Close a kept-alive connection after this long with "
                             "no request.")
    args = parser.parse_args()

    install_graceful_stop()

    status = SessionStatus(os.path.abspath(args.out), args.session_id)
    status.publish_initial()
    Handler.timeout = args.idle_timeout

    httpd = StatusServer((args.bind, args.port), Handler, status)
    thread = threading.Thread(target=httpd.serve_forever,
                              kwargs={"poll_interval": 0.5},
                              name="status-server", daemon=True)
    thread.start()
    print(f"[status] serving on http://{args.bind}:{args.port}/ "
          f"(POST /event, GET /status, GET /health)", flush=True)
    print(f"[status] publishing to {status.out_path}", flush=True)
    print(f"[status] scoped to session {args.session_id!r}"
          if args.session_id else
          "[status] NOT scoped to a session: any ProVoice reaching this port "
          "can signal it", flush=True)

    try:
        while True:
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("[status] stopping...", flush=True)
    finally:
        httpd.shutdown()
        httpd.server_close()
        st = status.snapshot()
        print(f"[status] {st['events']} event(s) handled; "
              f"started={st['collection_started_iso']} "
              f"ended={st['ended_iso']}", flush=True)


if __name__ == "__main__":
    main()
