#!/usr/bin/env python3
"""
REMOTE vehicle-state bridge: reads a local CARLA instance and serves the state
over HTTP for a ProVoice running on ANOTHER machine.

    python scripts/vehicle_state_server.py --port 8080 --vehicle-id 42

start_experiment.py --remote starts this for you, passing the vehicle id it has
already validated, and prints the matching ProVoice command line for the other
machine. Over the internet rather than a LAN, put a tunnel in front of it:

    ngrok http 8080

and pass the tunnel URL as vehicle_state_url=... to ProVoice.

NOTE: for a LOCAL, same-machine setup do NOT use this script -- use
scripts/vehicle_state_file_bridge.py instead. It publishes the same fields
through a file rather than a socket, which keeps the network stack out of the
loop entirely (see that file's header for the bugcheck history).

--------------------------------------------------------------------------
CONNECTION HANDLING -- why this is not a plain BaseHTTPRequestHandler
--------------------------------------------------------------------------
The first version of this script spoke HTTP/1.0 and its client used
urllib.urlopen, which sends "Connection: close". At ProVoice's 20 Hz poll rate
that is ~20 TCP connect/teardown cycles per SECOND, ~36,000 per 30-minute
session, each one churning kernel socket state (and, on Windows, leaving a
TIME_WAIT entry behind). That connection churn is the single kernel path the
HTTP bridge introduces and nothing else in the experiment touches, and it is
the prime suspect for the two machine bugchecks of 2026-07-28.

So this version:

  * speaks HTTP/1.1 with keep-alive and always sends an accurate
    Content-Length, so one connection carries the whole session's polls;
  * is threaded, which keep-alive makes mandatory -- a persistent connection
    occupies its handler thread for as long as it is open, so a single-threaded
    server would serve exactly one client, forever;
  * drops idle connections after --idle-timeout so a peer that vanishes without
    a FIN does not leak a thread for the rest of the session;
  * reports "N requests served over M TCP connection(s)" every --stats-every
    seconds. With a keep-alive client, M stays at 1 per client for the whole
    run -- that line is how you confirm the churn is actually gone.

The matching client change is in ProVoice's DataCollector._poll_vehicle_state,
which now holds one http.client connection open instead of calling urlopen per
poll. Keep-alive needs BOTH ends: a server that offers it and a client that
closes anyway saves nothing.

--------------------------------------------------------------------------
CARLA HANDLING
--------------------------------------------------------------------------
A background thread samples CARLA at a fixed rate and requests are served from
the last cached record. This decouples three things that used to be one:

  * CARLA's RPC load no longer depends on how many clients poll, or how fast --
    the old version issued ~8 RPC calls per REQUEST, straight from the handler;
  * a slow or hung CARLA read can no longer stall an HTTP request (a request is
    now a dict lookup and a socket write);
  * a lost CARLA connection is recoverable in place: the sampler rebuilds the
    client, re-resolves the actor and carries on, instead of the process dying.

Every response carries X-State-Age, the age of the cached record in seconds,
measured on THIS machine with a monotonic clock. The reader uses that header
rather than the record's wall-clock "ts" to detect a stalled feed, because the
two machines' wall clocks do not agree and a few seconds of skew would
otherwise make a perfectly fresh feed look stale.
"""

import argparse
import json
import os
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, ".."))
sys.path.insert(0, _HERE)

try:
    import carla
except ImportError:
    print("[httpbridge] ERROR: CARLA Python API not found. Run from the project venv.")
    sys.exit(1)

# ONE definition of the record, shared with the local file bridge. ProVoice
# consumes whichever bridge it is pointed at, so a field present in one and
# missing from the other would silently change what a column means between a
# local run and a remote one.
from vehicle_state_file_bridge import (  # noqa: E402
    collect,
    install_graceful_stop,
    read_vehicle_id,
)


class Stats:
    """Connection/request counters, shared by the handler and the sampler."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.connections = 0      # cumulative TCP connections accepted
        self.open = 0             # currently open
        self.requests = 0         # cumulative HTTP requests served

    def connection_opened(self) -> None:
        with self._lock:
            self.connections += 1
            self.open += 1

    def connection_closed(self) -> None:
        with self._lock:
            self.open -= 1

    def request_served(self) -> None:
        with self._lock:
            self.requests += 1

    def snapshot(self) -> dict:
        with self._lock:
            return {"connections": self.connections, "open": self.open,
                    "requests": self.requests}


class VehicleStateSampler:
    """Owns the CARLA client; republishes one serialized record at a fixed rate.

    Holds a thread rather than subclassing Thread. Thread reserves a long list
    of private attribute names, and one of them is _stop -- a METHOD that
    join() calls internally. A "_stop" event here silently shadowed it, and the
    only symptom was a TypeError at shutdown, i.e. exactly when a clean CARLA
    teardown matters most. Composition keeps the namespaces separate.
    """

    def __init__(self, args, stats: Stats) -> None:
        self.args = args
        self.stats = stats
        self.interval = 1.0 / max(1e-3, args.hz)
        # --vehicle-id pins the actor for the process lifetime. Without it the
        # id is re-read from vehicle_id.txt on every (re)connect, so a bridge
        # left running across two Drive sessions attaches to the new ego.
        self._pinned_id = args.vehicle_id
        self.vehicle_id = args.vehicle_id

        self._lock = threading.Lock()
        self._payload: "bytes | None" = None
        self._sampled_at = 0.0     # monotonic
        self._thread: "threading.Thread | None" = None
        self._stopping = threading.Event()
        self.fatal = threading.Event()   # "give up, let the supervisor restart us"

        self.samples = 0
        self.errors = 0            # CONSECUTIVE failures
        self.reconnects = 0
        self._client = None
        self._world = None
        self._actor = None
        self._map = None

    # --- lifecycle --------------------------------------------------------

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, name="carla-sampler",
                                        daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stopping.set()

    def join(self, timeout: "float | None" = None) -> None:
        if self._thread is not None:
            self._thread.join(timeout)

    # --- state exposed to the HTTP handler --------------------------------

    def snapshot(self) -> "tuple[bytes | None, float]":
        """The last record and its age in seconds. (None, 0.0) before the first."""
        with self._lock:
            if self._payload is None:
                return None, 0.0
            return self._payload, max(0.0, time.monotonic() - self._sampled_at)

    # --- CARLA ------------------------------------------------------------

    def _release(self) -> None:
        """Drop the CARLA handles so the RPC connection is closed properly.

        Order matters: the actor and the world are views onto the client's
        connection, and outliving it leaves them pointing at a torn-down
        transport. A cleanly closed connection also matters to the SERVER --
        an abruptly severed one is what it struggles to survive when the next
        client connects.
        """
        self._actor = None
        self._map = None
        self._world = None
        self._client = None

    def _connect(self) -> None:
        a = self.args
        self._release()

        if self._pinned_id is not None:
            vid = self._pinned_id
        else:
            vid = read_vehicle_id(a.vehicle_id_path, wait=a.connect_timeout)
            if vid is None:
                raise RuntimeError(
                    f"{a.vehicle_id_path} not readable within {a.connect_timeout:.0f}s")

        client = carla.Client(a.carla_host, a.carla_port)
        client.set_timeout(a.carla_timeout)
        world = client.get_world()
        actor = world.get_actor(vid)
        if actor is None:
            raise RuntimeError(f"actor id={vid} is not present in the CARLA world")

        self._client, self._world, self._actor = client, world, actor
        # Cached: get_map() re-parses the whole OpenDRIVE description and must
        # not be called per sample.
        self._map = world.get_map()
        self.vehicle_id = vid
        print(f"[httpbridge] tracking actor id={vid} type={actor.type_id}", flush=True)

    # --- loop -------------------------------------------------------------

    def _run(self) -> None:
        a = self.args
        next_t = time.monotonic()
        last_stats = time.monotonic()

        while not self._stopping.is_set():
            try:
                if self._actor is None:
                    self._connect()
                sample = collect(self._actor, self._world, self._map)
                payload = json.dumps(sample).encode("utf-8")
                with self._lock:
                    self._payload = payload
                    self._sampled_at = time.monotonic()
                self.samples += 1
                if self.errors:
                    print(f"[httpbridge] recovered after {self.errors} failed read(s)",
                          flush=True)
                    self.errors = 0
            except Exception as e:  # noqa: BLE001 — a read failure must not end the run
                self.errors += 1
                if self.errors == 1 or self.errors % 20 == 0:
                    print(f"[httpbridge] CARLA read FAILED (#{self.errors}): "
                          f"{type(e).__name__}: {e}", flush=True)
                # A run of failures means the client or the actor is gone (the
                # ego was destroyed, the server restarted, the connection
                # dropped). Rebuild rather than reading a dead handle until the
                # error budget runs out.
                if a.reconnect_after and self.errors % a.reconnect_after == 0:
                    self.reconnects += 1
                    print(f"[httpbridge] rebuilding the CARLA client "
                          f"(reconnect #{self.reconnects})", flush=True)
                    self._release()
                if self.errors >= a.max_consecutive_errors:
                    print(f"[httpbridge] {self.errors} consecutive failures; exiting "
                          f"so the supervisor can restart with a fresh client.",
                          flush=True)
                    self.fatal.set()
                    break

            now = time.monotonic()
            if a.stats_every and now - last_stats >= a.stats_every:
                last_stats = now
                st = self.stats.snapshot()
                # The point of this line: with a keep-alive client, requests
                # climbs into the thousands while connections stays at 1.
                print(f"[httpbridge] {st['requests']} requests over "
                      f"{st['connections']} TCP connection(s) ({st['open']} open); "
                      f"{self.samples} CARLA samples, {self.reconnects} reconnect(s)",
                      flush=True)

            next_t += self.interval
            if next_t < now:          # fell behind: skip the missed ticks
                next_t = now
            self._stopping.wait(max(0.0, next_t - now))

        self._release()
        print(f"[httpbridge] sampler stopped after {self.samples} samples.", flush=True)


class Handler(BaseHTTPRequestHandler):
    # HTTP/1.1 is what enables keep-alive. It also makes an accurate
    # Content-Length mandatory on EVERY response: without one the client cannot
    # tell where the body ends and waits for the connection to close, which is
    # exactly the close we are trying to avoid. _respond() is the only place
    # that writes a response, so that header can never be forgotten.
    protocol_version = "HTTP/1.1"
    server_version = "VehicleStateBridge/2"
    sys_version = ""
    disable_nagle_algorithm = True   # TCP_NODELAY: these payloads are ~400 bytes

    def setup(self):
        # socketserver applies self.timeout to the connection in setup(); it is
        # set from the CLI on the class in main().
        super().setup()
        self.server.stats.connection_opened()

    def finish(self):
        try:
            super().finish()
        finally:
            self.server.stats.connection_closed()

    # --- routes -----------------------------------------------------------

    def do_GET(self):
        route = self.path.split("?", 1)[0].rstrip("/")
        if route in ("", "/state", "/vehicle_state"):
            self._serve_state()
        elif route in ("/health", "/healthz"):
            self._serve_health()
        else:
            self._respond(404, b'{"error": "not found"}')

    # HEAD shares the routing; _respond() suppresses the body for it.
    do_HEAD = do_GET

    def _serve_state(self):
        payload, age = self.server.sampler.snapshot()
        if payload is None:
            # Nothing sampled yet (CARLA still connecting). 503 rather than an
            # empty record: ProVoice logs a failed poll and holds its last
            # values, which is right, where zeros would be written as data.
            self._respond(503, b'{"error": "no vehicle state sampled yet"}',
                          extra={"Retry-After": "1"})
            return
        self.server.stats.request_served()
        self._respond(200, payload, extra={"X-State-Age": f"{age:.3f}"})

    def _serve_health(self):
        sampler = self.server.sampler
        payload, age = sampler.snapshot()
        st = self.server.stats.snapshot()
        body = json.dumps({
            "ok": payload is not None and sampler.errors == 0,
            "vehicle_id": sampler.vehicle_id,
            "state_age_s": round(age, 3) if payload is not None else None,
            "samples": sampler.samples,
            "consecutive_errors": sampler.errors,
            "reconnects": sampler.reconnects,
            "requests": st["requests"],
            "connections": st["connections"],
            "open_connections": st["open"],
        }).encode("utf-8")
        self._respond(200, body)

    # --- plumbing ---------------------------------------------------------

    def _respond(self, code: int, body: bytes, ctype: str = "application/json",
                 extra: "dict | None" = None) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        for key, value in (extra or {}).items():
            self.send_header(key, value)
        self.end_headers()
        if self.command != "HEAD" and body:
            self.wfile.write(body)

    def handle_one_request(self):
        # A client that disappears mid-response is routine here (ProVoice
        # restarts, the tunnel reconnects) and must not print a traceback per
        # occurrence. Idle keep-alive connections time out instead of leaking a
        # thread; the base class turns that into close_connection itself.
        try:
            super().handle_one_request()
        except (ConnectionResetError, ConnectionAbortedError, BrokenPipeError):
            self.close_connection = True

    def log_message(self, *_args):
        pass  # per-request logging at 20 Hz would bury everything else

    def log_error(self, *_args):
        pass  # includes the routine "Request timed out" on an idle connection


class BridgeServer(ThreadingHTTPServer):
    daemon_threads = True
    # Keep-alive connections live as long as their client, so joining them at
    # shutdown would stall exit until every peer disconnects (or the idle
    # timeout expires). They are daemon threads and die with the process.
    block_on_close = False
    allow_reuse_address = True
    request_queue_size = 32

    def __init__(self, addr, handler_cls, sampler: VehicleStateSampler, stats: Stats):
        super().__init__(addr, handler_cls)
        self.sampler = sampler
        self.stats = stats


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--bind", default="localhost",
                        help="ngrok connected to localhost.")
    parser.add_argument("--hz", type=float, default=20.0,
                        help="Rate at which CARLA is sampled, independent of how "
                             "often clients poll. Should match the collection "
                             "loop: every field is held constant between samples, "
                             "so this is the true sampling rate of steer/brake.")
    parser.add_argument("--carla-host", default="localhost")
    parser.add_argument("--carla-port", type=int, default=2000)
    parser.add_argument("--carla-timeout", type=float, default=10.0)
    parser.add_argument("--vehicle-id-path", default="vehicle_id.txt")
    parser.add_argument("--vehicle-id", type=int, default=None,
                        help="Skip vehicle_id.txt discovery and pin this actor. "
                             "The launcher passes it, having already read and "
                             "validated the id.")
    parser.add_argument("--connect-timeout", type=float, default=60.0,
                        help="How long to wait for vehicle_id.txt when --vehicle-id "
                             "is not given.")
    parser.add_argument("--idle-timeout", type=float, default=30.0,
                        help="Close a kept-alive connection after this many seconds "
                             "with no request, so a peer that vanished without a FIN "
                             "does not hold its handler thread forever. Must stay "
                             "well above the client's poll interval or the "
                             "connection reuse this server exists for is lost.")
    parser.add_argument("--stats-every", type=float, default=60.0,
                        help="Seconds between the 'N requests over M connections' "
                             "lines. 0 disables them.")
    parser.add_argument("--reconnect-after", type=int, default=10,
                        help="Rebuild the CARLA client after this many consecutive "
                             "failed reads. 0 never rebuilds.")
    parser.add_argument("--max-consecutive-errors", type=int, default=200,
                        help="Exit (for the supervisor to restart us) after this "
                             "many failed reads in a row. A single bad read is "
                             "normal; a long run of them means CARLA is gone.")
    args = parser.parse_args()

    install_graceful_stop()

    stats = Stats()
    sampler = VehicleStateSampler(args, stats)
    Handler.timeout = args.idle_timeout

    httpd = BridgeServer((args.bind, args.port), Handler, sampler, stats)
    server_thread = threading.Thread(
        target=httpd.serve_forever, kwargs={"poll_interval": 0.5},
        name="http-server", daemon=True)

    sampler.start()
    server_thread.start()
    print(f"[httpbridge] serving on http://{args.bind}:{args.port}/ "
          f"(HTTP/1.1 keep-alive, sampling CARLA at {args.hz:.1f} Hz)", flush=True)
    print(f"[httpbridge] endpoints: / (state), /health", flush=True)

    rc = 0
    try:
        # Short waits so the Ctrl-Break/SIGTERM handler installed above gets a
        # chance to run in the main thread.
        while not sampler.fatal.wait(0.5):
            pass
        print("[httpbridge] sampler gave up; exiting for the supervisor to restart.",
              flush=True)
        rc = 1
    except KeyboardInterrupt:
        print("[httpbridge] stopping...", flush=True)
    finally:
        sampler.stop()
        httpd.shutdown()
        httpd.server_close()
        sampler.join(timeout=5.0)
        st = stats.snapshot()
        print(f"[httpbridge] served {st['requests']} requests over "
              f"{st['connections']} TCP connection(s).", flush=True)

    sys.exit(rc)


if __name__ == "__main__":
    main()
