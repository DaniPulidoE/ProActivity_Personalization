#!/usr/bin/env python3
"""
Experiment launcher (CARLA + Drive + ProVoice)
Improved version: process-controlled instead of terminal-based
"""

"""
MAIN OPTIONS:
--no-popup: no LoA selection popup
--fixed: fixed starting position
--calibration-only: provoice only on calibration mode, no inference afterwards
--data-collection: don't perform calibration, jump directly to data collection, no inference
--test-drive: don't launch provoice
--test-popup: teaching mode, simulator UI + immediate LoA popups, nothing logged
--remote: CARLA + traffic + Drive (with the LoA popups) run HERE and vehicle
    state is served over HTTP; ProVoice runs on ANOTHER machine and is started
    there by hand, with the command line this launcher prints

LoA POPUP INTERFACE (pick one; default = wheel if one is attached, else keyboard):
--wheel-input: paddles move the cursor, front button ticks, CONFIRM row submits
--keyboard-input: number keys 0-4 tick (same key again unticks), ENTER confirms

LoA POPUP FUNCTION:
--functionname: one function for the whole run, one popup per 20 s window
--random-function: function drawn per popup from the five study functions; each
    20 s window then holds TWO popups about two different ones (mutually
    exclusive with --functionname)


EXPERIMENT SETUP:
0. Teaching the LoA control: --test-popup --fullscreen
1. Driver adaptation phase: --test-drive --no-popup --fullscreen
2. Calibration: --fixed --no-popup --calibration-only --fullscreen --participantid
3. Inference: --data-collection --participantid --random-function --fullscreen

SPLIT ACROSS TWO MACHINES (--remote): run 2. and 3. as above plus --remote on
the CARLA machine, then paste the printed command on the ProVoice machine.
"""

import html
import os
import re
import socket
import sys
import uuid
import time
import signal
import subprocess
import argparse
from pathlib import Path


# =========================
# CONFIG
# =========================

# Used when neither --functionname nor --random-function is given. Kept in sync
# with Drive's and ProVoice's own defaults.
DEFAULT_FUNCTIONNAME = "Adjust seat positioning"

# Port nothing listens on, for --provoice-no-carla. 1 is reserved (tcpmux) and
# never bound in practice, so connections fail instantly with ECONNREFUSED
# rather than hanging until a timeout.
_DEAD_PORT = 1

# Drive's default wait for ProVoice's first logged frame. Resolved late (the
# flag defaults to None) so "the user asked for this" can be told apart from
# "nobody said", which --remote needs: there ProVoice logs on a DIFFERENT
# machine and the wait can only ever time out.
DEFAULT_POPUP_WAIT_TIMEOUT = 180.0


def outbound_ip() -> str:
    """This machine's LAN address, as the other machine would reach it.

    Connecting a UDP socket sends no packets — it only makes the OS pick the
    interface it would route through, which is exactly the address to print.
    gethostname() is not a substitute: on a machine with Hyper-V or several
    NICs (this one has both) it regularly resolves to a virtual adapter the
    other machine cannot reach.
    """
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))
        return s.getsockname()[0]
    except OSError:
        try:
            return socket.gethostbyname(socket.gethostname())
        except OSError:
            return "127.0.0.1"
    finally:
        s.close()


def shell_quote(cmd) -> str:
    """Render a command list so it can be PASTED into a shell and survive.

    Only quoting matters here, and only for spaces: without it
    `functionname=Adjust seat positioning` arrives as three arguments, ProVoice
    reports two "unrecognized" tokens, and the run silently proceeds under the
    default function. Double quotes behave identically in bash, PowerShell and
    cmd for these values, and none of them contain a quote of their own.
    """
    return " ".join(f'"{t}"' if " " in str(t) else str(t) for t in cmd)


# =========================
# CRASH DIAGNOSIS
# =========================

# Exit codes >= 0xC0000000 are NT status codes: the OS killed the process, so
# there is NO Python traceback anywhere and the bare number is all the launcher
# has. Decoding it is the difference between "exited with code 3221226505" and
# knowing the process was killed for a memory fault.
_NT_STATUS = {
    0xC0000005: ("ACCESS_VIOLATION",
                 "read or write through a bad pointer, inside native code"),
    0xC0000409: ("STACK_BUFFER_OVERRUN / __fastfail",
                 "MSVC fastfail: usually an unhandled C++ exception reaching "
                 "std::terminate, e.g. an RPC layer whose server vanished"),
    0xC000041D: ("FATAL_USER_CALLBACK_EXCEPTION",
                 "exception escaping a callback"),
    0xC00000FD: ("STACK_OVERFLOW", "runaway recursion"),
    0xC0000374: ("HEAP_CORRUPTION", "the heap was already corrupted earlier"),
    0xC000013A: ("CONTROL_C_EXIT", "Ctrl-C or console close"),
}


def describe_exit_code(code: int) -> str:
    """Render a child's exit code with its NT status name where there is one."""
    if code == 0:
        return "0 (clean exit)"
    status = code & 0xFFFFFFFF
    known = _NT_STATUS.get(status)
    # ASCII only in anything printed: the Windows console is cp1252, so an
    # em dash here comes out as mojibake in the middle of a crash report.
    if known:
        return f"{code} (0x{status:08X} {known[0]}: {known[1]})"
    if status >= 0xC0000000:
        return f"{code} (0x{status:08X}: NT status; the OS killed the process)"
    return str(code)


def _carla_crash_dirs():
    """Unreal crash-report directories for the CARLA server, if they exist."""
    local = os.environ.get("LOCALAPPDATA")
    if not local:
        return []
    return [p for p in (Path(local) / game / "Saved" / "Crashes"
                        for game in ("CarlaUnreal", "CarlaUE4"))
            if p.is_dir()]


def find_carla_crash(since: float):
    """Newest CARLA server crash report written after ``since`` (epoch seconds).

    This exists because the failure mode is counter-intuitive: when the CARLA
    server dies, its Python clients die 2-13 s LATER with a native exit code
    that names the CLIENT. Read literally, "[CRASH] NPC_TRAFFIC exited with
    code 3221226505" sends you to debug NPC_TRAFFIC, which was a victim. The
    server's own crash report is the actual evidence, so surface it here.
    """
    newest = None
    for d in _carla_crash_dirs():
        try:
            entries = list(d.iterdir())
        except OSError:
            continue
        for sub in entries:
            try:
                if not sub.is_dir():
                    continue
                mtime = sub.stat().st_mtime
            except OSError:
                continue
            if mtime >= since and (newest is None or mtime > newest[0]):
                newest = (mtime, sub)
    if newest is None:
        return None

    mtime, sub = newest
    info = {"path": sub, "when": mtime, "error": "", "thread": "", "uptime": ""}
    try:
        raw = (sub / "CrashContext.runtime-xml").read_text(
            encoding="utf-8", errors="replace")
    except OSError:
        return info

    def tag(name):
        m = re.search(rf"<{name}>(.*?)</{name}>", raw, re.S)
        return html.unescape(m.group(1)).strip() if m else ""

    info["error"] = tag("ErrorMessage")
    info["uptime"] = tag("SecondsSinceStart")
    for thread in re.findall(r"<Thread>(.*?)</Thread>", raw, re.S):
        if "<IsCrashed>true</IsCrashed>" in thread:
            m = re.search(r"<ThreadName>(.*?)</ThreadName>", thread, re.S)
            info["thread"] = html.unescape(m.group(1)).strip() if m else ""
            break
    return info


def report_probable_cause(name: str, code: int, since: float) -> None:
    """Print the likely root cause of a child's death. Never raises."""
    try:
        crash = find_carla_crash(since)
        if crash is None:
            # The crash reporter can take a few seconds to write the report,
            # and the client dies first — so one short re-check is worth it.
            time.sleep(3.0)
            crash = find_carla_crash(since)

        if crash is not None:
            gap = time.time() - crash["when"]
            print(f"[CAUSE] The CARLA SERVER crashed ~{gap:.0f}s ago. "
                  f"{name} is a client and almost")
            print(f"        certainly died downstream of it: debug CARLA, not {name}.")
            if crash["error"]:
                print(f"        {crash['error']}")
            detail = []
            if crash["thread"]:
                detail.append(f"thread {crash['thread']}")
            if crash["uptime"]:
                try:
                    detail.append(f"CARLA uptime {int(crash['uptime']) // 60} min")
                except ValueError:
                    pass
            if detail:
                print(f"        ({', '.join(detail)})")
            print(f"        report: {crash['path']}")
            return

        if (code & 0xFFFFFFFF) >= 0xC0000000:
            print("[CAUSE] No CARLA crash report newer than this session, so the "
                  "fault looks local to this process.")
            if name == "PROVOICE":
                fault_log = Path.cwd() / "logs" / "provoice_faults.log"
                if fault_log.exists():
                    print(f"        ProVoice runs faulthandler: {fault_log}")
                    print("        holds a Python stack for EVERY thread as of the fault.")
                # Measured 2026-07-28, twice consecutively with a clean control
                # run in between: after a ProVoice crash, the NEXT ProVoice
                # connection killed the CARLA server within ~10-12 s.
                #   16:44 ProVoice crash -> 16:51 ProVoice start -> CARLA dead +12s
                #   17:09 ProVoice crash -> 17:12 ProVoice start -> CARLA dead +10s
                # The first ProVoice run against a freshly launched CARLA was
                # clean, --test-drive runs never trip it, and an idle CARLA
                # survives indefinitely -- so it is orphaned client state that
                # only the next connection touches, not decay over time.
                #
                # There is a SECOND route into the same CARLA fault that this
                # advice does not cover. Of the nine recorded 0x1b8 crashes,
                # four had no ProVoice crash anywhere in that CARLA session
                # (2026-05-27, 07-25 13:16, 07-25 13:33, 07-27 18:44) -- CARLA
                # dying during ordinary operation. All four predate the
                # set_hybrid_physics_mode(False) change of 2026-07-28 12:52 in
                # src/drive/fixed_npc_traffic.py, and none has recurred since,
                # so that change plausibly closed this route. Only two crashes
                # have been observed post-fix though, which under the previous
                # rate would come up clean by luck ~18% of the time -- suggestive,
                # not settled.
                #
                # The route below is the one still live, and its root cause is
                # ProVoice's own heap corruption, not CARLA -- see
                # --provoice-no-carla.
                print()
                print("  ***  RESTART CARLA BEFORE THE NEXT RUN.  ***")
                print("       After a ProVoice crash, the next ProVoice connection")
                print("       has killed the CARLA server within ~10-12s. Relaunching")
                print("       CARLA broke that chain every time it was tried.")
                print("       Not a guarantee: CARLA has also died on fresh sessions")
                print("       with no prior ProVoice crash, though every such case")
                print("       predates the 2026-07-28 traffic-manager fix.")
    except Exception as e:  # noqa: BLE001 — diagnostics must never mask the crash
        print(f"[CAUSE] (could not determine: {e})")



# =========================
# SESSION
# =========================

def write_session_id(root: Path) -> str:
    session = str(uuid.uuid4())
    path = root / ".session_id"
    tmp = root / ".session_id.tmp"

    tmp.write_text(session)
    tmp.replace(path)

    print(f"[SESSION] {session}")
    return session


# =========================
# COMMAND BUILDERS (FIXED)
# =========================

def build_drive_cmd(session, args):
    return [
        sys.executable,
        "-m",
        "src.drive.drive_improved",
        "--control", "test",
        "--session-id", session,
        "--participantid", args.participantid,
        "--environment", args.environment,
        "--secondary-task", args.secondary_task,
        "--functionname", args.functionname,
        "--modeltype", args.modeltype,
        "--state-model", args.state_model,
        "--w-fcd", str(args.w_fcd),
        "--host", args.host,
        "--port", str(args.port),
        *(["--fullscreen"] if args.fullscreen else ["--res", args.res]),
        *(["--no-popup"] if args.no_popup else []),
        *(["--test-popup"] if args.test_popup else []),
        *(["--fixed"] if args.fixed else []),
        # Omitted when neither flag was given, so Drive keeps its own default
        # (the wheel when one is bound, the keyboard otherwise).
        *([f"--{args.popup_input}-input"] if args.popup_input else []),
        # Drive owns the draw: it is the process that shows the popups, so the
        # pool lives there and only the switch is forwarded.
        *(["--random-function"] if args.random_function else []),
        "--popup-wait-timeout", str(args.popup_wait_timeout),
    ]


def build_provoice_cmd(session, args, vehicle_id, remote_url=None):
    """Command line for ProVoice.

    ``remote_url`` (--remote) is the one case where the result is PRINTED for a
    human to run on another machine rather than spawned here. It is built by
    this same function on purpose: a separately hand-written command line is
    how the two ends end up disagreeing about the session id, the participant
    or the model, and every one of those mismatches only shows up in the data.
    """
    return [
        sys.executable,
        "src/ProVoice/main.py",
        f"session_id={session}",
        f"vehicle_id={vehicle_id}",
        f"participantid={args.participantid}",
        f"environment={args.environment}",
        f"secondary_task={args.secondary_task}",
        # Under --random-function this is the fallback default, not what the
        # popups ask about: ProVoice takes one function for the whole run, and in
        # a --data-collection run it is only a tag on the raw rows. The function
        # per label lives in data/user_loa_labels.csv, written by Drive.
        f"functionname={args.functionname}",
        f"modeltype={args.modeltype}",
        f"state_model={args.state_model}",
        f"w_fcd={args.w_fcd}",
        *([f"window_seconds={args.window_seconds}"] if args.window_seconds is not None else []),
        f"decision_hz={args.decision_hz}",
        f"host={args.host}",
        f"port={args.port}",
        # DIAGNOSTIC (--provoice-no-carla): ProVoice's main() skips the direct
        # CARLA connection entirely when vehicle_state_url is set, expecting a
        # bridge to serve vehicle state over HTTP. Pointing it at a dead port
        # gives us a ProVoice that runs completely normally -- every perception
        # thread, every model, all logging -- while making ZERO CARLA calls.
        # _poll_vehicle_state swallows the connection errors and the vehicle
        # fields stay at their defaults, so this run's speed/steer/brake columns
        # are unusable. It exists to answer one question: is ProVoice's heap
        # corruption coming from the CARLA client?
        # Both paths below set vehicle_state_url, and setting it AT ALL is what
        # matters: main.py then skips get_carla_vehicle_by_id() entirely, so no
        # carla.Client is ever constructed, carla_vehicle stays None, and the
        # per-frame CARLA block in DataCollector._carla_process is skipped
        # wholesale. --vehicle-bridge points at a real server so the vehicle
        # columns are still populated; --provoice-no-carla points at a dead port
        # so they are not (diagnostic only).
        *([f"vehicle_state_file={args.bridge_file}",
           f"state_poll_hz={args.bridge_poll_hz}"]
          if args.vehicle_bridge else []),
        # --remote: the same "no carla.Client in this process" contract as the
        # file bridge, but the state crosses a network instead of a file, so
        # THIS process is the one on the other machine.
        *([f"vehicle_state_url={remote_url}",
           f"state_poll_hz={args.bridge_poll_hz}"]
          if remote_url else []),
        *([f"vehicle_state_url=http://127.0.0.1:{_DEAD_PORT}"]
          if args.provoice_no_carla and not args.vehicle_bridge else []),
        # Bare flags, not key=value: ProVoice parses these with store_true, which
        # rejects an explicit "=value". _normalize_argv passes "-"-prefixed
        # tokens through untouched.
        *(["--calibration-only"] if args.calibration_only else []),
        *(["--data-collection"] if args.data_collection else []),
    ]


# =========================
# PROCESS MANAGER
# =========================

class ProcessManager:
    # A supervised child that keeps dying is a symptom, not something to paper
    # over forever: past this many restarts inside the window we stop and say so
    # rather than spinning silently for a whole session.
    RESTART_WINDOW_S = 120.0
    MAX_RESTARTS_IN_WINDOW = 6

    def __init__(self):
        self.processes = []
        # name -> {"cmd", "below_normal", "restarts": [monotonic timestamps],
        #          "total", "gave_up"}
        self.supervised = {}

    def start(self, cmd, name, below_normal=False, restart=False,
              restart_delay=0.0):
        """Spawn a child process.

        ``below_normal`` drops it to BELOW_NORMAL priority on Windows. Used for
        PROVOICE: it shares the machine with the CARLA server and Drive, and its
        perception workers will otherwise take whatever cores the Windows
        scheduler hands them. Starving CARLA's render/physics threads or Drive's
        render loop shows up as dropped frames and input lag for the participant.
        Priority does NOT reduce ProVoice's work, it only settles who yields when
        both want the same core, so it complements the thread caps in
        src/ProVoice/main.py rather than replacing them. No-op off Windows.

        CARLA runs ASYNCHRONOUSLY here — do not assume otherwise. Every
        sim_world.tick() in drive_improved.py sits behind ``if args.sync:``,
        ``--sync`` is store_true (so it defaults to False), and build_drive_cmd()
        below never passes it. Drive therefore takes the wait_for_tick() branch,
        fixed_delta_seconds is never set, and the simulator advances on its own
        clock rather than on any client's tick. src/drive/fixed_npc_traffic.py
        matches this: SYNC_MODE=False, so its traffic manager is asynchronous too.
        Stated explicitly because a previous version of this docstring claimed
        synchronous mode, and that claim misled a debugging session into
        "fixing" a nonexistent sync mismatch, which froze all NPC traffic.
        """
        print(f"[START] {name}{' (below-normal priority)' if below_normal else ''}")
        print("        ", " ".join(cmd))

        kwargs = {}
        if os.name == "nt":
            # CREATE_NEW_PROCESS_GROUP is what makes a graceful stop possible:
            # stop_all() signals CTRL_BREAK_EVENT, which is delivered to an
            # entire process group, so each child needs its own or the launcher
            # would be signalled along with it.
            #
            # Trade-off to know about: a new process group also stops Ctrl-C in
            # this console from reaching the children. That is now handled
            # explicitly -- the KeyboardInterrupt path runs stop_all(), which
            # signals each child -- so the shutdown is more deterministic than
            # relying on console-wide Ctrl-C delivery, not less.
            flags = subprocess.CREATE_NEW_PROCESS_GROUP
            if below_normal:
                flags |= subprocess.BELOW_NORMAL_PRIORITY_CLASS
            kwargs["creationflags"] = flags

        p = subprocess.Popen(
            cmd,
            text=True,
            **kwargs,
        )

        self.processes.append((name, p))
        if restart:
            self.supervised.setdefault(name, {
                "cmd": cmd, "below_normal": below_normal,
                "restarts": [], "total": 0, "gave_up": False,
                "delay": restart_delay,
            })
        return p

    def restart_supervised(self, name):
        """Relaunch a supervised child that exited. Returns True if relaunched.

        Used for the vehicle-state bridge and the --remote HTTP server: those
        are the processes whose death should not end the session. They hold
        nothing but a CARLA client and a publisher, so a fresh one recovers the
        vehicle feed within a second, and the alternative -- tearing down a live
        participant run because a read-only side channel died -- is far worse.
        """
        info = self.supervised.get(name)
        if info is None or info["gave_up"]:
            return False

        now = time.monotonic()
        info["restarts"] = [t for t in info["restarts"]
                            if now - t < self.RESTART_WINDOW_S]
        if len(info["restarts"]) >= self.MAX_RESTARTS_IN_WINDOW:
            info["gave_up"] = True
            print(f"[GIVE UP] {name} died {len(info['restarts'])} times in "
                  f"{self.RESTART_WINDOW_S:.0f}s: not restarting again.")
            print(f"          The session continues WITHOUT it; ProVoice will "
                  f"hold its last known vehicle state.")
            return False

        info["restarts"].append(now)
        info["total"] += 1
        # Drop the dead entry so the main loop does not keep re-reporting it.
        self.processes = [(n, p) for n, p in self.processes if n != name]
        delay = info.get("delay", 0.0)
        if delay:
            # ProVoice owns the camera. The OS releases the device when the
            # process dies, but not instantly, and a replacement that grabs it
            # too early gets a failed VideoCapture and runs blind for the rest
            # of the session.
            print(f"[RESTART] {name} in {delay:.0f}s (letting the camera and "
                  f"file handles be released)...")
            time.sleep(delay)
        print(f"[RESTART] {name} (attempt {info['total']})")
        self.start(info["cmd"], name, below_normal=info["below_normal"],
                   restart=True, restart_delay=delay)
        return True

    @staticmethod
    def _ask_to_stop(p) -> None:
        """Request a GRACEFUL exit, so the child runs its own cleanup.

        This used to be a bare p.terminate(). On Windows that is an alias for
        kill(): it calls TerminateProcess, which delivers NO signal and runs no
        atexit handler, no `finally` block and no destructor. ProVoice's
        SIGINT/SIGTERM handlers (src/ProVoice/main.py) therefore never ran once,
        and fixed_npc_traffic.py's KeyboardInterrupt cleanup never ran either.

        Why that matters beyond tidiness: ProVoice holds a CARLA RPC connection.
        Killed that way, the connection is severed exactly as if the process had
        crashed -- so CARLA cannot tell a normal end-of-run from a crash, and
        the observed "first run fine, second run dies" pattern follows. Sending
        CTRL_BREAK lets the child close that connection itself.

        CTRL_BREAK_EVENT goes to a whole process group, which is why start()
        spawns each child with CREATE_NEW_PROCESS_GROUP -- without it the signal
        would hit this launcher too.
        """
        try:
            if os.name == "nt":
                p.send_signal(signal.CTRL_BREAK_EVENT)
            else:
                p.terminate()
        except Exception as e:  # noqa: BLE001 — already-dead child, bad pid, ...
            print(f"        (graceful stop failed, will force: {e})")

    def stop_all(self, grace: float = 8.0):
        print("[CLEANUP] stopping processes...")

        alive = [(n, p) for n, p in self.processes if p.poll() is None]
        for name, p in alive:
            print(f"[STOP] {name} (asking to exit)")
            self._ask_to_stop(p)

        # Give them a bounded window to shut down on their own. ProVoice has the
        # most to do (stop the collector, flush and close the logger, drop the
        # CARLA client), so the budget is shared rather than per-process.
        deadline = time.monotonic() + grace
        for name, p in alive:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            try:
                p.wait(timeout=remaining)
            except subprocess.TimeoutExpired:
                pass

        for name, p in alive:
            if p.poll() is None:
                print(f"[KILL] {name} (no clean exit within {grace:.0f}s)")
                p.kill()
                # kill() only REQUESTS termination — TerminateProcess returns
                # before the process is gone. Wait so this method's
                # postcondition is "the children are dead", not "we asked".
                try:
                    p.wait(timeout=5.0)
                except subprocess.TimeoutExpired:
                    print(f"        WARNING: {name} (pid {p.pid}) still alive "
                          f"after kill; it may hold the camera or CARLA port.")
            else:
                print(f"[STOPPED] {name} exited cleanly")


# =========================
# WAIT HELPERS (IMPORTANT FOR CARLA)
# =========================

def wait_for_carla_ready():
    print("[WAIT] CARLA warmup...")
    time.sleep(5)


def clear_vehicle_id(root: Path):
    """Delete any vehicle_id.txt left behind by a previous run.

    drive_improved.py writes the file but never removes it on shutdown, so
    without this wait_for_vehicle_id() would instantly return the *previous*
    session's id and ProVoice would attach to a vehicle that no longer exists.
    """
    for name in ("vehicle_id.txt", "vehicle_id.txt.tmp"):
        path = root / name
        try:
            path.unlink()
            print(f"[CLEANUP] removed stale {name}")
        except FileNotFoundError:
            pass
        except OSError as e:
            print(f"[WARN] could not remove {name}: {e}")


def wait_for_vehicle_id(root: Path, drive_proc, timeout, poll=0.2):
    """Block until Drive publishes the spawned vehicle's id, and return it.

    drive_improved.py writes the file atomically (tmp + os.replace) only AFTER
    the world tick, so its appearance is the signal that Drive is fully
    initialised and ProVoice can safely connect — this replaces the fixed sleep
    that merely assumed the write had happened by then.
    """
    path = root / "vehicle_id.txt"
    print(f"[WAIT] vehicle id at {path} (timeout {timeout:.0f}s)...")

    started = time.monotonic()
    deadline = started + timeout
    warned = None  # last bad content reported, so a partial write warns once

    while time.monotonic() < deadline:
        # Drive dying before it spawns is a hard failure: waiting out the full
        # timeout would only delay an error we can already report.
        code = drive_proc.poll()
        if code is not None:
            raise RuntimeError(f"DRIVE exited with code {code} before writing vehicle_id.txt")

        try:
            raw = path.read_text().strip()
        except FileNotFoundError:
            raw = ""
        except OSError:
            raw = ""  # sharing violation mid-os.replace on Windows; retry

        if raw:
            try:
                vehicle_id = int(raw)
            except ValueError:
                if raw != warned:
                    warned = raw
                    print(f"[WARN] invalid vehicle id content {raw!r}, waiting...")
            else:
                print(f"[WAIT] vehicle id {vehicle_id} ready after "
                      f"{time.monotonic() - started:.1f}s")
                return vehicle_id

        time.sleep(poll)

    raise TimeoutError(f"vehicle id not written within {timeout:.0f}s")


# =========================
# MAIN
# =========================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--participantid", default="001")
    parser.add_argument("--environment", default="city")
    parser.add_argument("--secondary-task", default="none")
    # Default deferred to DEFAULT_FUNCTIONNAME below: left as None here so
    # "--functionname was given" can be told apart from "nobody said", which is
    # what makes the --random-function conflict detectable.
    parser.add_argument("--functionname", default=None,
                        help="Function the LoA popups ask about, for the whole run "
                             "(default: %s). Mutually exclusive with "
                             "--random-function." % DEFAULT_FUNCTIONNAME)
    parser.add_argument("--random-function", dest="random_function", action="store_true",
                        help="Draw the function each popup asks about at random from "
                             "the five study functions instead of fixing one with "
                             "--functionname. Every 20 s window then holds TWO prompts "
                             "about two DIFFERENT functions, doubling the labels per "
                             "window. The pool lives in src/drive/drive_improved.py "
                             "(RANDOM_FUNCTION_POOL).")
    parser.add_argument("--modeltype", default="combined")
    parser.add_argument("--state-model", default="xlstm")
    parser.add_argument("--w-fcd", type=float, default=0.7)
    parser.add_argument("--window-seconds", type=float, default=None,
                        help="Time span (s) of the driver-state window fed to the xLSTM. "
                             "Unset = inherit the window the checkpoint was trained with. "
                             "0 disables the time cap.")
    parser.add_argument("--decision-hz", type=float, default=4.0,
                        help="Rate of the decision thread, decoupled from data collection. "
                             "Sets the decisions.csv row rate; capped by the achieved "
                             "collection rate (one decision per distinct frame).")
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--fullscreen", action="store_true",
                        help="Run the drive window fullscreen at the desktop resolution.")
    parser.add_argument("--res", default="1280x720",
                        help="Drive window resolution, e.g. 2560x1440. Ignored with --fullscreen.")
    parser.add_argument("--calibration-only", dest="calibration_only", action="store_true",
                        help="Run ProVoice's 180 s calibration, store the baseline for "
                             "this participant, then shut the whole experiment down. "
                             "Always measures fresh, ignoring any stored calibration.")
    parser.add_argument("--data-collection", dest="data_collection", action="store_true",
                        help="Data-collection run: ProVoice records raw data only, with "
                             "no decision engine and no interventions. The live "
                             "calibration is skipped (the participant's stored baseline "
                             "is reused, or neutral defaults if there is none).")
    parser.add_argument("--vehicle-bridge", dest="vehicle_bridge",
                        action="store_true",
                        help="Run scripts/vehicle_state_file_bridge.py in its own "
                             "process and have ProVoice read vehicle state from "
                             "the file it publishes, instead of holding its own "
                             "CARLA client. ProVoice reliably corrupts its heap "
                             "when it polls CARLA directly; this keeps libcarla "
                             "out of that process while preserving the vehicle "
                             "columns. The bridge is supervised and restarted if "
                             "it dies, so its failure costs a short gap in "
                             "vehicle state rather than the session. Uses a FILE, "
                             "not HTTP: the earlier socket version generated ~20 "
                             "TCP connections/second, the kernel path suspected "
                             "in the 2026-07-28 machine bugchecks.")
    parser.add_argument("--bridge-file", dest="bridge_file",
                        default="vehicle_state.json",
                        help="File the vehicle-state bridge publishes into and "
                             "ProVoice reads. Written atomically, so a reader "
                             "never sees a partial record.")
    parser.add_argument("--bridge-poll-hz", dest="bridge_poll_hz", type=float,
                        default=20.0,
                        help="How often ProVoice polls the vehicle state, for "
                             "--vehicle-bridge and for the command --remote "
                             "prints. Defaults to "
                             "20 Hz to match the collection loop; ProVoice's own "
                             "built-in default is 2 Hz, which would sample "
                             "steer/brake ten times slower than a direct CARLA "
                             "read and visibly degrade those features.")
    parser.add_argument("--remote", action="store_true",
                        help="TWO-MACHINE setup. This machine runs CARLA, the "
                             "NPC traffic and Drive with the LoA popups, and "
                             "publishes vehicle state over HTTP from "
                             "scripts/vehicle_state_server.py. ProVoice is NOT "
                             "started here: it runs on the other machine, which "
                             "polls that URL and holds no CARLA client (the same "
                             "contract as --vehicle-bridge, over a network "
                             "instead of a file). The launcher prints the exact "
                             "ProVoice command line to paste there, with this "
                             "run's session id already filled in. The server is "
                             "supervised and restarted if it dies. NOTE: the two "
                             "halves log to their own machines — the LoA labels "
                             "land here in data/user_loa_labels.csv, ProVoice's "
                             "raw_data.jsonl lands there — so the files have to "
                             "be brought together before the dataset is built.")
    parser.add_argument("--remote-port", dest="remote_port", type=int, default=8080,
                        help="Port the vehicle-state server listens on (--remote).")
    parser.add_argument("--remote-bind", dest="remote_bind", default="0.0.0.0",
                        help="Interface the vehicle-state server binds (--remote). "
                             "0.0.0.0 accepts from the LAN, which is the point; "
                             "127.0.0.1 would only serve a tunnel running here.")
    parser.add_argument("--remote-sample-hz", dest="remote_sample_hz", type=float,
                        default=20.0,
                        help="Rate at which the vehicle-state server samples CARLA "
                             "(--remote). Independent of how often ProVoice polls: "
                             "requests are served from the last sample, so CARLA's "
                             "load no longer depends on the client.")
    parser.add_argument("--restart-provoice", dest="restart_provoice",
                        action="store_true",
                        help="Relaunch ProVoice if it crashes instead of ending "
                             "the session. Salvages a participant run at the "
                             "cost of a ~30-60s hole in raw_data.jsonl and "
                             "decisions.csv while the models reload; Drive keeps "
                             "running so the LoA labels are unbroken. STRONGLY "
                             "recommended only with --vehicle-bridge: without it "
                             "the new ProVoice reconnects to CARLA, and a "
                             "reconnect after a crash has killed the server "
                             "within ~10-12s every time it was observed.")
    parser.add_argument("--bridge-zeros", dest="bridge_zeros", action="store_true",
                        help="DIAGNOSTIC, use with --vehicle-bridge: the bridge "
                             "still reads CARLA at the normal rate but publishes "
                             "zeros. Isolates the ONE variable that separates a "
                             "crashing --vehicle-bridge run from a clean "
                             "--provoice-no-carla run: the numbers ProVoice "
                             "receives. Crashes anyway -> the values are "
                             "irrelevant; stays clean -> they are implicated. "
                             "NOT usable participant data.")
    parser.add_argument("--provoice-no-carla", dest="provoice_no_carla",
                        action="store_true",
                        help="DIAGNOSTIC: run ProVoice with NO CARLA connection "
                             "(everything else normal) to test whether the CARLA "
                             "client is the source of its heap corruption. The "
                             "vehicle-state columns (speed, steer, brake, ...) "
                             "stay at their defaults, so the run is NOT usable as "
                             "participant data.")
    parser.add_argument("--test-drive", dest="test_drive", action="store_true",
                        help="Test drive: launch NPC traffic and Drive only, without "
                             "the ProVoice proactive-assistance process. No "
                             "interventions, no decisions.csv, no driver-state model.")
    parser.add_argument("--fixed", action="store_true",
                        help="Always spawn the ego at the same map spawn point instead "
                             "of a random one. For calibration runs, which have to start "
                             "from an identical position.")
    parser.add_argument("--test-popup", dest="test_popup", action="store_true",
                        help="Teaching mode: open the simulator UI and show LoA "
                             "selection popups straight away, so the participant can "
                             "practise the control. No traffic, no ProVoice, and the "
                             "practice selections are not logged.")
    parser.add_argument("--no-popup", dest="no_popup", action="store_true",
                        help="Suppress Drive's LoA selection popups for the whole "
                             "session: no scene freezes and no user labels. ProVoice "
                             "still runs and logs its own decisions.")
    popup_input = parser.add_mutually_exclusive_group()
    popup_input.add_argument("--wheel-input", dest="popup_input", action="store_const",
                             const="wheel",
                             help="Answer the LoA popups from the steering wheel: the "
                                  "paddles move the cursor, the front button ticks the "
                                  "level under it, the CONFIRM row submits. Default "
                                  "when a wheel is attached.")
    popup_input.add_argument("--keyboard-input", dest="popup_input", action="store_const",
                             const="keyboard",
                             help="Answer the LoA popups from the keyboard: number keys "
                                  "0-4 tick a level, the same number again unticks it, "
                                  "ENTER confirms. The popup ignores the wheel, which "
                                  "still steers. Default when no wheel is attached.")
    parser.set_defaults(popup_input=None)
    parser.add_argument("--popup-wait-timeout", dest="popup_wait_timeout",
                        type=float, default=None,
                        help="Seconds Drive waits for ProVoice's first logged frame "
                             "before opening the first LoA window anyway. ProVoice "
                             "needs the better part of a minute to load its models, "
                             "and a window opened before it logs has no driver-state "
                             "data behind it, so its labels are dropped when the "
                             "dataset is built. Driving starts immediately either "
                             "way; only the windows wait. 0 disables the wait. "
                             "Default: %.0f, or 0 with --remote, where ProVoice "
                             "logs on the other machine and the wait could only "
                             "ever time out." % DEFAULT_POPUP_WAIT_TIMEOUT)
    parser.add_argument("--vehicle-id-timeout", type=float, default=120.0,
                        help="How long to wait for Drive to spawn the vehicle and write "
                             "vehicle_id.txt before giving up.")

    args = parser.parse_args()

    if args.calibration_only and args.data_collection:
        parser.error("--calibration-only and --data-collection are mutually exclusive.")
    if args.calibration_only and args.test_drive:
        parser.error("--calibration-only needs ProVoice, which --test-drive disables.")
    if args.restart_provoice and args.calibration_only:
        parser.error("--restart-provoice and --calibration-only are "
                     "incompatible: a baseline assembled from two partial "
                     "180 s measurements either side of a crash is not a "
                     "baseline. Re-run the calibration instead.")
    if args.restart_provoice and (args.test_drive or args.test_popup or args.remote):
        parser.error("--restart-provoice has nothing to restart: %s does not "
                     "start ProVoice." % ("--test-drive" if args.test_drive
                                          else "--test-popup" if args.test_popup
                                          else "--remote"))
    if args.remote and args.vehicle_bridge:
        parser.error("--remote and --vehicle-bridge both publish the vehicle "
                     "state, for different readers: the bridge feeds a ProVoice "
                     "on THIS machine, which --remote does not start. Pick the "
                     "one that matches where ProVoice runs.")
    if args.remote and args.provoice_no_carla:
        parser.error("--provoice-no-carla only affects a ProVoice started here, "
                     "and --remote starts none. Pass the diagnostic on the "
                     "ProVoice machine instead.")
    if args.remote and (args.test_drive or args.test_popup):
        parser.error("--remote and %s contradict each other: %s exists to run "
                     "without ProVoice at all, --remote to run it elsewhere."
                     % (("--test-drive",) * 2 if args.test_drive
                        else ("--test-popup",) * 2))
    if args.restart_provoice and not args.vehicle_bridge:
        # Allowed, because the alternative is losing the session outright, but
        # the odds are poor and the user should not discover that afterwards.
        print("[WARN] --restart-provoice WITHOUT --vehicle-bridge: the relaunched "
              "ProVoice will open its own CARLA connection, and a reconnect "
              "after a ProVoice crash has killed the CARLA server within "
              "~10-12s in every observed case. Add --vehicle-bridge so the "
              "restart never touches CARLA.")
    if args.bridge_zeros and not args.vehicle_bridge:
        parser.error("--bridge-zeros only affects the vehicle bridge; add "
                     "--vehicle-bridge (without it, nothing publishes at all, "
                     "which is what --provoice-no-carla already gives you).")
    if args.bridge_zeros:
        print("[WARN] --bridge-zeros: the bridge will publish ZEROS. Vehicle "
              "columns will be inert, so THIS RUN IS NOT USABLE PARTICIPANT "
              "DATA. It exists to isolate whether real vehicle values are what "
              "distinguishes a crashing run from a clean one.")
    if args.vehicle_bridge and args.provoice_no_carla:
        parser.error("--vehicle-bridge and --provoice-no-carla contradict each "
                     "other: both keep CARLA out of ProVoice, but the bridge "
                     "supplies real vehicle state while --provoice-no-carla "
                     "deliberately supplies none. Pick one.")
    if args.vehicle_bridge and (args.test_drive or args.test_popup):
        parser.error("--vehicle-bridge only feeds ProVoice, which %s does not "
                     "start." % ("--test-drive" if args.test_drive else "--test-popup"))
    if args.provoice_no_carla and (args.test_drive or args.test_popup):
        parser.error("--provoice-no-carla only affects ProVoice, which %s does not "
                     "start." % ("--test-drive" if args.test_drive else "--test-popup"))
    if args.provoice_no_carla:
        # Loud, because the run looks completely normal and silently produces
        # rows whose vehicle columns are all defaults.
        print("[WARN] --provoice-no-carla: ProVoice will make NO CARLA calls. "
              "speed/steer/brake/junction and every other vehicle field stay at "
              "their defaults, so THIS RUN IS NOT USABLE PARTICIPANT DATA. It is "
              "a diagnostic for the heap corruption only.")
    if args.test_popup and args.no_popup:
        parser.error("--test-popup and --no-popup contradict each other: one is "
                     "nothing but popups, the other suppresses them.")
    if args.test_popup and args.calibration_only:
        parser.error("--test-popup and --calibration-only are separate modes: the "
                     "first teaches the control, the second records a baseline.")
    if args.no_popup and not (args.calibration_only or args.test_drive):
        # A normal session exists to collect the labels, so suppressing the
        # popups there would produce a run with nothing to show for it.
        parser.error("--no-popup requires --calibration-only or --test-drive: a normal "
                     "session has to collect the LoA labels.")
    if args.calibration_only and not args.no_popup:
        # The popup freezes the scene for as long as the driver deliberates, and
        # a baseline recorded while they stare at a menu is not a driving
        # baseline. Worth saying out loud rather than silently forcing.
        print("[WARN] --calibration-only without --no-popup: the LoA popups will "
              "interrupt the 180 s baseline. Add --no-popup unless that is intended.")
    if args.popup_input and args.no_popup:
        # Harmless, but it means the run was configured for an interface it will
        # never show — worth saying rather than silently ignoring the flag.
        print("[WARN] --%s-input has no effect with --no-popup: this run shows no "
              "LoA popups." % args.popup_input)
    if args.random_function and args.functionname is not None:
        parser.error("--random-function and --functionname contradict each other: "
                     "one draws the function per popup, the other fixes it for the "
                     "whole run.")
    if args.random_function and args.no_popup:
        print("[WARN] --random-function has no effect with --no-popup: the function "
              "is only ever asked about in a popup.")
    if args.random_function and not (args.data_collection or args.test_drive
                                     or args.test_popup or args.calibration_only):
        # ProVoice's decision engine is driven by ONE functionname for the whole
        # run (its FCD vector comes from it), and there is no channel for Drive to
        # tell it which function a given popup drew. The user labels stay correct
        # either way, but system_* and user rows would then describe different
        # functions, which is exactly the comparison the analysis makes.
        print("[WARN] --random-function with a live ProVoice decision engine: its "
              "decisions all assume functionname='%s', while the popups ask about "
              "whichever function they drew. Intended for --data-collection runs."
              % (args.functionname or DEFAULT_FUNCTIONNAME))

    # Resolve the default only after the conflict check above, which needs to see
    # whether --functionname was actually given.
    if args.functionname is None:
        args.functionname = DEFAULT_FUNCTIONNAME

    # Drive decides ProVoice is up by watching data/raw_data.jsonl for a row
    # tagged with this session id. Under --remote that file is written on the
    # OTHER machine, so the watcher can only ever run out its timeout — three
    # silent minutes with the participant driving and no windows opening. Skip
    # the wait by default there, and if a value was asked for explicitly, say
    # what it will actually do rather than quietly overriding it.
    if args.popup_wait_timeout is None:
        args.popup_wait_timeout = 0.0 if args.remote else DEFAULT_POPUP_WAIT_TIMEOUT
    elif args.remote and args.popup_wait_timeout > 0:
        print("[WARN] --popup-wait-timeout %.0f with --remote: Drive watches a "
              "LOCAL data/raw_data.jsonl that the remote ProVoice never writes, "
              "so the first LoA window will simply open %.0f s late. Start "
              "ProVoice on the other machine BEFORE driving instead."
              % (args.popup_wait_timeout, args.popup_wait_timeout))
    if args.remote and args.no_popup:
        print("[WARN] --remote with --no-popup: this machine's only job besides "
              "serving vehicle state is collecting the LoA labels, and there "
              "will be none.")
    if args.remote and args.calibration_only:
        # Locally this launcher watches for ProVoice's clean exit and shuts the
        # session down; with ProVoice on the other machine there is nothing to
        # watch, so say who ends the run.
        print("[WARN] --remote --calibration-only: the remote ProVoice exits by "
              "itself once the baseline is stored, but this launcher cannot see "
              "that happen. Stop it with Ctrl-C when the other machine says it "
              "is done.")

    root = Path.cwd()

    # Cut-off for "did CARLA die during THIS session?". Taken before anything is
    # launched, so crash reports left by earlier sessions (CARLA is normally
    # left running across many runs) are never mistaken for this one's.
    session_start = time.time()

    session = write_session_id(root)

    pm = ProcessManager()

    try:
        # =========================
        # START CARLA FIRST (IMPORTANT)
        # =========================
        # if CARLA has been started，removce this line
        # pm.start(["./CarlaUnreal.sh"], "CARLA")

        wait_for_carla_ready()

        # =========================
        # START NPC / TRAFFIC
        # =========================
        if args.test_popup:
            # Teaching the popup control needs the simulator window and nothing
            # else; the scene is frozen whenever a popup is open anyway.
            print("[SKIP] NPC_TRAFFIC not started (--test-popup)")
        else:
            pm.start(
                [sys.executable, "-m", "src.drive.fixed_npc_traffic", "--host", args.host, "--port", str(args.port)],
                "NPC_TRAFFIC"
            )

            time.sleep(6)

        # =========================
        # START DRIVE
        # =========================
        # Clear first: Drive must publish a FRESH id, and a leftover file from
        # the last run would satisfy the wait below immediately.
        clear_vehicle_id(root)

        drive_cmd = build_drive_cmd(session, args)
        drive_proc = pm.start(drive_cmd, "DRIVE")

        # Wait for the vehicle to actually exist instead of guessing at a sleep.
        vehicle_id = wait_for_vehicle_id(root, drive_proc, args.vehicle_id_timeout)

        # =========================
        # START VEHICLE-STATE BRIDGE
        # =========================
        # Started here, after the id exists and before ProVoice, so the feed is
        # already serving when ProVoice's poller makes its first request. The id
        # is passed explicitly rather than letting the bridge read
        # vehicle_id.txt: it has already been read and validated above, and a
        # restart mid-session must not race a rewrite of that file.
        if args.vehicle_bridge and not (args.test_drive or args.test_popup):
            # Remove any file left by a previous session BEFORE either process
            # opens it. Without this, ProVoice's first polls would read the last
            # run's vehicle state; the record's "ts" makes that detectable as
            # stale, but not reading it at all is better than detecting it.
            # Must happen here: once ProVoice holds the file open, Windows will
            # refuse the delete.
            try:
                Path(args.bridge_file).unlink()
                print(f"[CLEANUP] removed stale {args.bridge_file}")
            except FileNotFoundError:
                pass
            except OSError as e:
                print(f"[WARN] could not remove {args.bridge_file}: {e}")

            bridge_cmd = [
                sys.executable,
                "scripts/vehicle_state_file_bridge.py",
                "--out", args.bridge_file,
                "--hz", str(args.bridge_poll_hz),
                "--carla-host", args.host,
                "--carla-port", str(args.port),
                "--vehicle-id", str(vehicle_id),
                *(["--zeros"] if args.bridge_zeros else []),
            ]
            pm.start(bridge_cmd, "VEHICLE_BRIDGE", restart=True)
            # The bridge connects to CARLA and caches the map before it binds;
            # a short head start keeps ProVoice's first polls from all failing.
            time.sleep(2)

        # =========================
        # START THE REMOTE VEHICLE-STATE SERVER
        # =========================
        # Same role as the file bridge — one process owning the CARLA client so
        # ProVoice never holds one — but the reader is on another machine, so it
        # is reached over HTTP instead of a file. Supervised for the same
        # reason: it carries a read-only side channel, and its death must cost a
        # gap in the vehicle columns rather than the participant's session.
        if args.remote:
            server_cmd = [
                sys.executable,
                "scripts/vehicle_state_server.py",
                "--bind", args.remote_bind,
                "--port", str(args.remote_port),
                "--hz", str(args.remote_sample_hz),
                "--carla-host", args.host,
                "--carla-port", str(args.port),
                "--vehicle-id", str(vehicle_id),
            ]
            pm.start(server_cmd, "VEHICLE_SERVER", restart=True)
            # It connects to CARLA and caches the map before the first sample
            # is available; a head start keeps the remote poller from opening
            # onto a run of 503s.
            time.sleep(2)

        # =========================
        # START PROVOICE
        # =========================
        if args.remote:
            remote_url = f"http://{outbound_ip()}:{args.remote_port}"
            remote_cmd = build_provoice_cmd(session, args, vehicle_id,
                                            remote_url=remote_url)
            # "uv run python", not this machine's interpreter path, which means
            # nothing on the other machine.
            remote_cmd[0:1] = ["uv", "run", "python"]
            print()
            print("=" * 72)
            print(f"[REMOTE] vehicle state is served at {remote_url}/")
            print(f"[REMOTE] health/keep-alive counters: {remote_url}/health")
            print("[REMOTE] ProVoice is NOT started here. On the OTHER machine, "
                  "in the project root, run:")
            print()
            print("    " + shell_quote(remote_cmd))
            print()
            print("[REMOTE] (drop the 'uv run' if that machine's venv is already "
                  "activated)")
            print("[REMOTE] The session id above must match; it is what ties "
                  "this machine's LoA labels to that machine's raw_data.jsonl.")
            print("[REMOTE] If the poll never connects, it is almost always the "
                  "Windows firewall on THIS machine: inbound TCP "
                  f"{args.remote_port} has to be allowed.")
            print("=" * 72)
            print()
        elif args.test_drive or args.test_popup:
            # Drive (+ traffic) only. The vehicle id was still waited for above:
            # it is the signal that Drive finished initialising, and nothing else
            # would catch a Drive that dies during startup.
            _why = "--test-popup" if args.test_popup else "--test-drive"
            print(f"[SKIP] PROVOICE not started ({_why}); "
                  f"vehicle id {vehicle_id} published anyway")
        else:
            # Pass the id explicitly so ProVoice skips its own file discovery
            # (read_vehicle_id) — the file has already been read and validated here.
            provoice_cmd = build_provoice_cmd(session, args, vehicle_id)
            # The same session_id is reused on a restart, so raw_data.jsonl and
            # decisions.csv stay one continuous session (both are opened in
            # append mode) with a visible timestamp gap rather than splitting
            # into two. Calibration is reloaded from the participant's stored
            # baseline, so a restart does not re-run the 180 s routine.
            pm.start(provoice_cmd, "PROVOICE", below_normal=True,
                     restart=args.restart_provoice, restart_delay=3.0)

        # =========================
        # MAIN LOOP (keep alive)
        # =========================
        print("[RUNNING] experiment started")

        while True:
            time.sleep(1)
            # Snapshot: restart_supervised() rewrites pm.processes.
            for name, p in list(pm.processes):
                code = p.poll()
                if code is None:
                    continue
                if name == "PROVOICE" and args.calibration_only and code == 0:
                    # Expected: ProVoice exits itself once the baseline is stored.
                    print("[DONE] calibration stored; stopping the experiment")
                    raise SystemExit(0)

                # A supervised side channel dying must not end a participant
                # session. Report it loudly -- the gap it leaves is real, the
                # vehicle fields freeze at their last values until it is back --
                # then bring it straight back up.
                if name in pm.supervised and not pm.supervised[name]["gave_up"]:
                    print(f"[DOWN] {name} exited with code "
                          f"{describe_exit_code(code)}")
                    if name == "VEHICLE_BRIDGE":
                        print("       Vehicle state is FROZEN at its last values "
                              "until the bridge is back; the run continues.")
                    elif name == "VEHICLE_SERVER":
                        print("       The remote ProVoice's polls will fail until "
                              "the server is back; it holds its last values and "
                              "keeps recording everything else. Its persistent "
                              "connection dies with the process, so it will "
                              "reconnect on the next poll.")
                    elif name == "PROVOICE":
                        # Worth stating plainly: this is a real hole in the
                        # participant's record, not a hiccup. Reloading torch,
                        # YOLO, MMRPhys and the emotion model takes tens of
                        # seconds, and nothing is logged to raw_data.jsonl or
                        # decisions.csv until it is back. Drive keeps running,
                        # so the LoA labels in user_loa_labels.csv are unbroken.
                        print("       NO ProVoice data is being recorded until it "
                              "is back (model reload takes ~30-60s).")
                        print("       Drive keeps running, so the LoA labels are "
                              "unaffected. The gap is visible as a jump in the "
                              "raw_data.jsonl timestamps.")
                    if pm.restart_supervised(name):
                        continue
                    # Fell through to gave_up: keep running without it.
                    continue

                print(f"[CRASH] {name} exited with code {describe_exit_code(code)}")
                report_probable_cause(name, code, session_start)
                raise SystemExit(1)  # or break, or restart it

    except KeyboardInterrupt:
        print("\n[EXIT] stopping experiment...")

    except (TimeoutError, RuntimeError) as e:
        # Startup never completed — report it plainly instead of a traceback.
        print(f"[FATAL] {e}")
        raise SystemExit(1)

    finally:
        pm.stop_all()
        print("[EXIT] experiment stopped")


if __name__ == "__main__":
    main()