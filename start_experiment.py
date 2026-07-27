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


EXPERIMENT SETUP:
0. Teaching the LoA control: --test-popup --fullscreen
1. Driver adaptation phase: --test-drive --no-popup --fullscreen
2. Calibration: --fixed --no-popup --calibration-only --fullscreen --participantid 
3. Inference: --data-collection --participantid --functionname --fullscreen
"""

import os
import sys
import uuid
import time
import subprocess
import argparse
from pathlib import Path


# =========================
# CONFIG
# =========================



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
    ]


def build_provoice_cmd(session, args, vehicle_id):
    return [
        sys.executable,
        "src/ProVoice/main.py",
        f"session_id={session}",
        f"vehicle_id={vehicle_id}",
        f"participantid={args.participantid}",
        f"environment={args.environment}",
        f"secondary_task={args.secondary_task}",
        f"functionname={args.functionname}",
        f"modeltype={args.modeltype}",
        f"state_model={args.state_model}",
        f"w_fcd={args.w_fcd}",
        *([f"window_seconds={args.window_seconds}"] if args.window_seconds is not None else []),
        f"decision_hz={args.decision_hz}",
        f"host={args.host}",
        f"port={args.port}",
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
    def __init__(self):
        self.processes = []

    def start(self, cmd, name, below_normal=False):
        """Spawn a child process.

        ``below_normal`` drops it to BELOW_NORMAL priority on Windows. Used for
        PROVOICE: CARLA runs in synchronous mode, so the simulation advances only
        when Drive calls world.tick() — whenever the Windows scheduler picks a
        ProVoice worker over Drive's or CARLA's threads, that shows up as sim lag.
        Priority does NOT reduce ProVoice's work, it only settles who yields when
        both want the same core, so it complements the thread caps in
        src/ProVoice/main.py rather than replacing them. No-op off Windows.
        """
        print(f"[START] {name}{' (below-normal priority)' if below_normal else ''}")
        print("        ", " ".join(cmd))

        kwargs = {}
        if below_normal and os.name == "nt":
            kwargs["creationflags"] = subprocess.BELOW_NORMAL_PRIORITY_CLASS

        p = subprocess.Popen(
            cmd,
            text=True,
            **kwargs,
        )

        self.processes.append((name, p))
        return p

    def stop_all(self):
        print("[CLEANUP] stopping processes...")

        for name, p in self.processes:
            print(f"[STOP] {name}")
            p.terminate()

        time.sleep(2)

        for name, p in self.processes:
            if p.poll() is None:
                print(f"[KILL] {name}")
                p.kill()


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
    parser.add_argument("--functionname", default="Adjust seat positioning")
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
    parser.add_argument("--vehicle-id-timeout", type=float, default=120.0,
                        help="How long to wait for Drive to spawn the vehicle and write "
                             "vehicle_id.txt before giving up.")

    args = parser.parse_args()

    if args.calibration_only and args.data_collection:
        parser.error("--calibration-only and --data-collection are mutually exclusive.")
    if args.calibration_only and args.test_drive:
        parser.error("--calibration-only needs ProVoice, which --test-drive disables.")
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

    root = Path.cwd()

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
        # START PROVOICE
        # =========================
        if args.test_drive or args.test_popup:
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
            pm.start(provoice_cmd, "PROVOICE", below_normal=True)

        # =========================
        # MAIN LOOP (keep alive)
        # =========================
        print("[RUNNING] experiment started")

        while True:
            time.sleep(1)
            for name, p in pm.processes:
                code = p.poll()
                if code is None:
                    continue
                if name == "PROVOICE" and args.calibration_only and code == 0:
                    # Expected: ProVoice exits itself once the baseline is stored.
                    print("[DONE] calibration stored; stopping the experiment")
                    raise SystemExit(0)
                print(f"[CRASH] {name} exited with code {code}")
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