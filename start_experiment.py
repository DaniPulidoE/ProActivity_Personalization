#!/usr/bin/env python3
"""
Experiment launcher (CARLA + Drive + ProVoice)
Improved version: process-controlled instead of terminal-based
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
    ]


def build_provoice_cmd(session, args):
    return [
        sys.executable,
        "-m",
        "provoice",
        f"session_id={session}",
        f"participantid={args.participantid}",
        f"environment={args.environment}",
        f"secondary_task={args.secondary_task}",
        f"functionname={args.functionname}",
        f"modeltype={args.modeltype}",
        f"state_model={args.state_model}",
        f"w_fcd={args.w_fcd}",
    ]


# =========================
# PROCESS MANAGER
# =========================

class ProcessManager:
    def __init__(self):
        self.processes = []

    def start(self, cmd, name):
        print(f"[START] {name}")
        print("        ", " ".join(cmd))

        p = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            creationflags=subprocess.CREATE_NEW_CONSOLE
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

    args = parser.parse_args()

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
        pm.start(
            [sys.executable, "-m", "src.drive.fixed_npc_traffic"],
            "NPC_TRAFFIC"
        )

        time.sleep(2)

        # =========================
        # START DRIVE
        # =========================
        drive_cmd = build_drive_cmd(session, args)
        pm.start(drive_cmd, "DRIVE")

        # =========================
        # START PROVOICE
        # =========================
        provoice_cmd = build_provoice_cmd(session, args)
        pm.start(provoice_cmd, "PROVOICE")

        # =========================
        # MAIN LOOP (keep alive)
        # =========================
        print("[RUNNING] experiment started")

        while True:
            time.sleep(1)

    except KeyboardInterrupt:
        print("\n[EXIT] stopping experiment...")
        pm.stop_all()


if __name__ == "__main__":
    main()