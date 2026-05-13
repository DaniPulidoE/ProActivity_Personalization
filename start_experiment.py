#!/usr/bin/env python3
"""
Start both drive_improved.py and provoice in separate terminal windows with shared session_id.
Usage:
  python start_experiment.py --participantid 001 --environment city --secondary-task none \
    --functionname "Adjust seat positioning" --modeltype combined --state-model xlstm --w-fcd 0.7
"""

import os
import sys
import uuid
import subprocess
import argparse

VEHICLE_CONFIGS = [
    (0, "vehicle.sprinter.mercedes"),
    (5, "vehicle.ambulance.ford"),
    (10, "vehicle.firetruck.actors"),
    (15, "vehicle.lincoln.mkz"),
    (20, "vehicle.dodgecop.charger"),
    (25, "vehicle.mini.cooper"),
    (30, "vehicle.dodge.charger"),
    (35, "vehicle.fuso.mitsubishi"),
    (40, "vehicle.nissan.patrol"),
    (45, "vehicle.carlacola.actors"),
    (50, "vehicle.taxi.ford"),
]

def write_session_id(root: str) -> str:
    """Generate and atomically write session_id to .session_id file."""
    session = str(uuid.uuid4())
    tmp = os.path.join(root, ".session_id.tmp")
    path = os.path.join(root, ".session_id")

    with open(tmp, "w", encoding="utf-8") as f:
        f.write(session)
    os.replace(tmp, path)

    print(f"[INFO] Generated session_id: {session}")
    print(f"[INFO] Wrote to: {path}")
    return session


def build_drive_command(root: str, session: str, args: dict) -> str:
    """Build drive_improved.py command."""
    cmd_parts = [
        "python -m src.drive.drive_improved",
        "--control test",
        f"--session-id {session}",
        f"--participantid {args['participantid']}",
        f"--environment {args['environment']}",
        f"--secondary-task {args['secondary_task']}",
        f"--functionname \"{args['functionname']}\"",
        f"--modeltype {args['modeltype']}",
        f"--state-model {args['state_model']}",
        f"--w-fcd {args['w_fcd']}",
    ]
    return " ".join(cmd_parts)


def build_provoice_command(session: str, args: dict) -> str:
    """Build provoice command."""
    cmd_parts = [
        "uv run provoice",
        f"session_id={session}",
        f"participantid={args['participantid']}",
        f"environment={args['environment']}",
        f"secondary_task={args['secondary_task']}",
        f"functionname={args['functionname']}",
        f"modeltype={args['modeltype']}",
        f"state_model={args['state_model']}",
        f"w_fcd={args['w_fcd']}",
    ]
    return " ".join(cmd_parts)


def start_on_windows(root: str, drive_cmd: str, provoice_cmd: str):
    """Start processes in new PowerShell windows on Windows."""
    print("[INFO] Starting on Windows (PowerShell)...")

    # Start drive_improved in new PowerShell window
    ps_drive = (
        f"Set-Location -Path '{root}'; "
        f"{drive_cmd}; "
        "Write-Host 'Press any key to close...'; "
        "$null = $Host.UI.RawUI.ReadKey('NoEcho,IncludeKeyDown')"
    )
    subprocess.Popen([
        "powershell.exe",
        "-NoExit",
        "-Command",
        ps_drive
    ])
    print("[INFO] Started drive_improved in new PowerShell window")

    # Start provoice in new PowerShell window
    ps_provoice = (
        f"Set-Location -Path '{root}'; "
        f"{provoice_cmd}; "
        "Write-Host 'Press any key to close...'; "
        "$null = $Host.UI.RawUI.ReadKey('NoEcho,IncludeKeyDown')"
    )
    subprocess.Popen([
        "powershell.exe",
        "-NoExit",
        "-Command",
        ps_provoice
    ])
    print("[INFO] Started provoice in new PowerShell window")


def start_on_unix(root: str, drive_cmd: str, provoice_cmd: str):
    """Start processes in new terminal windows on macOS/Linux."""
    print("[INFO] Starting on Unix (macOS/Linux)...")

    system = sys.platform

    if system == "darwin":
        # macOS: use osascript to open Terminal
        print("[INFO] Attempting to open new Terminal windows on macOS...")

        # Start drive_improved
        script_drive = f"""
        tell application "Terminal"
            do script "cd '{root}' && {drive_cmd}"
        end tell
        """
        try:
            subprocess.run([
                "osascript",
                "-e",
                script_drive
            ], check=True)
            print("[INFO] Started drive_improved in new Terminal window")
        except Exception as e:
            print(f"[WARN] Failed to open Terminal for drive_improved: {e}")
            print(f"[INFO] You can run manually: cd {root} && {drive_cmd}")

        # Start provoice
        script_provoice = f"""
        tell application "Terminal"
            do script "cd '{root}' && {provoice_cmd}"
        end tell
        """
        try:
            subprocess.run([
                "osascript",
                "-e",
                script_provoice
            ], check=True)
            print("[INFO] Started provoice in new Terminal window")
        except Exception as e:
            print(f"[WARN] Failed to open Terminal for provoice: {e}")
            print(f"[INFO] You can run manually: cd {root} && {provoice_cmd}")

    elif system.startswith("linux"):
        # Linux: try common terminal emulators
        print("[INFO] Attempting to open new terminal windows on Linux...")

        terminals = ["gnome-terminal", "konsole", "xterm", "xfce4-terminal"]
        found = False

        for term in terminals:
            try:
                # Try to start drive_improved
                if term == "gnome-terminal":
                    subprocess.Popen([term, "--", "bash", "-c", f"cd '{root}' && {drive_cmd}; exec bash"])
                elif term == "konsole":
                    subprocess.Popen([term, "-e", "bash", "-c", f"cd '{root}' && {drive_cmd}; exec bash"])
                elif term == "xfce4-terminal":
                    subprocess.Popen([term, "-e", f"bash -c 'cd {root} && {drive_cmd}; exec bash'"])
                else:  # xterm
                    subprocess.Popen([term, "-e", f"bash -c 'cd {root} && {drive_cmd}; exec bash'"])

                print(f"[INFO] Started drive_improved using {term}")
                found = True
                break
            except FileNotFoundError:
                continue

        if not found:
            print("[WARN] Could not find a suitable terminal emulator.")
            print(f"[INFO] You can run manually in two separate terminals:")
            print(f"  Terminal 1: cd {root} && {drive_cmd}")
            print(f"  Terminal 2: cd {root} && {provoice_cmd}")
            return

        # Try the same terminal for provoice
        for term in terminals:
            try:
                if term == "gnome-terminal":
                    subprocess.Popen([term, "--", "bash", "-c", f"cd '{root}' && {provoice_cmd}; exec bash"])
                elif term == "konsole":
                    subprocess.Popen([term, "-e", "bash", "-c", f"cd '{root}' && {provoice_cmd}; exec bash"])
                elif term == "xfce4-terminal":
                    subprocess.Popen([term, "-e", f"bash -c 'cd {root} && {provoice_cmd}; exec bash'"])
                else:  # xterm
                    subprocess.Popen([term, "-e", f"bash -c 'cd {root} && {provoice_cmd}; exec bash'"])

                print(f"[INFO] Started provoice using {term}")
                break
            except FileNotFoundError:
                continue


def main():
    parser = argparse.ArgumentParser(
        description="Start drive_improvedUE5.py and provoice with shared session_id"
    )
    parser.add_argument("--participantid", default="001", help="Participant ID (default: 001)")
    parser.add_argument("--environment", default="city", help="Environment (default: city)")
    parser.add_argument("--secondary-task", default="none", help="Secondary task (default: none)")
    parser.add_argument("--functionname", default="Adjust seat positioning", help="Function name")
    parser.add_argument("--modeltype", default="combined", help="Model type (default: combined)")
    parser.add_argument("--state-model", default="xlstm", help="State model (default: xlstm)")
    parser.add_argument("--w-fcd", default="0.7", help="FCD weight (default: 0.7)")

    args = parser.parse_args()

    root = os.getcwd()

    # Generate and write session_id
    session = write_session_id(root)

    # Build commands
    drive_cmd = build_drive_command(root, session, vars(args))
    provoice_cmd = build_provoice_command(session, vars(args))

    print("\n[INFO] Drive command:")
    print(f"  {drive_cmd}\n")
    print("[INFO] ProVoice command:")
    print(f"  {provoice_cmd}\n")

    # Start processes based on platform
    if sys.platform.startswith("win"):
        start_on_windows(root, drive_cmd, provoice_cmd)
    else:
        start_on_unix(root, drive_cmd, provoice_cmd)

    print("\n[INFO] Both processes started. Check the opened windows.")
    print("[INFO] Session ID stored in: .session_id")


if __name__ == "__main__":
    main()

