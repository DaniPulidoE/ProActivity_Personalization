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
--webcam: use the external USB webcam instead of the laptop's built-in camera
    (ProVoice uses camera index 1 and refuses to start if nothing is there;
     index 0 is the built-in one)
--test-popup: teaching mode, simulator UI + immediate LoA popups, nothing logged
--remote: CARLA + traffic + Drive (with the LoA popups) run HERE and vehicle
    state is served over HTTP; ProVoice runs on ANOTHER machine and is started
    there by hand, with the command line this launcher prints

LoA POPUP INTERFACE (default = keyboard on every rig, wheel attached or not):
--keyboard-input: the default, stated explicitly -- number keys 0-4 tick a level
    (same key again unticks), N ticks INVALID FRAME instead, ENTER commits
    whatever is ticked (an invalid frame writes no label)
--wheel-input: override -- paddles move the cursor, front button ticks, CONFIRM
    row submits

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

EXPERIMENT SETUP (remote):
0. Launch CARLA:
    CARLA MACHINE: ./CarlaUnreal --RenderOffScreen 
1. Driver adaptation phase:
    CARLA MACHINE:    uv run python start_experiment.py --experiment-adaptation
2. Teaching the LoA control:
    CARLA MACHINE:    uv run python start_experiment.py --experiment-popup
3. Calibration:
    CARLA MACHINE:    uv run python start_experiment.py --experiment-calibration-carla-remote --participantid <participantid>
    PROVOICE MACHINE: uv run python start_experiment.py --experiment-calibration-provoice-remote
4. Data collection:
    CARLA MACHINE:    uv run python start_experiment.py --experiment-data-collection-carla-remote --participantid <participantid> --run <1|2>
    PROVOICE MACHINE: uv run python start_experiment.py --experiment-data-collection-provoice-remote

No addresses anywhere: both machines run THIS file, so both take the link
address from CARLA_MACHINE_IP below -- the CARLA side to publish and bind, the
ProVoice side to connect. The participant id is typed once, on the CARLA
machine, and read from the bridge on the other. Off the rig, the ProVoice
presets still accept an explicit address (or PV_BRIDGE_URL). Setup for the link
itself: docs/remote_setup.md.

--participantid is given ONCE, on the CARLA machine: the ProVoice machine reads
it (and the session id) from the bridge, so there is no second place to mistype
it. The ProVoice side needs only the address to read it FROM -- a bare IP, an
IP:PORT or a full URL, or nothing at all if PV_BRIDGE_URL is set in that
shell's environment.

TWO BRIDGES, BOTH ON THE CARLA MACHINE, BOTH STARTED BY --remote
    scripts/vehicle_state_server.py  :8080  CARLA state -> ProVoice (polled)
    scripts/provoice_status_server.py :8081  ProVoice's lifecycle -> Drive
The second one is what makes a split session behave like a single-machine one:
ProVoice posts 'collection_started' at its first logged frame, so Drive holds
its LoA windows until then exactly as it holds them for the first line of a
local raw_data.jsonl, and posts 'provoice_ended' when it exits, on which Drive
stops the car and shows the end-of-session screen. ProVoice is told where to
post through the vehicle bridge's /session, so still nothing is typed twice --
but BOTH inbound ports have to be open on the CARLA machine.

WHAT EACH --experiment- PRESET EXPANDS TO
Presets set flags and nothing else: every one of them is exactly equivalent to
the line beside it, and any flag they do not mention (--webcam, --environment,
--res, --host, ...) can still be added on the same command line.

    --experiment-popup
        = --test-popup --fullscreen
    --experiment-adaptation
        = --test-drive --no-popup --fullscreen
    --experiment-calibration-carla-remote
        = --fixed --remote --no-popup --fullscreen
          --remote-host <CARLA_MACHINE_IP> --remote-bind <CARLA_MACHINE_IP>
          (--remote alone means "ProVoice runs on the other machine, not
           here", so --test-drive would add nothing; --no-popup because the
           baseline drive collects no labels)
    --experiment-data-collection-carla-remote
        = --remote --data-collection --random-function --fullscreen
          --remote-host <CARLA_MACHINE_IP> --remote-bind <CARLA_MACHINE_IP>

    This one also needs --run (1 or 2). The traffic scenario is derived from
    --participantid and --run through TRAFFIC_SEED_PLAN below, so the two
    together are what say which session this is; there is no seed to look up
    or retype. The calibration and adaptation presets take neither -- they
    always run the study scenario.

    Both pin the study link's address, since it is a property of the rig
    rather than of the run: --remote-host is what gets published to the
    ProVoice machine (the automatic guess names the Wi-Fi card instead), and
    --remote-bind keeps the bridges off every other network this machine is
    on. Edit CARLA_MACHINE_IP if the rig is re-addressed; passing either flag
    explicitly still wins for a one-off.

The two ProVoice-machine presets have no equivalent flag spelling, because
that machine runs neither CARLA nor Drive and this launcher previously had no
mode for "ProVoice alone". They start ProVoice and nothing else:

    --experiment-calibration-provoice-remote [URL]
        = ProVoice with --calibration-only --webcam, reading vehicle state from
          <URL>/ and its participant/session ids AND the status-bridge address
          from <URL>/session. Runs the 180 s baseline, stores it under the
          participant id the bridge reports, and exits by itself -- which is
          also what ends the drive on the other machine. Nothing is written to
          raw_data.jsonl.
    --experiment-data-collection-provoice-remote [URL]
        = ProVoice with --data-collection --webcam --data-collection-timeout 900,
          same id handling. Records raw_data.jsonl only: no decision engine, no
          interventions, and the participant's stored baseline is reused rather
          than re-measured. Its first logged frame is what opens the LoA
          windows over there. After 15 minutes it stops itself exactly as
          --calibration-only stops itself when its baseline finishes, which
          posts 'provoice_ended' and makes Drive show the same end-of-session
          screen it shows at the end of calibration -- deferred until any LoA
          prompt on screen at that moment has been answered (or skipped), so
          the cutoff never discards a label mid-window.

    [URL] defaults to PV_BRIDGE_URL and then to CARLA_MACHINE_IP, so on the
    rig neither preset takes an argument at all. Give one to point at a
    different machine, or a full URL for a tunnel. The source that won is
    printed at startup -- worth a glance, since the pinned default is only
    right while both machines are running the same revision of this file.

    Both imply --webcam, i.e. the external driver-facing camera rather than
    this machine's built-in one, and both imply it so the baseline and the
    session it normalizes are measured through the SAME lens. --no-webcam
    cancels it; use that for both phases or neither.

    Neither ProVoice-machine preset starts a bridge of its own: both bridges
    live on the CARLA machine (see above), so this machine needs no inbound
    port and no address beyond the one URL given here.

SPLIT ACROSS TWO MACHINES (--remote): the CARLA machine also prints a full
ProVoice command line for the other machine. The presets above are the short
way in; that printed line is the long way, and still worth pasting when the run
needs a flag the presets do not set (a different --modeltype, --environment,
--w-fcd, ...), since it carries every one of them already filled in.

Either way the session id and the participant id do NOT have to be retyped: the
vehicle-state bridge publishes both at <url>/session and ProVoice adopts them
from the URL it is already given. Supplying them anyway is fine -- ProVoice
checks them against the bridge and REFUSES TO START if they disagree, which is
the whole point: a mistyped participant id is otherwise invisible until the
data is analysed.
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

# Traffic manager port for the whole rig. ONE port, deliberately: CARLA's traffic
# manager runs inside the client process that creates it, so under --sync the
# process that ticks must be the process that owns the traffic manager the NPCs
# are registered to. Two ports means one of them goes unticked and its vehicles
# stop dead -- which is exactly what happened the last time --sync was attempted
# (see src/drive/fixed_npc_traffic.py's module docstring). Passed to both
# children so neither can drift from the other.
TM_PORT = 9000

# Traffic scenario for the CALIBRATION and ADAPTATION drives, and the default
# everywhere. Pinned rather than varied, and pinned to the value the rig has
# always run on, because those drives are a fixed condition: the participant is
# being compared against other participants in the same arm, so the traffic they
# are compared in has to be the same traffic. --traffic-seed is refused on those
# runs for that reason, not merely ignored.
#
# Data collection is the opposite case and varies it deliberately -- see the
# --traffic-seed help text. Keeping this value OUT of the collection seeds is
# what makes the study a test on traffic the population model has never been
# trained on, rather than a rerun of a scenario it has already seen.
STUDY_TRAFFIC_SEED = 42

# =========================
# TRAFFIC SCENARIO PLAN (data collection)
# =========================
#
# WHICH SCENARIO EACH PARTICIPANT DRIVES, decided once and written down, rather
# than typed per session. The seed is DERIVED from --participantid and --run;
# there is nothing to look up on the morning and nothing to mistype.
#
# That matters more than convenience. The assignment below is a counterbalanced
# design, and a design is only a design if it is actually executed: one row read
# off the wrong line of a notebook produces an unbalanced corpus, and nothing
# downstream can detect it -- the run looks normal, the data looks normal, and
# the imbalance is only ever visible to someone who reconstructs what should
# have happened. Deriving it removes the step where that can go wrong.
#
# THE DESIGN. Four scenarios over twelve participants x two runs. P(4,2) = 12 --
# four choices for run 1 times three remaining for run 2 -- so every participant
# gets a DIFFERENT ORDERED PAIR and every ordered pair occurs exactly once. Each
# scenario is then used six times: three as a first run and three as a second.
# Verified at startup by _validate_traffic_plan().
#
#  - Ordered pairs all distinct means every UNORDERED pair occurs exactly twice,
#    once in each order. So the scenario effect is estimable, not tangled with
#    any one participant, AND separable from the order it was met in -- which
#    the previous ten-participant design (each unordered pair once, in one
#    arbitrary order) could not do.
#  - The 3/3 split across run positions is the counterbalance that matters:
#    run 2 differs from run 1 for everyone -- they have learned the rig, learned
#    the route, and are fifteen minutes more tired. A scenario that always
#    landed in run 1 would have that baked into it inseparably.
#  - Holding out any single scenario removes exactly six runs spread over six
#    different participants, which is an honest generalisation test for the
#    population model rather than a test on one person.
#  - THE ROWS ARE IN BLOCKS OF FOUR (001-004, 005-008, 009-012) and each block
#    is already balanced on its own: within it every scenario appears once as
#    run 1 and once as run 2. Recruitment that stops at four or eight
#    participants therefore still yields a complete counterbalance rather than a
#    broken one -- which is the realistic failure mode, since the design is only
#    balanced at exactly 12 and participants drop out. Run the rows IN ORDER for
#    that to hold. (The three blocks are the "rotate by 1 / by 2 / by 3" shifts
#    of the scenario list, which is what makes them cover the 12 arcs exactly.)
#
# WHY 42 IS ABSENT. It is the calibration/adaptation scenario (STUDY_TRAFFIC_SEED
# above). Keeping it out of collection is what makes the study a test on traffic
# the model has never been trained on.
#
# THE RUN NUMBER IS NOT LOGGED SEPARATELY, and does not need to be: no
# participant sees the same scenario twice (enforced below), so run number is
# recoverable from the participantid and traffic_seed that every raw_data.jsonl
# frame already carries. Note this is a property of the pairs, not of the
# design: it survives dropping to four scenarios only because no row repeats a
# scenario, which _validate_traffic_plan makes fatal.
#
# CHANGING THIS: edit before the first participant, then leave it alone. If
# someone drops out, re-run THEIR row with the replacement rather than adding a
# new one -- a fresh pair would duplicate one ordered pair and leave another
# unused. For a 13th participant onwards, restart the cycle (P13 = P001's pair,
# P14 = P002's, ...) so each block of four keeps the balance growing evenly. A
# pilot or a re-run that should not follow the plan uses an explicit
# --traffic-seed, which is honoured and announced.
#
# The seed VALUES are arbitrary labels -- nothing is derived from the number and
# any four distinct integers would do. They only have to be distinct, stable
# once collection starts, and different from STUDY_TRAFFIC_SEED. What the seed
# selects is a whole scenario, density included: fixed_npc_traffic.py's
# SCENARIO_FLEET_SIZE maps 11/22/44/55 to 40/30/20/15 vehicles, so these four
# are a dose series and their ORDER in COLLECTION_SEEDS is the dose order.
# 33 (25 vehicles) was the fifth and is RETIRED -- the middle rung, dropped
# rather than an endpoint so the 15-40 range is unchanged. It is no longer a
# legal planned scenario (an entry naming it is fatal below), though
# --traffic-seed 33 still runs it off-plan for a pilot.
COLLECTION_SEEDS = (11, 22, 44, 55)

TRAFFIC_SEED_PLAN = {
    # participantid: (run 1 scenario, run 2 scenario)
    # Block 1 -- each scenario once as run 1, once as run 2
    "001": (11, 22),
    "002": (22, 44),
    "003": (44, 55),
    "004": (55, 11),
    # Block 2
    "005": (11, 44),
    "006": (22, 55),
    "007": (44, 11),
    "008": (55, 22),
    # Block 3
    "009": (11, 55),
    "010": (22, 11),
    "011": (44, 22),
    "012": (55, 44),
}

RUNS_PER_PARTICIPANT = 2

# Width the plan's participant ids are zero-padded to. Used ONLY to recognise
# "1" and "01" as participant "001" at lookup time -- see _canonical_participant.
PARTICIPANT_ID_WIDTH = 3


def _canonical_participant(pid):
    """The plan's spelling of a participant id, or the id unchanged.

    "001" and "1" are different dict keys but the same person, and the operator
    typing the short form at 9am is not a mistake worth failing a session over.
    What IS worth avoiding is the two spellings both reaching the data: the
    participant id is what the corpus is grouped by, so one session filed under
    "1" and the next under "001" splits a participant in half, and nothing
    downstream notices. Recognising the short form here and using the plan's
    spelling from then on keeps every session for a participant under one name.
    """
    pid = (pid or "").strip()
    if pid in TRAFFIC_SEED_PLAN:
        return pid
    if pid.isdigit():
        padded = pid.zfill(PARTICIPANT_ID_WIDTH)
        if padded in TRAFFIC_SEED_PLAN:
            return padded
    return pid


def _validate_traffic_plan():
    """Check the plan still is what its comment says, at every startup.

    Cheap, and it guards the one failure this whole arrangement exists to
    prevent: an edit that quietly breaks the balance. A plan that no longer
    counterbalances looks exactly like one that does from inside a session, and
    the damage is only visible once the corpus is complete.

    Errors are fatal rather than warnings for the properties that are ALWAYS
    wrong -- a participant driving one scenario twice (which would also make the
    run number unrecoverable), a scenario outside COLLECTION_SEEDS, the study
    seed leaking into collection. The count balance is reported rather than
    enforced, because a plan is legitimately unbalanced while it is being
    extended past twelve participants, or while a block is half-recruited.
    """
    problems = []
    for pid, pair in TRAFFIC_SEED_PLAN.items():
        if len(pair) != RUNS_PER_PARTICIPANT:
            problems.append("participant %s has %d run(s), expected %d"
                            % (pid, len(pair), RUNS_PER_PARTICIPANT))
            continue
        if len(set(pair)) != len(pair):
            problems.append(
                "participant %s drives scenario %d twice. Beyond wasting a run, "
                "it makes the run number unrecoverable from the logged data, "
                "which is the only reason --run is not recorded separately."
                % (pid, pair[0]))
        for seed in pair:
            if seed == STUDY_TRAFFIC_SEED:
                problems.append(
                    "participant %s is assigned the study scenario %d. That is "
                    "the calibration/adaptation condition; using it for "
                    "collection means the study is evaluated on traffic the "
                    "model was trained on." % (pid, seed))
            elif seed not in COLLECTION_SEEDS:
                problems.append("participant %s is assigned %d, which is not in "
                                "COLLECTION_SEEDS" % (pid, seed))
    if problems:
        raise SystemExit("Bad TRAFFIC_SEED_PLAN in start_experiment.py:\n  "
                         + "\n  ".join(problems))

    # Reported, not fatal -- and printed only when something is off, so a
    # correct plan is silent. This is the half of the design that a bad edit
    # breaks WITHOUT breaking anything a single session can notice: the plan
    # still resolves a scenario for every participant and run, it just no
    # longer counterbalances. Saying so at startup is the only moment anyone
    # would find out before the corpus is complete.
    for line in traffic_plan_imbalances():
        print("[PLAN][warn] " + line)


def traffic_plan_summary():
    """One line per scenario: how often it is used, and in which run position.

    Plus one line for the ordered-pair coverage, which is what the twelve-row
    design actually claims: P(4,2) = 12 pairs, each used exactly once.
    """
    total, first = {}, {}
    for pair in TRAFFIC_SEED_PLAN.values():
        for i, seed in enumerate(pair):
            total[seed] = total.get(seed, 0) + 1
            if i == 0:
                first[seed] = first.get(seed, 0) + 1
    lines = ["scenario %d: %d run(s), %d as run 1, %d as run 2"
             % (s, total[s], first.get(s, 0), total[s] - first.get(s, 0))
             for s in sorted(total)]
    n_seeds = len(COLLECTION_SEEDS)
    lines.append("ordered pairs: %d distinct of %d rows, %d possible"
                 % (len(set(TRAFFIC_SEED_PLAN.values())),
                    len(TRAFFIC_SEED_PLAN),
                    n_seeds * (n_seeds - 1)))
    return lines


def traffic_plan_imbalances():
    """Ways the plan departs from the counterbalance, worst first. [] if none.

    Everything here is a COUNT, i.e. a property of the plan as a whole rather
    than of any one row -- which is exactly why it needs stating out loud. A row
    that is wrong on its own is caught above and is fatal; a plan whose rows are
    each individually fine but which uses one scenario five times and another
    seven is indistinguishable from a correct one at every point where anyone
    looks, until the analysis.

    A part-recruited plan trips these legitimately (a half-finished block is
    unbalanced by construction), so they are warnings and the message says which
    complete size would clear them.
    """
    out = []
    pairs = list(TRAFFIC_SEED_PLAN.values())
    block = len(COLLECTION_SEEDS)
    full = block * (block - 1)

    dupes = sorted({p for p in pairs if pairs.count(p) > 1})
    if dupes:
        out.append("ordered pair(s) used more than once: %s. Each one used "
                   "twice means another is unused, so that scenario order is "
                   "never observed."
                   % ", ".join("(%d, %d)" % p for p in dupes))

    if len(pairs) % block:
        out.append("%d participants planned; the design is balanced in blocks "
                   "of %d (%d, %d, %d, ...), so this one stops mid-block."
                   % (len(pairs), block, block, 2 * block, 3 * block))

    total, first = {}, {}
    for pair in pairs:
        for i, seed in enumerate(pair):
            total[seed] = total.get(seed, 0) + 1
            if i == 0:
                first[seed] = first.get(seed, 0) + 1
    missing = [s for s in COLLECTION_SEEDS if s not in total]
    if missing:
        out.append("scenario(s) %s appear in no run at all."
                   % ", ".join(str(s) for s in missing))
    if total and len(set(total.values())) > 1:
        out.append("scenarios are not driven equally often: "
                   + ", ".join("%d x%d" % (s, total[s]) for s in sorted(total)))
    lopsided = [s for s in sorted(total)
                if abs(2 * first.get(s, 0) - total[s]) > 1]
    if lopsided:
        out.append("scenario(s) %s sit mostly in one run position, so their "
                   "effect is confounded with practice and fatigue: %s"
                   % (", ".join(str(s) for s in lopsided),
                      ", ".join("%d = %d run 1 / %d run 2"
                                % (s, first.get(s, 0), total[s] - first.get(s, 0))
                                for s in lopsided)))
    if out and len(pairs) != full:
        out.append("(the complete design is %d participants: every ordered "
                   "pair of the %d scenarios exactly once.)" % (full, block))
    return out


def resolve_traffic_seed(args, parser):
    """The scenario for this session: derived from the plan, or overridden.

    Order matters. An explicit --traffic-seed wins, because a pilot, a re-run
    after a crash, or a one-off check has to be possible without editing the
    plan -- but it says so out loud, since a session that silently departed from
    the design would be indistinguishable afterwards from one that followed it.

    SIDE EFFECT, deliberate and announced: on a planned run this rewrites
    args.participantid to the plan's spelling of the id, so a participant typed
    as "1" is recorded as "001" like their other session. See
    _canonical_participant for why that matters more than it looks.
    """
    if not args.data_collection:
        if args.traffic_seed is not None:
            parser.error(
                "--traffic-seed %d is only accepted on --data-collection runs; "
                "this one is not. The calibration and adaptation drives are "
                "pinned to scenario %d so every participant is measured in the "
                "same traffic. Drop --traffic-seed, or add --data-collection if "
                "this is a collection session."
                % (args.traffic_seed, STUDY_TRAFFIC_SEED))
        if args.run is not None:
            parser.error(
                "--run only selects a data-collection scenario and this is not a "
                "--data-collection run. The calibration and adaptation drives "
                "always use scenario %d." % STUDY_TRAFFIC_SEED)
        return STUDY_TRAFFIC_SEED

    if args.traffic_seed is not None:
        print("[PLAN] OVERRIDE: --traffic-seed %d given, so the plan is NOT "
              "being followed for this session." % args.traffic_seed)
        if args.run is not None:
            planned = TRAFFIC_SEED_PLAN.get(_canonical_participant(args.participantid))
            if planned and 1 <= args.run <= len(planned) \
                    and planned[args.run - 1] != args.traffic_seed:
                print("[PLAN]   the plan says participant %s run %d is scenario "
                      "%d." % (args.participantid, args.run,
                               planned[args.run - 1]))
        return args.traffic_seed

    if args.run is None:
        parser.error(
            "--data-collection needs --run (1..%d): the traffic scenario is "
            "derived from --participantid and --run via TRAFFIC_SEED_PLAN. Pass "
            "--traffic-seed instead to run a scenario off-plan (a pilot, or a "
            "re-run)." % RUNS_PER_PARTICIPANT)

    pid = _canonical_participant(args.participantid)
    if pid != (args.participantid or "").strip():
        print("[PLAN] participant id %r -> %r, the plan's spelling, so this "
              "session is filed with the participant's other run."
              % ((args.participantid or "").strip(), pid))
        args.participantid = pid
    pair = TRAFFIC_SEED_PLAN.get(pid)
    if pair is None:
        parser.error(
            "no TRAFFIC_SEED_PLAN entry for participant %r. Planned "
            "participants are: %s. If this is a pilot or an extra participant, "
            "either add a row to the plan in start_experiment.py or pass an "
            "explicit --traffic-seed."
            % (pid, ", ".join(sorted(TRAFFIC_SEED_PLAN, key=lambda k: (len(k), k)))))
    if not 1 <= args.run <= len(pair):
        parser.error("--run %d is out of range; participant %s has %d planned "
                     "run(s)." % (args.run, pid, len(pair)))

    seed = pair[args.run - 1]
    print("[PLAN] participant %s, run %d -> traffic scenario %d "
          "(the other run is %d)."
          % (pid, args.run, seed,
             pair[0] if args.run == 2 else pair[1]))
    return seed


# Under --sync, how Drive asks the clock owner to hold the simulation while a LoA
# popup is open. Defined HERE and passed to both children, rather than agreed on
# by two files, because a silent disagreement about the path would not fail --
# Drive would write a pause nobody reads and fall back to freezing every vehicle
# individually, which is the unnatural behaviour the pause exists to remove.
CLOCK_PAUSE_FILE = "clock_paused.flag"

# THE RIG. Address of the CARLA machine on the dedicated Ethernet link between
# the two study machines. Fixed for the whole data-collection run, so the
# --experiment-*-carla-remote presets set it themselves rather than having it
# retyped at 9am with a participant waiting. CHANGE IT HERE if the rig is
# re-addressed; an explicit --remote-host / --remote-bind still overrides it.
#
# It fills two different roles, and both need it:
#   --remote-host  the address PUBLISHED to the ProVoice machine. Left to
#                  itself the launcher guesses with outbound_ip(), which
#                  answers "which interface reaches the internet" -- the Wi-Fi
#                  card, not the cable carrying this pairing.
#   --remote-bind  the interface the two bridges LISTEN on. Pinning it here
#                  means they are not reachable from campus Wi-Fi at all,
#                  instead of being exposed there and merely firewalled off.
CARLA_MACHINE_IP = "192.168.50.1"


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


def advertised_host(args) -> str:
    """The address to hand the other machine: --remote-host, else the guess.

    Worth pinning whenever the two machines are joined by a dedicated link.
    outbound_ip() answers "which interface reaches the internet", and on a rig
    with Wi-Fi for the internet and Ethernet for the pairing, that is precisely
    the wrong one -- the URLs would name an address the other machine may not
    route to at all, and the symptom is a POST that TIMES OUT rather than
    anything that names the cause.
    """
    if args.remote_host:
        return args.remote_host.strip()
    return outbound_ip()


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
        # Under --sync this makes Drive wait on the clock instead of ticking, and
        # points it at the clock owner's traffic manager. Both halves have to
        # travel together -- see the --sync help text below.
        # The pause file goes in under BOTH clocks: NPC_TRAFFIC can hold a
        # free-running world too (by switching it into synchronous mode and not
        # ticking), so the popup freeze is a real freeze either way and traffic
        # never stops dead and pulls away from rest. Only --sync/--tm-port are
        # conditional.
        "--clock-pause-file", CLOCK_PAUSE_FILE,
        *(["--sync", "--tm-port", str(TM_PORT)] if args.sync else []),
        # Omitted when neither flag was given, so Drive keeps its own default
        # (the keyboard, on every rig).
        *([f"--{args.popup_input}-input"] if args.popup_input else []),
        # Drive owns the draw: it is the process that shows the popups, so the
        # pool lives there and only the switch is forwarded.
        *(["--random-function"] if args.random_function else []),
        # Only under --remote: locally Drive reads both facts itself (the log
        # for the start, this launcher for the end), and pointing it at a
        # status file nothing writes would just add a file to ignore.
        *(["--provoice-status-file", args.status_file] if args.remote else []),
        *(["--popup-immediate"] if args.test else []),
        *(["--speed"] if args.speed else []),
        *(["--render-scale", str(args.render_scale)]
          if args.render_scale != 1.0 else []),
        # Forwarded ONLY when the operator typed them, so the default lives in
        # exactly one place (DEFAULT_AMBIENT_GAIN in src/drive/ambience.py) and
        # the launcher cannot drift from Drive and hand two participants
        # different conditions. Drive logs whatever it ends up using into every
        # label row, so nothing depends on the launcher knowing the number.
        #
        # This is also the whole reason the ambience is confined to the CARLA
        # machine: the audio lives inside the Drive process, and the ProVoice
        # presets never reach this function -- they route to run_provoice_only(),
        # which starts no Drive at all.
        *(["--ambient-gain", str(args.ambient_gain)]
          if args.ambient_gain is not None else []),
        *(["--ambient-seed", str(args.ambient_seed)]
          if args.ambient_seed is not None else []),
        # Scripted precipitation ramp -- see drive_improved.py's World class
        # (start_weather_schedule/update_weather_schedule) for the mechanics.
        # Drive is where CARLA weather is actually set, so this only reaches
        # Drive's command line; ProVoice already reads live precipitation off
        # the shared CARLA world every frame, no flag needed on that side.
        *(["--condition-sun-rain"] if args.condition_sun_rain else []),
        *(["--condition-rain-sun"] if args.condition_rain_sun else []),
        "--popup-wait-timeout", str(args.popup_wait_timeout),
    ]


def build_provoice_cmd(session, args, vehicle_id, remote_url=None,
                       ids_from_bridge=False):
    """Command line for ProVoice.

    ``remote_url`` (--remote) is the one case where the result is PRINTED for a
    human to run on another machine rather than spawned here. It is built by
    this same function on purpose: a separately hand-written command line is
    how the two ends end up disagreeing about the session id, the participant
    or the model, and every one of those mismatches only shows up in the data.

    ``ids_from_bridge`` is the ProVoice-machine presets: this launcher is
    running on the machine that has neither CARLA nor Drive, so it knows
    NEITHER id. Omitting them is what makes ProVoice adopt the pair the bridge
    publishes at <url>/session; passing this machine's defaults instead
    (participantid="001", a session id minted here) would look authoritative
    and be wrong. ``session`` and ``vehicle_id`` are ignored in that mode.
    """
    return [
        sys.executable,
        "src/ProVoice/main.py",
        *([] if ids_from_bridge else [
            f"session_id={session}",
            f"vehicle_id={vehicle_id}",
            f"participantid={args.participantid}",
            # Rides with the ids, and is omitted for the same reason they are:
            # under --remote the bridge publishes all of them at /session and
            # ProVoice adopts them there. Passing it here as well would give
            # ProVoice two sources for one fact, and resolve_traffic_seed()
            # treats a disagreement between them as fatal.
            f"traffic_seed={args.traffic_seed}",
        ]),
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
        # status_url is deliberately NOT passed here: ProVoice adopts it from
        # the vehicle bridge's /session along with the two ids, so the printed
        # command stays about what this run IS rather than how its machines are
        # wired. Pass status_url=... by hand only to override it (a tunnel).
        # bridge_session_timeout=0: the dead port has no /session either, and
        # ProVoice would otherwise spend its startup retrying a URL that exists
        # to be unreachable. Both ids are on this command line already.
        *([f"vehicle_state_url=http://127.0.0.1:{_DEAD_PORT}",
           "bridge_session_timeout=0"]
          if args.provoice_no_carla and not args.vehicle_bridge else []),
        # Bare flags, not key=value: ProVoice parses these with store_true, which
        # rejects an explicit "=value". _normalize_argv passes "-"-prefixed
        # tokens through untouched.
        *(["--calibration-only"] if args.calibration_only else []),
        *(["--data-collection"] if args.data_collection else []),
        *(["--data-collection-timeout", str(args.data_collection_timeout)]
          if args.data_collection_timeout is not None else []),
        # Forwarded rather than resolved here on purpose: ProVoice owns the
        # camera, and probing an index means OPENING the device. Doing that in
        # this process would leave the launcher holding a handle that Windows
        # has not necessarily released by the time ProVoice tries to open the
        # same device -- exactly the failure the restart delay below exists for.
        *(["--webcam"] if args.webcam else []),
    ]


# =========================
# EXPERIMENT PRESETS
# =========================
#
# One flag per phase of the study protocol, because the protocol is what the
# operator is following at 9am with a participant waiting -- not a flag
# combination. Each preset does nothing but SET the flags it is documented as
# setting (see the table in the module docstring), so everything downstream --
# every conflict check, every warning -- sees exactly what a hand-typed command
# line would have produced. Nothing here has behaviour of its own.
#
# The two CARLA-machine presets and the two ProVoice-machine ones are different
# in kind: the first pair configures the normal CARLA+Drive(+bridge) run, the
# second pair runs ProVoice ALONE against a bridge on the other machine, which
# is a mode this launcher did not have before.

_CARLA_PRESETS = {
    # attribute name -> {flag: value} applied to the parsed args
    "experiment_popup": {"test_popup": True, "fullscreen": True},
    "experiment_adaptation": {"test_drive": True, "no_popup": True,
                              "fullscreen": True},
    # The two remote presets also pin the link addresses (see CARLA_MACHINE_IP):
    # they are a property of the rig, not of the run, and getting either wrong
    # fails as a TIMEOUT minutes later rather than as an error at startup.
    "experiment_calibration_carla_remote": {
        "fixed": True, "remote": True, "no_popup": True, "fullscreen": True,
        "remote_host": CARLA_MACHINE_IP, "remote_bind": CARLA_MACHINE_IP},
    "experiment_data_collection_carla_remote": {
        "remote": True, "data_collection": True, "random_function": True,
        "fullscreen": True,
        "remote_host": CARLA_MACHINE_IP, "remote_bind": CARLA_MACHINE_IP},
}

_PROVOICE_PRESETS = {
    # attribute name -> flags for the ProVoice this machine will start alone.
    #
    # webcam=True in BOTH, and that is the point of setting it here rather than
    # leaving it to the operator: the driver-facing camera is the external one
    # on this machine, and the setting has to be IDENTICAL in the two phases.
    # The two cameras differ in framing and frame rate, and the achieved frame
    # rate is the rPPG sampling rate -- so calibrating on one and collecting on
    # the other silently rescales that participant's HR/RR against a baseline
    # measured from a different signal. Nothing in the data says so; it shows
    # up, if at all, as one participant whose physiological features sit oddly.
    # Tying both presets to the same choice removes the chance to get it half
    # right. --no-webcam overrides, for a machine that only has a built-in one.
    "experiment_calibration_provoice_remote": {"calibration_only": True,
                                               "webcam": True},
    "experiment_data_collection_provoice_remote": {"data_collection": True,
                                                   "webcam": True,
                                                   "data_collection_timeout": 900.0},
}


def normalize_bridge_url(value: str, default_port: int) -> str:
    """Accept a bare IP, IP:PORT or a full URL and return a full URL.

    The operator on the ProVoice machine knows one thing about the other
    machine: its address. Requiring "http://" and the port around it is a
    typo surface for no information gain, so all three spellings are accepted
    and normalized here rather than in three places downstream.
    """
    v = value.strip().rstrip("/")
    if "://" in v:
        # Fully specified: left exactly as given, port included. Appending the
        # LAN default to an https tunnel URL (which means 443) would break a
        # URL the operator got right.
        return v
    if ":" not in v:
        v = f"{v}:{default_port}"
    return "http://" + v


def apply_experiment_presets(parser, args) -> None:
    """Expand any --experiment-* preset into the plain flags it stands for.

    Runs BEFORE every validation and warning in main(), so a preset cannot
    smuggle a combination past a check that a hand-typed equivalent would hit.
    Two presets at once is an error rather than a silent merge: they describe
    different phases of the session, and merging them would produce a run that
    is neither.
    """
    chosen = [name for name in (*_CARLA_PRESETS, *_PROVOICE_PRESETS)
              if getattr(args, name, None) not in (None, False)]
    if len(chosen) > 1:
        parser.error("only one --experiment-* preset at a time; got: "
                     + ", ".join("--" + n.replace("_", "-") for n in chosen))

    args.provoice_only = False
    args.bridge_url = None
    if not chosen:
        return
    name = chosen[0]

    if name in _CARLA_PRESETS:
        overrides = _CARLA_PRESETS[name]
    else:
        overrides = _PROVOICE_PRESETS[name]
        args.provoice_only = True
        # Three sources, most specific first. The pinned constant is last and
        # is why this needs no argument on the rig: the CARLA machine's address
        # is a property of the pairing, both machines run THIS file, and the
        # presets on the other side pin the very same value. Requiring it here
        # only ever caught "forgot to type it" -- typing it WRONG failed
        # identically either way, ten seconds later.
        #
        # Which source won is printed, because the one way this defaulting
        # misleads is a repo out of sync between the two machines: then the
        # address is confidently wrong, and the printed line is what makes that
        # visible instead of mysterious.
        raw = getattr(args, name)
        source = "command line"
        if not raw:
            raw, source = os.environ.get("PV_BRIDGE_URL", ""), "PV_BRIDGE_URL"
        if not raw:
            raw, source = CARLA_MACHINE_IP, "the pinned rig address"
        args.bridge_url = normalize_bridge_url(raw, args.remote_port)
        print("[PRESET] bridge at %s (from %s)" % (args.bridge_url, source))

    if args.no_webcam:
        # Honoured rather than overridden: a preset expresses the usual rig,
        # and the operator looking at the actual machine outranks it.
        overrides = {k: v for k, v in overrides.items() if k != "webcam"}
        args.webcam = False

    applied = []
    for flag, value in overrides.items():
        if (isinstance(value, (str, int, float)) and not isinstance(value, bool)
                and getattr(args, flag) is not None):
            # A VALUE the operator typed beats the preset's. Switches cannot
            # collide this way -- a preset only ever turns one on, and the
            # operator asking for the same thing is not a disagreement -- but
            # an address or a timeout is a real choice, and overriding it for
            # one run must not need the source edited.
            print("[PRESET] keeping --%s=%s from the command line (preset "
                  "would have used %s)"
                  % (flag.replace("_", "-"), getattr(args, flag), value))
            continue
        setattr(args, flag, value)
        applied.append("--" + flag.replace("_", "-")
                       + ("" if value is True else " " + str(value)))
    print("[PRESET] --%s = %s" % (name.replace("_", "-"), " ".join(applied)))


def run_provoice_only(args) -> None:
    """ProVoice alone, against a bridge on the machine running CARLA.

    This machine has no CARLA server, no traffic and no Drive, so none of the
    startup sequence in main() applies: there is no vehicle id to wait for, no
    session id to mint (the bridge publishes the one Drive is already using)
    and no local vehicle-state bridge to supervise.

    ProVoice is NOT restarted if it dies. A calibration run restarted halfway
    would splice two partial baselines, which is why --restart-provoice already
    refuses to combine with --calibration-only; a data-collection run restarted
    here would silently re-adopt the ids and keep recording, hiding a crash
    that the operator needs to see, since the other machine keeps collecting
    labels either way.
    """
    mode = "calibration" if args.calibration_only else "data collection"
    print(f"[PROVOICE-ONLY] {mode} run; bridge at {args.bridge_url}")
    print(f"[PROVOICE-ONLY] participant and session ids come from "
          f"{args.bridge_url}/session — nothing is typed twice.")
    if args.calibration_only:
        # Worth stating up front: the baseline is filed under the participant
        # id, and ProVoice refuses to start without one for exactly this
        # reason (a 180 s measurement that lands nowhere).
        print("[PROVOICE-ONLY] ProVoice exits by itself once the baseline is "
              "stored; nothing is written to raw_data.jsonl during calibration.")

    pm = ProcessManager()
    cmd = build_provoice_cmd(None, args, None, remote_url=args.bridge_url,
                             ids_from_bridge=True)
    try:
        proc = pm.start(cmd, "PROVOICE", below_normal=True)
        while True:
            code = proc.poll()
            if code is None:
                time.sleep(1)
                continue
            if code == 0:
                print("[DONE] ProVoice finished cleanly"
                      + (" — baseline stored" if args.calibration_only else ""))
                raise SystemExit(0)
            print(f"[CRASH] PROVOICE exited with code {describe_exit_code(code)}")
            raise SystemExit(1)
    except KeyboardInterrupt:
        print("\n[EXIT] stopping ProVoice...")
    finally:
        pm.stop_all()
        print("[EXIT] experiment stopped")


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

        WHICH CLOCK CARLA IS ON depends on ``--sync``, and no code below should
        assume either. Both children are told, together, by this launcher.

        Without ``--sync`` (the default, and how every session up to now has
        run): nobody sets synchronous_mode, fixed_delta_seconds is never set, and
        the simulator advances on its own variable clock. Both Drive and
        fixed_npc_traffic.py sit in wait_for_tick().

        With ``--sync``: fixed_npc_traffic.py owns the clock. It sets
        synchronous_mode plus a fixed --delta step and ticks in a loop paced
        against the wall clock; Drive gets ``--sync --tm-port`` and paces itself
        to that clock with wait_for_tick(), ticking nothing and touching no world
        settings.

        That split is not a stylistic choice. CARLA's traffic manager runs inside
        the client process that created it, so the process that ticks must be the
        process that owns the traffic manager the NPCs are registered to -- here,
        the one that spawns them. src/drive/fixed_npc_traffic.py's module
        docstring has the full mechanism.

        Worth knowing because this docstring has now been wrong in both
        directions. It once claimed synchronous mode when the system was async,
        which sent a debugging session after an imaginary mismatch. The
        correction then hard-coded "async" as a permanent fact, and concluded
        that the sync mismatch had never existed -- but it had: --sync used to
        synchronise Drive's own traffic manager on port 8000, which no NPC is
        registered to, leaving the port-9000 one that drives them unticked. That
        is why every NPC froze, and it is fixed rather than merely documented.
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
    parser.add_argument("--data-collection-timeout", dest="data_collection_timeout",
                        type=float, default=None,
                        help="Hard wall-clock limit, in seconds, on a --data-collection "
                             "run. Once it elapses, ProVoice stops itself exactly as it "
                             "does at the end of --calibration-only, which (under "
                             "--remote) posts 'provoice_ended' and makes Drive show the "
                             "same end-of-session screen it shows after calibration. "
                             "Unset = unlimited, which is the default for a plain "
                             "--data-collection run; --experiment-data-collection-"
                             "provoice-remote sets it to 900 (15 min) unless given here.")
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
    parser.add_argument("--remote-bind", dest="remote_bind", default=None,
                        help="Interface the two bridges bind (--remote). Default "
                             "0.0.0.0, which accepts from every network this "
                             "machine is on; the --experiment-*-carla-remote "
                             "presets narrow it to the study link instead, so the "
                             "bridges are not listening on campus Wi-Fi at all. "
                             "127.0.0.1 would only serve a tunnel running here.")
    parser.add_argument("--status-port", dest="status_port", type=int, default=8081,
                        help="Port the REVERSE bridge listens on (--remote): "
                             "scripts/provoice_status_server.py, which receives "
                             "the remote ProVoice's started/ended signals. Its "
                             "URL is handed to ProVoice through the vehicle "
                             "bridge's /session, so it is never typed by hand. "
                             "Needs the same inbound firewall allowance as "
                             "--remote-port.")
    parser.add_argument("--remote-host", dest="remote_host", default=None,
                        help="This machine's address AS THE OTHER MACHINE SEES "
                             "IT, for the URLs published to it (--remote). "
                             "Default: whichever interface routes to the "
                             "internet — which is the WRONG one as soon as the "
                             "two machines are joined by a dedicated link: that "
                             "guess returns the Wi-Fi address while the pairing "
                             "runs over Ethernet. Set it to this machine's "
                             "Ethernet IP for a direct-cable setup.")
    parser.add_argument("--status-url", dest="status_url", default=None,
                        help="Address the ProVoice machine should post its "
                             "signals to, when this machine cannot work it out: "
                             "behind a tunnel (ngrok) the LAN address published "
                             "by default is unreachable from there, so expose "
                             "the status port too and give its public URL here. "
                             "On a plain LAN leave it unset.")
    parser.add_argument("--status-file", dest="status_file",
                        default="provoice_status.json",
                        help="File the reverse bridge publishes into and Drive "
                             "polls for those signals. Written atomically.")
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
    parser.add_argument("--allow-long-vehicles", dest="allow_long_vehicles",
                        action="store_true",
                        help="Put the three off-tracking vehicles back into the "
                             "NPC fleet (firetruck, fuso, carlacola). The sprinter "
                             "and ambulance are no longer among them -- they were "
                             "driven without off-tracking, so they stay in the "
                             "fleet regardless and serve as substitutes for the "
                             "three that do. Those three are replaced by default "
                             "because they "
                             "OFF-TRACK: the traffic manager steers a reference "
                             "point along the lane centreline and does not model "
                             "the body behind it, so on tight corners their tails "
                             "sweep through the parked cars at the kerb, block the "
                             "road, and the rest of the fleet piles into them. "
                             "Parked cars are scenery, not actors, so the traffic "
                             "manager cannot see them and no setting avoids this. "
                             "The substitution keeps the same number of vehicles "
                             "at the same spawn points, so traffic density -- part "
                             "of the driving task -- is unchanged.")
    parser.add_argument("--num-vehicles", dest="num_vehicles", type=int, default=0,
                        help="Spawn only N of the configured NPC vehicles, evenly "
                             "spaced through the list (0 = all of them; the fleet "
                             "is 15 at the time of writing). A "
                             "DIAGNOSTIC for --sync slow motion: this rig spends "
                             "~1.17 s of server work per SIMULATED second, which no "
                             "tick rate can outrun. Run once with --num-vehicles 3 "
                             "and compare the [SYNC] sim speed -- a big jump means "
                             "vehicle physics is the cost and the fleet size is the "
                             "knob; no change rules it out. If you end up keeping a "
                             "reduced fleet, fix it across all participants: the "
                             "amount of traffic is part of the driving task.")
    parser.add_argument("--run", dest="run", type=int, default=None,
                        choices=tuple(range(1, RUNS_PER_PARTICIPANT + 1)),
                        help="Which of this participant's %d data-collection runs "
                             "this is. Together with --participantid it SELECTS "
                             "THE TRAFFIC SCENARIO from TRAFFIC_SEED_PLAN at the "
                             "top of this file, so the counterbalanced design is "
                             "executed by the code rather than read off a sheet "
                             "and retyped. Required on --data-collection runs "
                             "unless --traffic-seed overrides the plan; refused "
                             "on any other run, where the scenario is always %d.\n"
                             "\n"
                             "Not recorded separately, and does not need to be: "
                             "no participant drives the same scenario twice, so "
                             "the run number is recoverable from the "
                             "participantid and traffic_seed already on every "
                             "logged frame."
                             % (RUNS_PER_PARTICIPANT, STUDY_TRAFFIC_SEED))
    parser.add_argument("--traffic-seed", dest="traffic_seed", type=int,
                        default=None,
                        help="OVERRIDE the planned traffic scenario. Normally "
                             "unset: on a --data-collection run the scenario "
                             "comes from --participantid and --run via "
                             "TRAFFIC_SEED_PLAN, and everything else runs "
                             "scenario %d. Use this for a pilot, an unplanned "
                             "participant, or a re-run that must not follow the "
                             "plan -- it is honoured and announced, so a session "
                             "that departed from the design says so in its log. "
                             "Naming the scenario by hand on every session is "
                             "exactly what the plan replaced, so reaching for "
                             "this routinely means the plan is wrong and should "
                             "be edited instead.\n"
                             "\n"
                             "REFUSED outside --data-collection. The calibration "
                             "and adaptation drives are pinned to scenario %d so "
                             "the study arms stay a fixed condition, comparable "
                             "across participants.\n"
                             "\n"
                             "Whatever scenario is used -- planned or overridden "
                             "-- the seed drives the traffic manager and the "
                             "per-vehicle draws in fixed_npc_traffic.py's "
                             "VEHICLE_TM_OVERRIDES, so one number identifies one "
                             "scenario: the same seed reproduces the same traffic "
                             "for every participant, a different one is a "
                             "different scenario from the same configuration. It "
                             "is carried to ProVoice and lands on every "
                             "raw_data.jsonl frame, which is the only record of "
                             "which scenario produced a session -- nothing else "
                             "in the data identifies it after the fact."
                             % (STUDY_TRAFFIC_SEED, STUDY_TRAFFIC_SEED))
    parser.add_argument("--substep-delta", dest="substep_delta", type=float,
                        default=0.01,
                        help="Maximum physics substep in seconds (default 0.01, "
                             "CARLA's own). THE LEVER ON SLOW MOTION under --sync, "
                             "and the non-obvious one: the server runs 1/this many "
                             "physics substeps per SIMULATED second no matter what "
                             "--delta is, so lowering the tick rate cannot buy back "
                             "real time but raising this can. 0.02 halves the "
                             "physics work. It trades against integration accuracy "
                             "-- coarse substeps are what made NPCs spin out -- so "
                             "raise it one step at a time and watch both the [SYNC] "
                             "sim-speed line and the traffic. Fix it across all "
                             "participants once chosen. Ignored without --sync.")
    parser.add_argument("--render-scale", dest="render_scale", type=float, default=1.0,
                        help="Render the drive camera at this fraction of the screen "
                             "size and upscale it (default 1.0 = unchanged). THE "
                             "LEVER TO REACH A HIGHER --delta: the CARLA server's "
                             "frame rate caps the whole simulation under --sync, and "
                             "at fullscreen most of that frame is the drive camera. "
                             "0.7 renders about half the pixels, 0.5 about a quarter. "
                             "The image gets softer, not smaller. Read the effect off "
                             "the [SYNC] 'ceiling ~N Hz' line, then set --delta to "
                             "match. Fix it across all participants once chosen.")
    parser.add_argument("--speed", action=argparse.BooleanOptionalAction, default=True,
                        help="Show the vehicle speed in km/h on screen, bottom-right. "
                             "A single large readout, not the F1 debug panel, so it "
                             "is safe to have in front of a participant. ON by "
                             "default, and it must be either on or off identically "
                             "for EVERY participant and BOTH study arms, since a "
                             "precise speed instrument is something the driver can "
                             "regulate against and plausibly shifts how they judge "
                             "the assistant's autonomy. Pass --no-speed to turn it "
                             "off.")
    parser.add_argument("--ambient-gain", dest="ambient_gain", type=float,
                        default=None,
                        help="Gain of the synthesised engine/road cabin sound, "
                             "0-1. ON by default (0.35, defined once in "
                             "src/drive/ambience.py); pass 0 for a silent rig, "
                             "which is what CARLA gives on its own -- it has no "
                             "audio in any version. Applies only to runs that "
                             "start Drive, i.e. the CARLA machine; the "
                             "--experiment-*-provoice-remote presets start no "
                             "Drive and are unaffected. It is an arousal "
                             "manipulation whether or not it is meant as one, "
                             "and hr_delta / rr_delta are model inputs, so keep "
                             "it identical for EVERY participant and BOTH arms. "
                             "Not a level -- set the amplifier once, measure "
                             "dB(A) at the driver's head, report that. Logged "
                             "per label row as ambient_gain.")
    parser.add_argument("--ambient-seed", dest="ambient_seed", type=int,
                        default=None,
                        help="Seed for the ambience noise layers (default 0). "
                             "Gain plus seed reproduce exactly what a "
                             "participant heard; both are logged per label row. "
                             "No reason to vary it across participants.")
    condition_group = parser.add_mutually_exclusive_group()
    condition_group.add_argument("--condition-sun-rain", dest="condition_sun_rain",
                                 action="store_true",
                                 help="Scripted weather: precipitation starts at 80%% "
                                      "and ramps linearly down to 0%% over 10 minutes, "
                                      "timed from when the driver starts driving (not "
                                      "process launch). Forwarded to Drive, which owns "
                                      "the CARLA weather; ProVoice picks up the change "
                                      "for free since it reads live precipitation off "
                                      "the same CARLA world every frame. Mutually "
                                      "exclusive with --condition-rain-sun.")
    condition_group.add_argument("--condition-rain-sun", dest="condition_rain_sun",
                                 action="store_true",
                                 help="Scripted weather: sunny (0%% precipitation) for "
                                      "the first 5 minutes of driving, then ramps "
                                      "linearly up to 80%% over the following 10 "
                                      "minutes. Mutually exclusive with "
                                      "--condition-sun-rain.")
    parser.add_argument("--test", action="store_true",
                        help="REHEARSAL for the CARLA-machine data-collection run: "
                             "the LoA popups fire from the very start instead of "
                             "waiting for ProVoice on the other machine to load its "
                             "models and report its first frame. Everything else is "
                             "the real run -- NPC traffic, fullscreen, the genuine "
                             "20 s interval, random functions -- so it exercises the "
                             "popup flow, the wheel bindings and the label file on "
                             "the actual rig without needing the ProVoice machine up. "
                             "THE LABELS IT PRODUCES ARE NOT PARTICIPANT DATA: the "
                             "windows cover driving no ProVoice was recording. Use "
                             "with --experiment-data-collection-carla-remote.")
    parser.add_argument("--sync", action=argparse.BooleanOptionalAction, default=None,
                        help="SYNCHRONOUS mode: run the simulation on a fixed time "
                             "step (--delta) driven by the NPC-traffic process, which "
                             "paces it against the wall clock. ON by default (except "
                             "under --test-popup, which starts no NPC_TRAFFIC to own "
                             "the clock). Fixes the two things the free-running mode "
                             "cannot: physics is integrated at a constant step, so "
                             "NPCs stop spinning out under machine load, and the "
                             "traffic manager's seed actually makes the traffic "
                             "identical for every participant. Costs real-time "
                             "fidelity IF the rig cannot hold the step -- the run "
                             "then goes into slow motion rather than dropping physics "
                             "accuracy, and NPC_TRAFFIC prints the achieved rate "
                             "every 30 s so you can tell. Sets up both children "
                             "correctly; do not pass --sync to either by hand. Pass "
                             "--no-sync to fall back to the free-running clock.")
    parser.add_argument("--delta", type=float, default=0.05,
                        help="Fixed time step in seconds for --sync (default 0.05 = "
                             "20 Hz). This is a DEMAND on the CARLA server, not a "
                             "quality setting: each tick asks for a full rendered "
                             "frame, and if the server cannot keep up the clock "
                             "falls behind and EVERYTHING runs in slow motion, the "
                             "participant's own car included. 40 Hz was tried here "
                             "and did exactly that, so do not raise it without "
                             "evidence -- the [SYNC] lines from NPC_TRAFFIC print "
                             "the measured server frame time and the rate it can "
                             "actually sustain. KEEP THIS FIXED ACROSS ALL "
                             "PARTICIPANTS AND BOTH STUDY ARMS, for the same reason "
                             "--decision-hz is fixed: it changes the simulation the "
                             "driver responds to. Ignored without --sync.")
    parser.add_argument("--test-popup", dest="test_popup", action="store_true",
                        help="Teaching mode: open the simulator UI and show LoA "
                             "selection popups straight away, so the participant can "
                             "practise the control. Each practice window holds the "
                             "same TWO consecutive prompts about two different "
                             "functions as a real one. No traffic, no ProVoice, and "
                             "the practice selections are not logged.")
    parser.add_argument("--no-popup", dest="no_popup", action="store_true",
                        help="Suppress Drive's LoA selection popups for the whole "
                             "session: no scene freezes and no user labels. ProVoice "
                             "still runs and logs its own decisions.")
    popup_input = parser.add_mutually_exclusive_group()
    popup_input.add_argument("--wheel-input", dest="popup_input", action="store_const",
                             const="wheel",
                             help="Answer the LoA popups from the steering wheel "
                                  "instead of the keyboard: the paddles move the "
                                  "cursor, the front button ticks the level under it, "
                                  "the CONFIRM row submits.")
    popup_input.add_argument("--keyboard-input", dest="popup_input", action="store_const",
                             const="keyboard",
                             help="Answer the LoA popups from the keyboard: number keys "
                                  "1-5 tick a level, the same number again unticks it, "
                                  "N ticks INVALID WINDOW instead, and ENTER commits "
                                  "whatever is ticked. The popup ignores the wheel, "
                                  "which still steers. THE DEFAULT - the flag only "
                                  "states it explicitly.")
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
    parser.add_argument("--webcam", action="store_true",
                        help="Tell ProVoice to use camera index 1 — the external USB "
                             "webcam, where index 0 is the built-in one. ProVoice "
                             "refuses to start if index 1 delivers no frame. Without "
                             "it ProVoice always takes index 0. Keep this setting FIXED "
                             "across all participants: the two cameras differ in "
                             "framing and frame rate, and the achieved frame rate is "
                             "the rPPG sampling rate, so switching mid-study changes "
                             "the HR/RR scale. The two ProVoice-machine presets set "
                             "this for you.")
    parser.add_argument("--no-webcam", dest="no_webcam", action="store_true",
                        help="Cancel the --webcam that the ProVoice-machine presets "
                             "set, for a machine whose only camera is the built-in "
                             "one. Use it for BOTH phases or neither: a participant "
                             "calibrated on one camera and recorded on the other has "
                             "an HR/RR baseline measured from a different signal.")

    # --- Protocol presets (see the table in the module docstring) ------------
    presets = parser.add_argument_group(
        "experiment presets",
        "One flag per phase of the study protocol. Each is exactly equivalent "
        "to the flags listed in its help text, applied before every check "
        "below, and any other flag can still be added alongside.")
    presets.add_argument("--experiment-popup", dest="experiment_popup",
                         action="store_true",
                         help="Phase 0, teaching the LoA control. "
                              "= --test-popup --fullscreen")
    presets.add_argument("--experiment-adaptation", dest="experiment_adaptation",
                         action="store_true",
                         help="Phase 1, driver adaptation. "
                              "= --test-drive --no-popup --fullscreen")
    presets.add_argument("--experiment-calibration-carla-remote",
                         dest="experiment_calibration_carla_remote",
                         action="store_true",
                         help="Phase 2 on the CARLA machine. = --fixed --remote "
                              "--no-popup --fullscreen. Give --participantid "
                              "here; the ProVoice machine reads it off the "
                              "bridge.")
    presets.add_argument("--experiment-data-collection-carla-remote",
                         dest="experiment_data_collection_carla_remote",
                         action="store_true",
                         help="Phase 3 on the CARLA machine. = --remote "
                              "--data-collection --random-function --fullscreen. "
                              "Give --participantid here.")
    presets.add_argument("--experiment-calibration-provoice-remote",
                         dest="experiment_calibration_provoice_remote",
                         nargs="?", const="", default=None, metavar="BRIDGE",
                         help="Phase 2 on the ProVoice machine: start ProVoice "
                              "ALONE in --calibration-only mode against the "
                              "bridge at BRIDGE (bare IP, IP:PORT or URL). "
                              "Defaults to PV_BRIDGE_URL, then to the pinned rig "
                              "address, so on the rig it needs no argument. "
                              "Participant and session ids come from "
                              "BRIDGE/session. Implies --webcam (the "
                              "driver-facing camera on this machine); "
                              "--no-webcam cancels it.")
    presets.add_argument("--experiment-data-collection-provoice-remote",
                         dest="experiment_data_collection_provoice_remote",
                         nargs="?", const="", default=None, metavar="BRIDGE",
                         help="Phase 3 on the ProVoice machine: start ProVoice "
                              "ALONE in --data-collection mode against BRIDGE. "
                              "Same id handling as the calibration preset, and "
                              "the same --webcam, which is what keeps the two "
                              "phases on ONE camera.")

    args = parser.parse_args()

    # Expanded first, so every check and warning below judges the flags the
    # preset stands for rather than the preset itself.
    apply_experiment_presets(parser, args)
    # Left as None until now so a preset can tell "nobody said" from an
    # explicit --remote-bind and only fill in the former.
    if args.remote_bind is None:
        args.remote_bind = "0.0.0.0"

    # --sync default resolved here rather than at the argparse default itself:
    # it is ON for a normal launch, but --test-popup starts no NPC_TRAFFIC, so
    # nothing would own the clock and the explicit-vs-default distinction
    # matters -- an operator who types --sync --test-popup by hand still hits
    # the parser.error below, only an unset --sync is auto-resolved.
    if args.sync is None:
        args.sync = not args.test_popup

    if args.provoice_only:
        # Nothing else in main() applies on the ProVoice machine: no CARLA
        # server to wait for, no Drive, no vehicle id, no local bridge. The
        # checks below are all about that machinery, so they are not merely
        # unnecessary here, they would be reasoning about processes that do not
        # exist.
        #
        # Which is exactly why this one check has to be HERE rather than with
        # its siblings below: this branch returns, so anything about --test
        # placed down there would never run on the machine where --test is
        # wrong. It was, and the flag silently started a real ProVoice instead.
        if args.test:
            parser.error("--test rehearses the LoA popups, which Drive shows -- "
                         "and the --experiment-*-provoice-remote presets start "
                         "ProVoice alone, with no Drive on this machine. Run "
                         "--test on the CARLA machine, with "
                         "--experiment-data-collection-carla-remote.")
        if args.functionname is None:
            args.functionname = DEFAULT_FUNCTIONNAME
        run_provoice_only(args)
        return

    # The traffic scenario, resolved once and written back onto args so every
    # child below is handed a plain integer and none of them has to know a plan
    # exists. Fatal on a bad combination -- see resolve_traffic_seed for which
    # combinations are bad and why.
    _validate_traffic_plan()
    args.traffic_seed = resolve_traffic_seed(args, parser)

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
    if args.remote and args.test_popup:
        parser.error("--remote and --test-popup contradict each other: "
                     "--test-popup exists to run without ProVoice at all "
                     "(and without traffic), --remote to run it elsewhere.")
    # --remote --test-drive is REDUNDANT, not contradictory: both say "no
    # ProVoice on this machine". Accepted rather than rejected because an
    # operator mid-session should get their run, not a usage error, over a
    # flag that changes nothing.
    if args.remote and args.test_drive:
        print("[NOTE] --test-drive with --remote does nothing: --remote already "
              "starts no ProVoice on this machine.")
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
    if args.webcam and (args.test_drive or args.test_popup):
        parser.error("--webcam only affects ProVoice, which %s does not start."
                     % ("--test-drive" if args.test_drive else "--test-popup"))
    if args.provoice_no_carla:
        # Loud, because the run looks completely normal and silently produces
        # rows whose vehicle columns are all defaults.
        print("[WARN] --provoice-no-carla: ProVoice will make NO CARLA calls. "
              "speed/steer/brake/junction and every other vehicle field stay at "
              "their defaults, so THIS RUN IS NOT USABLE PARTICIPANT DATA. It is "
              "a diagnostic for the heap corruption only.")
    # (--test on a ProVoice-only preset is rejected earlier, before that branch
    # returns.)
    if args.test and args.no_popup:
        parser.error("--test and --no-popup contradict each other: --test exists "
                     "to fire the popups early, --no-popup suppresses them "
                     "entirely. Note --experiment-calibration-carla-remote "
                     "implies --no-popup, so --test does not apply to it.")
    if args.test and args.test_popup:
        print("[WARN] --test adds nothing to --test-popup: that mode already "
              "opens popups immediately and never waits for ProVoice.")
    if args.sync and args.test_popup:
        parser.error("--sync and --test-popup are incompatible: --test-popup "
                     "does not start NPC_TRAFFIC, and that is the process that "
                     "drives the clock in synchronous mode. Drive would sit "
                     "waiting on a clock nobody advances. Teach the control "
                     "without --sync -- the scene is frozen for every popup "
                     "anyway, so the time step changes nothing there.")
    if args.delta != parser.get_default("delta") and not args.sync:
        print("[WARN] --delta only applies with --sync; the server free-runs on "
              "its own variable step without it, so this value is ignored.")
    if args.test_popup and args.no_popup:
        parser.error("--test-popup and --no-popup contradict each other: one is "
                     "nothing but popups, the other suppresses them.")
    if args.test_popup and args.calibration_only:
        parser.error("--test-popup and --calibration-only are separate modes: the "
                     "first teaches the control, the second records a baseline.")
    if args.no_popup and not (args.calibration_only or args.test_drive
                              or args.remote):
        # A normal session exists to collect the labels, so suppressing the
        # popups there would produce a run with nothing to show for it.
        #
        # --remote counts because the calibration phase of a two-machine run is
        # exactly this: Drive here with no popups, ProVoice measuring the
        # baseline over there. --calibration-only is a statement about a
        # ProVoice this launcher starts, and under --remote it starts none, so
        # it is not available to say it with.
        parser.error("--no-popup requires --calibration-only, --test-drive or "
                     "--remote: a normal session has to collect the LoA labels.")
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
    # OTHER machine — so the wait used to be skipped there, because a watcher
    # on a file nobody writes can only run out its timeout.
    #
    # The status bridge is what changed that: the remote ProVoice now POSTs
    # collection_started, Drive reads it out of the status file, and the wait
    # means the same thing on both sides again. So --remote takes the SAME
    # default as a local run.
    if args.popup_wait_timeout is None:
        args.popup_wait_timeout = DEFAULT_POPUP_WAIT_TIMEOUT
    if args.test:
        # Applied AFTER the default above so it wins, and stated out loud because
        # it is the whole point of the flag: the wait exists so no window is
        # labelled without driver-state data behind it, and a rehearsal has no
        # ProVoice to wait for. Drive also gets --popup-immediate, so the first
        # window opens at once rather than one interval in.
        if args.popup_wait_timeout:
            print("[TEST] popups will NOT wait for ProVoice "
                  "(--popup-wait-timeout %g -> 0)" % args.popup_wait_timeout)
        args.popup_wait_timeout = 0.0
        print("[TEST] rehearsal run: first LoA window opens immediately, then "
              "every 20 s as normal. The labels are NOT participant data.")
    if args.remote and args.no_popup:
        # Deliberately stated as a fact with both readings rather than as a
        # warning: this launcher cannot tell them apart. Whether ProVoice on
        # the other machine is calibrating (where a label-free drive is the
        # whole point) or collecting data (where this would waste the session)
        # is decided over there, by a flag this process never sees.
        print("[NOTE] --remote with --no-popup: no LoA labels will be recorded "
              "here. That is the calibration phase; for a data-collection run "
              "it means the session collects nothing, so drop --no-popup.")
    if args.remote and args.calibration_only:
        # The status bridge reports the remote ProVoice's exit to DRIVE, which
        # ends the drive and shows the end screen. This launcher process still
        # keeps CARLA, the traffic and the bridges up, so it is the operator
        # who decides the phase is over.
        print("[NOTE] --remote --calibration-only: the remote ProVoice exits by "
              "itself once the baseline is stored, and Drive will end the drive "
              "when it reports that. Stop this launcher with Ctrl-C afterwards.")

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
                [sys.executable, "-m", "src.drive.fixed_npc_traffic",
                 "--host", args.host, "--port", str(args.port),
                 "--tm-port", str(TM_PORT),
                 *(["--num-vehicles", str(args.num_vehicles)]
                   if args.num_vehicles else []),
                 *(["--allow-long-vehicles"] if args.allow_long_vehicles else []),
                 # THE SCENARIO IDENTITY for this session. Always passed
                 # explicitly, never left to the child's default, so the value
                 # that reaches the traffic manager is the same one that
                 # reaches ProVoice and the collected data -- a default agreed
                 # in two places is a default that eventually disagrees.
                 "--traffic-seed", str(args.traffic_seed),
                 # This child owns the simulation clock under --sync. It is
                 # started BEFORE Drive, which is what the arrangement needs:
                 # the clock has to be running by the time Drive waits on it.
                 "--pause-file", CLOCK_PAUSE_FILE,
                 *(["--sync", "--delta", str(args.delta),
                    "--substep-delta", str(args.substep_delta)]
                   if args.sync else [])],
                "NPC_TRAFFIC"
            )

            time.sleep(6)

        # =========================
        # START THE REVERSE (STATUS) BRIDGE
        # =========================
        # Before Drive, because Drive polls the file this publishes and should
        # find a fresh record for THIS session there from its first tick.
        # Scoped to the session id for the same reason the file is republished
        # empty: a status file left by the previous participant must not be
        # able to end this drive. Supervised — the signals are one-shot, so a
        # bridge that died and stayed dead would silently cost the end screen.
        status_url = None
        if args.remote:
            try:
                Path(args.status_file).unlink()
                print(f"[CLEANUP] removed stale {args.status_file}")
            except FileNotFoundError:
                pass
            except OSError as e:
                print(f"[WARN] could not remove {args.status_file}: {e}")

            status_cmd = [
                sys.executable,
                "scripts/provoice_status_server.py",
                "--bind", args.remote_bind,
                "--port", str(args.status_port),
                "--out", args.status_file,
                "--session-id", session,
            ]
            pm.start(status_cmd, "STATUS_SERVER", restart=True)
            # A GUESS, and published as one: outbound_ip() reports the interface
            # that routes to the internet from HERE, which is not necessarily
            # the address the ProVoice machine can reach — a second NIC, or a
            # tunnel in front of the vehicle bridge, breaks it. ProVoice probes
            # this before trusting it and falls back to the host it actually
            # reached the vehicle bridge on, so a wrong guess costs a line in
            # its log rather than the end-of-session signal.
            status_url = (args.status_url
                          or f"http://{advertised_host(args)}:{args.status_port}")

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
                # Published at /session so the ProVoice on the other machine can
                # adopt this session's identity from the URL it already has,
                # instead of depending on a human retyping it. The printed
                # command below still carries both, and ProVoice cross-checks
                # them against these — a disagreement is an operator error that
                # must surface at startup, not in the logs afterwards.
                "--session-id", session,
                "--participantid", args.participantid,
                # Same channel, same reason: the seed is known only on this
                # machine, and it is the only record of which traffic scenario
                # produced the frames ProVoice is about to write.
                "--traffic-seed", str(args.traffic_seed),
                # And the address of the reverse bridge, so the whole pairing
                # follows from the ONE url the ProVoice machine is given.
                *(["--status-url", status_url] if status_url else []),
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
            remote_url = f"http://{advertised_host(args)}:{args.remote_port}"
            remote_cmd = build_provoice_cmd(session, args, vehicle_id,
                                            remote_url=remote_url)
            # "uv run python", not this machine's interpreter path, which means
            # nothing on the other machine.
            remote_cmd[0:1] = ["uv", "run", "python"]
            print()
            print("=" * 72)
            print(f"[REMOTE] vehicle state is served at {remote_url}/")
            print(f"[REMOTE] health/keep-alive counters: {remote_url}/health")
            print(f"[REMOTE] ProVoice's start/end signals come back to "
                  f"{status_url}/ (status: {status_url}/status), and Drive here "
                  f"acts on them: the LoA windows wait for the first, the drive "
                  f"ends on the second.")
            print("[REMOTE] ProVoice is NOT started here. On the OTHER machine, "
                  "in the project root, run:")
            print()
            print("    " + shell_quote(remote_cmd))
            print()
            print("[REMOTE] (drop the 'uv run' if that machine's venv is already "
                  "activated)")
            print(f"[REMOTE] session_id={session} participantid="
                  f"{args.participantid or '(unset)'} are also published at "
                  f"{remote_url}/session, and ProVoice adopts them from there "
                  f"when they are absent from its command line. Pasting the line "
                  f"above is still the safe path — it also pins the model, the "
                  f"environment and the function, which the bridge does not "
                  f"carry. What the bridge removes is the failure where those two "
                  f"ids are retyped and quietly disagree.")
            print("[REMOTE] If the poll never connects, it is almost always the "
                  "Windows firewall on THIS machine: inbound TCP "
                  f"{args.remote_port} AND {args.status_port} have to be "
                  f"allowed (state out, signals back).")
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
                    elif name == "STATUS_SERVER":
                        print("       ProVoice's start/end signals cannot arrive "
                              "until it is back. Signals already received "
                              "survive (the restarted server reloads the status "
                              "file for this session), but one SENT during the "
                              "gap is lost: the sender retries for seconds, not "
                              "minutes. If ProVoice ends in that window, end the "
                              "drive here by hand.")
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