"""Driver-fold definitions — the ONE place the study's splits are decided.

Both the hyperparameter sweep and the LODO runner read their splits from here,
so a fold cannot mean one thing in stage 1 and another in stage 2.

WHY THE VALIDATION PAIRS LOOK THE WAY THEY DO
---------------------------------------------
``start_experiment.TRAFFIC_SEED_PLAN`` assigns each driver a PAIR of traffic
scenarios, in blocks of four:

    block 1: 001 002 003 004      block 2: 005 006 007 008
    block 3: 009 010 011 012

Each block runs the same set of scenarios through a different rotation, so the
block is the counterbalancing unit. Pairing validation folds sequentially
(001+002, 003+004, ...) would therefore put both drivers of a fold inside ONE
rotation, and every configuration would be validated against a signal carrying
that rotation's idiosyncrasies. Pairing ACROSS blocks spreads each fold over
different rotations instead.

The pairing is fixed and explicit rather than seeded-random: it has to be
identical across all 6 configurations and all 5 seeds, or the comparison between
configurations is confounded by which drivers each was validated on.
"""
from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

# The 12 population-study drivers, in recruitment order.
ALL_PIDS: Tuple[str, ...] = tuple(f"{i:03d}" for i in range(1, 13))

# Counterbalancing blocks from start_experiment.TRAFFIC_SEED_PLAN.
SCENARIO_BLOCKS: Tuple[Tuple[str, ...], ...] = (
    ("001", "002", "003", "004"),
    ("005", "006", "007", "008"),
    ("009", "010", "011", "012"),
)

# 6 validation folds of 2 drivers, covering all 12 exactly once. Folds 0-3 pair
# block 1 with block 2; block 3 has no third partner block, so its four drivers
# pair within themselves (4 and 5) — unavoidable with 3 blocks and 6 folds, and
# noted here so it is a known property rather than an accident.
VALIDATION_FOLDS: Tuple[Tuple[str, str], ...] = (
    ("001", "005"),
    ("002", "006"),
    ("003", "007"),
    ("004", "008"),
    ("009", "011"),
    ("010", "012"),
)


def _check() -> None:
    flat = [p for fold in VALIDATION_FOLDS for p in fold]
    if sorted(flat) != sorted(ALL_PIDS):
        raise AssertionError(
            f"VALIDATION_FOLDS must cover each of {ALL_PIDS} exactly once, got {sorted(flat)}")
    blocks_of: Dict[str, int] = {p: i for i, blk in enumerate(SCENARIO_BLOCKS) for p in blk}
    same = [f for f in VALIDATION_FOLDS if blocks_of[f[0]] == blocks_of[f[1]]]
    if len(same) > 2:
        raise AssertionError(
            f"at most the two block-3 folds may pair within a block, got {same}")


_check()


def train_pids_for_validation_fold(fold: Sequence[str]) -> List[str]:
    """The 10 drivers a config is TRAINED on when ``fold`` is the validation pair."""
    held = set(fold)
    return [p for p in ALL_PIDS if p not in held]


def lodo_folds() -> List[Tuple[str, List[str]]]:
    """``(test_pid, train_pids)`` for each of the 12 leave-one-driver-out folds.

    The test driver is absent from ``train_pids`` — that absence is the entire
    guarantee the LODO estimate rests on, and it is asserted at the call site in
    ``run_lodo_population`` rather than trusted here.
    """
    return [(p, [q for q in ALL_PIDS if q != p]) for p in ALL_PIDS]
