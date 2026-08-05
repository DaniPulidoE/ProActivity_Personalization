from __future__ import annotations
from functools import lru_cache
from typing import Dict, Set

FCD_NAMES = [
    "Safety Risk", "Increased Safety", "Relevance", "Magicality",
    "Privacy", "Trust", "Time Consumption", "Repetitiveness",
    "Situational Context", "Social Risk", "Urgency", "Complexity",
]

def _canon(s: str) -> str:
    return "".join(ch.lower() for ch in s if ch.isalnum())

FUNCTIONS = [
    ("adjustseatpositioning",      "Adjust seat positioning"),
    ("adjustincartemperature",     "Adjust in-car temperature"),
    ("sendatextmessage",           "Send a text message"),
    ("startaphonecall",            "Start a phone call"),
    ("navigationcontrol",          "Navigation control"),
    ("changedrivingmode",          "Change driving mode"),
    ("selectparkingspace",         "Select parking space"),
    ("overtakevehicleahead",       "Overtake vehicle ahead"),
    ("startdriving",               "Start driving"),
    ("provideweatherupdate",       "Provide weather update"),
    ("adaptambientlight",          "Adapt ambient light"),
    ("changesong",                 "Change song"),
    ("providetrafficnews",         "Provide traffic news"),
    ("startamovie",                "Start a movie"),
]

_BASE_NUMS = {
    1:  [2,3,3,4,1,2,4,2,2,2,2,2],   # Adjust seat positioning
    2:  [2,3,3,2,2,2,4,4,2,2,2,2],   # Adjust in-car temperature
    3:  [2,3,2,4,4,3,4,4,3,4,2,3],   # Send a text message
    4:  [2,3,2,2,4,2,4,3,4,3,2,2],   # Start a phone call
    5:  [2,3,4,2,2,3,4,3,3,2,2,3],   # Navigation control
    6:  [2,5,4,4,2,4,3,3,4,3,4,3],   # Change driving mode
    7:  [3,3,2,5,2,4,3,2,3,3,2,4],   # Select parking space
    8:  [4,3,2,5,2,5,4,3,4,3,2,3],   # Overtake vehicle ahead
    9:  [3,3,4,4,2,4,4,3,4,2,3,3],   # Start driving
    10: [2,4,4,2,2,4,4,4,2,2,4,2],   # Provide weather update
    11: [2,2,1,3,2,2,5,4,2,2,2,2],   # Adapt ambient light
    12: [2,2,3,2,2,2,4,4,3,2,2,2],   # Change song
    13: [1,3,3,1,1,1,4,4,1,1,3,1],   # Provide traffic news
    14: [5,1,1,2,1,4,3,2,2,3,1,1],   # Start a movie
}

BASE_FCD_CONFIG: Dict[str, Dict[str, int]] = {}
for idx, (key, _pretty) in enumerate(FUNCTIONS, start=1):
    nums = _BASE_NUMS[idx]
    BASE_FCD_CONFIG[key] = {name: int(nums[i]) for i, name in enumerate(FCD_NAMES)}

_ALIASES = {key: key for key, _ in FUNCTIONS}
for key, pretty in FUNCTIONS:
    _ALIASES[_canon(pretty)] = key
    _ALIASES[_canon(pretty.replace(" ", "_"))] = key
    _ALIASES[_canon(pretty.replace("-", " "))] = key

# Sentinel key for a name that matches no known function. Deliberately NOT a
# member of BASE_FCD_CONFIG: get_fcd_for_function() therefore falls through to
# NEUTRAL_FCD for it, and the string itself travels into the logs (the `profile`
# column of decisions.csv, and any FCD dict written to raw_data.jsonl is all-3s)
# so an unresolved name is identifiable after the fact instead of being
# indistinguishable from a real 'Adjust seat positioning' run.
#
# Spelled unlike the other keys (space + capital, vs. 'adjustseatpositioning')
# precisely so it stands out when scanning a log.
UNKNOWN_FUNCTION_KEY = "Unknown function"

# Mid-scale FCD for an unresolved function: no dimension asserts anything.
NEUTRAL_FCD: Dict[str, int] = {name: 3 for name in FCD_NAMES}

# Registering the sentinel makes resolve_function_key IDEMPOTENT and silent on
# its own output. decision_engine._loa0_result() re-resolves a key it was already
# given, so without this a single bad name would warn twice — once for the
# original typo and once for 'Unknown function' itself.
_ALIASES[_canon(UNKNOWN_FUNCTION_KEY)] = UNKNOWN_FUNCTION_KEY

# Canonicalized names that failed to resolve, so each distinct one is reported
# EXACTLY ONCE. Deduplication is not a nicety here: resolve_function_key runs per
# FRAME (encode_frame -> get_fcd_for_function), i.e. ~200x per decision at 4 Hz
# and once per row over the whole training set, so an unguarded print would both
# flood the log and dominate the hot path.
_unknown_names: Set[str] = set()


@lru_cache(maxsize=512)
def _resolve_cached(name: str) -> str:
    """Cached body of :func:`resolve_function_key`; see it for semantics.

    Keyed on the RAW name so a hit skips ``_canon`` too — that character-by-
    character rebuild is 1.56 us of the 1.63 us this used to cost, and it ran
    once per FRAME. The set of distinct names a run ever sees is ~14 plus
    typos, so this is effectively a permanent table.

    Caching a function with a side effect (the warning) is safe here only
    because the side effect is already once-per-name by construction: the body
    runs on a miss, and ``_unknown_names`` still guards the warning if an
    eviction ever makes it run twice for the same name.
    """
    ck = _canon(name)
    key = _ALIASES.get(ck)
    if key is None:
        if ck not in _unknown_names:
            _unknown_names.add(ck)
            if ck:
                print(f"[fcd][warn] unknown function name {name!r} -> "
                      f"{UNKNOWN_FUNCTION_KEY!r} with a neutral FCD vector "
                      f"(all 3s). Check --functionname / the label CSV against "
                      f"fcd_config.FUNCTIONS.")
            else:
                print(f"[fcd][warn] empty/missing function name -> "
                      f"{UNKNOWN_FUNCTION_KEY!r} with a neutral FCD vector "
                      f"(all 3s).")
        return UNKNOWN_FUNCTION_KEY
    return key


def resolve_function_key(name: str) -> str:
    """Canonical key for a function name, or ``UNKNOWN_FUNCTION_KEY``.

    An unresolved name no longer masquerades as the first table entry. It
    returns the sentinel, which (a) carries a neutral FCD vector through
    :func:`get_fcd_for_function` and (b) is visible in the logs, so a run
    against a mistyped ``--functionname`` is distinguishable after the fact from
    a genuine 'Adjust seat positioning' run.

    Idempotent: resolving the sentinel returns the sentinel, without warning.

    Thin coercing wrapper over the cached body, which pins the cache key to a
    plain ``str``. Falsy input collapses to ``""`` exactly as the previous
    ``name or ""`` did.

    DELIBERATE BEHAVIOUR CHANGE: a non-string name (e.g. a JSONL row carrying
    ``"functionname": 5``) used to raise ``TypeError`` out of ``_canon``, which
    escaped ``decide()`` — ``_loa0_result`` re-resolves the same bad value, so
    even the error path re-raised — and killed the whole decision tick. It now
    stringifies, fails to resolve, warns once, and yields the neutral FCD, which
    is the same graceful-and-visible degradation an unrecognised name gets.
    """
    return _resolve_cached(str(name) if name else "")


def get_fcd_for_function(name: str) -> Dict[str, int]:
    """FCD vector for a function; NEUTRAL_FCD (all 3s) if the name is unknown.

    The default arm is live, not defensive padding: UNKNOWN_FUNCTION_KEY is
    deliberately absent from BASE_FCD_CONFIG, so it is the only thing that maps
    an unresolved name to a neutral vector. Do not "simplify" it away.
    """
    return BASE_FCD_CONFIG.get(resolve_function_key(name), NEUTRAL_FCD).copy()

def adjust_fcd_by_state(base_fcd: Dict[str, int], _state=None) -> Dict[str, int]:
    def _clamp(v: int) -> int: return max(1, min(5, int(v)))
    out = {}
    for k in FCD_NAMES:
        out[k] = _clamp(base_fcd.get(k, 3))
    return out
