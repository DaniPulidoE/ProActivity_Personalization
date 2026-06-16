from __future__ import annotations
from typing import Dict

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

def resolve_function_key(name: str) -> str:
    ck = _canon(name or "")
    return _ALIASES.get(ck, FUNCTIONS[0][0])  

def get_fcd_for_function(name: str) -> Dict[str, int]:
    return BASE_FCD_CONFIG.get(resolve_function_key(name), {k: 3 for k in FCD_NAMES}).copy()

def adjust_fcd_by_state(base_fcd: Dict[str, int], _state=None) -> Dict[str, int]:
    def _clamp(v: int) -> int: return max(1, min(5, int(v)))
    out = {}
    for k in FCD_NAMES:
        out[k] = _clamp(base_fcd.get(k, 3))
    return out
