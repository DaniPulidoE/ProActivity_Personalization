# Steering Wheel Support

`src/drive/drive_improved.py` drives the ego vehicle from a steering wheel when
one is attached, and falls back to the keyboard when it is not. The wheel also
answers the Level-of-Proactivity (LoA) prompt, so a participant never has to
reach for the keyboard mid-drive.

Nothing needs to be enabled: the wheel is detected at startup. Pass
`--no-wheel` to ignore it and force keyboard control.

---

## 1. Detection and axis layout

At startup the first attached joystick is bound and the axis layout is chosen
from the axis count, so the same build works with the wheel in either mode:

| Axes reported | Mode | Layout |
|---|---|---|
| 3 or more | **Native** | separate steer / throttle / brake axes, pedals resting at `+1.0` |
| exactly 2 | **Compatibility** | steering on axis 0, throttle and brake **summed** onto axis 1 |
| fewer than 2 | unusable | warns and falls back to the keyboard |

Startup prints which wheel was bound and which mode it is in.

### Compatibility mode is degraded

A Logitech wheel with no vendor driver installed enumerates as the generic
"Driving Force" device (USB PID `0xC294`) and reports only **2 axes**. In that
mode:

- throttle and brake share one axis, so they **cancel each other out** — no
  trail-braking and no left-foot braking
- steering travel is ~240° instead of 900°
- there is no force feedback

Driving works, but the reduced lock changes steering behaviour enough to matter
for a study. To get native mode, install **Logitech Gaming Software 5.10** —
G HUB dropped support for the G25/G27. Once LGS switches the wheel over it
re-enumerates (a G25 becomes PID `0xC299`), reports 4 axes, and
`drive_improved.py` picks the native branch automatically.

### Tuning constants

Near the top of `src/drive/drive_improved.py`:

| Constant | Purpose |
|---|---|
| `WHEEL_AXIS_STEER` / `_THROTTLE` / `_BRAKE` | native-mode axis indices |
| `WHEEL_AXIS_COMBINED` | compatibility-mode shared pedal axis |
| `WHEEL_COMBINED_THROTTLE_SIGN` | which direction of the combined axis is throttle — **flip this if throttle and brake come out swapped** |
| `WHEEL_STEER_DEADZONE`, `WHEEL_PEDAL_DEADZONE` | deadzones |
| `WHEEL_STEER_LIMIT` | steering clamp (`1.0` = full lock; the keyboard path caps at `0.7`) |

Steering is taken raw rather than quantised to `0.1` like the keyboard path —
rounding makes a wheel feel notched.

---

## 2. Button mapping (required for LoA input)

Button indices are device-specific, and most of the indices a Logitech wheel
reports are phantom — a G25 advertises its detachable shifter whether or not it
is attached. Map them once per rig:

```bash
python scripts/map_wheel_buttons.py
```

It asks you to press each control in turn and prints constants to paste into the
top of `src/drive/drive_improved.py`:

```python
LOA_WHEEL_BUTTON_CONFIRM = 6   # front button: ticks a level / submits
LOA_WHEEL_BUTTON_PREV    = 5   # left paddle:  moves the cursor down
LOA_WHEEL_BUTTON_NEXT    = 4   # right paddle: moves the cursor up
WHEEL_BUTTON_QUIT        = 7   # other front button: closes the simulation
```

Buttons are polled directly rather than read from the event queue, so the script
needs no window and no window focus.

**Until these are set, the wheel cannot answer the LoA prompt.** The popup says
so on screen and keeps the keyboard available, rather than leaving a participant
pressing a dead button.

---

## 3. Answering the LoA prompt from the wheel

The prompt appears every 20 s and freezes the scene. A driver may mark
**every** level they would accept, not just one.

- **Right paddle** — move the cursor up (towards higher LoA)
- **Left paddle** — move the cursor down
- **Front button** — tick or untick the level under the cursor
- **Front button on the `CONFIRM` row** — submit

The cursor spans the five LoA rows plus a trailing `CONFIRM` row. That extra row
exists because the wheel has only one free front button, so ticking and
submitting have to share it.

Ticked levels show as `[x]` and in green; the cursor is marked with `> <` and in
yellow — markers as well as colour, so the state survives a projector or a
colour-blind participant.

Two deliberate behaviours:

- **Nothing is pre-selected.** The cursor starts on no row and submitting with
  nothing ticked does nothing. A default-selected option would anchor the
  answer.
- **The cursor clamps and never wraps**, so holding a paddle cannot roll from
  LoA 0 straight round to LoA 4.

The keyboard still works throughout: `0`–`4` toggle a level and `ENTER`
submits. (Number keys *toggle* rather than submit immediately — otherwise a
second level could never be added.)

### Quitting

`WHEEL_BUTTON_QUIT` closes the simulation exactly like the window's X, and works
on all three screens (start screen, driving, and during an LoA prompt). It goes
through the same shutdown path, so CARLA settings are restored and actors
destroyed.

> **Note:** both front buttons sit next to each other — one submits, one ends the
> session. There is no undo. If that is a risk for your protocol, consider
> leaving `WHEEL_BUTTON_QUIT` unset and quitting with `ESC`.

---

## 4. What the driver's answer becomes

A multi-mark is written to `data/user_loa_labels.csv` as a `;`-joined list:

| `user_selected_loa` | Meaning |
|---|---|
| `2` | only LoA 2 acceptable |
| `2;3` | LoA 2 **and** 3 both acceptable |

A single mark stays a bare integer, so labels recorded before multi-select
still parse unchanged.

`scripts/build_loa_dataset.py` turns this into a **multi-hot** `Level_1..5`.
See the "State Model" section of the main [README](../README.md) for which
losses can consume more than one marked level.

---

## 5. Troubleshooting

**No wheel detected at all.** Check `pygame.joystick.get_count()`:

```bash
python -c "import pygame; pygame.init(); pygame.joystick.init(); print(pygame.joystick.get_count())"
```

**Buttons do nothing during the LoA prompt.** The indices are unmapped — run
`scripts/map_wheel_buttons.py`. The popup shows a warning when this is the case.

**Throttle and brake are swapped** (compatibility mode only). Flip
`WHEEL_COMBINED_THROTTLE_SIGN` between `-1.0` and `1.0`.

**Steering is coarse or the pedals fight each other.** The wheel is in
compatibility mode — see section 1.

**The wheel works but the car does not respond.** The wheel replaces the
keyboard for steer/throttle/brake when bound; they cannot both drive, since the
keyboard path zeroes throttle on every frame no key is held. Use `--no-wheel`
to force keyboard control.
