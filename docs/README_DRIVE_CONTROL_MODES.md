# Drive Improved - Control Mode Changes

### Control Modes

1. **`--control test` (default)**
   - Only basic driving controls are available:
     - **W**: Throttle
     - **A**: Steer left
     - **S**: Brake
     - **D**: Steer right
     - **Q**: Toggle reverse
     - **Space**: Hand-brake
     - **F1**: Toggle HUD
     - **H/?**: Toggle help
     - **ESC**: Quit

2. **`--control full`**
   - All controls are available (same as before):
     - All basic controls from test mode
     - **P**: Toggle autopilot
     - **M**: Toggle manual transmission
     - **,/.**: Gear up/down
     - **L, I, Z, X**: Light controls
     - **TAB, N, 1-9**: Camera/sensor controls
     - **C, G, V, B**: Weather, radar, map layer controls
     - **O, T**: Door and telemetry controls
     - **R, CTRL+R, CTRL+P**: Recording controls
     - **Backspace**: Change vehicle
     - And more...

> **Note:** Notification is only displayed in `full` control mode.

### Steering Wheel

When a wheel is attached it is bound automatically and **replaces the keyboard
for steering, throttle and brake** in both control modes. The two cannot drive
at once — the keyboard path zeroes throttle on every frame no key is held, so it
would fight the pedals. Pass `--no-wheel` to force keyboard control.

Everything else stays on the keyboard, including the hand-brake (**Space**) and
all the `full`-mode bindings above.

Wheel controls during the LoA prompt:

| Control | Action |
|---|---|
| Right paddle | Move the cursor up the LoA list |
| Left paddle | Move the cursor down |
| Front button | Tick / untick the level under the cursor |
| Front button on `CONFIRM` | Submit the marked levels |
| Other front button | Close the simulation (same as the window's X) |

Button indices are device-specific and must be mapped once per rig with
`python scripts/map_wheel_buttons.py`. See
**[Steering Wheel Setup](README_STEERING_WHEEL.md)** for the full guide.

## Usage

### Default (test mode):
```bash
python -m drive.drive_improvedUE5
```
or explicitly:
```bash
python -m drive.drive_improvedUE5 --control test
```

### Full control mode:
```bash
python -m drive.drive_improvedUE5 --control full
```
