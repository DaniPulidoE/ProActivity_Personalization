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
