#!/usr/bin/env python3
"""Map the steering wheel's buttons for LoA selection.

Button indices are device-specific and most of the indices a Logitech wheel
reports are phantom (the G25 advertises its detachable shifter whether or not it
is attached). Run this once per rig, then paste the output into the constants at
the top of src/drive/drive_improved.py.

    python scripts/map_wheel_buttons.py

Buttons are polled directly rather than read from the event queue, so no window
and no window focus are needed.
"""
import os
import sys
import time

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_JOYSTICK_ALLOW_BACKGROUND_EVENTS", "1")

import pygame


def wait_release(js, nbtn, timeout=10.0):
    """Block until every button is up, so one press is not captured twice."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        pygame.event.pump()
        if not any(js.get_button(b) for b in range(nbtn)):
            return
        time.sleep(0.02)


def capture(js, nbtn, role, timeout=60.0):
    wait_release(js, nbtn)
    sys.stdout.write("  Press %s ... " % role)
    sys.stdout.flush()
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        pygame.event.pump()
        for b in range(nbtn):
            if js.get_button(b):
                print("button %d" % b)
                return b
        time.sleep(0.02)
    print("nothing pressed")
    return None


def main():
    pygame.init()
    pygame.joystick.init()
    if pygame.joystick.get_count() == 0:
        sys.exit("No wheel detected. Is it plugged in?")

    js = pygame.joystick.Joystick(0)
    js.init()
    nbtn = js.get_numbuttons()
    print("%r: %d button indices, %d axes, %d hat(s)\n"
          % (js.get_name(), nbtn, js.get_numaxes(), js.get_numhats()))
    print("Press each control when prompted. If nothing ever registers, the")
    print("wheel is not reaching SDL at all.\n")

    try:
        nxt = capture(js, nbtn, "the RIGHT paddle  (raises the LoA)")
        prev = capture(js, nbtn, "the LEFT paddle   (lowers the LoA)")
        confirm = capture(js, nbtn, "the FRONT button you want as SELECT")
        quit_btn = capture(js, nbtn, "the OTHER front button (CLOSES the simulation)")
        wait_release(js, nbtn)
    except KeyboardInterrupt:
        sys.exit("\naborted")

    print("\n" + "=" * 64)
    captured = [prev, nxt, confirm, quit_btn]
    if any(c is None for c in captured):
        print("Timed out waiting for a press — nothing written.")
    elif len(set(captured)) < 4:
        print("The same button was captured more than once:")
        print("  prev=%s next=%s confirm=%s quit=%s" % (prev, nxt, confirm, quit_btn))
        print("These must be four different buttons. Run it again.")
    else:
        print("Paste into src/drive/drive_improved.py:\n")
        print("LOA_WHEEL_BUTTON_CONFIRM = %d" % confirm)
        print("LOA_WHEEL_BUTTON_PREV = %d" % prev)
        print("LOA_WHEEL_BUTTON_NEXT = %d" % nxt)
        print("WHEEL_BUTTON_QUIT = %d" % quit_btn)
    print("=" * 64)
    pygame.quit()


if __name__ == "__main__":
    main()
