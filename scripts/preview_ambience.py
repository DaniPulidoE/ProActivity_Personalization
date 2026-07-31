#!/usr/bin/env python
"""Render the drive ambience to a wav, so it can be auditioned and tuned
without starting CARLA.

    python scripts/preview_ambience.py --out ambience.wav

The point is the tuning loop. The constants in src/drive/ambience.py
(ROAD_LAYERS levels and exponents, ENGINE_* , GEAR_TOP_KMH) are hand-balanced by
ear, and doing that through a CARLA launch each time is minutes per iteration
for a five-second judgement. This renders a scripted drive -- idle, pull away
through the gears, cruise, slow to a stop -- in about a second.

It shares mix_levels() and engine_blend() with the live path rather than
reimplementing the curves, so what comes out of the wav IS the mix a participant
hears. Only the playback differs: pygame crossfades channels in real time, this
sums numpy arrays block by block.

Nothing here is used at run time; the drive does not import it.
"""

import argparse
import os
import sys
import wave

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                '..', 'src', 'drive'))

import ambience as A  # noqa: E402


# (seconds, speed_kmh, throttle) waypoints, linearly interpolated. Shaped to
# expose the things that are hard to judge from a static tone: the pull through
# the shift points, a steady cruise, and the drop back to idle.
PROFILE = (
    (0.0, 0.0, 0.0),
    (3.0, 0.0, 0.0),
    (5.0, 15.0, 0.8),
    (9.0, 45.0, 0.9),
    (14.0, 85.0, 0.7),
    (20.0, 95.0, 0.35),
    (26.0, 50.0, 0.0),
    (30.0, 0.0, 0.0),
    (34.0, 0.0, 0.0),
)


def _at(t):
    """(speed, throttle) at time t by linear interpolation of PROFILE."""
    if t <= PROFILE[0][0]:
        return PROFILE[0][1], PROFILE[0][2]
    for (t0, s0, g0), (t1, s1, g1) in zip(PROFILE, PROFILE[1:]):
        if t <= t1:
            f = (t - t0) / (t1 - t0) if t1 > t0 else 0.0
            return s0 + f * (s1 - s0), g0 + f * (g1 - g0)
    return PROFILE[-1][1], PROFILE[-1][2]


def render(rate, seed, gain, loop_seconds, block=1024):
    rng = np.random.default_rng(seed)

    texture = A._texture_envelope(loop_seconds, rate, rng)
    road = []
    for name, lo, hi, tilt, _rest, _full, _exp in A.ROAD_LAYERS:
        buf = A._band_noise(loop_seconds, rate, lo, hi, tilt, rng, 2)
        if name == 'rumble':
            buf = buf * texture[:, None]
        peak = float(np.max(np.abs(buf)))
        road.append(buf / peak if peak > 0 else buf)

    engine = []
    for i in range(A.ENGINE_BUCKETS):
        frac = i / float(max(1, A.ENGINE_BUCKETS - 1))
        rpm = A.IDLE_RPM + frac * (A.REDLINE_RPM - A.IDLE_RPM)
        cyc = A._engine_cycle(rpm, A.ENGINE_LOOP_S, rate, rng)
        peak = float(np.max(np.abs(cyc)))
        engine.append(cyc / peak if peak > 0 else cyc)

    total = int(PROFILE[-1][0] * rate)
    out = np.zeros((total, 2), dtype=np.float64)

    # Same smoothing the live path applies, so shift points land where they do
    # in the car rather than instantaneously.
    speed = throttle = 0.0
    gear = 1
    dt = block / float(rate)
    for start in range(0, total, block):
        n = min(block, total - start)
        t = start / float(rate)
        tgt_speed, tgt_throttle = _at(t)
        speed += (1.0 - np.exp(-dt / A.SPEED_TAU_S)) * (tgt_speed - speed)
        throttle += (1.0 - np.exp(-dt / A.THROTTLE_TAU_S)) * (tgt_throttle
                                                              - throttle)

        gear, rpm, road_levels, engine_level = A.mix_levels(speed, throttle,
                                                            gear)
        lo, hi, frac = A.engine_blend(rpm)

        for buf, level in zip(road, road_levels):
            idx = (np.arange(start, start + n) % len(buf))
            out[start:start + n] += level * buf[idx]

        eidx = (np.arange(start, start + n) % len(engine[lo]))
        cyc = ((1.0 - frac) * engine[lo][eidx] + frac * engine[hi][eidx])
        out[start:start + n] += engine_level * cyc[:, None]

    peak = float(np.max(np.abs(out)))
    if peak > 0:
        out = out / peak
    return np.rint(out * gain * 32767.0).astype(np.int16)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out', default='ambience_preview.wav')
    ap.add_argument('--rate', type=int, default=A.MIXER_RATE)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--gain', type=float, default=0.9,
                    help='Peak of the rendered file (default 0.9). This is a '
                         'file level for auditioning, NOT --ambient-gain: the '
                         'mix is normalised before it is applied, so the '
                         'balance between layers is what you are judging.')
    ap.add_argument('--loop-seconds', type=float, default=A.LOOP_SECONDS)
    args = ap.parse_args()

    pcm = render(args.rate, args.seed, args.gain, args.loop_seconds)
    with wave.open(args.out, 'wb') as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(args.rate)
        w.writeframes(pcm.tobytes())

    print('%s  %.1f s, %d Hz stereo' % (args.out, len(pcm) / float(args.rate),
                                        args.rate))
    print('Profile: idle -> pull away through the gears -> cruise -> stop.')
    print('Tune ROAD_LAYERS / ENGINE_* / GEAR_TOP_KMH in '
          'src/drive/ambience.py and re-run.')


if __name__ == '__main__':
    main()
