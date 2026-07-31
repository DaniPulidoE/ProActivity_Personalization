#!/usr/bin/env python

"""Cabin/road ambience for the drive UI.

CARLA has no audio of any kind -- no ambient sound, no engine sound, no audio
sensor -- in ANY version, 0.10 included. There is nothing in the simulator to
switch on and nothing in its shipped content to extract: whatever the
participant hears, this process plays. Without it the rig is silent, which is
the single most obvious way the simulation announces that it is a simulation.

The bed is SYNTHESISED rather than sampled, which is the right call twice over:

* Road and cabin noise physically IS filtered broadband noise. Synthesis here is
  not a stand-in for a recording, it is the same object.
* A study has to be able to state exactly what every participant heard. A gain
  and a seed in the label rows reproduce this bit-for-bit years later; a wav
  needs provenance, a licence, and a copy that never drifts.

The loop is built in the frequency domain (fixed magnitude spectrum, random
phases, inverse FFT), so it is exactly periodic BY CONSTRUCTION and ``loops=-1``
wraps with no click. Hand-trimming a recording to loop that cleanly is fiddly,
and the residual tick is exactly the kind of thing a participant stops noticing
consciously after ten minutes and keeps responding to physiologically -- which
is a signal this project MEASURES (hr_delta, rr_delta), so a periodic startle
every N seconds would land straight in the model's input.

Everything here degrades to silence rather than failing. A rig with no audio
device, a mixer that will not open, an unexpected sample format: all of them end
with ``effective_gain == 0.0``, a printed warning, and a drive that behaves
exactly as it did before this module existed.

Level calibration is NOT done here and cannot be: what reaches the driver
depends on the amplifier and the room. Set the physical volume once, measure
dB(A) at the driver's head, and report that number -- ``--ambient-gain 0.35``
means nothing to a reader.
"""

import math

import numpy as np
import pygame

# Mixer format. -16 is signed 16-bit, matching the int16 buffers built below.
# The 512-sample buffer is ~12 ms of latency at 44.1 kHz, which is irrelevant
# for a continuous bed (nothing here is synchronised to a visual event) and
# still small enough that a future engine-sound layer would stay responsive.
MIXER_RATE = 44100
MIXER_SIZE = -16
MIXER_CHANNELS = 2
MIXER_BUFFER = 512

# Loop length. Long enough that the repeat is not perceptible as a rhythm,
# short enough to build in well under a tenth of a second and to cost only a
# few MB resident.
LOOP_SECONDS = 20.0

# Spectral tilt in amplitude, as an exponent on frequency: -1.0 is pink,
# -2.0 brown. Road/cabin noise sits between the two -- rubber and glass roll off
# the top end hard, and the body of the sound is low.
SPECTRAL_TILT = -1.5

# Band limits. The high-pass matters more than it looks: with a negative tilt
# the sub-audible bins carry almost all the amplitude, so without it the
# normalisation below spends the entire headroom on content nobody can hear and
# the audible part comes out whisper-quiet.
HIGHPASS_HZ = 30.0
LOWPASS_HZ = 8000.0

# Peak amplitude of the generated buffer, as a fraction of full scale. Noise has
# a high crest factor, so this is headroom against the rare peak, not the
# perceived level -- that is set by the channel volume and the physical knob.
PEAK_SCALE = 0.6

# Speed -> level mapping. Volume follows sqrt(speed/SPEED_FULL_KMH) so the sound
# opens up early and then flattens, roughly how road noise behaves; IDLE_LEVEL
# is what remains at a standstill so the cabin never goes dead silent.
SPEED_FULL_KMH = 90.0
IDLE_LEVEL = 0.25

# Time constant for volume changes. Without it, the step to idle when the scene
# freezes for a LoA popup is an audible click, and a click at the exact moment a
# prompt appears is a cue no participant should be getting.
VOLUME_TAU_S = 0.35


def configure_mixer():
    """Request the mixer format. MUST be called BEFORE ``pygame.init()``.

    ``pre_init`` only sets the parameters the eventual mixer init will use, so
    once ``pygame.init()`` has opened the device this call does nothing at all
    -- and the buffer size, the one parameter that cannot be changed afterwards,
    is silently left at SDL's default. Hence the ordering requirement, which is
    the whole reason this is a separate function instead of living in
    ``Ambience.__init__``.
    """
    try:
        pygame.mixer.pre_init(MIXER_RATE, MIXER_SIZE, MIXER_CHANNELS,
                              MIXER_BUFFER)
    except pygame.error as exc:  # no audio subsystem at all
        print('[WARN] Could not pre-configure the audio mixer (%s).' % exc)


def _looping_noise(seconds, rate, channels, tilt=SPECTRAL_TILT, seed=0):
    """Seamless band-limited noise loop as an int16 array the mixer accepts.

    Every component is a harmonic of 1/seconds, so the buffer is exactly
    periodic and the loop point is not a discontinuity. The two channels get
    INDEPENDENT phases from the same magnitude spectrum: identical channels
    collapse to a point in the middle of the head on headphones, whereas
    decorrelated ones sound like a space the driver is sitting inside.
    """
    n = int(round(seconds * rate))
    rng = np.random.default_rng(seed)

    freqs = np.fft.rfftfreq(n, 1.0 / rate)
    mag = np.zeros_like(freqs)
    band = (freqs >= HIGHPASS_HZ) & (freqs <= LOWPASS_HZ)
    mag[band] = freqs[band] ** tilt

    rendered = []
    for _ in range(max(1, channels)):
        phase = rng.uniform(0.0, 2.0 * np.pi, freqs.shape)
        rendered.append(np.fft.irfft(mag * np.exp(1j * phase), n))

    buf = rendered[0] if channels < 2 else np.stack(rendered, axis=1)
    peak = float(np.max(np.abs(buf)))
    if peak > 0.0:
        buf = buf / peak

    # rint, not a bare astype: astype truncates TOWARDS ZERO, which is a
    # signal-correlated error rather than noise. Measured on this spectrum it
    # left a DC offset and pushed sub-20 Hz content ~9 dB above the 16-bit
    # floor -- inaudible either way, but rounding is free and correct, and the
    # LF end is where the tilt already concentrates the energy.
    return np.ascontiguousarray(
        np.rint(buf * PEAK_SCALE * 32767.0).astype(np.int16))


class Ambience(object):
    """Speed-coupled cabin noise, or a silent no-op.

    Construct once per session, call :meth:`update` every frame with the current
    speed, call :meth:`stop` on the way out. ``effective_gain`` is the honest
    record of what was actually played: it stays 0.0 when audio was not asked
    for AND when it was asked for but could not be started, so a label row never
    claims a participant heard something they did not.
    """

    def __init__(self, gain=0.0, seed=0, loop_seconds=LOOP_SECONDS):
        self.requested_gain = max(0.0, float(gain))
        self.seed = int(seed)
        self.effective_gain = 0.0
        self._sound = None
        self._channel = None
        self._level = 0.0
        self._last_ms = None

        if self.requested_gain <= 0.0:
            return

        try:
            if not pygame.mixer.get_init():
                pygame.mixer.init(MIXER_RATE, MIXER_SIZE, MIXER_CHANNELS,
                                  MIXER_BUFFER)
            init = pygame.mixer.get_init()
            if not init:
                raise pygame.error('mixer did not open')

            # Build to the format the device ACTUALLY opened with, not the one
            # requested. SDL is free to substitute (48 kHz is a common
            # substitution for 44.1), and make_sound rejects any array that does
            # not match the open device -- so adapting here is the difference
            # between working on a rig with a fussy sound card and silently
            # printing a warning on it.
            rate, fmt, channels = init[0], init[1], init[2]
            if fmt != MIXER_SIZE:
                raise pygame.error(
                    'mixer opened as %d-bit, this module builds signed 16-bit'
                    % abs(fmt))

            buf = _looping_noise(loop_seconds, rate, channels, seed=self.seed)
            self._sound = pygame.sndarray.make_sound(buf)
            self._channel = self._sound.play(loops=-1)
            if self._channel is None:
                raise pygame.error('no free mixer channel')
        except Exception as exc:
            # Deliberately broad: a missing sound card, an SDL backend failure
            # and a numpy/sndarray format mismatch all raise different things,
            # and NONE of them is a reason to lose a participant's session.
            print('[WARN] Ambience off: could not start audio (%s). The drive '
                  'runs silent and ambient_gain is logged as 0.' % exc)
            self._sound = None
            self._channel = None
            return

        self.effective_gain = self.requested_gain
        self.update(0.0)
        print('[INFO] Ambience on: gain=%.2f seed=%d (%.0f s loop, %d Hz, %d ch). '
              'Fix the gain AND the physical volume across all participants and '
              'both study arms, and report the measured dB(A), not this number.'
              % (self.effective_gain, self.seed, loop_seconds, rate, channels))

    @property
    def active(self):
        return self._channel is not None

    def update(self, speed_kmh):
        """Track the target level for ``speed_kmh``. Safe to call every frame.

        Pass 0.0 whenever the scene is frozen or the car is not under the
        driver's control -- during a LoA popup the HUD speed is stale, so the
        bed would otherwise hold motorway noise over a motionless picture.
        """
        if self._channel is None:
            return

        speed = float(speed_kmh)
        if not speed == speed:  # NaN
            speed = 0.0
        speed = min(max(speed, 0.0), SPEED_FULL_KMH)
        target = IDLE_LEVEL + (1.0 - IDLE_LEVEL) * math.sqrt(
            speed / SPEED_FULL_KMH)

        # dt is measured here rather than passed in so the smoothing is correct
        # whatever rate the caller happens to run at -- the drive loop cap moves
        # with --sync, and the popup branch is a different path through it.
        #
        # Only the FIRST call snaps to the target (there is no previous level to
        # glide from). Deciding that on dt == 0 instead would be a latent click:
        # get_ticks has 1 ms resolution, so any two updates landing in the same
        # millisecond would jump the volume to wherever the speed says.
        now = pygame.time.get_ticks()
        if self._last_ms is None:
            self._level = target
        else:
            dt = max(0, now - self._last_ms) / 1000.0
            if dt > 0.0:
                alpha = 1.0 - math.exp(-dt / VOLUME_TAU_S)
                self._level += alpha * (target - self._level)
        self._last_ms = now

        try:
            self._channel.set_volume(self.effective_gain * self._level)
        except pygame.error:
            pass

    def stop(self):
        if self._channel is not None:
            try:
                self._channel.stop()
            except pygame.error:
                pass
        self._sound = None
        self._channel = None
