#!/usr/bin/env python

"""Engine and road ambience for the drive UI.

CARLA has no audio of any kind -- no ambient sound, no engine sound, no audio
sensor -- in ANY version, 0.10 included. There is nothing in the simulator to
switch on and nothing in its shipped content to extract: whatever the
participant hears, this process synthesises. Without it the rig is silent, which
is the most obvious way the simulation announces that it is a simulation.

Nothing is sampled. A study has to be able to state exactly what every
participant heard, and a gain plus a seed reproduce this bit-for-bit years
later, whereas a wav needs provenance, a licence and a copy that never drifts.

WHY IT IS BUILT THIS WAY
------------------------
The naive version of this file -- one band-limited noise loop with its volume
tied to speed -- sounds like wind, and no amount of tuning fixes that, because
two things are wrong at the level of the model:

* A car's sound changes SHAPE with speed, not just level. Tyre and wind noise
  climb far faster at the top end than the low rumble does, so the spectrum
  tilts brighter as the car accelerates. One fixed spectrum behind a volume
  knob is, by construction, "the same sound, louder".
* A car is TONAL. The engine produces a harmonic stack at the firing frequency,
  and that stack -- rising through a gear, dropping at each shift, rising again
  -- is the cue that says "vehicle" rather than "weather". Filtered noise has no
  harmonic content at all, so it can never sound like an engine.

So the bed is four layers mixed live:

  rumble / body / hiss   three noise bands with DIFFERENT speed exponents, so
                         their balance (the spectral tilt) moves with speed
  engine                 a harmonic stack whose fundamental tracks a simulated
                         gearbox, pre-rendered at ENGINE_BUCKETS speeds and
                         crossfaded

Every buffer is built in the frequency domain (magnitudes on bin centres,
random phases, inverse FFT), which makes it exactly periodic BY CONSTRUCTION so
``loops=-1`` wraps with no click. That matters more here than it looks: a
periodic tick is exactly the kind of thing a participant stops noticing
consciously after ten minutes and keeps responding to physiologically, and this
project MEASURES that (hr_delta, rr_delta feed the model).

It still sounds synthesised -- it is a synthesiser. It sounds like a car rather
than like weather, which is the bar that matters for presence.

Everything degrades to silence rather than failing: no audio device, a mixer
that will not open, an unexpected sample format all end with
``effective_gain == 0.0``, a printed warning, and a drive that behaves exactly
as it did before this module existed.

Level calibration is NOT done here and cannot be: what reaches the driver
depends on the amplifier and the room. Set the physical volume once, measure
dB(A) at the driver's head, and report that -- ``--ambient-gain 0.35`` means
nothing to a reader.
"""

import math

import numpy as np
import pygame

# Mixer format. -16 is signed 16-bit, matching the int16 buffers built below.
# The 512-sample buffer is ~12 ms of latency at 44.1 kHz, small enough that the
# engine note still feels connected to the throttle.
MIXER_RATE = 44100
MIXER_SIZE = -16
MIXER_CHANNELS = 2
MIXER_BUFFER = 512

# Default gain, and ON by default -- the ONLY definition of it in the codebase,
# so the launcher and Drive cannot drift apart and hand two participants
# different conditions.
#
# On rather than off because the two failure modes are not symmetric. Forgetting
# a flag under an off-by-default gives ONE participant silence while the rest
# drove with sound, an unbalanced condition discovered (if at all) during
# analysis; forgetting it under an on-by-default gives everyone the same thing.
# Silence is the deliberate choice, spelled --ambient-gain 0.
#
# The number is a starting point, NOT a level: what reaches the driver is set by
# the amplifier. Set that once against a meter and leave both alone.
DEFAULT_AMBIENT_GAIN = 0.35

# Speed treated as "full" for the layer mix. Not a cap on the car, just the
# reference the exponents below are expressed against.
SPEED_FULL_KMH = 90.0

# Road layers: (name, lo_hz, hi_hz, tilt, level_at_rest, level_at_full, exponent)
#
# The EXPONENTS are what stop this sounding like wind. Level for a layer is
#   rest + (full - rest) * (speed/SPEED_FULL_KMH) ** exponent
# so hiss (2.0) is almost absent at a standstill and dominant at speed, while
# rumble (0.6) is already there at idle and grows slowly. The mix therefore
# tilts brighter with speed instead of merely getting louder.
#
# tilt is the exponent on frequency within the band: -2 is brown, -1 pink, 0
# white. Levels are hand-balanced by ear against the equal-loudness curves --
# low frequencies at equal amplitude sound much quieter, hence rumble's large
# weight.
ROAD_LAYERS = (
    ('rumble',   25.0,  180.0, -2.0, 0.30, 0.75, 0.6),
    ('body',    150.0, 1200.0, -1.2, 0.06, 0.55, 1.0),
    ('hiss',   1000.0, 7000.0, -0.6, 0.01, 0.65, 2.0),
)

# Road-texture modulation baked into the rumble layer: slow random level drift,
# as the surface under the tyres changes. Built periodic like everything else,
# so it costs nothing at runtime and does not break the loop.
TEXTURE_BAND_HZ = (0.15, 2.5)
TEXTURE_DEPTH = 0.35

# --- Engine -----------------------------------------------------------------
# A 4-stroke fires CYLINDERS/2 times per revolution, so the fundamental is
# rpm/60 * CYLINDERS/2 -- 25 Hz at idle, ~140 Hz at redline for these numbers.
ENGINE_CYLINDERS = 4
IDLE_RPM = 750.0
SHIFT_RPM = 2600.0
REDLINE_RPM = 4200.0
ENGINE_HARMONICS = 14
ENGINE_HARMONIC_ROLLOFF = 1.15

# Each harmonic is smeared over a few bins rather than placed on exactly one.
# A pure line spectrum sounds like a church organ; real combustion is rough,
# and this width is most of the difference between "engine" and "synth tone".
ENGINE_PARTIAL_WIDTH_HZ = 1.5
ENGINE_PARTIAL_WIDTH_REL = 0.008
# Broadband combustion/induction noise mixed under the harmonics.
ENGINE_NOISE_LEVEL = 0.30

# Pre-rendered rev buckets, crossfaded in pairs. pygame cannot pitch-shift a
# playing Sound, so continuous revving is 20 fixed points with a crossfade
# between neighbours -- standard game-audio practice, and at this spacing the
# steps are inaudible.
ENGINE_BUCKETS = 20
ENGINE_LOOP_S = 1.0
# Engine level: present at idle, louder under load.
ENGINE_REST_LEVEL = 0.30
ENGINE_FULL_LEVEL = 0.85
ENGINE_THROTTLE_LIFT = 0.35

# Simulated gearbox. CARLA's own gear is server-side under automatic
# transmission, and reading it would cost an RPC per frame on a loop this
# project already tunes for frame rate -- so the box is simulated from speed.
# The DOWNSHIFT margin is hysteresis: without it the box chatters between two
# gears whenever the driver holds a speed near a shift point.
GEAR_TOP_KMH = (28.0, 52.0, 80.0, 112.0, 150.0, 200.0)
DOWNSHIFT_MARGIN = 0.80

# Loop length for the road layers. Long enough that the repeat is not audible
# as a rhythm, short enough to build quickly and cost only a few MB.
LOOP_SECONDS = 20.0

# Peak amplitude of each generated buffer, as a fraction of full scale. Noise
# has a high crest factor, so this is headroom against the rare peak. Layers
# sum, hence the conservative value.
PEAK_SCALE = 0.5

# Time constants. Speed is tracked faster than the mix so the engine still feels
# connected to the pedal; without any smoothing the step to idle when the scene
# freezes for a LoA popup is an audible click, and a click at the exact moment a
# prompt appears is a cue no participant should be getting.
SPEED_TAU_S = 0.18
THROTTLE_TAU_S = 0.12


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


def _to_int16(buf, channels):
    """Normalise to PEAK_SCALE and quantise, as the array shape the mixer wants.

    rint, not a bare astype: astype truncates TOWARDS ZERO, which is a
    signal-correlated error rather than noise. Measured on these spectra it left
    a DC offset and pushed sub-20 Hz content ~9 dB above the 16-bit floor.
    Inaudible either way, but rounding is free and correct, and the low end is
    where the tilts already concentrate the energy.
    """
    peak = float(np.max(np.abs(buf)))
    if peak > 0.0:
        buf = buf / peak
    if buf.ndim == 1 and channels >= 2:
        buf = np.repeat(buf[:, None], channels, axis=1)
    return np.ascontiguousarray(np.rint(buf * PEAK_SCALE * 32767.0)
                                .astype(np.int16))


def _spectrum_to_signal(mag, n, rng):
    """One periodic real signal with the given magnitude spectrum.

    Random phases on bin centres: every component is a harmonic of 1/duration,
    so the result is exactly periodic and the loop point is not a
    discontinuity.
    """
    phase = rng.uniform(0.0, 2.0 * np.pi, mag.shape)
    return np.fft.irfft(mag * np.exp(1j * phase), n)


def _band_noise(seconds, rate, lo_hz, hi_hz, tilt, rng, channels=2):
    """Band-limited noise with a spectral tilt, as a periodic loop.

    The two channels get INDEPENDENT phases from the same magnitude spectrum:
    identical channels collapse to a point in the middle of the head on
    headphones, whereas decorrelated ones sound like a space you sit inside.
    """
    n = int(round(seconds * rate))
    freqs = np.fft.rfftfreq(n, 1.0 / rate)
    mag = np.zeros_like(freqs)
    # The high-pass is not cosmetic: with a negative tilt the sub-audible bins
    # carry almost all the amplitude, so without it normalisation spends the
    # entire headroom on content nobody can hear.
    band = (freqs >= lo_hz) & (freqs <= hi_hz)
    mag[band] = freqs[band] ** tilt
    return np.stack([_spectrum_to_signal(mag, n, rng)
                     for _ in range(channels)], axis=1)


def _texture_envelope(seconds, rate, rng):
    """Slow periodic level drift in [1-depth, 1+depth] for the rumble layer."""
    n = int(round(seconds * rate))
    freqs = np.fft.rfftfreq(n, 1.0 / rate)
    mag = np.zeros_like(freqs)
    band = (freqs >= TEXTURE_BAND_HZ[0]) & (freqs <= TEXTURE_BAND_HZ[1])
    mag[band] = 1.0
    env = _spectrum_to_signal(mag, n, rng)
    peak = float(np.max(np.abs(env)))
    if peak > 0.0:
        env = env / peak
    return 1.0 + TEXTURE_DEPTH * env


def _engine_cycle(rpm, seconds, rate, rng):
    """Harmonic stack for one engine speed, as a periodic loop.

    Mono on purpose: the engine is in front of the driver and belongs in the
    middle of the image, unlike the road layers which surround them.
    """
    n = int(round(seconds * rate))
    freqs = np.fft.rfftfreq(n, 1.0 / rate)
    mag = np.zeros_like(freqs)
    nyquist = 0.45 * rate

    f0 = rpm / 60.0 * ENGINE_CYLINDERS / 2.0
    for k in range(1, ENGINE_HARMONICS + 1):
        centre = f0 * k
        if centre >= nyquist:
            break
        # Gaussian smear instead of a single bin: a line spectrum is an organ,
        # combustion is rough. Width grows with frequency, as the jitter is a
        # roughly constant fraction of the period.
        width = ENGINE_PARTIAL_WIDTH_HZ + ENGINE_PARTIAL_WIDTH_REL * centre
        mag += (k ** -ENGINE_HARMONIC_ROLLOFF) * np.exp(
            -0.5 * ((freqs - centre) / width) ** 2)

    # Induction/combustion hiss under the harmonics, so the stack sits in
    # something rather than floating in silence between its partials.
    #
    # Matched on ENERGY, not on peak magnitude. Scaling by peak looks right and
    # is badly wrong: the harmonics occupy a few bins each while the noise is
    # spread over three kilohertz, so peak-matching at 0.30 put roughly NINE
    # TIMES more energy in the noise than in the stack -- an engine layer that
    # was mostly hiss, which is the exact "it sounds like wind" failure this
    # rewrite exists to fix.
    noise_band = (freqs >= 80.0) & (freqs <= 3500.0)
    noise_mag = np.zeros_like(freqs)
    noise_mag[noise_band] = freqs[noise_band] ** -1.0
    harmonic_energy = float(np.sqrt(np.sum(mag ** 2)))
    noise_energy = float(np.sqrt(np.sum(noise_mag ** 2)))
    if noise_energy > 0.0 and harmonic_energy > 0.0:
        noise_mag *= ENGINE_NOISE_LEVEL * harmonic_energy / noise_energy
    return _spectrum_to_signal(mag + noise_mag, n, rng)


def mix_levels(speed_kmh, throttle, previous_gear):
    """Layer levels for one vehicle state: (gear, rpm, [road levels], engine).

    Shared by :meth:`Ambience.update` and by scripts/preview_ambience.py rather
    than written twice, so what the preview renders to a wav is by construction
    the mix the participant hears -- a second copy of these curves would drift
    and quietly turn the tuning tool into a liar.
    """
    gear = _gear_for(speed_kmh, previous_gear)
    rpm = _rpm_for(speed_kmh, gear)

    v = min(max(speed_kmh, 0.0) / SPEED_FULL_KMH, 1.0)
    road = [rest + (full - rest) * v ** exponent
            for _n, _lo, _hi, _tilt, rest, full, exponent in ROAD_LAYERS]

    engine = min(1.0, (
        ENGINE_REST_LEVEL
        + (ENGINE_FULL_LEVEL - ENGINE_REST_LEVEL)
        * (rpm - IDLE_RPM) / max(1.0, REDLINE_RPM - IDLE_RPM)
        + ENGINE_THROTTLE_LIFT * min(max(throttle, 0.0), 1.0)))
    return gear, rpm, road, engine


def engine_blend(rpm):
    """(lo bucket, hi bucket, hi weight) for the crossfade at this engine speed."""
    pos = ((rpm - IDLE_RPM) / max(1.0, REDLINE_RPM - IDLE_RPM)
           * (ENGINE_BUCKETS - 1))
    lo = int(min(max(pos, 0.0), ENGINE_BUCKETS - 1))
    frac = min(max(pos - lo, 0.0), 1.0)
    return lo, min(lo + 1, ENGINE_BUCKETS - 1), frac


def _gear_for(speed_kmh, previous_gear):
    """Simulated automatic box with hysteresis. Gears are 1-based."""
    gear = min(max(previous_gear, 1), len(GEAR_TOP_KMH))
    while gear < len(GEAR_TOP_KMH) and speed_kmh > GEAR_TOP_KMH[gear - 1]:
        gear += 1
    while gear > 1 and speed_kmh < GEAR_TOP_KMH[gear - 2] * DOWNSHIFT_MARGIN:
        gear -= 1
    return gear


def _rpm_for(speed_kmh, gear):
    """Engine speed in this gear. Drops on each upshift, which is the point."""
    top = GEAR_TOP_KMH[gear - 1]
    rpm = IDLE_RPM + (speed_kmh / top) * (SHIFT_RPM - IDLE_RPM)
    return min(max(rpm, IDLE_RPM), REDLINE_RPM)


class Ambience(object):
    """Speed-coupled engine and road sound, or a silent no-op.

    Construct once per session, call :meth:`update` every frame with the current
    speed (and throttle, if it is free to read), call :meth:`stop` on the way
    out. ``effective_gain`` is the honest record of what was actually played: it
    stays 0.0 when audio was not asked for AND when it was asked for but could
    not be started, so a label row never claims a participant heard something
    they did not.
    """

    def __init__(self, gain=DEFAULT_AMBIENT_GAIN, seed=0,
                 loop_seconds=LOOP_SECONDS):
        self.requested_gain = max(0.0, float(gain))
        self.seed = int(seed)
        self.effective_gain = 0.0
        self._road = []           # [(channel, rest, full, exponent)]
        self._engine_sounds = []
        self._engine_ch = [None, None]
        self._engine_bucket = [-1, -1]
        self._speed = 0.0
        self._throttle = 0.0
        self._gear = 1
        self._last_ms = None
        self._started = False

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
            # requested. SDL is free to substitute (48 kHz is common), and
            # make_sound rejects any array that does not match the open device
            # -- adapting here is the difference between working on a rig with a
            # fussy sound card and printing a warning on it.
            rate, fmt, channels = init[0], init[1], init[2]
            if fmt != MIXER_SIZE:
                raise pygame.error(
                    'mixer opened as %d-bit, this module builds signed 16-bit'
                    % abs(fmt))
            channels = max(1, channels)

            # One channel per road layer plus two for the engine crossfade.
            needed = len(ROAD_LAYERS) + 2
            if pygame.mixer.get_num_channels() < needed:
                pygame.mixer.set_num_channels(needed)

            rng = np.random.default_rng(self.seed)
            started_ms = pygame.time.get_ticks()

            # Channels are claimed BY INDEX, not via find_channel(): two
            # consecutive find_channel() calls with nothing played in between
            # return the SAME idle channel, which would silently collapse the
            # engine's two crossfade slots into one.
            texture = _texture_envelope(loop_seconds, rate, rng)
            for idx, layer in enumerate(ROAD_LAYERS):
                name, lo, hi, tilt, rest, full, exponent = layer
                buf = _band_noise(loop_seconds, rate, lo, hi, tilt, rng,
                                  channels)
                if name == 'rumble':
                    buf = buf * texture[:, None]
                sound = pygame.sndarray.make_sound(_to_int16(buf, channels))
                ch = pygame.mixer.Channel(idx)
                ch.set_volume(0.0)
                ch.play(sound, loops=-1)
                # The Sound is held on the instance too: one that is garbage
                # collected while its channel plays takes the audio with it.
                self._road.append((ch, sound, rest, full, exponent))

            for i in range(ENGINE_BUCKETS):
                frac = i / float(max(1, ENGINE_BUCKETS - 1))
                rpm = IDLE_RPM + frac * (REDLINE_RPM - IDLE_RPM)
                cycle = _engine_cycle(rpm, ENGINE_LOOP_S, rate, rng)
                self._engine_sounds.append(
                    pygame.sndarray.make_sound(_to_int16(cycle, channels)))

            for slot in (0, 1):
                ch = pygame.mixer.Channel(len(ROAD_LAYERS) + slot)
                ch.set_volume(0.0)
                self._engine_ch[slot] = ch

            build_ms = pygame.time.get_ticks() - started_ms
        except Exception as exc:
            # Deliberately broad: a missing sound card, an SDL backend failure
            # and a numpy/sndarray format mismatch all raise different things,
            # and NONE of them is a reason to lose a participant's session.
            print('[WARN] Ambience off: could not start audio (%s). The drive '
                  'runs silent and ambient_gain is logged as 0.' % exc)
            self._silence()
            return

        self.effective_gain = self.requested_gain
        self._started = True
        self.update(0.0)
        print('[INFO] Ambience on: gain=%.2f seed=%d (%d road layers + %d rev '
              'buckets, %d Hz, %d ch, built in %d ms). Fix the gain AND the '
              'physical volume across all participants and both study arms, '
              'and report the measured dB(A), not this number.'
              % (self.effective_gain, self.seed, len(self._road),
                 ENGINE_BUCKETS, rate, channels, build_ms))

    def _silence(self):
        for entry in self._road:
            try:
                entry[0].stop()
            except pygame.error:
                pass
        for ch in self._engine_ch:
            if ch is not None:
                try:
                    ch.stop()
                except pygame.error:
                    pass
        self._road = []
        self._engine_sounds = []
        self._engine_ch = [None, None]
        self._started = False

    @property
    def active(self):
        return self._started

    @property
    def rpm(self):
        """Current simulated engine speed. Diagnostic only."""
        return _rpm_for(self._speed, self._gear)

    def update(self, speed_kmh, throttle=0.0):
        """Track speed (km/h) and throttle [0,1]. Safe to call every frame.

        Pass 0.0 whenever the scene is frozen or the car is not under the
        driver's control -- during a LoA popup the HUD speed is stale, so the
        bed would otherwise hold motorway noise over a motionless picture.
        """
        if not self._started:
            return

        target_speed = float(speed_kmh)
        if not target_speed == target_speed:  # NaN
            target_speed = 0.0
        target_speed = max(target_speed, 0.0)
        target_throttle = min(max(float(throttle or 0.0), 0.0), 1.0)

        # dt is measured here rather than passed in so the smoothing is correct
        # whatever rate the caller happens to run at -- the drive loop cap moves
        # with --sync, and the popup branch is a different path through it.
        #
        # Only the FIRST call snaps to the target. Deciding that on dt == 0
        # instead would be a latent click: get_ticks has 1 ms resolution, so any
        # two updates landing in the same millisecond would jump the mix.
        now = pygame.time.get_ticks()
        if self._last_ms is None:
            self._speed = target_speed
            self._throttle = target_throttle
        else:
            dt = max(0, now - self._last_ms) / 1000.0
            if dt > 0.0:
                self._speed += (1.0 - math.exp(-dt / SPEED_TAU_S)) * (
                    target_speed - self._speed)
                self._throttle += (1.0 - math.exp(-dt / THROTTLE_TAU_S)) * (
                    target_throttle - self._throttle)
        self._last_ms = now

        self._gear, rpm, road_levels, engine_level = mix_levels(
            self._speed, self._throttle, self._gear)

        for (ch, _sound, _rest, _full, _exp), level in zip(self._road,
                                                           road_levels):
            try:
                ch.set_volume(self.effective_gain * level)
            except pygame.error:
                pass

        level = self.effective_gain * engine_level
        lo, hi, frac = engine_blend(rpm)

        # Buckets are assigned to slots BY PARITY, which is what makes the
        # crossfade seamless. Restarting a Sound on a channel restarts it from
        # sample 0; with parity, the slot whose bucket changes as `lo` advances
        # is always the one whose weight has just fallen to ~0, so the restart
        # is inaudible. A fixed lo->slot0 / hi->slot1 assignment would restart
        # the slot carrying nearly all the level at every bucket boundary.
        #
        # Weights are accumulated per slot rather than applied in sequence,
        # because at the very top of the range lo == hi: both entries name the
        # same bucket and the same slot, and applying them in order would set
        # full level and then immediately overwrite it with frac == 0, i.e.
        # silence the engine exactly at redline.
        targets = {}
        for bucket, weight in ((lo, 1.0 - frac), (hi, frac)):
            slot = bucket % 2
            if slot in targets and targets[slot][0] == bucket:
                targets[slot] = (bucket, targets[slot][1] + weight)
            else:
                targets[slot] = (bucket, weight)

        for slot in (0, 1):
            ch = self._engine_ch[slot]
            if ch is None:
                continue
            bucket, weight = targets.get(slot, (None, 0.0))
            try:
                if bucket is not None and self._engine_bucket[slot] != bucket:
                    ch.play(self._engine_sounds[bucket], loops=-1)
                    self._engine_bucket[slot] = bucket
                ch.set_volume(level * weight)
            except pygame.error:
                pass

    def stop(self):
        self._silence()
