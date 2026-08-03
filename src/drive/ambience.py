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
  engine                 firing impulses through a fixed resonant body, whose
                         firing rate tracks a simulated gearbox, pre-rendered at
                         ENGINE_BUCKETS speeds and crossfaded

WHY THE ENGINE IS AN IMPULSE TRAIN AND NOT A HARMONIC STACK
-----------------------------------------------------------
This layer was first written the same way as the road bed -- harmonic
MAGNITUDES on bin centres with RANDOM PHASES -- and it sounded like a church
organ, which is the one thing everybody who has done this reports. Three
separate reasons, all of them structural:

* PHASE. A cylinder firing is an IMPULSE, and at 25-140 Hz its period is
  7-40 ms, far longer than the ear's ~2 ms temporal resolution -- so the
  individual firings are RESOLVED, and what the listener hears is the pulse
  train, the "putt-putt" at idle and the bark under load. Random phase spreads
  each period's energy evenly across the period: identical magnitude spectrum,
  no impulse, and the result is a drone. Phase-deafness (Ohm's law) is about
  steady tones well above the resolution limit; it does not apply down here.
* A FIXED SPECTRAL ENVELOPE. Exhaust pipe, intake, block and panels resonate at
  frequencies that DO NOT MOVE, and rpm slides the harmonics through that fixed
  envelope. A rolloff expressed per harmonic INDEX (k**-a) stretches with rpm
  instead, so the whole timbre sweeps together -- a siren, not an engine. Worse,
  a fixed harmonic COUNT means the bandwidth scales with rpm: fourteen
  harmonics of a 25 Hz idle stop at 350 Hz, a muffled boom, while real idle has
  content past 2 kHz.
* HALF ORDERS. Cylinders are not identical, so the pattern repeats over the
  full four-stroke cycle (two revolutions, ENGINE_CYLINDERS firings), not over
  one firing. That puts energy at 1/4, 1/2, 3/4 of the firing frequency, and
  that lumpiness is a large part of what reads as "engine" rather than "tone
  generator", especially at idle.

So the engine is built the way the physics is: an excitation (one impulse per
firing, per-cylinder amplitude trims, per-cycle timing jitter, plus combustion
noise gated by the firings) convolved with a fixed body impulse response (a sum
of decaying resonances). The convolution is CIRCULAR, which is what keeps the
loop exactly periodic -- the ring-out of the last firing wraps into the first,
which is precisely what a steady engine does.

The road buffers are still built in the frequency domain (magnitudes on bin
centres, random phases, inverse FFT) -- for broadband noise, phase carries no
information the ear can use, and that construction makes them exactly periodic
BY CONSTRUCTION so ``loops=-1`` wraps with no click. Both layers therefore loop
seamlessly. That matters more here than it looks: a periodic tick is exactly the
kind of thing a participant stops noticing consciously after ten minutes and
keeps responding to physiologically, and this project MEASURES that (hr_delta,
rr_delta feed the model).

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

try:
    from . import ambience_assets
except ImportError:
    # Plain-script launch: no package context for a relative import.
    import ambience_assets

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
# A 4-stroke fires CYLINDERS/2 times per revolution, so the FIRING frequency is
# rpm/60 * CYLINDERS/2 -- 25 Hz at idle, 140 Hz at redline for these numbers.
# Four is also the right number for the car actually spawned: the MKZ's base
# engine is a 2.0 l four. Set 6 and every order below moves with it.
ENGINE_CYLINDERS = 4
IDLE_RPM = 750.0
SHIFT_RPM = 2600.0
REDLINE_RPM = 4200.0

# The body: (frequency Hz, Q, gain) of the resonances the firing impulses are
# played through -- tailpipe, block/panel boom, and the two mid formants that
# carry most of the "car" of it. These frequencies are FIXED and do not move
# with rpm, which is the whole point: the harmonics slide through a stationary
# envelope, exactly as they do in the car. Each entry contributes a decaying
# sinusoid exp(-pi*f*t/Q)*sin(2*pi*f*t) to the impulse response, so Q is
# literally how long it rings (tau = Q/(pi*f): 16 ms for the first, 0.3 ms for
# the last), and the gain is its RESPONSE at resonance -- see _engine_cycle for
# why that is not the same as the amplitude of the decaying sinusoid.
#
# The Qs of the two boom resonances are the punch/drone knob and were measured,
# not guessed: at Q=7 the 92 Hz one rings for 24 ms, which is longer than the
# firing period above ~1200 rpm, so consecutive firings overlap and the layer
# smooths back into the drone this rewrite exists to remove. Keep tau below a
# firing period across most of the rev range.
ENGINE_RESONANCES = (
    (92.0,   4.5, 1.00),
    (188.0,  4.0, 0.65),
    (430.0,  3.5, 0.38),
    (1150.0, 3.0, 0.20),
    (2450.0, 2.5, 0.10),
)
# Band limits on the finished loop. The high one keeps the impulses from
# sounding like digital clicks; the low one removes the sub-audible content that
# would otherwise eat the headroom (same argument as the road bands).
ENGINE_LOW_HZ = 28.0
ENGINE_HIGH_HZ = 5200.0

# Per-cylinder amplitude trim, one per cylinder, mean 1. Real cylinders differ
# in fuelling and compression, and that difference is what makes the pattern
# repeat over the four-stroke cycle instead of over one firing -- which is where
# the half orders come from. Constants rather than draws from the session rng on
# purpose: this is a property of the CAR, so it must be identical in every
# bucket (a different pattern per bucket would smear across the crossfade) and
# in every session.
ENGINE_CYLINDER_TRIM = (1.00, 0.88, 1.06, 0.94)
# Cycle-to-cycle roughness. Timing jitter is a fraction of the firing period.
# This is the real source of the roughness that smearing each partial across
# bins was faking -- combustion is irregular in TIME, and a magnitude smear with
# independent random phases is a detuned chorus, not an engine.
ENGINE_TIMING_JITTER = 0.012
ENGINE_AMPLITUDE_JITTER = 0.07
# Combustion/induction noise, gated by the firings (loudest just after each one)
# and pushed through the same body. Matched on RMS against the impulse train,
# not on peak -- see the same lesson in the road layers.
ENGINE_NOISE_LEVEL = 0.35
ENGINE_NOISE_DECAY_S = 0.020

# Pre-rendered rev buckets, crossfaded in pairs. pygame cannot pitch-shift a
# playing Sound, so continuous revving is a set of fixed points with a crossfade
# between neighbours -- standard game-audio practice.
#
# Spaced GEOMETRICALLY, because the ear hears pitch ratios. Linear spacing over
# 750-4200 rpm puts 181 rpm between neighbours everywhere, which is +4.5% at the
# top of the range and +24% at the bottom -- nearly four semitones, so the whole
# lower half of the rev range was crossfading between two audibly different
# notes and beating between them. Geometric spacing is 6.6% per step at every
# rpm, a bit over a semitone, and 28 of them cost 8 MB and 0.45 s to build
# (measured; see _smooth_length for why it is not four times that).
ENGINE_BUCKETS = 28
ENGINE_LOOP_S = 1.6
# RMS every bucket is normalised to. NOT peak: an impulse train has a crest
# factor that falls as the firings crowd together, so peak-normalising every
# bucket (which is what the road layers do) would make the engine quietly ramp
# up in loudness with rpm ON TOP of the level curve below, for no reason the
# curve knows about. Matching RMS is matching loudness, which is what the
# crossfade needs to be transparent.
ENGINE_RMS = 0.13
# Peak above which the soft knee starts. Transparent below it, asymptotic to 1
# above, so a rare impulse cannot clip without the ordinary ones being squashed.
ENGINE_KNEE = 0.80
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

# --- Recorded path ----------------------------------------------------------
# Level curve applied to recorded clips (see asset_level). Gentler than the
# synthesised layers on purpose: the clips carry the character themselves, so
# this only restores the loudness difference that RMS-matching them removed.
ASSET_REST_LEVEL = 0.45
ASSET_FULL_LEVEL = 1.0
ASSET_LEVEL_EXPONENT = 0.7

# Time constants. Speed is tracked faster than the mix so the engine still feels
# connected to the pedal; without any smoothing the step to idle when the scene
# freezes for a LoA popup is an audible click, and a click at the exact moment a
# prompt appears is a cue no participant should be getting.
SPEED_TAU_S = 0.18
THROTTLE_TAU_S = 0.12

# Ducking is separate from the speed-derived idle bed. update(0.0) already
# settles the mix to "engine idling, car stopped", which is correct for a
# frozen scene but is not silence -- a stopped car still runs. Ducking is for
# the popup itself, which is asked to go fully quiet. Its own tau, fast enough
# to feel responsive to the popup opening but not an audible step.
DUCK_TAU_S = 0.15


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


def _to_int16(buf, channels, normalise=True):
    """Normalise to PEAK_SCALE and quantise, as the array shape the mixer wants.

    ``normalise=False`` skips the peak normalisation and scales the buffer as
    given. The engine buckets need this: they are matched to each other on RMS
    (see ENGINE_RMS), and peak-normalising them here would throw that match
    away one bucket at a time.

    rint, not a bare astype: astype truncates TOWARDS ZERO, which is a
    signal-correlated error rather than noise. Measured on these spectra it left
    a DC offset and pushed sub-20 Hz content ~9 dB above the 16-bit floor.
    Inaudible either way, but rounding is free and correct, and the low end is
    where the tilts already concentrate the energy.
    """
    if normalise:
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


def _smooth_length(n):
    """Smallest 7-smooth integer >= n, i.e. the next length numpy likes.

    numpy's FFT is fast when the length factorises into small primes and falls
    back to Bluestein's algorithm when it does not. Measured across this rev
    ladder that is not a detail: the buckets whose length happened to come out
    smooth built in 16-26 ms and the rest took ~70 ms, four times the cost, for
    two thirds of the module's whole start-up.

    Rounding up moves the firing frequency by at most a few tenths of a percent
    -- a small fraction of the 6.6% between neighbouring buckets, so the
    crossfade cannot tell and neither can a listener.
    """
    while True:
        m = n
        for p in (2, 3, 5, 7):
            while m % p == 0:
                m //= p
        if m == 1:
            return n
        n += 1


def _soft_limit(buf, knee=ENGINE_KNEE):
    """Round off peaks above ``knee``, asymptotic to 1, transparent below it.

    A plain rescale-to-fit would be wrong here: it would undo the RMS match
    between buckets that the crossfade depends on, and it would pay for one rare
    impulse by making every ordinary one quieter -- i.e. it would spend the
    headroom on exactly the transients this synthesis exists to produce.
    """
    out = np.array(buf, dtype=np.float64, copy=True)
    a = np.abs(out)
    over = a > knee
    if np.any(over):
        head = 1.0 - knee
        out[over] = np.sign(out[over]) * (
            knee + head * np.tanh((a[over] - knee) / head))
    return out


def _engine_cycle(rpm, seconds, rate, rng):
    """One engine speed as a periodic loop: firing impulses through a body.

    Excitation (impulse per firing + gated combustion noise) circularly
    convolved with a fixed resonant impulse response. See the module docstring
    for why it is built this way and not as a harmonic stack.

    ``seconds`` is a TARGET, not the length: the loop is rounded to a whole
    number of four-stroke cycles so the per-cylinder pattern -- and therefore
    the half orders it generates -- survives the loop point instead of being
    cut mid-cycle.

    Mono on purpose: the engine is in front of the driver and belongs in the
    middle of the image, unlike the road layers which surround them.
    """
    f0 = rpm / 60.0 * ENGINE_CYLINDERS / 2.0          # firings per second
    cycles = max(1, int(round(f0 * seconds / ENGINE_CYLINDERS)))
    fires = cycles * ENGINE_CYLINDERS
    n = _smooth_length(int(round(fires * rate / f0)))
    period = n / float(fires)                          # samples per firing

    # One impulse per firing, placed at a FRACTIONAL sample position and split
    # across the two neighbouring samples. Rounding to the nearest sample
    # instead would quantise the firing interval to 23 us steps, which at these
    # periods is an audible mistuning of the higher orders.
    exc = np.zeros(n)
    for i in range(fires):
        pos = (i * period
               + ENGINE_TIMING_JITTER * period * float(rng.standard_normal()))
        amp = ENGINE_CYLINDER_TRIM[i % len(ENGINE_CYLINDER_TRIM)] * (
            1.0 + ENGINE_AMPLITUDE_JITTER * float(rng.standard_normal()))
        base = int(math.floor(pos))
        frac = pos - base
        exc[base % n] += amp * (1.0 - frac)
        exc[(base + 1) % n] += amp * frac

    # Combustion noise, loudest just after each firing and decaying between
    # them: white noise windowed by the impulse train convolved with an
    # exponential. Circular again, so the tail of the last firing gates the
    # noise at the top of the loop just as it gates the impulses.
    decay = np.exp(-np.arange(n) / max(1.0, ENGINE_NOISE_DECAY_S * rate))
    gate = np.fft.irfft(np.fft.rfft(np.abs(exc)) * np.fft.rfft(decay), n)
    noise = rng.standard_normal(n) * np.maximum(gate, 0.0)
    exc_rms = float(np.sqrt(np.mean(exc ** 2)))
    noise_rms = float(np.sqrt(np.mean(noise ** 2)))
    if noise_rms > 0.0 and exc_rms > 0.0:
        noise *= ENGINE_NOISE_LEVEL * exc_rms / noise_rms

    # The body. Same impulse response at every rpm -- that is what makes the
    # spectral envelope stationary while the orders move through it.
    #
    # The 2*pi*f/Q factor turns the tabulated gain into the RESPONSE at
    # resonance, which is what makes the table mean what it says. A decaying
    # sinusoid of unit amplitude has |H(f)| = Q/(2*pi*f) at its peak, so writing
    # the amplitudes directly buries a 1/f tilt in the table: the 2450 Hz entry
    # at gain 0.10 would land 40 dB below the number written next to it, and the
    # layer comes out with no top at all -- measured, that was -48 dB in the
    # 2-4 kHz octave, i.e. an engine with a blanket over it.
    t = np.arange(n) / float(rate)
    body = np.zeros(n)
    for freq, q, gain in ENGINE_RESONANCES:
        body += gain * (2.0 * np.pi * freq / q) * np.exp(
            -np.pi * freq * t / q) * np.sin(2.0 * np.pi * freq * t)

    freqs = np.fft.rfftfreq(n, 1.0 / rate)
    # Zero-phase band limits. Applied to the transfer function rather than to
    # the excitation so the impulses themselves stay sharp.
    shape = (freqs ** 2) / (freqs ** 2 + ENGINE_LOW_HZ ** 2)
    shape /= np.sqrt(1.0 + (freqs / ENGINE_HIGH_HZ) ** 4)

    sig = np.fft.irfft(
        np.fft.rfft(exc + noise) * np.fft.rfft(body) * shape, n)

    rms = float(np.sqrt(np.mean(sig ** 2)))
    if rms > 0.0:
        sig *= ENGINE_RMS / rms
    return _soft_limit(sig)


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


def asset_level(speed_kmh):
    """Overall level for the RECORDED path at this speed.

    Deliberately gentler than the synthesised curves. There, level was doing all
    the work of conveying speed; here the clips themselves carry the character
    -- a 100 km/h recording already sounds like 100 km/h -- so this only has to
    supply the loudness difference the RMS matching in ambience_assets removed.
    Overdriving it would undo the point of having recordings.
    """
    v = min(max(speed_kmh, 0.0) / SPEED_FULL_KMH, 1.0)
    return (ASSET_REST_LEVEL
            + (ASSET_FULL_LEVEL - ASSET_REST_LEVEL) * v ** ASSET_LEVEL_EXPONENT)


def asset_weights(speed_kmh, voice_speeds):
    """Equal-power crossfade weights across the clips, bracketing this speed.

    Equal-power (sqrt) rather than linear because these beds are uncorrelated
    recordings: a linear crossfade would dip ~3 dB halfway between two clips, so
    holding a speed exactly between them would sound like a hole rather than a
    blend. Outside the recorded range the nearest clip simply holds.
    """
    count = len(voice_speeds)
    weights = [0.0] * count
    if count == 0:
        return weights
    if count == 1 or speed_kmh <= voice_speeds[0]:
        weights[0] = 1.0
        return weights
    if speed_kmh >= voice_speeds[-1]:
        weights[-1] = 1.0
        return weights
    for i in range(count - 1):
        lo, hi = voice_speeds[i], voice_speeds[i + 1]
        if lo <= speed_kmh <= hi:
            span = hi - lo
            f = (speed_kmh - lo) / span if span > 0 else 0.0
            weights[i] = math.sqrt(1.0 - f)
            weights[i + 1] = math.sqrt(f)
            break
    return weights


def engine_bucket_rpm(index):
    """Engine speed bucket ``index`` is rendered at. Geometric, see ENGINE_BUCKETS.

    The one definition of the bucket ladder: Ambience builds from it and
    engine_blend() inverts it, so the two cannot disagree about which rpm a
    bucket holds.
    """
    frac = index / float(max(1, ENGINE_BUCKETS - 1))
    return IDLE_RPM * (REDLINE_RPM / IDLE_RPM) ** frac


def engine_blend(rpm):
    """(lo bucket, hi bucket, hi weight) for the crossfade at this engine speed.

    The inverse of engine_bucket_rpm(), hence the log: a linear position would
    put the blend in the wrong place at every rpm but the two ends, and audibly
    so low down where the buckets are furthest apart in Hz.
    """
    rpm = min(max(float(rpm), IDLE_RPM), REDLINE_RPM)
    pos = (math.log(rpm / IDLE_RPM) / math.log(REDLINE_RPM / IDLE_RPM)
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
                 loop_seconds=LOOP_SECONDS, assets_dir=None):
        self.requested_gain = max(0.0, float(gain))
        self.seed = int(seed)
        self.effective_gain = 0.0
        # What was actually played, for the label rows: 'off', 'synth', or the
        # short hash of the recorded clip set. See the `source` property.
        self.assets = None
        self._road = []           # [(channel, sound, rest, full, exponent)]
        self._engine_sounds = []
        self._engine_ch = [None, None]
        self._engine_bucket = [-1, -1]
        self._voices = []         # [(speed_kmh, channel, sound)] when recorded
        self._speed = 0.0
        self._throttle = 0.0
        self._duck = 1.0
        self._duck_target = 1.0
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

            started_ms = pygame.time.get_ticks()

            # Recordings win when they are present. The synthesiser sounds like
            # *a* car; a recording sounds like *this* car, and no amount of
            # tuning closes that gap -- so the synth is the fallback for a
            # machine that has not copied the clips, not the preferred path.
            self.assets = ambience_assets.load(assets_dir, rate, channels,
                                               SPEED_FULL_KMH)
            if self.assets is not None:
                needed = len(self.assets)
                if pygame.mixer.get_num_channels() < needed:
                    pygame.mixer.set_num_channels(needed)
                for idx, (speed, sound) in enumerate(self.assets.voices):
                    ch = pygame.mixer.Channel(idx)
                    ch.set_volume(0.0)
                    ch.play(sound, loops=-1)
                    self._voices.append((speed, ch, sound))
                build_ms = pygame.time.get_ticks() - started_ms
                self.effective_gain = self.requested_gain
                self._started = True
                self.update(0.0)
                print('[INFO] Ambience on (recorded): gain=%.2f %s, %d Hz, '
                      '%d ch, loaded in %d ms. Fix the gain AND the physical '
                      'volume across all participants and both study arms, and '
                      'report the measured dB(A), not this number.'
                      % (self.effective_gain, self.assets.describe(), rate,
                         channels, build_ms))
                return

            # One channel per road layer plus two for the engine crossfade.
            needed = len(ROAD_LAYERS) + 2
            if pygame.mixer.get_num_channels() < needed:
                pygame.mixer.set_num_channels(needed)

            rng = np.random.default_rng(self.seed)

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
                cycle = _engine_cycle(engine_bucket_rpm(i), ENGINE_LOOP_S,
                                      rate, rng)
                self._engine_sounds.append(
                    pygame.sndarray.make_sound(
                        _to_int16(cycle, channels, normalise=False)))

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
        print('[INFO] Ambience on (synthesised): gain=%.2f seed=%d (%d road '
              'layers + %d rev buckets, %d Hz, %d ch, built in %d ms). Fix the '
              'gain AND the physical volume across all participants and both '
              'study arms, and report the measured dB(A), not this number.'
              % (self.effective_gain, self.seed, len(self._road),
                 ENGINE_BUCKETS, rate, channels, build_ms))

    def _silence(self):
        for _speed, ch, _sound in self._voices:
            try:
                ch.stop()
            except pygame.error:
                pass
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
        self._voices = []
        self._road = []
        self._engine_sounds = []
        self._engine_ch = [None, None]
        self._started = False

    @property
    def active(self):
        return self._started

    @property
    def source(self):
        """What was played, for the label rows.

        'off' when there was no audio, the clip set's short hash when it came
        from recordings, 'synth' when it came from the synthesiser. This is the
        column that makes a run interpretable after the fact: with recordings
        the seed means nothing, and a clip swapped halfway through a study is
        otherwise an undetectable change to the stimulus.
        """
        if not self._started:
            return 'off'
        return self.assets.assets_id if self.assets is not None else 'synth'

    @property
    def rpm(self):
        """Current simulated engine speed. Diagnostic only."""
        return _rpm_for(self._speed, self._gear)

    def set_ducked(self, ducked):
        """Fade the whole bed toward silence (True) or its normal level (False).

        Independent of speed: passing 0.0 to ``update`` settles the mix to
        idle -- engine running, car stopped -- which is right for "the scene
        is frozen" but not for "this should be silent". Call this every frame
        from whichever branch is driving the call to ``update`` (popup open,
        popup closed, session ended) rather than only on the transition, so
        ducking always matches the current state and can never stick from a
        state that was skipped (e.g. a popup that closes into a session-end
        overlay without a normal-driving frame in between).
        """
        self._duck_target = 0.0 if ducked else 1.0

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
            self._duck = self._duck_target
        else:
            dt = max(0, now - self._last_ms) / 1000.0
            if dt > 0.0:
                self._speed += (1.0 - math.exp(-dt / SPEED_TAU_S)) * (
                    target_speed - self._speed)
                self._throttle += (1.0 - math.exp(-dt / THROTTLE_TAU_S)) * (
                    target_throttle - self._throttle)
                self._duck += (1.0 - math.exp(-dt / DUCK_TAU_S)) * (
                    self._duck_target - self._duck)
        self._last_ms = now

        if self._voices:
            self._update_recorded()
            return

        self._gear, rpm, road_levels, engine_level = mix_levels(
            self._speed, self._throttle, self._gear)

        for (ch, _sound, _rest, _full, _exp), level in zip(self._road,
                                                           road_levels):
            try:
                ch.set_volume(self.effective_gain * self._duck * level)
            except pygame.error:
                pass

        level = self.effective_gain * self._duck * engine_level
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

    def _update_recorded(self):
        """Crossfade the recorded clips bracketing the current speed."""
        level = self.effective_gain * self._duck * asset_level(self._speed)
        weights = asset_weights(self._speed, [s for s, _c, _s in self._voices])
        for (_speed, ch, _sound), weight in zip(self._voices, weights):
            try:
                ch.set_volume(level * weight)
            except pygame.error:
                pass

    def stop(self):
        self._silence()
