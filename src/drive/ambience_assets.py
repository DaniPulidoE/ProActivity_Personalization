#!/usr/bin/env python

"""Recorded cabin ambience: discovery, loop repair, provenance.

The synthesiser in ambience.py sounds like *a* car. A recording sounds like
*this* car, and no amount of tuning closes that gap -- so if clips are present
they win, and the synthesiser becomes the fallback for a machine that has not
copied them.

DROP FILES IN, NO CODE CHANGES
------------------------------
    assets/ambience/
        manifest.json
        background_audio.wav  <- the default drop-in name
    or
        interior_030.wav      <- recorded at ~30 km/h
        interior_060.wav
        interior_100.wav

The number in the filename is the speed the clip was recorded at, in km/h. At
run time the two clips bracketing the current speed are crossfaded, so the
character of the sound moves with speed instead of one bed just getting louder
-- which is the thing that made the synthesised version read as wind.

ONE clip is a complete, supported setup: `background_audio.wav` on its own (or
`interior.wav`, or a single tagged clip) is expanded into SINGLE_CLIP_VOICES
gently pitch-shifted variants spanning the speed range. That is a garnish on the
level curve, not a substitute for real recordings at different speeds -- the
range is deliberately small, because a recording stretched far enough to cover
0-90 km/h sounds like a tape machine, not a car.

WHY THE LOOP REPAIR IS NOT OPTIONAL
-----------------------------------
Freesound clips are not loop-ready: the last sample rarely continues into the
first, so `loops=-1` clicks once per pass. That click is the worst possible
artifact for THIS study -- a periodic startle every N seconds, which
participants stop noticing consciously and keep responding to physiologically,
landing directly in hr_delta and rr_delta, which are model inputs.
_make_loopable() fixes it by crossfading the tail over the head (equal-power,
so noise-like material holds its level through the blend), which means any clip
can be dropped in without hand-editing it in an audio editor first.

PROVENANCE IS PART OF THE DATA
------------------------------
With synthesis, ambient_seed reproduced the sound exactly. With recordings the
guarantee lives in the files, so this module hashes them and hands back a short
assets_id that Drive writes into every label row. Swapping a clip mid-study
changes that id; without it the swap is an undetectable confound and the thesis
cannot state what the stimulus was. manifest.json carries the human half --
Freesound id, author, licence, url -- and doubles as the attribution appendix.

Licence note for whoever fills that manifest in: prefer CC0. Avoid CC-BY-NC in
particular -- a non-commercial clause with an industrial partner involved is a
real question rather than a formality, and CC0 costs nothing.
"""

import hashlib
import json
import os
import re

import numpy as np
import pygame

# background_audio.wav (the default drop-in name) or interior_60.wav /
# interior_060.ogg / interior.wav. The speed tag is optional on both: without
# one the clip is used in single-clip mode.
CLIP_PATTERN = re.compile(
    r'^(?:background_audio|interior)(?:_(\d{1,3}))?\.(wav|ogg|flac)$',
    re.IGNORECASE)

MANIFEST_NAME = 'manifest.json'

# Longest excerpt actually used from any one clip. Downloads are routinely
# minutes long, and every second is paid for several times over: single-clip
# mode builds SINGLE_CLIP_VOICES resampled variants, each of which is copied
# again by the loop repair and the normalisation. A three-minute 48 kHz stereo
# file costs about a gigabyte of transients and several seconds of startup that
# way, for a bed nobody can tell from a shorter one -- the loop repair means the
# excerpt joins to itself cleanly whatever length it is.
#
# Long enough that the repetition is not perceptible as a rhythm; raise it if a
# recording has slow structure worth keeping.
MAX_CLIP_SECONDS = 45.0

# Tail-over-head crossfade used to make an arbitrary clip loop seamlessly.
# Long enough to hide a mismatch in broadband material, short enough not to eat
# a clip with a short usable section.
LOOP_CROSSFADE_S = 0.75

# Every clip is normalised to this RMS (about -18 dBFS), NOT to its peak and NOT
# left at its recorded level. Recorded levels off Freesound reflect whatever mic
# gain the uploader used, so a 30 km/h clip is perfectly capable of arriving
# louder than a 100 km/h one -- which would invert the speed cue and make the
# crossfade lurch. Matching RMS makes the crossfade transparent and leaves the
# speed-to-loudness relationship to the level curve in ambience.py, where it is
# explicit and tunable.
TARGET_RMS = 0.12

# Single-clip mode: how many pitch variants, and the ratio at rest / at full
# speed. Kept subtle on purpose (see the module docstring). Set the range to
# (1.0, 1.0) to disable pitch tracking entirely.
SINGLE_CLIP_VOICES = 5
SINGLE_CLIP_PITCH = (0.94, 1.10)

# Where clips live when nothing overrides it. Resolved from THIS file rather
# than the working directory: Drive is started from more than one place, and a
# cwd-relative default would silently fall back to synthesis depending on where
# the operator happened to be standing.
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                          '..', '..'))
DEFAULT_SUBDIR = os.path.join('assets', 'ambience')
ENV_VAR = 'PROVOICE_AMBIENCE_DIR'


def default_directory():
    """Assets directory: explicit env var first, else <repo>/assets/ambience."""
    return os.environ.get(ENV_VAR) or os.path.join(_REPO_ROOT, DEFAULT_SUBDIR)


class AssetSet(object):
    """Loaded, loop-repaired clips plus the provenance Drive has to log."""

    def __init__(self, voices, clips, assets_id, directory):
        self.voices = voices          # [(speed_kmh, pygame.mixer.Sound)], sorted
        self.clips = clips            # [{name, sha256, speed_kmh, manifest}]
        self.assets_id = assets_id    # short hash of the whole set
        self.directory = directory

    def __len__(self):
        return len(self.voices)

    def describe(self):
        speeds = ', '.join('%.0f' % s for s, _ in self.voices)
        return ('%d clip(s) -> %d voice(s) at %s km/h, id=%s'
                % (len(self.clips), len(self.voices), speeds, self.assets_id))


def _sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def _discover(directory):
    """[(path, speed_or_None)] for every clip in the directory, sorted."""
    try:
        names = sorted(os.listdir(directory))
    except OSError:
        return []
    found = []
    for name in names:
        m = CLIP_PATTERN.match(name)
        if m:
            speed = float(m.group(1)) if m.group(1) is not None else None
            found.append((os.path.join(directory, name), speed))
    return found


def _load_samples(path):
    """Clip as float array in [-1, 1], shape (n, channels).

    SDL converts the file to the OPEN MIXER FORMAT here, so sample rate and
    channel count in the file do not have to match anything -- which is what
    lets an arbitrary Freesound download work untouched.
    """
    sound = pygame.mixer.Sound(path)
    arr = pygame.sndarray.array(sound).astype(np.float64)
    if arr.ndim == 1:
        arr = arr[:, None]
    return arr / 32768.0


def _resample(x, ratio):
    """Pitch-shift by playback-rate change. ratio > 1 is higher and shorter."""
    if abs(ratio - 1.0) < 1e-6:
        return x
    n = len(x)
    pos = np.arange(0.0, n - 1.0, ratio)
    src = np.arange(n, dtype=np.float64)
    return np.stack([np.interp(pos, src, x[:, c]) for c in range(x.shape[1])],
                    axis=1)


def _make_loopable(x, rate, crossfade_s=LOOP_CROSSFADE_S):
    """Crossfade the tail over the head so the clip loops without a click.

    The output is the clip minus the crossfade length. Its first samples are the
    original tail fading out under the original head fading in -- and because
    that tail is exactly what preceded the (now removed) end of the clip, the
    wrap from last sample to first is continuous by construction.

    Equal-power (sqrt) rather than linear: these beds are noise-like and
    uncorrelated across the blend, and a linear crossfade would dip ~3 dB in the
    middle, i.e. a periodic hole instead of a periodic click.
    """
    n = len(x)
    length = int(min(crossfade_s * rate, n // 4))
    if length < 32:
        return x
    t = np.linspace(0.0, 1.0, length, endpoint=False)[:, None]
    out = x[:n - length].copy()
    out[:length] = x[:length] * np.sqrt(t) + x[n - length:] * np.sqrt(1.0 - t)
    return out


def _normalize(x):
    rms = float(np.sqrt(np.mean(x ** 2)))
    if rms > 0.0:
        x = x * (TARGET_RMS / rms)
    peak = float(np.max(np.abs(x)))
    if peak > 0.99:
        x = x * (0.99 / peak)
    return x


def _to_sound(x, channels):
    if x.shape[1] == 1 and channels >= 2:
        x = np.repeat(x, channels, axis=1)
    elif x.shape[1] > channels:
        x = x[:, :channels]
    pcm = np.ascontiguousarray(np.rint(np.clip(x, -1.0, 1.0) * 32767.0)
                               .astype(np.int16))
    if channels == 1:
        pcm = np.ascontiguousarray(pcm[:, 0])
    return pygame.sndarray.make_sound(pcm)


def _read_manifest(directory):
    path = os.path.join(directory, MANIFEST_NAME)
    if not os.path.exists(path):
        return None
    try:
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as exc:
        print('[WARN] Ambience: %s could not be read (%s); provenance will not '
              'be checked.' % (path, exc))
        return None
    if isinstance(data, dict) and isinstance(data.get('clips'), dict):
        return data['clips']
    if isinstance(data, dict):
        return data
    print('[WARN] Ambience: %s is not an object keyed by filename.' % path)
    return None


def _report_provenance(directory, clips, manifest):
    """Warn about anything that would leave the thesis unable to cite a clip."""
    required = ('freesound_id', 'author', 'licence', 'url')
    if manifest is None:
        print('[WARN] Ambience: no %s in %s. The audio is logged by hash so the '
              'DATA stays interpretable, but nothing records where these clips '
              'came from or under what licence. Create it as:'
              % (MANIFEST_NAME, directory))
        stub = {'clips': {c['name']: {'freesound_id': '', 'author': '',
                                      'licence': 'CC0', 'url': ''}
                          for c in clips}}
        print(json.dumps(stub, indent=2))
        return
    for clip in clips:
        entry = manifest.get(clip['name'])
        if not isinstance(entry, dict):
            print('[WARN] Ambience: %s has no entry in %s -- it cannot be '
                  'attributed.' % (clip['name'], MANIFEST_NAME))
            continue
        clip['manifest'] = entry
        missing = [k for k in required if not str(entry.get(k, '')).strip()]
        if missing:
            print('[WARN] Ambience: %s is missing %s in %s.'
                  % (clip['name'], '/'.join(missing), MANIFEST_NAME))
        licence = str(entry.get('licence', '')).upper().replace(' ', '')
        if 'NC' in licence:
            print('[WARN] Ambience: %s is licensed %s. A non-commercial clause '
                  'is a real question with an industrial partner involved -- '
                  'prefer a CC0 replacement.' % (clip['name'],
                                                 entry.get('licence')))


def load(directory, rate, channels, speed_full_kmh, verbose=True):
    """Load an AssetSet, or return None to fall back to synthesis.

    Never raises: a broken or half-populated assets directory falls back rather
    than taking a participant's session down with it.
    """
    directory = os.path.abspath(directory or default_directory())
    found = _discover(directory)
    if not found:
        if verbose:
            print('[INFO] Ambience: no clips in %s, using the synthesiser. '
                  'Drop interior_<kmh>.wav files there to use recordings.'
                  % directory)
        return None

    tagged = [(p, s) for p, s in found if s is not None]
    untagged = [(p, s) for p, s in found if s is None]
    if tagged and untagged:
        for path, _ in untagged:
            print('[WARN] Ambience: ignoring %s -- untagged clips only make '
                  'sense on their own; name it interior_<kmh>%s to use it.'
                  % (os.path.basename(path),
                     os.path.splitext(path)[1]))
        found = tagged
    found.sort(key=lambda ps: (ps[1] if ps[1] is not None else 0.0))

    clips, loaded = [], []
    for path, speed in found:
        name = os.path.basename(path)
        try:
            samples = _load_samples(path)
        except Exception as exc:
            print('[WARN] Ambience: could not load %s (%s); skipping. wav and '
                  'ogg are the safe formats -- mp3 support varies by SDL build.'
                  % (name, exc))
            continue
        if len(samples) < rate // 2:
            print('[WARN] Ambience: %s is shorter than half a second; skipping.'
                  % name)
            continue
        limit = int(MAX_CLIP_SECONDS * rate)
        if len(samples) > limit:
            if verbose:
                print('[INFO] Ambience: using the first %.0f s of %s (%.0f s '
                      'on disk).' % (MAX_CLIP_SECONDS, name,
                                     len(samples) / float(rate)))
            samples = samples[:limit]
        samples -= samples.mean(axis=0, keepdims=True)  # field recordings drift
        loaded.append((speed, samples))
        clips.append({'name': name, 'sha256': _sha256(path),
                      'speed_kmh': speed, 'manifest': None})

    if not loaded:
        print('[WARN] Ambience: found clip files in %s but none could be used; '
              'falling back to the synthesiser.' % directory)
        return None

    voices = []
    if len(loaded) == 1:
        base_speed, samples = loaded[0]
        lo_ratio, hi_ratio = SINGLE_CLIP_PITCH
        count = max(1, SINGLE_CLIP_VOICES if hi_ratio > lo_ratio else 1)
        for i in range(count):
            f = i / float(count - 1) if count > 1 else 0.0
            ratio = lo_ratio + f * (hi_ratio - lo_ratio)
            # Resample BEFORE the loop repair: the repair is what guarantees the
            # wrap is continuous, and resampling afterwards would move the very
            # samples it just matched up.
            voice = _make_loopable(_resample(samples, ratio), rate)
            voices.append((f * speed_full_kmh, _to_sound(_normalize(voice),
                                                         channels)))
        if verbose and base_speed is not None:
            print('[INFO] Ambience: single clip tagged %.0f km/h; its speed tag '
                  'is unused in single-clip mode.' % base_speed)
    else:
        for speed, samples in loaded:
            voices.append((speed, _to_sound(
                _normalize(_make_loopable(samples, rate)), channels)))

    digest = hashlib.sha256()
    for clip in clips:
        digest.update(('%s:%s\n' % (clip['name'], clip['sha256'])).encode())
    assets_id = digest.hexdigest()[:8]

    _report_provenance(directory, clips, _read_manifest(directory))
    return AssetSet(voices, clips, assets_id, directory)
