# Cabin ambience clips

Drop recorded car-interior audio here and the drive uses it instead of the
synthesiser in `src/drive/ambience.py`. No code changes — the loader reads
whatever is in this directory at startup.

## Naming

`background_audio.wav` is the default drop-in name and is what is in use now. A
single untitled clip is a complete, supported setup: it gets expanded into five
gently pitch-shifted variants across the speed range, crossfaded as the car
speeds up.

For a better result, supply clips recorded at *different* speeds and tag them:

```text
interior_030.wav     recorded at ~30 km/h
interior_060.wav     recorded at ~60 km/h
interior_100.wav     recorded at ~100 km/h
```

(`background_audio_030.wav` works identically — either stem is accepted.) The
number is the speed the clip was recorded at, in km/h. At run time the two clips
bracketing the current speed are crossfaded, which is what makes the sound
change *character* with speed rather than just getting louder — the thing the
synthesiser could never quite do, and the reason multiple clips beat one.

Only the first **45 seconds** of any clip is used (`MAX_CLIP_SECONDS` in
`src/drive/ambience_assets.py`). Longer files cost startup time and a lot of
memory for no audible gain, since the loop repair joins the excerpt to itself
cleanly at any length.

`wav` and `ogg` are the safe formats. Avoid mp3: SDL's decoder support varies by
build, so a file that works on your laptop can fail on the rig. Sample rate, bit
depth and channel count are all converted on load, so they do not have to match
anything.

## Where to get them

[freesound.org](https://freesound.org), searching **"car interior"**, **"car
cabin driving"**, or **"onboard"**. You want an in-cabin recording at roughly
constant speed. Avoid exterior drive-by recordings: they have Doppler and a
perspective that sounds wrong from the driver's seat.

**Set the licence filter to CC0.** Avoid CC-BY-NC in particular — a
non-commercial clause with an industrial partner involved is a real question
rather than a formality, and CC0 costs nothing. CC-BY is fine as long as the
attribution actually reaches the thesis, which is what `manifest.json` below is
for.

Clips do not need trimming or looping first. The loader crossfades each one's
tail over its head so it loops without a click, which matters more here than it
sounds: a click once per loop is a periodic startle that participants stop
noticing consciously and keep responding to physiologically — landing directly
in `hr_delta` / `rr_delta`, which are model inputs.

## manifest.json

```json
{
  "clips": {
    "interior_060.wav": {
      "freesound_id": "123456",
      "author": "somebody",
      "licence": "CC0",
      "url": "https://freesound.org/s/123456/"
    }
  }
}
```

The drive warns at startup about any clip missing an entry, and prints a
ready-to-paste stub if the file is absent entirely. This doubles as the
attribution appendix for the thesis.

## What gets logged

Every row of `data/user_loa_labels.csv` carries `ambient_source`: the short hash
of the clip set when recordings were used, `synth` when the synthesiser was,
`off` when there was no audio. Swapping a clip changes the hash, so a stimulus
change mid-study is visible in the data instead of silently confounding it.
`ambient_seed` only reproduces the synthesised path and means nothing for
recordings.

## Committing

Commit the clips. They are small, and unlike model weights they are an
experimental *stimulus*: they have to be byte-identical on both machines and
citable in the write-up. This directory is deliberately not gitignored.
