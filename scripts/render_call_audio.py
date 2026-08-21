#!/usr/bin/env python

"""Render the call event's spoken lines to .wav, once, offline.

    uv run python scripts/render_call_audio.py
    uv run python scripts/render_call_audio.py --engine sapi --list-voices

WHY OFFLINE
-----------
pygame cannot synthesise speech -- ``pygame.mixer`` plays audio, nothing more.
So the lines are rendered HERE, by hand, and the .wav files are what ship. That
also makes the audio deterministic: byte-identical in every condition, block and
participant, which a study needs and a live synthesiser cannot promise.

ENGINES
-------
``piper``  (default) local neural TTS, ``pip install piper-tts``. Voices are
           ONNX files under ``models/piper``, fetched with
           ``python -m piper.download_voices NAME --data-dir models/piper``.
``sapi``   pyttsx3 / Windows SAPI5. Kept as a fallback, but UNRELIABLE for batch
           work: on this machine it hangs after the second file, with or without
           a fresh engine per line, exactly as pyttsx3's reputation and the
           comment in src/drive/simcall_simulation.py suggest. It also has only
           two voices installed, both female, so the caller ends up sounding
           like the assistant.

LICENSING
---------
Piper the ENGINE is GPL (it embeds espeak-ng). That governs the software, not
its output -- rendering wavs with it and shipping them is fine. The question
that does matter is the VOICE MODEL's licence, which is per-voice: check the
model card for whichever voice ships with the study, and record it in
assets/calls/manifest.json alongside the icon credits.

VOICE
-----
Assistant: a female en-US voice, the convention for in-car assistants.
Caller: a DIFFERENT voice, deliberately male by default -- with one voice the
driver hears the assistant announce the call and then answer itself, which
undercuts the whole "it acted on your behalf" framing at LoA 3 and 4.

THE VOICE LANGUAGE MUST MATCH THE PARTICIPANTS AND THE ON-SCREEN TEXT. The panel
is written in English; if the study runs in another language, both move together.
"""

import argparse
import os
import sys

import wave

try:
    from piper import PiperVoice, SynthesisConfig
except ImportError:                                           # pragma: no cover
    PiperVoice = SynthesisConfig = None

try:
    import pyttsx3
except ImportError:                                           # pragma: no cover
    pyttsx3 = None


_HERE = os.path.dirname(os.path.abspath(__file__))
ASSETS = os.path.join(_HERE, '..', 'assets', 'calls')
PIPER_MODELS = os.path.join(_HERE, '..', 'models', 'piper')
# LICENCE-DRIVEN CHOICE, not a quality one. Checked 2026-08-21 against the
# model cards at huggingface.co/rhasspy/piper-voices:
#
#   en_US-lessac-medium   Blizzard 2013 Lessac -- RESEARCH ONLY, no commercial
#                         use, and "the User is not permitted to allow the
#                         Materials to be used by any third party". Unusable.
#   en_US-ryan-medium     RyanSpeech -- CC BY-NC-SA 4.0 (non-commercial +
#                         share-alike).
#   en_US-hfc_female      Hi-Fi Captain -- CC BY-NC-SA 4.0.
#   en_US-amy-medium      mimic3-voices -- licence not stated on the card.
#   en_US-libritts_r      LibriTTS-R -- **CC BY 4.0**: attribution only,
#                         commercial and redistribution both permitted.
#
# NONE of the Piper English voices is CC0. LibriTTS-R is the cleanest available
# and the only one whose terms survive an industrial partner and a published
# artefact. It is multi-speaker (904), so ONE licence-clean model provides both
# the assistant and a distinctly different caller.
DEFAULT_PIPER_MODEL = 'en_US-libritts_r-medium'
DEFAULT_ASSISTANT_SPEAKER = 92
DEFAULT_CALLER_SPEAKER = 256
DEFAULT_PIPER_ASSISTANT = DEFAULT_PIPER_MODEL
DEFAULT_PIPER_CALLER = DEFAULT_PIPER_MODEL

# The assistant's lines, keyed by the file name call_event.py looks for.
#
# LoA 0 has no entry and never will: silence IS the content of "no assistive
# action is taken".
#
# LoA 3 says "unless you cancel", not "unless you veto". The on-screen button
# reads [B] Cancel, and a participant should not have to translate between the
# word they hear and the word they see -- "veto" is the design's vocabulary, not
# theirs. The screen shows the live countdown alongside, so the voice states the
# RULE and the display shows the CLOCK.
LINES = {
    'loa1_line.wav': 'Call from {caller}.',
    'loa2_line.wav': 'Call from {caller}. Want me to answer?',
    'loa3_line.wav': 'Call from {caller}. Answering unless you cancel.',
    'loa4_line.wav': 'Call from {caller}. Answering now.',
}

# The caller, once the call is answered. Same everywhere a call connects,
# including after a manual ACCEPT at LoA 0/1, so only the PATH to the outcome
# varies between conditions.
CALLER_LINE = ('caller_reply.wav', 'Hey, are you still coming tonight?')


def pick_voice(engine, wanted):
    for v in engine.getProperty('voices'):
        if wanted.lower() in v.name.lower():
            return v
    return None


def render_piper(jobs, outdir, model_dir):
    """Synthesise with Piper. One PiperVoice per model, reused across lines.

    `jobs` carries a speaker id in the rate slot for multi-speaker models; it is
    ignored by single-speaker ones.
    """
    if PiperVoice is None:
        sys.exit('piper-tts is not installed. `uv add piper-tts`, then '
                 '`uv run python -m piper.download_voices %s --data-dir %s`.'
                 % (DEFAULT_PIPER_ASSISTANT, model_dir))
    cache = {}
    for name, text, model, speaker in jobs:
        path = os.path.join(model_dir, model + '.onnx')
        if not os.path.exists(path):
            sys.exit('Voice model not found: %s\n'
                     'Fetch it with:  uv run python -m piper.download_voices '
                     '%s --data-dir %s' % (path, model, model_dir))
        if model not in cache:
            cache[model] = PiperVoice.load(path)
        voice = cache[model]
        out = os.path.join(outdir, name)
        kw = {}
        if getattr(voice.config, 'num_speakers', 1) > 1 and speaker is not None:
            kw['syn_config'] = SynthesisConfig(speaker_id=int(speaker))
        with wave.open(out, 'wb') as f:
            voice.synthesize_wav(text, f, **kw)
        with wave.open(out, 'rb') as f:
            secs = f.getnframes() / float(f.getframerate())
        print('  %-18s %6.2f s  [%s spk=%s]  %r'
              % (name, secs, model, speaker, text))


def render_sapi(jobs, outdir):
    """Synthesise with pyttsx3. See the module docstring: batch-unreliable."""
    if pyttsx3 is None:
        sys.exit('pyttsx3 is not installed.')
    for name, text, vid, rate in jobs:
        out = os.path.join(outdir, name)
        # A fresh engine per file. Reusing one hangs after the second call --
        # and on this machine a fresh one hangs there too, which is why piper
        # is the default.
        eng = pyttsx3.init()
        eng.setProperty('voice', vid)
        eng.setProperty('rate', rate)
        eng.save_to_file(text, out)
        eng.runAndWait()
        try:
            eng.stop()
        except Exception:                                     # noqa: BLE001
            pass
        del eng
        size = os.path.getsize(out) if os.path.exists(out) else 0
        print('  %-18s %6d B  %r' % (name, size, text))


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--engine', choices=('piper', 'sapi'), default='piper')
    ap.add_argument('--caller', default='Mark',
                    help='Caller name spoken in every line. Keep it FIXED and '
                         'neutral across the whole study (default: %(default)s).')
    ap.add_argument('--voice', default=None,
                    help='Assistant voice: a piper model name, or a substring '
                         'of a SAPI voice name. Default: %s / Zira.'
                         % DEFAULT_PIPER_ASSISTANT)
    ap.add_argument('--caller-voice', dest='caller_voice', default=None,
                    help='Voice for the caller line, deliberately DIFFERENT '
                         'from the assistant. Default: %s.'
                         % DEFAULT_PIPER_CALLER)
    ap.add_argument('--speaker', type=int, default=DEFAULT_ASSISTANT_SPEAKER,
                    help='Speaker id for the assistant on a multi-speaker piper '
                         'model (default: %(default)s of 904). AUDITION THIS: '
                         'the ids are anonymous LibriTTS reader numbers, so '
                         'which one sounds like an in-car assistant can only be '
                         'settled by listening.')
    ap.add_argument('--caller-speaker', dest='caller_speaker', type=int,
                    default=DEFAULT_CALLER_SPEAKER,
                    help='Speaker id for the caller (default: %(default)s). '
                         'Must be audibly DIFFERENT from --speaker.')
    ap.add_argument('--rate', type=int, default=170,
                    help='SAPI only, words per minute (default: %(default)s).')
    ap.add_argument('--caller-rate', dest='caller_rate', type=int, default=195,
                    help='SAPI only.')
    ap.add_argument('--outdir', default=ASSETS)
    ap.add_argument('--model-dir', dest='model_dir', default=PIPER_MODELS)
    ap.add_argument('--list-voices', dest='list_voices', action='store_true')
    args = ap.parse_args()

    outdir = os.path.abspath(args.outdir)
    os.makedirs(outdir, exist_ok=True)
    model_dir = os.path.abspath(args.model_dir)

    if args.list_voices:
        if args.engine == 'piper':
            print('Installed piper models in %s:' % model_dir)
            for f in sorted(os.listdir(model_dir)) if os.path.isdir(model_dir) else []:
                if f.endswith('.onnx'):
                    print('  %s' % f[:-5])
            print('\nMore:  uv run python -m piper.download_voices')
        else:
            for v in pyttsx3.init().getProperty('voices'):
                print('%-52s | %s' % (v.name, getattr(v, 'gender', '?')))
        return

    if args.engine == 'piper':
        assistant = args.voice or DEFAULT_PIPER_ASSISTANT
        caller = args.caller_voice or DEFAULT_PIPER_CALLER
    else:
        eng = pyttsx3.init()
        av = pick_voice(eng, args.voice or 'Zira')
        if av is None:
            sys.exit('No SAPI voice matches %r. Try --list-voices.' % args.voice)
        cv = pick_voice(eng, args.caller_voice) if args.caller_voice else av
        if cv is av:
            print('[WARN] caller uses the SAME voice as the assistant, at a '
                  'different rate -- the driver hears the assistant answer '
                  'itself. piper avoids this; sapi here has only two voices.')
        assistant, caller = av.id, cv.id

    print('engine: %s\nassistant voice: %s\ncaller voice: %s'
          % (args.engine, assistant, caller))

    a_arg = args.speaker if args.engine == 'piper' else args.rate
    c_arg = args.caller_speaker if args.engine == 'piper' else args.caller_rate
    jobs = [(name, text.format(caller=args.caller), assistant, a_arg)
            for name, text in sorted(LINES.items())]
    jobs.append((CALLER_LINE[0], CALLER_LINE[1], caller, c_arg))

    if args.engine == 'piper':
        render_piper(jobs, outdir, model_dir)
    else:
        render_sapi(jobs, outdir)

    print('\nWrote %d file(s) to %s' % (len(jobs), outdir))
    print('Check the loa1-loa4 DURATIONS above are close: speech length must '
          'not become a second variable riding along with autonomy.')


if __name__ == '__main__':
    main()
