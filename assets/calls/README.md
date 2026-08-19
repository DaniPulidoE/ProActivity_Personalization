# Call event audio

Clips for `src/drive/call_event.py` (the live study's interactive event —
see `docs/live_study_setup.md` §5.4, §6.4). Drop the files in; no code changes.

| File | Content |
|---|---|
| `ring.wav` | The phone ringing. Played in **every** condition, identically — it marks the world event, not the assistant's involvement. |
| `loa1_line.wav` | "Call from Mark." |
| `loa2_line.wav` | "Call from Mark. Want me to answer?" |
| `loa3_line.wav` | "Call from Mark. Answering in 3, 2, 1." |
| `loa4_line.wav` | "Call from Mark. Answering now." |
| `caller_reply.wav` | The caller's canned line (~2–3 s), played whenever the call is answered — including after a manual ACCEPT at LoA 0/1. |

Rules that are part of the experimental design, not preferences:

- **Render offline, once.** No TTS at run time: `pyttsx3.runAndWait()` blocks and
  the engine needs re-initialising per utterance, which stutters the render loop.
- **Keep `loa1`–`loa4` close in duration**, or speech length becomes a second
  manipulated variable riding along with autonomy.
- **One caller name, fixed** across every window, condition and participant, and
  neutral — "Mum" or "Boss" imports urgency and social obligation that would
  swamp the manipulation (Privacy 4 / Social Risk 3 in this function's FCD).
- **`caller_reply.wav` is identical everywhere.** Only the *path* to being
  answered varies between conditions; the outcome must not.

Missing files are not fatal — `call_event.py` warns once and runs silently, so
the interaction can be piloted before the audio exists. Override the directory
with `PROVOICE_CALL_ASSETS_DIR`.
