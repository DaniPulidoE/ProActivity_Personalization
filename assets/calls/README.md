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

## Icon

`viber.png` (CC0, credited in `manifest.json`) is the receiver glyph drawn in
every panel header. Two things about how it is consumed:

- **It is knockout art.** The rounded badge is opaque and the handset is punched
  *through* it to transparency, so blitting the file paints a solid block with a
  hole in it. `_load_glyph_mask` recovers the hole as an alpha mask, which is
  then tinted — so the icon recolours with the panel (neutral on the driver's
  card, assistant blue on the assistant's) rather than being a fixed sticker.
  Any replacement in the same knockout style works unchanged; a normal
  filled-glyph PNG would need the inversion removed.
- **It already contains the ring arcs**, so the icon looks the same whether the
  phone is ringing or connected. The drawn fallback in `_draw_handset` still
  varies, and takes over if the file is missing or numpy is unavailable.

Only the rendered handset is used — no brand mark appears in the panel — but the
filename is worth changing to something neutral if the asset set grows.

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
