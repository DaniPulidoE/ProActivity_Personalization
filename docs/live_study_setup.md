# Live follow-up study — setup

Design record for the satisfaction study that follows the population data
collection. Written 2026-08-19; supersedes the shorter sketch in `CLAUDE.md`
("Live Follow-Up Study"), which now points here for the operational detail.

Status markers used below: **DECIDED** — settled, implement against it.
**OPEN** — still needs a call, listed again under "Open decisions".

---

# 1. What the study measures

One question: **how much personalization data does a driver need before they are
satisfied with the system?**

It does *not* re-compare the two personalization arms. L2-SP and ANIL are
compared **offline** by leave-one-driver-out (MAE/QWK), and only the winner is
carried into the live study. Re-opening the arm comparison live would cost a
factor of two in conditions and reintroduce the deployment risk the offline
protocol exists to avoid.

Uncertainty is deliberately **not** wired into the live loop. The Laplace layer
runs offline only. Gating actuation on posterior width would confound
"personalization got better" with "the system got more confident and stopped
deferring" — which is precisely the measurement the study exists to produce.

---

# 2. Design

**One factor, three levels, within-subject.** The factor is **K**, the number of
the driver's own labels the served head was adapted on.

| | |
|---|---|
| Factor | K (personalization data) |
| Levels | 3 — K=0 plus two more (**OPEN**, see §3) |
| Design | within-subject, blocked: K held fixed for a block of windows |
| Counterbalancing | **all 3! = 6 block orders** |
| Participants | 12 (the population cohort, reused) = exactly 2 full replicates |

**Why all 6 orders and not a Latin square.** A 3×3 Latin square covers only 3 of
the 6 sequences — balanced for *position*, not for order pairs — and needs a
multiple of 3. Complete counterbalancing is strictly stronger, and 12 divides
evenly into 6. This matters more than usual because **K=0 is one of the levels**:
a driver who meets the unpersonalized condition *after* a well-personalized
block may rate it harsher by contrast than one who meets it first, so
first-order carryover is a live effect rather than a formality.

Assignment of participants to sequences must be **random and independent** of
their traffic-scenario pair from the population collection
(`TRAFFIC_SEED_PLAN`).

---

# 3. The three K conditions

**K=0 is the driver's own LODO population model, not a shared one.** Every K
level for participant *p* is served from a backbone that never saw *p*'s data.
Stage 2 of the offline pipeline (`training_scripts/run_lodo_population.py`)
already writes `trained_models/lodo/pop_heldout_<pid>.pt` ×12, and stage 3 adapts
the K>0 heads from those same files.

Two consequences:

- **K=0 costs no extra training.** It is a by-product of the offline analysis;
  the live study serves the stage-2 artifact directly.
- **The K contrast is clean by construction.** All three conditions for a driver
  share one backbone, so no part of the K effect can be "the backbone had already
  seen your data".

The fold used to mint a live checkpoint **must** be the same `folds.py` fold as
the offline sweep the K values were read off. Otherwise the served model does not
correspond to the curve that selected it.

**OPEN — the two non-zero K values.** They should be read off the offline
sweep curve, which is currently blocked: there is no `state_xlstm.pt`, the
widened-K ANIL sweep is queued but unrun, and τ is unsettled. The UI is buildable
ahead of this; only the checkpoint loader needs the final set.

## 3.1 Checkpoint naming and provenance

- Do **not** mint a `personalized_k0_*` copy. Nothing about the K=0 file is
  personalized, and `pop_heldout_<pid>.pt` already names it correctly.
- For K>0, mirror that convention and encode the **K value, zero-padded**
  (`k010`, `k030`) — not a condition index. `k1` does not record whether it meant
  10 labels or 30.
- The winning arm is not yet decided and both may sit on disk at once, so keep
  arm and τ in the name: `p001_l2sp_tau0.05_k010.pt`.
- Independently of the filename, record `participant_id`, `arm`, `tau`, `K` and
  the base checkpoint hash in the checkpoint's **`arch` dict**. The codebase
  already treats `arch` as the data contract (`head_type`, `context_length`,
  `window_seconds`, `resample_hz`); it survives a rename.
- The serving loader **must assert the held-out pid equals `--participantid`**.
  `run_lodo_population.py` already asserts a held-out driver is absent from its
  own training set at build time; the loader must carry the same guarantee at
  serve time. Serving 004's head to 003 is silent and unrecoverable afterwards.

---

# 4. Session structure

A loop of fixed **~20 s windows** (matching the model's own label window) at a
small set of pre-chosen, fixed spawn points, all under `STUDY_TRAFFIC_SEED = 42`
— already reserved in `start_experiment.py` for exactly this "fixed condition
across participants" purpose and excluded from the population-collection
scenarios, so the study also doubles as a genuine generalization test.

Per window:

1. A short **pre-set instruction** ("turn left, then right"). Deliberately not
   live turn-by-turn navigation — that is CARLA route-guidance engineering and
   bug risk for no measurement gain.
2. Drive ~20 s.
3. An **incoming phone call**, handled according to the predicted LoA (§5).
4. A **Likert satisfaction pop-up**, optionally also the driver's own accepted
   LoA level(s), reusing the existing labeling pop-up.

**What the Likert now measures.** With the interactive event, the rating is about
*how the interaction felt*, not *whether the driver agrees with a displayed LoA
number*. That is the better construct for the research question, but it means
these ratings are **not comparable** with anything in the offline analysis. State
this in the write-up rather than implying continuity.

---

# 5. The interactive event

**DECIDED: a simulated incoming phone call**, replacing the earlier plan of a
pop-up that merely *displays* the predicted LoA.

## 5.1 Why this function

`'Respond to a phone call'` is already in `RANDOM_FUNCTION_POOL`
(`drive_improved.py`) with a real FCD vector (`fcd_config.py` id 4), so the
population model was trained on labels for this exact function — the served head
is in-distribution for the staged event.

More decisively, it is the only pool function whose driver-chosen LoA mass sits
in the **middle** of the ladder. Measured over `data/user_loa_labels.csv`
(multi-label rows expanded; n = label occurrences):

| Function | n | L0 | L1 | L2 | L3 | L4 |
|---|---|---|---|---|---|---|
| **Respond to a phone call** | 338 | 17.2% | **23.4%** | **35.8%** | 21.6% | 2.1% |
| Respond to a text message | 337 | 33.2% | 33.2% | 19.9% | 11.6% | 2.1% |
| Provide traffic news | 313 | 29.4% | 19.5% | 4.5% | 8.0% | 38.7% |
| Provide weather update | 280 | 36.8% | 11.4% | 2.5% | 6.8% | 42.5% |
| Change song | 285 | 51.6% | 12.3% | 5.6% | 7.7% | 22.8% |

`Change song` is bimodal at the extremes — its middle rungs would almost never
fire and the study would degenerate into "did it do nothing, or everything". The
call exercises exactly the region where a 1-level MAE difference is meaningful.

Two planning consequences: rungs **1, 2 and 3 are ~81% of what gets served**, so
that is where the design effort belongs; and **LoA 4 is 2.1%** (≈7 of 338), so
keep it for completeness but do not invest in it — it is also the cheapest
rendering, having no input path.

## 5.2 The ladder

The call is a **world event**, not a system action. The phone rings in *every*
condition, identically. What the LoA governs is how involved the assistant gets.

This is load-bearing: if LoA 0 produced no on-screen event at all, the driver
would perceive nothing and the Likert would have no referent — and since
unpersonalized heads plausibly predict LoA 0 more often, the unratable windows
would concentrate in the K=0 condition. That is missing data correlated with the
independent variable.

| LoA | Ring | Assistant says | Driver's affordance | If driver does nothing |
|---|---|---|---|---|
| 0 | ✔ | *(silent)* | ACCEPT / DECLINE | rings out |
| 1 | ✔ | "Call from Mark." | ACCEPT / DECLINE | rings out |
| 2 | ✔ | "Call from Mark. Want me to answer?" | YES / NO | **not** answered |
| 3 | ✔ | "Call from Mark. Answering in 3… 2… 1…" | CANCEL | **answered** |
| 4 | ✔ | "Call from Mark. Answering now." | none | answered (already) |

The "does nothing" column is the cleanest formal statement of the ladder: no
input yields *nothing* for 0–2 and *the action* for 3–4, and that flip is exactly
what "veto" means.

## 5.3 Why LoA 1 and LoA 2 are separated visually, not verbally

For a binary, immediate action, "suggest" and "ask approval" **collapse in
natural language** — any natural phrasing of a suggestion about answering a call
("want to take it?", "worth taking?") *is* asking approval. No wording separates
these two rungs, and they are the two most common labels for this function.

So the distinction is carried by **who owns the screen**:

| | LoA 0 | LoA 1 | LoA 2 |
|---|---|---|---|
| Widget | phone card | phone card | **assistant dialogue** |
| Whose UI | driver's | driver's | assistant's |
| Assistant voice | silent | "Call from Mark." | "Call from Mark. Want me to answer?" |
| Recommendation | none | **ACCEPT badged ★ RECOMMENDED** | implicit in the offer |
| Buttons | ACCEPT / DECLINE | ACCEPT / DECLINE | YES / NO |
| Phone state | ringing | ringing | ringing, **held by the assistant** |

At LoA 1 the driver is operating their phone and the assistant *annotates the
driver's own control* — the same idiom as a suggested-reply chip. The assistant
never enters the interaction loop. At LoA 2 the phone card is replaced by a
visually distinct assistant panel and the driver answers the *assistant*.

**0 → 1 changes one marker. 1 → 2 changes the entire panel.** That asymmetry is
deliberate and mirrors the semantics the repo already encodes: `_LOA_POLICY`
groups `0:low, 1:low, 2:medium, 3:high, 4:high`, so the visual hinge sits exactly
where the policy hinge sits — 0 and 1 are the driver acting, 2 onward is the
assistant acting.

**Pilot check.** Verify on one pilot participant that they can articulate the
1-vs-2 difference unprompted. No amount of analysis recovers this afterwards.

## 5.4 Fixed elements

- **Direction of the suggestion is always "answer"**, never "let it go to
  voicemail". If direction varies, the ladder mixes autonomy with action-choice
  and stops being a clean autonomy manipulation.
- **One caller, fixed name, every window and participant.** Pick a neutral one —
  "Mum" or "Boss" imports urgency and social obligation that would swamp the
  manipulation, and both are live FCD dimensions for this function (Privacy 4,
  Social Risk 3).
- **What "answered" sounds like**: a short canned caller line (~2–3 s) then an
  automatic hang-up. Identical wherever a call gets answered, including after a
  manual ACCEPT at LoA 0/1, so only the *path* to the outcome varies. Without
  this, LoA 4 gives the driver nothing to perceive at all.
- **Timing**: fire at a fixed offset from window start (~t+5 s) with a hard cap
  (~8 s) so the interaction resolves before the rating pop-up opens.
- **One event type only** — call, no SMS. No dependence on live driving state.

Accept knowingly: conditions will differ in duration (LoA 4 resolves in ~2 s,
LoA 0–2 only on driver action or timeout). That is intrinsic — autonomy *is*
partly a claim about interaction time. Hold the **onset** identical and let only
the resolution vary, and keep the spoken lines close in length so speech duration
is not a second hidden variable.

---

# 6. UI specification

## 6.1 Placement — HUD band, right of the speed

`_render_speed` (`drive_improved.py`) puts the readout at
`y = 0.8·height − block_h`, at the base of the windshield. The call panel sits
**to its right, on the same baseline**, with the speed nudged further left to
open the room (decided 2026-08-19 from looking at the running UI).

```
                    horizon ≈ 0.45 h
    ─────────────────────────────────────────
                                              ← road, keep clear

                    ╔══════════════════════════╗   ≈ 0.66 h
                    ║ ◆ ASSISTANT              ║
                    ║   "Call from Mark.       ║    fixed box, same size
                    ║    Want me to answer?"   ║    for all five LoAs
      120  km/h     ║  [A] Yes      [B] No     ║
    ────────────────╚══════════════════════════╝   ≈ 0.80 h  ← shared baseline
         ↑ moved left                    (steering wheel / dash below)
```

Two elements in one horizontal instrument row, which is how real clusters read,
and it uses space that is already empty: the speed sits at ~29% from the left
(`x = (width − block_w) // 3.5`) and the `_show_info` telemetry column is on the
*left* edge, so the right two-thirds of the band is unused.

Three constraints on the move:

- **Anchor both elements in absolute coordinates.** `block_w` is
  `value.get_width() + pad + unit.get_width()`, so the current formula makes the
  speed block **drift horizontally as the digit count changes** (9 → 10 → 100
  km/h). The existing code fixes this only for the unit — its comment says the
  unit sits on the baseline "so it does not bounce as the number changes width" —
  not for the block. A panel positioned *relative* to the speed would inherit
  that jitter and break the fixed-footprint invariant of §6.3. Pin both to
  constants.
- **Bottom-align the panel to the speed baseline; do not centre it on the line.**
  The speed is at `0.8·h` precisely because that is the base of the windshield —
  below it is the rendered steering wheel. A five-row panel centred on that line
  would reach ~0.87 h and overlap the wheel. Growing upward from the shared
  baseline puts it at roughly 0.66 h–0.80 h, in the windshield.
- **Fix `// 3.5` and its docstring in the same edit.** The docstring claims
  "Centered horizontally" while the code divides by 3.5, placing it at ~29%.
  Since the placement is being changed anyway, make the code and the comment
  agree rather than leaving the next reader to trip on it.

The **backing plate behind the speed digits is commented out**; its own comment
explains why it existed — *"white text over a bright road surface is unreadable
exactly when the driver is looking for it."* A five-row panel needs that plate.
Reuse the pattern (`pygame.Surface`, `set_alpha(110)`, black fill).

## 6.2 The five renderings

Marker **and** colour in every state, per the codebase convention — state must
survive a projector or a colour-blind participant.

```
LoA 0 — driver's phone, assistant absent
┌────────────────────────────────┐
│  ▓ INCOMING CALL               │   neutral grey border
│    Mark                        │
│  [A] Answer      [B] Decline   │
└────────────────────────────────┘

LoA 1 — identical card plus a recommendation marker
┌────────────────────────────────┐
│  ▓ INCOMING CALL          ◆    │   ◆ = assistant present, passive
│    Mark                        │
│  [A] Answer  ★ RECOMMENDED     │   ★ + blue (140,200,255)
│  [B] Decline                   │
└────────────────────────────────┘

LoA 2 — the assistant's panel replaces the phone
╔════════════════════════════════╗
║ ◆ ASSISTANT                    ║   double border, assistant colour
║   "Call from Mark.             ║
║    Want me to answer?"         ║
║  [A] Yes           [B] No      ║   dialogue verbs, not phone verbs
╟────────────────────────────────╢
║  ▓ Mark — ringing, on hold     ║   phone demoted to a status strip
╚════════════════════════════════╝

LoA 3 — countdown, only a way out
╔════════════════════════════════╗
║ ◆ ASSISTANT                    ║
║   "Call from Mark.             ║
║    Answering in 2…"            ║
║   ██████████░░░░░░             ║   draining bar — appears nowhere else
║  [B] Cancel                    ║   no affirmative button exists
╟────────────────────────────────╢
║  ▓ Mark — ringing, on hold     ║
╚════════════════════════════════╝

LoA 4 — no affordance, already done
╔════════════════════════════════╗
║ ◆ ASSISTANT                    ║
║   "Answering now."             ║
╟────────────────────────────────╢
║  ▓ Mark — connected     00:03  ║
╚════════════════════════════════╝
```

## 6.3 Invariants

- **Fixed footprint.** Same box, same position, same size for all five; rows
  appear and disappear *inside* it. If LoA 4's panel is visibly smaller than
  LoA 3's, panel size becomes a salience confound alongside the content.
- **Fixed button meaning.** `[A]` is always the affirmative (Answer / Yes),
  `[B]` always the negative (Decline / No / Cancel). At LoA 3 there is no `[A]`
  — doing nothing *is* the affirmative, which is the whole content of "veto".
  If ACCEPT and YES were different physical buttons, a motor confound would ride
  along with the condition.
- **Does not freeze the sim.** Unlike the label pop-up, which blits a full-screen
  dimming overlay and holds the clock, the driver keeps driving throughout.
- **`_show_info` off during sessions.** The 220 px telemetry column down the left
  edge is a distractor competing with the event being timed.

## 6.4 Audio

Pre-render every line to `.wav` **offline** and play through `pygame.mixer`,
which the drive process already initializes via `ambience.configure_mixer()`.

Keep `pyttsx3` out of the live path: `runAndWait()` blocks, and
`simcall_simulation.py`'s own comment concedes the engine only plays once unless
re-initialized. A blocking TTS call anywhere near the render loop stutters the
sim. The existing strings there are also German — check against the participants.

Duck the ambience bed while call audio plays. `Ambience.set_ducked(True/False)`
already exists and is already used for the label pop-up.

---

# 7. Two-machine setup and LoA transport

Same rig as the population collection: **CARLA on the GPU machine**
(`192.168.50.1`, simulator + NPC traffic + Drive UI + both bridges), **ProVoice
on the laptop** (`192.168.50.2`, perception + models), joined by a dedicated
Ethernet cable. See `docs/remote_setup.md` for addressing, adapters and firewall.

**The reverse channel already exists.** TCP **8081** already runs ProVoice →
CARLA, currently carrying `collection_started` / `provoice_ended`. The predicted
LoA goes on that bridge — no third listener.

**Push, don't pull.** ProVoice pushes the decision; the CARLA side holds it and
`drive_improved` reads it locally. A pull would put a socket call on the render
thread at the exact moment the event fires, so a network stall would hitch the
sim in the one window being measured.

**One decision per window, read just before the call.** That is ~0.05 Hz. For
context, the two machine bugchecks of 2026-07-28 (0x1E, then 0xD1) were traced to
the HTTP bridge's ~20 TCP connections *per second*; this design sits four hundred
times below that, and stays there only if it never drifts into polling.

**Send the minimum.** Only the integer LoA and the frame timestamp — not `probs`
(the full 5-way PMF is a much richer function of driver state), not `message`,
not `fcd`.

> **Ethics paperwork must be updated.** `remote_setup.md` currently states, in
> the section it says the ethics paperwork has to match: *"No participant data
> crosses the link. Camera frames and everything derived from them … stay
> there."* The predicted LoA **is** derived from those signals, so that sentence
> becomes literally false. In substance one integer in 0–4 is about as
> de-identified as a derived value gets, but the claim is categorical and both
> the doc and possibly the submission need revising rather than quietly
> diverging.

**Freshness, not just availability.** `decisions.csv` already stamps the *frame
timestamp the decision was computed from*, not the wall clock. Carry that across
and check it at call time, or a ProVoice hiccup silently serves an LoA computed
from driver state 30 s stale.

**Clock skew.** Two machines, two wall clocks. Have ProVoice send an `age_ms`
computed on its own clock, or a sequence number — do not difference timestamps
across the cable.

**No decision available ⇒ skip the event and log the skip.** Never default to a
level; a fabricated LoA is indistinguishable from a served one in the results.

---

# 8. Calibration — re-run, to a new file

**DECIDED: re-run calibration for each participant on the study day.**

The pipeline stores per-participant baselines
(`data/calibration_data/calibration_<pid>.json`, loaded by `load_calibration`),
and `--data-collection` skipping calibration *is* that reuse — both of a
participant's collection sessions ran off one stored calibration. Reuse is the
established contract, so the question is fair. But the multi-week gap breaks it:

- **Geometry-bound signals must be re-run.** The `gaze_score` threshold
  (`mean + 2.5·std`), the EAR threshold, and PERCLOS (which runs off the EAR
  threshold) are dominated by camera mount, seat position and lighting, not by
  the person. On a different day the camera is re-mounted and the driver
  re-seats. A stale gaze threshold can fire constantly or never — and
  `gaze_distracted` and `perclos` are model inputs.
- **Physiological baselines drift.** Resting HR moves with sleep, caffeine, time
  of day. `hr_delta = (hr − mean)/std` with a stale mean puts a **constant offset
  on that column for the whole session**.

The decisive argument: during collection the calibration window was recorded
close in time to the drive, so `hr_delta` as the model saw it meant *within-session
arousal*, centred near zero. Reuse across weeks preserves the arithmetic but not
the semantics — a day-level shift gets encoded as sustained arousal the driver is
not experiencing. Re-running restores the condition the training features were
built under.

> **Do not let it overwrite `calibration_<pid>.json`.** That file is the artifact
> the entire training set was normalized against.
> `data_preprocessing/heart_rate_preprocessing.py` is explicit that it never
> modifies it, writing derived baselines elsewhere. A live recalibration at
> session start has no such scruple — if it writes back to the same path, the
> reproducibility of every offline number is silently invalidated, with no error
> and no diff. **Write the study-day baseline to a distinct, session-keyed
> file, and verify where the live path writes before the first participant.**

Two follow-ups:

- **Check the duration.** The preprocessing notes mention that a calibration file
  failing `load_calibration` triggers a **180 s** live recalibration at session
  start — not the 60 s quoted elsewhere. Confirm which path the study takes;
  3 minutes per participant is a real slice of a reused-participant session.
- **Verify it transferred.** After each session, flag any whose **median
  `hr_delta` sits too far from zero**. `heart_rate_preprocessing.py` already
  implements this diagnostic, so it costs nothing.

---

# 9. Logging

A new `data/call_events.csv`, one row per call event, written by the drive
process as a sibling of `append_user_loa_selection` — reuse the existing
`_read_csv_headers` / `_ensure_csv_columns` / `_normalize_csv_value` helpers so
header-migration behaviour matches.

Minimum fields: `session_id`, `participantid`, `window_idx`, `K_condition`,
`block_idx`, `checkpoint_id`, `served_loa`, `loa_frame_timestamp`, `loa_age_ms`,
`event_onset_ms`, `driver_response` (accept/decline/yes/no/cancel/timeout),
`response_latency_ms`, `outcome` (answered/not answered), `skipped_reason`.

The Likert rating continues through the existing label pop-up path into
`user_loa_labels.csv`; the two join on `session_id` + `window_idx`.

---

# 10. Implementation plan

## 10.1 Files

| File | Status | Contents |
|---|---|---|
| `src/drive/call_event.py` | **new** | The whole feature: 5-mode state machine, panel rendering, timeouts, outcome resolution |
| `assets/calls/` | **new** | Pre-rendered `.wav`s — ring, one line per LoA, caller reply |
| `src/drive/study_blocks.py` | **new** | K-condition sequence: the 6 counterbalance orders, participant → sequence map |
| `src/drive/drive_improved.py` | edit | Wiring only (§10.4) |
| `scripts/provoice_status_server.py` | edit | Carry the served LoA in `SessionStatus` (§10.5) |
| `src/ProVoice/main.py` | edit | POST the decision to `/event` once per window |
| `src/ProVoice/decision_engine.py` | edit | Load the per-driver adapted head; serve-time pid assert |
| `start_experiment.py` | edit | Study preset; force `STUDY_TRAFFIC_SEED`; pass K-condition through |
| `docs/remote_setup.md` | edit | Add the LoA row to "what crosses the cable" + the ethics sentence (§7) |
| `CLAUDE.md` | edit | Already points here; update if any DECIDED item changes |
| `src/drive/simcall_simulation.py` | leave | Untouched, or move to `examples/` per `CODE_REVIEW.md` |

**Why a new module rather than more of `drive_improved.py`.** That file is
already ~3.9k lines. The precedent for a self-contained subsystem is
`ambience.py` (898 lines), imported with the try/except relative-then-absolute
pattern. `LoASelectionPopup` lives *inside* `drive_improved.py` because it is
coupled to the main loop's clock-freezing; the call panel deliberately is not, so
it separates cleanly — and being separate is what lets the five renderings be
exercised standalone, without CARLA or a second machine.

**`simcall_simulation.py` is not a head start.** 71 lines, never imported by
anything, random 5–15 s timer, blocking `time.sleep(5)`, no rendering, no LoA
input, no logging. Only the TTS helper is reusable, and §6.4 rules that out too.

## 10.2 New — `src/drive/call_event.py`

Self-contained; imports `pygame` and nothing from `drive_improved`. Constructed
with the screen dims so it can place itself in the HUD band (§6.1).

Surface it needs to expose:

- `CallEvent(dim, assets_dir=None, caller_name=..., onset_offset_s=5.0, cap_s=8.0)`
- `arm(now_ms, loa, window_idx)` — schedule this window's event at the fixed
  offset. `loa is None` ⇒ do not arm; the caller logs the skip (§7).
- `update(now_ms)` — advance the state machine; returns the outcome dict on the
  tick it resolves, else `None`.
- `handle_event(event)` — consume a pygame event while active; returns
  `'affirmative' | 'negative' | None`. Wheel and keyboard both, per §6.3.
- `render(display)` — draw the panel. No-op when inactive.
- `active` — property; the main loop routes input on it.
- `stop()` — release audio channels, mirroring `Ambience.stop()`.

Internal states: `IDLE → RINGING → (AWAIT_INPUT | COUNTDOWN | AUTO) →
CONNECTED → DONE`. Which of the three middle states is entered is the only place
the LoA is branched on; everything else is shared, which is what keeps the five
renderings one component rather than five.

## 10.3 New — `assets/calls/`

Follow the `assets/ambience/` convention (`ambience_assets.py`: drop files in,
no code changes, optional `manifest.json`). Required files:

```
assets/calls/
    ring.wav            played in every condition, identical (§5.2)
    loa1_line.wav       "Call from Mark."
    loa2_line.wav       "Call from Mark. Want me to answer?"
    loa3_line.wav       "Call from Mark. Answering in 3, 2, 1."
    loa4_line.wav       "Call from Mark. Answering now."
    caller_reply.wav    the ~2-3 s canned line played whenever answered
```

Rendered offline, once, and committed or documented — **not** synthesized at run
time (§6.4). Keep `loa1..loa4` close in duration so speech length is not a second
manipulated variable (§5.4).

## 10.4 Edit — `src/drive/drive_improved.py`

Wiring only; no call logic lands here.

1. **Import** in the try/except block alongside `ambience`
   (`from .call_event import CallEvent` / `from call_event import CallEvent`).
2. **Instantiate** next to `ambience = Ambience(...)`, passing `hud.dim`. Mirror
   the `ambience = None` pre-declaration so the `finally` teardown is safe.
3. **Arm per window** — at window start, read the served LoA (§10.5) and call
   `arm()`. On `None`, write a `call_events.csv` row with `skipped_reason` and
   arm nothing.
4. **Tick + render** immediately after `hud.render(display)`, which gives HUD-band
   placement without touching the `HUD` class.
5. **Duck the ambience** while call audio plays, reusing the existing
   `ambience.set_ducked(True/False)` calls.
6. **Input routing** — when `call.active`, wheel buttons route to
   `call.handle_event` before the pop-up sees them. No collision: the call
   resolves before the rating pop-up opens. Button 7 stays quit, honoured
   everywhere, and must not be remapped.
7. **Teardown** — `call.stop()` beside `ambience.stop()`.
8. **Logging** — the `call_events.csv` writer of §9, as a sibling of
   `append_user_loa_selection`, reusing `_read_csv_headers` /
   `_ensure_csv_columns` / `_normalize_csv_value` so header migration behaves
   identically.
9. **Speed placement** — apply §6.1: move the speed left, pin both elements to
   absolute x, fix the `// 3.5` / "Centered horizontally" mismatch, restore the
   backing plate for the panel.

## 10.5 Edit — the LoA transport

The channel already exists and needs extending, not building.
`scripts/provoice_status_server.py` runs on **8081** as a `ThreadingHTTPServer`
with `POST /event`, `GET /status`, `GET /health`; its `SessionStatus` class
publishes a JSON file that the drive UI reads **locally** via `_status_event_ts`.
That is already push-on-the-wire, read-from-a-file at the consumer — exactly the
property §7 requires, so no socket call lands on the render thread.

- **`scripts/provoice_status_server.py`** — add `latest_loa`,
  `latest_loa_frame_ts`, `latest_loa_age_ms`, `latest_loa_window_idx` to
  `SessionStatus._state`, and accept a `decision` event in `record()`. The
  existing body cap and single-writer lock already cover it; the payload is
  smaller than the lifecycle events.
- **`src/ProVoice/main.py`** — POST one `decision` event per window. Send only
  the integer LoA, the frame timestamp it was computed from, and an `age_ms`
  computed on the sender's own clock (§7 — never difference timestamps across the
  cable). Not `probs`, not `message`, not `fcd`.
- **`src/drive/drive_improved.py`** — read it with a sibling of
  `_status_event_ts`, session-id matched exactly as the existing reader is, and
  **check freshness** before arming.

Local (single-machine) runs keep working through
`_load_latest_system_decision_snapshot`, which is already session-guarded; the
status path is the `--remote` branch, mirroring `ProVoiceReadyWatcher`'s
two-sources-one-meaning pattern.

## 10.6 Edit — `src/ProVoice/decision_engine.py`

- Load the per-driver adapted head named in §3.1 instead of the population
  checkpoint.
- **Assert the checkpoint's held-out pid equals `--participantid`** before the
  first inference, and abort loudly on mismatch. `run_lodo_population.py` already
  makes the equivalent assertion at build time; without the serve-time twin, a
  mis-copied file is undetectable after the session.
- Log the resolved checkpoint id into `decisions.csv` and into the POSTed event,
  so every served LoA is traceable to the exact file that produced it.

## 10.7 New — `src/drive/study_blocks.py`

The 6 counterbalance orders of §2 and the participant → sequence assignment, as a
literal table with a `_check()` at import, mirroring how `TRAFFIC_SEED_PLAN` and
`folds.py` are both written and self-validating. Assert: every sequence is a
permutation of the 3 K-conditions; all 6 appear; 12 participants map 2:1 onto
them; and no participant's sequence is derivable from their `TRAFFIC_SEED_PLAN`
pair (§2 — assignment must be independent).

`start_experiment.py` reads it, forces `STUDY_TRAFFIC_SEED`, and refuses to run a
study session without a sequence entry — the same failure mode
`TRAFFIC_SEED_PLAN` already implements for an unplanned participant.

## 10.8 Verify before the first participant

Not code to write, but checks that must pass — each one is silent if wrong:

- **Where the live recalibration writes.** It must not touch
  `calibration_<pid>.json` (§8). Confirm the path before, not after.
- **Which recalibration duration runs** — 60 s or the 180 s fallback (§8).
- **Frame rate with inference on.** Collection ran 14.3–19.5 Hz with the decision
  engine *off*; measure `fps_avg` on the laptop with it on (§7).
- **`[fcd][warn]` absent** from the console — a mistyped function name silently
  substitutes a neutral FCD vector.
- **The pid assert actually fires** — try loading a wrong-pid checkpoint once, on
  purpose, and confirm it aborts.

## 10.9 Build order

1. **Smoke test first**: one LoA — even hardcoded — travelling from the laptop
   through `/event` on 8081 and lighting a panel in the drive UI, with `fps_avg`
   logged on both sides. This is the piece most likely to eat an unbudgeted day.
2. The five renderings and the input matrix (§5.2, §6.2), testable standalone.
3. `study_blocks.py` + checkpoint loading + the pid assert.
4. The call audio layer **last**, so a slip costs the interactive feature rather
   than the study.

---

# 11. Open decisions

- **The two non-zero K values** (§3). Blocked on the offline sweep.
- **Which arm wins** (§3). Blocked on the same. Affects only which checkpoint the
  loader points at.
- **Caller name** (§5.4) — needs to be neutral and fixed.
- **Spawn points** — "a small set, pre-chosen"; not yet chosen.
- **Number of windows per block**, and therefore session length.
- **Likert scale wording and number of points.**
- Whether the driver's own accepted LoA is collected alongside the Likert
  (`CLAUDE.md` says "optionally"). It doubles the pop-up burden per window but is
  the only way to compute served-vs-preferred error live.

---

# 12. Known limitations to state in the write-up

- **Support sets differ by driver.** Each driver's personalization support comes
  from their own scenario pair (`TRAFFIC_SEED_PLAN`), which differs by driver.
  This is noise in individual-driver comparisons, not bias in the K contrast.
  Mitigate by defining each driver's support to span *both* their assigned
  scenarios rather than only the chronologically first.
- **Backbone quality varies between participants.** Twelve LODO backbones means
  each participant's *absolute* satisfaction partly reflects how good their
  particular 11-driver training subset was. K is within-subject and fully
  counterbalanced, so this inflates between-subject variance without biasing the
  K effect — and `results/lodo/lodo_population.csv` already carries each driver's
  floor MAE/QWK, so it enters the analysis as a per-driver covariate rather than
  an unexplained residual.
- **Eyes-off-road time varies by condition** by construction: LoA 2 requires
  reading a question, LoA 4 requires reading nothing. Intrinsic to the
  manipulation, not a placement flaw; HUD-band placement is what keeps it minimal.
- **Likert ratings do not chain to the offline analysis** (§4).
