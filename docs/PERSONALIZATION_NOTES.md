# Personalization: RL, or a simpler baseline? — design notes

> Working notes on *how* to personalize the LoA decision in the real vehicle.
> Not a final spec — a reasoned recommendation plus the criteria for changing it.

## TL;DR

**Start supervised: per-driver adaptation of the xLSTM. Treat RL as an
escalation, not the starting point.** RL has to *earn its place* with evidence
that the simpler approach is insufficient. Instrument the logging now (implicit
feedback + action propensities) so the door to RL stays open — but don't open it
until the data says you need to.

## Why this is supervised-shaped, not RL-shaped

The driver gives an **explicit label** — the preferred LoA — every 20 s
(`user_loa_labels.csv → user_selected_loa`). RL exists for the case where you
have a *reward* but **not** the right answer. Here we (mostly) have the answer.

- The task is "predict this driver's preferred LoA from context" — a per-step
  prediction **with labels available**, short horizon. That is textbook
  supervised / imitation learning (behavioral cloning), not sequential control.
- RL's distinctive value — long-horizon credit assignment and exploration — is
  largely wasted when the horizon is one decision and the target is observed.
- Driving adds the worst possible conditions for online RL: **safety-critical,
  tiny per-driver data, sim→real gap, liability.**

So the real question is narrower than "RL vs not": **is the stated-preference
label good enough, or do we need to optimize the true interaction objective
(helpful without annoying/endangering) that the label only partially captures?**

## What each approach is actually good for

| Approach | Good when | Cost / risk |
|---|---|---|
| Supervised / imitation (BC) | You have labels, short horizon, want to match stated preference | Low — safe, interpretable, A/B-able |
| Contextual bandit | Explicit label too sparse/biased; richer **implicit** feedback (accept/veto) available | Medium — needs propensity logging + off-policy eval |
| Offline RL (IQL/AWAC/CQL) | Genuine sequential dynamics (trust, fatigue) that per-step prediction misses | High — reward design, OPE, stability |
| Online RL | All the above proven necessary, with full safety infra | Highest — unsafe/expensive in a real car |

## The escalation ladder (use the simplest rung that clears the bar)

0. **Population xLSTM** — one model for everyone (current state). No personalization.
1. **Per-driver adapted xLSTM** ← *recommended baseline.* A per-user head /
   embedding fine-tuned on that driver's accumulated `(features → preferred LoA)`
   labels, anchored to the population model (regularized) so sparse per-driver
   data degrades gracefully to the default. Pure supervised: safe, data-efficient,
   interpretable, directly A/B-testable, **no live exploration.** Likely captures
   most of the achievable personalization value.
2. **Contextual bandit on implicit feedback** — the bridge. Reuse the xLSTM as
   the feature extractor; learn from accept/veto/takeover (richer and more
   frequent than the 20 s label). "RL-lite": gets the feedback benefit without
   full sequential RL. The sweet spot **if** rung 1 plateaus.
3. **Offline RL** — only if sequential effects are demonstrably real and the
   bandit underperforms.
4. **Online RL** — only after all the above, with shield + OPE + cautious,
   uncertainty-gated exploration.

## When does RL actually earn its place? (escalation triggers)

Move past supervised only when the **data shows** at least two of:

- The 20 s self-report is **too sparse / inconsistent** to learn a good per-driver
  model (measure: label rate, intra-driver consistency on repeated contexts).
- **Stated preference diverges from revealed behavior** — drivers *accept* a
  different LoA than they *report* wanting (this is the strongest signal RL would
  help: the label is a biased target, the behavior is the real objective).
- Evidence of **sequential dynamics** (trust building, fatigue) that a per-step
  predictor can't capture.
- The safety infrastructure exists: action shield, off-policy evaluation,
  **propensity logging**, per-driver model store.

Until those fire, supervised wins on every axis that matters (safety, cost,
data-efficiency, debuggability).

## Why the xLSTM is a strong baseline *specifically here*

- It already models the temporal driver-state sequence → LoA — the right
  inductive bias for "how this person's state maps to wanted automation."
- Personalization = a small per-user adapter on their labels: supervised, safe,
  data-efficient with the population prior as a floor.
- No reward engineering, no reward hacking, no off-policy headaches; you can
  validate it offline and A/B it cleanly.

## Why not jump straight to RL

- **Reward engineering + reward hacking** (e.g. a policy that just stays silent
  to avoid vetoes — useless but high "reward").
- **Off-policy evaluation** complexity; requires logging action propensities from
  day one or the logs are unusable for OPE.
- **Data scarcity** per driver, **sim→real** shift, **safety/liability** of
  automating real actions.
- Much harder to interpret, validate, and certify than a supervised model.

## What to measure before deciding (cheap, do during data collection)

- Density and intra-driver consistency of the explicit LoA labels.
- **Divergence between stated LoA and accepted/vetoed LoA** — the make-or-break
  signal for whether RL adds anything.
- Realistic per-driver data volume.
- Outcome metrics either way: agreement with preferred LoA, intervention/veto
  rate, acceptance rate, takeovers, trust/comfort surveys, safety events.

## Recommendation

1. Build **rung 1** (per-driver xLSTM adaptation) and make it the personalization
   baseline. It's the honest first answer to "personalize the LoA."
2. **Now**, while it's cheap, add the logging that keeps RL viable later:
   per-decision **action propensities** + **implicit feedback** (accept / veto /
   takeover / correction) joined into `(state, action, propensity, reward)` tuples
   — an extension of `scripts/build_loa_dataset.py`.
3. Escalate to the **bandit** (rung 2) only if the measured triggers fire, and to
   **full RL** (rungs 3–4) only if the bandit is shown to be insufficient.

The spirit: *use the simplest method that meets the objective; make RL prove it's
needed before paying for it.* The richer RL design (reward shaping, offline→online
loop, safety shield) is documented separately and remains the target **if** the
evidence calls for it — but it is an escalation from this baseline, not the
default.
