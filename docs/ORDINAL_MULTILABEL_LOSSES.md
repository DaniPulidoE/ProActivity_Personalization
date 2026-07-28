# Loss functions for multi-labelled ordinal LoA data

Analysis of how `--loss {ce, corn, coral}` behaves now that a driver may mark
**several acceptable Levels of Automation** per window, plus a survey of loss
functions designed for this setting.

All numbers below are reproduced from `data/user_loa_labels.csv` (the real
labels) with oracle-capacity models, so they isolate what each *objective and
head* can represent from what the xLSTM backbone happens to learn.

---

## 0. What the label actually is

The pipeline stores `user_selected_loa` as a `;`-separated set (`"1;3"`) and
`SeqDataset` turns it into a multi-hot `lvl ∈ {0,1}^5`. Three different
statistical readings of that set are possible, and they imply *different* losses:

| Reading | Meaning of `S = {marked levels}` | Correct objective |
|---|---|---|
| **(a) Label distribution (LDL)** | driver is indifferent; true LoA ~ Uniform(S) | CE / CORAL / CORN against a soft PMF |
| **(b) Partial label (PLL / superset)** | one LoA is truly right, `S` is an ambiguous candidate set | set-marginal, PRODEN, RC/CC |
| **(c) Set acceptance** | *every* element of `S` is acceptable; predict any one | set-marginal `−log Σ_{k∈S} p_k` |

`levels_to_distribution()` / `levels_to_cumulative()` commit the project to
**(a)**. But `set_accuracy()` and `set_mae()` — the metrics the run is *selected*
on — score reading **(c)**. That mismatch is the root cause of everything in
§2–§3.

### Empirical structure of the labels

```
N = 55 labelled windows
  single-label      36  (65.5 %)
  multi-label       19  (34.5 %)   set sizes: 18×|S|=2, 1×|S|=4
  of the multi-label rows:
    contiguous      13  ({0,1}, {1,2}, {2,3} …)
    GAPPED           6  ({0,2}, {0,4}, {1,3}, {2,4}, {1,3}, {0,2})
```

**Roughly a third of the multi-label rows are non-contiguous**, including a
maximal `{Level_1, Level_5}`. That single fact rules out every unimodality- or
interval-based treatment (§3.6) and is what breaks CORAL (§1.3).

---

## 1. Compatibility of the three implemented losses

### 1.1 `--loss ce` — softmax + `nn.CrossEntropyLoss(soft target)` → **compatible**

`CrossEntropyLoss` accepts probability targets (torch ≥ 1.10) and computes
`−Σ_k t_k log softmax(z)_k = KL(t‖p) + H(t)`, minimised at `p = t`. It is a
proper scoring rule for reading (a), reduces exactly to the one-hot case when
`|S| = 1`, and a softmax head can represent *any* PMF, gapped or not.

Oracle check: **set-accuracy 1.000** on all rows including gapped ones.

Two caveats, neither fatal:

- **Loss values are not comparable across rows.** The irreducible `H(t)` term is
  0 for `|S|=1`, `log 2 ≈ 0.693` for `|S|=2`, `log 4` for `|S|=4`. Reported
  training loss is inflated by a label-dependent constant, so it must not be
  used for early stopping or for comparing runs with different multi-label
  ratios. (Model selection here is on `set_mae`, so this is currently harmless.)
- **CE is nominal.** It has no notion that Level_2 is closer to Level_3 than to
  Level_5 — pre-existing, not a multi-label issue, but §3 fixes it for free.

### 1.2 `--loss corn` — **incompatible, and the guard is correct**

`corn_loss` partitions the batch into *hard* conditional subsets:

```python
label_mask   = y_train > i-1                       # is this sample in task i?
label_tensor = (y_train[label_mask] > i).long()    # its binary target
```

Subset membership is a discrete event, so it needs an integer `y`. Two failure
modes, both verified:

- Passing the multi-hot `(B, K)` tensor raises `IndexError` (the 2-D boolean mask
  hits `logits[mask, task_index]`). Loud — fine.
- Passing a **fractional scalar** (e.g. the mean marked level) is silently
  accepted and hard-thresholded through `1[y > k]`:

  ```
  y = 3    -> corn_loss = 1.075567
  y = 2.5  -> corn_loss = 1.075567     <- identical to y = 3
  y = 2.1  -> corn_loss = 1.075567     <- identical to y = 3
  y = 2    -> corn_loss = 0.836857
  ```

  So a soft label behaves like `ceil(y)`, not like a distribution. The
  `SystemExit` guards in `train_XLSTM.py:439` and `fine_tune_XLSTM.py:235` are
  the right call. (Minor doc nit: the comment says CORN "rounds" the target;
  it applies `1[y > k]`, which for fractional `y` is a ceiling, not rounding.)

**But this is a limitation of the reference implementation, not of CORN.** §3.1
derives a soft-label CORN that is an exact generalisation.

### 1.3 `--loss coral` — **the loss is compatible; the head is not**

`coral_loss` is `Σ_k BCE(σ(f_k), t_k)` with no rounding anywhere, so soft targets
in `[0,1]` are mathematically valid, and `levels_to_cumulative()` produces a
genuine complementary CDF. The construction is invertible, and because `q` is
monotone non-increasing by construction, CORAL's rank-consistency argument still
holds. All of that is correct.

**The problem is the head.** CORAL uses one shared weight vector plus `K−1`
biases (`xlstm_model.py:498-504`), so the achievable cumulative vectors form a
*one-dimensional curve* `(σ(s+b_1), …, σ(s+b_{K−1}))` — the proportional-odds /
parallel-thresholds model. Per-sample soft targets are only fit up to a
projection onto that curve, and the `b_k` are shared across the whole dataset.

For a gapped set the target is unreachable. `S = {Level_1, Level_5}` gives
`q = (0.5, 0.5, 0.5, 0.5)`, which requires `b_1 = b_2 = b_3 = b_4` — impossible
once other rows pull the thresholds apart.

**Oracle experiment** (free latent score per sample, i.e. a perfect backbone;
biases learned jointly on the real 55 rows):

```
learned biases b = [0.918, -3.117, -5.627, -7.554]

set-accuracy   all = 0.891   single = 1.000   multi = 0.684   GAPPED = 0.000
set-MAE        all = 0.127                                    GAPPED = 1.167

gapped rows (marked set -> CORAL oracle argmax):
  {1,3} -> 2     {0,4} -> 2     {0,2} -> 1
  {2,4} -> 3     {0,2} -> 1     {1,3} -> 2
```

Every gapped row decodes to a level the driver **did not mark**, and in every
case it lands *strictly between* the marked levels. The model splits the
difference: told "full manual or full automation", it proposes Level_3. For a
driver-facing LoA recommendation that is the worst available answer, and no
amount of training data or backbone capacity fixes it — the constraint is
structural.

**Isolating the cause.** Keeping the identical loss and identical soft target but
giving each threshold its own logit (a Frank & Hall style binary decomposition,
i.e. dropping the shared `w`):

```
(a) cumulative-BCE + shared-w CORAL head   ->  GAPPED set-acc = 0.000
(b) cumulative-BCE + free per-threshold logits ->  GAPPED set-acc = 1.000
```

So the failure is attributable **entirely to the shared weight vector**, not to
`coral_loss` and not to `levels_to_cumulative()`. The docstring's claim that
CORAL is "the only option that accepts more than one marked LoA" is true at the
loss level but misleading at the head level: it is the option that accepts soft
targets and then cannot represent a third of them.

### 1.4 Summary

| | soft target accepted | reduces to single-label | gapped sets | verdict |
|---|---|---|---|---|
| `ce` | yes | yes | **1.000** | compatible; nominal; `H(t)` offset |
| `corn` | no (`IndexError`, or silent `ceil`) | — | — | correctly rejected; fixable (§3.1) |
| `coral` | yes | yes | **0.000** | loss OK, **head cannot represent them** |

---

## 2. The deeper mismatch: the objective optimises the wrong thing

Even where CE is representationally fine, it targets reading (a) while the
metrics score reading (c). Uniform-over-`S` CE is minimised **only** when the
model outputs the uniform distribution over `S` — so it actively penalises a
confident, *correct* prediction.

Model predicts `p = [0.865, 0.006, 0.006, 0.006, 0.117]` for `S = {L1, L5}`.
The argmax is Level_1, which the driver marked acceptable — `set_accuracy` scores
this a hit. Yet:

```
CE-uniform    dL/dz = [+0.365, +0.006, +0.006, +0.006, -0.383]
                       ^^^^^^ pushes the CORRECT level DOWN

set-marginal  dL/dz = [-0.015, +0.006, +0.006, +0.006, -0.002]   loss = 0.0176
                       keeps mass inside S
```

CE spends capacity forcing the model to hedge between acceptable options.
Under the reading the metrics assume, hedging is not required — and with 55
training windows, capacity is the binding constraint.

**The fix is the target semantics, not the head.** Same CORAL head, same
capacity, only the objective swapped:

```
CORAL head + cumulative-BCE on uniform-CDF target (current)  GAPPED set-acc = 0.000
CORAL head + set-marginal  -log Σ_{k∈S} p_k                  GAPPED set-acc = 1.000
```

---

## 3. Alternatives

### 3.1 Soft-CORN — makes CORN multi-label-capable (recommended)

Take the expectation of CORN's conditional loss under the target distribution:
weight sample *i* in task *k* by its probability of being in that conditional
subset, `w_ik = P(y_i > k−1)`, with target `P(y_i > k | y_i > k−1) = q_k/q_{k−1}`.
The weights telescope, and the result collapses to something very simple:

> **soft-CORN = cross-entropy between the soft label PMF and the PMF the CORN
> chain rule already decodes.**

Closed form (`q_k = P(y>k)` = `levels_to_cumulative`, `p_k` = `levels_to_distribution`):

```
L = − Σ_{k=0}^{K−2} [ q_k · log σ(z_k) + p_k · log(1 − σ(z_k)) ]
```

Verified: identical to the chain-rule cross-entropy for both one-hot and
multi-hot targets, and on one-hot data the gradient has **cosine 0.9999999** with
`corn_loss` (they differ only by a scalar — `corn_loss` normalises by total task
memberships, this by batch size). So it is a strict generalisation: single-label
runs are unchanged up to an effective learning-rate constant.

```python
def soft_corn_loss(logits, lvl):
    """CORN loss for multi-hot LoA labels. (B, K-1) logits, (B, K) multi-hot.

    Equivalent to coral_pytorch.losses.corn_loss (up to a constant factor) when
    exactly one level is marked, but accepts a set of marked levels.
    """
    p = levels_to_distribution(lvl)          # (B, K)     P(y = k)
    q = levels_to_cumulative(lvl)            # (B, K-1)   P(y > k)
    return -(q * F.logsigmoid(logits)
             + p[..., :-1] * F.logsigmoid(-logits)).sum(-1).mean()
```

Oracle: **set-accuracy 1.000 including gapped rows.** CORN's head has `K−1` free
logits, so unlike CORAL it has no parallel-threshold constraint, while
`torch.cumprod` still guarantees rank consistency. This removes the need for the
`SystemExit` guard entirely.

### 3.2 Set-marginal / partial-label losses (recommended)

`L = −log Σ_{k∈S} p_k` — the negative log-probability the model assigns to the
acceptable set. Head-agnostic (works off `logits_to_probs`, so it drops onto
softmax, CORN *and* CORAL), reduces exactly to CE when `|S| = 1`, and is the
direct differentiable surrogate for `set_accuracy`. Oracle: **1.000 everywhere**,
including on the unmodified CORAL head.

Known weakness: it lets the model pick whichever candidate is easiest, which can
drift toward the majority class. The literature's remedies:

- **PRODEN** ([Lv et al., ICML 2020](https://proceedings.mlr.press/v119/lv20a.html),
  [code](https://github.com/lvjiaqi77/PRODEN)) — progressive identification:
  re-weight the target within `S` by the model's own normalised probabilities each
  step. Classifier-consistent, model- and loss-independent, ~5 lines over the above.
- **RC / CC losses** ([Feng et al., NeurIPS 2020](https://proceedings.neurips.cc/paper/2020/file/7bd28f15a49d5e5848d6ec70e584e625-Paper.pdf))
  — provably risk-consistent and classifier-consistent estimators; RC coincides
  algorithmically with PRODEN.
- **LWS** ([Wen et al. 2021](https://arxiv.org/pdf/2106.05731)) — leveraged weighted
  loss, adds an explicit penalty on non-candidate labels.
- [Naive PLL](https://arxiv.org/pdf/2010.11600) shows the plain set-marginal loss is
  a surprisingly strong baseline; [consistency regularisation](https://proceedings.mlr.press/v162/wu22l/wu22l.pdf)
  is the current SOTA direction.
- [Label-set loss functions for partial supervision](https://arxiv.org/pdf/2107.03846)
  gives a general axiomatic recipe for converting *any* per-label loss into a
  label-set loss (from segmentation, but the construction is generic).

Given 55 windows, start with plain set-marginal; add PRODEN only if you observe
collapse toward one level.

### 3.3 SORD — soft ordinal labels (cheap ordinal upgrade, keeps the softmax head)

[Diaz & Marathe, CVPR 2019](https://openaccess.thecvf.com/content_CVPR_2019/papers/Diaz_Soft_Labels_for_Ordinal_Regression_CVPR_2019_paper.pdf).
Replace the one-hot with `t_k ∝ exp(−α·d(k, y))` and train with ordinary CE.
Multi-label extension is natural — use **distance to the nearest marked level**,
`d(k, S) = min_{j∈S}|k−j|`. This is a drop-in replacement for
`levels_to_distribution()`, keeps the existing softmax head, and unlike CORAL it
handles gapped sets (the softmax head has no unimodality constraint, so the
target is simply bimodal). Reported to outperform other ordinal losses alongside
OLL. Lowest-effort way to get ordinality into the current `ce` path.

### 3.4 OLL — ordinal log-loss

[Castagnos et al., COLING 2022](https://aclanthology.org/2022.coling-1.407/) ·
[code](https://github.com/glanceable-io/ordinal-log-loss).
`L = −Σ_{k≠y} |k−y|^α log(1−p_k)` — penalises misclassification in proportion to
ordinal distance. Extends to sets with `d(k,S)` as above (terms for `k ∈ S`
vanish since `d = 0`), which makes it *automatically* set-aware: it never
penalises predicting one acceptable level over another. Benchmarked as
state-of-the-art alongside SORD.

### 3.5 Distribution-distance losses

- **Squared EMD / Wasserstein** ([Hou, Yu & Samaras](https://arxiv.org/abs/1611.05916)) —
  `‖CDF(p) − CDF(t)‖²`, natural for soft ordinal targets, directly penalises
  ordinal distance. Works with any multi-hot-normalised target. Note it *does*
  inherit the reading-(a) hedging problem of §2.
- **Weighted-kappa loss** ([de La Torre et al., PRL 2018](https://www.sciencedirect.com/science/article/abs/pii/S0167865517301666)) —
  differentiable QWK, aligns training with the QWK you already report.
- **SLACE** ([AAAI](https://ojs.aaai.org/index.php/AAAI/article/download/34158/36313)) —
  monotone, balance-sensitive ordinal loss, relevant given the class skew here
  (20/55 rows are Level_1).
- [Ordinal classification with label-dependent loss](https://link.springer.com/article/10.1007/s10994-026-07023-z)
  for asymmetric penalties — worth considering, since over-automating is not
  symmetric with under-automating in a driving context.

### 3.6 Explicitly **not** recommended here

- **Unimodal output losses** — [Beckham & Pal](https://arxiv.org/pdf/1705.05278)
  (binomial/Poisson), [optimal-transport + unimodal](https://arxiv.org/pdf/2011.07607),
  unimodal regularisation. They *enforce* a single-peaked PMF, which is exactly
  what makes CORAL fail on `{L1, L5}`. With 32 % of multi-label rows gapped, these
  would reproduce the §1.3 failure by construction.
- **Interval-censored ordinal regression** (`−log(q_{a−1} − q_b)` for `y ∈ [a,b]`)
  would be the textbook answer *if* marked sets were contiguous. They are not
  (6/19), so it is only usable with a fallback branch for gapped rows.
- **Plain binary relevance BCE** (true multi-label, `K` independent sigmoids) —
  discards ordinality and still needs an arbitrary rule to pick the single LoA to
  actually serve at inference.

---

## 4. Recommendation

1. **Fix the CORAL claim.** Either drop the shared-`w` constraint on multi-label
   data (§1.3(b)) or, better, switch its target semantics to set-marginal (§2) —
   the latter needs no architecture change and takes gapped set-accuracy from
   0.000 to 1.000. As it stands, `--loss coral` is the *worst* of the three
   options for multi-label rows despite being documented as the only one that
   supports them.
2. **Add `soft_corn_loss` (§3.1)** and remove the two `SystemExit` guards. It is
   a ~5-line function, provably reduces to `corn_loss` on single-label data, and
   gives a rank-consistent ordinal head with no representational blind spot.
3. **Align objective with metric.** Decide explicitly between reading (a) and
   reading (c). If `set_accuracy`/`set_mae` are the metrics you report, train
   with a set-marginal objective (§3.2); if you genuinely mean "the driver is
   indifferent", keep CE/soft-CORN but report a distributional metric (NLL, EMD)
   alongside.
4. **Cheapest single improvement to the default path:** swap
   `levels_to_distribution()` for a SORD-style nearest-marked-level soft target
   (§3.3). One function, no head change, adds ordinality to `--loss ce`.
5. **Caveat on all of this:** N = 55, with 6 gapped rows. The oracle experiments
   above are representational statements — they show what each objective *can*
   express — and are valid at any N. The relative *generalisation* of these
   losses cannot be settled on this dataset; treat §3 as a shortlist to evaluate,
   not a ranking.
