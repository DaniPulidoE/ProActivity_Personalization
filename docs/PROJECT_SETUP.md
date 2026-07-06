# PROJECT ARCHITECTURE & SETUP

## CURRENT ARCHITECTURE (population average)

**FUNCTION MODEL:** fcd $\rightarrow$ LoA_fcd (XGBoost)
**STATE MODEL**: fcd + driver_state + carla_state + environment $\rightarrow$ LoA_state (xLSTM)
**FINAL PREDICTION**: w_fcd \* LoA_fcd + (1 - w_fcd) \* LoA_state $\rightarrow$ LoA_pop

## PERSONALIZATION ARCHITECTURE

### FCD GP

**Personalization target**: $\delta$(fcd) = driver_preferred_LoA(ctx) - LoA_pop(ctx)
**SIMPLIFICATION**: ignores state (GP would have too many variables) → personalized state effect captured by modifying w_fcd instead
**INPUT SPACE**: fcd vector (14 evaluation points — one per function)
**GP prior**: $\mu_0(x) = 0$ — assume population model is correct; GP learns residuals only
**GP covariance**: Matérn-5/2 with ARD

* $K(x,x') = \sigma_f^2(1 + \sqrt{5}r + 5r^2/3)e^{-\sqrt{5}r}$
* $r = \sqrt{\sum_i (x_i - x'_i)^2/l_i^2}$
* $l_i$: per-dimension length scale (ARD — one per FCD dimension)
* $\sigma^2_f$: signal variance

**Convergence risk — ARD with few observations**: ARD requires optimizing 12 independent length scales plus $\sigma_f$ via marginal likelihood. With fewer than ~15 observations the marginal likelihood is nearly flat w.r.t. individual length scales — the optimizer finds degenerate solutions (all length scales collapse to the same value, or extreme values that either interpolate exactly or ignore all structure). This can produce falsely low $\sigma_n$ (spurious confidence) before the kernel is actually identifiable.

**Mitigation — staged kernel complexity**:

| Phase | Kernel | Condition |
| --- | --- | --- |
| Cold start | Isotropic Matérn-5/2 (1 length scale) | $N < 15$ |
| Warm | ARD Matérn-5/2 (12 length scales) | $N \geq 15$ |

Always bound length scales: `length_scale_bounds=(0.3, 8.0)` to prevent degenerate extremes.

**Secondary convergence risk — $\sigma_n < \varepsilon$ at wrong $\mu_n$**: the convergence criterion (max $\sigma_n < \varepsilon$) measures epistemic uncertainty, not accuracy. If early observations are predominantly high-noise implicit accepts ($\sigma^2 = 0.8$), $\sigma_n$ can fall below $\varepsilon$ while $\mu_n$ is still offset from the true preference. Mitigate by prioritising low-noise post-session labels ($\sigma^2 = 0.2$) in the active learning budget — they reduce both $\sigma_n$ and variance in $\mu_n$ faster.

**Observation model**: $y_i = f(x_i) + \epsilon_i$, $\epsilon_i \sim \mathcal{N}(0, \sigma^2_i)$

Noise is **heteroscedastic** (different per observation source):

| Source | $\sigma^2$ |
| --- | --- |
| Explicit online / offline label | 0.2–0.3 |
| Post-session questionnaire | 0.2 |
| Implicit accept | 0.8 |
| Implicit override (exact LoA) | 0.4 |

**Formulation note — direct ratings vs. pairwise comparisons**: Most GP preference learning literature (Chu & Ghahramani 2005; Granley & Beyeler, NeurIPS 2023) collects pairwise comparisons ("did you prefer LoA 2 or LoA 3?") and models them with a probit/Thurstone likelihood — non-conjugate to the GP prior, requiring approximate inference (Laplace, EP, or skew-GP). Here the driver selects their exact preferred LoA on a 0–4 scale, giving a direct rating $y = \text{loa\_selected} - \text{loa\_pop} \in \mathbb{R}$. The resulting Gaussian likelihood is conjugate to the GP prior → exact posterior updates, no approximation needed, and sklearn's GaussianProcessRegressor suffices. Direct ratings are also more informative per query than 1-bit pairwise comparisons.

**Posterior update**: standard noisy GP update conditioned on data (see slide 7 in GP slides of Bayesian Deep Learning)

**Acquisition function**: maximum variance ≡ BALD (Bayesian Active Learning by Disagreement, Houlsby et al. 2011) → $k^* = \arg\max_i \, \sigma_n(\text{fcd}_i)$ across all 14 functions

* For GP regression with Gaussian likelihood, BALD (maximise expected information gain about $f$) reduces exactly to max variance: $\arg\max_i \, \mathbb{I}(f(\text{fcd}_i);\, y^*) = \arg\max_i \, \sigma_n(\text{fcd}_i)$
* Information-gain maximisation has sublinear cumulative regret bounds (Srinivas et al. 2010), providing theoretical backing for convergence and directly supporting research question (b)
* Used for **function selection** in all data collection modes (online, offline, implicit)
* UCB ($\mu_n + \beta\sigma_n$) is **not suitable for LoA proposal**: it always adds variance, creating a directional bias toward higher LoA values — it would never explore corrections downward
* EI requires a known optimum and is not meaningful for preference learning
* Thompson Sampling is used for LoA proposal in implicit scenarios (see DATA COLLECTION)

**Inference formula** (applied every frame at 20 Hz):

```python
loa_fcd   = xgboost.predict(fcd_current)
loa_xlstm = xlstm.predict(fcd_current, driver_state, carla_state)
loa_pop   = w_fcd * loa_fcd + (1 - w_fcd) * loa_xlstm

mu, sigma = gp.predict([fcd_current], return_std=True)
confidence = 1 / (1 + sigma[0])          # → 0 with no data, → 1 when certain
loa_final = clamp(loa_pop + confidence * mu[0], 0, safety_ceiling(fcd_current))
```

* `confidence` weighting ensures graceful degradation: with zero observations the GP correction is zeroed out and `loa_final = loa_pop`

**Safety ceiling**: for functions with Safety Risk ≥ 4 (per FCD vector in `fcd_config.py`), `loa_final` is capped at LoA 3 regardless of GP correction, to prevent the personalization layer from autonomously enabling the highest-autonomy actions for safety-critical tasks.

**Convergence criterion**: $\max_i \, \sigma_n(\text{fcd}_i) < \varepsilon$ (e.g. $\varepsilon = 0.25$ LoA units) across all 14 functions.

* Log this value at the end of each session
* This is the primary dependent variable for research question (b): *how much data is needed before personalization converges?*

**Per-driver profile** stored at `data/driver_profiles/{participant_id}.json`:

```json
{
  "gp_X":       "n × 12 array of observed FCD vectors",
  "gp_y":       "n-vector of residuals (driver_preferred − loa_pop)",
  "gp_noise":   "n-vector of per-observation σ²",
  "w_fcd":      "personalized fusion weight (float in [0, 1])",
  "n_sessions": "session count (used to weight w_fcd regularization)"
}
```

### STATE GP

Possibility — probably won't implement it

* **OPTION A**: same as FCD GP but with the whole input space
* **OPTION B**: bottleneck xLSTM, GP over the last (8-dim) hidden layer

### W_FCD PERSONALIZATION

Given a whole batch of user labels, use 1D bounded optimization (e.g. `scipy.minimize_scalar`) to find the best value for w_fcd.
Apply regularization proportional to `|w_new − w_previous| / n_sessions` to avoid large swings early in training when data is sparse.

## DATA COLLECTION

### EXPLICIT ONLINE DATA COLLECTION

Use active learning to select the function to ask about: $k^* = \arg\max_i \, \sigma_n(\text{fcd}_i)$.
Every 20 s the driver selects their preferred LoA for function $k^*$; observe $y = \text{loa\_selected} - \text{loa\_pop}(\text{fcd}_{k^*})$, noise $\sigma^2 = 0.3$.

**POTENTIAL ISSUE**: excessive cognitive workload if the queried function changes every question

### EXPLICIT OFFLINE DATA COLLECTION

Same maximum variance acquisition as online, applied post-session to fill coverage gaps for under-queried functions (2–3 targeted questions via questionnaire). Noise $\sigma^2 = 0.2$.

### IMPLICIT ONLINE DATA COLLECTION

Modify UI so that simulator events trigger a specific in-vehicle function (e.g. incoming call → Function 4) and the system proposes a LoA. The driver either accepts the proposed LoA or overrides it to their exact preferred value.

**Function selection**: maximum variance over the subset of functions currently triggerable in the simulator:

$$k^* = \arg\max_{i \in \text{triggerable}} \sigma_n(\text{fcd}_i)$$

**HOW TO SELECT THE LoA**: Thompson Sampling from the GP posterior

* Sample $\delta_\text{sample} \sim \mathcal{N}(\mu_n(\text{fcd}_{k^*}),\; \sigma_n^2(\text{fcd}_{k^*}))$ (GP posterior — Gaussian, not Beta)
* Propose $\text{LoA}_\text{proposed} = \text{clamp}(\text{LoA}_\text{pop} + \delta_\text{sample},\; 0,\; \text{safety\_ceiling})$
* Symmetric exploration: samples range both above and below the posterior mean → no directional bias
* Natural annealing: as posterior narrows with more data, proposals concentrate around $\mu_n$ (exploitation)

**GP update after implicit scenario**:

* Driver **accepts**: $y = \delta_\text{sample}$, noise $\sigma^2 = 0.8$ (weak — we don't know the exact preference, only that it's close)
* Driver **overrides** to exact LoA $k$: $y = k - \text{LoA}_\text{pop}$, noise $\sigma^2 = 0.4$ (strong — the override reveals the exact preference; this is the primary informative signal)

**Approximation note — accept observations**: the principled likelihood for an accept is probit, $P(\text{accept}) = \Phi((f(\text{fcd}_i) - \text{threshold})/\sigma)$ (non-conjugate, would require Laplace inference and break sklearn compatibility). Treating an accept as a Gaussian observation $y = \delta_\text{sample}$ with high noise $\sigma^2 = 0.8$ is a deliberate simplification that preserves exact inference throughout. Override observations (direct ratings) are unaffected by this approximation and should dominate the posterior in practice.
