# tiny-causal-inference

Tiny, readable implementations of causal inference methods.

## Meta-learners, Double ML, IPW/AIPW, and propensity score matching

`meta_learners.py` implements five meta-learners for the conditional average
treatment effect tau(x) = E[Y(1) - Y(0) | X = x], plus Double ML, IPW, AIPW,
and propensity score matching for the average treatment effect, all estimated
from observational data using any sklearn-compatible regressor/classifier as
base models:

| Learner | Idea | When it helps |
|---|---|---|
| **S-learner** | One model with treatment as a feature; CATE = f(x, 1) - f(x, 0) | Simple, data-efficient; can bias effects toward 0 |
| **T-learner** | Separate outcome model per arm; CATE = mu1(x) - mu0(x) | Flexible; struggles when one arm is small |
| **X-learner** | T-learner + imputed individual effects, blended by propensity | Imbalanced treatment/control groups |
| **R-learner** | Residual-on-residual regression (Robinson decomposition), cross-fitted | Robust to imperfect nuisance models |
| **DR-learner** | Regression on doubly robust (AIPW) pseudo-outcomes, cross-fitted | Consistent if either outcome or propensity model is right |
| **Double ML** | Neyman-orthogonal residual-on-residual moment for a scalar ATE, cross-fitted | Root-n consistent ATE with an honest confidence interval (assumes constant effect) |
| **IPW** | Reweight each arm by 1/e(x) or 1/(1-e(x)) so both look like the population (Hajek form) | No outcome model needed; consistent if the propensity model is right |
| **AIPW** | Average of the doubly robust pseudo-outcome, cross-fitted | Consistent if either outcome or propensity model is right, with an honest CI |
| **PS matching** | 1-NN matching on the propensity score logit, with replacement; ATT/ATC/ATE from matched pairs | Intuitive and auditable (inspect pairs and covariate balance); noisier than model-based estimators |

### Estimators

Notation: outcome $Y$, binary treatment $W$, covariates $X$; nuisance
functions $\mu_w(x) = E[Y \mid X=x, W=w]$, $m(x) = E[Y \mid X=x]$, and the
propensity score $e(x) = P(W=1 \mid X=x)$. Hats denote fitted models.

**S-learner.** Fit one model $\hat\mu(x, w)$ with the treatment as a feature:

$$\hat\tau(x) = \hat\mu(x, 1) - \hat\mu(x, 0)$$

**T-learner.** Fit $\hat\mu_1$ on treated units and $\hat\mu_0$ on controls:

$$\hat\tau(x) = \hat\mu_1(x) - \hat\mu_0(x)$$

**X-learner.** Stage 1 fits $\hat\mu_0, \hat\mu_1$ as in the T-learner.
Stage 2 imputes each unit's individual effect using the opposite arm's model,

$$\tilde D_i = \begin{cases} Y_i - \hat\mu_0(X_i) & W_i = 1 \\ \hat\mu_1(X_i) - Y_i & W_i = 0 \end{cases}$$

then regresses $\tilde D$ on $X$ within each arm to get $\hat\tau_1(x)$ (from
treated units) and $\hat\tau_0(x)$ (from controls), blended by the propensity
score:

$$\hat\tau(x) = \hat e(x) \hat\tau_0(x) + (1 - \hat e(x)) \hat\tau_1(x)$$

**R-learner.** Cross-fit $\hat m$ and $\hat e$, then solve the Robinson
residual-on-residual problem

$$\hat\tau = \arg\min_\tau \sum_i \Big[ \big(Y_i - \hat m(X_i)\big) - \tau(X_i)\big(W_i - \hat e(X_i)\big) \Big]^2,$$

implemented as a regression of $(Y_i - \hat m(X_i)) / (W_i - \hat e(X_i))$ on
$X_i$ with sample weights $(W_i - \hat e(X_i))^2$.

**DR-learner.** Cross-fit $\hat\mu_0, \hat\mu_1, \hat e$, build the doubly
robust (AIPW) pseudo-outcome

$$\hat\varphi_i = \hat\mu_1(X_i) - \hat\mu_0(X_i) + \frac{W_i\big(Y_i - \hat\mu_1(X_i)\big)}{\hat e(X_i)} - \frac{(1 - W_i)\big(Y_i - \hat\mu_0(X_i)\big)}{1 - \hat e(X_i)},$$

which satisfies $E[\hat\varphi \mid X = x] = \tau(x)$ if either the outcome
models or the propensity model is correct, then regress $\hat\varphi$ on $X$.

**Double ML.** Assume the partially linear model
$Y = \theta W + f(X) + \varepsilon$. Cross-fit $\hat m$ and $\hat e$, then
solve the Neyman-orthogonal moment on the residuals:

$$\hat\theta = \frac{\sum_i \big(W_i - \hat e(X_i)\big)\big(Y_i - \hat m(X_i)\big)}{\sum_i \big(W_i - \hat e(X_i)\big)^2}$$

**IPW.** Cross-fit $\hat e$, then reweight each arm so it resembles the full
population (the normalized/Hajek form):

$$\widehat{ATE} = \frac{\sum_i W_i Y_i / \hat e(X_i)}{\sum_i W_i / \hat e(X_i)} - \frac{\sum_i (1 - W_i) Y_i / (1 - \hat e(X_i))}{\sum_i (1 - W_i) / (1 - \hat e(X_i))}$$

**AIPW.** Cross-fit $\hat\mu_0, \hat\mu_1, \hat e$ and average the doubly
robust pseudo-outcome $\hat\varphi$ from the DR-learner:
$\widehat{ATE} = \frac{1}{n} \sum_i \hat\varphi_i$, with the influence-function
standard error $\widehat{se} = \mathrm{sd}(\hat\varphi) / \sqrt{n}$.

**PS matching.** Match each unit to its nearest opposite-arm neighbor on the
propensity logit, $j(i) = \arg\min_{j: W_j \ne W_i} |\mathrm{logit}(\hat e(X_i)) - \mathrm{logit}(\hat e(X_j))|$, and use the match's outcome as the counterfactual:

$$\widehat{ATT} = \frac{1}{n_1} \sum_{i: W_i=1} \big(Y_i - Y_{j(i)}\big), \qquad \widehat{ATC} = \frac{1}{n_0} \sum_{i: W_i=0} \big(Y_{j(i)} - Y_i\big),$$

$$\widehat{ATE} = \frac{n_1 \widehat{ATT} + n_0 \widehat{ATC}}{n_0 + n_1}$$

An optional caliper (in SDs of the logit score; 0.2 is the conventional
choice) drops pairs whose scores are too far apart. The fit also reports
standardized mean differences per covariate before and after matching
(`smd_before_`, `smd_after_`) — matching is only trustworthy if it actually
balances the covariates, and |SMD| below ~0.1 is the conventional bar.

Where "cross-fit" appears above (R-learner, DR-learner, Double ML), the
nuisance predictions $\hat m(X_i)$, $\hat e(X_i)$, $\hat\mu_w(X_i)$ are
out-of-fold: the data is split into K folds and each unit's nuisance values
are predicted by models trained on the other K-1 folds, which prevents
overfitting bias from leaking into the effect estimate.

All learners share one interface:

```python
learner.fit(X, w, y)          # w: binary treatment, y: outcome
tau_hat = learner.predict_cate(X)
```

Double ML, IPW, and AIPW additionally expose `ate_`, `se_`, and
`confint(alpha)` for inference on the average effect; PS matching exposes
`att_`, `atc_`, `ate_`, and the balance diagnostics `smd_before_` /
`smd_after_`.

### Demo

```
python3 demo_meta_learners.py
```

Generates synthetic observational data with confounded treatment assignment
and a known heterogeneous effect, then compares the learners on PEHE (RMSE of
individual effect estimates) and ATE bias:

```
n = 5000, treated fraction = 0.50
True ATE                       : +0.997
Naive difference in means      : +2.446  (bias +1.449, from confounding)

learner          PEHE   ATE est  ATE bias
-----------------------------------------
S-learner       0.248    +0.939    -0.057
T-learner       0.200    +1.005    +0.008
X-learner       0.164    +0.977    -0.020
R-learner       0.277    +0.986    -0.011
DR-learner      0.255    +0.994    -0.002
Double ML       0.832    +0.973    -0.023
IPW             0.832    +0.996    -0.000
AIPW            0.832    +0.995    -0.001
PS matching     0.832    +1.023    +0.026

Double ML  ATE 95% CI: [+0.927, +1.020]
IPW        ATE 95% CI: [+0.873, +1.119]
AIPW       ATE 95% CI: [+0.953, +1.037]

PS matching balance, max |SMD| over covariates: 1.068 before -> 0.028 after matching
```

Double ML, IPW, AIPW, and PS matching estimate average effects only, so their
high PEHE just reflects the heterogeneity they ignore by design. Note the
CI ordering: IPW (propensity only) is widest, Double ML tightens it with an
outcome model, and AIPW is tightest here while staying consistent if either
nuisance model is wrong. PS matching is the classic auditable estimator but
is noisier, since each counterfactual rests on a single matched neighbor.

### Tests

```
python3 -m pytest
```

The suite checks each estimator against data-generating processes with known
effects: recovering a constant ATE under confounding, recovering heterogeneous
tau(x) (CATE learners), confidence interval coverage, double robustness of
AIPW under a broken outcome model, and covariate balance before/after
matching.

Requires `numpy` and `scikit-learn` (plus `pytest` for the tests).
