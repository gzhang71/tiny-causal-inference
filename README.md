# tiny_causal_inference

Tiny, readable implementations of causal inference methods.

## Meta-learners and Double ML

`meta_learners.py` implements five meta-learners for the conditional average
treatment effect tau(x) = E[Y(1) - Y(0) | X = x], plus Double ML for the
average treatment effect, all estimated from observational data using any
sklearn-compatible regressor/classifier as base models:

| Learner | Idea | When it helps |
|---|---|---|
| **S-learner** | One model with treatment as a feature; CATE = f(x, 1) - f(x, 0) | Simple, data-efficient; can bias effects toward 0 |
| **T-learner** | Separate outcome model per arm; CATE = mu1(x) - mu0(x) | Flexible; struggles when one arm is small |
| **X-learner** | T-learner + imputed individual effects, blended by propensity | Imbalanced treatment/control groups |
| **R-learner** | Residual-on-residual regression (Robinson decomposition), cross-fitted | Robust to imperfect nuisance models |
| **DR-learner** | Regression on doubly robust (AIPW) pseudo-outcomes, cross-fitted | Consistent if either outcome or propensity model is right |
| **Double ML** | Neyman-orthogonal residual-on-residual moment for a scalar ATE, cross-fitted | Root-n consistent ATE with an honest confidence interval (assumes constant effect) |

All learners share one interface:

```python
learner.fit(X, w, y)          # w: binary treatment, y: outcome
tau_hat = learner.predict_cate(X)
```

Double ML additionally exposes `ate_`, `se_`, and `confint(alpha)` for
inference on the average effect.

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

Double ML ATE 95% CI: [+0.927, +1.020]
```

Double ML's high PEHE is expected: it assumes a constant effect, so the number
reflects the heterogeneity it ignores by design — its target is the (variance-
weighted) average effect, which it pins down with an honest confidence
interval.

Requires `numpy` and `scikit-learn`.
