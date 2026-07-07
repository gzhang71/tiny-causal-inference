"""Tiny implementations of meta-learners for heterogeneous treatment effect (CATE) estimation.

Each learner estimates tau(x) = E[Y(1) - Y(0) | X = x] from observational data
(X, w, y), where w is a binary treatment indicator. All learners share the same
interface:

    learner.fit(X, w, y)
    tau_hat = learner.predict_cate(X)

References:
    - Kunzel et al. (2019), "Metalearners for estimating heterogeneous treatment
      effects using machine learning" (S-, T-, X-learner)
    - Nie & Wager (2021), "Quasi-oracle estimation of heterogeneous treatment
      effects" (R-learner)
    - Kennedy (2023), "Towards optimal doubly robust estimation of heterogeneous
      causal effects" (DR-learner)
    - Chernozhukov et al. (2018), "Double/debiased machine learning for
      treatment and structural parameters" (Double ML)
"""

import numpy as np
from sklearn.base import clone
from sklearn.model_selection import KFold

EPS = 0.01  # propensity clipping bound


def _clip_propensity(e):
    return np.clip(e, EPS, 1 - EPS)


def _cross_fit_regression(model, X, y, n_splits, random_state):
    """Out-of-fold predictions of E[y | X] via K-fold cross-fitting."""
    preds = np.zeros(len(y), dtype=float)
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    for train_idx, test_idx in kf.split(X):
        m = clone(model).fit(X[train_idx], y[train_idx])
        preds[test_idx] = m.predict(X[test_idx])
    return preds


def _cross_fit_propensity(model, X, w, n_splits, random_state):
    """Out-of-fold predictions of P(w = 1 | X) via K-fold cross-fitting."""
    preds = np.zeros(len(w), dtype=float)
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    for train_idx, test_idx in kf.split(X):
        m = clone(model).fit(X[train_idx], w[train_idx])
        preds[test_idx] = m.predict_proba(X[test_idx])[:, 1]
    return _clip_propensity(preds)


class SLearner:
    """Single model with the treatment indicator as an extra feature.

    CATE is the difference between predictions with w set to 1 vs 0. Simple and
    data-efficient, but the model may shrink the treatment feature toward zero,
    biasing effects toward 0.
    """

    def __init__(self, outcome_model):
        self.outcome_model = outcome_model

    def fit(self, X, w, y):
        Xw = np.column_stack([X, w])
        self.model_ = clone(self.outcome_model).fit(Xw, y)
        return self

    def predict_cate(self, X):
        ones = np.ones(len(X))
        y1 = self.model_.predict(np.column_stack([X, ones]))
        y0 = self.model_.predict(np.column_stack([X, 1 - ones]))
        return y1 - y0


class TLearner:
    """Two separate outcome models, one per treatment arm.

    CATE = mu1(x) - mu0(x). Flexible, but each model only sees its own arm's
    data, which hurts when one arm is small or the arms' errors don't cancel.
    """

    def __init__(self, outcome_model):
        self.outcome_model = outcome_model

    def fit(self, X, w, y):
        self.mu0_ = clone(self.outcome_model).fit(X[w == 0], y[w == 0])
        self.mu1_ = clone(self.outcome_model).fit(X[w == 1], y[w == 1])
        return self

    def predict_cate(self, X):
        return self.mu1_.predict(X) - self.mu0_.predict(X)


class XLearner:
    """T-learner refined with imputed individual effects and a propensity blend.

    Stage 1 fits mu0/mu1 as in the T-learner. Stage 2 imputes each unit's
    treatment effect using the opposite arm's model and regresses those imputed
    effects on X. The two stage-2 estimates are blended with the propensity
    score, so the arm with more data dominates where it should. Shines under
    treatment/control imbalance.
    """

    def __init__(self, outcome_model, effect_model, propensity_model):
        self.outcome_model = outcome_model
        self.effect_model = effect_model
        self.propensity_model = propensity_model

    def fit(self, X, w, y):
        X0, y0 = X[w == 0], y[w == 0]
        X1, y1 = X[w == 1], y[w == 1]

        mu0 = clone(self.outcome_model).fit(X0, y0)
        mu1 = clone(self.outcome_model).fit(X1, y1)

        # Imputed individual treatment effects.
        d1 = y1 - mu0.predict(X1)  # treated: observed y1 minus predicted y0
        d0 = mu1.predict(X0) - y0  # control: predicted y1 minus observed y0

        self.tau1_ = clone(self.effect_model).fit(X1, d1)
        self.tau0_ = clone(self.effect_model).fit(X0, d0)
        self.propensity_ = clone(self.propensity_model).fit(X, w)
        return self

    def predict_cate(self, X):
        # Kunzel et al. weighting: where treated units are scarce (low g), lean
        # on tau1, whose imputation used mu0 fit on the abundant controls; and
        # symmetrically for tau0.
        g = _clip_propensity(self.propensity_.predict_proba(X)[:, 1])
        return g * self.tau0_.predict(X) + (1 - g) * self.tau1_.predict(X)


class RLearner:
    """Residual-on-residual regression (Robinson decomposition).

    Cross-fits m(x) = E[y|x] and e(x) = P(w=1|x), then solves
        min_tau sum_i [ (y_i - m(x_i)) - tau(x_i) (w_i - e(x_i)) ]^2,
    implemented as a weighted regression of (y - m)/(w - e) on X with weights
    (w - e)^2. Orthogonal to errors in m and e, so it tolerates imperfect
    nuisance models.
    """

    def __init__(self, outcome_model, effect_model, propensity_model,
                 n_splits=5, random_state=0):
        self.outcome_model = outcome_model
        self.effect_model = effect_model
        self.propensity_model = propensity_model
        self.n_splits = n_splits
        self.random_state = random_state

    def fit(self, X, w, y):
        m_hat = _cross_fit_regression(
            self.outcome_model, X, y, self.n_splits, self.random_state)
        e_hat = _cross_fit_propensity(
            self.propensity_model, X, w, self.n_splits, self.random_state)

        y_res = y - m_hat
        w_res = w - e_hat
        pseudo = y_res / w_res
        weights = w_res ** 2

        self.effect_model_ = clone(self.effect_model).fit(
            X, pseudo, sample_weight=weights)
        return self

    def predict_cate(self, X):
        return self.effect_model_.predict(X)


class DoubleML:
    """Double/debiased ML for the ATE in a partially linear model.

    Assumes y = theta * w + f(x) + noise with a constant effect theta.
    Cross-fits m(x) = E[y|x] and e(x) = P(w=1|x), then solves the Neyman-
    orthogonal moment equation on the residuals:

        theta = sum_i (w_i - e(x_i)) (y_i - m(x_i)) / sum_i (w_i - e(x_i))^2

    i.e. an OLS of y-residuals on w-residuals (Frisch-Waugh-Lovell with ML
    nuisances). Orthogonality plus cross-fitting makes theta root-n consistent
    and asymptotically normal even with slow-converging ML nuisance models, so
    it comes with an honest standard error. The R-learner is its nonparametric
    generalization; here tau(x) is constant by assumption.

    After fit: `ate_`, `se_`, and `confint(alpha)`. `predict_cate` returns the
    constant theta for interface compatibility.
    """

    def __init__(self, outcome_model, propensity_model, n_splits=5,
                 random_state=0):
        self.outcome_model = outcome_model
        self.propensity_model = propensity_model
        self.n_splits = n_splits
        self.random_state = random_state

    def fit(self, X, w, y):
        m_hat = _cross_fit_regression(
            self.outcome_model, X, y, self.n_splits, self.random_state)
        e_hat = _cross_fit_propensity(
            self.propensity_model, X, w, self.n_splits, self.random_state)

        y_res = y - m_hat
        w_res = w - e_hat

        denom = np.mean(w_res ** 2)
        self.ate_ = np.mean(w_res * y_res) / denom

        # Standard error from the influence function of the moment condition.
        psi = (y_res - self.ate_ * w_res) * w_res / denom
        self.se_ = psi.std(ddof=1) / np.sqrt(len(y))
        return self

    def confint(self, alpha=0.05):
        from statistics import NormalDist
        z = NormalDist().inv_cdf(1 - alpha / 2)
        return self.ate_ - z * self.se_, self.ate_ + z * self.se_

    def predict_cate(self, X):
        return np.full(len(X), self.ate_)


class DRLearner:
    """Doubly robust pseudo-outcome regression.

    Cross-fits mu0, mu1, and e, builds the AIPW pseudo-outcome
        phi = mu1(x) - mu0(x) + w (y - mu1(x)) / e(x)
                              - (1 - w)(y - mu0(x)) / (1 - e(x)),
    whose conditional mean is tau(x) if either the outcome models or the
    propensity model is correct, then regresses phi on X.
    """

    def __init__(self, outcome_model, effect_model, propensity_model,
                 n_splits=5, random_state=0):
        self.outcome_model = outcome_model
        self.effect_model = effect_model
        self.propensity_model = propensity_model
        self.n_splits = n_splits
        self.random_state = random_state

    def fit(self, X, w, y):
        n = len(y)
        mu0_hat = np.zeros(n)
        mu1_hat = np.zeros(n)
        kf = KFold(n_splits=self.n_splits, shuffle=True,
                   random_state=self.random_state)
        for train_idx, test_idx in kf.split(X):
            Xtr, wtr, ytr = X[train_idx], w[train_idx], y[train_idx]
            m0 = clone(self.outcome_model).fit(Xtr[wtr == 0], ytr[wtr == 0])
            m1 = clone(self.outcome_model).fit(Xtr[wtr == 1], ytr[wtr == 1])
            mu0_hat[test_idx] = m0.predict(X[test_idx])
            mu1_hat[test_idx] = m1.predict(X[test_idx])
        e_hat = _cross_fit_propensity(
            self.propensity_model, X, w, self.n_splits, self.random_state)

        pseudo = (mu1_hat - mu0_hat
                  + w * (y - mu1_hat) / e_hat
                  - (1 - w) * (y - mu0_hat) / (1 - e_hat))

        self.effect_model_ = clone(self.effect_model).fit(X, pseudo)
        return self

    def predict_cate(self, X):
        return self.effect_model_.predict(X)
