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
    - Rosenbaum & Rubin (1983), "The central role of the propensity score in
      observational studies for causal effects" (propensity score matching)
    - Horvitz & Thompson (1952), "A generalization of sampling without
      replacement from a finite universe" (IPW)
    - Robins, Rotnitzky & Zhao (1994), "Estimation of regression coefficients
      when some regressors are not always observed" (AIPW)
"""

import numpy as np
from sklearn.base import clone
from sklearn.model_selection import KFold
from sklearn.neighbors import NearestNeighbors

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


def _cross_fit_aipw_pseudo_outcome(outcome_model, propensity_model, X, w, y,
                                   n_splits, random_state):
    """Out-of-fold AIPW pseudo-outcome phi, with E[phi | X = x] = tau(x)
    if either the outcome models or the propensity model is correct."""
    n = len(y)
    mu0_hat = np.zeros(n)
    mu1_hat = np.zeros(n)
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    for train_idx, test_idx in kf.split(X):
        Xtr, wtr, ytr = X[train_idx], w[train_idx], y[train_idx]
        m0 = clone(outcome_model).fit(Xtr[wtr == 0], ytr[wtr == 0])
        m1 = clone(outcome_model).fit(Xtr[wtr == 1], ytr[wtr == 1])
        mu0_hat[test_idx] = m0.predict(X[test_idx])
        mu1_hat[test_idx] = m1.predict(X[test_idx])
    e_hat = _cross_fit_propensity(propensity_model, X, w, n_splits,
                                  random_state)

    return (mu1_hat - mu0_hat
            + w * (y - mu1_hat) / e_hat
            - (1 - w) * (y - mu0_hat) / (1 - e_hat))


def _normal_confint(center, se, alpha):
    from statistics import NormalDist
    z = NormalDist().inv_cdf(1 - alpha / 2)
    return center - z * se, center + z * se


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
        return _normal_confint(self.ate_, self.se_, alpha)

    def predict_cate(self, X):
        return np.full(len(X), self.ate_)


class IPW:
    """Inverse propensity weighting (Hajek estimator) for the ATE.

    Cross-fits e(x) = P(w=1|x), then reweights each arm by 1/e (treated) and
    1/(1-e) (control) so both look like the full population:

        ATE = sum_i w_i y_i / e_i        / sum_i w_i / e_i
            - sum_i (1-w_i) y_i / (1-e_i) / sum_i (1-w_i) / (1-e_i)

    This is the normalized (Hajek) form, which is less variable than dividing
    by n (Horvitz-Thompson). Needs no outcome model at all; consistent if the
    propensity model is right, but sensitive to extreme weights -- hence the
    propensity clipping. The standard error treats the propensities as known,
    which is (mildly) conservative when they are estimated.

    After fit: `ate_`, `se_`, and `confint(alpha)`. `predict_cate` returns the
    constant ATE for interface compatibility.
    """

    def __init__(self, propensity_model, n_splits=5, random_state=0):
        self.propensity_model = propensity_model
        self.n_splits = n_splits
        self.random_state = random_state

    def fit(self, X, w, y):
        e = _cross_fit_propensity(
            self.propensity_model, X, w, self.n_splits, self.random_state)

        u1 = w / e            # weights that map the treated to the population
        u0 = (1 - w) / (1 - e)
        mu1 = np.sum(u1 * y) / np.sum(u1)
        mu0 = np.sum(u0 * y) / np.sum(u0)
        self.ate_ = mu1 - mu0

        # Influence function of the Hajek difference (propensities as known).
        psi = u1 * (y - mu1) / u1.mean() - u0 * (y - mu0) / u0.mean()
        self.se_ = psi.std(ddof=1) / np.sqrt(len(y))
        return self

    def confint(self, alpha=0.05):
        return _normal_confint(self.ate_, self.se_, alpha)

    def predict_cate(self, X):
        return np.full(len(X), self.ate_)


class AIPW:
    """Augmented IPW (doubly robust) estimator of the ATE.

    Cross-fits mu0, mu1, and e, builds the AIPW pseudo-outcome phi (the same
    one the DR-learner regresses on X), and simply averages it:

        ATE = mean(phi),    se = sd(phi) / sqrt(n)

    Consistent if either the outcome models or the propensity model is
    correct, and the influence-function standard error is honest under
    cross-fitting -- the ATE analogue of the DR-learner.

    After fit: `ate_`, `se_`, and `confint(alpha)`. `predict_cate` returns the
    constant ATE for interface compatibility.
    """

    def __init__(self, outcome_model, propensity_model, n_splits=5,
                 random_state=0):
        self.outcome_model = outcome_model
        self.propensity_model = propensity_model
        self.n_splits = n_splits
        self.random_state = random_state

    def fit(self, X, w, y):
        phi = _cross_fit_aipw_pseudo_outcome(
            self.outcome_model, self.propensity_model, X, w, y,
            self.n_splits, self.random_state)
        self.ate_ = phi.mean()
        self.se_ = phi.std(ddof=1) / np.sqrt(len(y))
        return self

    def confint(self, alpha=0.05):
        return _normal_confint(self.ate_, self.se_, alpha)

    def predict_cate(self, X):
        return np.full(len(X), self.ate_)


class PropensityScoreMatching:
    """1-nearest-neighbor matching on the propensity score, with replacement.

    Fits e(x) = P(w=1|x), then matches each unit to its nearest neighbor in
    the opposite arm on the *logit* of the propensity score (matching on the
    logit is standard practice: it spreads out scores near 0 and 1). Each
    unit's counterfactual outcome is its match's observed outcome:

        ATT = mean over treated of  y_i - y_matched_control(i)
        ATC = mean over control of  y_matched_treated(i) - y_i
        ATE = (n_treated * ATT + n_control * ATC) / n

    The classic, intuitive estimator -- easy to explain and to audit (you can
    inspect the matched pairs) -- but noisier than model-based estimators
    because each counterfactual rests on a single neighbor, and it estimates
    averages only, not tau(x).

    A `caliper` (in standard deviations of the logit score; Austin (2011)
    recommends 0.2) drops pairs whose scores are further apart than that, at
    the cost of estimating the effect on the matchable units only.

    Matching is only trustworthy if it actually balances the covariates, so
    fit also computes standardized mean differences per covariate,

        SMD_j = (mean of X_j, treated - mean of X_j, control) / pooled SD_j,

    before matching (`smd_before_`, treated vs control arms as observed) and
    after (`smd_after_`, each arm pooled with its matches' counterfactuals),
    both scaled by the *pre-matching* pooled SD so they are comparable.
    |SMD| below ~0.1 is conventionally considered balanced.

    After fit: `att_`, `atc_`, `ate_`, `smd_before_`, `smd_after_`,
    `n_matched_treated_`, `n_matched_control_`. `predict_cate` returns the
    constant ATE for interface compatibility.
    """

    def __init__(self, propensity_model, caliper=None):
        self.propensity_model = propensity_model
        self.caliper = caliper

    def fit(self, X, w, y):
        e = _clip_propensity(
            clone(self.propensity_model).fit(X, w).predict_proba(X)[:, 1])
        score = np.log(e / (1 - e)).reshape(-1, 1)

        s0, X0, y0 = score[w == 0], X[w == 0], y[w == 0]
        s1, X1, y1 = score[w == 1], X[w == 1], y[w == 1]

        # For each treated unit, the nearest control (and vice versa).
        match_for_treated = NearestNeighbors(n_neighbors=1).fit(s0)
        match_for_control = NearestNeighbors(n_neighbors=1).fit(s1)
        d0, idx0 = (a[:, 0] for a in match_for_treated.kneighbors(s1))
        d1, idx1 = (a[:, 0] for a in match_for_control.kneighbors(s0))

        if self.caliper is not None:
            max_dist = self.caliper * score.std()
            keep1, keep0 = d0 <= max_dist, d1 <= max_dist
        else:
            keep1 = np.ones(len(s1), dtype=bool)
            keep0 = np.ones(len(s0), dtype=bool)
        self.n_matched_treated_ = int(keep1.sum())
        self.n_matched_control_ = int(keep0.sum())

        self.att_ = np.mean(y1[keep1] - y0[idx0[keep1]])
        self.atc_ = np.mean(y1[idx1[keep0]] - y0[keep0])
        n1m, n0m = self.n_matched_treated_, self.n_matched_control_
        self.ate_ = (n1m * self.att_ + n0m * self.atc_) / (n0m + n1m)

        # Balance diagnostics: the matched "treated" arm is every matched unit
        # plus every match standing in for a treated unit, and symmetrically.
        sd = np.sqrt((X1.var(axis=0) + X0.var(axis=0)) / 2)
        self.smd_before_ = (X1.mean(axis=0) - X0.mean(axis=0)) / sd
        X1_matched = np.vstack([X1[keep1], X1[idx1[keep0]]])
        X0_matched = np.vstack([X0[idx0[keep1]], X0[keep0]])
        self.smd_after_ = (
            X1_matched.mean(axis=0) - X0_matched.mean(axis=0)) / sd
        return self

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
        pseudo = _cross_fit_aipw_pseudo_outcome(
            self.outcome_model, self.propensity_model, X, w, y,
            self.n_splits, self.random_state)
        self.effect_model_ = clone(self.effect_model).fit(X, pseudo)
        return self

    def predict_cate(self, X):
        return self.effect_model_.predict(X)
