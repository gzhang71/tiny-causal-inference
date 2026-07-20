"""Tests: every estimator must recover known effects on synthetic data.

The data-generating processes are linear (with a logistic propensity), so the
linear base models are correctly specified and any systematic miss is the
estimator's fault, not the models'. Run with:

    python3 -m pytest
"""

import numpy as np
import pytest
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression

from meta_learners import (AIPW, IPW, DoubleML, DRLearner,
                           PropensityScoreMatching, RLearner, SLearner,
                           TLearner, XLearner, _clip_propensity, EPS)

N = 4000
TRUE_ATE = 1.0


def make_confounded(n=N, seed=0, heterogeneous=False):
    """Confounded data: x0 drives both treatment and outcome.

    Constant effect: y = tau*w + x0 + 0.5*x1 + noise, tau = TRUE_ATE.
    Heterogeneous:   tau(x) = TRUE_ATE + x0 (mean effect still TRUE_ATE).
    """
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 3))
    e = 1 / (1 + np.exp(-1.2 * X[:, 0]))
    w = rng.binomial(1, e)
    tau = TRUE_ATE + X[:, 0] if heterogeneous else np.full(n, TRUE_ATE)
    y = w * tau + X[:, 0] + 0.5 * X[:, 1] + rng.normal(0, 0.5, size=n)
    return X, w, y, tau


def build_all(seed=0):
    outcome = LinearRegression()
    effect = LinearRegression()
    propensity = LogisticRegression()
    return {
        "S-learner": SLearner(outcome),
        "T-learner": TLearner(outcome),
        "X-learner": XLearner(outcome, effect, propensity),
        "R-learner": RLearner(outcome, effect, propensity, random_state=seed),
        "DR-learner": DRLearner(outcome, effect, propensity,
                                random_state=seed),
        "Double ML": DoubleML(outcome, propensity, random_state=seed),
        "IPW": IPW(propensity, random_state=seed),
        "AIPW": AIPW(outcome, propensity, random_state=seed),
        "PS matching": PropensityScoreMatching(propensity),
    }


def test_naive_estimate_is_confounded():
    X, w, y, tau = make_confounded()
    naive = y[w == 1].mean() - y[w == 0].mean()
    assert abs(naive - TRUE_ATE) > 0.5  # else the tests below prove nothing


@pytest.mark.parametrize("name,learner", build_all().items())
def test_recovers_constant_ate_under_confounding(name, learner):
    X, w, y, _ = make_confounded()
    ate = learner.fit(X, w, y).predict_cate(X).mean()
    tol = 0.15 if name == "PS matching" else 0.1
    assert ate == pytest.approx(TRUE_ATE, abs=tol)


@pytest.mark.parametrize(
    "name", ["T-learner", "X-learner", "R-learner", "DR-learner"])
def test_cate_learners_recover_heterogeneous_effect(name):
    X, w, y, tau = make_confounded(heterogeneous=True)
    learner = build_all()[name]
    tau_hat = learner.fit(X, w, y).predict_cate(X)
    pehe = np.sqrt(np.mean((tau_hat - tau) ** 2))
    assert pehe < 0.15
    assert tau_hat.mean() == pytest.approx(tau.mean(), abs=0.1)


@pytest.mark.parametrize("name,learner", build_all().items())
def test_predict_cate_shape(name, learner):
    X, w, y, _ = make_confounded(n=500)
    tau_hat = learner.fit(X, w, y).predict_cate(X[:7])
    assert tau_hat.shape == (7,)


@pytest.mark.parametrize("cls", [DoubleML, IPW, AIPW])
def test_ate_confidence_intervals(cls):
    X, w, y, _ = make_confounded()
    if cls is IPW:
        est = cls(LogisticRegression())
    elif cls is DoubleML:
        est = cls(LinearRegression(), LogisticRegression())
    else:
        est = cls(LinearRegression(), LogisticRegression())
    est.fit(X, w, y)
    assert est.se_ > 0
    lo, hi = est.confint()
    assert lo < TRUE_ATE < hi
    lo99, hi99 = est.confint(alpha=0.01)
    assert lo99 < lo and hi < hi99  # wider at higher confidence


def test_aipw_is_doubly_robust():
    # Break the outcome models entirely; the correct propensity model must
    # still carry the estimate to the truth.
    X, w, y, _ = make_confounded()
    aipw = AIPW(DummyRegressor(), LogisticRegression()).fit(X, w, y)
    assert aipw.ate_ == pytest.approx(TRUE_ATE, abs=0.1)


def test_matching_improves_balance():
    X, w, y, _ = make_confounded()
    psm = PropensityScoreMatching(LogisticRegression()).fit(X, w, y)
    # x0 is the confounder: badly imbalanced before, balanced after.
    assert abs(psm.smd_before_[0]) > 0.5
    assert abs(psm.smd_after_[0]) < 0.1
    assert np.all(np.abs(psm.smd_after_) < 0.1)
    assert psm.n_matched_treated_ == (w == 1).sum()
    assert psm.n_matched_control_ == (w == 0).sum()


def test_matching_caliper_drops_bad_pairs():
    X, w, y, _ = make_confounded()
    loose = PropensityScoreMatching(LogisticRegression()).fit(X, w, y)
    tight = PropensityScoreMatching(
        LogisticRegression(), caliper=0.05).fit(X, w, y)
    assert tight.n_matched_treated_ <= loose.n_matched_treated_
    assert tight.n_matched_control_ <= loose.n_matched_control_
    assert tight.ate_ == pytest.approx(TRUE_ATE, abs=0.15)
    assert np.all(np.abs(tight.smd_after_) < 0.1)


def test_clip_propensity_bounds():
    e = np.array([0.0, 0.005, 0.5, 0.999, 1.0])
    clipped = _clip_propensity(e)
    assert clipped.min() == EPS
    assert clipped.max() == 1 - EPS
    assert clipped[2] == 0.5
