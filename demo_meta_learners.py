"""Demonstration: meta-learners for CATE estimation on synthetic data.

Generates observational data with confounded treatment assignment and a known
heterogeneous treatment effect, fits S/T/X/R/DR-learners, and compares them on
PEHE (root mean squared error of the individual effect estimates) and ATE bias.
Because assignment is confounded, the naive treated-vs-control difference in
means is badly biased -- the learners have to adjust for X to do better.

Usage:
    python3 demo_meta_learners.py [--n 5000] [--seed 0]
"""

import argparse

import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression

from meta_learners import (SLearner, TLearner, XLearner, RLearner, DRLearner,
                           DoubleML, IPW, AIPW, PropensityScoreMatching)


def make_data(n, seed):
    """Confounded observational data with heterogeneous treatment effect.

    x0 drives both the propensity and the baseline outcome (confounding);
    the true effect tau(x) varies with x0 and x1.
    """
    rng = np.random.default_rng(seed)
    X = rng.uniform(-1, 1, size=(n, 5))

    # Propensity: healthier-looking units (high x0) get treated more often.
    e = 1 / (1 + np.exp(-2.0 * X[:, 0] - 0.5 * X[:, 1]))
    w = rng.binomial(1, e)

    # Baseline outcome depends on the same confounder x0.
    mu0 = 2.0 * X[:, 0] + X[:, 1] ** 2 + 0.5 * X[:, 2]

    # True CATE: nonlinear, heterogeneous, sometimes negative.
    tau = 1.0 + X[:, 0] * X[:, 1] + np.sin(2.0 * X[:, 0])

    y = mu0 + w * tau + rng.normal(0, 0.5, size=n)
    return X, w, y, tau


def build_learners(seed):
    outcome = GradientBoostingRegressor(random_state=seed)
    effect = GradientBoostingRegressor(random_state=seed)
    propensity = LogisticRegression()
    return {
        "S-learner": SLearner(outcome),
        "T-learner": TLearner(outcome),
        "X-learner": XLearner(outcome, effect, propensity),
        "R-learner": RLearner(outcome, effect, propensity, random_state=seed),
        "DR-learner": DRLearner(outcome, effect, propensity, random_state=seed),
        "Double ML": DoubleML(outcome, propensity, random_state=seed),
        "IPW": IPW(propensity, random_state=seed),
        "AIPW": AIPW(outcome, propensity, random_state=seed),
        "PS matching": PropensityScoreMatching(propensity, caliper=0.2),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=5000, help="sample size")
    parser.add_argument("--seed", type=int, default=0, help="random seed")
    args = parser.parse_args()

    X, w, y, tau = make_data(args.n, args.seed)
    X_test, _, _, tau_test = make_data(args.n, args.seed + 1)

    true_ate = tau_test.mean()
    naive_ate = y[w == 1].mean() - y[w == 0].mean()

    print(f"n = {args.n}, treated fraction = {w.mean():.2f}")
    print(f"True ATE                       : {true_ate:+.3f}")
    print(f"Naive difference in means      : {naive_ate:+.3f}  "
          f"(bias {naive_ate - true_ate:+.3f}, from confounding)")
    print()
    print(f"{'learner':<12} {'PEHE':>8} {'ATE est':>9} {'ATE bias':>9}")
    print("-" * 41)

    fitted = {}
    for name, learner in build_learners(args.seed).items():
        learner.fit(X, w, y)
        fitted[name] = learner
        tau_hat = learner.predict_cate(X_test)
        pehe = np.sqrt(np.mean((tau_hat - tau_test) ** 2))
        ate_hat = tau_hat.mean()
        print(f"{name:<12} {pehe:>8.3f} {ate_hat:>+9.3f} "
              f"{ate_hat - true_ate:>+9.3f}")

    print()
    for name in ("Double ML", "IPW", "AIPW"):
        lo, hi = fitted[name].confint()
        print(f"{name:<10} ATE 95% CI: [{lo:+.3f}, {hi:+.3f}]")

    psm = fitted["PS matching"]
    print()
    print("PS matching balance, max |SMD| over covariates: "
          f"{np.abs(psm.smd_before_).max():.3f} before -> "
          f"{np.abs(psm.smd_after_).max():.3f} after matching")
    print("Note: Double ML, IPW, AIPW, and PS matching estimate average")
    print("effects only, so their PEHE reflects the heterogeneity they")
    print("ignore by design.")


if __name__ == "__main__":
    main()
