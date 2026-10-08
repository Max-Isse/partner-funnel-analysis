"""Lightweight checks. Run:  python tests/test_core.py   (or pytest, if installed)."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import bayes, experiment, funnel  # noqa: E402


def test_wilson_known_value():
    lo, hi = funnel.wilson(50, 100)
    assert abs(lo - 0.4038) < 1e-3 and abs(hi - 0.5962) < 1e-3


def test_wilson_coverage_near_nominal():
    rng = np.random.default_rng(0)
    k = rng.binomial(200, 0.3, 20000)
    lo, hi = funnel.wilson(k, 200)
    cover = ((lo <= 0.3) & (0.3 <= hi)).mean()
    assert 0.93 < cover < 0.97


def test_sampler_recovers_known_hyperparameters():
    rng = np.random.default_rng(1)
    J, n = 40, 500
    theta = rng.normal(-1.0, 0.5, J)
    p = 1 / (1 + np.exp(-theta))
    y = rng.binomial(n, p)
    fit = bayes.fit(y, np.full(J, n), n_chains=3, n_iter=4000, burn=1500)
    lo, hi = np.quantile(fit["mu"], [0.01, 0.99])
    assert lo < -1.0 < hi
    assert lo < -1.0 < hi and fit["diag"]["rhat_mu"] < 1.05 and fit["diag"]["rhat_tau"] < 1.05
    # posterior for tau should include the truth too
    tlo, thi = np.quantile(fit["tau"], [0.01, 0.99])
    assert tlo < 0.5 < thi


def test_fixed_horizon_sample_size_matches_textbook():
    n = experiment.fixed_horizon_n(0.50, 0.05)  # textbook value is about 1,565 per arm
    assert 1500 < n < 1620


def test_strict_threshold_controls_false_ships():
    oc = experiment.operating_characteristics(0.48, 0.0, ship_threshold=0.99, reps=600, seed=3)
    assert oc["ship_rate"] < 0.09


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print("ok  ", t.__name__)
    print(f"{len(tests)} tests passed")
