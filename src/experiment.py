"""Sequential Bayesian monitoring of a funnel-fix experiment.

Two arms, binomial outcome (e.g. started -> submitted), Beta(1, 1) priors.
Each week we look at the data and apply a pre-registered rule:
    ship      if P(lift > 0) >= ship_threshold
    stop (no) if P(lift > MDE) < futility_threshold  (only from week 3)
    otherwise keep collecting, up to max_weeks.

Bayesian stopping rules do not automatically control the false-positive rate
under repeated looks, so we measure it by simulation instead of assuming it.
"""
import numpy as np
from scipy import stats


def fixed_horizon_n(p_a, lift, alpha=0.05, power=0.80):
    """Per-arm sample size for a two-sided two-proportion z-test."""
    p_b = p_a + lift
    pbar = (p_a + p_b) / 2
    za, zb = stats.norm.ppf(1 - alpha / 2), stats.norm.ppf(power)
    num = (za * np.sqrt(2 * pbar * (1 - pbar)) + zb * np.sqrt(p_a * (1 - p_a) + p_b * (1 - p_b))) ** 2
    return int(np.ceil(num / lift**2))


def operating_characteristics(
    p_a,
    lift,
    n_week=400,
    max_weeks=10,
    ship_threshold=0.975,
    futility_threshold=0.05,
    mde=0.01,
    reps=1000,
    draws=2000,
    seed=0,
):
    rng = np.random.default_rng(seed)
    ya = rng.binomial(n_week, p_a, (reps, max_weeks)).cumsum(1)
    yb = rng.binomial(n_week, p_a + lift, (reps, max_weeks)).cumsum(1)
    active = np.ones(reps, bool)
    decision = np.zeros(reps, int)  # +1 ship, -1 futility, 0 none by the end
    stop_week = np.full(reps, max_weeks)
    for k in range(max_weeks):
        n = n_week * (k + 1)
        a = rng.beta(1 + ya[:, [k]], 1 + n - ya[:, [k]], (reps, draws))
        b = rng.beta(1 + yb[:, [k]], 1 + n - yb[:, [k]], (reps, draws))
        d = b - a
        p_pos = (d > 0).mean(1)
        p_mde = (d > mde).mean(1)
        ship = active & (p_pos >= ship_threshold)
        fut = active & ~ship & (p_mde < futility_threshold) & (k >= 2)
        decision[ship], decision[fut] = 1, -1
        stop_week[ship | fut] = k + 1
        active &= ~(ship | fut)
    return {
        "ship_rate": float((decision == 1).mean()),
        "futility_rate": float((decision == -1).mean()),
        "undecided_rate": float((decision == 0).mean()),
        "mean_weeks": float(stop_week.mean()),
        "mean_users_per_arm": float(stop_week.mean() * n_week),
    }
