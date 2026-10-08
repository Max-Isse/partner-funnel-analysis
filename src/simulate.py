"""Simulate a partner-sourced SME lending funnel.

iwoca's data is private, so this repo uses a synthetic funnel whose true
parameters are known. That lets every method be checked against ground truth.

Funnel: started -> submitted -> decisioned -> (auto | manual route)
        -> approved -> drawdown
Partners differ in volume (a few big, many small) and in a latent
"partner quality" that shifts every stage, as real integrations do.
"""
import numpy as np
import pandas as pd


def expit(x):
    return 1.0 / (1.0 + np.exp(-x))


def logit(p):
    return np.log(p / (1.0 - p))


def simulate_partner_funnel(n_partners: int = 18, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    volume = np.clip(rng.lognormal(6.2, 1.1, n_partners).astype(int), 40, 8000)
    quality = rng.normal(0, 0.45, n_partners)  # latent partner quality

    def stage_p(base, loading, noise=0.12):
        return expit(logit(base) + loading * quality + rng.normal(0, noise, n_partners))

    p_submit = stage_p(0.55, 1.0)  # started -> submitted (the big leak)
    p_decision = stage_p(0.90, 0.4)  # submitted -> decisioned
    p_auto = np.clip(stage_p(0.60, 0.8, 0.2), 0.05, 0.95)  # share auto-decisioned
    p_appr_auto = stage_p(0.70, 0.5)
    p_appr_man = stage_p(0.55, 0.5)
    p_draw_auto = stage_p(0.68, 0.7)  # fast decision -> better drawdown
    p_draw_man = stage_p(0.48, 0.7)

    started = volume
    submitted = rng.binomial(started, p_submit)
    decisioned = rng.binomial(submitted, p_decision)
    auto = rng.binomial(decisioned, p_auto)
    manual = decisioned - auto
    appr_auto = rng.binomial(auto, p_appr_auto)
    appr_man = rng.binomial(manual, p_appr_man)
    draw_auto = rng.binomial(appr_auto, p_draw_auto)
    draw_man = rng.binomial(appr_man, p_draw_man)

    p_true = p_submit * p_decision * (
        p_auto * p_appr_auto * p_draw_auto + (1 - p_auto) * p_appr_man * p_draw_man
    )
    return pd.DataFrame(
        {
            "partner": [f"P{j + 1:02d}" for j in range(n_partners)],
            "started": started,
            "submitted": submitted,
            "decisioned": decisioned,
            "auto": auto,
            "manual": manual,
            "approved_auto": appr_auto,
            "approved_manual": appr_man,
            "drawdown_auto": draw_auto,
            "drawdown_manual": draw_man,
            "approved": appr_auto + appr_man,
            "drawdown": draw_auto + draw_man,
            "p_true": p_true,
        }
    )
