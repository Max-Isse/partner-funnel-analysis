"""Funnel metrics with honest uncertainty (Wilson intervals)."""
import numpy as np
import pandas as pd

STAGES = ["started", "submitted", "decisioned", "approved", "drawdown"]


def wilson(k, n, z: float = 1.96):
    """Wilson score interval for a binomial proportion."""
    k = np.asarray(k, float)
    n = np.asarray(n, float)
    p = k / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return centre - half, centre + half


def stage_table(df: pd.DataFrame) -> pd.DataFrame:
    totals = df[STAGES].sum()
    rows = []
    for prev, cur in zip(STAGES[:-1], STAGES[1:]):
        k, n = totals[cur], totals[prev]
        lo, hi = wilson(k, n)
        rows.append(
            {
                "step": f"{prev} -> {cur}",
                "entered": int(n),
                "passed": int(k),
                "lost": int(n - k),
                "conversion": k / n,
                "ci_low": float(lo),
                "ci_high": float(hi),
                "share_of_all_losses": (n - k) / (totals["started"] - totals["drawdown"]),
            }
        )
    return pd.DataFrame(rows)


def route_table(df: pd.DataFrame) -> pd.DataFrame:
    """Auto vs manual routing: approval and drawdown conversion."""
    rows = []
    for route in ["auto", "manual"]:
        n_dec = df[route].sum()
        n_app = df[f"approved_{route}"].sum()
        n_draw = df[f"drawdown_{route}"].sum()
        lo, hi = wilson(n_draw, n_app)
        rows.append(
            {
                "route": route,
                "decisioned": int(n_dec),
                "approval_rate": n_app / n_dec,
                "drawdown_given_approved": n_draw / n_app,
                "drawdown_ci_low": float(lo),
                "drawdown_ci_high": float(hi),
                "decisioned_to_drawdown": n_draw / n_dec,
            }
        )
    return pd.DataFrame(rows)
