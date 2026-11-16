"""A stylised generator for the decision-level table.

It is a teaching device, not a market model. Its one job is to reproduce the
mechanism that matters: fills are correlated with the part of the future move you
could not predict, so the orders that fill are worse than average and the orders
that miss are better than average. ``adverse`` sets how strong that link is.

Features are signed in the direction of the trade, like the outcomes.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timedelta

import numpy as np
import polars as pl
from scipy.special import expit, logit, ndtri

MARKOUT_SECONDS = (30, 60, 120, 300, 900)
REGIMES = {
    "none": "No change",
    "informed": "More informed flow late in the sample",
    "undercut": "Quotes undercut late in the sample",
}


@dataclass(frozen=True)
class SimParams:
    n_days: int = 120
    orders_per_day: int = 150
    base_spread: float = 4.0  # typical spread, price units per MWh
    alpha_mean: float = 3.0  # average edge of the entry signal
    vol_mult: float = 1.5  # exit-horizon volatility as a multiple of the spread
    base_fill: float = 0.5  # fill probability of a typical order
    adverse: float = 0.6  # 0 = fills unrelated to the future move, 1 = fully determined by it
    regime: str = "none"  # see REGIMES
    regime_start: float = 0.67  # fraction of the sample after which the regime applies
    deadline_s: int = 300  # how long a quote rests before you give up
    horizon_s: int = 1800  # decision-to-exit horizon
    seed: int = 7

    def as_dict(self) -> dict:
        return asdict(self)


def simulate(params: SimParams | None = None) -> pl.DataFrame:
    p = params or SimParams()
    if p.regime not in REGIMES:
        raise ValueError(f"regime must be one of {sorted(REGIMES)}")
    rng = np.random.default_rng(p.seed)
    n = p.n_days * p.orders_per_day
    day = np.repeat(np.arange(p.n_days), p.orders_per_day)

    # --- what you can see at decision time -------------------------------- #
    ttg_min = rng.uniform(5.0, 240.0, n)  # minutes to gate closure
    far = ttg_min / 240.0
    event = rng.random(n) < 0.12  # forecast update, UMM or imbalance print in the window
    spread = p.base_spread * (0.6 + 0.8 * far) * np.exp(rng.normal(0.0, 0.3, n)) * np.where(event, 1.25, 1.0)
    book_imbalance = np.clip(rng.normal(0.0, 0.45, n), -1.0, 1.0)  # + = book leans your way
    flow_imbalance = rng.normal(0.0, 1.0, n)  # + = recent aggressors trade your way
    signal = rng.gamma(4.0, p.alpha_mean / 4.0, n)  # the strategy's own expected move
    impact = 0.08 * spread * np.exp(rng.normal(0.0, 0.5, n))

    # --- what happens next ------------------------------------------------- #
    sigma = p.vol_mult * spread * np.where(event, 1.4, 1.0)
    mu = signal + 0.35 * spread * book_imbalance + 0.15 * spread * flow_imbalance
    eps = rng.normal(size=n)  # the surprise in the move
    fav_move = mu + sigma * eps

    late = day >= p.regime_start * p.n_days
    rho = p.adverse + 0.2 * event  # informed flow is heavier around events
    fill_logit = logit(np.clip(p.base_fill, 0.02, 0.98)) + 1.1 * (0.5 - far) - 1.3 * book_imbalance
    fill_logit = fill_logit - 0.45 * flow_imbalance + 0.2 * event
    if p.regime == "informed":
        rho = rho + 0.25 * late
    elif p.regime == "undercut":
        fill_logit = fill_logit - 1.0 * late
    rho = np.clip(rho, 0.0, 0.95)
    p_fill = expit(fill_logit)

    # Gaussian copula: a bad surprise (eps < 0) pushes the price into your quote.
    latent = -rho * eps + np.sqrt(1.0 - rho**2) * rng.normal(size=n)
    filled = latent > ndtri(1.0 - p_fill)

    columns: dict[str, np.ndarray] = {}
    for h in MARKOUT_SECONDS:
        f = h / p.horizon_s
        columns[f"mk_{h}"] = mu * f + sigma * np.sqrt(f) * (0.8 * eps + 0.6 * rng.normal(size=n))
    deadline_col = f"mk_{p.deadline_s}"
    if deadline_col in columns:
        wait_move = columns[deadline_col]
    else:
        f = p.deadline_s / p.horizon_s
        wait_move = mu * f + sigma * np.sqrt(f) * (0.8 * eps + 0.6 * rng.normal(size=n))
    chase_cost = wait_move + rng.normal(0.0, 0.1, n) * spread

    start = datetime(2026, 6, 1)
    seconds = rng.uniform(7 * 3600, 22 * 3600, n)
    ts = [start + timedelta(days=int(d), seconds=float(s)) for d, s in zip(day, seconds)]

    return (
        pl.DataFrame(
            {
                "ts": ts,
                "ttg_min": ttg_min,
                "spread": spread,
                "impact": impact,
                "signal": signal,
                "book_imbalance": book_imbalance,
                "flow_imbalance": flow_imbalance,
                "event": event,
                "filled": filled,
                "fav_move": fav_move,
                "chase_cost": chase_cost,
                **columns,
            }
        )
        .lazy()
        .with_columns(pl.col("ts").cast(pl.Datetime("us")))
        .sort("ts")
        .collect()
    )
