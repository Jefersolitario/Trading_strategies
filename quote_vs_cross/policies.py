"""Realised PnL per execution policy, and risk summaries.

PnL is per MWh per order, measured from the decision mid, so a policy's total is what
a 1 MWh clip on every order would have made.

Policies
--------
cross   cross the spread on every order
quote   quote the near touch on every order
rule    choose per order from the causal predictions in ``models``
oracle  choose per order with hindsight (an upper bound, not a strategy)

Sharpe is deliberately absent: the payoffs are asymmetric, so the summary reports
expected shortfall and the worst day instead.
"""

from __future__ import annotations

import math

import polars as pl

MODES = {
    "optional": "Optional entry",
    "must": "Must-complete",
}
POLICIES = ("cross", "quote", "rule", "oracle")
POLICY_LABELS = {
    "cross": "Always cross",
    "quote": "Always quote",
    "rule": "Model rule",
    "oracle": "Oracle (hindsight)",
}
ACTIONS = ("cross", "quote", "skip")

_HALF = pl.col("spread") / 2.0


def _check_mode(mode: str) -> None:
    if mode not in MODES:
        raise ValueError(f"mode must be one of {sorted(MODES)}")


def add_realised_pnl(df: pl.DataFrame, mode: str) -> pl.DataFrame:
    """Add ``pnl_cross`` and ``pnl_quote``: what each static policy made on each order."""
    _check_mode(mode)
    crossed = pl.col("fav_move") - _HALF - pl.col("impact")
    on_fill = pl.col("fav_move") + _HALF
    if mode == "optional":
        on_miss = pl.lit(0.0)  # no fill, no trade
    else:
        on_miss = crossed - pl.col("chase_cost")  # cross at the deadline, after the price moved
    return df.with_columns(
        crossed.alias("pnl_cross"),
        pl.when(pl.col("filled")).then(on_fill).otherwise(on_miss).alias("pnl_quote"),
    )


def add_rule(df: pl.DataFrame, mode: str, *, allow_skip: bool = False) -> pl.DataFrame:
    """Add the model rule's expected values, its ``action`` and ``pnl_rule``, plus ``pnl_oracle``.

    Needs the prediction columns from ``models.walk_forward`` and the columns from
    ``add_realised_pnl``. ``allow_skip`` only applies to optional entries.
    """
    _check_mode(mode)
    skip = allow_skip and mode == "optional"
    p = pl.col("p_fill_hat")
    ev_cross = pl.col("alpha_hat") - _HALF - pl.col("impact")
    if mode == "optional":
        ev_quote = p * (pl.col("alpha_fill_hat") + _HALF)
    else:
        ev_quote = ev_cross + p * (pl.col("spread") + pl.col("impact")) - (1.0 - p) * pl.col("chase_hat")

    df = df.with_columns(ev_cross.alias("ev_cross"), ev_quote.alias("ev_quote"))
    best_ev = pl.max_horizontal("ev_cross", "ev_quote")
    action = pl.when(pl.col("ev_quote") > pl.col("ev_cross")).then(pl.lit("quote")).otherwise(pl.lit("cross"))
    oracle = pl.max_horizontal("pnl_cross", "pnl_quote")
    if skip:
        action = pl.when(best_ev < 0).then(pl.lit("skip")).otherwise(action)
        oracle = pl.max_horizontal(oracle, pl.lit(0.0))
    df = df.with_columns(action.alias("action"), oracle.alias("pnl_oracle"))
    return df.with_columns(
        pl.when(pl.col("action") == "quote")
        .then(pl.col("pnl_quote"))
        .when(pl.col("action") == "cross")
        .then(pl.col("pnl_cross"))
        .otherwise(0.0)
        .alias("pnl_rule")
    )


def daily_pnl(df: pl.DataFrame) -> pl.DataFrame:
    """One row per day with the summed PnL of each policy, plus cumulative columns."""
    cols = [f"pnl_{p}" for p in POLICIES]
    return (
        df.lazy()
        .group_by("day")
        .agg(pl.col(cols).sum(), pl.len().alias("orders"))
        .sort("day")
        .with_columns([pl.col(c).cum_sum().alias(f"cum_{c.removeprefix('pnl_')}") for c in cols])
        .collect()
    )


def expected_shortfall(values: pl.Series, q: float = 0.05) -> float:
    """Mean of the worst ``q`` share of observations (at least one)."""
    n = values.len()
    if n == 0:
        return float("nan")
    k = max(1, math.ceil(q * n))
    return float(values.sort().head(k).mean())


def summarise(df: pl.DataFrame, *, es_q: float = 0.05) -> pl.DataFrame:
    """Per-policy summary: mean per order, total, and daily tail risk."""
    daily = daily_pnl(df)
    rows = []
    for policy in POLICIES:
        col = f"pnl_{policy}"
        per_day = daily[col]
        rows.append(
            {
                "policy": POLICY_LABELS[policy],
                "mean_per_order": float(df[col].mean()),
                "total": float(df[col].sum()),
                "days_positive": float((per_day > 0).mean()),
                "es_daily": expected_shortfall(per_day, es_q),
                "worst_day": float(per_day.min()),
            }
        )
    return pl.DataFrame(rows)


def action_mix(df: pl.DataFrame) -> pl.DataFrame:
    """Share of orders the rule crossed, quoted and skipped."""
    total = df.height
    counts = dict(df.group_by("action").agg(pl.len().alias("n")).iter_rows())
    return pl.DataFrame(
        {
            "action": list(ACTIONS),
            "orders": [int(counts.get(a, 0)) for a in ACTIONS],
            "share": [counts.get(a, 0) / total if total else 0.0 for a in ACTIONS],
        }
    )


def rule_sorting(df: pl.DataFrame) -> pl.DataFrame:
    """Did the rule send each order the right way?

    For the orders the rule crossed, quoted and skipped: what the rule made per order,
    and what always-cross and always-quote would have made on those same orders.
    The rule sorts well if crossing wins on the orders it crossed and quoting wins on
    the orders it quoted.
    """
    order = pl.DataFrame({"action": list(ACTIONS), "rank": list(range(len(ACTIONS)))})
    return (
        df.lazy()
        .group_by("action")
        .agg(
            pl.len().alias("orders"),
            pl.col("pnl_rule").mean().alias("rule"),
            pl.col("pnl_cross").mean().alias("if_crossed"),
            pl.col("pnl_quote").mean().alias("if_quoted"),
        )
        .join(order.lazy(), on="action", how="left")
        .sort("rank")
        .drop("rank")
        .collect()
    )
