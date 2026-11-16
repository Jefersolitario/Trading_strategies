"""Fill-selection diagnostics: how different are the orders that fill from the ones that miss?"""

from __future__ import annotations

import polars as pl

from .schema import markout_columns

_FAV = pl.col("fav_move")
_HIT = pl.col("filled")


def _selection_aggs() -> list[pl.Expr]:
    return [
        pl.len().alias("orders"),
        _HIT.mean().alias("fill_rate"),
        _FAV.mean().alias("alpha_all"),
        _FAV.filter(_HIT).mean().alias("alpha_fill"),
        _FAV.filter(~_HIT).mean().alias("alpha_miss"),
        pl.col("chase_cost").filter(~_HIT).mean().alias("chase_miss"),
        pl.col("spread").mean().alias("spread"),
        pl.col("impact").mean().alias("impact"),
    ]


def selection_summary(df: pl.DataFrame) -> dict[str, float]:
    """Sample-wide fill rate, alpha overall / on fills / on misses, chase cost on misses."""
    row = df.select(_selection_aggs()).row(0, named=True)
    return {k: (float(v) if v is not None else float("nan")) for k, v in row.items()}


def bucketise(df: pl.DataFrame, feature: str, n_buckets: int = 5) -> pl.DataFrame:
    """Add ``bucket`` (label) and ``bucket_order`` (sort key) for one feature.

    Low-cardinality features (booleans, flags) are grouped by value; continuous ones
    are cut into equal-count buckets and labelled with the range they cover.
    """
    if df.schema[feature] == pl.Boolean or df[feature].n_unique() <= n_buckets:
        return df.with_columns(
            pl.col(feature).rank("dense").cast(pl.Int32).alias("bucket_order"),
            pl.col(feature).cast(pl.String).alias("bucket"),
        )
    keyed = df.with_columns(
        ((pl.col(feature).rank("ordinal") - 1) * n_buckets // pl.len()).cast(pl.Int32).alias("bucket_order")
    )
    edges = keyed.group_by("bucket_order").agg(pl.col(feature).min().alias("lo"), pl.col(feature).max().alias("hi"))
    span = float(edges["hi"].max() - edges["lo"].min())
    digits = 0 if span >= 50 else 1 if span >= 5 else 2
    labels = edges.select(
        "bucket_order",
        pl.format("{} to {}", pl.col("lo").round(digits).cast(pl.String), pl.col("hi").round(digits).cast(pl.String))
        .str.replace_all(r"\.0( |$)", "$1")
        .alias("bucket"),
    )
    return keyed.join(labels, on="bucket_order", how="left")


def selection_by_bucket(df: pl.DataFrame, feature: str, n_buckets: int = 5) -> pl.DataFrame:
    """The selection summary within buckets of one feature."""
    return (
        bucketise(df, feature, n_buckets)
        .lazy()
        .group_by("bucket_order", "bucket")
        .agg(_selection_aggs())
        .sort("bucket_order")
        .drop("bucket_order")
        .collect()
    )


def action_mix_by_bucket(df: pl.DataFrame, feature: str, actions: tuple[str, ...], n_buckets: int = 5) -> pl.DataFrame:
    """Share of orders the rule sent down each action, within buckets of one feature."""
    return (
        bucketise(df, feature, n_buckets)
        .lazy()
        .group_by("bucket_order", "bucket")
        .agg(pl.len().alias("orders"), *[(pl.col("action") == a).mean().alias(a) for a in actions])
        .sort("bucket_order")
        .drop("bucket_order")
        .collect()
    )


def markout_curve(df: pl.DataFrame) -> pl.DataFrame:
    """Mean favourable mid move by horizon, for filled, missed and all orders.

    Empty if the table has no ``mk_<seconds>`` columns.
    """
    cols = markout_columns(df)
    if not cols:
        return pl.DataFrame(schema={"seconds": pl.Int64, "filled": pl.Float64, "missed": pl.Float64, "all": pl.Float64})
    rows = [
        {
            "seconds": seconds,
            "filled": df.filter(_HIT)[col].mean(),
            "missed": df.filter(~_HIT)[col].mean(),
            "all": df[col].mean(),
        }
        for seconds, col in cols
    ]
    return pl.DataFrame(rows)


def rolling_selection(df: pl.DataFrame, window_days: int = 10) -> pl.DataFrame:
    """Rolling fill rate and alpha on fills by day.

    Rolling sums of counts and of PnL are divided at the end, so each window is
    order-weighted rather than an average of daily averages.
    """
    daily = (
        df.lazy()
        .with_columns(pl.col("ts").dt.date().alias("day"))
        .group_by("day")
        .agg(
            pl.len().alias("orders"),
            _HIT.sum().alias("fills"),
            _FAV.filter(_HIT).sum().alias("fav_fill"),
            _FAV.sum().alias("fav_all"),
        )
        .sort("day")
        .collect()
    )
    w = max(1, min(window_days, daily.height))
    roll = [pl.col(c).rolling_sum(window_size=w).alias(f"r_{c}") for c in ("orders", "fills", "fav_fill", "fav_all")]
    return (
        daily.with_columns(roll)
        .with_columns(
            (pl.col("r_fills") / pl.col("r_orders")).alias("fill_rate"),
            (pl.col("r_fav_fill") / pl.col("r_fills")).alias("alpha_fill"),
            (pl.col("r_fav_all") / pl.col("r_orders")).alias("alpha_all"),
        )
        .select("day", "fill_rate", "alpha_fill", "alpha_all")
        .drop_nulls()
    )


def calibration(df: pl.DataFrame, pred: str, actual: pl.Expr, n_bins: int = 10) -> pl.DataFrame:
    """Mean prediction versus mean outcome in equal-count bins of the prediction."""
    return (
        df.lazy()
        .with_columns(((pl.col(pred).rank("ordinal") - 1) * n_bins // pl.len()).cast(pl.Int32).alias("bin"))
        .group_by("bin")
        .agg(pl.col(pred).mean().alias("predicted"), actual.mean().alias("realised"), pl.len().alias("orders"))
        .sort("bin")
        .collect()
    )
