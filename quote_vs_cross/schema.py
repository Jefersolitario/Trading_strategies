"""The decision-level table the dashboard works from.

One row per entry decision. All price columns are per MWh and signed in the
direction of the trade (positive = good for you), so buys and sells share one table.
"""

from __future__ import annotations

import re

import polars as pl

# name -> (description, kind)
REQUIRED: dict[str, tuple[str, str]] = {
    "ts": ("Decision time", "datetime"),
    "spread": ("Best ask minus best bid at decision time", "numeric"),
    "impact": ("Extra cost of crossing your size beyond the touch (>= 0), read off the visible book", "numeric"),
    "filled": ("Would a quote at the near touch have filled before the deadline (queue-aware)", "boolean"),
    "fav_move": ("Favourable mid move from decision to exit: side x (exit mid - decision mid)", "numeric"),
    "chase_cost": (
        "Adverse move of the far touch from decision to deadline: side x (far touch at deadline - far touch at decision)",
        "numeric",
    ),
}

OPTIONAL: dict[str, str] = {
    "mk_<seconds>": "Favourable mid move <seconds> after the decision, e.g. mk_30, mk_300 (markout curve)",
    "any other numeric column": "Used as a model feature, so it must be known at decision time",
}

OUTCOME_COLUMNS = ("filled", "fav_move", "chase_cost")
_MARKOUT = re.compile(r"^mk_(\d+)$")


def markout_columns(df: pl.DataFrame) -> list[tuple[int, str]]:
    """(seconds, column) pairs for the optional markout columns, sorted by horizon."""
    found = []
    for name in df.columns:
        m = _MARKOUT.match(name)
        if m and df.schema[name].is_numeric():
            found.append((int(m.group(1)), name))
    return sorted(found)


def feature_columns(df: pl.DataFrame) -> list[str]:
    """Columns the models may use: every numeric or boolean column known at decision time.

    That is everything except the outcomes and the markouts. ``spread`` and ``impact``
    are features too, since both are visible when you decide.
    """
    excluded = set(OUTCOME_COLUMNS) | {"ts"} | {c for _, c in markout_columns(df)}
    out = []
    for name, dtype in df.schema.items():
        if name in excluded:
            continue
        if dtype.is_numeric() or dtype == pl.Boolean:
            out.append(name)
    return out


def validate(df: pl.DataFrame) -> list[str]:
    """Human-readable problems with ``df``. Empty list means it is usable."""
    problems: list[str] = []
    for name, (_, kind) in REQUIRED.items():
        if name not in df.columns:
            problems.append(f"Missing column `{name}`.")
            continue
        dtype = df.schema[name]
        if kind == "datetime" and not dtype.is_temporal():
            problems.append(f"`{name}` must be a datetime, found {dtype}.")
        elif kind == "numeric" and not dtype.is_numeric():
            problems.append(f"`{name}` must be numeric, found {dtype}.")
        elif kind == "boolean" and not (dtype == pl.Boolean or dtype.is_integer()):
            problems.append(f"`{name}` must be boolean or 0/1, found {dtype}.")
    if problems:
        return problems

    if df.height == 0:
        return ["The table is empty."]
    for name in REQUIRED:
        nulls = df[name].null_count()
        if nulls:
            problems.append(f"`{name}` has {nulls:,} null values.")
    if df.filter(pl.col("spread") <= 0).height:
        problems.append("`spread` must be positive on every row.")
    if df.filter(pl.col("impact") < 0).height:
        problems.append("`impact` must be zero or positive on every row.")
    return problems


def normalise(df: pl.DataFrame) -> pl.DataFrame:
    """Cast to the types the rest of the package assumes and sort by time."""
    return (
        df.lazy()
        .with_columns(
            pl.col("ts").cast(pl.Datetime("us")),
            pl.col("filled").cast(pl.Boolean),
            pl.col("spread", "impact", "fav_move", "chase_cost").cast(pl.Float64),
        )
        .sort("ts")
        .collect()
    )
