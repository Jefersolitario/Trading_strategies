"""Causal estimates of the terms in the quote-or-cross rule.

Four quantities are predicted for each decision, from features known at decision time:

p_fill_hat      P(fill)
alpha_hat       E[favourable move]                (fit on every training row)
alpha_fill_hat  E[favourable move | fill]         (fit on filled rows only)
chase_hat       E[chase cost | no fill]           (fit on unfilled rows only)

Estimation is walk-forward by trading day: the model that scores day d has only seen
days before d. Days inside the warm-up carry no prediction and are flagged ``oos = False``
so every policy is compared on the same out-of-sample orders.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance

PREDICTION_COLUMNS = ("p_fill_hat", "alpha_hat", "alpha_fill_hat", "chase_hat")
MIN_ROWS = 200  # per fitted model


@dataclass(frozen=True)
class WalkForwardParams:
    warmup_days: int = 20
    refit_days: int = 5
    window_days: int | None = None  # None = expanding window

    def key(self) -> tuple:
        return (self.warmup_days, self.refit_days, self.window_days)


def _classifier() -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(
        max_depth=3, max_iter=80, learning_rate=0.08, min_samples_leaf=100, l2_regularization=1.0, random_state=0
    )


def _regressor() -> HistGradientBoostingRegressor:
    # Outcomes are mostly noise, so the trees are kept shallow and heavily regularised.
    return HistGradientBoostingRegressor(
        max_depth=2, max_iter=60, learning_rate=0.06, min_samples_leaf=200, l2_regularization=5.0, random_state=0
    )


def _matrix(df: pl.DataFrame, features: list[str]) -> np.ndarray:
    return df.select(pl.col(features).cast(pl.Float64)).to_numpy()


def _fit_predict(model, x_train, y_train, x_test, *, proba: bool = False) -> np.ndarray:
    """Fit and predict, falling back to the training mean when there is too little data."""
    if len(y_train) < MIN_ROWS or (proba and len(np.unique(y_train)) < 2):
        fallback = float(np.mean(y_train)) if len(y_train) else np.nan
        return np.full(x_test.shape[0], fallback)
    model.fit(x_train, y_train)
    return model.predict_proba(x_test)[:, 1] if proba else model.predict(x_test)


def walk_forward(df: pl.DataFrame, features: list[str], params: WalkForwardParams | None = None) -> pl.DataFrame:
    """Return ``df`` (sorted by time) with a ``day`` column, the four predictions and ``oos``."""
    wp = params or WalkForwardParams()
    if not features:
        raise ValueError("At least one feature column is needed.")

    df = df.lazy().sort("ts").with_columns(pl.col("ts").dt.date().alias("day")).collect()
    day_idx = df.select((pl.col("day").rank("dense") - 1).cast(pl.Int64)).to_series().to_numpy()
    n_days = int(day_idx.max()) + 1 if len(day_idx) else 0

    x = _matrix(df, features)
    filled = df["filled"].to_numpy().astype(bool)
    fav = df["fav_move"].to_numpy()
    chase = df["chase_cost"].to_numpy()

    preds = {name: np.full(df.height, np.nan) for name in PREDICTION_COLUMNS}
    for start in range(wp.warmup_days, n_days, wp.refit_days):
        train = day_idx < start
        if wp.window_days is not None:
            train &= day_idx >= start - wp.window_days
        test = (day_idx >= start) & (day_idx < start + wp.refit_days)
        if not test.any() or not train.any():
            continue
        x_test = x[test]
        preds["p_fill_hat"][test] = _fit_predict(_classifier(), x[train], filled[train], x_test, proba=True)
        preds["alpha_hat"][test] = _fit_predict(_regressor(), x[train], fav[train], x_test)
        hit, miss = train & filled, train & ~filled
        preds["alpha_fill_hat"][test] = _fit_predict(_regressor(), x[hit], fav[hit], x_test)
        preds["chase_hat"][test] = _fit_predict(_regressor(), x[miss], chase[miss], x_test)

    out = df.with_columns([pl.Series(name, values) for name, values in preds.items()])
    return out.with_columns(pl.all_horizontal(pl.col(PREDICTION_COLUMNS).is_not_nan()).alias("oos"))


def fill_feature_importance(
    df: pl.DataFrame, features: list[str], *, max_rows: int = 6000, seed: int = 0
) -> pl.DataFrame:
    """Permutation importance of each feature for the fill model.

    Fit on the first 70% of the sample by time and scored on the last 30%, so the
    number is the drop in out-of-sample log-loss skill when the feature is shuffled.
    """
    df = df.sort("ts")
    cut = int(df.height * 0.7)
    train, test = df.head(cut), df.tail(df.height - cut)
    if test.height > max_rows:
        test = test.sample(max_rows, seed=seed)
    y_train = train["filled"].to_numpy().astype(bool)
    y_test = test["filled"].to_numpy().astype(bool)
    if train.height < MIN_ROWS or len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
        return pl.DataFrame({"feature": features, "importance": [0.0] * len(features)})
    model = _classifier().fit(_matrix(train, features), y_train)
    result = permutation_importance(
        model, _matrix(test, features), y_test, scoring="neg_log_loss", n_repeats=4, random_state=seed
    )
    return pl.DataFrame({"feature": features, "importance": result.importances_mean}).sort(
        "importance", descending=True
    )
