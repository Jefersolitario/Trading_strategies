"""The data path: simulator, schema, causal models, policies and diagnostics."""

from datetime import datetime

import polars as pl
import pytest

from quote_vs_cross import diagnostics as diag
from quote_vs_cross import policies as pol
from quote_vs_cross import schema
from quote_vs_cross.models import PREDICTION_COLUMNS, WalkForwardParams, walk_forward
from quote_vs_cross.simulate import SimParams, simulate

SMALL = SimParams(n_days=40, orders_per_day=80, seed=3)


@pytest.fixture(scope="module")
def table() -> pl.DataFrame:
    return schema.normalise(simulate(SMALL))


@pytest.fixture(scope="module")
def scored(table) -> pl.DataFrame:
    return walk_forward(table, schema.feature_columns(table), WalkForwardParams(warmup_days=15, refit_days=5))


def test_simulated_table_matches_the_schema(table):
    assert schema.validate(table) == []
    assert table.height == SMALL.n_days * SMALL.orders_per_day
    features = schema.feature_columns(table)
    assert not set(features) & {"filled", "fav_move", "chase_cost", "ts"}
    assert not [f for f in features if f.startswith("mk_")]


def test_simulation_is_reproducible():
    assert simulate(SMALL).equals(simulate(SMALL))


def test_validation_reports_problems():
    assert schema.validate(pl.DataFrame({"ts": [datetime(2026, 1, 1)]}))  # missing columns
    bad = pl.DataFrame(
        {
            "ts": [datetime(2026, 1, 1)],
            "spread": [-1.0],
            "impact": [0.1],
            "filled": [True],
            "fav_move": [0.5],
            "chase_cost": [0.2],
        }
    )
    assert any("spread" in p for p in schema.validate(bad))


def test_adverse_selection_shows_up_in_fills(table):
    s = diag.selection_summary(table)
    assert s["alpha_fill"] < s["alpha_all"] < s["alpha_miss"]
    rebuilt = s["fill_rate"] * s["alpha_fill"] + (1 - s["fill_rate"]) * s["alpha_miss"]
    assert rebuilt == pytest.approx(s["alpha_all"])


def test_no_adverse_selection_means_no_gap():
    flat = simulate(SimParams(n_days=60, orders_per_day=200, adverse=0.0, seed=5))
    # Events still add a little selection in the simulator, so compare outside them.
    s = diag.selection_summary(flat.filter(~pl.col("event")))
    strong = diag.selection_summary(simulate(SimParams(n_days=60, orders_per_day=200, adverse=0.8, seed=5)))
    assert abs(s["alpha_fill"] - s["alpha_miss"]) < 0.25 * abs(strong["alpha_fill"] - strong["alpha_miss"])


def test_walk_forward_is_causal(table, scored):
    """Changing the outcomes of the last days must not change any earlier prediction."""
    assert scored.filter(~pl.col("oos"))["day"].n_unique() == 15  # warm-up carries no prediction
    days = scored["day"].unique().sort()
    cutoff = days[30]
    tampered = table.with_columns(
        pl.when(pl.col("ts").dt.date() >= cutoff).then(pl.col("fav_move") + 100.0).otherwise(pl.col("fav_move")).alias("fav_move"),
        pl.when(pl.col("ts").dt.date() >= cutoff).then(~pl.col("filled")).otherwise(pl.col("filled")).alias("filled"),
    )
    rescored = walk_forward(tampered, schema.feature_columns(tampered), WalkForwardParams(warmup_days=15, refit_days=5))
    # The refit that first sees the tampered days predicts from day 35 (refits at 15, 20, 25, 30, 35).
    before = pl.col("day") < days[35]
    for col in PREDICTION_COLUMNS:
        a = scored.filter(before & pl.col("oos"))[col]
        b = rescored.filter(before & pl.col("oos"))[col]
        assert a.equals(b), col


def test_realised_pnl(table):
    opt = pol.add_realised_pnl(table, "optional")
    must = pol.add_realised_pnl(table, "must")
    half = pl.col("spread") / 2
    crossed = pl.col("fav_move") - half - pl.col("impact")
    assert opt.select((pl.col("pnl_cross") - crossed).abs().max()).item() < 1e-12
    # Optional: a miss is no trade. Must-complete: a miss is a late cross.
    assert opt.filter(~pl.col("filled"))["pnl_quote"].abs().max() == 0.0
    late = must.filter(~pl.col("filled")).select((pl.col("pnl_quote") - (crossed - pl.col("chase_cost"))).abs().max())
    assert late.item() < 1e-12
    fill = opt.filter(pl.col("filled")).select((pl.col("pnl_quote") - (pl.col("fav_move") + half)).abs().max())
    assert fill.item() < 1e-12


@pytest.mark.parametrize("mode", ["optional", "must"])
def test_rule_is_bounded_by_the_oracle(scored, mode):
    oos = scored.filter(pl.col("oos"))
    bt = pol.add_rule(pol.add_realised_pnl(oos, mode), mode)
    assert set(bt["action"].unique()) <= {"cross", "quote"}
    assert (bt["pnl_oracle"] >= bt["pnl_rule"] - 1e-12).all()
    assert (bt["pnl_oracle"] >= bt["pnl_cross"] - 1e-12).all()
    stats = pol.summarise(bt)
    assert stats["policy"].to_list() == [pol.POLICY_LABELS[p] for p in pol.POLICIES]
    assert pol.action_mix(bt)["share"].sum() == pytest.approx(1.0)


def test_skip_only_applies_to_optional_entries(scored):
    oos = scored.filter(pl.col("oos"))
    must = pol.add_rule(pol.add_realised_pnl(oos, "must"), "must", allow_skip=True)
    assert "skip" not in must["action"].unique().to_list()
    opt = pol.add_rule(pol.add_realised_pnl(oos, "optional"), "optional", allow_skip=True)
    skipped = opt.filter(pl.col("action") == "skip")
    assert (skipped["pnl_rule"] == 0.0).all()
    assert (opt["pnl_oracle"] >= 0.0).all()


def test_expected_shortfall():
    values = pl.Series([-10.0, -4.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
    assert pol.expected_shortfall(values, 0.2) == pytest.approx(-7.0)
    assert pol.expected_shortfall(values, 0.01) == pytest.approx(-10.0)  # at least one observation


def test_bucket_summaries(table):
    by_time = diag.selection_by_bucket(table, "ttg_min", 5)
    assert by_time.height == 5 and by_time["orders"].sum() == table.height
    by_event = diag.selection_by_bucket(table, "event")
    assert by_event["bucket"].to_list() == ["false", "true"]
    curve = diag.markout_curve(table)
    assert curve["seconds"].to_list() == [30, 60, 120, 300, 900]
    assert diag.markout_curve(table.select(list(schema.REQUIRED))).height == 0
