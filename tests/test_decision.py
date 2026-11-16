"""The decision maths, checked against worked examples small enough to do by hand.

Mid 100, bid 98, ask 102 (spread 4). The signal says the price ends at 103 (alpha 3).
"""

import numpy as np
import pytest

from quote_vs_cross import decision as dec


def test_cross_pays_half_the_spread():
    # Buy at 102, exit at 103.
    assert dec.ev_cross(alpha=3, spread=4) == pytest.approx(1.0)
    assert dec.ev_cross(alpha=3, spread=4, impact=0.25) == pytest.approx(0.75)


def test_quote_wins_when_fills_are_mildly_selected():
    # On fills the price ends at 100.5; bought at 98, so 2.5 per fill, half the time.
    assert dec.ev_quote_optional(p_fill=0.5, alpha_fill=0.5, spread=4) == pytest.approx(1.25)


def test_cross_wins_when_fills_are_badly_selected():
    # On fills the price ends at 99; bought at 98, so 1 per fill, half the time.
    assert dec.ev_quote_optional(p_fill=0.5, alpha_fill=-1.0, spread=4) == pytest.approx(0.5)
    assert dec.ev_quote_optional(0.5, -1.0, 4) < dec.ev_cross(3, 4)


def test_missed_orders_carry_the_alpha_the_fills_lack():
    # alpha = p * alpha_fill + (1 - p) * alpha_no_fill
    assert dec.alpha_no_fill(alpha=3, p_fill=0.5, alpha_fill=0.5) == pytest.approx(5.5)
    assert dec.alpha_no_fill(alpha=3, p_fill=0.5, alpha_fill=-1.0) == pytest.approx(7.0)


def test_optional_breakeven_is_where_the_two_are_equal():
    p_star = dec.breakeven_p_fill_optional(alpha=3, alpha_fill=0.5, spread=4)
    assert p_star == pytest.approx(0.4)
    assert dec.ev_quote_optional(p_star, 0.5, 4) == pytest.approx(dec.ev_cross(3, 4))
    # A fill that loses money even after earning the half-spread has no breakeven.
    assert np.isnan(dec.breakeven_p_fill_optional(alpha=3, alpha_fill=-2.5, spread=4))


def test_must_complete_rule():
    # 60% fill saves the 4 spread; 40% miss costs a 5 chase: 2.4 against 2.0.
    assert dec.quote_edge_must_complete(p_fill=0.6, spread=4, chase_cost=5) == pytest.approx(0.4)
    assert dec.quote_edge_must_complete(p_fill=0.5, spread=4, chase_cost=5) == pytest.approx(-0.5)


def test_must_complete_edge_is_the_difference_in_costs():
    for p, s, chase, impact in [(0.6, 4, 5, 0), (0.3, 2.5, 1.2, 0.4), (0.9, 6, 9, 1.0)]:
        saving = dec.cost_cross(s, impact) - dec.cost_quote_must_complete(p, s, chase, impact)
        assert saving == pytest.approx(dec.quote_edge_must_complete(p, s, chase, impact))


def test_must_complete_breakeven():
    p_star = dec.breakeven_p_fill_must_complete(spread=4, chase_cost=5)
    assert p_star == pytest.approx(5 / 9)
    assert dec.quote_edge_must_complete(p_star, 4, 5) == pytest.approx(0.0)


def test_functions_accept_arrays():
    p = np.array([0.0, 0.5, 1.0])
    np.testing.assert_allclose(dec.ev_quote_optional(p, 0.5, 4), [0.0, 1.25, 2.5])
    np.testing.assert_allclose(dec.quote_edge_must_complete(p, 4, 5), [-5.0, -0.5, 4.0])
