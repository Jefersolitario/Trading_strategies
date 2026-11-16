"""Expected-value maths for the quote-or-cross entry decision.

Conventions
-----------
Everything is per MWh, in price units, measured from the mid at decision time and
signed in the direction of the trade: positive is good for you.

spread       best ask minus best bid at decision time (s)
alpha        expected favourable move from the decision mid to the exit (the signal's edge)
impact       extra cost of crossing your size beyond the touch (>= 0)
p_fill       probability a quote at the near touch fills before the deadline
alpha_fill   expected favourable move on the occasions the quote filled, E[alpha | fill]
chase_cost   how far the price you would cross at moved against you while you waited,
             on the occasions the quote did not fill, E[chase | no fill]

The functions accept floats or numpy arrays.
"""

from __future__ import annotations

import numpy as np

# --------------------------------------------------------------------------- #
# Optional entry: if the quote does not fill, you skip the trade.
# --------------------------------------------------------------------------- #


def ev_cross(alpha, spread, impact=0.0):
    """Expected PnL of crossing now: pay half the spread plus impact, keep alpha."""
    return alpha - spread / 2.0 - impact


def ev_quote_optional(p_fill, alpha_fill, spread):
    """Expected PnL of quoting when a miss means no trade.

    A fill buys at mid - s/2, so it earns alpha_fill + s/2. A miss earns nothing.
    """
    return p_fill * (alpha_fill + spread / 2.0)


def alpha_no_fill(alpha, p_fill, alpha_fill):
    """Alpha on the orders that did not fill, implied by the identity

    alpha = p_fill * alpha_fill + (1 - p_fill) * alpha_no_fill.
    """
    p_fill = np.asarray(p_fill, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = (alpha - p_fill * alpha_fill) / (1.0 - p_fill)
    return float(out) if out.ndim == 0 else out


def breakeven_p_fill_optional(alpha, alpha_fill, spread, impact=0.0):
    """Fill probability above which quoting beats crossing (optional entry).

    Returns nan when a fill is not worth having (alpha_fill + s/2 <= 0), and can
    exceed 1 when crossing wins at every fill probability.
    """
    per_fill = np.asarray(alpha_fill + spread / 2.0, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.where(per_fill > 0, ev_cross(alpha, spread, impact) / per_fill, np.nan)
    return float(out) if out.ndim == 0 else out


# --------------------------------------------------------------------------- #
# Must-complete: if the quote does not fill, you cross at the deadline.
# Alpha cancels because you hold the position either way; only cost matters.
# --------------------------------------------------------------------------- #


def cost_cross(spread, impact=0.0):
    """Execution cost versus mid of crossing now."""
    return spread / 2.0 + impact


def cost_quote_must_complete(p_fill, spread, chase_cost, impact=0.0):
    """Expected execution cost versus mid of quoting first, then chasing a miss.

    Fill: you are paid half the spread (cost -s/2).
    Miss: you cross later at a price that moved by chase_cost (cost s/2 + impact + chase).
    """
    return p_fill * (-spread / 2.0) + (1.0 - p_fill) * (spread / 2.0 + impact + chase_cost)


def quote_edge_must_complete(p_fill, spread, chase_cost, impact=0.0):
    """Expected saving from quoting instead of crossing now (must-complete).

    Equals cost_cross - cost_quote_must_complete:
        p_fill * (s + impact) - (1 - p_fill) * chase_cost
    With impact = 0 this is the rule "quote only if P(fill) * s > P(no fill) * chase".
    """
    return p_fill * (spread + impact) - (1.0 - p_fill) * chase_cost


def breakeven_p_fill_must_complete(spread, chase_cost, impact=0.0):
    """Fill probability above which quoting beats crossing (must-complete)."""
    denom = np.asarray(spread + impact + chase_cost, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.where(denom > 0, chase_cost / denom, np.nan)
    return float(out) if out.ndim == 0 else out
