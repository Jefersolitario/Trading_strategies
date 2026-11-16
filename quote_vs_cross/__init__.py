"""Quote-or-cross entry execution for power intraday.

The package is split so the maths can be reused outside the dashboard:

- ``decision``  expected-value formulas (pure functions, floats or arrays)
- ``schema``    the decision-level table the dashboard expects
- ``simulate``  a stylised generator for that table
- ``models``    causal (walk-forward) estimates of the terms in the rule
- ``policies``  realised PnL per policy and risk summaries
- ``diagnostics`` fill-selection summaries, markouts, rolling terms, calibration
- ``charts``    Plotly figure builders
"""

__all__ = ["decision", "schema", "simulate", "models", "policies", "diagnostics", "charts"]
