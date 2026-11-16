"""Plotly figure builders for the quote-or-cross dashboard.

Colour follows the entity everywhere: crossing is blue, quoting is orange, the model
rule is aqua, filled orders are violet and missed orders are yellow. Context series
(all orders, the hindsight oracle, skipped orders) are grey. Titles live in the page,
not in the figure, so they wrap on a phone.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import plotly.graph_objects as go
import polars as pl

from . import decision as dec

FONT = 'system-ui, -apple-system, "Segoe UI", sans-serif'


@dataclass(frozen=True)
class Palette:
    mode: str
    surface: str
    ink: str
    ink2: str
    muted: str
    grid: str
    axis: str
    cross: str
    quote: str
    rule: str
    filled: str
    missed: str
    div_mid: str

    @property
    def context(self) -> str:
        return self.muted

    def policy(self, name: str) -> str:
        return {"cross": self.cross, "quote": self.quote, "rule": self.rule, "oracle": self.muted}[name]

    def action(self, name: str) -> str:
        return {"cross": self.cross, "quote": self.quote, "skip": self.muted}[name]


LIGHT = Palette(
    mode="light",
    surface="#fcfcfb",
    ink="#0b0b0b",
    ink2="#52514e",
    muted="#898781",
    grid="#e1e0d9",
    axis="#c3c2b7",
    cross="#2a78d6",
    quote="#eb6834",
    rule="#1baf7a",
    filled="#6250d6",
    missed="#eda100",
    div_mid="#f0efec",
)
DARK = Palette(
    mode="dark",
    surface="#1a1a19",
    ink="#f0efec",
    ink2="#c3c2b7",
    muted="#898781",
    grid="#2c2c2a",
    axis="#383835",
    cross="#3987e5",
    quote="#d95926",
    rule="#199e70",
    filled="#9085e9",
    missed="#c98500",
    div_mid="#383835",
)


def palette(mode: str | None) -> Palette:
    return DARK if mode == "dark" else LIGHT


# --------------------------------------------------------------------------- #
# Shared chrome
# --------------------------------------------------------------------------- #


def _axis(pal: Palette, title: str | None = None, **kw) -> dict:
    base = dict(
        title=dict(text=title or "", font=dict(color=pal.ink2, size=12), standoff=8),
        showgrid=True,
        gridcolor=pal.grid,
        gridwidth=1,
        zeroline=False,
        showline=True,
        linecolor=pal.axis,
        linewidth=1,
        ticks="",
        tickfont=dict(color=pal.muted, size=11),
        automargin=True,
    )
    base.update(kw)
    return base


def _figure(pal: Palette, *, height: int = 340, legend: bool = True, right: int = 24) -> go.Figure:
    fig = go.Figure()
    fig.update_layout(
        font=dict(family=FONT, size=12, color=pal.ink2),
        paper_bgcolor=pal.surface,
        plot_bgcolor=pal.surface,
        height=height,
        margin=dict(l=8, r=right, t=36 if legend else 16, b=8),
        showlegend=legend,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0,
            font=dict(color=pal.ink2, size=12),
            bgcolor="rgba(0,0,0,0)",
            itemclick=False,
            itemdoubleclick=False,
        ),
        hoverlabel=dict(bgcolor=pal.surface, bordercolor=pal.axis, font=dict(family=FONT, size=12, color=pal.ink)),
        dragmode=False,
    )
    return fig


def _zero_line(fig: go.Figure, pal: Palette) -> None:
    fig.add_hline(y=0, line=dict(color=pal.axis, width=1))


def _marker(pal: Palette, color: str, size: int = 10, symbol: str = "circle") -> dict:
    """A dot with the 2px surface ring that keeps it legible over lines and other dots."""
    return dict(color=color, size=size, symbol=symbol, line=dict(color=pal.surface, width=2))


def compact(x: float) -> str:
    """1,284 / 12.9K / 4.2M."""
    a = abs(x)
    if a >= 1e6:
        return f"{x / 1e6:.1f}M"
    if a >= 1e4:
        return f"{x / 1e3:.1f}K"
    return f"{x:,.0f}"


def _bar_width(n_bars: int, plot_px: int = 420) -> float:
    """Bar width as a share of its slot, so bars stay near 24px at the expected plot width."""
    return min(0.5, 24 * max(n_bars, 1) / plot_px)


def _nice_ticks(lo: float, hi: float, target: int = 5) -> tuple[list[float], list[str]]:
    """Round tick values with one label style per axis: in full under 10,000, compact (10K) above."""
    span = (hi - lo) or 1.0
    raw = span / target
    mag = 10 ** np.floor(np.log10(raw))
    step = min((1, 2, 2.5, 5, 10), key=lambda m: abs(m * mag - raw)) * mag
    vals = np.arange(np.ceil(lo / step) * step, hi + step * 0.25, step)
    vals = [0.0 if abs(v) < step * 1e-9 else float(v) for v in vals]
    if max(abs(lo), abs(hi)) >= 1e6:
        text = ["0" if v == 0 else f"{v / 1e6:g}M" for v in vals]
    elif max(abs(lo), abs(hi)) >= 1e4:
        text = ["0" if v == 0 else f"{v / 1e3:g}K" for v in vals]
    else:
        text = [f"{v:,.0f}" if step >= 1 else f"{v:,.2f}" for v in vals]
    return vals, text


def fmt_seconds(seconds: int) -> str:
    return f"{seconds}s" if seconds < 60 else f"{seconds // 60}m" if seconds < 3600 else f"{seconds / 3600:g}h"


# --------------------------------------------------------------------------- #
# Decision rule
# --------------------------------------------------------------------------- #


def ev_vs_p_fill(
    pal: Palette,
    mode: str,
    *,
    spread: float,
    alpha: float,
    impact: float,
    p_fill: float,
    alpha_fill: float,
    chase_cost: float,
    unit: str,
) -> go.Figure:
    """Cross versus quote as the fill probability varies, with the breakeven marked.

    Optional entry plots expected PnL (higher is better). Must-complete plots expected
    execution cost versus mid (lower is better), because alpha cancels.
    """
    p = np.linspace(0.0, 1.0, 101)
    if mode == "optional":
        y_cross = np.full_like(p, dec.ev_cross(alpha, spread, impact))
        y_quote = dec.ev_quote_optional(p, alpha_fill, spread)
        y_now = dec.ev_quote_optional(p_fill, alpha_fill, spread)
        breakeven = dec.breakeven_p_fill_optional(alpha, alpha_fill, spread, impact)
        y_title = f"Expected PnL ({unit})"
    else:
        y_cross = np.full_like(p, dec.cost_cross(spread, impact))
        y_quote = dec.cost_quote_must_complete(p, spread, chase_cost, impact)
        y_now = dec.cost_quote_must_complete(p_fill, spread, chase_cost, impact)
        breakeven = dec.breakeven_p_fill_must_complete(spread, chase_cost, impact)
        y_title = f"Expected cost vs mid ({unit})"

    fig = _figure(pal, right=16)
    line = dict(width=2, shape="linear")
    fig.add_trace(
        go.Scatter(x=p, y=y_cross, name="Cross", mode="lines", line=dict(color=pal.cross, **line), hovertemplate="%{y:.2f}")
    )
    fig.add_trace(
        go.Scatter(x=p, y=y_quote, name="Quote", mode="lines", line=dict(color=pal.quote, **line), hovertemplate="%{y:.2f}")
    )
    fig.add_trace(
        go.Scatter(
            x=[p_fill],
            y=[y_now],
            name="Your inputs",
            mode="markers",
            marker=_marker(pal, pal.ink, 11),
            hovertemplate="%{y:.2f}",
        )
    )
    if np.isfinite(breakeven) and 0.0 < breakeven < 1.0:
        fig.add_vline(x=float(breakeven), line=dict(color=pal.muted, width=1, dash="dot"))
        fig.add_annotation(
            x=float(breakeven),
            y=1.0,
            yref="paper",
            text=f"Breakeven {breakeven:.0%}",
            showarrow=False,
            xanchor="left" if breakeven < 0.7 else "right",
            xshift=6 if breakeven < 0.7 else -6,
            yanchor="top",
            font=dict(color=pal.ink2, size=11),
        )
    _zero_line(fig, pal)
    fig.update_layout(
        hovermode="x unified",
        xaxis=_axis(pal, "Fill probability", tickformat=".0%", range=[0, 1], hoverformat=".0%"),
        yaxis=_axis(pal, y_title),
    )
    return fig


def edge_heatmap(
    pal: Palette,
    mode: str,
    *,
    spread: float,
    alpha: float,
    impact: float,
    p_fill: float,
    alpha_fill: float,
    chase_cost: float,
    unit: str,
) -> go.Figure:
    """Quote minus cross over a grid of the two terms you have to estimate.

    Orange is where quoting wins, blue where crossing wins, grey is a wash.
    The solid line is the breakeven.
    """
    p = np.linspace(0.02, 0.98, 49)
    if mode == "optional":
        lo = min(alpha_fill, -spread) - 0.25 * spread
        hi = max(alpha_fill, alpha) + 0.75 * spread
        y = np.linspace(lo, hi, 49)
        pp, yy = np.meshgrid(p, y)
        z = dec.ev_quote_optional(pp, yy, spread) - dec.ev_cross(alpha, spread, impact)
        cross_ev = dec.ev_cross(alpha, spread, impact)
        with np.errstate(divide="ignore", invalid="ignore"):
            be = cross_ev / p - spread / 2.0  # alpha_fill at which the two are equal
        y_now, y_title, y_hover = alpha_fill, f"Alpha on fills ({unit})", "Alpha on fills"
    else:
        hi = max(chase_cost, spread) * 2.0
        y = np.linspace(0.0, hi, 49)
        pp, yy = np.meshgrid(p, y)
        z = dec.quote_edge_must_complete(pp, spread, yy, impact)
        be = p * (spread + impact) / (1.0 - p)  # chase cost at which the two are equal
        y_now, y_title, y_hover = chase_cost, f"Chase cost on misses ({unit})", "Chase cost"

    bound = float(np.nanmax(np.abs(z))) or 1.0
    fig = _figure(pal, legend=True, right=8)
    fig.add_trace(
        go.Heatmap(
            x=p,
            y=y,
            z=z,
            zmin=-bound,
            zmax=bound,
            zmid=0.0,
            colorscale=[[0.0, pal.cross], [0.5, pal.div_mid], [1.0, pal.quote]],
            colorbar=dict(
                title=dict(text=f"Quote minus cross ({unit})", side="right", font=dict(color=pal.ink2, size=11)),
                thickness=10,
                outlinewidth=0,
                tickfont=dict(color=pal.muted, size=11),
                len=0.9,
            ),
            hovertemplate=f"Fill probability %{{x:.0%}}<br>{y_hover} %{{y:.2f}}<br><b>%{{z:+.2f}}</b> quote minus cross<extra></extra>",
            showlegend=False,
        )
    )
    inside = (be >= y.min()) & (be <= y.max())
    if inside.any():
        fig.add_trace(
            go.Scatter(
                x=p[inside],
                y=be[inside],
                name="Breakeven",
                mode="lines",
                line=dict(color=pal.ink, width=2),
                hoverinfo="skip",
            )
        )
    fig.add_trace(
        go.Scatter(
            x=[p_fill],
            y=[y_now],
            name="Your inputs",
            mode="markers",
            marker=_marker(pal, pal.ink, 12),
            hoverinfo="skip",
        )
    )
    for wins, label in ((z > 0, "Quote wins"), (z < 0, "Cross wins")):
        if wins.mean() > 0.06:
            fig.add_annotation(
                x=float(pp[wins].mean()),
                y=float(yy[wins].mean()),
                text=f"<b>{label}</b>",
                showarrow=False,
                font=dict(color=pal.ink, size=12),
            )
    fig.update_layout(
        xaxis=_axis(pal, "Fill probability", tickformat=".0%", showgrid=False, range=[0.02, 0.98]),
        yaxis=_axis(pal, y_title, showgrid=False, range=[float(y.min()), float(y.max())]),
    )
    return fig


# --------------------------------------------------------------------------- #
# Fill selection
# --------------------------------------------------------------------------- #


def selection_dumbbell(pal: Palette, buckets: pl.DataFrame, unit: str) -> go.Figure:
    """Alpha on fills versus alpha on misses in each bucket, with all orders as the reference."""
    x = buckets["bucket"].to_list()
    fig = _figure(pal)
    xs, ys = [], []
    for label, lo, hi in zip(x, buckets["alpha_fill"], buckets["alpha_miss"]):
        if lo is None or hi is None:
            continue
        xs += [label, label, None]
        ys += [lo, hi, None]
    fig.add_trace(
        go.Scatter(x=xs, y=ys, mode="lines", line=dict(color=pal.axis, width=2), hoverinfo="skip", showlegend=False)
    )
    series = (
        ("Missed", "alpha_miss", pal.missed, "circle", 11),
        ("All orders", "alpha_all", pal.context, "diamond", 9),
        ("Filled", "alpha_fill", pal.filled, "circle", 11),
    )
    for name, col, color, symbol, size in series:
        fig.add_trace(
            go.Scatter(
                x=x,
                y=buckets[col].to_list(),
                name=name,
                mode="markers",
                marker=_marker(pal, color, size, symbol),
                hovertemplate="%{y:.2f}",
            )
        )
    _zero_line(fig, pal)
    fig.update_layout(
        hovermode="x unified",
        xaxis=_axis(pal, showgrid=False, type="category"),
        yaxis=_axis(pal, f"Favourable move to exit ({unit})"),
    )
    return fig


def fill_rate_bars(pal: Palette, buckets: pl.DataFrame) -> go.Figure:
    """Fill rate per bucket. One series, so no legend: the title names it."""
    x = buckets["bucket"].to_list()
    rates = buckets["fill_rate"].to_list()
    fig = _figure(pal, legend=False)
    fig.add_trace(
        go.Bar(
            x=x,
            y=rates,
            width=_bar_width(len(x)),
            marker=dict(color=pal.filled, cornerradius=4),
            text=[f"{r:.0%}" for r in rates],
            textposition="outside",
            textfont=dict(color=pal.ink2, size=11),
            constraintext="none",
            cliponaxis=False,
            hovertemplate="%{x}<br><b>%{y:.1%}</b> filled<extra></extra>",
        )
    )
    top = max(rates) if rates else 1.0
    fig.update_layout(
        xaxis=_axis(pal, showgrid=False, type="category"),
        yaxis=_axis(pal, "Fill rate", tickformat=".0%", range=[0, min(1.0, top * 1.2 + 0.02)]),
    )
    return fig


def markout_chart(pal: Palette, curve: pl.DataFrame, unit: str) -> go.Figure:
    """Mean favourable mid move after the decision, split by whether the quote filled."""
    seconds = curve["seconds"].to_list()
    fig = _figure(pal, right=16)
    for name, col, color in (
        ("Missed", "missed", pal.missed),
        ("All orders", "all", pal.context),
        ("Filled", "filled", pal.filled),
    ):
        fig.add_trace(
            go.Scatter(
                x=seconds,
                y=curve[col].to_list(),
                name=name,
                mode="lines+markers",
                line=dict(color=color, width=2),
                marker=_marker(pal, color, 9),
                hovertemplate="%{y:.2f}",
            )
        )
    _zero_line(fig, pal)
    fig.update_layout(
        hovermode="x unified",
        xaxis=_axis(
            pal,
            "Time since decision",
            type="log",
            tickvals=seconds,
            ticktext=[fmt_seconds(s) for s in seconds],
            showgrid=False,
        ),
        yaxis=_axis(pal, f"Favourable mid move ({unit})"),
    )
    return fig


def rolling_line(
    pal: Palette,
    rolling: pl.DataFrame,
    column: str,
    *,
    y_title: str,
    percent: bool = False,
    marker_day=None,
    marker_text: str = "",
) -> go.Figure:
    """A single rolling series through time, with an optional dated marker."""
    fig = _figure(pal, legend=False, right=16)
    fig.add_trace(
        go.Scatter(
            x=rolling["day"].to_list(),
            y=rolling[column].to_list(),
            mode="lines",
            line=dict(color=pal.filled, width=2),
            hovertemplate="%{x|%d %b %Y}<br><b>%{y:.1%}</b><extra></extra>"
            if percent
            else "%{x|%d %b %Y}<br><b>%{y:.2f}</b><extra></extra>",
        )
    )
    if marker_day is not None:
        fig.add_shape(
            type="line",
            x0=marker_day,
            x1=marker_day,
            y0=0,
            y1=1,
            yref="paper",
            line=dict(color=pal.muted, width=1, dash="dot"),
        )
        fig.add_annotation(
            x=marker_day,
            y=1.0,
            yref="paper",
            text=marker_text,
            showarrow=False,
            xanchor="right",
            xshift=-6,
            yanchor="top",
            font=dict(color=pal.ink2, size=11),
        )
    if not percent:
        _zero_line(fig, pal)
    fig.update_layout(
        height=260,
        xaxis=_axis(pal, showgrid=False),
        yaxis=_axis(pal, y_title, tickformat=".0%" if percent else None),
    )
    return fig


# --------------------------------------------------------------------------- #
# Policy backtest
# --------------------------------------------------------------------------- #


def cumulative_pnl(pal: Palette, daily: pl.DataFrame, policies: list[str], labels: dict[str, str], unit: str) -> go.Figure:
    """Cumulative PnL per policy. End values are labelled only where they do not collide."""
    days = daily["day"].to_list()
    fig = _figure(pal, right=56)
    ends: dict[str, float] = {}
    for name in policies:
        y = daily[f"cum_{name}"].to_list()
        ends[name] = y[-1]
        fig.add_trace(
            go.Scatter(
                x=days,
                y=y,
                name=labels[name],
                mode="lines",
                line=dict(color=pal.policy(name), width=2, dash="dash" if name == "oracle" else "solid"),
                hovertemplate="%{y:,.0f}",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=[days[-1]],
                y=[y[-1]],
                mode="markers",
                marker=_marker(pal, pal.policy(name), 9),
                hoverinfo="skip",
                showlegend=False,
            )
        )
    lo = min(0.0, *[daily[f"cum_{n}"].min() for n in policies])
    hi = max(0.0, *[daily[f"cum_{n}"].max() for n in policies])
    gap = 0.07 * (hi - lo or 1.0)
    placed: list[float] = []
    for name in sorted(policies, key=lambda n: ("rule", "quote", "cross", "oracle").index(n)):
        if all(abs(ends[name] - other) > gap for other in placed):
            placed.append(ends[name])
            fig.add_annotation(
                x=days[-1],
                y=ends[name],
                text=compact(ends[name]),
                showarrow=False,
                xanchor="left",
                xshift=8,
                font=dict(color=pal.ink, size=11),
            )
    _zero_line(fig, pal)
    pad = 0.05 * (hi - lo or 1.0)
    tickvals, ticktext = _nice_ticks(lo, hi)
    fig.update_layout(
        hovermode="x unified",
        xaxis=_axis(pal, showgrid=False, hoverformat="%d %b %Y"),
        yaxis=_axis(
            pal,
            f"Cumulative PnL ({unit}, 1 MWh clips)",
            tickvals=tickvals,
            ticktext=ticktext,
            range=[lo - pad if lo < 0 else 0, hi + pad],
        ),
    )
    return fig


def action_mix_bars(pal: Palette, mix: pl.DataFrame, actions: list[str]) -> go.Figure:
    """Share of orders the rule crossed, quoted or skipped in each bucket (stacked to 100%)."""
    x = mix["bucket"].to_list()
    names = {"cross": "Cross", "quote": "Quote", "skip": "Skip"}
    fig = _figure(pal)
    for action in actions:
        shares = mix[action].to_list()
        fig.add_trace(
            go.Bar(
                x=x,
                y=shares,
                name=names[action],
                width=_bar_width(len(x), 480),
                marker=dict(color=pal.action(action), line=dict(color=pal.surface, width=2)),
                hovertemplate="%{y:.0%}",
            )
        )
    fig.update_layout(
        barmode="stack",
        legend_traceorder="normal",
        hovermode="x unified",
        xaxis=_axis(pal, showgrid=False, type="category"),
        yaxis=_axis(pal, "Share of orders", tickformat=".0%", range=[0, 1]),
    )
    return fig


# --------------------------------------------------------------------------- #
# Model check
# --------------------------------------------------------------------------- #


def calibration_chart(pal: Palette, calib: pl.DataFrame, *, x_title: str, y_title: str, percent: bool) -> go.Figure:
    """Predicted versus realised by bin. Points on the diagonal mean the model is calibrated."""
    predicted, realised = calib["predicted"].to_list(), calib["realised"].to_list()
    lo, hi = min(predicted + realised), max(predicted + realised)
    pad = 0.06 * (hi - lo or 1.0)
    lo, hi = lo - pad, hi + pad
    fmt = ".0%" if percent else ".2f"
    fig = _figure(pal, legend=True)
    fig.add_trace(
        go.Scatter(
            x=[lo, hi],
            y=[lo, hi],
            name="Perfect calibration",
            mode="lines",
            line=dict(color=pal.muted, width=1),
            hoverinfo="skip",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=predicted,
            y=realised,
            name="Out-of-sample bins",
            mode="markers",
            marker=_marker(pal, pal.filled, 11),
            hovertemplate=f"Predicted %{{x:{fmt}}}<br><b>%{{y:{fmt}}}</b> realised<extra></extra>",
        )
    )
    tick = ".0%" if percent else None
    fig.update_layout(
        xaxis=_axis(pal, x_title, range=[lo, hi], tickformat=tick),
        yaxis=_axis(pal, y_title, range=[lo, hi], tickformat=tick),
    )
    return fig


def importance_bars(pal: Palette, importance: pl.DataFrame) -> go.Figure:
    """Ranked horizontal bars, largest first, one hue."""
    ranked = importance.sort("importance")  # plotly draws the first row at the bottom
    features = ranked["feature"].to_list()
    values = [max(v, 0.0) for v in ranked["importance"].to_list()]  # below zero is noise; the table keeps it
    fig = _figure(pal, legend=False, height=30 * len(features) + 70, right=48)
    fig.add_trace(
        go.Bar(
            x=values,
            y=features,
            orientation="h",
            width=0.55,
            marker=dict(color=pal.filled, cornerradius=4),
            text=[f"{v:.3f}" for v in values],
            textposition="outside",
            textfont=dict(color=pal.ink2, size=11),
            constraintext="none",
            cliponaxis=False,
            hovertemplate="%{y}<br><b>%{x:.4f}</b><extra></extra>",
        )
    )
    fig.update_layout(
        xaxis=_axis(pal, "Loss of out-of-sample log-likelihood when shuffled", rangemode="tozero"),
        yaxis=_axis(pal, showgrid=False, tickfont=dict(color=pal.ink2, size=12)),
    )
    return fig
