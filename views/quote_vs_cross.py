"""Quote or cross: when to rest a quote and when to pay the spread on entry."""

from __future__ import annotations

import hashlib
import io
import math

import polars as pl
import streamlit as st

from quote_vs_cross import charts, schema
from quote_vs_cross import decision as dec
from quote_vs_cross import diagnostics as diag
from quote_vs_cross import policies as pol
from quote_vs_cross.models import WalkForwardParams, fill_feature_importance, walk_forward
from quote_vs_cross.simulate import REGIMES, SimParams, simulate

PLOT_CONFIG = {"displayModeBar": False, "scrollZoom": False}
CALC_DEFAULTS = {
    "calc_spread": 4.0,
    "calc_alpha": 3.0,
    "calc_impact": 0.0,
    "calc_p_pct": 50,
    "calc_alpha_fill": 0.5,
    "calc_chase": 5.0,
}
ROLLING_DAYS = 10


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def money(x: float | None, ccy: str, *, signed: bool = False, digits: int = 2) -> str:
    if x is None or not math.isfinite(x):
        return "n/a"
    sign = "-" if x < 0 else "+" if signed and x > 0 else ""
    return f"{sign}{ccy}{abs(x):,.{digits}f}"


def theme_mode() -> str:
    try:
        return st.context.theme.type or "light"
    except Exception:  # older Streamlit, or no browser context yet
        return "light"


def show(fig, title: str, note: str | None = None, table: pl.DataFrame | None = None, *, key: str) -> None:
    """A chart with its title, a one-line reading aid and a table-view twin."""
    st.markdown(f"**{title}**")
    if note:
        st.caption(note)
    st.plotly_chart(fig, width="stretch", theme=None, config=PLOT_CONFIG, key=key)
    if table is not None:
        with st.expander("Table view"):
            st.dataframe(table, width="stretch", hide_index=True)


def rounded(df: pl.DataFrame, digits: int = 3) -> pl.DataFrame:
    return df.with_columns(pl.col(pl.Float64).round(digits))


@st.cache_data(show_spinner=False)
def load_simulated(params: dict) -> pl.DataFrame:
    return simulate(SimParams(**params))


@st.cache_data(show_spinner="Fitting the walk-forward models")
def run_walk_forward(data_key: str, _df: pl.DataFrame, features: tuple[str, ...], wf_key: tuple) -> pl.DataFrame:
    warmup, refit, window = wf_key
    return walk_forward(_df, list(features), WalkForwardParams(warmup, refit, window))


@st.cache_data(show_spinner=False)
def run_importance(data_key: str, _df: pl.DataFrame, features: tuple[str, ...]) -> pl.DataFrame:
    return fill_feature_importance(_df, list(features))


def read_upload(file) -> pl.DataFrame:
    raw = file.getvalue()
    if file.name.lower().endswith(".parquet"):
        return pl.read_parquet(io.BytesIO(raw))
    return pl.read_csv(io.BytesIO(raw), try_parse_dates=True)


# --------------------------------------------------------------------------- #
# Sidebar: data source and estimation settings
# --------------------------------------------------------------------------- #

with st.sidebar:
    st.header("Data")
    source = st.radio(
        "Source",
        ["Simulated market", "Your decisions"],
        help="The simulated market is a teaching device. Upload a decision-level table to use your own orders.",
    )
    upload = None
    sim = SimParams()
    if source == "Your decisions":
        upload = st.file_uploader("Parquet or CSV", type=["parquet", "csv"], help="The Data tab lists the columns.")
    else:
        with st.expander("Simulator settings", expanded=True):
            adverse = st.slider(
                "Adverse selection",
                0.0,
                0.9,
                sim.adverse,
                0.05,
                help="How strongly a fill is tied to the price then moving against you. 0 means fills are luck.",
            )
            base_fill = st.slider("Typical fill probability", 0.1, 0.9, sim.base_fill, 0.05)
            alpha_mean = st.slider("Signal edge (average alpha)", 0.0, 8.0, sim.alpha_mean, 0.25)
            base_spread = st.slider("Typical spread", 1.0, 10.0, sim.base_spread, 0.5)
            vol_mult = st.slider(
                "Volatility (multiple of spread)",
                0.5,
                4.0,
                sim.vol_mult,
                0.25,
                help="Size of the move to exit relative to the spread. Higher means larger chase costs.",
            )
            regime = st.selectbox(
                "Regime change",
                list(REGIMES),
                format_func=REGIMES.get,
                help="Applies to the last third of the sample, to show what each kind of decay looks like.",
            )
            n_days = st.slider("Trading days", 60, 240, sim.n_days, 20)
            seed = st.number_input("Random seed", 0, 9999, sim.seed, 1)
        sim = SimParams(
            n_days=n_days,
            adverse=adverse,
            base_fill=base_fill,
            alpha_mean=alpha_mean,
            base_spread=base_spread,
            vol_mult=vol_mult,
            regime=regime,
            seed=int(seed),
        )

    st.header("Walk-forward")
    warmup_days = st.slider("Warm-up days", 5, 60, 20, 5, help="Days used only for the first fit. Not scored.")
    refit_days = st.slider("Refit every (days)", 1, 20, 5, 1)
    window_choice = st.selectbox("Training window", ["Expanding", "Rolling 30 days", "Rolling 60 days"])
    window_days = {"Expanding": None, "Rolling 30 days": 30, "Rolling 60 days": 60}[window_choice]

# --------------------------------------------------------------------------- #
# Load and validate
# --------------------------------------------------------------------------- #

if source == "Your decisions":
    if upload is None:
        st.title("Quote or cross")
        st.info("Upload a decision-level table in the sidebar, or switch back to the simulated market.")
        st.markdown("**Columns the table needs**")
        st.dataframe(
            pl.DataFrame(
                {
                    "Column": list(schema.REQUIRED),
                    "Type": [kind for _, kind in schema.REQUIRED.values()],
                    "Meaning": [text for text, _ in schema.REQUIRED.values()],
                }
            ),
            width="stretch",
            hide_index=True,
        )
        st.stop()
    try:
        raw_df = read_upload(upload)
    except Exception as exc:  # unreadable file: say so and stop
        st.error(f"Could not read {upload.name}: {exc}")
        st.stop()
    data_key = hashlib.md5(upload.getvalue()).hexdigest()
    is_simulated = False
else:
    raw_df = load_simulated(sim.as_dict())
    data_key = repr(sorted(sim.as_dict().items()))
    is_simulated = True

problems = schema.validate(raw_df)
if problems:
    st.title("Quote or cross")
    st.error("The table cannot be used yet:\n\n" + "\n".join(f"- {p}" for p in problems))
    st.stop()

df = schema.normalise(raw_df)
features = schema.feature_columns(df)
summary = diag.selection_summary(df)

# --------------------------------------------------------------------------- #
# Header and the one filter row that scopes everything below
# --------------------------------------------------------------------------- #

st.title("Quote or cross")
st.caption("Entry execution for power intraday: when to rest a quote and when to pay the spread.")

f1, f2 = st.columns([3, 1])
mode = (
    f1.segmented_control(
        "Entry type",
        list(pol.MODES),
        format_func=pol.MODES.get,
        default="optional",
        help="Optional entry: if the quote does not fill, you skip the trade. "
        "Must-complete: if it does not fill, you cross at the deadline (a hedge leg).",
    )
    or "optional"
)
ccy = f2.selectbox("Currency", ["£", "€"])
unit = f"{ccy}/MWh"
pal = charts.palette(theme_mode())

tab_rule, tab_selection, tab_backtest, tab_model, tab_data = st.tabs(
    ["Decision rule", "Fill selection", "Policy backtest", "Model check", "Data"]
)

# --------------------------------------------------------------------------- #
# Tab 1: the decision rule as a calculator
# --------------------------------------------------------------------------- #


def use_dataset_averages(values: dict[str, float]) -> None:
    st.session_state["calc_spread"] = round(values["spread"], 2)
    st.session_state["calc_alpha"] = round(values["alpha_all"], 2)
    st.session_state["calc_impact"] = round(values["impact"], 2)
    st.session_state["calc_p_pct"] = int(min(99, max(1, round(values["fill_rate"] * 100))))
    st.session_state["calc_alpha_fill"] = round(values["alpha_fill"], 2)
    st.session_state["calc_chase"] = round(max(values["chase_miss"], 0.0), 2)


for _key, _value in CALC_DEFAULTS.items():
    st.session_state.setdefault(_key, _value)
    st.session_state[_key] = st.session_state[_key]  # keep values of inputs hidden in the other mode

with tab_rule:
    if mode == "optional":
        st.markdown(
            "You have a signal and may skip the trade. Crossing pays half the spread for certain. "
            "Quoting earns half the spread, but only on the orders that fill, and those are not a random sample."
        )
    else:
        st.markdown(
            "You will hold the position either way, so alpha cancels and only the entry price matters. "
            "A fill saves the whole spread against crossing now. A miss costs however far the price ran while you waited."
        )

    cols = st.columns(5 if mode == "optional" else 4)
    cols[0].number_input(f"Spread ({unit})", 0.05, 100.0, step=0.25, key="calc_spread", help="Best ask minus best bid.")
    cols[1].number_input(
        f"Impact ({unit})", 0.0, 50.0, step=0.05, key="calc_impact", help="Extra cost of crossing your size beyond the touch."
    )
    cols[2].slider("Fill probability", 1, 99, key="calc_p_pct", format="%d%%", help="Chance the quote fills before the deadline.")
    if mode == "optional":
        cols[3].number_input(
            f"Alpha ({unit})",
            -50.0,
            50.0,
            step=0.25,
            key="calc_alpha",
            help="Expected favourable move from the mid now to your exit, over all orders.",
        )
        cols[4].number_input(
            f"Alpha on fills ({unit})",
            -50.0,
            50.0,
            step=0.25,
            key="calc_alpha_fill",
            help="The same move, averaged only over the orders whose quote filled.",
        )
    else:
        cols[3].number_input(
            f"Chase cost on misses ({unit})",
            0.0,
            100.0,
            step=0.25,
            key="calc_chase",
            help="How far the price you would cross at moved against you while the quote rested, when it did not fill.",
        )
    st.button(
        "Use dataset averages",
        on_click=use_dataset_averages,
        args=(summary,),
        help="Fill the inputs with the averages of the loaded table.",
    )

    s = float(st.session_state["calc_spread"])
    impact = float(st.session_state["calc_impact"])
    p_fill = st.session_state["calc_p_pct"] / 100.0
    alpha = float(st.session_state["calc_alpha"])
    alpha_fill = float(st.session_state["calc_alpha_fill"])
    chase = float(st.session_state["calc_chase"])
    calc = dict(spread=s, alpha=alpha, impact=impact, p_fill=p_fill, alpha_fill=alpha_fill, chase_cost=chase)

    k1, k2, k3, k4 = st.columns(4)
    if mode == "optional":
        cross_ev = dec.ev_cross(alpha, s, impact)
        quote_ev = dec.ev_quote_optional(p_fill, alpha_fill, s)
        edge = quote_ev - cross_ev
        breakeven = dec.breakeven_p_fill_optional(alpha, alpha_fill, s, impact)
        k1.metric("Cross, expected PnL", money(cross_ev, ccy))
        k2.metric("Quote, expected PnL", money(quote_ev, ccy))
        k3.metric("Quote minus cross", money(edge, ccy, signed=True))
        if not math.isfinite(breakeven):
            be_text = "None"
        elif breakeven <= 0:
            be_text = "0%"
        elif breakeven >= 1:
            be_text = "Above 100%"
        else:
            be_text = f"{breakeven:.0%}"
        k4.metric("Breakeven fill probability", be_text, help="Quote beats cross above this fill probability.")

        if max(cross_ev, quote_ev) < 0:
            verdict = f"**Skip.** Both are negative: crossing {money(cross_ev, ccy)}, quoting {money(quote_ev, ccy)} per MWh."
        elif edge > 0:
            verdict = f"**Quote.** It earns {money(quote_ev, ccy)} per MWh against {money(cross_ev, ccy)} for crossing."
        else:
            verdict = f"**Cross.** It earns {money(cross_ev, ccy)} per MWh against {money(quote_ev, ccy)} for quoting."
        st.markdown(verdict)

        miss_alpha = dec.alpha_no_fill(alpha, p_fill, alpha_fill)
        mid, bid, ask = 100.0, 100.0 - s / 2, 100.0 + s / 2
        with st.expander("Worked example in prices (a buy, mid at 100)", expanded=True):
            st.markdown(
                f"- **Cross:** buy at {ask + impact:.2f}, the price ends at {mid + alpha:.2f} on average, "
                f"so every order earns {cross_ev:.2f} on average.\n"
                f"- **Quote:** bid {bid:.2f} and get filled {p_fill:.0%} of the time. On fills the price ends at "
                f"{mid + alpha_fill:.2f}, so a fill earns {alpha_fill + s / 2:.2f} and an order earns "
                f"{p_fill:.2f} x {alpha_fill + s / 2:.2f} = {quote_ev:.2f}.\n"
                f"- **What quoting misses:** the unfilled orders end at {mid + miss_alpha:.2f} on average. "
                f"The overall average is still {mid + alpha:.2f}: "
                f"{p_fill:.2f} x {mid + alpha_fill:.2f} + {1 - p_fill:.2f} x {mid + miss_alpha:.2f}."
            )
        line_table = pl.DataFrame(
            {
                "Fill probability": [f"{p:.0%}" for p in (i / 10 for i in range(11))],
                "Cross": [cross_ev] * 11,
                "Quote": [dec.ev_quote_optional(i / 10, alpha_fill, s) for i in range(11)],
            }
        )
        grid_table = pl.DataFrame(
            {
                "Fill probability": [f"{i / 10:.0%}" for i in range(1, 10)],
                "Breakeven alpha on fills": [cross_ev / (i / 10) - s / 2 for i in range(1, 10)],
                "Quote minus cross at your alpha on fills": [
                    dec.ev_quote_optional(i / 10, alpha_fill, s) - cross_ev for i in range(1, 10)
                ],
            }
        )
        line_title, line_note = "Expected PnL as the fill probability changes", "Higher is better."
        grid_note = "Orange: quoting wins. Blue: crossing wins. The solid line is the breakeven."
    else:
        cross_cost = dec.cost_cross(s, impact)
        quote_cost = dec.cost_quote_must_complete(p_fill, s, chase, impact)
        edge = dec.quote_edge_must_complete(p_fill, s, chase, impact)
        breakeven = dec.breakeven_p_fill_must_complete(s, chase, impact)
        k1.metric("Cross now, cost vs mid", money(cross_cost, ccy))
        k2.metric("Quote then chase, cost vs mid", money(quote_cost, ccy))
        k3.metric("Saving from quoting", money(edge, ccy, signed=True))
        k4.metric(
            "Breakeven fill probability",
            f"{breakeven:.0%}" if math.isfinite(breakeven) else "None",
            help="Quote beats cross above this fill probability.",
        )
        if edge > 0:
            verdict = f"**Quote.** Expected cost {money(quote_cost, ccy)} per MWh against {money(cross_cost, ccy)} for crossing now."
        else:
            verdict = f"**Cross.** Expected cost {money(cross_cost, ccy)} per MWh against {money(quote_cost, ccy)} for quoting first."
        st.markdown(verdict)

        mid, bid, ask = 100.0, 100.0 - s / 2, 100.0 + s / 2
        with st.expander("Worked example in prices (a buy, mid at 100)", expanded=True):
            st.markdown(
                f"- **Cross now:** buy at {ask + impact:.2f}, which is {cross_cost:.2f} above mid.\n"
                f"- **Quote, filled ({p_fill:.0%}):** buy at {bid:.2f}. Against crossing now you save "
                f"{s + impact:.2f}.\n"
                f"- **Quote, missed ({1 - p_fill:.0%}):** the ask has moved up by {chase:.2f}, so you buy at "
                f"{ask + impact + chase:.2f}. Against crossing now you lose {chase:.2f}.\n"
                f"- **Net:** {p_fill:.2f} x {s + impact:.2f} - {1 - p_fill:.2f} x {chase:.2f} = {edge:+.2f} per MWh."
            )
        line_table = pl.DataFrame(
            {
                "Fill probability": [f"{i / 10:.0%}" for i in range(11)],
                "Cross": [cross_cost] * 11,
                "Quote": [dec.cost_quote_must_complete(i / 10, s, chase, impact) for i in range(11)],
            }
        )
        grid_table = pl.DataFrame(
            {
                "Fill probability": [f"{i / 10:.0%}" for i in range(1, 10)],
                "Breakeven chase cost": [(i / 10) * (s + impact) / (1 - i / 10) for i in range(1, 10)],
                "Saving from quoting at your chase cost": [
                    dec.quote_edge_must_complete(i / 10, s, chase, impact) for i in range(1, 10)
                ],
            }
        )
        line_title, line_note = "Expected cost as the fill probability changes", "Lower is better."
        grid_note = "Orange: quoting wins. Blue: crossing wins. The solid line is the breakeven."

    c1, c2 = st.columns(2)
    with c1:
        show(charts.ev_vs_p_fill(pal, mode, unit=unit, **calc), line_title, line_note, rounded(line_table, 2), key="rule_line")
    with c2:
        show(
            charts.edge_heatmap(pal, mode, unit=unit, **calc),
            "Where quoting wins",
            grid_note,
            rounded(grid_table, 2),
            key="rule_grid",
        )

# --------------------------------------------------------------------------- #
# Tab 2: how different are the orders that fill?
# --------------------------------------------------------------------------- #

with tab_selection:
    gap_fill = summary["alpha_fill"] - summary["alpha_all"]
    gap_miss = summary["alpha_miss"] - summary["alpha_all"]
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Fill rate", f"{summary['fill_rate']:.0%}")
    k2.metric("Alpha, all orders", money(summary["alpha_all"], ccy))
    k3.metric("Alpha on fills", money(summary["alpha_fill"], ccy), f"{gap_fill:+.2f} vs all orders")
    k4.metric("Alpha on misses", money(summary["alpha_miss"], ccy), f"{gap_miss:+.2f} vs all orders")
    k5.metric("Chase cost on misses", money(summary["chase_miss"], ccy))
    st.caption(
        f"{summary['orders']:,.0f} orders. The split adds back up: {summary['fill_rate']:.2f} x "
        f"{summary['alpha_fill']:.2f} + {1 - summary['fill_rate']:.2f} x {summary['alpha_miss']:.2f} = "
        f"{summary['alpha_all']:.2f}. Quoting only collects the first term."
    )

    default_feature = features.index("ttg_min") if "ttg_min" in features else 0
    split_by = st.selectbox("Split by", features, index=default_feature, key="split_selection")
    buckets = diag.selection_by_bucket(df, split_by)
    bucket_table = rounded(
        buckets.select("bucket", "orders", "fill_rate", "alpha_all", "alpha_fill", "alpha_miss", "chase_miss")
    )
    c1, c2 = st.columns(2)
    with c1:
        show(
            charts.selection_dumbbell(pal, buckets, unit),
            f"Alpha on fills and on misses, by {split_by}",
            "The wider the gap between the dots, the more a fill tells you the price is about to go against you.",
            bucket_table,
            key="sel_dumbbell",
        )
    with c2:
        show(
            charts.fill_rate_bars(pal, buckets),
            f"Fill rate, by {split_by}",
            "Equal-count buckets, so every bar rests on the same number of orders.",
            bucket_table.select("bucket", "orders", "fill_rate"),
            key="sel_fill_rate",
        )

    curve = diag.markout_curve(df)
    if curve.height:
        show(
            charts.markout_chart(pal, curve, unit),
            "Markout after the decision",
            "Mean favourable move in the mid. Filled orders drifting below zero is adverse selection as it happens.",
            rounded(curve),
            key="sel_markout",
        )

    rolling = diag.rolling_selection(df, ROLLING_DAYS)
    if rolling.height > 1:
        marker_day, marker_text = None, ""
        if is_simulated and sim.regime != "none":
            days = df.select(pl.col("ts").dt.date().unique().sort()).to_series()
            marker_day = days[min(days.len() - 1, int(sim.regime_start * days.len()))]
            marker_text = "Regime change"
        st.markdown("**Which term moved?**")
        st.caption(
            f"{ROLLING_DAYS}-day rolling windows. A falling fill rate means you are being undercut. "
            "Falling alpha on fills means more informed flow is hitting you."
        )
        c1, c2 = st.columns(2)
        with c1:
            show(
                charts.rolling_line(
                    pal, rolling, "fill_rate", y_title="Fill rate", percent=True, marker_day=marker_day, marker_text=marker_text
                ),
                "Rolling fill rate",
                key="sel_roll_fill",
            )
        with c2:
            show(
                charts.rolling_line(
                    pal,
                    rolling,
                    "alpha_fill",
                    y_title=f"Alpha on fills ({unit})",
                    marker_day=marker_day,
                    marker_text=marker_text,
                ),
                "Rolling alpha on fills",
                key="sel_roll_alpha",
            )
        with st.expander("Table view"):
            st.dataframe(rounded(rolling), width="stretch", hide_index=True)

# --------------------------------------------------------------------------- #
# Tab 3: does a per-order rule beat the two static policies out of sample?
# --------------------------------------------------------------------------- #

wf = run_walk_forward(data_key, df, tuple(features), (warmup_days, refit_days, window_days))
oos = wf.filter(pl.col("oos"))

with tab_backtest:
    if oos.height == 0:
        st.warning(
            "No out-of-sample orders: the warm-up is as long as the sample. Shorten the warm-up or load more days."
        )
    else:
        t1, t2 = st.columns(2)
        allow_skip = t1.toggle(
            "Let the rule skip negative-EV orders",
            value=False,
            disabled=mode != "optional",
            help="Optional entries only. Off keeps the comparison about execution alone.",
        )
        show_oracle = t2.toggle(
            "Show the hindsight oracle", value=False, help="The best choice per order knowing the outcome. A ceiling, not a strategy."
        )
        bt = pol.add_rule(pol.add_realised_pnl(oos, mode), mode, allow_skip=allow_skip)
        stats = pol.summarise(bt)
        mean = dict(zip(pol.POLICIES, stats["mean_per_order"]))
        best_static = max(("cross", "quote"), key=lambda name: mean[name])
        lift = round(mean["rule"] - mean[best_static], 2) or 0.0  # no "-0.00"
        headroom = mean["oracle"] - mean[best_static]
        captured = round(lift / headroom, 2) or 0.0 if headroom > 1e-9 else float("nan")
        mix = pol.action_mix(bt)
        share = dict(zip(mix["action"], mix["share"]))

        k1, k2, k3, k4 = st.columns(4)
        k1.metric(
            "Model rule, per order",
            money(mean["rule"], ccy),
            f"{lift:+.2f} vs {pol.POLICY_LABELS[best_static].lower()}",
            delta_color="normal" if lift else "off",
        )
        k2.metric("Always cross, per order", money(mean["cross"], ccy))
        k3.metric("Always quote, per order", money(mean["quote"], ccy))
        k4.metric(
            "Oracle gap captured",
            f"{captured:.0%}" if math.isfinite(captured) else "n/a",
            help="The rule's gain over the best static policy, as a share of what hindsight would have gained.",
        )
        n_days_oos = bt["day"].n_unique()
        mix_text = f"The rule crossed {share['cross']:.0%} and quoted {share['quote']:.0%}"
        mix_text = (
            f"The rule crossed {share['cross']:.0%}, quoted {share['quote']:.0%} and skipped {share['skip']:.0%}."
            if share["skip"] > 0
            else mix_text + "."
        )
        st.caption(
            f"Out of sample: {bt.height:,} orders over {n_days_oos} days. The first {warmup_days} days are warm-up. "
            f"Each day is scored by models fitted only on earlier days. {mix_text}"
        )

        shown = [p for p in pol.POLICIES if show_oracle or p != "oracle"]
        daily = pol.daily_pnl(bt)
        show(
            charts.cumulative_pnl(pal, daily, shown, pol.POLICY_LABELS, ccy),
            "Cumulative PnL by policy",
            "One 1 MWh clip per order, measured from the decision mid.",
            rounded(daily.select("day", "orders", *[f"cum_{p}" for p in shown]), 1),
            key="bt_cum",
        )

        st.markdown("**Policy summary**")
        st.caption("Tail risk is reported per day. Sharpe is left out because these payoffs are asymmetric.")
        st.dataframe(
            stats.with_columns(pl.col("total").round(0)),
            width="stretch",
            hide_index=True,
            column_config={
                "policy": st.column_config.TextColumn("Policy"),
                "mean_per_order": st.column_config.NumberColumn(f"Mean per order ({unit})", format="%.2f"),
                "total": st.column_config.NumberColumn(f"Total ({ccy})", format="localized"),
                "days_positive": st.column_config.NumberColumn("Days positive", format="percent"),
                "es_daily": st.column_config.NumberColumn(f"Daily ES, worst 5% ({ccy})", format="%.0f"),
                "worst_day": st.column_config.NumberColumn(f"Worst day ({ccy})", format="%.0f"),
            },
        )

        actions = [a for a in pol.ACTIONS if a != "skip" or share["skip"] > 0]
        c1, c2 = st.columns([3, 2])
        with c1:
            default_mix = features.index("ttg_min") if "ttg_min" in features else 0
            mix_by = st.selectbox("Split the rule's choices by", features, index=default_mix, key="split_mix")
            mix_buckets = diag.action_mix_by_bucket(bt, mix_by, tuple(actions))
            show(
                charts.action_mix_bars(pal, mix_buckets, actions),
                f"What the rule chose, by {mix_by}",
                "Share of out-of-sample orders in each bucket.",
                rounded(mix_buckets),
                key="bt_mix",
            )
        with c2:
            st.markdown("**Did the rule sort the orders correctly?**")
            st.caption(
                "Mean PnL per order on the orders the rule sent each way. Crossing should win on the orders it "
                "crossed, and quoting on the orders it quoted."
            )
            st.dataframe(
                pol.rule_sorting(bt).with_columns(pl.col("action").str.to_titlecase()),
                width="stretch",
                hide_index=True,
                column_config={
                    "action": st.column_config.TextColumn("Rule chose"),
                    "orders": st.column_config.NumberColumn("Orders", format="localized"),
                    "rule": st.column_config.NumberColumn("Rule", format="%.2f"),
                    "if_crossed": st.column_config.NumberColumn("If crossed", format="%.2f"),
                    "if_quoted": st.column_config.NumberColumn("If quoted", format="%.2f"),
                },
            )

# --------------------------------------------------------------------------- #
# Tab 4: are the estimates behind the rule any good?
# --------------------------------------------------------------------------- #

with tab_model:
    if oos.height == 0:
        st.warning("No out-of-sample predictions to check yet.")
    else:
        st.markdown(
            "The rule is only as good as its inputs. Each chart bins out-of-sample predictions and compares "
            "the average prediction with what happened."
        )
        fill_calib = diag.calibration(oos, "p_fill_hat", pl.col("filled").cast(pl.Float64))
        if mode == "optional":
            second = diag.calibration(oos.filter(pl.col("filled")), "alpha_fill_hat", pl.col("fav_move"))
            second_title = "Alpha on fills: predicted vs realised"
            second_axes = (f"Predicted ({unit})", f"Realised ({unit})")
        else:
            second = diag.calibration(oos.filter(~pl.col("filled")), "chase_hat", pl.col("chase_cost"))
            second_title = "Chase cost on misses: predicted vs realised"
            second_axes = (f"Predicted ({unit})", f"Realised ({unit})")
        c1, c2 = st.columns(2)
        with c1:
            show(
                charts.calibration_chart(
                    pal, fill_calib, x_title="Predicted fill probability", y_title="Realised fill rate", percent=True
                ),
                "Fill probability: predicted vs realised",
                "Ten equal-count bins. Points on the line mean the probabilities can be trusted as probabilities.",
                rounded(fill_calib.drop("bin")),
                key="mod_fill",
            )
        with c2:
            show(
                charts.calibration_chart(pal, second, x_title=second_axes[0], y_title=second_axes[1], percent=False),
                second_title,
                "Noisier than the fill model: the outcome is mostly price noise, so read the slope, not each point.",
                rounded(second.drop("bin")),
                key="mod_second",
            )
        importance = run_importance(data_key, df, tuple(features))
        show(
            charts.importance_bars(pal, importance),
            "What drives the fill model",
            "Permutation importance: fitted on the first 70% of the sample, shuffled and scored on the last 30%.",
            rounded(importance, 4),
            key="mod_importance",
        )

# --------------------------------------------------------------------------- #
# Tab 5: the table itself
# --------------------------------------------------------------------------- #

with tab_data:
    st.markdown("**Columns the table needs**")
    st.caption(
        "One row per entry decision. Price columns are per MWh and signed in the direction of the trade, "
        "so positive is good for you and buys and sells share one table."
    )
    spec = pl.DataFrame(
        {
            "Column": list(schema.REQUIRED) + list(schema.OPTIONAL),
            "Required": ["yes"] * len(schema.REQUIRED) + ["no"] * len(schema.OPTIONAL),
            "Meaning": [text for text, _ in schema.REQUIRED.values()] + list(schema.OPTIONAL.values()),
        }
    )
    st.table(spec)
    st.caption(
        "`filled` has to come from a queue-aware simulation. Marking a quote as filled whenever the price "
        "touches it overstates quoting. Every extra numeric column is used as a model feature, so leave out "
        "anything you would not have known at decision time."
    )

    st.markdown(f"**Loaded table**: {df.height:,} rows, model features: {', '.join(features)}")
    st.dataframe(rounded(df.head(200)), width="stretch", hide_index=True)
    buffer = io.BytesIO()
    df.write_parquet(buffer)
    st.download_button(
        "Download this table as Parquet",
        buffer.getvalue(),
        file_name="decisions.parquet",
        mime="application/octet-stream",
        help="Use it as a template for your own decisions.",
    )
