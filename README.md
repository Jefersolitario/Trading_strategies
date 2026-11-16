# Trading_strategies

My trading ideas, each as a page of one Streamlit dashboard.

## Run it

```bash
pip install -r requirements.txt
streamlit run app.py
```

Tests: `pip install pytest && pytest`.

## Strategy 1: quote or cross

Entry execution for power intraday. Always quoting and always crossing are the two
static endpoints. The edge is in choosing per order.

### Notation

Everything is per MWh, measured from the mid at decision time, and signed in the
direction of the trade (positive is good for you).

| Symbol | Meaning |
|---|---|
| `s` | Spread: best ask minus best bid |
| `alpha` | Expected favourable move from the mid now to the exit, over all orders |
| `impact` | Extra cost of crossing your size beyond the touch |
| `P(fill)` | Probability a quote at the near touch fills before the deadline |
| `E[alpha \| fill]` | The same move, averaged only over the orders whose quote filled |
| chase cost | How far the price you would cross at moved against you while the quote rested, on a miss |

### The rule

**Optional entry** (a miss means no trade):

- Cross earns `alpha - s/2 - impact`.
- Quote earns `P(fill) x (E[alpha | fill] + s/2)`.

Adverse selection is the gap `E[alpha | fill] < alpha`. The identity
`alpha = P(fill) x E[alpha | fill] + P(no fill) x E[alpha | no fill]` always holds,
and quoting only collects the first term.

**Must-complete** (a miss means crossing at the deadline, as on a hedge leg). You hold
the position either way, so alpha cancels:

- Quote only if `P(fill) x (s + impact) > P(no fill) x E[chase cost | no fill]`.

Worked example with mid 100, bid 98, ask 102 and alpha 3: crossing earns 1. Quoting at a
50% fill rate earns 1.25 if fills end at 100.5 on average, and 0.5 if they end at 99.

### The dashboard

| Tab | Question it answers |
|---|---|
| Decision rule | Given my numbers, quote or cross, and where is the breakeven? |
| Fill selection | How much worse are the orders that fill than the ones that miss, and where? |
| Policy backtest | Does a per-order rule beat always-cross and always-quote out of sample? |
| Model check | Can the fill probabilities and conditional alphas behind the rule be trusted? |
| Data | What table does this need, and what is loaded? |

The backtest is causal: each day is scored by models fitted only on earlier days
(walk-forward, expanding or rolling window, with a warm-up that is not scored). The
hindsight oracle is shown as a ceiling. Risk is reported as daily expected shortfall and
worst day, not Sharpe, because the payoffs are asymmetric.

### Your data

The dashboard opens on a simulated market. To use real decisions, upload a Parquet or
CSV file with one row per entry decision:

| Column | Required | Meaning |
|---|---|---|
| `ts` | yes | Decision time |
| `spread` | yes | Best ask minus best bid at decision time |
| `impact` | yes | Extra cost of crossing your size beyond the touch |
| `filled` | yes | Would a quote at the near touch have filled before the deadline |
| `fav_move` | yes | `side x (exit mid - decision mid)` |
| `chase_cost` | yes | `side x (far touch at deadline - far touch at decision)` |
| `mk_<seconds>` | no | Favourable mid move that many seconds after the decision, for the markout curve |
| anything else numeric | no | Used as a model feature |

Two things decide whether the results mean anything:

1. `filled` must come from a queue-aware fill simulation. "Filled if the price touched
   my level" overstates quoting.
2. Every extra numeric column becomes a model feature, so include only what was known
   at decision time.

### What is simulated and what is not

The simulated market (`quote_vs_cross/simulate.py`) is a teaching device. It reproduces
one mechanism: fills are correlated with the part of the future move you could not
predict. It is not calibrated to any market, and its results say nothing about what the
rule would earn on real orders. The "Regime change" setting shows what each kind of
decay looks like in the diagnostics: quotes being undercut (fill rate falls) versus
more informed flow (alpha on fills falls).

## Layout

```
app.py                     entry point; one st.Page per strategy
views/quote_vs_cross.py    the dashboard page
quote_vs_cross/
  decision.py              expected-value formulas (pure functions)
  schema.py                the decision-level table and its validation
  simulate.py              stylised market generator
  models.py                walk-forward estimates of P(fill), alpha, alpha | fill, chase | miss
  policies.py              realised PnL per policy, expected shortfall, summaries
  diagnostics.py           fill-selection summaries, markouts, rolling terms, calibration
  charts.py                Plotly figures and the colour palette
tests/                     worked examples and pipeline checks, including a causality test
```

To add a strategy, add a page under `views/` and register it in `app.py`.
