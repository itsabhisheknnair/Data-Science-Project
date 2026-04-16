from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from crashrisk.models.score import assign_risk_buckets


# ── Risk metrics ──────────────────────────────────────────────────────────────

def _annualized_return(weekly_returns: np.ndarray) -> float:
    """Geometric annualized return from weekly return series."""
    if len(weekly_returns) < 2:
        return float("nan")
    return float(np.prod(1.0 + weekly_returns) ** (52.0 / len(weekly_returns)) - 1.0)


def _sharpe_ratio(weekly_returns: np.ndarray, weekly_rf: float = 0.04 / 52) -> float:
    """Annualized Sharpe ratio (excess return / total volatility)."""
    excess = weekly_returns - weekly_rf
    std = np.std(excess, ddof=1)
    if len(excess) < 4 or std == 0:
        return float("nan")
    return float(np.mean(excess) / std * np.sqrt(52.0))


def _sortino_ratio(weekly_returns: np.ndarray, weekly_rf: float = 0.04 / 52) -> float:
    """Annualized Sortino ratio (excess return / downside volatility)."""
    excess = weekly_returns - weekly_rf
    downside = excess[excess < 0.0]
    if len(downside) < 2:
        return float("nan")
    downside_std = np.std(downside, ddof=1)
    if downside_std == 0:
        return float("nan")
    return float(np.mean(excess) / downside_std * np.sqrt(52.0))


def _max_drawdown(weekly_returns: np.ndarray) -> float:
    """Maximum peak-to-trough drawdown of a return series."""
    if len(weekly_returns) < 2:
        return float("nan")
    cumulative = np.cumprod(1.0 + weekly_returns)
    running_peak = np.maximum.accumulate(cumulative)
    drawdowns = (cumulative - running_peak) / running_peak
    return float(np.min(drawdowns))


def _var_cvar(weekly_returns: np.ndarray, confidence: float = 0.95) -> tuple[float, float]:
    """
    Historical VaR and CVaR (Expected Shortfall) at given confidence level.

    Both are returned as positive numbers (losses expressed as positive values).
    """
    if len(weekly_returns) < 10:
        return float("nan"), float("nan")
    sorted_rets = np.sort(weekly_returns)
    cutoff_idx = max(1, int(len(sorted_rets) * (1.0 - confidence)))
    var = float(-sorted_rets[cutoff_idx - 1])
    cvar = float(-np.mean(sorted_rets[:cutoff_idx]))
    return var, cvar


# ── Portfolio construction ────────────────────────────────────────────────────

def _build_weekly_portfolio_returns(
    price_history: pd.DataFrame,
    scores: pd.DataFrame,
    eval_start_quantile: float = 0.60,
) -> pd.DataFrame:
    """
    Build equal-weight strategy and benchmark weekly return series.

    Strategy: each week, hold only stocks NOT in the 'High' risk bucket.
    Benchmark: equal-weight all stocks each week.

    Uses the last (1 - eval_start_quantile) fraction of the price history
    as the evaluation window so there is no look-ahead from the training data.
    """
    ph = price_history.copy()
    ph["date"] = pd.to_datetime(ph["date"])
    ph = ph.sort_values(["ticker", "date"])
    ph["weekly_return"] = ph.groupby("ticker", group_keys=False)["adj_close"].pct_change()

    all_dates = ph["date"].dropna().sort_values().unique()
    if len(all_dates) < 10:
        return pd.DataFrame()
    eval_start = all_dates[int(len(all_dates) * eval_start_quantile)]
    eval_data = ph[ph["date"] > eval_start].copy()
    if eval_data.empty:
        return pd.DataFrame()

    ticker_bucket = scores.set_index("ticker")["risk_bucket"].to_dict()
    eval_data["risk_bucket"] = eval_data["ticker"].map(ticker_bucket).fillna("Low")

    portfolio_weeks: list[dict] = []
    for date, group in eval_data.groupby("date"):
        valid = group.dropna(subset=["weekly_return"])
        if valid.empty:
            continue
        benchmark_ret = float(valid["weekly_return"].mean())
        strategy_group = valid[valid["risk_bucket"] != "High"]
        strategy_ret = float(strategy_group["weekly_return"].mean()) if not strategy_group.empty else benchmark_ret
        portfolio_weeks.append({"date": date, "strategy": strategy_ret, "benchmark": benchmark_ret})

    if not portfolio_weeks:
        return pd.DataFrame()
    return pd.DataFrame(portfolio_weeks).sort_values("date").set_index("date")


def _positive_class_index(model: Any) -> int:
    classifier = getattr(model, "named_steps", {}).get("classifier")
    classes = list(getattr(classifier, "classes_", [0, 1]))
    return classes.index(1) if 1 in classes else len(classes) - 1


def build_weekly_forward_portfolio_returns(
    panel: pd.DataFrame,
    model: Any,
    eval_start_quantile: float = 0.60,
    high_share: float = 0.20,
) -> pd.DataFrame:
    """
    Build a forward-looking weekly overlay backtest.

    Risk scores at week t decide the equal-weight portfolio held for the
    following week t+1. The benchmark is the equal-weight return of all
    available stocks over the same following week.
    """
    required = {"ticker", "date"}
    missing = required - set(panel.columns)
    if missing:
        raise ValueError(f"panel is missing required column(s): {', '.join(sorted(missing))}")

    feature_columns = list(getattr(model, "feature_columns_", []))
    if not feature_columns:
        raise ValueError("model must expose feature_columns_ for weekly forward portfolio scoring")
    feature_missing = [column for column in feature_columns if column not in panel.columns]
    if feature_missing:
        raise ValueError(f"panel is missing feature column(s): {', '.join(feature_missing)}")

    ph = panel.copy()
    ph["date"] = pd.to_datetime(ph["date"])
    ph = ph.sort_values(["ticker", "date"]).reset_index(drop=True)
    if "weekly_return" not in ph.columns:
        if "adj_close" not in ph.columns:
            raise ValueError("panel must contain weekly_return or adj_close")
        ph["weekly_return"] = ph.groupby("ticker", group_keys=False)["adj_close"].pct_change()
    ph["next_week_return"] = ph.groupby("ticker", group_keys=False)["weekly_return"].shift(-1)

    all_dates = pd.Series(ph["date"].dropna().unique()).sort_values().tolist()
    if len(all_dates) < 3:
        return pd.DataFrame(columns=["date", "return_date", "strategy", "benchmark", "n_holdings", "n_excluded", "excluded_tickers"])
    eval_start = all_dates[int(len(all_dates) * eval_start_quantile)]
    positive_index = _positive_class_index(model)

    rows: list[dict[str, object]] = []
    for rebalance_date in all_dates:
        if rebalance_date <= eval_start:
            continue
        week = ph.loc[ph["date"] == rebalance_date].dropna(subset=["next_week_return"]).copy()
        if week.empty:
            continue

        probabilities = pd.Series(
            model.predict_proba(week[feature_columns])[:, positive_index],
            index=week.index,
            name="crash_probability",
        )
        week["crash_probability"] = probabilities
        week["risk_bucket"] = assign_risk_buckets(probabilities, high_share=high_share)

        strategy_group = week.loc[week["risk_bucket"] != "High"]
        high_group = week.loc[week["risk_bucket"] == "High"].sort_values("ticker")
        benchmark_ret = float(week["next_week_return"].mean())
        strategy_ret = float(strategy_group["next_week_return"].mean()) if not strategy_group.empty else benchmark_ret
        return_dates = [date for date in all_dates if date > rebalance_date]

        rows.append(
            {
                "date": pd.Timestamp(rebalance_date),
                "return_date": pd.Timestamp(return_dates[0]) if return_dates else pd.NaT,
                "strategy": strategy_ret,
                "benchmark": benchmark_ret,
                "n_holdings": int(len(strategy_group)),
                "n_excluded": int(len(high_group)),
                "excluded_tickers": ";".join(high_group["ticker"].astype(str).tolist()),
            }
        )

    return pd.DataFrame(rows)


# ── Main public function ──────────────────────────────────────────────────────

def compute_business_analysis(
    price_history: pd.DataFrame,
    scores: pd.DataFrame | None = None,
    model: Any | None = None,
    portfolio_returns: pd.DataFrame | None = None,
    fund_aum: float = 1_000_000_000.0,
    annual_risk_free: float = 0.04,
    team_annual_cost: float = 800_000.0,
) -> dict[str, object]:
    """
    Compute portfolio-level performance metrics and economic value of the crash-risk model.

    Strategy
    --------
    Equal-weight all stocks NOT in the 'High' crash-risk bucket each week.
    Benchmark: equal-weight all stocks.

    Parameters
    ----------
    price_history:    DataFrame with columns [ticker, date, adj_close].
    scores:           DataFrame with columns [ticker, risk_bucket].
    fund_aum:         Hypothetical fund size in USD (default $1 billion).
    annual_risk_free: Annual risk-free rate (default 4%).
    team_annual_cost: Annual cost of the four-person implementation team.

    Returns
    -------
    Dict containing:
      strategy_annual_return, benchmark_annual_return, alpha_annualized
      strategy_sharpe, benchmark_sharpe, strategy_sortino, benchmark_sortino
      max_drawdown_strategy, max_drawdown_benchmark, drawdown_improvement
      var_95_weekly, benchmark_var_95_weekly, cvar_95_weekly, benchmark_cvar_95_weekly
      evaluation_weeks, high_risk_excluded_pct
      fund_aum, economic_gain_annual, team_annual_cost, team_roi, justifies_team
    """
    if price_history.empty:
        return {"error": "insufficient data for business analysis"}

    weekly_rf = annual_risk_free / 52.0
    if portfolio_returns is not None:
        perf = portfolio_returns.copy()
    elif model is not None:
        perf = build_weekly_forward_portfolio_returns(price_history, model)
    elif scores is not None and not scores.empty:
        perf = _build_weekly_portfolio_returns(price_history, scores)
    else:
        return {"error": "business analysis requires either a model or risk scores"}

    if perf.empty or len(perf) < 8:
        return {"error": "not enough evaluation weeks for business analysis"}

    strategy_rets = perf["strategy"].fillna(0.0).to_numpy()
    benchmark_rets = perf["benchmark"].fillna(0.0).to_numpy()

    annual_strategy = _annualized_return(strategy_rets)
    annual_benchmark = _annualized_return(benchmark_rets)
    alpha = annual_strategy - annual_benchmark

    strategy_sharpe = _sharpe_ratio(strategy_rets, weekly_rf)
    benchmark_sharpe = _sharpe_ratio(benchmark_rets, weekly_rf)
    strategy_sortino = _sortino_ratio(strategy_rets, weekly_rf)
    benchmark_sortino = _sortino_ratio(benchmark_rets, weekly_rf)

    mdd_strategy = _max_drawdown(strategy_rets)
    mdd_benchmark = _max_drawdown(benchmark_rets)

    var_95, cvar_95 = _var_cvar(strategy_rets)
    benchmark_var_95, benchmark_cvar_95 = _var_cvar(benchmark_rets)

    economic_gain = fund_aum * alpha
    team_roi = economic_gain / team_annual_cost if team_annual_cost > 0 else float("nan")

    if {"n_excluded", "n_holdings"}.issubset(perf.columns):
        weekly_universe = perf["n_excluded"] + perf["n_holdings"]
        high_pct = float((perf["n_excluded"] / weekly_universe.replace(0, np.nan)).mean())
    elif scores is not None and "risk_bucket" in scores:
        ticker_buckets = scores["risk_bucket"].tolist()
        high_pct = sum(1 for b in ticker_buckets if b == "High") / max(1, len(ticker_buckets))
    else:
        high_pct = float("nan")

    def _fmt(v: float, digits: int = 4) -> float | None:
        return round(v, digits) if np.isfinite(v) else None

    return {
        "strategy_annual_return": _fmt(annual_strategy),
        "benchmark_annual_return": _fmt(annual_benchmark),
        "alpha_annualized": _fmt(alpha),
        "benchmark_alpha_annualized": 0.0,
        "strategy_sharpe": _fmt(strategy_sharpe, 3),
        "benchmark_sharpe": _fmt(benchmark_sharpe, 3),
        "strategy_sortino": _fmt(strategy_sortino, 3),
        "benchmark_sortino": _fmt(benchmark_sortino, 3),
        "max_drawdown_strategy": _fmt(mdd_strategy),
        "max_drawdown_benchmark": _fmt(mdd_benchmark),
        "drawdown_improvement": _fmt(mdd_benchmark - mdd_strategy),
        "var_95_weekly": _fmt(var_95),
        "benchmark_var_95_weekly": _fmt(benchmark_var_95),
        "cvar_95_weekly": _fmt(cvar_95),
        "benchmark_cvar_95_weekly": _fmt(benchmark_cvar_95),
        "evaluation_weeks": int(len(strategy_rets)),
        "high_risk_excluded_pct": _fmt(high_pct, 3),
        "benchmark_high_risk_excluded_pct": 0.0,
        "business_analysis_method": "weekly_forward_overlay" if model is not None or portfolio_returns is not None else "latest_score_overlay",
        "fund_aum": fund_aum,
        "economic_gain_annual": round(economic_gain, 0) if np.isfinite(economic_gain) else None,
        "benchmark_economic_gain_annual": 0.0,
        "team_annual_cost": team_annual_cost,
        "team_roi": _fmt(team_roi, 2),
        "benchmark_team_roi": "-",
        "justifies_team": bool(np.isfinite(economic_gain) and economic_gain > team_annual_cost),
        "business_analysis_note": (
            "The economic results are based on a stylised simulation and should be interpreted as illustrative. "
            "In practice, transaction costs, market impact, capacity limits, model uncertainty, "
            "and live implementation slippage would likely reduce realised returns."
        ),
    }


def business_analysis_to_dataframe(analysis: dict[str, object]) -> pd.DataFrame:
    """Convert the business analysis dict to a two-column (metric, value) DataFrame."""
    rows = [{"metric": k, "value": str(v)} for k, v in analysis.items()]
    return pd.DataFrame(rows)


# ── Quarter snapshot backtest ─────────────────────────────────────────────────

def quarter_snapshot_backtest(
    panel: pd.DataFrame,
    model: Any,
    cutoff_date: str | pd.Timestamp | None = None,
    forward_weeks: int = 13,
    high_share: float = 0.20,
    fund_aum: float = 1_000_000_000.0,
) -> dict:
    """
    True out-of-sample quarter backtest.

    At ``cutoff_date`` (default: the date that leaves exactly ``forward_weeks``
    of history remaining), score all stocks using the already-trained model and
    assign risk buckets.  Then track the equal-weight strategy (excluding High-
    risk stocks) versus the equal-weight benchmark over the following
    ``forward_weeks`` weeks.

    This is NOT a new train/test split — it reuses the model as-is and simply
    isolates a single named quarter to produce a clear, story-driven result for
    academic and business-pitch purposes.

    Parameters
    ----------
    panel:        Full feature panel returned by build_feature_panel().
    model:        Trained sklearn pipeline with .feature_columns_ attribute.
    cutoff_date:  Snapshot date for scoring.  If None, auto-detects as the
                  last available date that still leaves forward_weeks of data.
    forward_weeks: Number of weekly periods to evaluate (default 13 ≈ one quarter).
    high_share:   Fraction of stocks to label High risk (default top 20%).
    fund_aum:     Hypothetical fund AUM in USD for dollar-impact calculations.

    Returns
    -------
    dict with keys:
        cutoff_date, quarter_label, forward_weeks,
        excluded_tickers  (list of dicts: ticker, crash_probability, quarter_return, outcome),
        weekly_series     (list of dicts: date, strategy_cumulative, benchmark_cumulative),
        strategy_quarter_return, benchmark_quarter_return,
        outperformance_bps, dollar_impact_quarter, dollar_impact_annualised,
        n_excluded, n_held, pct_excluded_correct
    """
    feature_columns = list(getattr(model, "feature_columns_", []))
    if not feature_columns:
        return {"error": "model must expose feature_columns_ for quarter_snapshot_backtest"}

    ph = panel.copy()
    ph["date"] = pd.to_datetime(ph["date"])
    ph = ph.sort_values(["ticker", "date"]).reset_index(drop=True)

    if "weekly_return" not in ph.columns:
        if "adj_close" not in ph.columns:
            return {"error": "panel must contain weekly_return or adj_close"}
        ph["weekly_return"] = ph.groupby("ticker", group_keys=False)["adj_close"].pct_change()

    sorted_dates = sorted(ph["date"].dropna().unique())
    if len(sorted_dates) < forward_weeks + 2:
        return {"error": "not enough dates in panel for quarter_snapshot_backtest"}

    # Resolve cutoff — default to leaving exactly forward_weeks future dates
    if cutoff_date is None:
        cutoff_ts = pd.Timestamp(sorted_dates[-(forward_weeks + 1)])
    else:
        cutoff_ts = pd.Timestamp(cutoff_date)
        # Snap to nearest available date at or before the requested cutoff
        available = [d for d in sorted_dates if pd.Timestamp(d) <= cutoff_ts]
        if not available:
            return {"error": f"cutoff_date {cutoff_date} is before any data"}
        cutoff_ts = pd.Timestamp(available[-1])

    forward_dates = [pd.Timestamp(d) for d in sorted_dates if pd.Timestamp(d) > cutoff_ts][:forward_weeks]
    if not forward_dates:
        return {"error": "no forward dates available after cutoff"}

    # Derive a quarter label from the first and last forward dates
    first_fwd, last_fwd = forward_dates[0], forward_dates[-1]
    quarter_num = (last_fwd.month - 1) // 3 + 1
    quarter_label = f"Q{quarter_num} {last_fwd.year}"

    # ── Score all stocks at the cutoff date ───────────────────────────────────
    cutoff_week = ph.loc[ph["date"] == cutoff_ts].copy()
    feat_missing = [c for c in feature_columns if c not in cutoff_week.columns]
    if feat_missing or cutoff_week.empty:
        return {"error": f"cannot score at cutoff {cutoff_ts}: missing features or no data"}

    valid_cutoff = cutoff_week.dropna(subset=feature_columns, how="all")
    if valid_cutoff.empty:
        return {"error": f"all stocks have NaN features at cutoff {cutoff_ts}"}

    positive_index = _positive_class_index(model)
    probabilities = pd.Series(
        model.predict_proba(valid_cutoff[feature_columns])[:, positive_index],
        index=valid_cutoff.index,
        name="crash_probability",
    )
    valid_cutoff = valid_cutoff.copy()
    valid_cutoff["crash_probability"] = probabilities
    valid_cutoff["risk_bucket"] = assign_risk_buckets(probabilities, high_share=high_share)

    ticker_bucket: dict[str, str] = valid_cutoff.set_index("ticker")["risk_bucket"].to_dict()
    ticker_prob: dict[str, float] = valid_cutoff.set_index("ticker")["crash_probability"].to_dict()

    high_tickers = {t for t, b in ticker_bucket.items() if b == "High"}

    # ── Compute per-ticker Q4 absolute return (price at end / price at cutoff - 1) ─
    ticker_q4_return: dict[str, float] = {}
    if "adj_close" in ph.columns:
        cutoff_prices = ph.loc[ph["date"] == cutoff_ts].set_index("ticker")["adj_close"].to_dict()
        end_prices = ph.loc[ph["date"] == last_fwd].set_index("ticker")["adj_close"].to_dict()
        for ticker in high_tickers:
            p0 = cutoff_prices.get(ticker)
            p1 = end_prices.get(ticker)
            if p0 and p1 and p0 != 0:
                ticker_q4_return[ticker] = float(p1 / p0 - 1)

    excluded_tickers = [
        {
            "ticker": t,
            "crash_probability": round(float(ticker_prob.get(t, float("nan"))), 4),
            "quarter_return": round(ticker_q4_return.get(t, float("nan")), 4)
            if t in ticker_q4_return else None,
            "outcome": "Avoided loss"
            if ticker_q4_return.get(t, 0) < 0
            else ("Model missed" if t in ticker_q4_return else "No price data"),
        }
        for t in sorted(high_tickers)
    ]
    excluded_tickers.sort(key=lambda x: -(ticker_prob.get(x["ticker"], 0)))

    # ── Build weekly cumulative return series for the forward window ───────────
    forward_data = ph.loc[ph["date"].isin(forward_dates)].copy()
    forward_data["risk_bucket"] = forward_data["ticker"].map(ticker_bucket).fillna("Low")

    weekly_series: list[dict] = []
    strat_cum = 1.0
    bench_cum = 1.0
    for date in forward_dates:
        week = forward_data.loc[forward_data["date"] == date].dropna(subset=["weekly_return"])
        if week.empty:
            continue
        bench_ret = float(week["weekly_return"].mean())
        strat_group = week.loc[week["risk_bucket"] != "High"]
        strat_ret = float(strat_group["weekly_return"].mean()) if not strat_group.empty else bench_ret
        strat_cum *= 1.0 + strat_ret
        bench_cum *= 1.0 + bench_ret
        weekly_series.append(
            {
                "date": date.strftime("%Y-%m-%d"),
                "strategy_cumulative": round(strat_cum, 6),
                "benchmark_cumulative": round(bench_cum, 6),
            }
        )

    if not weekly_series:
        return {"error": "no weekly returns available in forward window"}

    strategy_q4 = float(strat_cum - 1.0)
    benchmark_q4 = float(bench_cum - 1.0)
    outperformance = strategy_q4 - benchmark_q4
    outperformance_bps = int(round(outperformance * 10_000))
    dollar_impact_quarter = float(fund_aum * outperformance)
    dollar_impact_annualised = float(fund_aum * outperformance * 4)

    correct = sum(1 for t in excluded_tickers if t["quarter_return"] is not None and t["quarter_return"] < 0)
    total_with_data = sum(1 for t in excluded_tickers if t["quarter_return"] is not None)
    pct_correct = round(correct / total_with_data, 4) if total_with_data > 0 else None

    return {
        "cutoff_date": cutoff_ts.strftime("%Y-%m-%d"),
        "quarter_label": quarter_label,
        "forward_weeks": len(weekly_series),
        "excluded_tickers": excluded_tickers,
        "weekly_series": weekly_series,
        "strategy_quarter_return": round(strategy_q4, 4),
        "benchmark_quarter_return": round(benchmark_q4, 4),
        "outperformance_bps": outperformance_bps,
        "dollar_impact_quarter": round(dollar_impact_quarter, 0),
        "dollar_impact_annualised": round(dollar_impact_annualised, 0),
        "n_excluded": len(high_tickers),
        "n_held": len(ticker_bucket) - len(high_tickers),
        "pct_excluded_correct": pct_correct,
    }
