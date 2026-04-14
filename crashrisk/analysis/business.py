from __future__ import annotations

import numpy as np
import pandas as pd


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


# ── Main public function ──────────────────────────────────────────────────────

def compute_business_analysis(
    price_history: pd.DataFrame,
    scores: pd.DataFrame,
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
      strategy_sharpe, benchmark_sharpe, strategy_sortino
      max_drawdown_strategy, max_drawdown_benchmark, drawdown_improvement
      var_95_weekly, cvar_95_weekly
      evaluation_weeks, high_risk_excluded_pct
      fund_aum, economic_gain_annual, team_annual_cost, team_roi, justifies_team
    """
    if price_history.empty or scores.empty:
        return {"error": "insufficient data for business analysis"}

    weekly_rf = annual_risk_free / 52.0
    perf = _build_weekly_portfolio_returns(price_history, scores)

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

    mdd_strategy = _max_drawdown(strategy_rets)
    mdd_benchmark = _max_drawdown(benchmark_rets)

    var_95, cvar_95 = _var_cvar(strategy_rets)

    economic_gain = fund_aum * alpha
    team_roi = economic_gain / team_annual_cost if team_annual_cost > 0 else float("nan")

    ticker_buckets = scores["risk_bucket"].tolist()
    high_pct = sum(1 for b in ticker_buckets if b == "High") / max(1, len(ticker_buckets))

    def _fmt(v: float, digits: int = 4) -> float | None:
        return round(v, digits) if np.isfinite(v) else None

    return {
        "strategy_annual_return": _fmt(annual_strategy),
        "benchmark_annual_return": _fmt(annual_benchmark),
        "alpha_annualized": _fmt(alpha),
        "strategy_sharpe": _fmt(strategy_sharpe, 3),
        "benchmark_sharpe": _fmt(benchmark_sharpe, 3),
        "strategy_sortino": _fmt(strategy_sortino, 3),
        "max_drawdown_strategy": _fmt(mdd_strategy),
        "max_drawdown_benchmark": _fmt(mdd_benchmark),
        "drawdown_improvement": _fmt(mdd_benchmark - mdd_strategy),
        "var_95_weekly": _fmt(var_95),
        "cvar_95_weekly": _fmt(cvar_95),
        "evaluation_weeks": int(len(strategy_rets)),
        "high_risk_excluded_pct": _fmt(high_pct, 3),
        "fund_aum": fund_aum,
        "economic_gain_annual": round(economic_gain, 0) if np.isfinite(economic_gain) else None,
        "team_annual_cost": team_annual_cost,
        "team_roi": _fmt(team_roi, 2),
        "justifies_team": bool(np.isfinite(economic_gain) and economic_gain > team_annual_cost),
        "business_analysis_note": (
            "Illustrative gross overlay result annualized from the evaluation window; "
            "excludes transaction costs, capacity limits, and live implementation slippage."
        ),
    }


def business_analysis_to_dataframe(analysis: dict[str, object]) -> pd.DataFrame:
    """Convert the business analysis dict to a two-column (metric, value) DataFrame."""
    rows = [{"metric": k, "value": str(v)} for k, v in analysis.items()]
    return pd.DataFrame(rows)
