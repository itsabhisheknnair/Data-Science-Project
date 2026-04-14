/* ── Demo fallback data ────────────────────────────────────────────────── */

const DEMO_ROWS = [
  { ticker: "CYRX", as_of_date: "2024-12-27", crash_probability: "0.84", risk_bucket: "High",   top_drivers: "controversy_spike_flag;downside_beta;detrended_turnover" },
  { ticker: "FINX", as_of_date: "2024-12-27", crash_probability: "0.76", risk_bucket: "High",   top_drivers: "controversy_change_13w;lagged_ncskew;leverage" },
  { ticker: "HVST", as_of_date: "2024-12-27", crash_probability: "0.68", risk_bucket: "High",   top_drivers: "controversy_rolling_mean_13w;realized_volatility;beta" },
  { ticker: "JETT", as_of_date: "2024-12-27", crash_probability: "0.55", risk_bucket: "Medium", top_drivers: "lagged_duvol;trailing_return;controversy_sector_percentile" },
  { ticker: "ECOR", as_of_date: "2024-12-27", crash_probability: "0.48", risk_bucket: "Medium", top_drivers: "market_to_book;detrended_turnover;controversy_change_4w" },
  { ticker: "GLOB", as_of_date: "2024-12-27", crash_probability: "0.41", risk_bucket: "Medium", top_drivers: "realized_volatility;trailing_return;roa" },
  { ticker: "ALPH", as_of_date: "2024-12-27", crash_probability: "0.30", risk_bucket: "Low",    top_drivers: "market_cap;roa;controversy_score" },
  { ticker: "BRCK", as_of_date: "2024-12-27", crash_probability: "0.22", risk_bucket: "Low",    top_drivers: "beta;leverage;market_to_book" },
  { ticker: "KNXT", as_of_date: "2024-12-27", crash_probability: "0.19", risk_bucket: "Low",    top_drivers: "roa;trailing_return;controversy_rolling_std_13w" },
  { ticker: "DYNM", as_of_date: "2024-12-27", crash_probability: "0.14", risk_bucket: "Low",    top_drivers: "controversy_score;downside_beta;market_cap" },
  { ticker: "IMUN", as_of_date: "2024-12-27", crash_probability: "0.11", risk_bucket: "Low",    top_drivers: "realized_volatility;roa;leverage" },
  { ticker: "LUXE", as_of_date: "2024-12-27", crash_probability: "0.09", risk_bucket: "Low",    top_drivers: "market_to_book;trailing_return;beta" },
];

const DEMO_IMPORTANCE = [
  { feature: "controversy_spike_flag",         importance: "0.412" },
  { feature: "controversy_change_13w",         importance: "0.374" },
  { feature: "lagged_ncskew",                  importance: "0.341" },
  { feature: "downside_beta",                  importance: "0.298" },
  { feature: "controversy_rolling_mean_13w",   importance: "0.267" },
  { feature: "realized_volatility",            importance: "0.245" },
  { feature: "detrended_turnover",             importance: "0.218" },
  { feature: "leverage",                       importance: "0.196" },
  { feature: "controversy_sector_percentile",  importance: "0.178" },
  { feature: "beta",                           importance: "0.152" },
  { feature: "trailing_return",                importance: "0.140" },
  { feature: "market_to_book",                 importance: "0.132" },
];

const DEMO_ALGO_ROWS = [
  { model: "logistic_regression",  split: "test", best_cv_roc_auc: "0.662", roc_auc: "0.631", precision_at_top_bucket: "0.417", crash_capture_at_top_bucket: "0.500" },
  { model: "random_forest",        split: "test", best_cv_roc_auc: "0.711", roc_auc: "0.693", precision_at_top_bucket: "0.500", crash_capture_at_top_bucket: "0.583" },
  { model: "gradient_boosting",    split: "test", best_cv_roc_auc: "0.698", roc_auc: "0.672", precision_at_top_bucket: "0.500", crash_capture_at_top_bucket: "0.583" },
];

const DEMO_COMPARISON_ROWS = [
  { model: "baseline_no_esg",    split: "test", roc_auc: "0.582", precision_at_top_bucket: "0.417", crash_capture_at_top_bucket: "0.333" },
  { model: "full_with_esg",      split: "test", roc_auc: "0.631", precision_at_top_bucket: "0.500", crash_capture_at_top_bucket: "0.500" },
  { model: "full_minus_baseline",split: "test", roc_auc: "0.049", precision_at_top_bucket: "0.083", crash_capture_at_top_bucket: "0.167" },
];

const DEMO_BIZ_ROWS = [
  { metric: "strategy_annual_return",  value: "0.0821" },
  { metric: "benchmark_annual_return", value: "0.0643" },
  { metric: "alpha_annualized",        value: "0.0178" },
  { metric: "strategy_sharpe",         value: "0.812" },
  { metric: "benchmark_sharpe",        value: "0.614" },
  { metric: "strategy_sortino",        value: "1.204" },
  { metric: "max_drawdown_strategy",   value: "-0.1423" },
  { metric: "max_drawdown_benchmark",  value: "-0.1891" },
  { metric: "drawdown_improvement",    value: "0.0468" },
  { metric: "var_95_weekly",           value: "0.0284" },
  { metric: "cvar_95_weekly",          value: "0.0401" },
  { metric: "evaluation_weeks",        value: "104" },
  { metric: "high_risk_excluded_pct",  value: "0.25" },
  { metric: "fund_aum",                value: "1000000000" },
  { metric: "economic_gain_annual",    value: "17800000" },
  { metric: "team_annual_cost",        value: "800000" },
  { metric: "team_roi",                value: "22.25" },
  { metric: "justifies_team",          value: "True" },
];

const DEMO_PRICE_HISTORY = [
  ["CYRX","2024-06-28",44],["CYRX","2024-07-05",42],["CYRX","2024-07-12",45],["CYRX","2024-07-19",43],
  ["CYRX","2024-07-26",40],["CYRX","2024-08-02",38],["CYRX","2024-08-09",36],["CYRX","2024-08-16",33],
  ["CYRX","2024-08-23",31],["CYRX","2024-08-30",29],["CYRX","2024-09-06",27],["CYRX","2024-09-13",25],
  ["ALPH","2024-06-28",88],["ALPH","2024-07-05",91],["ALPH","2024-07-12",90],["ALPH","2024-07-19",93],
  ["ALPH","2024-07-26",95],["ALPH","2024-08-02",94],["ALPH","2024-08-09",97],["ALPH","2024-08-16",99],
  ["ALPH","2024-08-23",98],["ALPH","2024-08-30",102],["ALPH","2024-09-06",104],["ALPH","2024-09-13",103],
].map(([ticker, date, adj_close]) => ({ ticker, date, adj_close }));

const DEMO_PRICE_SCENARIOS = [
  { ticker: "CYRX", as_of_date: "2024-09-13", latest_price: 25, horizon_weeks: 13, price_p05: 14, price_p50: 25, price_p95: 31, crash_probability: 0.84, risk_bucket: "High",   scenario_method: "demo" },
  { ticker: "ALPH", as_of_date: "2024-09-13", latest_price: 103, horizon_weeks: 13, price_p05: 87, price_p50: 103, price_p95: 122, crash_probability: 0.30, risk_bucket: "Low", scenario_method: "demo" },
];

/* ── App state ─────────────────────────────────────────────────────────── */
let allRows        = [];
let priceHistory   = [];
let priceScenarios = [];
let comparisonRows = [];
let algoRows       = [];
let importanceRows = [];
let bizRows        = [];
let selectedTicker = "";

/* ── DOM refs ──────────────────────────────────────────────────────────── */
const uploadInput       = document.querySelector("#scoreUpload");
const historyUploadInput = document.querySelector("#historyUpload");
const apiUploadForm     = document.querySelector("#apiUploadForm");
const sampleGuideButton = document.querySelector("#sampleGuideButton");
const sampleGuideModal  = document.querySelector("#sampleGuideModal");
const sampleGuideClose  = document.querySelector("#sampleGuideClose");
const sectionInfoModal  = document.querySelector("#sectionInfoModal");
const sectionInfoClose  = document.querySelector("#sectionInfoClose");
const sectionInfoTitle  = document.querySelector("#sectionInfoTitle");
const sectionInfoBody   = document.querySelector("#sectionInfoBody");
const rawPrices         = document.querySelector("#rawPrices");
const rawBenchmark      = document.querySelector("#rawBenchmark");
const rawFundamentals   = document.querySelector("#rawFundamentals");
const rawControversies  = document.querySelector("#rawControversies");
const rawPricesStatus   = document.querySelector("#rawPricesStatus");
const rawBenchmarkStatus = document.querySelector("#rawBenchmarkStatus");
const rawFundamentalsStatus = document.querySelector("#rawFundamentalsStatus");
const rawControversiesStatus = document.querySelector("#rawControversiesStatus");
const apiTune           = document.querySelector("#apiTune");
const apiStatus         = document.querySelector("#apiStatus");
const runLiveScoreButton = document.querySelector("#runLiveScoreButton");
const bucketFilter      = document.querySelector("#bucketFilter");
const tickerSearch      = document.querySelector("#tickerSearch");
const dataStatus        = document.querySelector("#dataStatus");
const priceStatus       = document.querySelector("#priceStatus");
const chartStatus       = document.querySelector("#chartStatus");
const chartTitle        = document.querySelector("#chartTitle");
const resultCount       = document.querySelector("#resultCount");
const scoreRows         = document.querySelector("#scoreRows");
const riskBars          = document.querySelector("#riskBars");
const priceChart        = document.querySelector("#priceChart");
const comparisonStatus  = document.querySelector("#comparisonStatus");
const comparisonRowsBody = document.querySelector("#comparisonRows");
const comparisonSummary = document.querySelector("#comparisonSummary");
const algoStatus        = document.querySelector("#algoStatus");
const algoRowsBody      = document.querySelector("#algoRows");
const importanceStatus  = document.querySelector("#importanceStatus");
const importanceBars    = document.querySelector("#importanceBars");
const bizStatus         = document.querySelector("#bizStatus");
const bizSummary        = document.querySelector("#bizSummary");
const bizRowsBody       = document.querySelector("#bizRows");
const metricTotal       = document.querySelector("#metricTotal");
const metricHigh        = document.querySelector("#metricHigh");
const metricAverage     = document.querySelector("#metricAverage");
const metricDate        = document.querySelector("#metricDate");
const latestPrice       = document.querySelector("#latestPrice");
const priceP05          = document.querySelector("#priceP05");
const priceP50          = document.querySelector("#priceP50");
const priceP95          = document.querySelector("#priceP95");
const API_BASE_URL      = "https://crashrisk-api.onrender.com";

const SECTION_INFO = {
  "live-scoring": {
    title: "Live scoring",
    lead: "This is the main product workflow. Upload four raw Bloomberg-style exports and the hosted Python backend builds the feature panel, scores the crash-risk model, and returns dashboard-ready results.",
    points: [
      "prices gives stock price and volume history. The backend turns this into weekly returns, volatility, turnover, and crash-risk inputs.",
      "benchmark_prices gives the market reference series, such as S&P 500 or SPY. It is used for firm-specific residual returns and downside beta.",
      "fundamentals gives market cap, shares outstanding, leverage, market-to-book, and ROA. These controls help separate ESG controversy risk from ordinary firm characteristics.",
      "controversies is the ESG controversy signal. The model uses the score level, changes, spikes, rolling behavior, and sector-relative position.",
      "The API URL is fixed in code, so the user does not need to type the Render backend link."
    ],
    note: "Use this section for fresh predictions from raw data."
  },
  "advanced-uploads": {
    title: "Advanced saved CSV loading",
    lead: "This is a replay mode, not the normal prediction flow. It skips the backend and simply loads files that have already been produced by a previous backend run.",
    points: [
      "stock_scores.csv contains final scored rows: ticker, as-of date, crash probability, Low/Medium/High bucket, and top model drivers.",
      "price_history.csv contains historical adjusted close prices for the chart.",
      "price_scenarios.csv contains the 13-week p05, p50, and p95 scenario range used in the projection graph.",
      "This is useful for backup demos, offline testing, or checking a previous model run without uploading raw Bloomberg files again."
    ],
    note: "Live scoring generates new results; Advanced only displays existing results."
  },
  "risk-summary": {
    title: "Risk summary",
    lead: "These tiles give a quick read of the scored universe currently loaded on the dashboard.",
    points: [
      "Total names is the number of stocks currently scored.",
      "High risk counts how many names fall into the top-risk bucket.",
      "Average probability is the mean crash-risk probability across the loaded universe.",
      "As of is the latest scoring date in the loaded results."
    ],
    note: "Use this as the first sanity check after a live upload."
  },
  "top-names": {
    title: "Top names",
    lead: "This ranks stocks by model-implied crash probability so a portfolio manager can quickly see which names deserve attention first.",
    points: [
      "Higher bars mean higher predicted crash vulnerability.",
      "The bucket color shows whether each stock is Low, Medium, or High risk.",
      "This is a triage view, not a buy/sell recommendation."
    ],
    note: "The stock table below gives the same signal with dates and top drivers."
  },
  "data-contract": {
    title: "Data contract",
    lead: "This is the minimum schema the backend expects for the raw Bloomberg-style input files.",
    points: [
      "The backend validates these columns before scoring so missing fields fail clearly.",
      "Dates are aligned carefully to avoid look-ahead bias.",
      "Fundamentals are treated as available after a reporting lag, rather than as if they were known immediately on the period-end date.",
      "The controversy file should include sector so the backend can compute sector-relative controversy percentiles."
    ],
    note: "The Sample file guide shows small CSV examples of this schema."
  },
  "filters": {
    title: "Filters",
    lead: "These controls narrow what you see without changing the model output.",
    points: [
      "Risk bucket filters the table and top-name view to High, Medium, Low, or All names.",
      "Ticker search lets you jump to a specific stock.",
      "When a ticker is selected, the 13-week scenario chart updates to that ticker."
    ],
    note: "Filtering is display-only; it does not rerun the model."
  },
  "price-scenario": {
    title: "13-week scenario range",
    lead: "This graph is a scenario range, not a single price forecast. It shows recent price history and a volatility-based 13-week range adjusted for crash probability.",
    points: [
      "The historical line comes from adjusted close prices.",
      "p50 is set to the latest price so we do not pretend to have a precise return forecast.",
      "p05 is the downside scenario. It is widened when crash probability is higher.",
      "p95 is the upside scenario based on historical volatility.",
      "This is useful for discussing downside exposure in a fund demo, but it is not a guaranteed future price path."
    ],
    note: "Funds usually trust scenario ranges more than point price predictions."
  },
  "stock-table": {
    title: "Stock risk table",
    lead: "This is the main stock-level output of the crash-risk model.",
    points: [
      "crash_probability is the model's estimated probability that a stock belongs in the future high crash-risk bucket.",
      "risk_bucket converts the probability into Low, Medium, or High for easier business use.",
      "top_drivers lists the strongest standardized coefficient contributions for that stock.",
      "Clicking a row selects that ticker and updates the scenario chart."
    ],
    note: "This is the clearest section to show after live scoring finishes."
  },
  "feature-importance": {
    title: "Feature importance",
    lead: "This explains which inputs matter most in the fitted model overall.",
    points: [
      "For logistic regression, importance is based on standardized coefficient magnitude.",
      "Large values mean the feature has a stronger association with the model's crash-risk classification.",
      "This helps explain whether ESG controversy features, turnover, downside beta, volatility, and fundamentals are driving the signal.",
      "Feature importance is not causal proof; it is model interpretation."
    ],
    note: "Use this to explain why the model is flagging names."
  },
  "algorithm-comparison": {
    title: "Algorithm comparison",
    lead: "This compares candidate classifiers on the same chronological finance split.",
    points: [
      "ROC-AUC measures how well the model ranks future risky names above safer names.",
      "Precision at top bucket asks how accurate the top-risk flags are.",
      "Crash capture asks how many future high-risk names were caught in the top bucket.",
      "Chronological splits matter because random splits leak future market information into the past."
    ],
    note: "This section supports model selection, not individual stock decisions."
  },
  "esg-validation": {
    title: "ESG controversy validation",
    lead: "This is the key research question: does ESG controversy data improve crash-risk prediction beyond a traditional non-ESG benchmark?",
    points: [
      "The baseline model uses crash, turnover, downside-risk, volatility, return, and fundamental variables.",
      "The full model adds ESG controversy variables such as score level, changes, rolling behavior, spike flag, and sector percentile.",
      "The lift row is full model minus baseline on the same out-of-sample split.",
      "Positive lift in ROC-AUC, precision, or crash capture supports the claim that ESG controversy adds risk signal."
    ],
    note: "This is the section to cite when explaining ESG and risk econometrically."
  },
  "business-analysis": {
    title: "Business analysis",
    lead: "This translates the model into a simple fund-use case: avoid or review names flagged as High crash risk, then compare the resulting portfolio behavior.",
    points: [
      "The strategy is illustrative: equal-weight stocks not flagged as High risk.",
      "The benchmark is the equal-weighted full universe.",
      "Metrics such as alpha, Sharpe, Sortino, drawdown, VaR, and CVaR describe whether the overlay could add economic value.",
      "The AUM and team-cost assumptions turn model performance into a rough business case."
    ],
    note: "Treat this as a product-demo overlay until tested on real Bloomberg history."
  }
};

/* ── CSV parser ────────────────────────────────────────────────────────── */
function parseCsvRecords(text) {
  const rows = [];
  let current = "", row = [], inQuotes = false;
  for (let i = 0; i < text.length; i += 1) {
    const ch = text[i], nx = text[i + 1];
    if (ch === '"' && nx === '"') { current += '"'; i += 1; }
    else if (ch === '"') { inQuotes = !inQuotes; }
    else if (ch === ',' && !inQuotes) { row.push(current); current = ""; }
    else if ((ch === '\n' || ch === '\r') && !inQuotes) {
      if (ch === '\r' && nx === '\n') i += 1;
      row.push(current);
      if (row.some(c => c.trim())) rows.push(row);
      row = []; current = "";
    } else { current += ch; }
  }
  row.push(current);
  if (row.some(c => c.trim())) rows.push(row);
  if (rows.length < 2) return [];
  const headers = rows[0].map(h => h.trim());
  return rows.slice(1).map(vals => {
    const rec = {};
    headers.forEach((h, i) => { rec[h] = (vals[i] || "").trim(); });
    return rec;
  });
}

/* ── Normalisers ───────────────────────────────────────────────────────── */
function normalizeBucket(b) {
  const v = String(b || "").trim().toLowerCase();
  return v === "high" ? "High" : v === "medium" ? "Medium" : "Low";
}
function numberOrNull(v) { const n = Number(v); return Number.isFinite(n) ? n : null; }
function numberOrZero(v)  { const n = Number(v); return Number.isFinite(n) ? n : 0; }

function normalizeScore(row) {
  return {
    ticker:            String(row.ticker || "").trim().toUpperCase(),
    as_of_date:        String(row.as_of_date || "").trim(),
    crash_probability: numberOrZero(row.crash_probability),
    risk_bucket:       normalizeBucket(row.risk_bucket),
    top_drivers:       String(row.top_drivers || "").trim(),
  };
}
function normalizeHistory(row) {
  return {
    ticker:    String(row.ticker || "").trim().toUpperCase(),
    date:      String(row.date || "").trim(),
    adj_close: numberOrZero(row.adj_close),
  };
}
function normalizeScenario(row) {
  return {
    ticker:            String(row.ticker || "").trim().toUpperCase(),
    as_of_date:        String(row.as_of_date || "").trim(),
    latest_price:      numberOrZero(row.latest_price),
    horizon_weeks:     numberOrZero(row.horizon_weeks),
    price_p05:         numberOrZero(row.price_p05),
    price_p50:         numberOrZero(row.price_p50),
    price_p95:         numberOrZero(row.price_p95),
    crash_probability: numberOrZero(row.crash_probability),
    risk_bucket:       normalizeBucket(row.risk_bucket),
    scenario_method:   String(row.scenario_method || "").trim(),
  };
}
function normalizeComparison(row) {
  return {
    model:                        String(row.model || "").trim(),
    split:                        String(row.split || "").trim(),
    roc_auc:                      Number(row.roc_auc),
    precision_at_top_bucket:      Number(row.precision_at_top_bucket),
    crash_capture_at_top_bucket:  Number(row.crash_capture_at_top_bucket),
  };
}
function normalizeAlgo(row) {
  return {
    model:                       String(row.model || "").trim(),
    split:                       String(row.split || "").trim(),
    best_cv_roc_auc:             numberOrNull(row.best_cv_roc_auc),
    roc_auc:                     numberOrNull(row.roc_auc),
    precision_at_top_bucket:     numberOrNull(row.precision_at_top_bucket),
    crash_capture_at_top_bucket: numberOrNull(row.crash_capture_at_top_bucket),
  };
}
function normalizeImportance(row) {
  return { feature: String(row.feature || "").trim(), importance: numberOrZero(row.importance) };
}
function normalizeBiz(row) {
  return { metric: String(row.metric || "").trim(), value: String(row.value || "").trim() };
}

/* ── Formatters ────────────────────────────────────────────────────────── */
function formatPercent(v)  { return `${Math.round(v * 100)}%`; }
function formatPrice(v)    { return Number.isFinite(v) && v > 0 ? `$${v.toFixed(2)}` : "-"; }
function formatMetric(v)   { return Number.isFinite(v) ? v.toFixed(3) : "-"; }
function formatSignedPts(v){ return Number.isFinite(v) ? `${v > 0 ? "+" : ""}${Math.round(v * 100)} pts` : "-"; }
function bucketClass(b)    { return b.toLowerCase(); }

function escapeHtml(v) {
  return String(v).replaceAll("&","&amp;").replaceAll("<","&lt;").replaceAll(">","&gt;")
                  .replaceAll('"',"&quot;").replaceAll("'","&#039;");
}

function driverPills(drivers) {
  const names = drivers.split(";").map(d => d.trim()).filter(Boolean);
  if (!names.length) return '<span class="driver-pill">No driver data</span>';
  return names.map(d => `<span class="driver-pill">${escapeHtml(d)}</span>`).join("");
}

/* ── Filtering ─────────────────────────────────────────────────────────── */
function filteredRows() {
  const bucket = bucketFilter.value;
  const search = tickerSearch.value.trim().toUpperCase();
  return allRows.filter(r => {
    const bMatch = bucket === "All" || r.risk_bucket === bucket;
    const tMatch = !search || r.ticker.includes(search);
    return bMatch && tMatch;
  });
}

function defaultTicker(rows) {
  const [top] = [...rows].sort((a, b) => b.crash_probability - a.crash_probability);
  return top ? top.ticker : "";
}

function ensureSelectedTicker() {
  if (!selectedTicker || !allRows.some(r => r.ticker === selectedTicker)) {
    selectedTicker = defaultTicker(allRows);
  }
}

function selectTickerFromFilters({ preferSearch = false } = {}) {
  const rows = filteredRows();
  const search = tickerSearch.value.trim().toUpperCase();

  if (preferSearch && search) {
    const exact = rows.find(r => r.ticker === search) || allRows.find(r => r.ticker === search);
    if (exact) {
      selectedTicker = exact.ticker;
      return rows;
    }

    if (rows.length) {
      selectedTicker = defaultTicker(rows);
      return rows;
    }
  }

  if (rows.length && !rows.some(r => r.ticker === selectedTicker)) {
    selectedTicker = defaultTicker(rows);
  }

  if (!rows.length) {
    ensureSelectedTicker();
  }

  return rows;
}

/* ── Render orchestration ──────────────────────────────────────────────── */
function render() {
  ensureSelectedTicker();
  const rows = selectTickerFromFilters();
  renderMetrics(allRows);
  renderBars(rows);
  renderTable(rows);
  renderChart();
}

/* ── Metric KPIs ───────────────────────────────────────────────────────── */
function renderMetrics(rows) {
  const high = rows.filter(r => r.risk_bucket === "High").length;
  const avg  = rows.length ? rows.reduce((s, r) => s + r.crash_probability, 0) / rows.length : 0;
  const date = rows.map(r => r.as_of_date).filter(Boolean).sort().at(-1) || "-";
  metricTotal.textContent   = String(rows.length);
  metricHigh.textContent    = String(high);
  metricAverage.textContent = formatPercent(avg);
  metricDate.textContent    = date;
}

/* ── Top-names bar chart ───────────────────────────────────────────────── */
function renderBars(rows) {
  const top = [...rows].sort((a, b) => b.crash_probability - a.crash_probability).slice(0, 10);
  if (!top.length) { riskBars.innerHTML = '<div class="empty-state">No matching scores.</div>'; return; }
  riskBars.innerHTML = top.map(r => {
    const w = Math.max(3, Math.round(r.crash_probability * 100));
    return `
      <div class="bar-row">
        <div class="bar-label">${escapeHtml(r.ticker)}</div>
        <div class="bar-track" aria-label="${escapeHtml(r.ticker)} ${formatPercent(r.crash_probability)}">
          <div class="bar-fill ${bucketClass(r.risk_bucket)}" style="width:${w}%"></div>
        </div>
        <div class="bar-value">${formatPercent(r.crash_probability)}</div>
      </div>`;
  }).join("");
}

/* ── Score table ───────────────────────────────────────────────────────── */
function renderTable(rows) {
  resultCount.textContent = `${rows.length} row${rows.length === 1 ? "" : "s"}`;
  if (!rows.length) {
    scoreRows.innerHTML = '<tr><td colspan="5" class="empty-state">No matching scores.</td></tr>';
    return;
  }
  scoreRows.innerHTML = [...rows]
    .sort((a, b) => b.crash_probability - a.crash_probability)
    .map(r => {
      const sel = r.ticker === selectedTicker ? "selected-row" : "";
      return `
        <tr class="${sel}" data-ticker="${escapeHtml(r.ticker)}">
          <td class="ticker-cell">${escapeHtml(r.ticker)}</td>
          <td>${escapeHtml(r.as_of_date)}</td>
          <td class="probability-cell">${formatPercent(r.crash_probability)}</td>
          <td><span class="bucket ${bucketClass(r.risk_bucket)}">${escapeHtml(r.risk_bucket)}</span></td>
          <td><div class="driver-list">${driverPills(r.top_drivers)}</div></td>
        </tr>`;
    }).join("");
  scoreRows.querySelectorAll("tr[data-ticker]").forEach(row => {
    row.addEventListener("click", () => { selectedTicker = row.dataset.ticker; render(); });
  });
}

/* ── Price chart ───────────────────────────────────────────────────────── */
function renderChart() {
  const history  = priceHistory.filter(r => r.ticker === selectedTicker && r.adj_close > 0)
                               .sort((a, b) => a.date.localeCompare(b.date)).slice(-52);
  const scenario = priceScenarios.find(r => r.ticker === selectedTicker);
  const score    = allRows.find(r => r.ticker === selectedTicker);
  chartTitle.textContent = selectedTicker ? `${selectedTicker} price scenario` : "Select a ticker";
  if (!selectedTicker || history.length < 2 || !scenario) {
    chartStatus.textContent = "Upload raw files for a live score, or load price_history.csv and price_scenarios.csv.";
    priceChart.innerHTML = '<div class="empty-state">No price scenario for selected ticker.</div>';
    latestPrice.textContent = priceP05.textContent = priceP50.textContent = priceP95.textContent = "-";
    return;
  }
  const prob = score ? score.crash_probability : scenario.crash_probability;
  chartStatus.textContent = `Scenario range — not a point forecast. Crash probability: ${formatPercent(prob)}.`;
  latestPrice.textContent = formatPrice(scenario.latest_price);
  priceP05.textContent    = formatPrice(scenario.price_p05);
  priceP50.textContent    = formatPrice(scenario.price_p50);
  priceP95.textContent    = formatPrice(scenario.price_p95);
  priceChart.innerHTML    = buildPriceSvg(history, scenario);
}

function buildPriceSvg(history, scenario) {
  const W = 900, H = 310;
  const pad = { top: 22, right: 36, bottom: 42, left: 56 };
  const iW = W - pad.left - pad.right, iH = H - pad.top - pad.bottom;
  const futureIdx = history.length + 13;
  const allPrices = [...history.map(r => r.adj_close), scenario.price_p05, scenario.price_p95]
    .filter(v => Number.isFinite(v) && v > 0);
  const mn = Math.min(...allPrices), mx = Math.max(...allPrices);
  const rng = mx - mn || 1;
  const yMin = Math.max(0, mn - rng * 0.1), yMax = mx + rng * 0.1;

  const x = i => pad.left + (i / futureIdx) * iW;
  const y = p => pad.top  + ((yMax - p) / (yMax - yMin)) * iH;
  const pts = history.map((r, i) => `${x(i).toFixed(1)},${y(r.adj_close).toFixed(1)}`).join(" ");
  const li  = history.length - 1, lp = history[li].adj_close;
  const poly = [
    `${x(li).toFixed(1)},${y(lp).toFixed(1)}`,
    `${x(futureIdx).toFixed(1)},${y(scenario.price_p95).toFixed(1)}`,
    `${x(futureIdx).toFixed(1)},${y(scenario.price_p05).toFixed(1)}`,
  ].join(" ");
  const p50ln = `${x(li).toFixed(1)},${y(lp).toFixed(1)} ${x(futureIdx).toFixed(1)},${y(scenario.price_p50).toFixed(1)}`;
  const grid = [0, 0.25, 0.5, 0.75, 1].map(r => {
    const yp = pad.top + r * iH, lv = yMax - r * (yMax - yMin);
    return `<line class="chart-grid" x1="${pad.left}" y1="${yp}" x2="${W-pad.right}" y2="${yp}"></line>
            <text class="chart-label" x="10" y="${yp + 4}">${formatPrice(lv)}</text>`;
  });
  return `
    <svg viewBox="0 0 ${W} ${H}" role="img" aria-label="${escapeHtml(selectedTicker)} price scenario">
      ${grid.join("")}
      <line class="chart-axis" x1="${pad.left}" y1="${H-pad.bottom}" x2="${W-pad.right}" y2="${H-pad.bottom}"></line>
      <line class="chart-axis" x1="${pad.left}" y1="${pad.top}" x2="${pad.left}" y2="${H-pad.bottom}"></line>
      <polygon class="scenario-band" points="${poly}"></polygon>
      <polyline class="projection-line" points="${p50ln}"></polyline>
      <polyline class="history-line" points="${pts}"></polyline>
      <circle cx="${x(li)}" cy="${y(lp)}" r="4" fill="#0d6b62"></circle>
      <circle cx="${x(futureIdx)}" cy="${y(scenario.price_p05)}" r="4" fill="#c0332e"></circle>
      <circle cx="${x(futureIdx)}" cy="${y(scenario.price_p50)}" r="4" fill="#b07d00"></circle>
      <circle cx="${x(futureIdx)}" cy="${y(scenario.price_p95)}" r="4" fill="#1a7f68"></circle>
      <text class="chart-label" x="${pad.left}" y="${H-12}">${escapeHtml(history[0].date)}</text>
      <text class="chart-label" x="${(x(li)-28).toFixed(0)}" y="${H-12}">${escapeHtml(history[li].date)}</text>
      <text class="chart-label" x="${(x(futureIdx)-78).toFixed(0)}" y="${H-12}">+${scenario.horizon_weeks}w</text>
    </svg>`;
}

/* ── Feature importance ────────────────────────────────────────────────── */
function renderImportance() {
  if (!importanceRows.length) {
    importanceBars.innerHTML = '<div class="empty-state">No feature importance loaded.</div>';
    return;
  }
  const sorted = [...importanceRows].sort((a, b) => b.importance - a.importance).slice(0, 15);
  const maxImp = sorted[0].importance || 1;
  importanceBars.innerHTML = sorted.map(r => {
    const pct = Math.max(2, Math.round((r.importance / maxImp) * 100));
    const label = r.feature.replaceAll("_", " ");
    return `
      <div class="imp-row">
        <div class="imp-label" title="${escapeHtml(r.feature)}">${escapeHtml(label)}</div>
        <div class="imp-track">
          <div class="imp-fill" style="width:${pct}%"></div>
        </div>
        <div class="imp-value">${r.importance.toFixed(3)}</div>
      </div>`;
  }).join("");
}

/* ── Algorithm comparison ──────────────────────────────────────────────── */
const ALGO_LABELS = {
  logistic_regression: ["Logistic Regression", "L2-penalised, class-balanced"],
  random_forest:       ["Random Forest",        "Balanced class weights, GridSearchCV-tuned"],
  gradient_boosting:   ["Gradient Boosting",    "Shrinkage + depth search, GridSearchCV-tuned"],
};

function renderAlgoComparison() {
  if (!algoRows.length) {
    algoRowsBody.innerHTML = '<tr><td colspan="6" class="empty-state">No algorithm comparison loaded.</td></tr>';
    return;
  }
  // Find best test ROC-AUC to highlight the winner
  const testRows = algoRows.filter(r => r.split === "test");
  const bestAuc  = Math.max(...testRows.map(r => r.roc_auc || 0));

  algoRowsBody.innerHTML = algoRows.map(r => {
    const [title, sub] = ALGO_LABELS[r.model] || [r.model, ""];
    const isBest = r.split === "test" && r.roc_auc === bestAuc;
    return `
      <tr class="${isBest ? "best-row" : ""}">
        <td class="algo-model">${escapeHtml(title)}<br><small style="color:var(--muted);font-weight:400">${escapeHtml(sub)}</small></td>
        <td><span class="split-pill">${escapeHtml(r.split)}</span></td>
        <td class="metric-value">${r.best_cv_roc_auc != null ? r.best_cv_roc_auc.toFixed(3) : "-"}</td>
        <td class="metric-value">${r.roc_auc != null ? r.roc_auc.toFixed(3) : "-"}${isBest ? " ★" : ""}</td>
        <td class="metric-value">${r.precision_at_top_bucket != null ? r.precision_at_top_bucket.toFixed(3) : "-"}</td>
        <td class="metric-value">${r.crash_capture_at_top_bucket != null ? r.crash_capture_at_top_bucket.toFixed(3) : "-"}</td>
      </tr>`;
  }).join("");
}

/* ── ESG lift comparison ───────────────────────────────────────────────── */
function renderComparison() {
  if (!comparisonRows.length) {
    comparisonSummary.innerHTML = '<div class="empty-state">No ESG comparison loaded.</div>';
    comparisonRowsBody.innerHTML = '<tr><td colspan="6" class="empty-state">No ESG comparison loaded.</td></tr>';
    return;
  }
  renderComparisonSummary();
  comparisonRowsBody.innerHTML = comparisonRows.map(r => `
    <tr>
      <td>${modelLabel(r.model)}</td>
      <td><span class="split-pill">${escapeHtml(r.split)}</span></td>
      <td class="metric-value ${metricClass(r)}">${formatMetric(r.roc_auc)}</td>
      <td class="metric-value ${metricClass(r)}">${formatMetric(r.precision_at_top_bucket)}</td>
      <td class="metric-value ${metricClass(r)}">${formatMetric(r.crash_capture_at_top_bucket)}</td>
      <td class="plain-read">${plainEnglishRead(r)}</td>
    </tr>`).join("");
}

function renderComparisonSummary() {
  const delta = comparisonRows.find(r => r.model === "full_minus_baseline" && r.split === "test")
             || comparisonRows.find(r => r.model === "full_minus_baseline");
  if (!delta) { comparisonSummary.innerHTML = '<div class="empty-state">No delta row found.</div>'; return; }
  comparisonSummary.innerHTML = [
    summaryCard("ROC-AUC lift",     delta.roc_auc,                      "Ranking improvement from ESG controversy."),
    summaryCard("Precision lift",   delta.precision_at_top_bucket,      "Change in accuracy for top-risk flags."),
    summaryCard("Crash capture lift", delta.crash_capture_at_top_bucket, "Change in future high-risk names caught."),
  ].join("");
}

function summaryCard(label, value, desc) {
  const cls = value >= 0 ? "positive" : "negative";
  return `<article class="lift-card ${cls}">
    <span>${escapeHtml(label)}</span>
    <strong>${formatSignedPts(value)}</strong>
    <p class="status-text">${escapeHtml(desc)}</p>
  </article>`;
}

function modelLabel(model) {
  const map = {
    baseline_no_esg:    ["Traditional risk model", "Crash, turnover, downside risk, fundamentals"],
    full_with_esg:      ["Risk model + ESG controversy", "Baseline plus 8 controversy features"],
    full_minus_baseline:["ESG added value",        "Full model minus traditional model"],
  };
  const [title, sub] = map[model] || [model, ""];
  return `<div class="model-label"><strong>${escapeHtml(title)}</strong><span>${escapeHtml(sub)}</span></div>`;
}

function plainEnglishRead(row) {
  if (row.model === "baseline_no_esg")    return "Baseline: crash-risk prediction without ESG controversy data.";
  if (row.model === "full_with_esg")      return "Full model: same framework with ESG controversy features added.";
  const lift = row.roc_auc + row.precision_at_top_bucket + row.crash_capture_at_top_bucket;
  if (lift > 0) return "Positive lift: ESG controversy adds out-of-sample signal.";
  if (lift < 0) return "Negative lift: ESG controversy does not improve this split.";
  return "Neutral: ESG controversy has no measurable lift here.";
}

function metricClass(row) {
  if (row.model !== "full_minus_baseline") return "";
  const total = row.roc_auc + row.precision_at_top_bucket + row.crash_capture_at_top_bucket;
  return total >= 0 ? "positive" : "negative";
}

/* ── Business analysis ─────────────────────────────────────────────────── */
const BIZ_LABELS = {
  strategy_annual_return:  "Strategy annual return",
  benchmark_annual_return: "Benchmark annual return",
  alpha_annualized:        "Alpha (annualised)",
  strategy_sharpe:         "Strategy Sharpe ratio",
  benchmark_sharpe:        "Benchmark Sharpe ratio",
  strategy_sortino:        "Strategy Sortino ratio",
  max_drawdown_strategy:   "Max drawdown — strategy",
  max_drawdown_benchmark:  "Max drawdown — benchmark",
  drawdown_improvement:    "Drawdown improvement",
  var_95_weekly:           "Weekly VaR (95%)",
  cvar_95_weekly:          "Weekly CVaR (95%)",
  evaluation_weeks:        "Evaluation weeks",
  high_risk_excluded_pct:  "High-risk excluded %",
  fund_aum:                "Fund AUM ($)",
  economic_gain_annual:    "Annual economic gain ($)",
  team_annual_cost:        "Team annual cost ($)",
  team_roi:                "Team ROI (×)",
  justifies_team:          "Justifies team hire?",
};

const BIZ_SUMMARY_KEYS = ["alpha_annualized", "strategy_sharpe", "economic_gain_annual"];

function renderBizAnalysis() {
  if (!bizRows.length) {
    bizSummary.innerHTML = '<div class="empty-state">No business analysis loaded.</div>';
    bizRowsBody.innerHTML = '<tr><td colspan="2" class="empty-state">No business analysis loaded.</td></tr>';
    return;
  }
  const lookup = Object.fromEntries(bizRows.map(r => [r.metric, r.value]));

  // Summary cards
  bizSummary.innerHTML = BIZ_SUMMARY_KEYS.map(key => {
    const raw   = lookup[key] ?? "-";
    const label = BIZ_LABELS[key] || key;
    let display = raw, cls = "neutral";
    if (key === "alpha_annualized") {
      const v = parseFloat(raw);
      display = Number.isFinite(v) ? `${(v * 100).toFixed(2)}%` : raw;
      cls = v >= 0 ? "positive" : "negative";
    } else if (key === "strategy_sharpe") {
      const v = parseFloat(raw);
      display = Number.isFinite(v) ? v.toFixed(2) : raw;
      cls = v >= 1 ? "positive" : v >= 0 ? "neutral" : "negative";
    } else if (key === "economic_gain_annual") {
      const v = parseFloat(raw);
      display = Number.isFinite(v) ? `$${(v / 1e6).toFixed(1)}M` : raw;
      cls = v > 0 ? "positive" : "negative";
    }
    return `<article class="biz-card ${cls}"><span>${escapeHtml(label)}</span><strong>${escapeHtml(display)}</strong></article>`;
  }).join("");

  // Full table
  bizRowsBody.innerHTML = bizRows.map(r => {
    const label = BIZ_LABELS[r.metric] || r.metric;
    let display = r.value;
    const v = parseFloat(r.value);
    if (["strategy_annual_return","benchmark_annual_return","alpha_annualized","high_risk_excluded_pct"].includes(r.metric))
      display = Number.isFinite(v) ? `${(v*100).toFixed(2)}%` : r.value;
    else if (["fund_aum","economic_gain_annual","team_annual_cost"].includes(r.metric))
      display = Number.isFinite(v) ? `$${v.toLocaleString()}` : r.value;
    else if (r.metric === "team_roi" && Number.isFinite(v))
      display = `${v.toFixed(2)}×`;
    return `<tr><td>${escapeHtml(label)}</td><td style="font-variant-numeric:tabular-nums">${escapeHtml(display)}</td></tr>`;
  }).join("");
}

/* ── Setters (called after load) ───────────────────────────────────────── */
function setRows(rows, msg) {
  allRows = rows.map(normalizeScore).filter(r => r.ticker);
  selectedTicker = defaultTicker(allRows);
  dataStatus.textContent = msg;
  render();
}

function setPriceData(historyRows, scenarioRows, msg) {
  if (historyRows)  priceHistory   = historyRows.map(normalizeHistory).filter(r => r.ticker && r.adj_close > 0);
  if (scenarioRows) priceScenarios = scenarioRows.map(normalizeScenario).filter(r => r.ticker && r.latest_price > 0);
  priceStatus.textContent = msg;
  render();
}

function setComparisonRows(rows, msg) {
  comparisonRows = rows.map(normalizeComparison).filter(r => r.model && r.split);
  comparisonStatus.textContent = msg;
  renderComparison();
}

function setAlgoRows(rows, msg) {
  algoRows = rows.map(normalizeAlgo).filter(r => r.model && r.split);
  algoStatus.textContent = msg;
  renderAlgoComparison();
}

function setImportanceRows(rows, msg) {
  importanceRows = rows.map(normalizeImportance).filter(r => r.feature);
  importanceStatus.textContent = msg;
  renderImportance();
}

function setBizRows(rows, msg) {
  bizRows = rows.map(normalizeBiz).filter(r => r.metric);
  bizStatus.textContent = msg;
  renderBizAnalysis();
}

/* ── File I/O ──────────────────────────────────────────────────────────── */
function selectedFile(input, label) {
  const [file] = Array.from(input.files || []);
  if (!file) throw new Error(`Choose ${label}.`);
  return file;
}

function fileListLabel(input) {
  const files = Array.from(input.files || []);
  if (!files.length) return "No file selected";
  if (files.length === 1) return files[0].name;
  return `${files.length} files selected`;
}

function updateFileStatus(input, statusEl) {
  if (input && statusEl) statusEl.textContent = fileListLabel(input);
}

function setLiveScoreBusy(isBusy) {
  runLiveScoreButton.disabled = isBusy;
  runLiveScoreButton.textContent = isBusy ? "Scoring..." : "Run live score";
  apiUploadForm.setAttribute("aria-busy", String(isBusy));
}

function apiEndpoint() {
  return `${API_BASE_URL}/predict`;
}

function openSampleGuide() {
  if (typeof sampleGuideModal.showModal === "function") sampleGuideModal.showModal();
  else sampleGuideModal.setAttribute("open", "");
}

function closeSampleGuide() {
  if (typeof sampleGuideModal.close === "function") sampleGuideModal.close();
  else sampleGuideModal.removeAttribute("open");
}

function renderSectionInfo(info) {
  sectionInfoTitle.textContent = info.title;
  sectionInfoBody.innerHTML = "";

  const lead = document.createElement("p");
  lead.className = "info-lead";
  lead.textContent = info.lead;
  sectionInfoBody.appendChild(lead);

  const list = document.createElement("ul");
  info.points.forEach(point => {
    const item = document.createElement("li");
    item.textContent = point;
    list.appendChild(item);
  });
  sectionInfoBody.appendChild(list);

  if (info.note) {
    const note = document.createElement("p");
    note.className = "info-note";
    note.textContent = info.note;
    sectionInfoBody.appendChild(note);
  }
}

function openSectionInfo(key) {
  const info = SECTION_INFO[key];
  if (!info) return;
  renderSectionInfo(info);
  if (typeof sectionInfoModal.showModal === "function") sectionInfoModal.showModal();
  else sectionInfoModal.setAttribute("open", "");
}

function closeSectionInfo() {
  if (typeof sectionInfoModal.close === "function") sectionInfoModal.close();
  else sectionInfoModal.removeAttribute("open");
}

function attachSectionInfoButtons() {
  document.querySelectorAll("[data-info-key]").forEach(section => {
    const key = section.dataset.infoKey;
    if (!SECTION_INFO[key]) return;

    const button = document.createElement("button");
    button.className = "section-info-button";
    button.type = "button";
    button.textContent = "i";
    button.setAttribute("aria-label", `Explain ${SECTION_INFO[key].title}`);
    button.addEventListener("click", event => {
      event.preventDefault();
      event.stopPropagation();
      openSectionInfo(key);
    });

    const target = Array.from(section.children).find(child =>
      child.classList?.contains("section-heading") || child.tagName === "SUMMARY"
    );

    if (target) target.appendChild(button);
    else {
      button.classList.add("floating");
      section.appendChild(button);
    }
  });
}

async function runLiveApiScore(event) {
  event.preventDefault();

  const form = new FormData();
  try {
    form.append("prices", selectedFile(rawPrices, "prices"));
    form.append("benchmark_prices", selectedFile(rawBenchmark, "benchmark_prices"));
    form.append("fundamentals", selectedFile(rawFundamentals, "fundamentals"));
    form.append("controversies", selectedFile(rawControversies, "controversies"));
    form.append("tune", apiTune.checked ? "true" : "false");
  } catch (err) {
    apiStatus.textContent = err.message;
    return;
  }

  setLiveScoreBusy(true);
  apiStatus.textContent = "Scoring raw files with the Python backend...";
  try {
    const response = await fetch(apiEndpoint(), { method: "POST", body: form });
    const payload = await response.json().catch(() => ({}));
    if (!response.ok) {
      const detail = Array.isArray(payload.detail) ? payload.detail.map(d => d.msg || d.detail || d).join("; ") : payload.detail;
      throw new Error(detail || `API request failed with status ${response.status}.`);
    }

    setRows(payload.scores || [], `Loaded ${payload.scores?.length || 0} scores from Render API.`);
    setPriceData(payload.price_history || [], payload.price_scenarios || [], "Loaded live price history and scenarios.");
    setComparisonRows(payload.model_comparison || [], "Loaded live ESG comparison.");
    setAlgoRows(payload.algorithm_comparison || [], "Loaded live algorithm comparison.");
    setImportanceRows(payload.feature_importance || [], "Loaded live feature importance.");
    setBizRows(payload.business_analysis || [], "Loaded live business analysis.");
    apiStatus.textContent = `Scored ${payload.metadata?.score_count || 0} rows from raw uploaded data.`;
  } catch (err) {
    apiStatus.textContent = `API scoring failed: ${err.message}`;
  } finally {
    setLiveScoreBusy(false);
  }
}

function loadUploadedFile(file) {
  const reader = new FileReader();
  reader.onload = () => {
    const rows = parseCsvRecords(String(reader.result || ""));
    setRows(rows, `Loaded ${rows.length} score rows from ${file.name}.`);
  };
  reader.onerror = () => setRows(DEMO_ROWS, "Could not read that file. Demo scores restored.");
  reader.readAsText(file);
}

async function loadUploadedPriceFiles(files) {
  let hist = null, scen = null;
  for (const file of files) {
    const rows = parseCsvRecords(await file.text());
    if (file.name.toLowerCase().includes("scenario")) scen = rows;
    else hist = rows;
  }
  const hc = hist ? hist.length : priceHistory.length;
  const sc = scen ? scen.length : priceScenarios.length;
  setPriceData(hist, scen, `Loaded ${hc} history and ${sc} scenario rows.`);
}

async function loadCsv(path) {
  const res = await fetch(path, { cache: "no-store" });
  if (!res.ok) throw new Error(`Could not load ${path}`);
  return parseCsvRecords(await res.text());
}

async function loadCsvFirst(paths) {
  let lastError = null;
  for (const path of paths) {
    try {
      return await loadCsv(path);
    } catch (err) {
      lastError = err;
    }
  }
  throw lastError || new Error("No CSV path supplied");
}

async function loadDefaultScores() {
  try {
    const rows = await loadCsvFirst(["outputs/stock_scores.csv", "../outputs/stock_scores.csv"]);
    if (!rows.length) throw new Error("empty");
    setRows(rows, `Loaded ${rows.length} scores from outputs/stock_scores.csv.`);
  } catch { setRows(DEMO_ROWS, "Demo scores are showing until you run a live upload."); }
}

async function loadDefaultPriceData() {
  try {
    const [h, s] = await Promise.all([
      loadCsvFirst(["outputs/price_history.csv", "../outputs/price_history.csv"]),
      loadCsvFirst(["outputs/price_scenarios.csv", "../outputs/price_scenarios.csv"]),
    ]);
    setPriceData(h, s, `Loaded ${h.length} history and ${s.length} scenario rows.`);
  } catch {
    setPriceData(DEMO_PRICE_HISTORY, DEMO_PRICE_SCENARIOS, "Demo price scenarios are showing until you run a live upload.");
  }
}

async function loadDefaultComparison() {
  try {
    const rows = await loadCsvFirst(["outputs/esg_model_comparison.csv", "../outputs/esg_model_comparison.csv"]);
    if (!rows.length) throw new Error("empty");
    setComparisonRows(rows, `Loaded ${rows.length} ESG comparison rows.`);
  } catch { setComparisonRows(DEMO_COMPARISON_ROWS, "Demo ESG lift is showing until you run a live upload."); }
}

async function loadDefaultAlgoComparison() {
  try {
    const rows = await loadCsvFirst(["outputs/algorithm_comparison.csv", "../outputs/algorithm_comparison.csv"]);
    if (!rows.length) throw new Error("empty");
    setAlgoRows(rows, `Loaded ${rows.length} algorithm comparison rows.`);
  } catch { setAlgoRows(DEMO_ALGO_ROWS, "Demo algorithm comparison is showing until you run a live upload."); }
}

async function loadDefaultImportance() {
  try {
    const rows = await loadCsvFirst(["outputs/feature_importance.csv", "../outputs/feature_importance.csv"]);
    if (!rows.length) throw new Error("empty");
    setImportanceRows(rows, `Loaded ${rows.length} feature importance rows.`);
  } catch { setImportanceRows(DEMO_IMPORTANCE, "Demo feature importance is showing until you run a live upload."); }
}

async function loadDefaultBizAnalysis() {
  try {
    const rows = await loadCsvFirst(["outputs/business_analysis.csv", "../outputs/business_analysis.csv"]);
    if (!rows.length) throw new Error("empty");
    setBizRows(rows, `Loaded ${rows.length} business analysis rows.`);
  } catch { setBizRows(DEMO_BIZ_ROWS, "Demo business analysis is showing until you run a live upload."); }
}

/* ── Event listeners ───────────────────────────────────────────────────── */
attachSectionInfoButtons();

uploadInput.addEventListener("change", e => {
  const [file] = e.target.files;
  if (file) loadUploadedFile(file);
});
historyUploadInput.addEventListener("change", e => {
  const files = Array.from(e.target.files || []);
  if (files.length) loadUploadedPriceFiles(files);
});
rawPrices.addEventListener("change", () => updateFileStatus(rawPrices, rawPricesStatus));
rawBenchmark.addEventListener("change", () => updateFileStatus(rawBenchmark, rawBenchmarkStatus));
rawFundamentals.addEventListener("change", () => updateFileStatus(rawFundamentals, rawFundamentalsStatus));
rawControversies.addEventListener("change", () => updateFileStatus(rawControversies, rawControversiesStatus));
apiUploadForm.addEventListener("submit", runLiveApiScore);
sampleGuideButton.addEventListener("click", openSampleGuide);
sampleGuideClose.addEventListener("click", closeSampleGuide);
sampleGuideModal.addEventListener("click", e => {
  if (e.target === sampleGuideModal) closeSampleGuide();
});
sectionInfoClose.addEventListener("click", closeSectionInfo);
sectionInfoModal.addEventListener("click", e => {
  if (e.target === sectionInfoModal) closeSectionInfo();
});
bucketFilter.addEventListener("change", () => {
  selectTickerFromFilters({ preferSearch: true });
  render();
});
tickerSearch.addEventListener("input", () => {
  selectTickerFromFilters({ preferSearch: true });
  render();
});

/* ── Boot ──────────────────────────────────────────────────────────────── */
loadDefaultScores();
loadDefaultPriceData();
loadDefaultComparison();
loadDefaultAlgoComparison();
loadDefaultImportance();
loadDefaultBizAnalysis();
