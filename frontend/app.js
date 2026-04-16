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

const DEMO_DATA_SUMMARY = [
  { section: "feature_panel",  metric: "rows",                   value: "—", detail: "Run the backend to see weekly feature rows." },
  { section: "feature_panel",  metric: "ticker_count",           value: "—", detail: "Run the backend to see ticker count." },
  { section: "model_dataset",  metric: "rows",                   value: "—", detail: "Run the backend to see labeled model rows." },
  { section: "configuration",  metric: "fundamentals_lag_days",  value: "45", detail: "Fundamentals become usable 45 calendar days after period_end." },
  { section: "configuration",  metric: "target_horizon_weeks",   value: "13", detail: "Crash-risk target uses a 13-week future window." },
];

const DEMO_CLEANING_LOG = [
  { dataset: "feature_engineering", check: "date_alignment_method", value: "weekly Friday observations", detail: "Daily raw inputs are aligned to Friday week-end observations before modeling." },
  { dataset: "target_creation",     check: "future_window",          value: "t+1 through t+13",           detail: "Targets use future weeks only; features at t use data available at or before t." },
];

const DEMO_SQL_SUMMARY = [
  { query_name: "demo_placeholder", query: "-- Run the backend locally to generate SQL evidence.", result_json: "[]", row_count: 0 },
];

const DEMO_TEXTUAL_ANALYSIS = [
  { status: "no_text_file", note: "Rubric evidence appears after a backend run. Supply controversy_text.csv or news_text.csv to enable direct headline analysis." },
];

const DEMO_TEXTUAL_TICKER_SUMMARY = [
  { status: "ok", ticker: "CYRX", latest_text_date: "2024-12-27", article_count: "7", negative_esg_controversy_score_0_100: "81.6", text_sentiment_score: "-0.1429", negative_word_count: "16", positive_word_count: "2", controversy_keyword_count: "11", score_band: "High" },
  { status: "ok", ticker: "FINX", latest_text_date: "2024-12-27", article_count: "5", negative_esg_controversy_score_0_100: "68.4", text_sentiment_score: "-0.1180", negative_word_count: "11", positive_word_count: "1", controversy_keyword_count: "8", score_band: "High" },
  { status: "ok", ticker: "HVST", latest_text_date: "2024-12-27", article_count: "4", negative_esg_controversy_score_0_100: "52.1", text_sentiment_score: "-0.0710", negative_word_count: "7", positive_word_count: "1", controversy_keyword_count: "5", score_band: "Medium" },
];

const DEMO_QUARTER_BT = {
  cutoff_date: "2024-06-28",
  quarter_label: "Q3 2024",
  forward_weeks: 12,
  strategy_quarter_return: 0.0489,
  benchmark_quarter_return: 0.0263,
  outperformance_bps: 226,
  dollar_impact_quarter: 2260000,
  dollar_impact_annualised: 9040000,
  n_excluded: 3,
  n_held: 9,
  pct_excluded_correct: 0.67,
  excluded_tickers: [
    { ticker: "CYRX", crash_probability: 0.8412, quarter_return: -0.3182, outcome: "Avoided loss" },
    { ticker: "FINX", crash_probability: 0.7634, quarter_return: -0.1547, outcome: "Avoided loss" },
    { ticker: "HVST", crash_probability: 0.6821, quarter_return:  0.0831, outcome: "Model missed" },
  ],
  weekly_series: [
    { date: "2024-07-05", strategy_cumulative: 1.0041, benchmark_cumulative: 1.0018 },
    { date: "2024-07-12", strategy_cumulative: 1.0097, benchmark_cumulative: 1.0052 },
    { date: "2024-07-19", strategy_cumulative: 1.0154, benchmark_cumulative: 1.0089 },
    { date: "2024-07-26", strategy_cumulative: 1.0198, benchmark_cumulative: 1.0107 },
    { date: "2024-08-02", strategy_cumulative: 1.0241, benchmark_cumulative: 1.0134 },
    { date: "2024-08-09", strategy_cumulative: 1.0287, benchmark_cumulative: 1.0158 },
    { date: "2024-08-16", strategy_cumulative: 1.0321, benchmark_cumulative: 1.0172 },
    { date: "2024-08-23", strategy_cumulative: 1.0369, benchmark_cumulative: 1.0192 },
    { date: "2024-08-30", strategy_cumulative: 1.0412, benchmark_cumulative: 1.0213 },
    { date: "2024-09-06", strategy_cumulative: 1.0451, benchmark_cumulative: 1.0234 },
    { date: "2024-09-13", strategy_cumulative: 1.0479, benchmark_cumulative: 1.0249 },
    { date: "2024-09-20", strategy_cumulative: 1.0489, benchmark_cumulative: 1.0263 },
  ],
};

const DEMO_TICKERS = new Set(DEMO_ROWS.map(row => row.ticker));

/* ── App state ─────────────────────────────────────────────────────────── */
let allRows        = [];
let priceHistory   = [];
let priceScenarios = [];
let comparisonRows = [];
let algoRows       = [];
let importanceRows = [];
let bizRows             = [];
let dataSummaryRows     = [];
let cleaningLogRows     = [];
let sqlSummaryRows      = [];
let textualAnalysisRows = [];
let textualTickerSummaryRows = [];
let textModelRows       = [];
let tuningRows          = [];
let confusionRows       = [];
let calibrationRows     = [];
let featureStatsRows    = [];
let textCoverageRows    = [];
let ldaTopicRows        = [];
let ldaTickerTopicRows  = [];
let quarterBt           = null;
let selectedTicker      = "";
let liveProgressTimer   = null;
let liveProgressPercent = 0;

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
const rawNewsText       = document.querySelector("#rawNewsText");
const rawControversyText = document.querySelector("#rawControversyText");
const rawPricesStatus   = document.querySelector("#rawPricesStatus");
const rawBenchmarkStatus = document.querySelector("#rawBenchmarkStatus");
const rawFundamentalsStatus = document.querySelector("#rawFundamentalsStatus");
const rawControversiesStatus = document.querySelector("#rawControversiesStatus");
const rawNewsTextStatus = document.querySelector("#rawNewsTextStatus");
const rawControversyTextStatus = document.querySelector("#rawControversyTextStatus");
const apiTune           = document.querySelector("#apiTune");
const apiStatus         = document.querySelector("#apiStatus");
const runLiveScoreButton = document.querySelector("#runLiveScoreButton");
const liveScoreProgress = document.querySelector("#liveScoreProgress");
const liveProgressLabel = document.querySelector("#liveProgressLabel");
const liveProgressValue = document.querySelector("#liveProgressValue");
const liveProgressTrack = document.querySelector("#liveProgressTrack");
const liveProgressBar   = document.querySelector("#liveProgressBar");
const liveProgressHint  = document.querySelector("#liveProgressHint");
const bucketFilter      = document.querySelector("#bucketFilter");
const tickerSearch      = document.querySelector("#tickerSearch");
const tickerOptions     = document.querySelector("#tickerOptions");
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
const bizHighlights     = document.querySelector("#bizHighlights");
const bizRowsBody       = document.querySelector("#bizRows");
const quarterBtSection  = document.querySelector("#quarterBacktestSection");
const quarterBtStatus   = document.querySelector("#quarterBtStatus");
const quarterBtHero     = document.querySelector("#quarterBtHero");
const quarterReturnChartEl   = document.querySelector("#quarterReturnChart");
const quarterExcludedBarChartEl = document.querySelector("#quarterExcludedBarChart");
const quarterExcludedTable   = document.querySelector("#quarterExcludedTable tbody");
const metricTotal       = document.querySelector("#metricTotal");
const metricHigh        = document.querySelector("#metricHigh");
const metricAverage     = document.querySelector("#metricAverage");
const metricDate        = document.querySelector("#metricDate");
const latestPrice            = document.querySelector("#latestPrice");
const priceP05               = document.querySelector("#priceP05");
const priceP50               = document.querySelector("#priceP50");
const priceP95               = document.querySelector("#priceP95");
const evidenceStatus         = document.querySelector("#evidenceStatus");
const evidenceHeadline       = document.querySelector("#evidenceHeadline");
const dataSummaryContent     = document.querySelector("#dataSummaryContent");
const cleaningLogContent     = document.querySelector("#cleaningLogContent");
const sqlEvidenceContent     = document.querySelector("#sqlEvidenceContent");
const textualAnalysisContent = document.querySelector("#textualAnalysisContent");
const modelDiagnosticsStatus = document.querySelector("#modelDiagnosticsStatus");
const tuningRowsBody         = document.querySelector("#tuningRows");
const textModelRowsBody      = document.querySelector("#textModelRows");
const confusionRowsBody      = document.querySelector("#confusionRows");
const calibrationRowsBody    = document.querySelector("#calibrationRows");
const tuningBadge            = document.querySelector("#tuningBadge");
const textModelBadge         = document.querySelector("#textModelBadge");
const confusionBadge         = document.querySelector("#confusionBadge");
const calibrationBadge       = document.querySelector("#calibrationBadge");
const textAnalyticsStatus    = document.querySelector("#textAnalyticsStatus");
const textAnalyticsSummary   = document.querySelector("#textAnalyticsSummary");
const textAnalyticsBadge     = document.querySelector("#textAnalyticsBadge");
const textTickerSummaryContent = document.querySelector("#textTickerSummaryContent");
const textWordCloudContent   = document.querySelector("#textWordCloudContent");
const textWordCloudStatus    = document.querySelector("#textWordCloudStatus");
const dataSummaryBadge       = document.querySelector("#dataSummaryBadge");
const cleaningLogBadge       = document.querySelector("#cleaningLogBadge");
const sqlEvidenceBadge       = document.querySelector("#sqlEvidenceBadge");
const textualAnalysisBadge   = document.querySelector("#textualAnalysisBadge");
const featureStatsContent    = document.querySelector("#featureStatsContent");
const featureStatsBadge      = document.querySelector("#featureStatsBadge");
const textCoverageContent    = document.querySelector("#textCoverageContent");
const textCoverageBadge      = document.querySelector("#textCoverageBadge");
const reportFigures          = document.querySelector("#reportFigures");
const reportDownloads        = document.querySelector("#reportDownloads");
const DEFAULT_API_BASE_URL   = "https://crashrisk-api.onrender.com";
const isLocalStaticHost      = ["", "localhost", "127.0.0.1"].includes(window.location.hostname);
const API_BASE_URL           = (
  window.CRASHRISK_API_BASE_URL ||
  (window.location.protocol.startsWith("http") && !isLocalStaticHost ? "/api" : DEFAULT_API_BASE_URL)
).replace(/\/$/, "");

const SECTION_INFO = {
  "live-scoring": {
    title: "Live scoring",
    lead: "This is the main product workflow. Upload four raw Bloomberg-style exports and the hosted Python backend builds the feature panel, scores the crash-risk model, and returns dashboard-ready results.",
    points: [
      "prices gives stock price and volume history. The backend turns this into weekly returns, volatility, turnover, and crash-risk inputs.",
      "benchmark_prices gives the market reference series, such as S&P 500 or SPY. It is used for firm-specific residual returns and downside beta.",
      "fundamentals gives market cap, shares outstanding, leverage, market-to-book, and ROA. These controls help separate ESG controversy risk from ordinary firm characteristics.",
      "controversies is the ESG controversy signal. The model uses the score level, changes, spikes, rolling behavior, and sector-relative position.",
      "Optional news_text and controversy_text files power the separate ESG news monitor and do not change the crash-risk model score in this version.",
      "On Netlify, the dashboard calls /api/predict and Netlify proxies the request to the hosted FastAPI backend."
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
    lead: "This explains which inputs matter most in the fitted model overall. The displayed values are normalized percentage shares that add to 100%.",
    points: [
      "For logistic regression, the raw signal starts from standardized coefficient magnitude.",
      "The frontend converts raw importance values into percentage shares so the chart is easier to read.",
      "Large percentages mean the feature has a stronger association with the model's crash-risk classification.",
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
  "model-diagnostics": {
    title: "Model diagnostics",
    lead: "This mirrors the latest backend outputs for model tuning, textual-signal testing, confusion-matrix behavior, and probability calibration.",
    points: [
      "Hyperparameter tuning reports the grid searched and the best TimeSeriesSplit CV result for each model family.",
      "Text signal comparison checks whether the text-derived ESG sentiment features change validation or test metrics.",
      "The confusion matrix gives TP, FP, TN, and FN counts under both a 0.50 probability threshold and the top-bucket decision rule.",
      "The calibration curve compares predicted probabilities against observed crash rates, which is important when probabilities are used in business overlays."
    ],
    note: "Use these tables when writing the machine-learning evaluation section."
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
  },
  "text-analytics": {
    title: "Text analytics",
    lead: "This section is separate from the crash-risk model. It scores ESG-negative news coverage from optional uploaded text files and ranks tickers by current controversy pressure.",
    points: [
      "The input can be a Bloomberg-style export or another CSV/XLSX file as long as it includes ticker, date, and some headline, body, text, or summary column.",
      "The 0 to 100 score blends negative sentiment intensity, ESG controversy keyword density, and article volume into one interpretable severity measure.",
      "The word cloud emphasizes controversy-heavy negative language, while the table ranks the latest score by ticker."
    ],
    note: "Use this when you want to explain why a name is attracting negative ESG news flow."
  },
  "fds-evidence": {
    title: "FDS Project Evidence",
    lead: "This section maps the dashboard to the FIN42110 marking rubric. It provides documented proof for the data, SQL, textual analysis, and ML components of the submission.",
    points: [
      "Data Summary documents sample coverage: ticker count, weekly observation count, model-ready rows, date range, and configuration settings such as the 45-day fundamentals lag and the 13-week target horizon.",
      "Cleaning Log records missing values, duplicate rows, invalid price checks, rows removed, the fundamentals availability lag, weekly Friday date alignment, and the target window definition.",
      "SQL Evidence shows five runnable queries with compact result tables: ticker observation counts, sector controversy averages, top controversy events, target class balance, and high-risk names by sector.",
      "Textual Analysis shows sentiment score, negative/positive keyword counts, and controversy keyword counts per ticker-week if a news_text or controversy_text file is supplied. Otherwise it explains the limitation clearly.",
      "Feature Descriptive Statistics lists count, mean, standard deviation, min, max, and null percentage for every configured model feature.",
      "Text Coverage and LDA shows article coverage by split plus topic words and ticker-level dominant topics.",
      "Report Figures shows SVG charts produced by a local backend run and can be copied into the written report.",
      "Download Report Artifacts provides links to the report outline, SQL summary, project report draft, and viva slides outline generated by the backend."
    ],
    note: "This section supports the report and viva — it does not change the crash-risk model signal."
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
function normalizeTextualSummary(row) {
  return {
    status: String(row.status || "").trim(),
    ticker: String(row.ticker || "").trim().toUpperCase(),
    latest_text_date: String(row.latest_text_date || row.date || "").trim(),
    article_count: numberOrZero(row.article_count),
    negative_esg_controversy_score_0_100: numberOrNull(row.negative_esg_controversy_score_0_100),
    text_sentiment_score: numberOrNull(row.text_sentiment_score),
    negative_word_count: numberOrZero(row.negative_word_count),
    positive_word_count: numberOrZero(row.positive_word_count),
    controversy_keyword_count: numberOrZero(row.controversy_keyword_count),
    score_band: String(row.score_band || "").trim(),
    note: String(row.note || "").trim(),
    source_file: String(row.source_file || "").trim(),
  };
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
function renderTickerOptions() {
  if (!tickerOptions) return;
  const tickers = [...new Set(allRows.map(r => r.ticker).filter(Boolean))].sort();
  tickerOptions.innerHTML = tickers.map(ticker => `<option value="${escapeHtml(ticker)}"></option>`).join("");
}

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
  const top = [...rows].sort((a, b) => b.crash_probability - a.crash_probability);
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
  const totalImportance = sorted.reduce((sum, row) => sum + Math.max(0, row.importance), 0) || 1;
  const rawShares = sorted.map(r => (Math.max(0, r.importance) / totalImportance) * 100);
  const flooredShares = rawShares.map(Math.floor);
  let remainder = 100 - flooredShares.reduce((sum, value) => sum + value, 0);
  rawShares
    .map((value, index) => ({ index, fraction: value - Math.floor(value) }))
    .sort((a, b) => b.fraction - a.fraction)
    .forEach(({ index }) => {
      if (remainder > 0) {
        flooredShares[index] += 1;
        remainder -= 1;
      }
    });

  importanceBars.innerHTML = sorted.map((r, index) => {
    const share = Math.max(0, r.importance) / totalImportance;
    const pct = Math.max(2, share * 100);
    const label = r.feature.replaceAll("_", " ");
    const displayValue = `${flooredShares[index]}%`;
    return `
      <div class="imp-row">
        <div class="imp-label" title="${escapeHtml(r.feature)}">${escapeHtml(label)}</div>
        <div class="imp-track">
          <div class="imp-fill" style="width:${pct}%"></div>
        </div>
        <div class="imp-value">${displayValue}</div>
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
function formatNumberCell(value, digits = 3) {
  const n = Number(value);
  return Number.isFinite(n) ? n.toFixed(digits) : escapeHtml(value ?? "-");
}

function compactParams(value) {
  if (!value) return "-";
  try {
    const parsed = typeof value === "string" ? JSON.parse(value) : value;
    return Object.entries(parsed)
      .map(([key, val]) => `${key.replace("classifier__", "")}: ${Array.isArray(val) ? val.join("/") : val}`)
      .join("; ") || "-";
  } catch {
    return String(value);
  }
}

function renderModelDiagnostics() {
  const totalRows = textModelRows.length + tuningRows.length + confusionRows.length + calibrationRows.length;
  if (modelDiagnosticsStatus) {
    modelDiagnosticsStatus.textContent = totalRows
      ? `Loaded ${totalRows} diagnostic rows from the latest backend artifacts.`
      : "No latest diagnostics loaded yet.";
  }

  if (tuningBadge) tuningBadge.textContent = tuningRows.length ? `${tuningRows.length} models` : "";
  if (tuningRowsBody) {
    tuningRowsBody.innerHTML = tuningRows.length ? tuningRows.map(row => `
      <tr>
        <td>${escapeHtml(String(row.model || ""))}</td>
        <td class="muted-cell">${escapeHtml(compactParams(row.best_params))}</td>
        <td class="metric-value">${formatNumberCell(row.best_cv_roc_auc)}</td>
        <td class="metric-value">${formatNumberCell(row.cv_roc_auc_std)}</td>
        <td class="metric-value">${escapeHtml(String(row.n_candidates ?? ""))}</td>
      </tr>`).join("") : '<tr><td colspan="5" class="empty-state">No tuning results loaded.</td></tr>';
  }

  if (textModelBadge) textModelBadge.textContent = textModelRows.length ? `${textModelRows.length} rows` : "";
  if (textModelRowsBody) {
    textModelRowsBody.innerHTML = textModelRows.length ? textModelRows.map(row => `
      <tr>
        <td>${escapeHtml(String(row.model || "").replaceAll("_", " "))}</td>
        <td><span class="split-pill">${escapeHtml(String(row.split || ""))}</span></td>
        <td class="metric-value">${escapeHtml(String(row.text_covered_rows ?? ""))}</td>
        <td class="metric-value">${formatNumberCell(row.roc_auc)}</td>
        <td class="metric-value">${formatNumberCell(row.precision_at_top_bucket)}</td>
        <td class="metric-value">${formatNumberCell(row.crash_capture_at_top_bucket)}</td>
      </tr>`).join("") : '<tr><td colspan="6" class="empty-state">No text model comparison loaded.</td></tr>';
  }

  if (confusionBadge) confusionBadge.textContent = confusionRows.length ? `${confusionRows.length} thresholds` : "";
  if (confusionRowsBody) {
    confusionRowsBody.innerHTML = confusionRows.length ? confusionRows.map(row => `
      <tr>
        <td>${escapeHtml(String(row.threshold || ""))}</td>
        <td class="metric-value">${escapeHtml(String(row.tp ?? ""))}</td>
        <td class="metric-value">${escapeHtml(String(row.fp ?? ""))}</td>
        <td class="metric-value">${escapeHtml(String(row.tn ?? ""))}</td>
        <td class="metric-value">${escapeHtml(String(row.fn ?? ""))}</td>
        <td class="metric-value">${formatNumberCell(row.precision)}</td>
        <td class="metric-value">${formatNumberCell(row.recall)}</td>
      </tr>`).join("") : '<tr><td colspan="7" class="empty-state">No confusion matrix loaded.</td></tr>';
  }

  if (calibrationBadge) calibrationBadge.textContent = calibrationRows.length ? `${calibrationRows.length} bins` : "";
  if (calibrationRowsBody) {
    calibrationRowsBody.innerHTML = calibrationRows.length ? calibrationRows.map(row => `
      <tr>
        <td>${escapeHtml(String(row.bin || ""))}</td>
        <td class="metric-value">${escapeHtml(String(row.n_rows ?? ""))}</td>
        <td class="metric-value">${formatNumberCell(row.mean_predicted_probability)}</td>
        <td class="metric-value">${formatNumberCell(row.observed_crash_rate)}</td>
      </tr>`).join("") : '<tr><td colspan="4" class="empty-state">No calibration curve loaded.</td></tr>';
  }
}

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
  benchmark_alpha_annualized: "Benchmark alpha",
  strategy_sharpe:         "Strategy Sharpe ratio",
  benchmark_sharpe:        "Benchmark Sharpe ratio",
  strategy_sortino:        "Strategy Sortino ratio",
  benchmark_sortino:       "Benchmark Sortino ratio",
  max_drawdown_strategy:   "Max drawdown - strategy",
  max_drawdown_benchmark:  "Max drawdown - benchmark",
  drawdown_improvement:    "Drawdown improvement",
  var_95_weekly:           "Weekly VaR (95%)",
  benchmark_var_95_weekly: "Benchmark weekly VaR (95%)",
  cvar_95_weekly:          "Weekly CVaR (95%)",
  benchmark_cvar_95_weekly:"Benchmark weekly CVaR (95%)",
  evaluation_weeks:        "Evaluation weeks",
  high_risk_excluded_pct:  "High-risk excluded %",
  benchmark_high_risk_excluded_pct: "Benchmark high-risk excluded %",
  fund_aum:                "Fund AUM ($)",
  economic_gain_annual:    "Annual economic gain ($)",
  benchmark_economic_gain_annual: "Benchmark economic gain ($)",
  team_annual_cost:        "Team annual cost ($)",
  team_roi:                "Team ROI (x)",
  benchmark_team_roi:      "Benchmark Team ROI",
  justifies_team:          "Justifies team hire?",
};
const BIZ_SUMMARY_KEYS = ["alpha_annualized", "strategy_sharpe", "economic_gain_annual"];
const BIZ_TABLE_EXCLUDED = new Set(["business_analysis_note"]);

function formatBizMetric(metric, raw) {
  const v = parseFloat(raw);
  if ([
    "strategy_annual_return",
    "benchmark_annual_return",
    "alpha_annualized",
    "benchmark_alpha_annualized",
    "high_risk_excluded_pct",
    "benchmark_high_risk_excluded_pct",
    "max_drawdown_strategy",
    "max_drawdown_benchmark",
    "drawdown_improvement",
    "var_95_weekly",
    "benchmark_var_95_weekly",
    "cvar_95_weekly",
    "benchmark_cvar_95_weekly",
  ].includes(metric))
    return Number.isFinite(v) ? `${(v * 100).toFixed(2)}%` : raw;
  if (["fund_aum","economic_gain_annual","benchmark_economic_gain_annual","team_annual_cost"].includes(metric))
    return Number.isFinite(v) ? `$${v.toLocaleString()}` : raw;
  if (metric === "team_roi" && Number.isFinite(v))
    return `${v.toFixed(2)}x`;
  if (metric === "benchmark_team_roi" && (raw === "None" || raw === "" || raw == null))
    return "-";
  return raw;
}

function renderBizHighlights(lookup) {
  if (!bizHighlights) return;
  const note = lookup.business_analysis_note || "Illustrative gross overlay result; not a guaranteed return forecast.";
  const cards = [
    {
      title: "Overlay return",
      value: formatBizMetric("strategy_annual_return", lookup.strategy_annual_return ?? "-"),
      detail: `Benchmark: ${formatBizMetric("benchmark_annual_return", lookup.benchmark_annual_return ?? "-")} | Alpha: ${formatBizMetric("alpha_annualized", lookup.alpha_annualized ?? "-")}`,
    },
    {
      title: "Risk control",
      value: formatBizMetric("max_drawdown_strategy", lookup.max_drawdown_strategy ?? "-"),
      detail: `Benchmark drawdown: ${formatBizMetric("max_drawdown_benchmark", lookup.max_drawdown_benchmark ?? "-")} | Weekly CVaR: ${formatBizMetric("cvar_95_weekly", lookup.cvar_95_weekly ?? "-")}`,
    },
    {
      title: "Business case",
      value: formatBizMetric("economic_gain_annual", lookup.economic_gain_annual ?? "-"),
      detail: `Team cost: ${formatBizMetric("team_annual_cost", lookup.team_annual_cost ?? "-")} | ROI: ${formatBizMetric("team_roi", lookup.team_roi ?? "-")}`,
    },
  ];

  bizHighlights.innerHTML = `
    ${cards.map(card => `
      <article class="business-value-card">
        <span>${escapeHtml(card.title)}</span>
        <strong>${escapeHtml(card.value)}</strong>
        <p>${escapeHtml(card.detail)}</p>
      </article>
    `).join("")}
    <article class="business-value-card business-note-card">
      <span>How to read this</span>
      <strong>Illustrative overlay</strong>
      <p>${escapeHtml(note)}</p>
    </article>`;
}

function renderBizAnalysis() {
  if (!bizRows.length) {
    bizSummary.innerHTML = '<div class="empty-state">No business analysis loaded.</div>';
    if (bizHighlights) bizHighlights.innerHTML = "";
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
  renderBizHighlights(lookup);

  // Full table
  bizRowsBody.innerHTML = bizRows.filter(r => !BIZ_TABLE_EXCLUDED.has(r.metric)).map(r => {
    const label = BIZ_LABELS[r.metric] || r.metric;
    let display = formatBizMetric(r.metric, r.value);
    const v = parseFloat(r.value);
    if (["strategy_annual_return","benchmark_annual_return","alpha_annualized","benchmark_alpha_annualized","high_risk_excluded_pct","benchmark_high_risk_excluded_pct"].includes(r.metric))
      display = Number.isFinite(v) ? `${(v*100).toFixed(2)}%` : r.value;
    else if (["fund_aum","economic_gain_annual","benchmark_economic_gain_annual","team_annual_cost"].includes(r.metric))
      display = Number.isFinite(v) ? `$${v.toLocaleString()}` : r.value;
    else if (r.metric === "team_roi" && Number.isFinite(v))
      display = `${v.toFixed(2)}x`;
    else if (r.metric === "benchmark_team_roi")
      display = "-";
    return `<tr><td>${escapeHtml(label)}</td><td style="font-variant-numeric:tabular-nums">${escapeHtml(display)}</td></tr>`;
  }).join("");
}

/* ── FDS Evidence ──────────────────────────────────────────────────────── */
const REPORT_FIGURES = [
  { name: "Risk probability ranking",  paths: ["outputs/figures/risk_probability_ranking.svg",  "../outputs/figures/risk_probability_ranking.svg"] },
  { name: "Sector controversy",        paths: ["outputs/figures/sector_controversy.svg",          "../outputs/figures/sector_controversy.svg"] },
  { name: "Controversy over time",     paths: ["outputs/figures/controversy_over_time.svg",       "../outputs/figures/controversy_over_time.svg"] },
  { name: "Feature importance",        paths: ["outputs/figures/feature_importance.svg",          "../outputs/figures/feature_importance.svg"] },
  { name: "ESG lift",                  paths: ["outputs/figures/esg_lift.svg",                    "../outputs/figures/esg_lift.svg"] },
  { name: "Price scenario range",      paths: ["outputs/figures/price_scenario_range.svg",        "../outputs/figures/price_scenario_range.svg"] },
  { name: "Text word cloud",           paths: ["outputs/figures/text_word_cloud.svg",             "../outputs/figures/text_word_cloud.svg"] },
  { name: "Feature correlation heatmap", paths: ["outputs/figures/feature_correlation_heatmap.png", "../outputs/figures/feature_correlation_heatmap.png"] },
  { name: "Representative price time series", paths: ["outputs/figures/price_time_series.png", "../outputs/figures/price_time_series.png"] },
  { name: "Probability calibration", paths: ["outputs/figures/probability_calibration.png", "../outputs/figures/probability_calibration.png"] },
  { name: "LDA topic distribution", paths: ["outputs/figures/lda_topic_distribution.png", "../outputs/figures/lda_topic_distribution.png"] },
];
const TEXT_WORD_CLOUD_PATHS = ["outputs/figures/text_word_cloud.svg", "../outputs/figures/text_word_cloud.svg"];

const REPORT_ARTIFACTS = [
  { name: "FDS report outline",   paths: ["outputs/fds_report_outline.md",        "../outputs/fds_report_outline.md"],        filename: "fds_report_outline.md" },
  { name: "SQL summary",          paths: ["outputs/sql_summary.md",               "../outputs/sql_summary.md"],               filename: "sql_summary.md" },
  { name: "Feature descriptive stats", paths: ["outputs/feature_descriptive_stats.csv", "../outputs/feature_descriptive_stats.csv"], filename: "feature_descriptive_stats.csv" },
  { name: "Text model comparison", paths: ["outputs/text_model_comparison.csv", "../outputs/text_model_comparison.csv"], filename: "text_model_comparison.csv" },
  { name: "Hyperparameter tuning", paths: ["outputs/hyperparameter_tuning_results.csv", "../outputs/hyperparameter_tuning_results.csv"], filename: "hyperparameter_tuning_results.csv" },
  { name: "Confusion matrix", paths: ["outputs/confusion_matrix.csv", "../outputs/confusion_matrix.csv"], filename: "confusion_matrix.csv" },
  { name: "Calibration curve", paths: ["outputs/calibration_curve.csv", "../outputs/calibration_curve.csv"], filename: "calibration_curve.csv" },
  { name: "Text coverage", paths: ["outputs/text_coverage.csv", "../outputs/text_coverage.csv"], filename: "text_coverage.csv" },
  { name: "LDA topic words", paths: ["outputs/lda_topic_words.csv", "../outputs/lda_topic_words.csv"], filename: "lda_topic_words.csv" },
  { name: "Project report draft", paths: ["reports/fds_project_report_draft.md",  "../reports/fds_project_report_draft.md"],  filename: "fds_project_report_draft.md" },
  { name: "Viva slides outline",  paths: ["reports/fds_viva_slides_outline.md",   "../reports/fds_viva_slides_outline.md"],   filename: "fds_viva_slides_outline.md" },
];

/* ── Quarter snapshot backtest ─────────────────────────────────────────── */

function setQuarterBt(data) {
  quarterBt = (data && !data.error && data.weekly_series) ? data : null;
  renderQuarterBacktest();
}

function buildQbCumReturnSvg(weeklySeries) {
  const W = 520, H = 220, padL = 48, padR = 16, padT = 14, padB = 36;
  const iW = W - padL - padR, iH = H - padT - padB;
  const n = weeklySeries.length;
  if (n < 2) return "";
  const sCum = weeklySeries.map(d => d.strategy_cumulative);
  const bCum = weeklySeries.map(d => d.benchmark_cumulative);
  const allVals = [...sCum, ...bCum];
  const yMin = Math.min(...allVals) - 0.002;
  const yMax = Math.max(...allVals) + 0.002;
  const xOf = i => padL + (i / (n - 1)) * iW;
  const yOf = v => padT + iH - ((v - yMin) / (yMax - yMin)) * iH;
  const pts = arr => arr.map((v, i) => `${xOf(i).toFixed(1)},${yOf(v).toFixed(1)}`).join(" ");
  const fmt = v => `${((v - 1) * 100).toFixed(1)}%`;
  const finalS = sCum[n - 1], finalB = bCum[n - 1];
  const tickDates = [0, Math.floor(n / 2), n - 1].map(i => weeklySeries[i].date.slice(0, 10));
  const tickX    = [0, Math.floor(n / 2), n - 1].map(xOf);
  return `<svg viewBox="0 0 ${W} ${H}" style="width:100%;max-height:220px;display:block">
    <polyline points="${pts(sCum)}" fill="none" stroke="#0d6b62" stroke-width="2.2"/>
    <polyline points="${pts(bCum)}" fill="none" stroke="#7f8c8d" stroke-width="1.8" stroke-dasharray="5,3"/>
    <text x="${xOf(n-1)+4}" y="${yOf(finalS)+4}" fill="#0d6b62" font-size="11" font-weight="bold">${fmt(finalS)}</text>
    <text x="${xOf(n-1)+4}" y="${yOf(finalB)+4}" fill="#7f8c8d" font-size="11">${fmt(finalB)}</text>
    ${tickDates.map((d, i) => `<text x="${tickX[i]}" y="${H - 4}" text-anchor="middle" fill="#555" font-size="10">${d}</text>`).join("")}
    <text x="${padL - 4}" y="${yOf(yMin + (yMax - yMin) * 0.5)}" text-anchor="end" fill="#555" font-size="10" transform="rotate(-90,${padL-4},${yOf(yMin+(yMax-yMin)*0.5)})">Cumul. Return</text>
    <circle cx="${W-120}" cy="${padT+8}" r="5" fill="#0d6b62"/>
    <text x="${W-112}" y="${padT+12}" fill="#0d6b62" font-size="10">Strategy</text>
    <line x1="${W-55}" y1="${padT+8}" x2="${W-40}" y2="${padT+8}" stroke="#7f8c8d" stroke-width="1.8" stroke-dasharray="4,2"/>
    <text x="${W-37}" y="${padT+12}" fill="#7f8c8d" font-size="10">Benchmark</text>
  </svg>`;
}

function buildQbExcludedBarSvg(excludedTickers) {
  const items = excludedTickers.filter(t => t.quarter_return != null);
  if (!items.length) return "";
  const sorted = [...items].sort((a, b) => a.quarter_return - b.quarter_return);
  const W = 400, barH = 22, gap = 4, padL = 48, padR = 60, padT = 10, padB = 10;
  const H = padT + sorted.length * (barH + gap) + padB;
  const vals = sorted.map(t => t.quarter_return);
  const absMax = Math.max(...vals.map(Math.abs), 0.01);
  const midX = padL + (W - padL - padR) / 2;
  const xScale = v => midX + (v / absMax) * (midX - padL - 4);
  const rows = sorted.map((t, i) => {
    const y = padT + i * (barH + gap);
    const x0 = Math.min(midX, xScale(t.quarter_return));
    const bW = Math.abs(xScale(t.quarter_return) - midX);
    const col = t.quarter_return < 0 ? "#1a7a4a" : "#c0392b";
    const pct = `${(t.quarter_return * 100).toFixed(1)}%`;
    return `<text x="${padL - 4}" y="${y + barH * 0.68}" text-anchor="end" font-size="10" fill="#333">${escapeHtml(t.ticker)}</text>
      <rect x="${x0}" y="${y+2}" width="${Math.max(bW,1)}" height="${barH-4}" fill="${col}" rx="2"/>
      <text x="${xScale(t.quarter_return) + (t.quarter_return >= 0 ? 3 : -3)}" y="${y + barH * 0.68}" text-anchor="${t.quarter_return >= 0 ? 'start' : 'end'}" font-size="9" fill="${col}" font-weight="bold">${pct}</text>`;
  });
  return `<svg viewBox="0 0 ${W} ${H}" style="width:100%;max-height:${H}px;display:block">
    <line x1="${midX}" y1="${padT}" x2="${midX}" y2="${H-padB}" stroke="#ccc" stroke-width="1"/>
    ${rows.join("")}
    <text x="${W/2}" y="${H-1}" text-anchor="middle" fill="#555" font-size="10">Q4 Return (%)</text>
  </svg>`;
}

function renderQuarterBacktest() {
  if (!quarterBtSection) return;
  if (!quarterBt) {
    quarterBtSection.style.display = "none";
    return;
  }
  quarterBtSection.style.display = "";
  const q = quarterBt;
  const alphaBps  = Number.isFinite(q.outperformance_bps) ? q.outperformance_bps : "—";
  const dollarImp = q.dollar_impact_quarter != null
    ? `$${(q.dollar_impact_quarter / 1e6).toFixed(1)}M`
    : "—";
  const pctCorr = q.pct_excluded_correct != null
    ? `${Math.round(q.pct_excluded_correct * 100)}%`
    : "—";

  if (quarterBtStatus)
    quarterBtStatus.textContent = `${q.quarter_label} · cutoff ${q.cutoff_date} · ${q.n_excluded} stocks excluded`;

  if (quarterBtHero) {
    const sign = alphaBps >= 0 ? "+" : "";
    quarterBtHero.innerHTML = [
      { label: "Alpha vs Benchmark", val: `${sign}${alphaBps} bps`, cls: alphaBps >= 0 ? "positive" : "negative" },
      { label: `Dollar Impact ($1B Fund, ${q.quarter_label})`, val: dollarImp, cls: "positive" },
      { label: `Flagged Stocks That Fell in ${q.quarter_label}`, val: pctCorr, cls: "" },
    ].map(c => `<article class="biz-card ${c.cls}"><span>${escapeHtml(c.label)}</span><strong>${escapeHtml(c.val)}</strong></article>`).join("");
  }

  if (quarterReturnChartEl)
    quarterReturnChartEl.innerHTML = buildQbCumReturnSvg(q.weekly_series || []);
  if (quarterExcludedBarChartEl)
    quarterExcludedBarChartEl.innerHTML = buildQbExcludedBarSvg(q.excluded_tickers || []);

  if (quarterExcludedTable) {
    quarterExcludedTable.innerHTML = (q.excluded_tickers || []).map(t => {
      const ret = t.quarter_return != null ? `${(t.quarter_return * 100).toFixed(1)}%` : "N/A";
      const retNum = t.quarter_return != null ? t.quarter_return : 0;
      const rowCls = t.outcome === "Avoided loss" ? "style=\"background:#efffef\"" : t.outcome === "Model missed" ? "style=\"background:#fff0f0\"" : "";
      return `<tr ${rowCls}><td>${escapeHtml(t.ticker)}</td><td>${(t.crash_probability * 100).toFixed(1)}%</td><td style="color:${retNum < 0 ? '#1a7a4a' : '#c0392b'};font-weight:600">${ret}</td><td>${escapeHtml(t.outcome || "—")}</td></tr>`;
    }).join("");
  }
}

function renderEvidenceHeadline() {
  if (!evidenceHeadline) return;
  const lookup = {};
  dataSummaryRows.forEach(r => { lookup[`${r.section}::${r.metric}`] = r.value; });
  const tickers   = lookup["feature_panel::ticker_count"] ?? "—";
  const weeklyObs = lookup["feature_panel::rows"]         ?? "—";
  const modelRows = lookup["model_dataset::rows"]         ?? "—";
  const dateStart = lookup["prices::date_start"]          ?? "";
  const dateEnd   = lookup["prices::date_end"]            ?? "";
  const dateRange = dateStart && dateEnd ? `${dateStart} → ${dateEnd}` : "—";
  const textStatus = textualAnalysisRows[0]?.status ?? "—";
  const textLabel  = textStatus === "ok" ? "Loaded" : textStatus === "no_text_file" ? "No text file" : textStatus.replaceAll("_", " ");
  const textCls    = textStatus === "ok" ? "positive" : "";
  evidenceHeadline.innerHTML = [
    `<article class="evidence-tile"><span>Tickers</span><strong>${escapeHtml(String(tickers))}</strong></article>`,
    `<article class="evidence-tile"><span>Weekly obs.</span><strong>${escapeHtml(String(weeklyObs))}</strong></article>`,
    `<article class="evidence-tile"><span>Model rows</span><strong>${escapeHtml(String(modelRows))}</strong></article>`,
    `<article class="evidence-tile"><span>Date range</span><strong style="font-size:.85rem">${escapeHtml(dateRange)}</strong></article>`,
    `<article class="evidence-tile ${textCls}"><span>ESG text</span><strong>${escapeHtml(textLabel)}</strong></article>`,
  ].join("");
}

function renderDataSummary() {
  if (!dataSummaryContent) return;
  dataSummaryBadge.textContent = dataSummaryRows.length ? `${dataSummaryRows.length} rows` : "";
  if (!dataSummaryRows.length) {
    dataSummaryContent.innerHTML = '<div class="empty-state">No data summary loaded.</div>';
    return;
  }
  const sections = {};
  dataSummaryRows.forEach(r => {
    if (!sections[r.section]) sections[r.section] = [];
    sections[r.section].push(r);
  });
  let html = "";
  for (const [section, rows] of Object.entries(sections)) {
    html += `<div class="evidence-subsection">
      <span class="evidence-subsection-title">${escapeHtml(section)}</span>
      <div class="table-shell"><table class="evidence-table">
        <thead><tr><th>Metric</th><th>Value</th><th>Detail</th></tr></thead>
        <tbody>${rows.map(r => `
          <tr>
            <td>${escapeHtml(r.metric)}</td>
            <td class="metric-value">${escapeHtml(String(r.value))}</td>
            <td class="muted-cell">${escapeHtml(r.detail)}</td>
          </tr>`).join("")}
        </tbody>
      </table></div>
    </div>`;
  }
  dataSummaryContent.innerHTML = html;
}

function renderCleaningLog() {
  if (!cleaningLogContent) return;
  cleaningLogBadge.textContent = cleaningLogRows.length ? `${cleaningLogRows.length} checks` : "";
  if (!cleaningLogRows.length) {
    cleaningLogContent.innerHTML = '<div class="empty-state">No cleaning log loaded.</div>';
    return;
  }
  cleaningLogContent.innerHTML = `<div class="table-shell"><table class="evidence-table">
    <thead><tr><th>Dataset</th><th>Check</th><th>Value</th><th>Detail</th></tr></thead>
    <tbody>${cleaningLogRows.map(r => `
      <tr>
        <td>${escapeHtml(r.dataset)}</td>
        <td>${escapeHtml(r.check)}</td>
        <td class="metric-value">${escapeHtml(String(r.value))}</td>
        <td class="muted-cell">${escapeHtml(r.detail)}</td>
      </tr>`).join("")}
    </tbody>
  </table></div>`;
}

function renderSqlEvidence() {
  if (!sqlEvidenceContent) return;
  sqlEvidenceBadge.textContent = sqlSummaryRows.length ? `${sqlSummaryRows.length} queries` : "";
  if (!sqlSummaryRows.length) {
    sqlEvidenceContent.innerHTML = '<div class="empty-state">No SQL evidence loaded.</div>';
    return;
  }
  const cards = sqlSummaryRows.map(r => {
    let resultHtml = "";
    try {
      const records = JSON.parse(r.result_json || "[]");
      if (Array.isArray(records) && records.length && typeof records[0] === "object") {
        const cols = Object.keys(records[0]);
        resultHtml = `<div class="table-shell"><table class="sql-result-table">
          <thead><tr>${cols.map(c => `<th>${escapeHtml(c)}</th>`).join("")}</tr></thead>
          <tbody>${records.slice(0, 10).map(rec =>
            `<tr>${cols.map(c => `<td>${escapeHtml(String(rec[c] ?? ""))}</td>`).join("")}</tr>`
          ).join("")}</tbody>
        </table></div>`;
        if (records.length > 10) resultHtml += `<p class="status-text">Showing 10 of ${records.length} rows.</p>`;
      } else {
        resultHtml = `<p class="status-text">No result rows for this query.</p>`;
      }
    } catch {
      resultHtml = `<p class="status-text">Could not parse result JSON.</p>`;
    }
    return `<div class="sql-query-card">
      <div class="sql-query-name">${escapeHtml(r.query_name.replaceAll("_", " "))}</div>
      <details class="sql-query-details">
        <summary>View SQL</summary>
        <pre class="sql-query-text">${escapeHtml(r.query)}</pre>
      </details>
      ${resultHtml}
    </div>`;
  });
  sqlEvidenceContent.innerHTML = cards.join("");
}

function renderTextualAnalysis() {
  if (!textualAnalysisContent) return;
  if (!textualAnalysisRows.length) {
    textualAnalysisBadge.textContent = "";
    textualAnalysisContent.innerHTML = '<div class="empty-state">No textual analysis loaded.</div>';
    return;
  }
  const status = textualAnalysisRows[0]?.status;
  if (status !== "ok") {
    const note = textualAnalysisRows[0]?.note || "";
    textualAnalysisBadge.textContent = (status || "").replaceAll("_", " ");
    textualAnalysisContent.innerHTML = `<div class="text-limitation-box">
      <p><strong>Status: ${escapeHtml(status || "unknown")}</strong></p>
      <p>${escapeHtml(note)}</p>
      <p class="status-text">Limitation: supply a <code>controversy_text.csv</code> or <code>news_text.csv</code>
      with ticker, date, and headline/text columns to enable direct sentiment analysis.</p>
    </div>`;
    return;
  }
  textualAnalysisBadge.textContent = `${textualAnalysisRows.length} weekly rows`;
  const totalArticles = textualAnalysisRows.reduce((s, r) => s + (Number(r.article_count) || 0), 0);
  const avgSentiment  = textualAnalysisRows.reduce((s, r) => s + (Number(r.text_sentiment_score) || 0), 0) / textualAnalysisRows.length;
  const totalNeg      = textualAnalysisRows.reduce((s, r) => s + (Number(r.negative_word_count) || 0), 0);
  const totalPos      = textualAnalysisRows.reduce((s, r) => s + (Number(r.positive_word_count) || 0), 0);
  const totalKw       = textualAnalysisRows.reduce((s, r) => s + (Number(r.controversy_keyword_count) || 0), 0);
  const tickers       = new Set(textualAnalysisRows.map(r => r.ticker).filter(Boolean)).size;
  textualAnalysisContent.innerHTML = `
    <div class="evidence-headline-grid" style="margin:12px 0 16px">
      <article class="evidence-tile"><span>Articles</span><strong>${totalArticles.toLocaleString()}</strong></article>
      <article class="evidence-tile"><span>Tickers</span><strong>${tickers}</strong></article>
      <article class="evidence-tile"><span>Avg sentiment</span><strong>${avgSentiment.toFixed(4)}</strong></article>
      <article class="evidence-tile"><span>Negative kw</span><strong>${totalNeg.toLocaleString()}</strong></article>
      <article class="evidence-tile"><span>Positive kw</span><strong>${totalPos.toLocaleString()}</strong></article>
      <article class="evidence-tile"><span>Controversy kw</span><strong>${totalKw.toLocaleString()}</strong></article>
    </div>
    <div class="table-shell"><table class="evidence-table">
      <thead><tr><th>Ticker</th><th>Date</th><th>Articles</th><th>Sentiment</th><th>Neg</th><th>Pos</th><th>Controversy kw</th></tr></thead>
      <tbody>${textualAnalysisRows.slice(0, 20).map(r => `
        <tr>
          <td>${escapeHtml(r.ticker || "")}</td>
          <td>${escapeHtml(String(r.date || ""))}</td>
          <td class="metric-value">${escapeHtml(String(r.article_count ?? ""))}</td>
          <td class="metric-value">${Number.isFinite(Number(r.text_sentiment_score)) ? Number(r.text_sentiment_score).toFixed(4) : "—"}</td>
          <td class="metric-value">${escapeHtml(String(r.negative_word_count ?? ""))}</td>
          <td class="metric-value">${escapeHtml(String(r.positive_word_count ?? ""))}</td>
          <td class="metric-value">${escapeHtml(String(r.controversy_keyword_count ?? ""))}</td>
        </tr>`).join("")}
      </tbody>
    </table></div>
    ${textualAnalysisRows.length > 20 ? `<p class="status-text">Showing 20 of ${textualAnalysisRows.length} rows.</p>` : ""}`;
}

function renderFeatureStats() {
  if (!featureStatsContent) return;
  if (featureStatsBadge) featureStatsBadge.textContent = featureStatsRows.length ? `${featureStatsRows.length} features` : "";
  if (!featureStatsRows.length) {
    featureStatsContent.innerHTML = '<div class="empty-state">No feature descriptive statistics loaded.</div>';
    return;
  }
  featureStatsContent.innerHTML = `<div class="table-shell"><table class="evidence-table">
    <thead><tr><th>Feature</th><th>Group</th><th>Count</th><th>Mean</th><th>Std</th><th>Min</th><th>Max</th><th>Null %</th></tr></thead>
    <tbody>${featureStatsRows.map(row => `
      <tr>
        <td>${escapeHtml(row.feature || "")}</td>
        <td>${escapeHtml(row.feature_group || "")}</td>
        <td class="metric-value">${escapeHtml(String(row.count ?? ""))}</td>
        <td class="metric-value">${formatNumberCell(row.mean, 4)}</td>
        <td class="metric-value">${formatNumberCell(row.std, 4)}</td>
        <td class="metric-value">${formatNumberCell(row.min, 4)}</td>
        <td class="metric-value">${formatNumberCell(row.max, 4)}</td>
        <td class="metric-value">${formatNumberCell(row.null_percent, 2)}</td>
      </tr>`).join("")}
    </tbody>
  </table></div>`;
}

function renderTextCoverageLda() {
  if (!textCoverageContent) return;
  const badgeParts = [];
  if (textCoverageRows.length) badgeParts.push(`${textCoverageRows.length} coverage rows`);
  if (ldaTopicRows.length) badgeParts.push(`${new Set(ldaTopicRows.map(r => r.topic).filter(Boolean)).size} topics`);
  if (textCoverageBadge) textCoverageBadge.textContent = badgeParts.join(" | ");
  if (!textCoverageRows.length && !ldaTopicRows.length && !ldaTickerTopicRows.length) {
    textCoverageContent.innerHTML = '<div class="empty-state">No text coverage or LDA outputs loaded.</div>';
    return;
  }

  const coverageTable = textCoverageRows.length ? `<div class="evidence-subsection">
    <span class="evidence-subsection-title">Coverage by split</span>
    <div class="table-shell"><table class="evidence-table">
      <thead><tr><th>Split</th><th>Status</th><th>Articles</th><th>Weekly rows</th><th>Tickers</th><th>Date range</th><th>Matched model rows</th></tr></thead>
      <tbody>${textCoverageRows.map(row => `
        <tr>
          <td>${escapeHtml(row.split || "")}</td>
          <td>${escapeHtml(row.status || "")}</td>
          <td class="metric-value">${escapeHtml(String(row.article_count ?? ""))}</td>
          <td class="metric-value">${escapeHtml(String(row.weekly_text_rows ?? ""))}</td>
          <td class="metric-value">${escapeHtml(String(row.unique_tickers ?? ""))}</td>
          <td>${escapeHtml(`${row.date_start || ""} to ${row.date_end || ""}`)}</td>
          <td class="metric-value">${escapeHtml(String(row.matched_model_rows ?? ""))}</td>
        </tr>`).join("")}</tbody>
    </table></div>
  </div>` : "";

  const topics = {};
  ldaTopicRows.forEach(row => {
    if (!row.topic) return;
    if (!topics[row.topic]) topics[row.topic] = [];
    topics[row.topic].push(row);
  });
  const topicTable = Object.keys(topics).length ? `<div class="evidence-subsection">
    <span class="evidence-subsection-title">LDA topic words</span>
    <div class="table-shell"><table class="evidence-table">
      <thead><tr><th>Topic</th><th>Top words</th></tr></thead>
      <tbody>${Object.entries(topics).map(([topic, rows]) => `
        <tr>
          <td>${escapeHtml(topic)}</td>
          <td>${escapeHtml(rows.sort((a, b) => Number(a.rank) - Number(b.rank)).slice(0, 8).map(row => row.word).join(", "))}</td>
        </tr>`).join("")}</tbody>
    </table></div>
  </div>` : "";

  const tickerTable = ldaTickerTopicRows.length ? `<div class="evidence-subsection">
    <span class="evidence-subsection-title">Ticker dominant topics</span>
    <div class="table-shell"><table class="evidence-table">
      <thead><tr><th>Ticker</th><th>Articles</th><th>Dominant topic</th><th>Topic probability</th><th>Date range</th></tr></thead>
      <tbody>${ldaTickerTopicRows.slice(0, 20).map(row => `
        <tr>
          <td>${escapeHtml(row.ticker || "")}</td>
          <td class="metric-value">${escapeHtml(String(row.article_count ?? ""))}</td>
          <td>${escapeHtml(row.dominant_topic || "")}</td>
          <td class="metric-value">${formatNumberCell(row.topic_probability)}</td>
          <td>${escapeHtml(`${row.date_start || ""} to ${row.date_end || ""}`)}</td>
        </tr>`).join("")}</tbody>
    </table></div>
    ${ldaTickerTopicRows.length > 20 ? `<p class="status-text">Showing 20 of ${ldaTickerTopicRows.length} ticker-topic rows.</p>` : ""}
  </div>` : "";

  textCoverageContent.innerHTML = coverageTable + topicTable + tickerTable;
}

async function renderTextAnalytics() {
  if (!textTickerSummaryContent || !textAnalyticsSummary || !textWordCloudContent) return;

  const validRows = textualTickerSummaryRows.filter(row => row.status === "ok" && row.ticker);
  const status = validRows.length ? "ok" : (textualTickerSummaryRows[0]?.status || textualAnalysisRows[0]?.status || "");
  const note = textualTickerSummaryRows[0]?.note || textualAnalysisRows[0]?.note || "";

  if (textAnalyticsBadge) {
    textAnalyticsBadge.textContent = validRows.length ? `${validRows.length} tickers` : (status || "").replaceAll("_", " ");
  }

  if (!validRows.length) {
    if (textAnalyticsStatus) {
      textAnalyticsStatus.textContent = note || "Upload optional text files to score ESG-negative coverage.";
    }
    textAnalyticsSummary.innerHTML = '<div class="empty-state">No text ticker summary loaded.</div>';
    textTickerSummaryContent.innerHTML = `<div class="text-limitation-box">
      <p><strong>Status: ${escapeHtml(status || "unknown")}</strong></p>
      <p>${escapeHtml(note || "Supply controversy_text.csv or news_text.csv to rank tickers by negative ESG controversy score.")}</p>
    </div>`;
  } else {
    const avgScore = validRows.reduce((sum, row) => sum + (row.negative_esg_controversy_score_0_100 || 0), 0) / validRows.length;
    const totalArticles = validRows.reduce((sum, row) => sum + row.article_count, 0);
    const topRow = validRows[0];
    if (textAnalyticsStatus) {
      textAnalyticsStatus.textContent = `Ranked ${validRows.length} tickers from optional text uploads.`;
    }
    textAnalyticsSummary.innerHTML = `
      <article class="metric-tile"><span>Average controversy score</span><strong>${avgScore.toFixed(1)}</strong></article>
      <article class="metric-tile"><span>Highest ticker score</span><strong>${(topRow.negative_esg_controversy_score_0_100 || 0).toFixed(1)}</strong></article>
      <article class="metric-tile"><span>Total text articles</span><strong>${totalArticles.toLocaleString()}</strong></article>
      <article class="metric-tile"><span>Covered tickers</span><strong>${validRows.length}</strong></article>
    `;
    textTickerSummaryContent.innerHTML = `
      <div class="table-shell"><table class="evidence-table">
        <thead><tr><th>Ticker</th><th>Latest date</th><th>Score</th><th>Band</th><th>Articles</th><th>Sentiment</th><th>Controversy kw</th></tr></thead>
        <tbody>${validRows.slice(0, 20).map(row => `
          <tr>
            <td>${escapeHtml(row.ticker)}</td>
            <td>${escapeHtml(row.latest_text_date || "")}</td>
            <td class="metric-value">${Number.isFinite(row.negative_esg_controversy_score_0_100) ? row.negative_esg_controversy_score_0_100.toFixed(1) : "—"}</td>
            <td><span class="score-band ${String(row.score_band || "").toLowerCase()}">${escapeHtml(row.score_band || "—")}</span></td>
            <td class="metric-value">${row.article_count.toLocaleString()}</td>
            <td class="metric-value">${Number.isFinite(row.text_sentiment_score) ? row.text_sentiment_score.toFixed(4) : "—"}</td>
            <td class="metric-value">${row.controversy_keyword_count.toLocaleString()}</td>
          </tr>`).join("")}
        </tbody>
      </table></div>
      ${validRows.length > 20 ? `<p class="status-text">Showing 20 of ${validRows.length} rows.</p>` : ""}`;
  }

  await renderTextWordCloud();
}

async function renderTextWordCloud() {
  if (!textWordCloudContent || !textWordCloudStatus) return;
  let src = null;
  for (const path of TEXT_WORD_CLOUD_PATHS) {
    try {
      const response = await fetch(path, { method: "HEAD", cache: "no-store" });
      if (response.ok) {
        src = path;
        break;
      }
    } catch { /* try next path */ }
  }
  if (!src) {
    textWordCloudStatus.textContent = "Word cloud appears after a local backend run writes outputs/figures/text_word_cloud.svg.";
    textWordCloudContent.innerHTML = '<div class="empty-state">Word cloud not generated yet.</div>';
    return;
  }
  textWordCloudStatus.textContent = "Word cloud loaded from generated backend output.";
  textWordCloudContent.innerHTML = `<img src="${escapeHtml(src)}" alt="Text word cloud" loading="lazy">`;
}

async function renderReportFigures() {
  if (!reportFigures) return;
  const cards = await Promise.all(REPORT_FIGURES.map(async fig => {
    let src = null;
    for (const path of fig.paths) {
      try {
        const res = await fetch(path, { method: "HEAD", cache: "no-store" });
        if (res.ok) { src = path; break; }
      } catch { /* try next path */ }
    }
    if (src) {
      return `<div class="report-figure-card">
        <div class="report-figure-img"><img src="${escapeHtml(src)}" alt="${escapeHtml(fig.name)}" loading="lazy"></div>
        <p class="report-figure-name">${escapeHtml(fig.name)}</p>
      </div>`;
    }
    return `<div class="report-figure-card report-figure-placeholder">
      <div class="report-figure-empty">Not generated yet</div>
      <p class="report-figure-name">${escapeHtml(fig.name)}</p>
    </div>`;
  }));
  const anyMissing = cards.some(c => c.includes("report-figure-placeholder"));
  reportFigures.innerHTML = cards.join("")
    + (anyMissing ? `<p class="status-text" style="grid-column:1/-1;margin-top:8px">Run the backend locally to regenerate report figures.</p>` : "");
}

async function renderReportDownloads() {
  if (!reportDownloads) return;
  const items = await Promise.all(REPORT_ARTIFACTS.map(async a => {
    let href = null;
    for (const path of a.paths) {
      try {
        const res = await fetch(path, { method: "HEAD", cache: "no-store" });
        if (res.ok) { href = path; break; }
      } catch { /* try next */ }
    }
    if (href) {
      return `<a class="download-link" href="${escapeHtml(href)}" download="${escapeHtml(a.filename)}">
        <span>${escapeHtml(a.name)}</span>
        <small>${escapeHtml(href)}</small>
      </a>`;
    }
    return `<div class="download-unavailable">
      <span>${escapeHtml(a.name)}</span>
      <small>Not generated yet</small>
    </div>`;
  }));
  const anyMissing = items.some(i => i.includes("download-unavailable"));
  reportDownloads.innerHTML = items.join("")
    + (anyMissing ? `<p class="status-text" style="grid-column:1/-1;margin-top:8px">Run the backend locally to generate these files.</p>` : "");
}

/* ── Setters (called after load) ───────────────────────────────────────── */
function setRows(rows, msg) {
  allRows = rows.map(normalizeScore).filter(r => r.ticker);
  selectedTicker = defaultTicker(allRows);
  dataStatus.textContent = msg;
  renderTickerOptions();
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

function setDataSummaryRows(rows, msg) {
  dataSummaryRows = rows;
  if (evidenceStatus) evidenceStatus.textContent = msg;
  renderEvidenceHeadline();
  renderDataSummary();
}

function setCleaningLogRows(rows) {
  cleaningLogRows = rows;
  renderCleaningLog();
}

function setSqlSummaryRows(rows) {
  sqlSummaryRows = rows;
  renderSqlEvidence();
}

function setTextualAnalysisRows(rows) {
  textualAnalysisRows = rows;
  renderTextualAnalysis();
  renderEvidenceHeadline();
  renderTextAnalytics();
}

function setTextualTickerSummaryRows(rows) {
  textualTickerSummaryRows = rows.map(normalizeTextualSummary);
  renderTextAnalytics();
}

/* ── File I/O ──────────────────────────────────────────────────────────── */
function setLatestArtifactRows(payload) {
  textModelRows = payload.text_model_comparison || [];
  tuningRows = payload.hyperparameter_tuning_results || [];
  confusionRows = payload.confusion_matrix || [];
  calibrationRows = payload.calibration_curve || [];
  featureStatsRows = payload.feature_descriptive_stats || [];
  textCoverageRows = payload.text_coverage || [];
  ldaTopicRows = payload.lda_topic_words || [];
  ldaTickerTopicRows = payload.lda_ticker_topics || [];
  renderModelDiagnostics();
  renderFeatureStats();
  renderTextCoverageLda();
}

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

function looksLikeDemoUniverse(rows) {
  const tickers = new Set(rows.map(row => String(row.ticker || "").toUpperCase()).filter(Boolean));
  if (!tickers.size) return false;
  let demoMatches = 0;
  tickers.forEach(ticker => { if (DEMO_TICKERS.has(ticker)) demoMatches += 1; });
  return demoMatches >= Math.min(4, tickers.size) && tickers.size <= DEMO_TICKERS.size;
}

function updateFileStatus(input, statusEl) {
  if (input && statusEl) statusEl.textContent = fileListLabel(input);
}

function progressStage(percent) {
  if (percent >= 100) return "Score complete";
  if (percent >= 82) return "Returning dashboard outputs";
  if (percent >= 62) return "Training and scoring";
  if (percent >= 36) return "Building crash-risk features";
  if (percent >= 16) return "Validating raw files";
  return "Uploading files";
}

function setLiveProgress(percent, hint) {
  if (!liveScoreProgress) return;
  liveProgressPercent = Math.max(0, Math.min(100, Math.round(percent)));
  liveScoreProgress.hidden = false;
  liveProgressLabel.textContent = progressStage(liveProgressPercent);
  liveProgressValue.textContent = `${liveProgressPercent}%`;
  liveProgressBar.style.width = `${liveProgressPercent}%`;
  liveProgressTrack.setAttribute("aria-valuenow", String(liveProgressPercent));
  if (hint) liveProgressHint.textContent = hint;
}

function startLiveProgress() {
  window.clearInterval(liveProgressTimer);
  setLiveProgress(7, "The hosted Python model is processing your uploaded files.");
  liveProgressTimer = window.setInterval(() => {
    const remaining = 92 - liveProgressPercent;
    if (remaining <= 0) return;
    const step = liveProgressPercent < 40 ? 7 : liveProgressPercent < 70 ? 4 : 2;
    setLiveProgress(liveProgressPercent + Math.min(step, Math.max(1, remaining)));
  }, 750);
}

function finishLiveProgress(success) {
  window.clearInterval(liveProgressTimer);
  liveProgressTimer = null;
  setLiveProgress(
    success ? 100 : Math.max(liveProgressPercent, 92),
    success
      ? "Scoring finished. The dashboard has been updated."
      : "Scoring did not finish. Check the status message above for the reason."
  );
  if (!success) liveProgressLabel.textContent = "Scoring failed";
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
    if ((rawNewsText.files || []).length) form.append("news_text", selectedFile(rawNewsText, "news_text"));
    if ((rawControversyText.files || []).length) form.append("controversy_text", selectedFile(rawControversyText, "controversy_text"));
    form.append("tune", apiTune.checked ? "true" : "false");
  } catch (err) {
    apiStatus.textContent = err.message;
    return;
  }

  setLiveScoreBusy(true);
  startLiveProgress();
  apiStatus.textContent = "Submitting files to the Python backend…";
  try {
    // ── Step 1: submit job (returns immediately with job_id) ────────────────
    const submitResp = await fetch(apiEndpoint(), { method: "POST", body: form });
    const submitData = await submitResp.json().catch(() => ({}));
    if (!submitResp.ok) {
      const detail = Array.isArray(submitData.detail) ? submitData.detail.map(d => d.msg || d.detail || d).join("; ") : submitData.detail;
      throw new Error(detail || `API request failed with status ${submitResp.status}.`);
    }

    const jobId = submitData.job_id;
    if (!jobId) {
      // Legacy path: server returned the full payload synchronously (local dev)
      _renderApiPayload(submitData);
      return;
    }

    // ── Step 2: poll GET /job/{job_id} every 4 seconds ─────────────────────
    const jobUrl = `${API_BASE_URL}/job/${jobId}`;
    const startTime = Date.now();
    while (true) {
      await new Promise(resolve => setTimeout(resolve, 4000));
      const elapsed = Math.round((Date.now() - startTime) / 1000);
      const mins = Math.floor(elapsed / 60);
      const secs = elapsed % 60;
      const elapsedStr = mins > 0 ? `${mins}m ${secs}s` : `${secs}s`;
      apiStatus.textContent = `Pipeline running… (${elapsedStr} elapsed — this typically takes 3–8 minutes)`;

      const pollResp = await fetch(jobUrl);
      const pollData = await pollResp.json().catch(() => ({}));
      if (!pollResp.ok) {
        const detail = Array.isArray(pollData.detail) ? pollData.detail.map(d => d.msg || d.detail || d).join("; ") : pollData.detail;
        throw new Error(detail || `Pipeline failed with status ${pollResp.status}.`);
      }

      if (pollData.status === "done") {
        _renderApiPayload(pollData.result);
        return;
      }
      // status === "running" → keep polling
    }
  } catch (err) {
    apiStatus.textContent = `API scoring failed: ${err.message}`;
    finishLiveProgress(false);
  } finally {
    setLiveScoreBusy(false);
  }
}

function _renderApiPayload(payload) {
  setRows(payload.scores || [], `Loaded ${payload.scores?.length || 0} scores from the hosted API.`);
  setPriceData(payload.price_history || [], payload.price_scenarios || [], "Loaded live price history and scenarios.");
  setComparisonRows(payload.model_comparison || [], "Loaded live ESG comparison.");
  setAlgoRows(payload.algorithm_comparison || [], "Loaded live algorithm comparison.");
  setImportanceRows(payload.feature_importance || [], "Loaded live feature importance.");
  setBizRows(payload.business_analysis || [], "Loaded live business analysis.");
  setQuarterBt(payload.quarter_backtest || {});
  setDataSummaryRows(payload.data_summary || [], `Loaded ${payload.data_summary?.length || 0} data summary rows from API.`);
  setCleaningLogRows(payload.cleaning_log || []);
  setSqlSummaryRows(payload.sql_summary || []);
  setTextualAnalysisRows(payload.textual_analysis || []);
  setTextualTickerSummaryRows(payload.textual_ticker_summary || []);
  setLatestArtifactRows(payload);
  renderReportFigures();
  renderReportDownloads();
  const scoreCount = payload.metadata?.score_count || 0;
  apiStatus.textContent = looksLikeDemoUniverse(payload.scores || [])
    ? `Scored ${scoreCount} rows, but they are the demo universe. Upload the 50-ticker files from data/raw or data/raw_yfinance.`
    : `Scored ${scoreCount} rows from raw uploaded data.`;
  finishLiveProgress(true);
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

async function loadDefaultQuarterBacktest() {
  try {
    const [retRows, exclRows] = await Promise.all([
      loadCsvFirst(["outputs/quarter_backtest_returns.csv", "../outputs/quarter_backtest_returns.csv"]),
      loadCsvFirst(["outputs/quarter_excluded_stocks.csv", "../outputs/quarter_excluded_stocks.csv"]),
    ]);
    if (!retRows.length || !exclRows.length) throw new Error("empty");
    const weeklySeries   = retRows.map(r => ({ date: r.date, strategy_cumulative: Number(r.strategy_cumulative), benchmark_cumulative: Number(r.benchmark_cumulative) }));
    const excl           = exclRows.map(r => ({ ticker: r.ticker, crash_probability: Number(r.crash_probability), quarter_return: r.quarter_return !== "" ? Number(r.quarter_return) : null, outcome: r.outcome }));
    const lastS  = weeklySeries[weeklySeries.length - 1].strategy_cumulative;
    const lastB  = weeklySeries[weeklySeries.length - 1].benchmark_cumulative;
    const alpha  = lastS - lastB;
    const correct = excl.filter(t => t.quarter_return != null && t.quarter_return < 0).length;
    const withData = excl.filter(t => t.quarter_return != null).length;
    setQuarterBt({
      cutoff_date: retRows[0] ? retRows[0].date : "",
      quarter_label: "Q4 2024",
      forward_weeks: weeklySeries.length,
      strategy_quarter_return: lastS - 1,
      benchmark_quarter_return: lastB - 1,
      outperformance_bps: Math.round(alpha * 10000),
      dollar_impact_quarter: Math.round(1e9 * alpha),
      dollar_impact_annualised: Math.round(1e9 * alpha * 4),
      n_excluded: excl.length,
      n_held: 50 - excl.length,
      pct_excluded_correct: withData > 0 ? correct / withData : null,
      excluded_tickers: excl,
      weekly_series: weeklySeries,
    });
  } catch { setQuarterBt(DEMO_QUARTER_BT); }
}

async function loadDefaultDataSummary() {
  try {
    const rows = await loadCsvFirst(["outputs/data_summary.csv", "../outputs/data_summary.csv"]);
    if (!rows.length) throw new Error("empty");
    setDataSummaryRows(rows, `Loaded ${rows.length} data summary rows.`);
  } catch { setDataSummaryRows(DEMO_DATA_SUMMARY, "Demo data summary is showing until you run a live upload."); }
}

async function loadDefaultCleaningLog() {
  try {
    const rows = await loadCsvFirst(["outputs/cleaning_log.csv", "../outputs/cleaning_log.csv"]);
    if (!rows.length) throw new Error("empty");
    setCleaningLogRows(rows);
  } catch { setCleaningLogRows(DEMO_CLEANING_LOG); }
}

async function loadDefaultSqlSummary() {
  try {
    const rows = await loadCsvFirst(["outputs/sql_summary.csv", "../outputs/sql_summary.csv"]);
    if (!rows.length) throw new Error("empty");
    setSqlSummaryRows(rows);
  } catch { setSqlSummaryRows(DEMO_SQL_SUMMARY); }
}

async function loadDefaultTextualAnalysis() {
  try {
    const rows = await loadCsvFirst(["outputs/textual_analysis.csv", "../outputs/textual_analysis.csv"]);
    if (!rows.length) throw new Error("empty");
    setTextualAnalysisRows(rows);
  } catch { setTextualAnalysisRows(DEMO_TEXTUAL_ANALYSIS); }
}

async function loadDefaultTextualTickerSummary() {
  try {
    const rows = await loadCsvFirst(["outputs/textual_ticker_summary.csv", "../outputs/textual_ticker_summary.csv"]);
    if (!rows.length) throw new Error("empty");
    setTextualTickerSummaryRows(rows);
  } catch { setTextualTickerSummaryRows(DEMO_TEXTUAL_TICKER_SUMMARY.filter(row => row.status === "ok")); }
}

/* ── Event listeners ───────────────────────────────────────────────────── */
async function loadDefaultLatestArtifacts() {
  const loadOptional = async paths => {
    try {
      return await loadCsvFirst(paths);
    } catch {
      return [];
    }
  };
  const [
    textModel,
    tuning,
    confusion,
    calibration,
    featureStats,
    textCoverage,
    ldaTopics,
    ldaTickerTopics,
  ] = await Promise.all([
    loadOptional(["outputs/text_model_comparison.csv", "../outputs/text_model_comparison.csv"]),
    loadOptional(["outputs/hyperparameter_tuning_results.csv", "../outputs/hyperparameter_tuning_results.csv"]),
    loadOptional(["outputs/confusion_matrix.csv", "../outputs/confusion_matrix.csv"]),
    loadOptional(["outputs/calibration_curve.csv", "../outputs/calibration_curve.csv"]),
    loadOptional(["outputs/feature_descriptive_stats.csv", "../outputs/feature_descriptive_stats.csv"]),
    loadOptional(["outputs/text_coverage.csv", "../outputs/text_coverage.csv"]),
    loadOptional(["outputs/lda_topic_words.csv", "../outputs/lda_topic_words.csv"]),
    loadOptional(["outputs/lda_ticker_topics.csv", "../outputs/lda_ticker_topics.csv"]),
  ]);
  setLatestArtifactRows({
    text_model_comparison: textModel,
    hyperparameter_tuning_results: tuning,
    confusion_matrix: confusion,
    calibration_curve: calibration,
    feature_descriptive_stats: featureStats,
    text_coverage: textCoverage,
    lda_topic_words: ldaTopics,
    lda_ticker_topics: ldaTickerTopics,
  });
}

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
rawNewsText.addEventListener("change", () => updateFileStatus(rawNewsText, rawNewsTextStatus));
rawControversyText.addEventListener("change", () => updateFileStatus(rawControversyText, rawControversyTextStatus));
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
loadDefaultQuarterBacktest();
loadDefaultDataSummary();
loadDefaultCleaningLog();
loadDefaultSqlSummary();
loadDefaultTextualAnalysis();
loadDefaultTextualTickerSummary();
loadDefaultLatestArtifacts();
renderTextAnalytics();
renderReportFigures();
renderReportDownloads();
