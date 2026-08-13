import { useCallback, useEffect, useMemo, useState } from "react";
import "../styles/dashboard.css";
import {
  Brush,
  CartesianGrid,
  Cell,
  Legend,
  Line,
  LineChart,
  Pie,
  PieChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis
} from "recharts";
import TooltipIcon from "../components/TooltipIcon";

const SECTOR_COLORS = ["#22c55e", "#0ea5e9", "#f97316", "#eab308", "#ef4444", "#8b5cf6", "#14b8a6"];
const SIGNAL_LABELS = { BUY: "BUY", HOLD: "HOLD", SELL: "SELL" };
const RANGE_MAP = { "1M": 21, "3M": 63, "6M": 126, "1Y": 252 };

const metricCopy = {
  expected_return: {
    label: "Estimated yearly return",
    tooltip: "This estimates how much your portfolio may gain in a year based on recent data."
  },
  sharpe: {
    label: "Return vs Risk Score (higher = better)",
    tooltip: "This shows how much return you get for the risk you take; higher is better."
  },
  volatility: {
    label: "How much prices swing day-to-day",
    tooltip: "If your portfolio has 20% volatility, it could swing up or down by about 20% in a year; higher means more risk."
  },
  var_95: {
    label: "Possible daily loss on a bad day (95% confidence)",
    tooltip: "This estimates a daily loss level that should only be exceeded on unusually bad days."
  },
  cvar_95: {
    label: "Loss during worst market days (average)",
    tooltip: "This estimates the average loss when days are worse than the normal bad-day estimate."
  },
  max_drawdown: {
    label: "Biggest drop seen from peak to bottom",
    tooltip: "This shows the largest fall from a previous high point during the measured period."
  },
  portfolio_beta: {
    label: "How much your portfolio moves with the market (1 = same as NIFTY)",
    tooltip: "A value near 1 means your portfolio tends to move like NIFTY; higher means stronger market movement."
  },
  diversification_score: {
    label: "How spread out your investments are (higher = safer mix)",
    tooltip: "This shows how well your money is spread across sectors; higher usually means less concentration risk."
  },
  benchmark_return: {
    label: "NIFTY yearly return",
    tooltip: "This shows the estimated yearly return of the NIFTY benchmark over the same data period."
  },
  portfolio_value: {
    label: "Current estimated portfolio value",
    tooltip: "This estimates what your invested amount would be worth after applying the portfolio's recent performance."
  }
};

const RISK_HELP = {
  low: {
    title: "CONSERVATIVE",
    sub: "Prioritise stability",
    desc: "You want stability. Less profit but less chance of big losses."
  },
  medium: {
    title: "MODERATE",
    sub: "Balance risk and return",
    desc: "Some risk, some reward. Good for most people."
  },
  high: {
    title: "AGGRESSIVE",
    sub: "Prioritise growth",
    desc: "Maximum growth with larger price swings."
  }
};

const GUIDE_STEPS = [
  {
    title: "Pick your stocks",
    body: "Choose 2 or more companies from the NIFTY 50 list. The more different sectors, the safer your mix."
  },
  {
    title: "Set your investment",
    body: "Enter how much money (in ₹) you want to invest."
  },
  {
    title: "Choose your risk level",
    body: "Conservative prioritises stability, Moderate balances risk and return, and Aggressive prioritises growth while accepting larger price swings."
  },
  {
    title: "Click Optimize",
    body: "We'll run the math and tell you the best way to split your money across your chosen stocks."
  }
];

const OPTIMIZATION_METHOD_LABELS = {
  min_variance: "Minimize Risk",
  max_sharpe: "Maximize Return Score",
  max_return: "Maximize Estimated Return"
};

const formatCurrency = (value) => {
  const numeric = Number(value);
  if (Number.isNaN(numeric)) return "--";
  return new Intl.NumberFormat("en-IN", {
    style: "currency",
    currency: "INR",
    maximumFractionDigits: 0
  }).format(numeric);
};

function Dashboard({ onLogout, initialStocks = [], onBackToUniverse }) {
  const [stocks, setStocks] = useState([]);
  const [risk, setRisk] = useState("medium");
  const [investment, setInvestment] = useState("");
  const [showResults, setShowResults] = useState(false);
  const [loading, setLoading] = useState(false);
  const [selectedRange, setSelectedRange] = useState("6M");
  const [showGuide, setShowGuide] = useState(false);
  const [guideStep, setGuideStep] = useState(0);

  const [ensemblePrediction, setEnsemblePrediction] = useState(null);
  const [ensembleLoading, setEnsembleLoading] = useState(false);
  const [ensembleError, setEnsembleError] = useState("");
  const [ensembleStatus, setEnsembleStatus] = useState("Select a stock");
  const [ensembleExplanations, setEnsembleExplanations] = useState([]);
  const [ensembleFeatureCards, setEnsembleFeatureCards] = useState([]);
  const [analyzedStock, setAnalyzedStock] = useState("");

  const [metrics, setMetrics] = useState({});
  const [summary, setSummary] = useState({});
  const [signals, setSignals] = useState([]);
  const [technicalDetails, setTechnicalDetails] = useState([]);
  const [portfolioSignal, setPortfolioSignal] = useState("HOLD");
  const [allocation, setAllocation] = useState([]);
  const [sectorAllocation, setSectorAllocation] = useState([]);
  const [performanceData, setPerformanceData] = useState([]);

  const safeMetrics = metrics || {};

  useEffect(() => {
    if (localStorage.getItem("hasSeenGuide") !== "true") {
      setShowGuide(true);
    }
  }, []);

  useEffect(() => {
    if (initialStocks.length) {
      setStocks([...new Set(initialStocks.map((stock) => String(stock.ticker || stock).toUpperCase()))]);
    }
  }, [initialStocks]);

  const selectedStockSignals = useMemo(
    () => Object.fromEntries(initialStocks.map((stock) => [stock.ticker || stock, stock])),
    [initialStocks]
  );

  const filteredPerformance = useMemo(() => {
    const count = RANGE_MAP[selectedRange] || RANGE_MAP["6M"];
    return performanceData.slice(-count);
  }, [performanceData, selectedRange]);

  const selectedAiStock = useMemo(() => stocks[stocks.length - 1] || "", [stocks]);

  const runEnsemblePrediction = useCallback(async (tickerToAnalyze, portfolioStocks) => {
    if (!tickerToAnalyze) {
      setEnsemblePrediction(null);
      setEnsembleExplanations([]);
      setEnsembleFeatureCards([]);
      setAnalyzedStock("");
      setEnsembleStatus("Select a stock");
      return;
    }

    setEnsembleLoading(true);
    setEnsembleError("");
    setEnsemblePrediction(null);
    setEnsembleStatus("Calculating...");

    try {
      const response = await fetch("http://localhost:5000/ensemble/explain", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          ticker: tickerToAnalyze,
          portfolio_stocks: portfolioStocks && portfolioStocks.length ? portfolioStocks : [tickerToAnalyze]
        })
      });
      const data = await response.json();
      if (!response.ok) {
        setEnsembleError(data.error || "Failed to get prediction");
        setEnsembleStatus("Error");
      } else {
        const predictionValue = Array.isArray(data.predictions) ? data.predictions[0] : null;
        setEnsemblePrediction(predictionValue);
        setEnsembleExplanations(Array.isArray(data.explanations) ? data.explanations : []);
        setEnsembleFeatureCards(Array.isArray(data.features) ? data.features : []);
        setAnalyzedStock(data.analyzed_stock || tickerToAnalyze);
        setEnsembleStatus("Connected");
      }
    } catch (error) {
      console.error(error);
      setEnsembleError("Backend connection failed");
      setEnsembleStatus("Error");
    } finally {
      setEnsembleLoading(false);
    }
  }, []);

  useEffect(() => {
    runEnsemblePrediction(selectedAiStock, stocks);
  }, [runEnsemblePrediction, selectedAiStock, stocks]);

  const sectorByStock = useMemo(() => {
    const map = {};
    signals.forEach((item) => {
      map[item.stock] = item.sector;
    });
    return map;
  }, [signals]);

  const optimize = async () => {
    if (stocks.length < 2) {
      alert("Please add at least 2 stocks");
      return;
    }
    const investmentValue = Number(investment);
    if (!investment || Number.isNaN(investmentValue) || investmentValue <= 0) {
      alert("Enter investment amount");
      return;
    }

    setLoading(true);
    try {
      const response = await fetch("http://localhost:5000/optimize", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ stocks, investment: investmentValue, risk })
      });
      const data = await response.json();
      if (!response.ok) {
        alert(data.error || "Failed to optimize portfolio");
        setShowResults(false);
        return;
      }

      setMetrics(data.portfolio_metrics || data.metrics || {});
      setSummary(data.portfolio_summary || {});
      setSignals(Array.isArray(data.signals) ? data.signals : []);
      setTechnicalDetails(Array.isArray(data.technical_details) ? data.technical_details : []);
      setPortfolioSignal(data.portfolio_signal || "HOLD");
      setAllocation(Array.isArray(data.allocation) ? data.allocation : []);
      setSectorAllocation(Array.isArray(data.sector_allocation) ? data.sector_allocation : []);
      setPerformanceData(Array.isArray(data.performance_curve || data.performance) ? (data.performance_curve || data.performance) : []);
      setShowResults(true);
    } catch (error) {
      console.error(error);
      alert("Backend connection failed");
    } finally {
      setLoading(false);
    }
  };

  const getSignalClass = (signal) => `badge badge-${String(signal || "HOLD").toLowerCase()}`;
  const formatRsi = (rsi) => {
    const numeric = Number(rsi);
    return Number.isNaN(numeric) ? "--" : numeric.toFixed(2);
  };
  const getConfidencePercent = (confidence) => {
    const numeric = Number(confidence);
    if (Number.isNaN(numeric)) return 0;
    return Math.round(Math.min(1, Math.max(0, numeric)) * 100);
  };
  const getRsiPosition = (rsi) => {
    const numeric = Number(rsi);
    if (Number.isNaN(numeric)) return 50;
    return Math.min(100, Math.max(0, numeric));
  };
  const getSignalExplanation = (item) => {
    const rsi = Number(item.rsi);
    const rsiStr = formatRsi(item.rsi);
    const signal = String(item.signal || "HOLD").toUpperCase();
    const sma20 = Number(item.sma20);
    const sma50 = Number(item.sma50);
    const trendRising = !Number.isNaN(sma20) && !Number.isNaN(sma50) && sma20 > sma50;
    const trendFalling = !Number.isNaN(sma20) && !Number.isNaN(sma50) && sma20 < sma50;

    if (signal === "BUY") {
      if (!Number.isNaN(rsi) && rsi < 35) {
        return `RSI is ${rsiStr} — the stock is near oversold territory, which can signal a potential buying opportunity. Short-term trend is also rising.`;
      }
      if (trendRising) {
        return `Short-term trend is rising (SMA20 > SMA50), which is a positive sign. RSI is ${rsiStr} — momentum is moderate but direction looks good.`;
      }
      return `Multiple indicators point positive. RSI is ${rsiStr}. Consider this a cautious entry opportunity.`;
    }

    if (signal === "SELL") {
      if (!Number.isNaN(rsi) && rsi > 65) {
        return `RSI is ${rsiStr} — the stock is near overbought territory. Short-term trend is also falling. Consider reducing your position.`;
      }
      if (trendFalling) {
        return `Short-term trend is falling (SMA20 < SMA50). RSI is ${rsiStr}. Momentum is weakening — watch for further decline.`;
      }
      return `Multiple indicators point negative. RSI is ${rsiStr}. Reducing exposure may be prudent.`;
    }

    // HOLD — add nuance based on RSI position
    if (!Number.isNaN(rsi) && rsi < 40) {
      return `RSI is ${rsiStr} — approaching oversold territory, but the short-term trend hasn't confirmed a reversal yet. Worth watching closely.`;
    }
    if (!Number.isNaN(rsi) && rsi > 60) {
      return `RSI is ${rsiStr} — approaching overbought territory. Momentum is fading. Monitor for a potential SELL trigger.`;
    }
    return `No strong momentum signal right now. RSI is ${rsiStr} — the stock is in neutral territory. Waiting is prudent.`;
  };
  const getPortfolioSignalCounts = () => {
    const total = signals.length;
    const buy = signals.filter((item) => item.signal === "BUY").length;
    const sell = signals.filter((item) => item.signal === "SELL").length;
    const hold = signals.filter((item) => item.signal === "HOLD").length;
    return { total, buy, sell, hold };
  };
  const getPortfolioRecommendation = () => {
    const { total, buy, sell } = getPortfolioSignalCounts();
    if (!total) return "Add stocks and optimize to see a clear recommendation.";

    // Count near-signal stocks
    const nearBuy = signals.filter(
      (item) => item.signal === "HOLD" && item.rsi !== null && Number(item.rsi) < 40
    ).length;
    const nearSell = signals.filter(
      (item) => item.signal === "HOLD" && item.rsi !== null && Number(item.rsi) > 60
    ).length;

    if (buy / total >= 0.5) return "At least half of your stocks look positive. This portfolio may be worth adding to carefully.";
    if (sell / total >= 0.5) return "At least half of your stocks look weak. Consider reducing exposure or waiting.";

    if (nearBuy > 0 && nearSell === 0) {
      return `${nearBuy} stock${nearBuy > 1 ? "s are" : " is"} approaching a BUY signal (RSI near oversold). Keep watching — a trigger may be close.`;
    }
    if (nearSell > 0 && nearBuy === 0) {
      return `${nearSell} stock${nearSell > 1 ? "s are" : " is"} approaching a SELL signal (RSI near overbought). Consider setting a stop-loss.`;
    }
    if (nearBuy > 0 && nearSell > 0) {
      return `Mixed signals: ${nearBuy} stock${nearBuy > 1 ? "s" : ""} near a BUY and ${nearSell} near a SELL. Monitor both carefully.`;
    }

    return "Most of your stocks look neutral. No strong reason to buy or sell right now.";
  };
  const openGuide = () => {
    setGuideStep(0);
    setShowGuide(true);
  };
  const closeGuide = () => {
    localStorage.setItem("hasSeenGuide", "true");
    setShowGuide(false);
  };
  const goToPreviousGuideStep = () => {
    setGuideStep((current) => Math.max(0, current - 1));
  };
  const goToNextGuideStep = () => {
    setGuideStep((current) => Math.min(GUIDE_STEPS.length - 1, current + 1));
  };
  const getOptimizationMethodLabel = () => {
    const mode = summary.optimize_mode || safeMetrics.optimize_mode;
    return summary.optimization_method || safeMetrics.optimization_method || OPTIMIZATION_METHOD_LABELS[mode] || "--";
  };
  const getMetricTone = (key, value) => {
    const numeric = Number(value);
    if (Number.isNaN(numeric)) return "neutral";

    if (key === "expected_return" || key === "benchmark_return") {
      if (numeric > 15) return "good";
      if (numeric >= 5) return "moderate";
      return "bad";
    }

    if (key === "volatility") {
      if (numeric < 15) return "good";
      if (numeric <= 25) return "moderate";
      return "bad";
    }

    if (key === "sharpe") {
      if (numeric > 1) return "good";
      if (numeric >= 0.5) return "moderate";
      return "bad";
    }

    if (key === "var_95" || key === "cvar_95") {
      if (numeric < 2) return "good";
      if (numeric <= 4) return "moderate";
      return "bad";
    }

    if (key === "diversification_score") {
      if (numeric > 60) return "good";
      if (numeric >= 30) return "moderate";
      return "bad";
    }

    if (key === "max_drawdown") {
      if (numeric < 10) return "good";
      if (numeric <= 20) return "moderate";
      return "bad";
    }

    if (key === "portfolio_beta") {
      if (numeric <= 1) return "good";
      if (numeric <= 1.2) return "moderate";
      return "bad";
    }

    if (key === "portfolio_value") {
      const invested = Number(safeMetrics.investment_amount || investment);
      if (Number.isNaN(invested) || invested <= 0) return "neutral";
      if (numeric > invested) return "good";
      if (numeric === invested) return "moderate";
      return "bad";
    }

    return "neutral";
  };
  const metricCards = [
    {
      key: "portfolio_value",
      value: formatCurrency(safeMetrics.portfolio_value),
      rawValue: safeMetrics.portfolio_value
    },
    {
      key: "expected_return",
      value: `${safeMetrics.expected_return ?? safeMetrics.returns ?? "--"}%`,
      rawValue: safeMetrics.expected_return ?? safeMetrics.returns
    },
    {
      key: "sharpe",
      value: safeMetrics.sharpe ?? "--",
      rawValue: safeMetrics.sharpe
    },
    {
      key: "volatility",
      value: `${safeMetrics.volatility ?? "--"}%`,
      rawValue: safeMetrics.volatility
    },
    {
      key: "var_95",
      value: `${safeMetrics.var_95 ?? "--"}%`,
      rawValue: safeMetrics.var_95
    },
    {
      key: "cvar_95",
      value: `${safeMetrics.cvar_95 ?? "--"}%`,
      rawValue: safeMetrics.cvar_95
    },
    {
      key: "max_drawdown",
      value: `${safeMetrics.max_drawdown ?? "--"}%`,
      rawValue: safeMetrics.max_drawdown
    },
    {
      key: "portfolio_beta",
      value: safeMetrics.portfolio_beta ?? "--",
      rawValue: safeMetrics.portfolio_beta
    },
    {
      key: "diversification_score",
      value: `${safeMetrics.diversification_score ?? "--"}%`,
      rawValue: safeMetrics.diversification_score
    },
    {
      key: "benchmark_return",
      value: `${safeMetrics.benchmark_return ?? "--"}%`,
      rawValue: safeMetrics.benchmark_return
    }
  ];
  const getEnsembleStatusClass = (status) => {
    const normalized = String(status || "").toLowerCase();
    if (normalized.includes("connected")) return "connected";
    if (normalized.includes("connecting")) return "connecting";
    if (normalized.includes("error")) return "error";
    return "ready";
  };
  const getReturnSignalLevel = () => {
    const score = Number(ensemblePrediction);
    if (Number.isNaN(score)) return "pending";
    if (score > 0.05) return "high";
    if (score >= -0.05) return "medium";
    return "low";
  };
  const getPredictionLabel = () => {
    const score = Number(ensemblePrediction);
    if (Number.isNaN(score)) return "Calculating...";
    if (score > 0.15) return "Strong Positive Outlook";
    if (score > 0.05) return "Moderate Positive Outlook";
    if (score >= -0.05) return "Neutral Outlook";
    if (score >= -0.15) return "Cautious Outlook";
    return "Negative Outlook";
  };

  const getPredictionColor = () => {
    const score = Number(ensemblePrediction);
    if (Number.isNaN(score)) return "neutral";
    if (score > 0.15) return "strong-positive";
    if (score > 0.05) return "moderate-positive";
    if (score >= -0.05) return "neutral-outlook";
    if (score >= -0.15) return "cautious";
    return "negative";
  };
  const formatFeatureValue = (key, value) => {
    const numeric = Number(value);
    if (Number.isNaN(numeric)) return "--";
    if (["recent_return", "volatility", "momentum", "sector_exposure"].includes(key)) {
      return `${(numeric * 100).toFixed(2)}%`;
    }
    return numeric.toFixed(4);
  };
  const getTopExplanationFeature = () => {
    if (!ensembleExplanations.length) return null;
    return ensembleExplanations.reduce((top, item) => (
      Math.abs(Number(item.contribution || 0)) > Math.abs(Number(top.contribution || 0)) ? item : top
    ), ensembleExplanations[0]);
  };
  const getContributionWidth = (contribution) => {
    const values = ensembleExplanations.map((item) => Math.abs(Number(item.contribution || 0)));
    const maxValue = Math.max(...values, 0.000001);
    return Math.min(100, (Math.abs(Number(contribution || 0)) / maxValue) * 100);
  };
  const formatContribution = (value) => {
    const numeric = Number(value || 0);
    return `${numeric >= 0 ? "+" : ""}${numeric.toFixed(4)}`;
  };
  const topExplanationFeature = getTopExplanationFeature();
  const portfolioCounts = getPortfolioSignalCounts();
  const activeGuideStep = GUIDE_STEPS[guideStep];

  return (
    <div className="dashboard-page">
      <header className="dashboard-header">
        <h2>Quant Portfolio Control Center</h2>
        <div className="header-actions">
          <button className="guide-help-btn" onClick={openGuide} aria-label="Open first-time user guide">?</button>
          <button className="logout-btn" onClick={onLogout}>Logout</button>
        </div>
      </header>

      {showGuide && (
        <div className="guide-overlay" role="dialog" aria-modal="true" aria-labelledby="guide-title">
          <div className="guide-card">
            <div className="guide-step-count">Step {guideStep + 1} of {GUIDE_STEPS.length}</div>
            <h3 id="guide-title">{activeGuideStep.title}</h3>
            <p>{activeGuideStep.body}</p>
            <div className="guide-dots" aria-hidden="true">
              {GUIDE_STEPS.map((step, index) => (
                <span key={step.title} className={index === guideStep ? "active" : ""} />
              ))}
            </div>
            <div className="guide-actions">
              <button type="button" onClick={goToPreviousGuideStep} disabled={guideStep === 0}>
                Previous
              </button>
              {guideStep < GUIDE_STEPS.length - 1 ? (
                <button type="button" className="guide-primary-btn" onClick={goToNextGuideStep}>
                  Next
                </button>
              ) : (
                <button type="button" className="guide-primary-btn" onClick={closeGuide}>
                  Got it
                </button>
              )}
            </div>
          </div>
        </div>
      )}

      {onBackToUniverse && (
        <button onClick={onBackToUniverse} style={{ marginBottom: 12, cursor: "pointer" }}>
          Back to stock universe
        </button>
      )}

      <section className="panel">
        <div className="section-title">Portfolio Setup</div>
        <p className="ensemble-note">These stocks were selected from the NIFTY universe. Go back to the universe if you want to change the portfolio basket.</p>
        <div className="stock-tags">
          {stocks.map((s) => (
            <span key={s} style={{
              background: selectedStockSignals[s]?.color === "green" ? "#dcfce7" : selectedStockSignals[s]?.color === "red" ? "#fee2e2" : "#fef9c3",
              borderColor: selectedStockSignals[s]?.color === "green" ? "#16a34a" : selectedStockSignals[s]?.color === "red" ? "#dc2626" : "#ca8a04",
            }}>
              {s}
              {selectedStockSignals[s]?.signal ? <b style={{ marginLeft: 5 }}>({selectedStockSignals[s].signal})</b> : null}
              {sectorByStock[s] ? <em className="sector-tag">{sectorByStock[s]}</em> : null}
            </span>
          ))}
        </div>
        <input
          type="number"
          className="investment-input"
          placeholder="Investment Amount (INR)"
          value={investment}
          onChange={(e) => setInvestment(e.target.value)}
        />
        <div className="risk-buttons">
          <button className={risk === "low" ? "risk-active" : ""} onClick={() => setRisk("low")}>CONSERVATIVE</button>
          <button className={risk === "medium" ? "risk-active" : ""} onClick={() => setRisk("medium")}>MODERATE</button>
          <button className={risk === "high" ? "risk-active" : ""} onClick={() => setRisk("high")}>AGGRESSIVE</button>
        </div>
        <div className="risk-help-grid">
          {Object.entries(RISK_HELP).map(([key, details]) => (
            <div key={key} className={`risk-help-card ${risk === key ? "active" : ""}`} title={details.desc}>
              <p>{details.title}</p>
              <small>{details.sub}</small>
              <span>{details.desc}</span>
            </div>
          ))}
        </div>

        {false && <section className="ensemble-panel">
          <div className="ensemble-header">
            <div>
              <div className="section-title">AI Signal Explainer — What's driving the prediction?</div>
              <p className="ensemble-note">Select a stock and the backend calculates live market features automatically.</p>
            </div>
          </div>
          <div className={`ensemble-preview preview-${getReturnSignalLevel()}`}>
            {selectedAiStock
              ? <>Based on live data for <strong>{analyzedStock || selectedAiStock}</strong>, the AI sees a <strong>{getPredictionLabel()}</strong> return signal.</>
              : "Select a stock to calculate the AI signal from live market data."}
          </div>
          {ensembleLoading && <div className="ai-loading">Fetching market data and calculating features...</div>}
          <div className="feature-row dynamic-feature-grid">
            {ensembleFeatureCards.map(({ key, label, value, description, support }) => (
              <div key={key} className={`feature-box dynamic-feature-card support-${String(support || "HOLD").toLowerCase()}`}>
                <div className="slider-label-row">
                  <label>{label}</label>
                  <strong>{formatFeatureValue(key, value)}</strong>
                </div>
                <small>{description}</small>
                <span className={getSignalClass(support)}>Supports {support}</span>
              </div>
            ))}
          </div>
          <div className="prediction-actions ai-status-row">
            <span className={`model-badge status-${getEnsembleStatusClass(ensembleStatus)}`}>
              {ensembleStatus}
            </span>
          </div>
          {ensemblePrediction !== null && (
            <div className="prediction-result">
              <div className="prediction-main">
                <span>AI Outlook</span>
                <strong className={`prediction-label prediction-${getPredictionColor()}`}>{getPredictionLabel()}</strong>
                <small className="prediction-score">(score: {Number(ensemblePrediction).toFixed(4)})</small>
              </div>
              {topExplanationFeature && (
                <p className="prediction-summary">
                  The main reason for this prediction is {topExplanationFeature.label}.
                </p>
              )}
              <div className="contribution-chart">
                {ensembleExplanations.map(({ key, label, contribution }) => {
                  const numericContribution = Number(contribution || 0);
                  const direction = numericContribution >= 0 ? "BUY" : "SELL";
                  const width = getContributionWidth(numericContribution);

                  return (
                    <div className="contribution-row" key={key}>
                      <div className="contribution-label">
                        {label} contributed {formatContribution(numericContribution)} toward {direction}
                      </div>
                      <div className="contribution-track">
                        <span className="contribution-center" />
                        <span
                          className={`contribution-bar ${numericContribution >= 0 ? "buy" : "sell"}`}
                          style={{
                            width: `${width / 2}%`,
                            left: numericContribution >= 0 ? "50%" : `${50 - width / 2}%`
                          }}
                        />
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>
          )}
          {ensembleError && <div className="error-text">{ensembleError}</div>}
        </section>}

        <button className="optimize-btn" onClick={optimize} disabled={loading}>
          {loading ? <span className="spinner" /> : null}
          {loading ? "Optimizing..." : "Optimize Portfolio"}
        </button>
      </section>

      {showResults && (
        <>
          <section className="panel">
            <div className="section-title">Your optimized portfolio</div>
            <div className="recommendation-card">
              <p>Portfolio recommendation: <strong className={getSignalClass(summary.portfolio_signal || portfolioSignal)}>{summary.portfolio_signal || portfolioSignal}</strong></p>
              <p>{getPortfolioRecommendation()}</p>
            </div>
            <div className="metrics" style={{ marginTop: 12 }}>
              {metricCards.filter(({ key }) => ["expected_return", "volatility", "sharpe", "diversification_score"].includes(key)).map(({ key, value, rawValue }) => (
                <div className={`metric-card metric-${getMetricTone(key, rawValue)}`} key={key}>
                  <p>{metricCopy[key].label} <TooltipIcon text={metricCopy[key].tooltip} /></p>
                  <h3>{value}</h3>
                </div>
              ))}
            </div>
            <div className="summary-grid">
              <div><span>Investment Amount</span><strong>{formatCurrency(summary.investment_amount ?? safeMetrics.investment_amount ?? investment)}</strong></div>
              <div><span>Stocks Selected</span><strong>{summary.stocks_selected ?? stocks.length}</strong></div>
              <div><span>Sectors Covered</span><strong>{summary.sectors_covered ?? "--"}</strong></div>
              <div><span>Risk Preference</span><strong>{RISK_HELP[summary.risk_level || risk]?.title || "MODERATE"}</strong></div>
              <div><span>Optimization method</span><strong>{getOptimizationMethodLabel()}</strong></div>
            </div>
          </section>

          <section className="panel">
            <div className="section-title">Portfolio vs NIFTY 50</div>
            <p className="ensemble-note">This compares the estimated value of your optimized portfolio with the NIFTY 50 benchmark over the same period.</p>
            <div className="range-buttons">
              {Object.keys(RANGE_MAP).map((range) => (
                <button
                  key={range}
                  className={selectedRange === range ? "active" : ""}
                  onClick={() => setSelectedRange(range)}
                >
                  {range}
                </button>
              ))}
            </div>
            <ResponsiveContainer width="100%" height={350}>
              <LineChart data={filteredPerformance}>
                <CartesianGrid strokeDasharray="3 3" stroke="#1f2937" />
                <XAxis dataKey="date" />
                <YAxis tickFormatter={(value) => formatCurrency(value)} />
                <Tooltip formatter={(value, name) => [formatCurrency(value), name]} />
                <Legend />
                <Line type="monotone" dataKey="portfolio" name="Portfolio Value" stroke="#22c55e" strokeWidth={2} dot={false} />
                <Line type="monotone" dataKey="benchmark" name="NIFTY Benchmark" stroke="#60a5fa" strokeWidth={2} dot={false} />
                <Brush dataKey="date" height={24} stroke="#60a5fa" />
              </LineChart>
            </ResponsiveContainer>
          </section>

          <section className="panel">
            <div className="section-title">Recommended allocation</div>
            <p className="ensemble-note">Invest the shown percentage of your amount in each selected stock. Card colours match each stock’s BUY, HOLD, or SELL signal.</p>
            <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(160px, 1fr))", gap: 10, marginBottom: 18 }}>
              {allocation.map((item) => {
                const stock = selectedStockSignals[item.name];
                const color = stock?.color === "green" ? "#16a34a" : stock?.color === "red" ? "#dc2626" : "#ca8a04";
                return (
                  <div key={item.name} style={{ border: `2px solid ${color}`, borderRadius: 8, padding: 10, background: `${color}12` }}>
                    <strong>{item.name}</strong>
                    <div style={{ color, fontWeight: 700 }}>{item.value}% to invest</div>
                    <small>{stock?.signal || "HOLD"} - {item.sector}</small>
                  </div>
                );
              })}
            </div>
            <div className="allocation-grid">
              <div className="chart-card">
                <h4>Sector-wise investment</h4>
                <ResponsiveContainer width="100%" height={320}>
                  <PieChart>
                    <Tooltip formatter={(value, name) => [`${value}%`, name]} />
                    <Pie
                      data={sectorAllocation}
                      dataKey="value"
                      nameKey="name"
                      outerRadius={110}
                      label={({ name, percent }) => `${name} ${(percent * 100).toFixed(1)}%`}
                    >
                      {sectorAllocation.map((entry, index) => (
                        <Cell key={`${entry.name}-${index}`} fill={SECTOR_COLORS[index % SECTOR_COLORS.length]} />
                      ))}
                    </Pie>
                  </PieChart>
                </ResponsiveContainer>
              </div>
            </div>
          </section>

          <section className="panel">
            <details>
              <summary style={{ cursor: "pointer", fontWeight: 700, color: "#f8fafc" }}>Technical indicator details (optional)</summary>
              <p className="ensemble-note" style={{ marginTop: 10 }}>Optional evidence behind the recommendation: leading signals, lagging trend signals, India VIX regime, and the ML feature inputs.</p>
            <div className="table-wrap" style={{ marginTop: 10 }}>
              <table className="indicator-table">
                <thead>
                  <tr>
                    <th>Stock</th>
                    <th>Lagging indicators</th>
                    <th>Leading indicators</th>
                    <th>ML feature inputs</th>
                  </tr>
                </thead>
                <tbody>
                  {technicalDetails.map((item) => (
                    <tr key={item.stock}>
                      <td><strong>{item.stock}</strong><br /><em className="sector-tag">{item.sector}</em></td>
                      <td>
                        Signal: <span className={getSignalClass(item.lagging.signal)}>{item.lagging.signal}</span><br />
                        RSI: {item.lagging.rsi ?? "--"}<br />
                        SMA 20 / 50: {item.lagging.sma20 ?? "--"} / {item.lagging.sma50 ?? "--"}<br />
                        10-day momentum: {item.lagging.momentum_10d == null ? "--" : `${(Number(item.lagging.momentum_10d) * 100).toFixed(2)}%`}
                      </td>
                      <td>
                        Signal: <span className={getSignalClass(item.leading.signal)}>{item.leading.signal}</span> (score {item.leading.score})<br />
                        Stochastic: {item.leading.detail.stochastic}<br />
                        Williams %R: {item.leading.detail.williams_r}<br />
                        OBV: {item.leading.detail.obv}<br />
                        ROC: {item.leading.detail.roc}<br />
                        India VIX: {item.leading.detail.india_vix?.regime || "unknown"}
                      </td>
                      <td>
                        Recent return: {(Number(item.ml_features.recent_return || 0) * 100).toFixed(2)}%<br />
                        Volatility: {(Number(item.ml_features.volatility || 0) * 100).toFixed(2)}%<br />
                        Momentum: {(Number(item.ml_features.momentum || 0) * 100).toFixed(2)}%<br />
                        Sector exposure: {(Number(item.ml_features.sector_exposure || 0) * 100).toFixed(1)}%<br />
                        Risk score: {Number(item.ml_features.risk_score || 0).toFixed(3)}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            </details>
          </section>
        </>
      )}
    </div>
  );
}

export default Dashboard;
