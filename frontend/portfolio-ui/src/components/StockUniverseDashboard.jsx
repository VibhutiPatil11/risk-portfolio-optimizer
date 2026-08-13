/*
Shown immediately on login: all 50 NIFTY stocks, color-coded by the fused
composite signal (Step 4). Matches the exact shape returned by
GET /dashboard/stocks (Step 5):
  { last_refreshed_at, count, stocks: [{ ticker, sector, signal, color,
    score, confidence, expected_return, risk_level, ... }] }
*/
import { useEffect, useState } from "react";

const COLOR_STYLES = {
  green:  { bg: "#dcfce7", border: "#16a34a", label: "BUY" },
  yellow: { bg: "#fef9c3", border: "#ca8a04", label: "HOLD" },
  red:    { bg: "#fee2e2", border: "#dc2626", label: "SELL" },
};

export default function StockUniverseDashboard({ onSelectStock, onViewStock, selectedTickers = [] }) {
  const [stocks, setStocks] = useState([]);
  const [lastRefreshed, setLastRefreshed] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  const load = () => {
    fetch("http://127.0.0.1:5000/dashboard/stocks")
      .then(r => {
        if (!r.ok) throw new Error("Dashboard not ready yet");
        return r.json();
      })
      .then(data => {
        setStocks(data.stocks || []);
        setLastRefreshed(data.last_refreshed_at);
        setError(null);
        setLoading(false);
      })
      .catch(err => {
        setError(err.message);
        setLoading(false);
      });
  };

  useEffect(() => {
    load();
    // Poll for the scheduler's periodic recompute (Step 8); this does NOT
    // trigger a recompute itself, just re-reads whatever's cached.
    const interval = setInterval(load, 5 * 60 * 1000);
    return () => clearInterval(interval);
  }, []);

  if (loading) return <div>Loading stock universe...</div>;
  if (error) return <div style={{ color: "#dc2626" }}>{error} — try again in a moment.</div>;

  return (
    <div>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "baseline", marginBottom: 12 }}>
        <h2 style={{ margin: 0 }}>NIFTY 50 — Live Signals</h2>
        <span style={{ color: "#666", fontSize: 13 }}>
          Last refreshed: {lastRefreshed ? new Date(lastRefreshed).toLocaleTimeString() : "—"}
        </span>
      </div>

      <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fill, minmax(170px, 1fr))", gap: 10 }}>
        {stocks.map(s => {
          const style = COLOR_STYLES[s.color] || COLOR_STYLES.yellow;
          return (
            <button
              key={s.ticker}
              onClick={() => onSelectStock(s)}
              aria-pressed={selectedTickers.includes(s.ticker)}
              style={{
                background: style.bg, border: `2px solid ${style.border}`,
                borderRadius: 8, padding: "10px 12px", textAlign: "left", cursor: "pointer",
                boxShadow: selectedTickers.includes(s.ticker) ? "0 0 0 3px #2563eb" : "none",
              }}
            >
              <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
                <div style={{ fontWeight: 700 }}>{s.ticker}</div>
                <button
                  type="button"
                  title={`View ${s.ticker} chart and indicators`}
                  aria-label={`View ${s.ticker} chart and indicators`}
                  onClick={(event) => {
                    event.stopPropagation();
                    onViewStock(s.ticker);
                  }}
                  style={{ border: "1px solid #94a3b8", borderRadius: "50%", width: 22, height: 22, cursor: "pointer", background: "white", fontWeight: 700 }}
                >
                  i
                </button>
              </div>
              <div style={{ fontSize: 11, color: "#555" }}>{s.sector}</div>
              <div style={{ fontWeight: 600, color: style.border, marginTop: 4 }}>{style.label}</div>
              <div style={{ fontSize: 11, color: "#555" }}>
                Exp. return: {(s.expected_return * 100).toFixed(1)}% · Conf: {(s.confidence * 100).toFixed(0)}%
              </div>
              <div style={{ fontSize: 11, color: "#555" }}>Risk: {s.risk_level}</div>
            </button>
          );
        })}
      </div>
    </div>
  );
}
