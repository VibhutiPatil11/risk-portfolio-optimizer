/* Select stocks for optimization; use the info button for one-stock chart details. */
import { useState } from "react";
import StockUniverseDashboard from "../components/StockUniverseDashboard";
import CandlestickChart from "../components/CandlestickChart";
import StockDetailPanel from "../components/StockDetailPanel";

export default function UniverseDashboard({ onLogout, onBuildPortfolio }) {
  const [selectedStocks, setSelectedStocks] = useState([]);
  const [detail, setDetail] = useState(null);
  const [detailTicker, setDetailTicker] = useState("");
  const [loadingDetail, setLoadingDetail] = useState(false);
  const [detailError, setDetailError] = useState("");

  const selectedTickers = selectedStocks.map((stock) => stock.ticker);

  const handleSelectStock = (stock) => {
    setSelectedStocks((current) => (
      current.some((item) => item.ticker === stock.ticker)
        ? current.filter((item) => item.ticker !== stock.ticker)
        : [...current, stock]
    ));
  };

  const handleViewStock = (ticker) => {
    setDetailTicker(ticker);
    setDetail(null);
    setDetailError("");
    setLoadingDetail(true);

    fetch(`http://127.0.0.1:5000/stocks/${ticker}/details`)
      .then((response) => {
        if (!response.ok) throw new Error(`Couldn't load details for ${ticker}`);
        return response.json();
      })
      .then(setDetail)
      .catch((error) => setDetailError(error.message))
      .finally(() => setLoadingDetail(false));
  };

  return (
    <div style={{ padding: 20 }}>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 16 }}>
        <h1 style={{ margin: 0, fontSize: 20 }}>RiskLens - NIFTY 50</h1>
        {onLogout && <button onClick={onLogout} style={{ padding: "6px 14px", cursor: "pointer" }}>Logout</button>}
      </div>

      <p style={{ color: "#475569" }}>Click a stock card to select it for your portfolio. Use the <b>i</b> button only to inspect one stock’s chart and indicators.</p>
      <StockUniverseDashboard
        onSelectStock={handleSelectStock}
        onViewStock={handleViewStock}
        selectedTickers={selectedTickers}
      />

      <section style={{ marginTop: 20, padding: 16, background: "#eff6ff", border: "1px solid #bfdbfe", borderRadius: 8 }}>
        <h2 style={{ margin: "0 0 6px" }}>Portfolio selection ({selectedStocks.length})</h2>
        <div style={{ display: "flex", flexWrap: "wrap", gap: 8, marginBottom: 12 }}>
          {selectedStocks.map((stock) => (
            <span key={stock.ticker} style={{ padding: "4px 9px", borderRadius: 12, background: stock.color === "green" ? "#dcfce7" : stock.color === "red" ? "#fee2e2" : "#fef9c3", border: `1px solid ${stock.color === "green" ? "#16a34a" : stock.color === "red" ? "#dc2626" : "#ca8a04"}` }}>
              {stock.ticker} - {stock.signal}
            </span>
          ))}
        </div>
        <button onClick={() => onBuildPortfolio(selectedStocks)} disabled={selectedStocks.length < 2} style={{ padding: "8px 14px", cursor: selectedStocks.length < 2 ? "not-allowed" : "pointer" }}>
          Continue to amount and risk level
        </button>
        {selectedStocks.length < 2 && <span style={{ marginLeft: 10, color: "#b45309", fontSize: 13 }}>Select at least 2 stocks.</span>}
      </section>

      {detailTicker && (
        <section style={{ marginTop: 24, border: "1px solid #ddd", borderRadius: 8, padding: 16 }}>
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 12 }}>
            <h2 style={{ margin: 0 }}>{detailTicker} details</h2>
            <button onClick={() => { setDetailTicker(""); setDetail(null); }} style={{ cursor: "pointer" }}>Close</button>
          </div>
          {loadingDetail && <div>Loading {detailTicker} details...</div>}
          {detailError && <div style={{ color: "#dc2626" }}>{detailError}</div>}
          {detail && <div style={{ display: "grid", gridTemplateColumns: "2fr 1fr", gap: 20, alignItems: "start" }}>
            <CandlestickChart candles={detail.candles} overlays={detail.overlays} rsi={detail.rsi} markers={detail.markers} />
            <StockDetailPanel composite={detail.composite} />
          </div>}
        </section>
      )}
    </div>
  );
}
