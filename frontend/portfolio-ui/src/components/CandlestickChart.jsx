/*
Replaces the pie/bar chart on individual stock views. Renders:
  - OHLC candlesticks
  - SMA20 / SMA50 overlay lines (lagging trend indicators, directly on price)
  - RSI in a separate pane below (its own 0-100 scale, doesn't belong on price)
  - A single marker on the latest candle, colored by the fused composite signal

Expects the exact shape returned by GET /stocks/<ticker>/details:
  { candles, overlays: { sma20, sma50 }, rsi, markers, composite }

npm install lightweight-charts
*/
import { useEffect, useRef } from "react";
import {
  createChart,
  createSeriesMarkers,
  CandlestickSeries,
  LineSeries,
} from "lightweight-charts";

const SIGNAL_COLOR = { BUY: "#16a34a", SELL: "#dc2626", HOLD: "#ca8a04" };

export default function CandlestickChart({ candles, overlays, rsi, markers = [], height = 340 }) {
  const priceContainerRef = useRef(null);
  const rsiContainerRef = useRef(null);

  useEffect(() => {
    if (!priceContainerRef.current || !candles?.length) return;

    // --- Price chart: candles + SMA20/SMA50 overlays ---
    const priceChart = createChart(priceContainerRef.current, {
      height,
      layout: { background: { color: "transparent" }, textColor: "#333" },
      grid: { vertLines: { color: "#eee" }, horzLines: { color: "#eee" } },
      timeScale: { borderColor: "#ddd" },
    });

    const candleSeries = priceChart.addSeries(CandlestickSeries, {
      upColor: "#16a34a", downColor: "#dc2626", borderVisible: false,
      wickUpColor: "#16a34a", wickDownColor: "#dc2626",
    });
    candleSeries.setData(candles);

    if (overlays?.sma20?.length) {
      const sma20Series = priceChart.addSeries(LineSeries, { color: "#2563eb", lineWidth: 1, title: "SMA 20" });
      sma20Series.setData(overlays.sma20);
    }
    if (overlays?.sma50?.length) {
      const sma50Series = priceChart.addSeries(LineSeries, { color: "#9333ea", lineWidth: 1, title: "SMA 50" });
      sma50Series.setData(overlays.sma50);
    }

    if (markers.length) {
      createSeriesMarkers(candleSeries, markers.map(m => ({
        time: m.time,
        position: m.signal === "SELL" ? "aboveBar" : "belowBar",
        color: SIGNAL_COLOR[m.signal] || "#666",
        shape: m.signal === "BUY" ? "arrowUp" : m.signal === "SELL" ? "arrowDown" : "circle",
        text: m.signal,
      })));
    }

    priceChart.timeScale().fitContent();

    // --- RSI pane: separate 0-100 scale, doesn't belong on the price chart ---
    let rsiChart;
    if (rsiContainerRef.current && rsi?.length) {
      rsiChart = createChart(rsiContainerRef.current, {
        height: 110,
        layout: { background: { color: "transparent" }, textColor: "#333" },
        grid: { vertLines: { color: "#eee" }, horzLines: { color: "#eee" } },
        timeScale: { borderColor: "#ddd" },
        rightPriceScale: { autoScale: false },
      });
      const rsiSeries = rsiChart.addSeries(LineSeries, { color: "#f97316", lineWidth: 1, title: "RSI (14)" });
      rsiSeries.setData(rsi);
      rsiSeries.createPriceLine({ price: 70, color: "#dc2626", lineStyle: 2, title: "Overbought" });
      rsiSeries.createPriceLine({ price: 30, color: "#16a34a", lineStyle: 2, title: "Oversold" });
      rsiChart.timeScale().fitContent();
      priceChart.timeScale().subscribeVisibleLogicalRangeChange(range => {
        rsiChart.timeScale().setVisibleLogicalRange(range);
      });
    }

    const handleResize = () => {
      priceChart.applyOptions({ width: priceContainerRef.current.clientWidth });
      if (rsiChart) rsiChart.applyOptions({ width: rsiContainerRef.current.clientWidth });
    };
    window.addEventListener("resize", handleResize);
    handleResize();

    return () => {
      window.removeEventListener("resize", handleResize);
      priceChart.remove();
      if (rsiChart) rsiChart.remove();
    };
  }, [candles, overlays, rsi, markers, height]);

  return (
    <div>
      <div ref={priceContainerRef} style={{ width: "100%" }} />
      <div ref={rsiContainerRef} style={{ width: "100%", marginTop: 4 }} />
    </div>
  );
}
