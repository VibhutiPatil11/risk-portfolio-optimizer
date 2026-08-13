/*
Shown next to the CandlestickChart when a user clicks a dashboard block.
This is the "why" panel -- shows the ML prediction + SHAP factors, leading
indicator detail, and lagging RSI that together produced the fused signal.
Matches the `composite` object returned by GET /stocks/<ticker>/details.
*/
const SIGNAL_COLOR = { BUY: "#16a34a", SELL: "#dc2626", HOLD: "#ca8a04" };

function Pill({ text, color }) {
  return (
    <span style={{
      display: "inline-block", padding: "2px 8px", borderRadius: 12,
      background: color + "22", color, fontWeight: 600, fontSize: 12,
    }}>{text}</span>
  );
}

export default function StockDetailPanel({ composite }) {
  if (!composite) return <div style={{ color: "#888" }}>No recommendation data available for this stock yet.</div>;

  const { signal, confidence, expected_return, risk_level, cvar_95, components } = composite;
  const color = SIGNAL_COLOR[signal] || "#666";

  return (
    <div style={{ fontSize: 13, lineHeight: 1.5 }}>
      <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 10 }}>
        <span style={{ fontSize: 20, fontWeight: 800, color }}>{signal}</span>
        <Pill text={`${(confidence * 100).toFixed(0)}% confidence`} color={color} />
        <Pill text={`Risk: ${risk_level}`} color="#555" />
      </div>

      <div style={{ marginBottom: 12, color: "#444" }}>
        Expected return: <b>{(expected_return * 100).toFixed(1)}%</b> · CVaR (95%): <b>{(cvar_95 * 100).toFixed(1)}%</b>
      </div>

      {/* ML component */}
      <div style={{ marginBottom: 10, padding: 10, background: "#f8f8f8", borderRadius: 6 }}>
        <div style={{ fontWeight: 700, marginBottom: 4 }}>ML Model</div>
        <div>Signal: <b>{components.ml.signal}</b> ({(components.ml.confidence * 100).toFixed(0)}% conf.)</div>
        <div style={{ marginTop: 4 }}>
          {Object.entries(components.ml.probabilities).map(([k, v]) => (
            <span key={k} style={{ marginRight: 10 }}>{k}: {(v * 100).toFixed(0)}%</span>
          ))}
        </div>
        {components.ml.top_factors && (
          <div style={{ marginTop: 6 }}>
            <div style={{ fontWeight: 600, fontSize: 12 }}>Top factors (SHAP):</div>
            {components.ml.top_factors.map(([feature, value]) => (
              <div key={feature} style={{ fontSize: 12 }}>
                {feature}: <span style={{ color: value >= 0 ? "#16a34a" : "#dc2626" }}>{value >= 0 ? "+" : ""}{value.toFixed(3)}</span>
              </div>
            ))}
          </div>
        )}
      </div>

      {/* Leading indicators */}
      <div style={{ marginBottom: 10, padding: 10, background: "#f8f8f8", borderRadius: 6 }}>
        <div style={{ fontWeight: 700, marginBottom: 4 }}>Leading Indicators</div>
        <div>Signal: <b>{components.leading.signal}</b></div>
        <div style={{ marginTop: 4, fontSize: 12 }}>
          {Object.entries(components.leading.detail).map(([k, v]) => (
            <div key={k}>{k}: {typeof v === "object" ? JSON.stringify(v) : v}</div>
          ))}
        </div>
      </div>

      {/* Lagging indicators */}
      <div style={{ padding: 10, background: "#f8f8f8", borderRadius: 6 }}>
        <div style={{ fontWeight: 700, marginBottom: 4 }}>Lagging Indicators</div>
        <div>Signal: <b>{components.lagging.signal}</b> · RSI: {components.lagging.rsi}</div>
      </div>
    </div>
  );
}
