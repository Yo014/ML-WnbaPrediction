import React, { useState } from 'react';

export default function ExplainabilityCard({ explainability }) {
  const [mode, setMode] = useState('spread'); // 'spread' or 'total'

  if (!explainability) return null;

  const data = mode === 'spread' ? explainability.spread : explainability.total;
  if (!data) return null;

  // Find max absolute value to scale the bars
  const maxVal = Math.max(...Object.values(data).map(Math.abs), 1.0);

  return (
    <div className="glass-card" style={{ marginTop: '20px' }}>
      <div className="card-title" style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <span>Explainability Engine (SHAP)</span>
        <div style={{ display: 'flex', gap: '5px' }}>
          <button
            className={`nav-tab ${mode === 'spread' ? 'active' : ''}`}
            style={{ padding: '4px 10px', fontSize: '12px', border: 'none', cursor: 'pointer' }}
            onClick={() => setMode('spread')}
          >
            Spread
          </button>
          <button
            className={`nav-tab ${mode === 'total' ? 'active' : ''}`}
            style={{ padding: '4px 10px', fontSize: '12px', border: 'none', cursor: 'pointer' }}
            onClick={() => setMode('total')}
          >
            Total
          </button>
        </div>
      </div>

      <div style={{ padding: '15px 0 5px' }}>
        <p style={{ fontSize: '13px', color: '#cbd5e1', marginBottom: '15px' }}>
          {mode === 'spread' 
            ? "Feature contributions to the Home Team's expected point margin:" 
            : "Feature contributions to the expected total score of the game:"}
        </p>

        <div style={{ display: 'flex', flexDirection: 'column', gap: '14px' }}>
          {Object.entries(data).map(([feature, val]) => {
            // Calculate width percentage relative to maxVal (capped at 100%)
            const pct = Math.min(100, (Math.abs(val) / maxVal) * 100);
            const isPositive = val >= 0;

            return (
              <div key={feature} style={{ display: 'flex', flexDirection: 'column', gap: '4px' }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '12px' }}>
                  <span style={{ fontWeight: '500', color: '#f1f5f9' }}>{feature}</span>
                  <span style={{ 
                    fontWeight: '600', 
                    color: val > 0 ? '#10b981' : (val < 0 ? '#ef4444' : '#94a3b8') 
                  }}>
                    {val > 0 ? '+' : ''}{val.toFixed(2)} pts
                  </span>
                </div>
                
                {/* Visual Bar representation */}
                <div style={{ 
                  height: '8px', 
                  width: '100%', 
                  background: 'rgba(255, 255, 255, 0.07)', 
                  borderRadius: '4px',
                  position: 'relative',
                  overflow: 'hidden'
                }}>
                  <div style={{
                    position: 'absolute',
                    left: '50%',
                    width: '1px',
                    height: '100%',
                    backgroundColor: 'rgba(255, 255, 255, 0.25)',
                    zIndex: 2
                  }} />
                  <div style={{
                    position: 'absolute',
                    left: isPositive ? '50%' : `calc(50% - ${pct / 2}%)`,
                    right: isPositive ? `calc(50% - ${pct / 2}%)` : 'auto',
                    width: `${pct / 2}%`,
                    height: '100%',
                    backgroundColor: val > 0 ? 'rgba(16, 185, 129, 0.7)' : 'rgba(239, 68, 68, 0.7)',
                    borderRadius: '4px',
                    transition: 'all 0.3s ease'
                  }} />
                </div>
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
}
