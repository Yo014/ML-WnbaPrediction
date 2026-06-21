import React from 'react';

export default function PerformanceDetailsCard({ differentials }) {
  const floorDiff = differentials?.talent_floor_diff ?? 0.0;
  const bias = differentials?.h2h_bias ?? 50.0;
  const net5 = differentials?.net_rating_diff_5 ?? 0.0;
  const net10 = differentials?.net_rating_diff_10 ?? 0.0;

  const floorClass = floorDiff > 0 ? 'emerald' : (floorDiff < 0 ? 'rose' : '');
  const net5Class = net5 > 0 ? 'emerald' : (net5 < 0 ? 'rose' : '');
  const net10Class = net10 > 0 ? 'emerald' : (net10 < 0 ? 'rose' : '');

  return (
    <div className="glass-card">
      <div className="card-title">Matchup Performance Details</div>
      <div className="detail-section">
        <div className="info-grid">
          <div className="info-item">
            <span className="info-label">Talent Floor Diff</span>
            <span className={`info-value ${floorClass}`}>
              {floorDiff > 0 ? '+' : ''}{floorDiff.toFixed(1)}
            </span>
          </div>
          <div className="info-item">
            <span className="info-label">H2H Historical Bias</span>
            <span className="info-value">{bias.toFixed(1)}%</span>
          </div>
          <div className="info-item">
            <span className="info-label">Net Rating Diff (EMA 5)</span>
            <span className={`info-value ${net5Class}`}>
              {net5 > 0 ? '+' : ''}{net5.toFixed(1)}
            </span>
          </div>
          <div className="info-item">
            <span className="info-label">Net Rating Diff (EMA 10)</span>
            <span className={`info-value ${net10Class}`}>
              {net10 > 0 ? '+' : ''}{net10.toFixed(1)}
            </span>
          </div>
        </div>
      </div>
    </div>
  );
}
