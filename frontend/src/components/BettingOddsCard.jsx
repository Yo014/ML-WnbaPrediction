import React from 'react';

export default function BettingOddsCard({ odds, restDiff }) {
  const homeOdds = odds?.bookie_home_odds ?? 1.90;
  const awayOdds = odds?.bookie_away_odds ?? 1.90;
  const impliedProb = odds?.implied_prob_home ?? 50.0;
  const closingSpread = odds?.closing_spread ?? 0.0;
  const openingSpread = odds?.opening_spread ?? 0.0;
  const overUnder = odds?.over_under ?? 160.0;
  const rest = restDiff ?? 0;

  const formattedBookieMl = `${homeOdds.toFixed(2)} / ${awayOdds.toFixed(2)}`;
  const formattedImplied = `${impliedProb.toFixed(1)}%`;
  const formattedClosing = closingSpread > 0 ? `+${closingSpread.toFixed(1)}` : closingSpread.toFixed(1);
  const formattedOpening = openingSpread > 0 ? `+${openingSpread.toFixed(1)}` : openingSpread.toFixed(1);
  const formattedOverUnder = overUnder.toFixed(1);
  const formattedRest = `${rest > 0 ? '+' : ''}${rest} days`;

  return (
    <div className="glass-card">
      <div className="card-title">Market Odds & ELO Implied</div>
      <div className="detail-section">
        <div className="info-grid">
          <div className="info-item">
            <span className="info-label">Bookie ML Home / Away</span>
            <span className="info-value">{formattedBookieMl}</span>
          </div>
          <div className="info-item">
            <span className="info-label">Bookie Implied Home Prob</span>
            <span className="info-value">{formattedImplied}</span>
          </div>
          <div className="info-item">
            <span className="info-label">Market Closing Spread</span>
            <span className="info-value">{formattedClosing}</span>
          </div>
          <div className="info-item">
            <span className="info-label">Market Opening Spread</span>
            <span className="info-value">{formattedOpening}</span>
          </div>
          <div className="info-item">
            <span className="info-label">Over / Under Line</span>
            <span className="info-value">{formattedOverUnder}</span>
          </div>
          <div className="info-item">
            <span className="info-label">Rest Days Difference</span>
            <span className="info-value">{formattedRest}</span>
          </div>
        </div>
      </div>
    </div>
  );
}
