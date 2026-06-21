import React from 'react';

export default function PredictionHero({
  homeTeam,
  awayTeam,
  predictedSpread = 0.0,
  homeProb = 50.0,
  awayProb = 50.0
}) {
  let spreadStyle = {
    color: 'var(--neon-indigo)',
    textShadow: '0 0 20px rgba(99, 102, 241, 0.35)'
  };
  let desc = 'Select teams to compute live spread.';

  if (homeTeam && awayTeam) {
    if (predictedSpread > 0) {
      spreadStyle = {
        color: 'var(--neon-emerald)',
        textShadow: '0 0 20px rgba(16, 185, 129, 0.35)'
      };
      desc = `${homeTeam} is favored to win by ${predictedSpread.toFixed(1)} points at home.`;
    } else if (predictedSpread < 0) {
      spreadStyle = {
        color: 'var(--neon-rose)',
        textShadow: '0 0 20px rgba(244, 63, 94, 0.35)'
      };
      desc = `${awayTeam} is favored to win by ${Math.abs(predictedSpread).toFixed(1)} points on the road.`;
    } else {
      spreadStyle = {
        color: 'var(--neon-indigo)',
        textShadow: '0 0 20px rgba(99, 102, 241, 0.35)'
      };
      desc = 'The matchup is projected to be a perfect draw (spread margin: 0.0).';
    }
  }

  const formattedSpread = predictedSpread > 0 ? `+${predictedSpread.toFixed(1)}` : predictedSpread.toFixed(1);

  return (
    <div className="glass-card prediction-hero-card">
      <div className="control-label">Expected Spread (Home - Away)</div>
      <div className="predicted-spread-value" style={spreadStyle}>
        {formattedSpread}
      </div>
      <div className="predicted-spread-desc">{desc}</div>
      
      <div className="probability-gauge-container">
        <div className="prob-labels">
          <div className="prob-team-name home">
            <span>{homeTeam ? homeTeam.toUpperCase() : 'HOME'}</span>
            <span className="prob-percent">{homeProb.toFixed(1)}%</span>
          </div>
          <div className="prob-team-name away">
            <span>{awayTeam ? awayTeam.toUpperCase() : 'AWAY'}</span>
            <span className="prob-percent">{awayProb.toFixed(1)}%</span>
          </div>
        </div>
        <div className="prob-bar-track">
          <div className="prob-bar-fill" style={{ width: `${homeProb}%` }}></div>
        </div>
      </div>
    </div>
  );
}
