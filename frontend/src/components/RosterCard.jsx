import React from 'react';
import PlayerCard from './PlayerCard';

export default function RosterCard({
  teamName,
  teamType,
  health,
  roster,
  onToggleInjury
}) {
  const missingUsg = health?.Missing_Usage_Pct ?? 0.0;
  const missingNet = health?.Missing_Net_Rating ?? health?.Missing_Net_Rating_Pct ?? health?.Missing_NET_Pct ?? health?.Missing_BPM_Pct ?? 0.0;
  const missingPie = health?.Missing_PIE ?? health?.Missing_PIE_Pct ?? 0.0;
  const missingMin = health?.Missing_Minutes_Pct ?? 0.0;
  const injuredCount = health?.Injured_Players_Count ?? 0;

  const badgeClass = teamType === 'home' ? 'team-badge home' : 'team-badge away';
  const badgeLabel = teamType === 'home' ? 'Home' : 'Away';

  const countStyle = {
    color: injuredCount > 0 ? 'var(--neon-rose)' : 'var(--color-text-main)'
  };

  return (
    <div className="glass-card roster-card">
      <div className="roster-header-info">
        <h3>{teamName || (teamType === 'home' ? 'Home Team' : 'Away Team')}</h3>
        <span className={badgeClass}>{badgeLabel}</span>
      </div>
      
      <div className="squad-health-summary">
        <div className="health-metric-box">
          <span className="health-metric-label">Missing USG%</span>
          <span className="health-metric-value">{missingUsg.toFixed(1)}%</span>
        </div>
        <div className="health-metric-box">
          <span className="health-metric-label">Missing NET</span>
          <span className="health-metric-value">{missingNet.toFixed(1)}</span>
        </div>
        <div className="health-metric-box">
          <span className="health-metric-label">Missing PIE</span>
          <span className="health-metric-value">{missingPie.toFixed(1)}%</span>
        </div>
        <div className="health-metric-box">
          <span className="health-metric-label">Missing Min%</span>
          <span className="health-metric-value">{missingMin.toFixed(1)}%</span>
        </div>
        <div className="health-metric-box">
          <span className="health-metric-label">Injured Count</span>
          <span className="health-metric-value impact" style={countStyle}>{injuredCount}</span>
        </div>
      </div>
      
      <div className="player-list">
        {roster.map((player) => (
          <PlayerCard 
            key={player.name} 
            player={player} 
            onToggleInjury={onToggleInjury} 
          />
        ))}
      </div>
    </div>
  );
}
