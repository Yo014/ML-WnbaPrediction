import React from 'react';

export default function PlayerCard({ player, onToggleInjury }) {
  const minVal = (player.min ?? 0).toFixed(1);
  const usgVal = ((player.usg_pct ?? 0) * 100).toFixed(1);
  
  const netRating = player.net_rating ?? player.net ?? 0.0;
  const netVal = netRating.toFixed(1);
  const netClass = netRating >= 0 ? 'positive-net' : 'negative-net';
  const netSign = netRating > 0 ? '+' : '';
  
  const pieRaw = player.pie ?? 0.0;
  const pieVal = (pieRaw <= 1.0 && pieRaw > 0 ? pieRaw * 100 : pieRaw).toFixed(1);
  
  let injuryBadge = null;
  if (player.injured) {
    const statusClass = (player.injury_status || '').toLowerCase() === 'out' ? 'out' : 'questionable';
    injuryBadge = (
      <span className={`injury-tag ${statusClass}`}>
        {player.injury_status || 'OUT'}
      </span>
    );
  }

  const handleCardClick = () => {
    onToggleInjury(player.name);
  };

  const handleCheckboxClick = (e) => {
    e.stopPropagation();
    onToggleInjury(player.name);
  };

  return (
    <div 
      className={`player-card ${player.injured ? 'checked-injured' : ''}`}
      onClick={handleCardClick}
    >
      <input 
        type="checkbox" 
        className="checkbox-custom" 
        checked={!!player.injured}
        onChange={() => {}} // Controlled input warning mitigation
        onClick={handleCheckboxClick}
      />
      <div className="player-info">
        <div className="player-name">{player.name}</div>
        <div className="player-sub">
          <span className="player-stats-pill">{minVal} MPG</span>
          <span className="player-stats-pill">{usgVal}% USG</span>
          <span className={`player-stats-pill ${netClass}`}>{netSign}{netVal} NET</span>
          <span className="player-stats-pill">{pieVal}% PIE</span>
          {injuryBadge}
        </div>
      </div>
    </div>
  );
}
