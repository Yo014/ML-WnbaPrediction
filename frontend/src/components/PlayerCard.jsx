import React from 'react';

export default function PlayerCard({ player, onToggleInjury }) {
  const minVal = (player.min ?? 0).toFixed(1);
  const usgVal = ((player.usg_pct ?? 0) * 100).toFixed(1);
  const bpmVal = (player.bpm ?? 0).toFixed(1);
  
  const bpmClass = player.bpm >= 0 ? 'positive-bpm' : 'negative-bpm';
  const bpmSign = player.bpm > 0 ? '+' : '';
  
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
          <span className={`player-stats-pill ${bpmClass}`}>{bpmSign}{bpmVal} BPM</span>
          {injuryBadge}
        </div>
      </div>
    </div>
  );
}
