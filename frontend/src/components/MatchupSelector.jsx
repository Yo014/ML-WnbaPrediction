import React from 'react';

export default function MatchupSelector({
  homeTeam,
  setHomeTeam,
  awayTeam,
  setAwayTeam,
  teamsList
}) {
  return (
    <div className="glass-card matchup-selector-card">
      <div className="team-select-wrapper">
        <div className="team-select-box">
          <label className="control-label" htmlFor="home-team-select">Home Team</label>
          <select 
            id="home-team-select" 
            className="select-input"
            value={homeTeam}
            onChange={(e) => setHomeTeam(e.target.value)}
          >
            {teamsList.map((team) => (
              <option key={`home-${team}`} value={team}>{team}</option>
            ))}
          </select>
        </div>
        
        <div className="vs-divider">VS</div>
        
        <div className="team-select-box">
          <label className="control-label" htmlFor="away-team-select">Away Team</label>
          <select 
            id="away-team-select" 
            className="select-input"
            value={awayTeam}
            onChange={(e) => setAwayTeam(e.target.value)}
          >
            {teamsList.map((team) => (
              <option key={`away-${team}`} value={team}>{team}</option>
            ))}
          </select>
        </div>
      </div>
    </div>
  );
}
