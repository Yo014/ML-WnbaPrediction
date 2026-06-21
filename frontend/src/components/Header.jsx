import React from 'react';

export default function Header({
  predictionDate,
  setPredictionDate,
  selectedCrewChief,
  setSelectedCrewChief,
  crewChiefs
}) {
  return (
    <header>
      <div className="brand-container">
        <h1>WNBA Live Matchup Spread Predictor <span className="badge">XGBoost ML</span></h1>
        
        <div className="header-controls">
          <div className="control-group">
            <label className="control-label" htmlFor="matchup-date">Matchup Date</label>
            <input 
              type="date" 
              id="matchup-date" 
              className="date-input" 
              value={predictionDate}
              onChange={(e) => setPredictionDate(e.target.value)}
            />
          </div>
          
          <div className="control-group">
            <label className="control-label" htmlFor="crew-chief">Crew Chief</label>
            <select 
              id="crew-chief" 
              className="select-input"
              value={selectedCrewChief}
              onChange={(e) => setSelectedCrewChief(e.target.value)}
            >
              <option value="None">Global Average</option>
              {crewChiefs.map((ref) => (
                <option key={ref} value={ref}>{ref}</option>
              ))}
            </select>
          </div>
        </div>
      </div>
    </header>
  );
}
