import React, { useState, useEffect } from 'react';
import Header from './components/Header';
import MatchupSelector from './components/MatchupSelector';
import PredictionHero from './components/PredictionHero';
import BettingOddsCard from './components/BettingOddsCard';
import KellyCalculatorCard from './components/KellyCalculatorCard';
import PerformanceDetailsCard from './components/PerformanceDetailsCard';
import ExplainabilityCard from './components/ExplainabilityCard';
import RosterCard from './components/RosterCard';
import SimulationBacktester from './components/SimulationBacktester';
import UpcomingBets from './components/UpcomingBets';
function App() {
  const [activeTab, setActiveTab] = useState('predictor');
  const [teamsList, setTeamsList] = useState([]);
  const [crewChiefs, setCrewChiefs] = useState([]);

  const [homeTeam, setHomeTeam] = useState('');
  const [awayTeam, setAwayTeam] = useState('');
  const [predictionDate, setPredictionDate] = useState('2026-06-15');
  const [selectedCrewChief, setSelectedCrewChief] = useState('None');

  const [homeRoster, setHomeRoster] = useState([]);
  const [awayRoster, setAwayRoster] = useState([]);
  const [predictionResult, setPredictionResult] = useState(null);
  const [error, setError] = useState(null);
  // Initialize dropdown options
  useEffect(() => {
    async function init() {
      try {
        const [resTeams, resRefs] = await Promise.all([
          fetch('/api/teams').then(r => r.json()),
          fetch('/api/crew_chiefs').then(r => r.json())
        ]);

        setTeamsList(resTeams);
        setCrewChiefs(resRefs);

        const defHome = resTeams.includes('Las Vegas Aces') ? 'Las Vegas Aces' : resTeams[0];
        let defAway = resTeams.includes('New York Liberty') ? 'New York Liberty' : resTeams[1];
        if (defHome === defAway && resTeams.length > 1) {
          defAway = resTeams[resTeams.length - 1];
        }

        setHomeTeam(defHome || '');
        setAwayTeam(defAway || '');
      } catch (err) {
        setError("Failed to initialize dashboard parameters: " + err.message);
      }
    }
    init();
  }, []);
  const updatePrediction = async (currentHomeRoster = homeRoster, currentAwayRoster = awayRoster) => {
    if (!homeTeam || !awayTeam || homeTeam === awayTeam) return;

    setError(null);

    const homeInjuredList = currentHomeRoster.filter(p => p.injured).map(p => p.name);
    const awayInjuredList = currentAwayRoster.filter(p => p.injured).map(p => p.name);

    try {
      const response = await fetch('/predict', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          home_team: homeTeam,
          away_team: awayTeam,
          crew_chief: selectedCrewChief,
          prediction_date: predictionDate,
          home_injured_players: homeInjuredList,
          away_injured_players: awayInjuredList
        })
      });

      const result = await response.json();
      if (result.error) {
        setError(result.error);
        return;
      }
      setPredictionResult(result);
    } catch (err) {
      setError("Prediction server returned an error: " + err.message);
    }
  };
  // Sync rosters and predict on team change
  useEffect(() => {
    if (!homeTeam || !awayTeam) return;
    if (homeTeam === awayTeam) {
      setError("Home team and Away team must be different.");
      return;
    }
    setError(null);
    async function loadRostersAndPredict() {
      try {
        const [resHome, resAway] = await Promise.all([
          fetch(`/api/roster/${encodeURIComponent(homeTeam)}`).then(r => r.json()),
          fetch(`/api/roster/${encodeURIComponent(awayTeam)}`).then(r => r.json())
        ]);
        setHomeRoster(resHome);
        setAwayRoster(resAway);

        // Use local responses directly to avoid race conditions with setting state asynchronously
        const homeInjuredList = resHome.filter(p => p.injured).map(p => p.name);
        const awayInjuredList = resAway.filter(p => p.injured).map(p => p.name);

        const response = await fetch('/predict', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json'
          },
          body: JSON.stringify({
            home_team: homeTeam,
            away_team: awayTeam,
            crew_chief: selectedCrewChief,
            prediction_date: predictionDate,
            home_injured_players: homeInjuredList,
            away_injured_players: awayInjuredList
          })
        });

        const result = await response.json();
        if (result.error) {
          setError(result.error);
          return;
        }
        setPredictionResult(result);
      } catch (err) {
        setError("Error loading team rosters: " + err.message);
      }
    }
    loadRostersAndPredict();
  }, [homeTeam, awayTeam]);
  // Sync prediction on parameter changes (date, crew chief)
  useEffect(() => {
    if (homeRoster.length > 0 && awayRoster.length > 0) {
      updatePrediction(homeRoster, awayRoster);
    }
  }, [predictionDate, selectedCrewChief]);
  const handleToggleInjuryHome = (playerName) => {
    const updated = homeRoster.map(p =>
      p.name === playerName ? { ...p, injured: !p.injured } : p
    );
    setHomeRoster(updated);
    updatePrediction(updated, awayRoster);
  };
  const handleToggleInjuryAway = (playerName) => {
    const updated = awayRoster.map(p =>
      p.name === playerName ? { ...p, injured: !p.injured } : p
    );
    setAwayRoster(updated);
    updatePrediction(homeRoster, updated);
  };
  return (
    <>
      {activeTab === 'predictor' ? (
        <Header
          predictionDate={predictionDate}
          setPredictionDate={setPredictionDate}
          selectedCrewChief={selectedCrewChief}
          setSelectedCrewChief={setSelectedCrewChief}
          crewChiefs={crewChiefs}
        />
      ) : activeTab === 'simulator' ? (
        <header>
          <div className="brand-container">
            <h1>WNBA Season Simulator & Backtester <span className="badge">Simulation Mode</span></h1>
          </div>
        </header>
      ) : (
        <header>
          <div className="brand-container">
            <h1>WNBA Upcoming Match Edge Finder <span className="badge">Live Odds Mode</span></h1>
          </div>
        </header>
      )}
      <div className="nav-tabs">
        <button
          className={`nav-tab ${activeTab === 'predictor' ? 'active' : ''}`}
          onClick={() => setActiveTab('predictor')}
        >
          Matchup Predictor
        </button>
        <button
          className={`nav-tab ${activeTab === 'upcoming' ? 'active' : ''}`}
          onClick={() => setActiveTab('upcoming')}
        >
          Upcoming Bets
        </button>
        <button
          className={`nav-tab ${activeTab === 'simulator' ? 'active' : ''}`}
          onClick={() => setActiveTab('simulator')}
        >
          Season Simulator & Backtester
        </button>
      </div>
      <div style={{ display: activeTab === 'predictor' ? 'block' : 'none' }}>
        <div className="dashboard-container">
          {/* Left Column: Prediction Info & Market stats */}
          <div className="left-panel">
            {error && <div className="error-alert">{error}</div>}

            <MatchupSelector
              homeTeam={homeTeam}
              setHomeTeam={setHomeTeam}
              awayTeam={awayTeam}
              setAwayTeam={setAwayTeam}
              teamsList={teamsList}
            />

            <PredictionHero
              homeTeam={homeTeam}
              awayTeam={awayTeam}
              predictedSpread={predictionResult?.predicted_spread}
              homeProb={predictionResult?.home_win_probability}
              awayProb={predictionResult?.away_win_probability}
              predictedTotal={predictionResult?.predicted_total}
              overProb={predictionResult?.over_probability}
              underProb={predictionResult?.under_probability}
              overUnderLine={predictionResult?.odds?.over_under}
            />

            <BettingOddsCard
              odds={predictionResult?.odds}
              restDiff={predictionResult?.differentials?.rest_diff}
            />

            <KellyCalculatorCard
              predictionResult={predictionResult}
              homeTeam={homeTeam}
              awayTeam={awayTeam}
            />

            <PerformanceDetailsCard
              differentials={predictionResult?.differentials}
            />

            <ExplainabilityCard
              explainability={predictionResult?.explainability}
            />
          </div>

          {/* Right Column: Rosters & Injury Selectors */}
          <div className="right-panel">
            <div className="roster-grid">
              <RosterCard
                teamName={homeTeam}
                teamType="home"
                health={predictionResult?.home_health}
                roster={homeRoster}
                onToggleInjury={handleToggleInjuryHome}
              />

              <RosterCard
                teamName={awayTeam}
                teamType="away"
                health={predictionResult?.away_health}
                roster={awayRoster}
                onToggleInjury={handleToggleInjuryAway}
              />
            </div>
          </div>
        </div>
      </div>

      <div style={{ display: activeTab === 'simulator' ? 'block' : 'none' }}>
        <SimulationBacktester />
      </div>

      <div style={{ display: activeTab === 'upcoming' ? 'block' : 'none' }}>
        <UpcomingBets />
      </div>
    </>
  );
}
export default App;