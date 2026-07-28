import React, { useState, useEffect } from 'react';

export default function KellyCalculatorCard({ predictionResult, homeTeam, awayTeam }) {
  // Bankroll state synced with localStorage (matching UpcomingBets)
  const [bankroll, setBankroll] = useState(() => {
    const saved = localStorage.getItem('wnba_initial_bankroll');
    return saved ? parseFloat(saved) : 1000;
  });

  const [kellyCap, setKellyCap] = useState(0.10); // 10% default cap
  const [customHomeOdds, setCustomHomeOdds] = useState('');
  const [customAwayOdds, setCustomAwayOdds] = useState('');
  const [customOverOdds, setCustomOverOdds] = useState('');
  const [customUnderOdds, setCustomUnderOdds] = useState('');

  // Persist bankroll updates to localStorage
  useEffect(() => {
    localStorage.setItem('wnba_initial_bankroll', bankroll);
  }, [bankroll]);

  if (!homeTeam || !awayTeam || !predictionResult) {
    return null;
  }

  // 1. Moneyline / Spread Odds & Edges
  const bookieHomeOdds = predictionResult?.odds?.bookie_home_odds ?? 1.90;
  const bookieAwayOdds = predictionResult?.odds?.bookie_away_odds ?? 1.90;

  const homeOdds = parseFloat(customHomeOdds) > 0 ? parseFloat(customHomeOdds) : bookieHomeOdds;
  const awayOdds = parseFloat(customAwayOdds) > 0 ? parseFloat(customAwayOdds) : bookieAwayOdds;

  const pHomeRaw = 1.0 / homeOdds;
  const pAwayRaw = 1.0 / awayOdds;
  const sumP = pHomeRaw + pAwayRaw;
  const homeMarketProb = sumP > 0 ? pHomeRaw / sumP : 0.5;
  const awayMarketProb = sumP > 0 ? pAwayRaw / sumP : 0.5;

  const homeModelProb = (predictionResult?.home_win_probability ?? 50.0) / 100.0;
  const awayModelProb = (predictionResult?.away_win_probability ?? 50.0) / 100.0;

  const homeEdge = homeModelProb - homeMarketProb;
  const awayEdge = awayModelProb - awayMarketProb;

  let mlPick = 'No Edge';
  let mlEdge = 0;
  let mlPrice = 0.5;
  let mlOdds = 1.90;

  if (homeEdge > 0 && homeEdge >= awayEdge) {
    mlPick = homeTeam;
    mlEdge = homeEdge;
    mlPrice = homeMarketProb;
    mlOdds = homeOdds;
  } else if (awayEdge > 0 && awayEdge >= homeEdge) {
    mlPick = awayTeam;
    mlEdge = awayEdge;
    mlPrice = awayMarketProb;
    mlOdds = awayOdds;
  }

  let mlWager = 0;
  let mlPct = 0;
  let mlWin = 0;

  if (mlPick !== 'No Edge' && mlPrice < 1.0) {
    let rawKelly = mlEdge / (1.0 - mlPrice);
    if (rawKelly > 0) {
      let kellyFraction = kellyCap * rawKelly;
      kellyFraction = Math.min(kellyCap, kellyFraction);
      mlPct = kellyFraction * 100;
      mlWager = bankroll * kellyFraction;
      mlWin = mlWager * (mlOdds - 1.0);
    }
  }

  // 2. Over / Under Totals Odds & Edges
  const overUnderLine = predictionResult?.odds?.over_under ?? 160.0;
  const overOdds = parseFloat(customOverOdds) > 0 ? parseFloat(customOverOdds) : 1.91;
  const underOdds = parseFloat(customUnderOdds) > 0 ? parseFloat(customUnderOdds) : 1.91;

  const pOverRaw = 1.0 / overOdds;
  const pUnderRaw = 1.0 / underOdds;
  const sumTotals = pOverRaw + pUnderRaw;
  const overMarketProb = sumTotals > 0 ? pOverRaw / sumTotals : 0.5;
  const underMarketProb = sumTotals > 0 ? pUnderRaw / sumTotals : 0.5;

  const overModelProb = (predictionResult?.over_probability ?? 50.0) / 100.0;
  const underModelProb = (predictionResult?.under_probability ?? 50.0) / 100.0;

  const overEdge = overModelProb - overMarketProb;
  const underEdge = underModelProb - underMarketProb;

  let totalsPick = 'No Edge';
  let totalsEdge = 0;
  let totalsPrice = 0.5;
  let totalsOdds = 1.91;

  if (overEdge > 0 && overEdge >= underEdge) {
    totalsPick = `OVER ${overUnderLine}`;
    totalsEdge = overEdge;
    totalsPrice = overMarketProb;
    totalsOdds = overOdds;
  } else if (underEdge > 0 && underEdge >= overEdge) {
    totalsPick = `UNDER ${overUnderLine}`;
    totalsEdge = underEdge;
    totalsPrice = underMarketProb;
    totalsOdds = underOdds;
  }

  let totalsWager = 0;
  let totalsPct = 0;
  let totalsWin = 0;

  if (totalsPick !== 'No Edge' && totalsPrice < 1.0) {
    let rawKellyTotal = totalsEdge / (1.0 - totalsPrice);
    if (rawKellyTotal > 0) {
      let kellyFraction = kellyCap * rawKellyTotal;
      kellyFraction = Math.min(kellyCap, kellyFraction);
      totalsPct = kellyFraction * 100;
      totalsWager = bankroll * kellyFraction;
      totalsWin = totalsWager * (totalsOdds - 1.0);
    }
  }

  return (
    <div className="glass-card kelly-calculator-card">
      <div className="card-title">
        <span>Kelly Criterion Wager Calculator</span>
        <span className="badge" style={{ background: 'rgba(99, 102, 241, 0.15)', color: 'var(--neon-indigo)' }}>
          Kelly Edge Sizing
        </span>
      </div>

      {/* Control Row: Bankroll & Kelly Cap */}
      <div className="info-grid" style={{ marginBottom: '16px' }}>
        <div className="info-item">
          <label className="info-label" htmlFor="predictor-bankroll">Available Bankroll ($)</label>
          <input
            id="predictor-bankroll"
            type="number"
            min="10"
            step="50"
            className="select-input"
            style={{ padding: '4px 8px', fontSize: '0.9rem', width: '100%', marginTop: '4px' }}
            value={bankroll}
            onChange={(e) => setBankroll(Math.max(0, parseFloat(e.target.value) || 0))}
          />
        </div>

        <div className="info-item">
          <label className="info-label" htmlFor="predictor-kelly-cap">
            Kelly Cap ({kellyCap === 1.0 ? 'No Cap' : `${Math.round(kellyCap * 100)}%`})
          </label>
          <select
            id="predictor-kelly-cap"
            className="select-input"
            style={{ padding: '4px 8px', fontSize: '0.9rem', width: '100%', marginTop: '4px' }}
            value={kellyCap}
            onChange={(e) => setKellyCap(parseFloat(e.target.value))}
          >
            <option value="0.10">1/10 (10% Cap)</option>
            <option value="0.15">15% Cap</option>
            <option value="0.20">20% Cap</option>
            <option value="0.25">25% Cap</option>
            <option value="0.30">30% Cap</option>
            <option value="0.50">50% Cap</option>
            <option value="1.00">No Cap (100%)</option>
          </select>
        </div>
      </div>

      {/* Results Section */}
      <div className="kelly-results-grid">
        {/* Moneyline / Spread Recommendation */}
        <div className="kelly-recommendation-box">
          <div className="kelly-box-header">
            <span className="kelly-market-type">Moneyline / Spread</span>
            <span className={`kelly-pick-badge ${mlWager > 0 ? 'active' : ''}`}>
              {mlPick}
            </span>
          </div>

          {mlWager > 0 ? (
            <div className="kelly-wager-details">
              <div className="kelly-wager-amount">
                ${mlWager.toFixed(2)}
                <span className="kelly-wager-pct">({mlPct.toFixed(1)}% bankroll)</span>
              </div>
              <div className="kelly-stat-row">
                <span>Model Edge:</span>
                <span style={{ color: 'var(--neon-emerald)', fontWeight: '700' }}>
                  +{(mlEdge * 100).toFixed(1)}%
                </span>
              </div>
              <div className="kelly-stat-row">
                <span>Odds:</span>
                <span>{mlOdds.toFixed(2)}</span>
              </div>
              <div className="kelly-stat-row">
                <span>Potential Win:</span>
                <span style={{ color: 'var(--neon-emerald)' }}>+${mlWin.toFixed(2)}</span>
              </div>
            </div>
          ) : (
            <div className="kelly-no-bet">No model edge detected on moneyline odds.</div>
          )}
        </div>

        {/* Over / Under Totals Recommendation */}
        <div className="kelly-recommendation-box">
          <div className="kelly-box-header">
            <span className="kelly-market-type">Over / Under Total</span>
            <span className={`kelly-pick-badge ${totalsWager > 0 ? 'active' : ''}`}>
              {totalsPick}
            </span>
          </div>

          {totalsWager > 0 ? (
            <div className="kelly-wager-details">
              <div className="kelly-wager-amount">
                ${totalsWager.toFixed(2)}
                <span className="kelly-wager-pct">({totalsPct.toFixed(1)}% bankroll)</span>
              </div>
              <div className="kelly-stat-row">
                <span>Model Edge:</span>
                <span style={{ color: 'var(--neon-emerald)', fontWeight: '700' }}>
                  +{(totalsEdge * 100).toFixed(1)}%
                </span>
              </div>
              <div className="kelly-stat-row">
                <span>Odds:</span>
                <span>{totalsOdds.toFixed(2)}</span>
              </div>
              <div className="kelly-stat-row">
                <span>Potential Win:</span>
                <span style={{ color: 'var(--neon-emerald)' }}>+${totalsWin.toFixed(2)}</span>
              </div>
            </div>
          ) : (
            <div className="kelly-no-bet">No model edge detected on game total odds.</div>
          )}
        </div>
      </div>
    </div>
  );
}
