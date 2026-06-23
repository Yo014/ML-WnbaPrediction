import React, { useState, useEffect } from 'react';

export default function UpcomingBets() {
  // Config inputs
  const [initialBankroll, setInitialBankroll] = useState(() => {
    const saved = localStorage.getItem('wnba_initial_bankroll');
    return saved !== null ? Math.max(0, parseFloat(saved)) : 100;
  });
  const [minEdgePct, setMinEdgePct] = useState(7.0); // entered as percentage, e.g. 7.0%
  const [flatWagerPct, setFlatWagerPct] = useState(12.0); // entered as percentage, e.g. 12.0%
  const [marketSource, setMarketSource] = useState('polymarket'); // 'polymarket' or 'bookie'
  const [customOdds, setCustomOdds] = useState(() => {
    const saved = localStorage.getItem('wnba_custom_odds');
    return saved !== null ? JSON.parse(saved) : {};
  });

  useEffect(() => {
    localStorage.setItem('wnba_custom_odds', JSON.stringify(customOdds));
  }, [customOdds]);

  const handleCustomOddsChange = (gameKey, teamSide, val) => {
    setCustomOdds(prev => {
      const current = prev[gameKey] || {};
      const nextOdds = {
        ...current,
        [teamSide]: val
      };

      const updated = {
        ...prev,
        [gameKey]: nextOdds
      };

      if (!nextOdds.home_odds && !nextOdds.away_odds) {
        delete updated[gameKey];
      }
      return updated;
    });
  };

  // Data and UI state
  const [bets, setBets] = useState([]);
  const [loading, setLoading] = useState(false);
  const [scraping, setScraping] = useState(false);
  const [error, setError] = useState(null);
  const [expandedGames, setExpandedGames] = useState({}); // { [gameIndex]: boolean }
  const [confirmedBets, setConfirmedBets] = useState([]);

  const fetchConfirmedBets = async () => {
    try {
      const res = await fetch('/api/confirmed_bets');
      if (!res.ok) {
        throw new Error(`Server returned status ${res.status}`);
      }
      const data = await res.json();
      setConfirmedBets(data);
    } catch (err) {
      console.error("Failed to fetch confirmed bets:", err);
    }
  };

  const handleConfirmBet = async (bet, wagerType, wagerAmount, odds, recommendedSide) => {
    setError(null);
    const gameKey = `${bet.date}_${bet.home_team_abbr}_${bet.away_team_abbr}`;
    const custom = customOdds[gameKey];
    const customHome = custom?.home_odds ? parseFloat(custom.home_odds) : null;
    const customAway = custom?.away_odds ? parseFloat(custom.away_odds) : null;

    try {
      const res = await fetch('/api/confirm_bet', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          match_date: bet.date,
          home_team: bet.home_team_abbr,
          away_team: bet.away_team_abbr,
          recommended_side: recommendedSide,
          wager_type: wagerType,
          wager_amount: wagerAmount,
          odds: odds,
          custom_home_odds: customHome,
          custom_away_odds: customAway
        })
      });
      if (!res.ok) {
        const errData = await res.json();
        throw new Error(errData.error || `Status ${res.status}`);
      }
      await fetchConfirmedBets();
    } catch (err) {
      setError(`Failed to confirm bet: ${err.message}`);
    }
  };

  const handleDeleteBet = async (bet) => {
    setError(null);
    try {
      const res = await fetch('/api/delete_bet', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          match_date: bet.date,
          home_team: bet.home_team_abbr,
          away_team: bet.away_team_abbr
        })
      });
      if (!res.ok) {
        const errData = await res.json();
        throw new Error(errData.error || `Status ${res.status}`);
      }
      await fetchConfirmedBets();
    } catch (err) {
      setError(`Failed to delete bet: ${err.message}`);
    }
  };

  const getDbAbbr = (name) => {
    if (!name) return '';
    const n = name.trim().toUpperCase();
    if (n.includes('FEVER') || n === 'IND') return 'IND';
    if (n.includes('SKY') || n === 'CHI') return 'CHI';
    if (n.includes('ACES') || n === 'LVA' || n === 'LV') return 'LVA';
    if (n.includes('LIBERTY') || n === 'NYL' || n === 'NY') return 'NYL';
    if (n.includes('STORM') || n === 'SEA') return 'SEA';
    if (n.includes('LYNX') || n === 'MIN') return 'MIN';
    if (n.includes('MERCURY') || n === 'PHO' || n === 'PHX') return 'PHO';
    if (n.includes('WINGS') || n === 'DAL') return 'DAL';
    if (n.includes('DREAM') || n === 'ATL') return 'ATL';
    if (n.includes('SUN') || n === 'CON') return 'CON';
    if (n.includes('SPARKS') || n === 'LAS' || n === 'LA') return 'LAS';
    if (n.includes('MYSTICS') || n === 'WAS') return 'WAS';
    if (n.includes('VALKYRIES') || n === 'GSV' || n === 'GS') return 'GSV';
    if (n.includes('FIRE') || n === 'PDX' || n === 'POR' || n === 'PTF') return 'PDX';
    if (n.includes('TEMPO') || n === 'TOR' || n === 'TOT') return 'TOR';
    return n;
  };

  const getTrackedBet = (bet) => {
    const betHomeAbbr = getDbAbbr(bet.home_team_abbr) || getDbAbbr(bet.home_team);
    const betAwayAbbr = getDbAbbr(bet.away_team_abbr) || getDbAbbr(bet.away_team);
    return confirmedBets.find(cb => {
      const cbHomeAbbr = getDbAbbr(cb.home_team);
      const cbAwayAbbr = getDbAbbr(cb.away_team);
      return cb.match_date === bet.date && cbHomeAbbr === betHomeAbbr && cbAwayAbbr === betAwayAbbr;
    });
  };

  useEffect(() => {
    localStorage.setItem('wnba_initial_bankroll', initialBankroll);
  }, [initialBankroll]);

  const fetchUpcomingBets = async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await fetch('/api/upcoming_bets');
      if (!res.ok) {
        throw new Error(`Server returned status ${res.status}`);
      }
      const data = await res.json();
      setBets(data);
    } catch (err) {
      setError(`Failed to fetch upcoming bets: ${err.message}`);
    } finally {
      setLoading(false);
    }
  };

  const handleScrape = async () => {
    setScraping(true);
    setError(null);
    try {
      const endpoint = marketSource === 'bookie' ? '/api/scrape_fanduel' : '/api/scrape_polymarket';
      const res = await fetch(endpoint, { method: 'POST' });
      if (!res.ok) {
        throw new Error(`Server returned status ${res.status}`);
      }
      const data = await res.json();
      setBets(data);
    } catch (err) {
      const sourceName = marketSource === 'bookie' ? 'FanDuel odds scraper' : 'live Polymarket scraper';
      setError(`Failed to run ${sourceName}: ${err.message}`);
    } finally {
      setScraping(false);
    }
  };



  useEffect(() => {
    fetchUpcomingBets();
    fetchConfirmedBets();
  }, []);

  const toggleExpand = (index) => {
    setExpandedGames(prev => ({
      ...prev,
      [index]: !prev[index]
    }));
  };

  // Calculations for bankroll & stats
  const pendingWagers = confirmedBets
    .filter(bet => bet.outcome === null)
    .reduce((sum, bet) => sum + (bet.wager_amount || 0), 0);

  const settledPnL = confirmedBets
    .filter(bet => bet.outcome !== null)
    .reduce((sum, bet) => sum + (bet.bankroll_change || 0), 0);

  const currentBankroll = initialBankroll + settledPnL - pendingWagers;

  const resolvedBets = confirmedBets.filter(bet => bet.outcome !== null);
  const wins = resolvedBets.filter(bet => bet.outcome?.toLowerCase() === 'won').length;
  const losses = resolvedBets.filter(bet => bet.outcome?.toLowerCase() === 'lost').length;
  const totalBets = resolvedBets.length;
  const winRate = totalBets > 0 ? ((wins / totalBets) * 100).toFixed(1) : '0.0';

  return (
    <div className="sim-dashboard-grid">
      {/* Live Bankroll Metrics Grid */}
      <div className="metrics-grid" style={{ gridTemplateColumns: 'repeat(auto-fit, minmax(220px, 1fr))', marginBottom: '16px', gap: '20px' }}>
        <div className="metric-card">
          <span className="metric-card-label">Available Bankroll</span>
          <span className="metric-card-value emerald">
            ${currentBankroll.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
          </span>
          <span className="metric-card-sub">
            Starting: ${initialBankroll.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
          </span>
        </div>

        <div className="metric-card">
          <span className="metric-card-label">Net P&L</span>
          <span className={`metric-card-value ${settledPnL >= 0 ? 'emerald' : 'rose'}`}>
            {settledPnL >= 0 ? '+' : ''}${settledPnL.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
          </span>
          <span className="metric-card-sub">
            Growth: {initialBankroll > 0 ? ((settledPnL / initialBankroll) * 100).toFixed(1) : '0.0'}%
          </span>
        </div>

        <div className="metric-card">
          <span className="metric-card-label">Capital at Risk</span>
          <span className="metric-card-value" style={{ color: 'var(--neon-amber)' }}>
            ${pendingWagers.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
          </span>
          <span className="metric-card-sub">
            {confirmedBets.filter(bet => bet.outcome === null).length} Pending Wagers
          </span>
        </div>

        <div className="metric-card">
          <span className="metric-card-label">Tracked Record</span>
          <span className="metric-card-value" style={{ color: 'var(--neon-indigo)' }}>
            {wins}W - {losses}L
          </span>
          <span className="metric-card-sub">
            Win Rate: {winRate}% ({totalBets} settled)
          </span>
        </div>
      </div>
      {/* Parameter Control Card */}
      <div className="glass-card" style={{ marginBottom: '8px' }}>
        <div className="card-title">
          <span>Upcoming Bets Dashboard</span>
          <span className="badge" style={{ background: 'rgba(16, 185, 129, 0.15)', borderColor: 'var(--neon-emerald)', color: 'var(--neon-emerald)' }}>
            Edge Finder
          </span>
        </div>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(180px, 1fr))', gap: '20px', alignItems: 'end' }}>

          <div className="control-group">
            <label className="control-label" htmlFor="bankroll-input">Initial Bankroll ($)</label>
            <input
              id="bankroll-input"
              type="number"
              className="select-input"
              value={initialBankroll}
              onChange={(e) => setInitialBankroll(Math.max(0, parseFloat(e.target.value) || 0))}
              disabled={loading || scraping}
              style={{ width: '100%' }}
            />
          </div>

          <div className="control-group">
            <label className="control-label" htmlFor="edge-input">Minimum Edge (%)</label>
            <input
              id="edge-input"
              type="number"
              step="0.1"
              className="select-input"
              value={minEdgePct}
              onChange={(e) => setMinEdgePct(Math.max(0, parseFloat(e.target.value) || 0))}
              disabled={loading || scraping}
              style={{ width: '100%' }}
            />
          </div>

          <div className="control-group">
            <label className="control-label" htmlFor="market-source-input">Market Odds Source</label>
            <select
              id="market-source-input"
              className="select-input"
              value={marketSource}
              onChange={(e) => setMarketSource(e.target.value)}
              disabled={loading || scraping}
              style={{ width: '100%' }}
            >
              <option value="polymarket">Polymarket Contract Prices</option>
              <option value="bookie">Traditional Bookmaker (FanDuel / ELO)</option>
            </select>
          </div>

          <div className="control-group">
            <label className="control-label" htmlFor="flat-wager-input">Flat Wager (%)</label>
            <input
              id="flat-wager-input"
              type="number"
              step="0.5"
              min="0.1"
              max="100"
              className="select-input"
              value={flatWagerPct}
              onChange={(e) => setFlatWagerPct(Math.max(0.1, parseFloat(e.target.value) || 0))}
              disabled={loading || scraping}
              style={{ width: '100%' }}
            />
          </div>

          <div className="control-group">
            <button
              onClick={handleScrape}
              className="select-input"
              style={{
                width: '100%',
                background: 'var(--neon-indigo)',
                color: '#fff',
                border: 'none',
                cursor: 'pointer',
                fontWeight: '700',
                padding: '10px 20px',
                borderRadius: '10px',
                filter: 'drop-shadow(0 2px 4px rgba(99, 102, 241, 0.3))',
                opacity: scraping ? 0.7 : 1
              }}
              disabled={loading || scraping}
            >
              {scraping ? 'Scraping & Predicting...' : marketSource === 'bookie' ? 'Scrape Live FanDuel Odds' : 'Scrape Live Polymarket'}
            </button>
          </div>
        </div>
      </div>

      {error && <div className="error-alert">{error}</div>}

      {/* Bets Table Card */}
      <div className="glass-card">
        <div className="card-title">
          <span>Active WNBA Markets & Value Bets</span>
          {bets.length > 0 && <span className="control-label" style={{ fontSize: '0.8rem' }}>{bets.length} Matches Found</span>}
        </div>

        {loading ? (
          <div style={{ textAlign: 'center', padding: '40px', color: 'var(--color-text-muted)' }}>
            Loading predictions & market odds...
          </div>
        ) : bets.length === 0 ? (
          <div style={{ textAlign: 'center', padding: '40px', color: 'var(--color-text-muted)' }}>
            No upcoming match markets found in the database.
            <br />
            <span style={{ fontSize: '0.85rem', display: 'block', marginTop: '12px' }}>
              Click <strong>{marketSource === 'bookie' ? 'Scrape Live FanDuel Odds' : 'Scrape Live Polymarket'}</strong> above to fetch current WNBA matchups.
            </span>
          </div>
        ) : (
          <div className="table-container">
            <table className="custom-table">
              <thead>
                <tr>
                  <th>Date</th>
                  <th>Matchup</th>
                  <th style={{ textAlign: 'center' }}>Model Win Prob (H/A)</th>
                  <th style={{ textAlign: 'center' }}>{marketSource === 'polymarket' ? 'Polymarket Price (H/A)' : 'Bookmaker Odds (H/A)'}</th>
                  <th style={{ textAlign: 'center' }}>Recommendation</th>
                  <th style={{ textAlign: 'right' }}>Suggested Flat Wager ({flatWagerPct}%)</th>
                  <th style={{ textAlign: 'right' }}>Suggested Kelly Wager</th>
                  <th style={{ textAlign: 'right' }}>Potential Win (Flat/Kelly)</th>
                  <th style={{ textAlign: 'right' }}>Potential Loss (Flat/Kelly)</th>
                  <th style={{ textAlign: 'center', width: '180px' }}>Track Bet</th>
                  <th style={{ width: '80px' }}></th>
                </tr>
              </thead>
              <tbody>
                {bets.map((bet, idx) => {
                  // Calculate edges and odds dynamically
                  const gameKey = `${bet.date}_${bet.home_team_abbr}_${bet.away_team_abbr}`;
                  const custom = customOdds[gameKey];
                  const trackedBet = getTrackedBet(bet);

                  const homeModelProb = bet.home_prob / 100;
                  const awayModelProb = bet.away_prob / 100;

                  let homeOdds = marketSource === 'polymarket'
                    ? (bet.home_price > 0 ? 1.0 / bet.home_price : 99.0)
                    : (bet.bookmaker ? bet.bookmaker.home_odds : 1.90);
                  let awayOdds = marketSource === 'polymarket'
                    ? (bet.away_price > 0 ? 1.0 / bet.away_price : 99.0)
                    : (bet.bookmaker ? bet.bookmaker.away_odds : 1.90);

                  if (marketSource === 'bookie' && !bet.bookmaker?.is_fanduel && custom) {
                    if (custom.home_odds) {
                      const parsed = parseFloat(custom.home_odds);
                      if (!isNaN(parsed) && parsed > 0) homeOdds = parsed;
                    }
                    if (custom.away_odds) {
                      const parsed = parseFloat(custom.away_odds);
                      if (!isNaN(parsed) && parsed > 0) awayOdds = parsed;
                    }
                  }

                  let homeMarketProb = 0.5;
                  let awayMarketProb = 0.5;

                  if (marketSource === 'polymarket') {
                    homeMarketProb = bet.home_price;
                    awayMarketProb = bet.away_price;
                  } else {
                    const hasCustom = custom && (
                      (custom.home_odds && !isNaN(parseFloat(custom.home_odds))) ||
                      (custom.away_odds && !isNaN(parseFloat(custom.away_odds)))
                    );
                    if (!bet.bookmaker?.is_fanduel && hasCustom) {
                      const p_home_raw = 1.0 / homeOdds;
                      const p_away_raw = 1.0 / awayOdds;
                      const sum_p = p_home_raw + p_away_raw;
                      homeMarketProb = sum_p > 0 ? p_home_raw / sum_p : 0.5;
                      awayMarketProb = sum_p > 0 ? p_away_raw / sum_p : 0.5;
                    } else {
                      homeMarketProb = bet.bookmaker ? bet.bookmaker.home_implied_prob / 100 : 0.5;
                      awayMarketProb = bet.bookmaker ? bet.bookmaker.away_implied_prob / 100 : 0.5;
                    }
                  }

                  const homeEdge = homeModelProb - homeMarketProb;
                  const awayEdge = awayModelProb - awayMarketProb;

                  const minEdge = minEdgePct / 100;

                  // Determine if there is a bet, and which side is favored
                  let suggestedBet = 'No Bet';
                  let activeEdge = 0;
                  let activeTeam = '';
                  let activePrice = 0;
                  let activeOdds = 1.90;
                  let isHomeBet = false;

                  if (homeEdge >= minEdge && homeEdge >= awayEdge) {
                    suggestedBet = 'Home';
                    activeEdge = homeEdge;
                    activeTeam = bet.home_team;
                    activePrice = homeMarketProb;
                    activeOdds = homeOdds;
                    isHomeBet = true;
                  } else if (awayEdge >= minEdge && awayEdge >= homeEdge) {
                    suggestedBet = 'Away';
                    activeEdge = awayEdge;
                    activeTeam = bet.away_team;
                    activePrice = awayMarketProb;
                    activeOdds = awayOdds;
                  }

                  console.log(`Edge Diagnostic [${bet.away_team_abbr}@${bet.home_team_abbr}]: homeEdge=${homeEdge.toFixed(4)}, awayEdge=${awayEdge.toFixed(4)}, minEdge=${minEdge.toFixed(4)}, marketSource=${marketSource}, suggestedBet=${suggestedBet}`);

                  // Suggested wagers
                  let flatBetSize = 0;
                  let kellyBetSize = 0;
                  let kellyPct = 0;
                  let flatWin = 0;
                  let flatLoss = 0;
                  let kellyWin = 0;
                  let kellyLoss = 0;

                  if (suggestedBet !== 'No Bet') {
                    // Flat wager is custom percentage of current bankroll
                    flatBetSize = currentBankroll * (flatWagerPct / 100);
                    flatLoss = flatBetSize;
                    flatWin = flatBetSize * (activeOdds - 1.0);

                    // Kelly sizing: f* = (p - price) / (1 - price)
                    const kellyFraction = (activeEdge) / (1.0 - activePrice);
                    if (kellyFraction > 0) {
                      kellyPct = kellyFraction * 100;
                      kellyBetSize = currentBankroll * kellyFraction;
                      kellyLoss = kellyBetSize;
                      kellyWin = kellyBetSize * (activeOdds - 1.0);
                    }
                  }

                  const isExpanded = expandedGames[idx] || false;

                  return (
                    <React.Fragment key={idx}>
                      <tr>
                        <td>{bet.date}</td>
                        <td style={{ fontWeight: '600' }}>
                          <div>
                            <span style={{ color: 'var(--color-text-muted)' }}>{bet.away_team_abbr}</span>
                            <span style={{ margin: '0 8px', color: 'var(--color-text-dim)' }}>@</span>
                            <span style={{ color: 'var(--color-text-main)' }}>{bet.home_team_abbr}</span>
                          </div>
                          {bet.has_happened && bet.home_score !== null && bet.away_score !== null && (
                            <div style={{ fontSize: '0.75rem', color: 'var(--neon-emerald)', marginTop: '4px' }}>
                              {bet.home_team_abbr} {bet.home_score} - {bet.away_score} {bet.away_team_abbr}
                            </div>
                          )}
                        </td>
                        <td style={{ textAlign: 'center' }}>
                          <span style={{ color: 'var(--neon-indigo)', fontWeight: '600' }}>{bet.home_prob}%</span>
                          <span style={{ margin: '0 6px', color: 'var(--color-text-dim)' }}>/</span>
                          <span style={{ color: 'var(--neon-purple)', fontWeight: '600' }}>{bet.away_prob}%</span>
                        </td>
                        <td style={{ textAlign: 'center' }}>
                          {marketSource === 'polymarket' ? (
                            <>
                              <span>${bet.home_price.toFixed(2)}</span>
                              <span style={{ margin: '0 6px', color: 'var(--color-text-dim)' }}>/</span>
                              <span>${bet.away_price.toFixed(2)}</span>
                            </>
                          ) : (
                            <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: '6px' }}>
                              {bet.bookmaker?.is_fanduel ? (
                                <div>
                                  <span>{bet.bookmaker?.home_odds ? bet.bookmaker.home_odds.toFixed(2) : '—'}</span>
                                  <span style={{ margin: '0 6px', color: 'var(--color-text-dim)' }}>/</span>
                                  <span>{bet.bookmaker?.away_odds ? bet.bookmaker.away_odds.toFixed(2) : '—'}</span>
                                </div>
                              ) : (
                                <div style={{ display: 'flex', alignItems: 'center', gap: '4px' }}>
                                  <input
                                    type="number"
                                    step="0.01"
                                    min="1.01"
                                    placeholder={bet.bookmaker?.home_odds ? bet.bookmaker.home_odds.toFixed(2) : '1.90'}
                                    value={customOdds[gameKey]?.home_odds || ''}
                                    onChange={(e) => handleCustomOddsChange(gameKey, 'home_odds', e.target.value)}
                                    style={{
                                      width: '64px',
                                      background: 'rgba(255, 255, 255, 0.05)',
                                      border: '1px solid var(--border-card)',
                                      color: 'var(--color-text-main)',
                                      borderRadius: '6px',
                                      padding: '4px 6px',
                                      fontSize: '0.8rem',
                                      textAlign: 'center',
                                      outline: 'none',
                                      transition: 'border-color 0.2s'
                                    }}
                                  />
                                  <span style={{ color: 'var(--color-text-dim)' }}>/</span>
                                  <input
                                    type="number"
                                    step="0.01"
                                    min="1.01"
                                    placeholder={bet.bookmaker?.away_odds ? bet.bookmaker.away_odds.toFixed(2) : '1.90'}
                                    value={customOdds[gameKey]?.away_odds || ''}
                                    onChange={(e) => handleCustomOddsChange(gameKey, 'away_odds', e.target.value)}
                                    style={{
                                      width: '64px',
                                      background: 'rgba(255, 255, 255, 0.05)',
                                      border: '1px solid var(--border-card)',
                                      color: 'var(--color-text-main)',
                                      borderRadius: '6px',
                                      padding: '4px 6px',
                                      fontSize: '0.8rem',
                                      textAlign: 'center',
                                      outline: 'none',
                                      transition: 'border-color 0.2s'
                                    }}
                                  />
                                </div>
                              )}
                              {bet.bookmaker?.is_fanduel ? (
                                <span style={{
                                  background: 'rgba(16, 185, 129, 0.15)',
                                  border: '1px solid var(--neon-emerald)',
                                  color: 'var(--neon-emerald)',
                                  padding: '2px 6px',
                                  borderRadius: '4px',
                                  fontSize: '0.65rem',
                                  fontWeight: '600',
                                  display: 'inline-block'
                                }}>
                                  [FanDuel]
                                </span>
                              ) : (
                                <span style={{
                                  background: customOdds[gameKey]?.home_odds || customOdds[gameKey]?.away_odds
                                    ? 'rgba(245, 158, 11, 0.15)'
                                    : 'rgba(255, 255, 255, 0.05)',
                                  border: customOdds[gameKey]?.home_odds || customOdds[gameKey]?.away_odds
                                    ? '1px solid var(--neon-amber)'
                                    : '1px solid var(--border-card)',
                                  color: customOdds[gameKey]?.home_odds || customOdds[gameKey]?.away_odds
                                    ? 'var(--neon-amber)'
                                    : 'var(--color-text-muted)',
                                  padding: '2px 6px',
                                  borderRadius: '4px',
                                  fontSize: '0.65rem',
                                  fontWeight: '600',
                                  display: 'inline-block'
                                }}>
                                  {customOdds[gameKey]?.home_odds || customOdds[gameKey]?.away_odds ? '[ELO - Custom]' : '[ELO]'}
                                </span>
                              )}
                            </div>
                          )}
                        </td>
                        <td style={{ textAlign: 'center' }}>
                          {suggestedBet !== 'No Bet' ? (
                            <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: '4px' }}>
                              <span style={{
                                background: 'rgba(16, 185, 129, 0.15)',
                                border: '1px solid var(--neon-emerald)',
                                color: 'var(--neon-emerald)',
                                padding: '4px 10px',
                                borderRadius: '6px',
                                fontSize: '0.75rem',
                                fontWeight: '700',
                                display: 'inline-block',
                                letterSpacing: '0.05em'
                              }}>
                                BET {isHomeBet ? bet.home_team_abbr : bet.away_team_abbr}
                              </span>
                              <span style={{ color: 'var(--neon-emerald)', fontSize: '0.75rem', fontWeight: '600' }}>
                                +{(activeEdge * 100).toFixed(1)}% Edge
                              </span>
                            </div>
                          ) : (
                            <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: '4px' }}>
                              <span style={{
                                background: 'rgba(255, 255, 255, 0.03)',
                                border: '1px solid var(--border-card)',
                                color: 'var(--color-text-dim)',
                                padding: '4px 10px',
                                borderRadius: '6px',
                                fontSize: '0.75rem',
                                fontWeight: '600',
                                display: 'inline-block'
                              }}>
                                NO BET
                              </span>
                              <span style={{ color: 'var(--color-text-muted)', fontSize: '0.7rem' }}>
                                Max Edge: {Math.max(homeEdge, awayEdge) > 0 ? `+${(Math.max(homeEdge, awayEdge) * 100).toFixed(1)}%` : `${(Math.max(homeEdge, awayEdge) * 100).toFixed(1)}%`}
                              </span>
                            </div>
                          )}
                        </td>
                        <td style={{ textAlign: 'right', fontWeight: '600' }}>
                          {suggestedBet !== 'No Bet' ? (
                            <span style={{ color: 'var(--color-text-main)' }}>
                              ${flatBetSize.toFixed(2)}
                            </span>
                          ) : (
                            <span style={{ color: 'var(--color-text-dim)' }}>—</span>
                          )}
                        </td>
                        <td style={{ textAlign: 'right', fontWeight: '700' }}>
                          {suggestedBet !== 'No Bet' && kellyBetSize > 0 ? (
                            <span style={{ color: 'var(--neon-indigo)' }}>
                              ${kellyBetSize.toFixed(2)} <span style={{ fontSize: '0.75rem', fontWeight: '500', color: 'var(--color-text-muted)' }}>({kellyPct.toFixed(1)}%)</span>
                            </span>
                          ) : (
                            <span style={{ color: 'var(--color-text-dim)' }}>—</span>
                          )}
                        </td>
                        <td style={{ textAlign: 'right', fontWeight: '600' }}>
                          {suggestedBet !== 'No Bet' ? (
                            <span style={{ color: 'var(--neon-emerald)' }}>
                              +${flatWin.toFixed(2)} <span style={{ color: 'var(--color-text-muted)', fontSize: '0.8rem', fontWeight: 'normal' }}>/</span> <span style={{ color: 'var(--neon-indigo)' }}>+${kellyWin.toFixed(2)}</span>
                            </span>
                          ) : (
                            <span style={{ color: 'var(--color-text-dim)' }}>—</span>
                          )}
                        </td>
                        <td style={{ textAlign: 'right', fontWeight: '600' }}>
                          {suggestedBet !== 'No Bet' ? (
                            <span style={{ color: 'var(--neon-rose)' }}>
                              -${flatLoss.toFixed(2)} <span style={{ color: 'var(--color-text-muted)', fontSize: '0.8rem', fontWeight: 'normal' }}>/</span> <span style={{ color: 'var(--neon-purple)' }}>-${kellyLoss.toFixed(2)}</span>
                            </span>
                          ) : (
                            <span style={{ color: 'var(--color-text-dim)' }}>—</span>
                          )}
                        </td>

                        <td style={{ textAlign: 'center' }}>
                          {trackedBet ? (
                            trackedBet.outcome === null ? (
                              <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '8px' }}>
                                <span className="badge" style={{
                                  background: 'rgba(245, 158, 11, 0.15)',
                                  borderColor: 'var(--neon-amber)',
                                  color: 'var(--neon-amber)',
                                  fontSize: '0.7rem',
                                  padding: '2px 6px'
                                }}>
                                  PENDING
                                </span>
                                <button
                                  onClick={() => handleDeleteBet(bet)}
                                  className="select-input"
                                  style={{
                                    padding: '2px 8px',
                                    fontSize: '0.7rem',
                                    cursor: 'pointer',
                                    background: 'rgba(244, 63, 94, 0.15)',
                                    border: '1px solid var(--neon-rose)',
                                    borderRadius: '6px',
                                    color: 'var(--neon-rose)'
                                  }}
                                >
                                  Cancel
                                </button>
                              </div>
                            ) : (
                              <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '8px' }}>
                                {trackedBet.outcome.toLowerCase() === 'won' ? (
                                  <>
                                    <span className="badge" style={{
                                      background: 'rgba(16, 185, 129, 0.15)',
                                      borderColor: 'var(--neon-emerald)',
                                      color: 'var(--neon-emerald)',
                                      fontSize: '0.7rem',
                                      padding: '2px 6px'
                                    }}>
                                      WON
                                    </span>
                                    <span style={{ color: 'var(--neon-emerald)', fontSize: '0.8rem', fontWeight: '600' }}>
                                      +${trackedBet.bankroll_change?.toFixed(2)}
                                    </span>
                                  </>
                                ) : (
                                  <>
                                    <span className="badge" style={{
                                      background: 'rgba(244, 63, 94, 0.15)',
                                      borderColor: 'var(--neon-rose)',
                                      color: 'var(--neon-rose)',
                                      fontSize: '0.7rem',
                                      padding: '2px 6px'
                                    }}>
                                      LOST
                                    </span>
                                    <span style={{ color: 'var(--neon-rose)', fontSize: '0.8rem', fontWeight: '600' }}>
                                      -${Math.abs(trackedBet.bankroll_change || 0)?.toFixed(2)}
                                    </span>
                                  </>
                                )}
                                <button
                                  onClick={() => handleDeleteBet(bet)}
                                  className="select-input"
                                  style={{
                                    padding: '2px 8px',
                                    fontSize: '0.7rem',
                                    cursor: 'pointer',
                                    background: 'rgba(255, 255, 255, 0.05)',
                                    border: '1px solid var(--border-card)',
                                    borderRadius: '6px',
                                    color: 'var(--color-text-muted)'
                                  }}
                                >
                                  Reset
                                </button>
                              </div>
                            )
                          ) : (
                            suggestedBet !== 'No Bet' ? (
                              <div style={{ display: 'flex', flexDirection: 'column', gap: '4px', alignItems: 'center' }}>
                                <button
                                  onClick={() => handleConfirmBet(bet, 'Flat', flatBetSize, activeOdds, suggestedBet)}
                                  className="select-input"
                                  style={{
                                    padding: '4px 8px',
                                    fontSize: '0.7rem',
                                    cursor: 'pointer',
                                    background: 'rgba(99, 102, 241, 0.15)',
                                    border: '1px solid var(--neon-indigo)',
                                    borderRadius: '6px',
                                    color: 'var(--neon-indigo)',
                                    fontWeight: '600',
                                    width: '100%',
                                    textAlign: 'center'
                                  }}
                                >
                                  Confirm Flat (${flatBetSize.toFixed(0)})
                                </button>
                                {kellyBetSize > 0 && (
                                  <button
                                    onClick={() => handleConfirmBet(bet, 'Kelly', kellyBetSize, activeOdds, suggestedBet)}
                                    className="select-input"
                                    style={{
                                      padding: '4px 8px',
                                      fontSize: '0.7rem',
                                      cursor: 'pointer',
                                      background: 'rgba(168, 85, 247, 0.15)',
                                      border: '1px solid var(--neon-purple)',
                                      borderRadius: '6px',
                                      color: 'var(--neon-purple)',
                                      fontWeight: '600',
                                      width: '100%',
                                      textAlign: 'center'
                                    }}
                                  >
                                    Confirm Kelly (${kellyBetSize.toFixed(0)})
                                  </button>
                                )}
                              </div>
                            ) : (
                              <span style={{ color: 'var(--color-text-dim)', fontSize: '0.8rem' }}>—</span>
                            )
                          )}
                        </td>

                        <td>
                          <button
                            onClick={() => toggleExpand(idx)}
                            className="select-input"
                            style={{
                              padding: '4px 10px',
                              fontSize: '0.75rem',
                              cursor: 'pointer',
                              background: isExpanded ? 'rgba(255,255,255,0.08)' : 'transparent',
                              border: '1px solid var(--border-card)',
                              borderRadius: '6px',
                              color: 'var(--color-text-main)'
                            }}
                          >
                            {isExpanded ? 'Hide Info' : 'Show Info'}
                          </button>
                        </td>
                      </tr>

                      {/* Collapsible details for health & injury impacts */}
                      {isExpanded && (
                        <tr>
                          <td colSpan="11" style={{ background: 'rgba(255, 255, 255, 0.01)', padding: '16px 24px' }}>
                            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '24px' }}>

                              {/* Home Team Health Detail */}
                              <div style={{ background: 'rgba(0, 0, 0, 0.2)', padding: '16px', borderRadius: '12px', border: '1px solid rgba(255,255,255,0.03)' }}>
                                <div style={{ display: 'flex', justifycontent: 'space-between', alignItems: 'center', marginBottom: '12px' }}>
                                  <h4 style={{ margin: 0, fontSize: '0.95rem', fontWeight: '700', color: 'var(--color-text-main)' }}>
                                    {bet.home_team} (Home)
                                  </h4>
                                  <span style={{
                                    fontSize: '0.75rem',
                                    fontWeight: '700',
                                    padding: '2px 8px',
                                    borderRadius: '10px',
                                    background: bet.home_health?.Injured_Players_Count > 0 ? 'rgba(244, 63, 94, 0.15)' : 'rgba(16, 185, 129, 0.15)',
                                    color: bet.home_health?.Injured_Players_Count > 0 ? 'var(--neon-rose)' : 'var(--neon-emerald)',
                                    border: `1px solid ${bet.home_health?.Injured_Players_Count > 0 ? 'var(--neon-rose)' : 'var(--neon-emerald)'}`
                                  }}>
                                    {bet.home_health?.Injured_Players_Count} Injured
                                  </span>
                                </div>

                                <div className="squad-health-summary" style={{ display: 'flex', gap: '10px', flexWrap: 'wrap', marginBottom: '14px', border: 'none', background: 'transparent', padding: 0 }}>
                                  <div className="health-metric-box" style={{ flex: '1 1 70px', padding: '6px 10px', background: 'rgba(255, 255, 255, 0.02)', borderRadius: '8px' }}>
                                    <span className="health-metric-label" style={{ fontSize: '0.65rem' }}>Missing USG%</span>
                                    <span className="health-metric-value" style={{ fontSize: '0.85rem' }}>{(bet.home_health?.Missing_Usage_Pct ?? 0).toFixed(1)}%</span>
                                  </div>
                                  <div className="health-metric-box" style={{ flex: '1 1 70px', padding: '6px 10px', background: 'rgba(255, 255, 255, 0.02)', borderRadius: '8px' }}>
                                    <span className="health-metric-label" style={{ fontSize: '0.65rem' }}>Missing BPM</span>
                                    <span className="health-metric-value" style={{ fontSize: '0.85rem' }}>{(bet.home_health?.Missing_BPM_Pct ?? 0).toFixed(1)}</span>
                                  </div>
                                  <div className="health-metric-box" style={{ flex: '1 1 70px', padding: '6px 10px', background: 'rgba(255, 255, 255, 0.02)', borderRadius: '8px' }}>
                                    <span className="health-metric-label" style={{ fontSize: '0.65rem' }}>Missing Min%</span>
                                    <span className="health-metric-value" style={{ fontSize: '0.85rem' }}>{(bet.home_health?.Missing_Minutes_Pct ?? 0).toFixed(1)}%</span>
                                  </div>
                                </div>

                                <h5 style={{ fontSize: '0.75rem', textTransform: 'uppercase', color: 'var(--color-text-muted)', marginBottom: '6px', fontWeight: '700' }}>Injured Roster Impact</h5>
                                {bet.home_injuries.length === 0 ? (
                                  <div style={{ fontSize: '0.8rem', color: 'var(--color-text-dim)', fontStyle: 'italic' }}>No active roster injuries.</div>
                                ) : (
                                  <div style={{ display: 'flex', flexDirection: 'column', gap: '6px' }}>
                                    {bet.home_injuries.map((player, pIdx) => (
                                      <div key={pIdx} style={{ display: 'flex', justifyContent: 'space-between', padding: '6px 10px', background: 'rgba(244, 63, 94, 0.04)', border: '1px solid rgba(244, 63, 94, 0.1)', borderRadius: '8px', fontSize: '0.8rem' }}>
                                        <span style={{ fontWeight: '600', color: 'var(--color-text-main)' }}>{player.name}</span>
                                        <span style={{ color: 'var(--neon-rose)' }}>{player.status || 'Out'} {player.expected_return ? `(Return: ${player.expected_return})` : ''}</span>
                                      </div>
                                    ))}
                                  </div>
                                )}
                              </div>

                              {/* Away Team Health Detail */}
                              <div style={{ background: 'rgba(0, 0, 0, 0.2)', padding: '16px', borderRadius: '12px', border: '1px solid rgba(255,255,255,0.03)' }}>
                                <div style={{ display: 'flex', justifycontent: 'space-between', alignItems: 'center', marginBottom: '12px' }}>
                                  <h4 style={{ margin: 0, fontSize: '0.95rem', fontWeight: '700', color: 'var(--color-text-main)' }}>
                                    {bet.away_team} (Away)
                                  </h4>
                                  <span style={{
                                    fontSize: '0.75rem',
                                    fontWeight: '700',
                                    padding: '2px 8px',
                                    borderRadius: '10px',
                                    background: bet.away_health?.Injured_Players_Count > 0 ? 'rgba(244, 63, 94, 0.15)' : 'rgba(16, 185, 129, 0.15)',
                                    color: bet.away_health?.Injured_Players_Count > 0 ? 'var(--neon-rose)' : 'var(--neon-emerald)',
                                    border: `1px solid ${bet.away_health?.Injured_Players_Count > 0 ? 'var(--neon-rose)' : 'var(--neon-emerald)'}`
                                  }}>
                                    {bet.away_health?.Injured_Players_Count} Injured
                                  </span>
                                </div>

                                <div className="squad-health-summary" style={{ display: 'flex', gap: '10px', flexWrap: 'wrap', marginBottom: '14px', border: 'none', background: 'transparent', padding: 0 }}>
                                  <div className="health-metric-box" style={{ flex: '1 1 70px', padding: '6px 10px', background: 'rgba(255, 255, 255, 0.02)', borderRadius: '8px' }}>
                                    <span className="health-metric-label" style={{ fontSize: '0.65rem' }}>Missing USG%</span>
                                    <span className="health-metric-value" style={{ fontSize: '0.85rem' }}>{(bet.away_health?.Missing_Usage_Pct ?? 0).toFixed(1)}%</span>
                                  </div>
                                  <div className="health-metric-box" style={{ flex: '1 1 70px', padding: '6px 10px', background: 'rgba(255, 255, 255, 0.02)', borderRadius: '8px' }}>
                                    <span className="health-metric-label" style={{ fontSize: '0.65rem' }}>Missing BPM</span>
                                    <span className="health-metric-value" style={{ fontSize: '0.85rem' }}>{(bet.away_health?.Missing_BPM_Pct ?? 0).toFixed(1)}</span>
                                  </div>
                                  <div className="health-metric-box" style={{ flex: '1 1 70px', padding: '6px 10px', background: 'rgba(255, 255, 255, 0.02)', borderRadius: '8px' }}>
                                    <span className="health-metric-label" style={{ fontSize: '0.65rem' }}>Missing Min%</span>
                                    <span className="health-metric-value" style={{ fontSize: '0.85rem' }}>{(bet.away_health?.Missing_Minutes_Pct ?? 0).toFixed(1)}%</span>
                                  </div>
                                </div>

                                <h5 style={{ fontSize: '0.75rem', textTransform: 'uppercase', color: 'var(--color-text-muted)', marginBottom: '6px', fontWeight: '700' }}>Injured Roster Impact</h5>
                                {bet.away_injuries.length === 0 ? (
                                  <div style={{ fontSize: '0.8rem', color: 'var(--color-text-dim)', fontStyle: 'italic' }}>No active roster injuries.</div>
                                ) : (
                                  <div style={{ display: 'flex', flexDirection: 'column', gap: '6px' }}>
                                    {bet.away_injuries.map((player, pIdx) => (
                                      <div key={pIdx} style={{ display: 'flex', justifyContent: 'space-between', padding: '6px 10px', background: 'rgba(244, 63, 94, 0.04)', border: '1px solid rgba(244, 63, 94, 0.1)', borderRadius: '8px', fontSize: '0.8rem' }}>
                                        <span style={{ fontWeight: '600', color: 'var(--color-text-main)' }}>{player.name}</span>
                                        <span style={{ color: 'var(--neon-rose)' }}>{player.status || 'Out'} {player.expected_return ? `(Return: ${player.expected_return})` : ''}</span>
                                      </div>
                                    ))}
                                  </div>
                                )}
                              </div>
                            </div>
                          </td>
                        </tr>
                      )}
                    </React.Fragment>
                  );
                })}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </div>
  );
}
