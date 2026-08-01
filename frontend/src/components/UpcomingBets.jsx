import React, { useState, useEffect } from 'react';

// Custom lightweight SVG Line Chart for Bankroll History over time (react-19 compatible, no dependencies)
function BankrollChart({ history, initialBankroll }) {
  const [hoveredPoint, setHoveredPoint] = useState(null);

  if (!history || history.length <= 1) {
    return <div className="text-muted" style={{ padding: '24px', textAlign: 'center' }}>No settled betting history to display.</div>;
  }

  // SVG dimensions
  const width = 1000;
  const height = 300;
  const paddingLeft = 70;
  const paddingRight = 20;
  const paddingTop = 20;
  const paddingBottom = 45;
  const chartWidth = width - paddingLeft - paddingRight;
  const chartHeight = height - paddingTop - paddingBottom;

  // Extract min/max values
  const bankrolls = history.map(d => d.bankroll);
  const minBankroll = Math.min(...bankrolls, initialBankroll);
  const maxBankroll = Math.max(...bankrolls, initialBankroll);

  // Add Y-axis padding
  const bankrollRange = maxBankroll - minBankroll;
  const yMin = Math.max(0, minBankroll - (bankrollRange * 0.1 || 100));
  const yMax = maxBankroll + (bankrollRange * 0.1 || 100);
  const yRange = yMax - yMin;
  const pointsCount = history.length;

  // Map to SVG coordinates
  const points = history.map((d, index) => {
    const x = paddingLeft + (index / (pointsCount - 1 || 1)) * chartWidth;
    const y = paddingTop + chartHeight - ((d.bankroll - yMin) / (yRange || 1)) * chartHeight;
    return { x, y, data: d, index };
  });

  // Path generator
  let pathD = '';
  if (points.length > 0) {
    pathD = `M ${points[0].x} ${points[0].y} ` + points.slice(1).map(p => `L ${p.x} ${p.y}`).join(' ');
  }

  // Area path generator (closed loop for gradient background)
  let areaD = '';
  if (points.length > 0) {
    areaD = `${pathD} L ${points[points.length - 1].x} ${paddingTop + chartHeight} L ${points[0].x} ${paddingTop + chartHeight} Z`;
  }

  // Break-even threshold line (initial bankroll)
  const breakEvenY = paddingTop + chartHeight - ((initialBankroll - yMin) / (yRange || 1)) * chartHeight;

  // Grid levels
  const gridLines = [];
  const gridCount = 5;
  for (let i = 0; i < gridCount; i++) {
    const ratio = i / (gridCount - 1);
    const value = yMin + ratio * yRange;
    const y = paddingTop + chartHeight - ratio * chartHeight;
    gridLines.push({ y, value });
  }

  // X ticks labels (dates)
  const xTicks = [];
  const tickCount = Math.min(6, pointsCount);
  if (pointsCount > 0) {
    const divisor = tickCount - 1 || 1;
    for (let i = 0; i < tickCount; i++) {
      const idx = Math.floor((i / divisor) * (pointsCount - 1));
      if (points[idx]) {
        xTicks.push(points[idx]);
      }
    }
  }

  return (
    <div style={{ position: 'relative', width: '100%' }}>
      <svg viewBox={`0 0 ${width} ${height}`} style={{ width: '100%', height: 'auto', overflow: 'visible' }}>
        <defs>
          <linearGradient id="upcoming-chart-glow" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor="#6366f1" stopOpacity="0.25" />
            <stop offset="100%" stopColor="#6366f1" stopOpacity="0.0" />
          </linearGradient>
        </defs>

        {/* Grid Lines */}
        {gridLines.map((line, i) => (
          <g key={i}>
            <line
              x1={paddingLeft}
              y1={line.y}
              x2={width - paddingRight}
              y2={line.y}
              stroke="rgba(255, 255, 255, 0.05)"
              strokeWidth="1"
              strokeDasharray="4 4"
            />
            <text
              x={paddingLeft - 10}
              y={line.y + 4}
              fill="var(--color-text-dim)"
              fontSize="10"
              fontWeight="600"
              textAnchor="end"
            >
              ${Math.round(line.value).toLocaleString()}
            </text>
          </g>
        ))}

        {/* Break-even marker line */}
        {breakEvenY >= paddingTop && breakEvenY <= paddingTop + chartHeight && (
          <g>
            <line
              x1={paddingLeft}
              y1={breakEvenY}
              x2={width - paddingRight}
              y2={breakEvenY}
              stroke="rgba(245, 158, 11, 0.25)"
              strokeWidth="1.5"
              strokeDasharray="5 5"
            />
            <text
              x={width - paddingRight}
              y={breakEvenY - 6}
              fill="var(--neon-amber)"
              fontSize="9"
              fontWeight="700"
              textAnchor="end"
            >
              Initial: ${initialBankroll.toLocaleString()}
            </text>
          </g>
        )}

        {/* Gradient fill */}
        {areaD && <path d={areaD} fill="url(#upcoming-chart-glow)" />}

        {/* The bankroll history line */}
        {pathD && (
          <path
            d={pathD}
            fill="none"
            stroke="var(--neon-indigo)"
            strokeWidth="3"
            strokeLinecap="round"
            strokeLinejoin="round"
            style={{ filter: 'drop-shadow(0px 0px 6px rgba(99, 102, 241, 0.4))' }}
          />
        )}

        {/* X-axis line */}
        <line
          x1={paddingLeft}
          y1={paddingTop + chartHeight}
          x2={width - paddingRight}
          y2={paddingTop + chartHeight}
          stroke="rgba(255, 255, 255, 0.1)"
        />

        {/* X-axis Labels */}
        {xTicks.map((p, i) => (
          <g key={i}>
            <line
              x1={p.x}
              y1={paddingTop + chartHeight}
              x2={p.x}
              y2={paddingTop + chartHeight + 6}
              stroke="rgba(255, 255, 255, 0.2)"
            />
            <text
              x={p.x}
              y={paddingTop + chartHeight + 20}
              fill="var(--color-text-dim)"
              fontSize="10"
              fontWeight="600"
              textAnchor="middle"
            >
              {p.data.date === 'Start' ? 'Start' : p.data.date.substring(5)}
            </text>
          </g>
        ))}

        {/* Interactive Circle Markers */}
        {points.map((p, i) => (
          <circle
            key={i}
            cx={p.x}
            cy={p.y}
            r={hoveredPoint?.index === p.index ? 6 : 2}
            fill={hoveredPoint?.index === p.index ? "var(--neon-purple)" : "var(--neon-indigo)"}
            stroke="var(--bg-main)"
            strokeWidth={hoveredPoint?.index === p.index ? 2 : 1}
            style={{ cursor: 'pointer', transition: 'all 0.1s ease' }}
            onMouseEnter={() => setHoveredPoint(p)}
            onMouseLeave={() => setHoveredPoint(null)}
          />
        ))}
      </svg>

      {/* Tooltip Overlay */}
      {hoveredPoint && hoveredPoint.data.date !== 'Start' && (
        <div
          className="glass-card"
          style={{
            position: 'absolute',
            left: `${(hoveredPoint.x / width) * 100}%`,
            top: `${(hoveredPoint.y / height) * 100}%`,
            transform: 'translate(-50%, -115%)',
            padding: '10px 14px',
            fontSize: '0.75rem',
            zIndex: 10,
            pointerEvents: 'none',
            border: '1px solid var(--neon-purple)',
            background: 'rgba(11, 12, 21, 0.95)',
            boxShadow: '0 6px 20px rgba(168, 85, 247, 0.3)',
            borderRadius: '8px',
            color: 'var(--color-text-main)',
            minWidth: '150px'
          }}
        >
          <div style={{ color: 'var(--color-text-dim)', fontSize: '0.7rem' }}>
            Date: {hoveredPoint.data.date}
          </div>
          <div style={{ fontWeight: '700', marginTop: '2px' }}>
            Bankroll: <span style={{ color: '#fff' }}>${hoveredPoint.data.bankroll.toLocaleString()}</span>
          </div>
          <div style={{ color: hoveredPoint.data.cumulative_profit >= 0 ? 'var(--neon-emerald)' : 'var(--neon-rose)', fontWeight: '600' }}>
            Profit: {hoveredPoint.data.cumulative_profit >= 0 ? '+' : ''}${hoveredPoint.data.cumulative_profit.toLocaleString()}
          </div>
        </div>
      )}
    </div>
  );
}

// High-precision approximation of the cumulative distribution function for a normal distribution
function normalCDF(x, mean, std) {
  const z = (x - mean) / std;
  const t = 1.0 / (1.0 + 0.2316419 * Math.abs(z));
  const d = 0.39894228 * Math.exp(-z * z / 2.0);
  const p = d * t * (0.31938153 + t * (-0.356563782 + t * (1.781477937 + t * (-1.821255978 + t * 1.330274429))));
  const val = z >= 0.0 ? 1.0 - p : p;
  return Math.round(val * 1000) / 10; // return as percentage, e.g. 52.3
}

// Helper utilities for American and Decimal odds conversion
export function parseOddsInputToDecimal(inputVal) {
  if (inputVal === null || inputVal === undefined) return null;
  const str = inputVal.toString().trim();
  if (str === '') return null;

  // Check if negative American odds (e.g. -110, -150, -270)
  if (str.startsWith('-')) {
    const val = parseFloat(str);
    if (!isNaN(val) && val < 0) {
      return 1.0 + (100.0 / Math.abs(val));
    }
  }

  // Check if positive American odds with '+' (e.g. +150, +200)
  if (str.startsWith('+')) {
    const val = parseFloat(str.substring(1));
    if (!isNaN(val) && val > 0) {
      return 1.0 + (val / 100.0);
    }
  }

  const val = parseFloat(str);
  if (isNaN(val) || val <= 0) return null;

  // If positive number >= 50 (e.g. 150, 200, 110 entered without +), treat as positive American odds
  if (val >= 50) {
    return 1.0 + (val / 100.0);
  }

  // Otherwise treat as standard decimal odds (e.g. 1.91, 2.50)
  return val;
}

export function formatOddsDisplay(decimalOdds, format = 'american') {
  if (!decimalOdds || isNaN(decimalOdds) || decimalOdds <= 1.0) return '';
  const d = parseFloat(decimalOdds);
  if (format === 'decimal') {
    return d.toFixed(2);
  }
  // American Format
  if (d >= 2.0) {
    const american = Math.round((d - 1.0) * 100);
    return `+${american}`;
  } else {
    const american = Math.round(100.0 / (d - 1.0));
    return `-${american}`;
  }
}

export function formatOddsPlaceholder(decimalVal, format) {
  if (!decimalVal || isNaN(decimalVal)) return format === 'american' ? '-110' : '1.91';
  return formatOddsDisplay(decimalVal, format);
}

export default function UpcomingBets() {
  // Config inputs
  const [initialBankroll, setInitialBankroll] = useState(() => {
    const saved = localStorage.getItem('wnba_initial_bankroll');
    return saved !== null ? Math.max(0, parseFloat(saved)) : 100;
  });
  const [bankrollAdjustment, setBankrollAdjustment] = useState(() => {
    const saved = localStorage.getItem('wnba_bankroll_adjustment');
    return saved !== null ? parseFloat(saved) || 0 : 0;
  });
  const [minEdgePct, setMinEdgePct] = useState(7.0); // entered as percentage, e.g. 7.0%
  const [flatWagerPct, setFlatWagerPct] = useState(12.0); // entered as percentage, e.g. 12.0%
  const [kellyCap, setKellyCap] = useState(0.10); // bankroll cap fraction, defaulting to 1/10
  const [marketSource, setMarketSource] = useState('polymarket'); // 'polymarket' or 'bookie'
  const [bettingMode, setBettingMode] = useState('spread'); // 'spread' or 'total'
  const [oddsFormat, setOddsFormat] = useState(() => {
    const saved = localStorage.getItem('wnba_odds_format');
    return saved || 'american';
  });
  const [customOdds, setCustomOdds] = useState(() => {
    const saved = localStorage.getItem('wnba_custom_odds');
    return saved !== null ? JSON.parse(saved) : {};
  });

  useEffect(() => {
    localStorage.setItem('wnba_custom_odds', JSON.stringify(customOdds));
  }, [customOdds]);

  useEffect(() => {
    localStorage.setItem('wnba_odds_format', oddsFormat);
  }, [oddsFormat]);

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

      if (!nextOdds.home_odds && !nextOdds.away_odds && !nextOdds.over_odds && !nextOdds.under_odds && !nextOdds.over_under && !nextOdds.poly_home_price && !nextOdds.poly_away_price) {
        delete updated[gameKey];
      }
      return updated;
    });
  };

  const handleUpdatePredictionMarketOdds = async (bet, teamSide, val) => {
    const gameKey = `${bet.date}_${bet.home_team_abbr}_${bet.away_team_abbr}`;
    const custom = customOdds[gameKey] || {};
    
    const homeVal = teamSide === 'poly_home_price' ? val : (custom.poly_home_price || '');
    const awayVal = teamSide === 'poly_away_price' ? val : (custom.poly_away_price || '');
    
    try {
      const res = await fetch('/api/update_prediction_market_odds', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          match_date: bet.date,
          home_team: bet.home_team_abbr,
          away_team: bet.away_team_abbr,
          home_yes_price: homeVal !== '' ? parseFloat(homeVal) : null,
          away_yes_price: awayVal !== '' ? parseFloat(awayVal) : null
        })
      });
      
      if (!res.ok) {
        const errData = await res.json();
        throw new Error(errData.error || `Status ${res.status}`);
      }
      
      await fetchUpcomingBets(true);
    } catch (err) {
      setError(`Failed to update prediction market odds: ${err.message}`);
    }
  };

  // Data and UI state
  const [bets, setBets] = useState([]);
  const [loading, setLoading] = useState(false);
  const [scraping, setScraping] = useState(false);
  const [error, setError] = useState(null);
  const [expandedGames, setExpandedGames] = useState({}); // { [gameIndex]: boolean }
  const [confirmedBets, setConfirmedBets] = useState([]);
  const [editingBetId, setEditingBetId] = useState(null);
  const [editSide, setEditSide] = useState('');
  const [editWager, setEditWager] = useState('');
  const [editOdds, setEditOdds] = useState('');
  const [editOutcome, setEditOutcome] = useState('');
  const [editBankrollChange, setEditBankrollChange] = useState('');

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
    
    const customHome = bettingMode === 'spread'
      ? (custom?.home_odds ? parseFloat(custom.home_odds) : null)
      : (custom?.over_odds ? parseFloat(custom.over_odds) : null);
      
    const customAway = bettingMode === 'spread'
      ? (custom?.away_odds ? parseFloat(custom.away_odds) : null)
      : (custom?.under_odds ? parseFloat(custom.under_odds) : null);
      
    const customOverUnder = bettingMode === 'spread'
      ? null
      : (custom?.over_under ? parseFloat(custom.over_under) : null);

    const customPolyHome = custom?.poly_home_price ? parseFloat(custom.poly_home_price) : null;
    const customPolyAway = custom?.poly_away_price ? parseFloat(custom.poly_away_price) : null;

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
          custom_away_odds: customAway,
          custom_over_under: customOverUnder,
          custom_poly_home_price: customPolyHome,
          custom_poly_away_price: customPolyAway
        })
      });
      if (!res.ok) {
        const errData = await res.json();
        throw new Error(errData.error || `Status ${res.status}`);
      }
      await fetchConfirmedBets();
      await fetchUpcomingBets(true);
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
          away_team: bet.away_team_abbr,
          is_totals: bettingMode === 'total'
        })
      });
      if (!res.ok) {
        const errData = await res.json();
        throw new Error(errData.error || `Status ${res.status}`);
      }
      await fetchConfirmedBets();
      await fetchUpcomingBets(true);
    } catch (err) {
      setError(`Failed to delete bet: ${err.message}`);
    }
  };

  const handleDeleteConfirmedBet = async (betId) => {
    setError(null);
    try {
      const res = await fetch('/api/delete_bet', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ id: betId })
      });
      if (!res.ok) {
        const errData = await res.json();
        throw new Error(errData.error || `Status ${res.status}`);
      }
      await fetchConfirmedBets();
      await fetchUpcomingBets(true);
    } catch (err) {
      setError(`Failed to delete bet: ${err.message}`);
    }
  };

  const updatePnLPreview = (wagerStr, oddsStr, outcomeStr) => {
    const wagerVal = parseFloat(wagerStr) || 0;
    const oddsVal = parseFloat(oddsStr) || 0;
    if (outcomeStr === 'WON') {
      setEditBankrollChange((wagerVal * (oddsVal - 1.0)).toFixed(2));
    } else if (outcomeStr === 'LOST') {
      setEditBankrollChange((-wagerVal).toFixed(2));
    } else if (outcomeStr === 'PUSH') {
      setEditBankrollChange('0.00');
    } else {
      setEditBankrollChange((-wagerVal).toFixed(2));
    }
  };

  const handleSaveEditBet = async (betId) => {
    setError(null);
    try {
      const res = await fetch('/api/edit_bet', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          id: betId,
          recommended_side: editSide,
          wager_amount: parseFloat(editWager),
          odds: parseFloat(editOdds),
          outcome: editOutcome === 'PENDING' ? null : editOutcome.toLowerCase(),
          bankroll_change: parseFloat(editBankrollChange)
        })
      });
      if (!res.ok) {
        const errData = await res.json();
        throw new Error(errData.error || `Status ${res.status}`);
      }
      setEditingBetId(null);
      await fetchConfirmedBets();
      await fetchUpcomingBets(true);
    } catch (err) {
      setError(`Failed to save bet: ${err.message}`);
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

  const getTrackedBet = (bet, type) => {
    if (!confirmedBets || confirmedBets.length === 0) return null;
    const betHomeAbbr = getDbAbbr(bet.home_team_abbr) || getDbAbbr(bet.home_team);
    const betAwayAbbr = getDbAbbr(bet.away_team_abbr) || getDbAbbr(bet.away_team);
    return confirmedBets.find(cb => {
      const cbHomeAbbr = getDbAbbr(cb.home_team);
      const cbAwayAbbr = getDbAbbr(cb.away_team);
      const matchesMeta = cb.match_date === bet.date && cbHomeAbbr === betHomeAbbr && cbAwayAbbr === betAwayAbbr;
      if (!matchesMeta) return false;
      const isTotalsBet = cb.recommended_side.trim().toUpperCase() === 'OVER' || cb.recommended_side.trim().toUpperCase() === 'UNDER';
      return type === 'total' ? isTotalsBet : !isTotalsBet;
    });
  };



  useEffect(() => {
    localStorage.setItem('wnba_initial_bankroll', initialBankroll);
  }, [initialBankroll]);

  useEffect(() => {
    localStorage.setItem('wnba_bankroll_adjustment', bankrollAdjustment);
  }, [bankrollAdjustment]);

  const fetchUpcomingBets = async (silent = false) => {
    if (!silent) setLoading(true);
    setError(null);
    try {
      const res = await fetch('/api/upcoming_bets');
      if (!res.ok) {
        throw new Error(`Server returned status ${res.status}`);
      }
      const data = await res.json();
      setBets(data);

      // Pre-populate customOdds state from backend payloads
      const initialCustom = {};
      data.forEach(bet => {
        const gameKey = `${bet.date}_${bet.home_team_abbr}_${bet.away_team_abbr}`;
        const bm = bet.bookmaker;
        if (bm && (bm.custom_home_odds || bm.custom_away_odds || bm.custom_over_odds || bm.custom_under_odds || bm.custom_over_under)) {
          initialCustom[gameKey] = {
            home_odds: bm.custom_home_odds ? bm.custom_home_odds.toString() : '',
            away_odds: bm.custom_away_odds ? bm.custom_away_odds.toString() : '',
            over_odds: bm.custom_over_odds ? bm.custom_over_odds.toString() : '',
            under_odds: bm.custom_under_odds ? bm.custom_under_odds.toString() : '',
            over_under: bm.custom_over_under ? bm.custom_over_under.toString() : ''
          };
        }
      });
      setCustomOdds(prev => ({
        ...initialCustom,
        ...prev
      }));
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
      const sourceName = marketSource === 'bookie' ? 'FanDuel odds scraper' : 'live Prediction Market scraper';
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
    .reduce((sum, bet) => sum + (bet.bankroll_change || 0), 0) + bankrollAdjustment;

  const currentBankroll = initialBankroll + settledPnL - pendingWagers;

  const resolvedBets = confirmedBets.filter(bet => bet.outcome !== null);
  const wins = resolvedBets.filter(bet => bet.outcome?.toLowerCase() === 'won').length;
  const losses = resolvedBets.filter(bet => bet.outcome?.toLowerCase() === 'lost').length;
  const totalBets = resolvedBets.length;
  const winRate = totalBets > 0 ? ((wins / totalBets) * 100).toFixed(1) : '0.0';

  const isTotals = (bet) => {
    const side = (bet.recommended_side || '').trim().toUpperCase();
    return side === 'OVER' || side === 'UNDER';
  };

  const totalsResolved = resolvedBets.filter(bet => isTotals(bet));
  const totalsWins = totalsResolved.filter(bet => bet.outcome?.toLowerCase() === 'won').length;
  const totalsLosses = totalsResolved.filter(bet => bet.outcome?.toLowerCase() === 'lost').length;
  const totalsPushes = totalsResolved.filter(bet => bet.outcome?.toLowerCase() === 'push').length;
  const totalsDecided = totalsWins + totalsLosses;
  const totalsWinRate = totalsDecided > 0 ? ((totalsWins / totalsDecided) * 100).toFixed(1) : '0.0';

  const spreadResolved = resolvedBets.filter(bet => !isTotals(bet));
  const spreadWins = spreadResolved.filter(bet => bet.outcome?.toLowerCase() === 'won').length;
  const spreadLosses = spreadResolved.filter(bet => bet.outcome?.toLowerCase() === 'lost').length;
  const spreadPushes = spreadResolved.filter(bet => bet.outcome?.toLowerCase() === 'push').length;
  const spreadDecided = spreadWins + spreadLosses;
  const spreadWinRate = spreadDecided > 0 ? ((spreadWins / spreadDecided) * 100).toFixed(1) : '0.0';

  const chartHistory = React.useMemo(() => {
    const settled = confirmedBets
      .filter(bet => bet.outcome !== null)
      .sort((a, b) => {
        if (a.match_date !== b.match_date) {
          return a.match_date.localeCompare(b.match_date);
        }
        return (a.confirmed_at || '').localeCompare(b.confirmed_at || '') || (a.id - b.id);
      });

    const history = [{ date: 'Start', bankroll: initialBankroll, cumulative_profit: 0 }];
    let runningBankroll = initialBankroll;

    settled.forEach(bet => {
      runningBankroll += (bet.bankroll_change || 0);
      const adjustedBankroll = runningBankroll + bankrollAdjustment;
      history.push({
        date: bet.match_date,
        bankroll: adjustedBankroll,
        cumulative_profit: adjustedBankroll - initialBankroll
      });
    });

    return history;
  }, [confirmedBets, initialBankroll, bankrollAdjustment]);

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
            {bankrollAdjustment !== 0 && ` (Adj: ${bankrollAdjustment >= 0 ? '+' : ''}${bankrollAdjustment.toFixed(2)})`}
          </span>
        </div>

        <div className="metric-card">
          <span className="metric-card-label">Net P&L</span>
          <span className={`metric-card-value ${settledPnL >= 0 ? 'emerald' : 'rose'}`}>
            {settledPnL >= 0 ? '+' : ''}${settledPnL.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
          </span>
          <span className="metric-card-sub">
            Growth: {initialBankroll > 0 ? ((settledPnL / initialBankroll) * 100).toFixed(1) : '0.0'}%
            {bankrollAdjustment !== 0 && ` (Adj: ${bankrollAdjustment >= 0 ? '+' : ''}${bankrollAdjustment.toFixed(2)})`}
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
          <span className="metric-card-label">Spread / ML Record</span>
          <span className="metric-card-value" style={{ color: 'var(--neon-indigo)' }}>
            {spreadWins}W - {spreadLosses}L
          </span>
          <span className="metric-card-sub">
            Win Rate: {spreadWinRate}% ({spreadDecided} settled{spreadPushes > 0 ? `, ${spreadPushes} P` : ''})
          </span>
        </div>

        <div className="metric-card">
          <span className="metric-card-label">Over / Under Record</span>
          <span className="metric-card-value" style={{ color: 'var(--neon-purple)' }}>
            {totalsWins}W - {totalsLosses}L
          </span>
          <span className="metric-card-sub">
            Win Rate: {totalsWinRate}% ({totalsDecided} settled{totalsPushes > 0 ? `, ${totalsPushes} P` : ''})
          </span>
        </div>
      </div>

      {/* Bankroll Growth History Chart */}
      <div className="glass-card chart-section" style={{ marginBottom: '16px' }}>
        <div className="card-title">
          <span>Bankroll Growth History</span>
          <span className="badge" style={{ background: 'rgba(99, 102, 241, 0.15)', borderColor: 'var(--neon-indigo)', color: 'var(--neon-indigo)' }}>
            Confirmed Bets Performance
          </span>
        </div>
        <div style={{ padding: '10px 0' }}>
          <BankrollChart history={chartHistory} initialBankroll={initialBankroll} />
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
            <label className="control-label" htmlFor="adjustment-input">Bankroll Adjustment ($)</label>
            <input
              id="adjustment-input"
              type="number"
              step="0.01"
              className="select-input"
              value={bankrollAdjustment}
              onChange={(e) => setBankrollAdjustment(parseFloat(e.target.value) || 0)}
              disabled={loading || scraping}
              style={{ width: '100%' }}
              placeholder="e.g. -2.30"
            />
          </div>

          <div className="control-group">
            <label className="control-label" htmlFor="betting-mode-input">Betting Target</label>
            <select
              id="betting-mode-input"
              className="select-input"
              value={bettingMode}
              onChange={(e) => setBettingMode(e.target.value)}
              disabled={loading || scraping}
              style={{ width: '100%' }}
            >
              <option value="spread">Point Spread / Money Line</option>
              <option value="total">Over / Under Totals</option>
            </select>
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
              <option value="polymarket">Prediction Market Odds</option>
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
            <label className="control-label" htmlFor="kelly-cap-input">Kelly Cap ({kellyCap === 1.0 ? 'No Cap' : `${Math.round(kellyCap * 100)}%`})</label>
            <select
              id="kelly-cap-input"
              className="select-input"
              value={kellyCap}
              onChange={(e) => setKellyCap(parseFloat(e.target.value))}
              disabled={loading || scraping}
              style={{ width: '100%' }}
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

          <div className="control-group">
            <label className="control-label" htmlFor="odds-format-input">Odds Format</label>
            <select
              id="odds-format-input"
              className="select-input"
              value={oddsFormat}
              onChange={(e) => setOddsFormat(e.target.value)}
              disabled={loading || scraping}
              style={{ width: '100%' }}
            >
              <option value="american">American Odds (-110, +150)</option>
              <option value="decimal">Decimal Odds (1.91, 2.50)</option>
            </select>
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
              {scraping ? 'Scraping & Predicting...' : marketSource === 'bookie' ? 'Scrape Live FanDuel Odds' : 'Scrape Live Prediction Market'}
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
              Click <strong>{marketSource === 'bookie' ? 'Scrape Live FanDuel Odds' : 'Scrape Live Prediction Market'}</strong> above to fetch current WNBA matchups.
            </span>
          </div>
        ) : (
          <div className="table-container">
            <table className="custom-table">
              <thead>
                <tr>
                  <th>Date</th>
                  <th>Matchup</th>
                  <th style={{ textAlign: 'center' }}>{bettingMode === 'spread' ? 'Model Win Prob (H/A)' : 'Model Over/Under Prob'}</th>
                  <th style={{ textAlign: 'center' }}>
                    {marketSource === 'polymarket'
                      ? (bettingMode === 'spread' ? 'Prediction Market Odds (H/A)' : 'Prediction Market Over/Under Price')
                      : `Bookmaker Line / Odds (${oddsFormat === 'american' ? 'American' : 'Decimal'})`}
                  </th>
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

                  let suggestedBet = 'No Bet';
                  let activeEdge = 0;
                  let activeTeam = '';
                  let activePrice = 0;
                  let activeOdds = 1.90;
                  let isHomeBet = false;

                  const trackedBet = getTrackedBet(bet, bettingMode);

                  let homeEdge = 0;
                  let awayEdge = 0;
                  let overEdge = 0;
                  let underEdge = 0;

                  let displayOverProb = bet.over_probability || 50.0;
                  let displayUnderProb = bet.under_probability || 50.0;

                  if (bettingMode === 'spread') {
                    const homeModelProb = bet.home_prob / 100;
                    const awayModelProb = bet.away_prob / 100;

                    let homeOdds = 1.90;
                    let awayOdds = 1.90;
                    let homeMarketProb = 0.5;
                    let awayMarketProb = 0.5;

                    if (marketSource === 'polymarket') {
                      let homePrice = bet.home_price;
                      let awayPrice = bet.away_price;
                      if (custom) {
                        if (custom.poly_home_price) {
                          const parsed = parseFloat(custom.poly_home_price);
                          if (!isNaN(parsed) && parsed > 0) homePrice = parsed;
                        }
                        if (custom.poly_away_price) {
                          const parsed = parseFloat(custom.poly_away_price);
                          if (!isNaN(parsed) && parsed > 0) awayPrice = parsed;
                        }
                      }
                      homeMarketProb = homePrice;
                      awayMarketProb = awayPrice;
                      homeOdds = homePrice > 0 ? 1.0 / homePrice : 99.0;
                      awayOdds = awayPrice > 0 ? 1.0 / awayPrice : 99.0;
                    } else {
                      homeOdds = bet.bookmaker ? bet.bookmaker.home_odds : 1.90;
                      awayOdds = bet.bookmaker ? bet.bookmaker.away_odds : 1.90;
                      if (custom) {
                        if (custom.home_odds) {
                          const parsed = parseOddsInputToDecimal(custom.home_odds);
                          if (parsed && parsed > 1.0) homeOdds = parsed;
                        }
                        if (custom.away_odds) {
                          const parsed = parseOddsInputToDecimal(custom.away_odds);
                          if (parsed && parsed > 1.0) awayOdds = parsed;
                        }
                      }
                      const hasCustom = custom && (
                        (custom.home_odds && parseOddsInputToDecimal(custom.home_odds) !== null) ||
                        (custom.away_odds && parseOddsInputToDecimal(custom.away_odds) !== null)
                      );
                      if (hasCustom) {
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

                    homeEdge = homeModelProb - homeMarketProb;
                    awayEdge = awayModelProb - awayMarketProb;
                    const minEdge = minEdgePct / 100;

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
                  } else {
                    let overUnderLine = bet.bookmaker ? bet.bookmaker.over_under : 160.0;
                    if (custom && custom.over_under && !isNaN(parseFloat(custom.over_under))) {
                      overUnderLine = parseFloat(custom.over_under);
                    }

                    const overModelProbPct = (custom && custom.over_under)
                      ? normalCDF(bet.predicted_total || 160.0, overUnderLine, bet.total_dynamic_sigma || 10.0)
                      : (bet.over_probability || 50.0);
                    
                    displayOverProb = Math.round(overModelProbPct * 10) / 10;
                    displayUnderProb = Math.round((100.0 - overModelProbPct) * 10) / 10;

                    const overModelProb = overModelProbPct / 100;
                    const underModelProb = 1.0 - overModelProb;
                    
                    let overOdds = bet.bookmaker?.over_odds || 1.91;
                    let underOdds = bet.bookmaker?.under_odds || 1.91;
                    
                    if (custom) {
                      if (custom.over_odds) {
                        const parsed = parseOddsInputToDecimal(custom.over_odds);
                        if (parsed && parsed > 1.0) overOdds = parsed;
                      }
                      if (custom.under_odds) {
                        const parsed = parseOddsInputToDecimal(custom.under_odds);
                        if (parsed && parsed > 1.0) underOdds = parsed;
                      }
                    }
                    
                    const rawOverProb = 1.0 / overOdds;
                    const rawUnderProb = 1.0 / underOdds;
                    const sumTotalsProb = rawOverProb + rawUnderProb;
                    const overMarketProb = sumTotalsProb > 0 ? rawOverProb / sumTotalsProb : 0.5;
                    const underMarketProb = sumTotalsProb > 0 ? rawUnderProb / sumTotalsProb : 0.5;
                    
                    overEdge = overModelProb - overMarketProb;
                    underEdge = underModelProb - underMarketProb;
                    const minEdge = minEdgePct / 100;
                    
                    if (overEdge >= minEdge && overEdge >= underEdge) {
                      suggestedBet = 'OVER';
                      activeEdge = overEdge;
                      activeTeam = `Over ${overUnderLine}`;
                      activePrice = overMarketProb;
                      activeOdds = overOdds;
                    } else if (underEdge >= minEdge && underEdge >= overEdge) {
                      suggestedBet = 'UNDER';
                      activeEdge = underEdge;
                      activeTeam = `Under ${overUnderLine}`;
                      activePrice = underMarketProb;
                      activeOdds = underOdds;
                    }
                  }

                  let flatBetSize = 0;
                  let kellyBetSize = 0;
                  let kellyPct = 0;
                  let flatWin = 0;
                  let flatLoss = 0;
                  let kellyWin = 0;
                  let kellyLoss = 0;

                  if (suggestedBet !== 'No Bet') {
                    flatBetSize = currentBankroll * (flatWagerPct / 100);
                    flatLoss = flatBetSize;
                    flatWin = flatBetSize * (activeOdds - 1.0);

                    let kellyFraction = (activeEdge) / (1.0 - activePrice);
                    if (kellyFraction > 0) {
                      kellyFraction = kellyCap * kellyFraction;
                      kellyFraction = Math.min(kellyCap, kellyFraction);
                      
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
                          {bettingMode === 'spread' ? (
                            <>
                              <span style={{ color: 'var(--neon-indigo)', fontWeight: '600' }}>{bet.home_prob}%</span>
                              <span style={{ margin: '0 6px', color: 'var(--color-text-dim)' }}>/</span>
                              <span style={{ color: 'var(--neon-purple)', fontWeight: '600' }}>{bet.away_prob}%</span>
                            </>
                          ) : (
                            <>
                              <span style={{ color: 'var(--neon-indigo)', fontWeight: '600' }}>O: {displayOverProb}%</span>
                              <span style={{ margin: '0 6px', color: 'var(--color-text-dim)' }}>/</span>
                              <span style={{ color: 'var(--neon-purple)', fontWeight: '600' }}>U: {displayUnderProb}%</span>
                            </>
                          )}
                        </td>
                        <td style={{ textAlign: 'center' }}>
                          {bettingMode === 'spread' ? (
                            marketSource === 'polymarket' ? (
                              <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: '6px' }}>
                                <div style={{ display: 'flex', alignItems: 'center', gap: '4px' }}>
                                  <span style={{ fontSize: '0.8rem', color: 'var(--color-text-dim)' }}>$</span>
                                  <input
                                    type="number"
                                    step="0.01"
                                    min="0.01"
                                    max="0.99"
                                    placeholder={bet.home_price ? bet.home_price.toFixed(2) : '0.50'}
                                    value={customOdds[gameKey]?.poly_home_price || ''}
                                    onChange={(e) => handleCustomOddsChange(gameKey, 'poly_home_price', e.target.value)}
                                    onBlur={(e) => handleUpdatePredictionMarketOdds(bet, 'poly_home_price', e.target.value)}
                                    onKeyDown={(e) => {
                                      if (e.key === 'Enter') {
                                        handleUpdatePredictionMarketOdds(bet, 'poly_home_price', e.target.value);
                                        e.target.blur();
                                      }
                                    }}
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
                                  <span style={{ fontSize: '0.8rem', color: 'var(--color-text-dim)' }}>$</span>
                                  <input
                                    type="number"
                                    step="0.01"
                                    min="0.01"
                                    max="0.99"
                                    placeholder={bet.away_price ? bet.away_price.toFixed(2) : '0.50'}
                                    value={customOdds[gameKey]?.poly_away_price || ''}
                                    onChange={(e) => handleCustomOddsChange(gameKey, 'poly_away_price', e.target.value)}
                                    onBlur={(e) => handleUpdatePredictionMarketOdds(bet, 'poly_away_price', e.target.value)}
                                    onKeyDown={(e) => {
                                      if (e.key === 'Enter') {
                                        handleUpdatePredictionMarketOdds(bet, 'poly_away_price', e.target.value);
                                        e.target.blur();
                                      }
                                    }}
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
                                <span style={{
                                  background: customOdds[gameKey]?.poly_home_price || customOdds[gameKey]?.poly_away_price
                                    ? 'rgba(245, 158, 11, 0.15)'
                                    : 'rgba(99, 102, 241, 0.15)',
                                  border: customOdds[gameKey]?.poly_home_price || customOdds[gameKey]?.poly_away_price
                                    ? '1px solid var(--neon-amber)'
                                    : '1px solid var(--neon-indigo)',
                                  color: customOdds[gameKey]?.poly_home_price || customOdds[gameKey]?.poly_away_price
                                    ? 'var(--neon-amber)'
                                    : 'var(--neon-indigo)',
                                  padding: '2px 6px',
                                  borderRadius: '4px',
                                  fontSize: '0.65rem',
                                  fontWeight: '600',
                                  display: 'inline-block'
                                }}>
                                  {customOdds[gameKey]?.poly_home_price || customOdds[gameKey]?.poly_away_price ? '[Prediction - Custom]' : '[Prediction Market]'}
                                </span>
                              </div>
                            ) : (
                              <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: '6px' }}>
                                <div style={{ display: 'flex', alignItems: 'center', gap: '4px' }}>
                                  <input
                                    type="text"
                                    placeholder={formatOddsPlaceholder(bet.bookmaker?.home_odds || 1.90, oddsFormat)}
                                    value={customOdds[gameKey]?.home_odds || ''}
                                    onChange={(e) => handleCustomOddsChange(gameKey, 'home_odds', e.target.value)}
                                    style={{
                                      width: oddsFormat === 'american' ? '68px' : '64px',
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
                                    type="text"
                                    placeholder={formatOddsPlaceholder(bet.bookmaker?.away_odds || 1.90, oddsFormat)}
                                    value={customOdds[gameKey]?.away_odds || ''}
                                    onChange={(e) => handleCustomOddsChange(gameKey, 'away_odds', e.target.value)}
                                    style={{
                                      width: oddsFormat === 'american' ? '68px' : '64px',
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
                                {bet.bookmaker?.is_fanduel ? (
                                  <span style={{
                                    background: customOdds[gameKey]?.home_odds || customOdds[gameKey]?.away_odds
                                      ? 'rgba(245, 158, 11, 0.15)'
                                      : 'rgba(16, 185, 129, 0.15)',
                                    border: customOdds[gameKey]?.home_odds || customOdds[gameKey]?.away_odds
                                      ? '1px solid var(--neon-amber)'
                                      : '1px solid var(--neon-emerald)',
                                    color: customOdds[gameKey]?.home_odds || customOdds[gameKey]?.away_odds
                                      ? 'var(--neon-amber)'
                                      : 'var(--neon-emerald)',
                                    padding: '2px 6px',
                                    borderRadius: '4px',
                                    fontSize: '0.65rem',
                                    fontWeight: '600',
                                    display: 'inline-block'
                                  }}>
                                    {customOdds[gameKey]?.home_odds || customOdds[gameKey]?.away_odds ? '[FanDuel - Custom]' : '[FanDuel]'}
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
                            )
                          ) : (
                            <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: '6px' }}>
                              <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: '6px' }}>
                                <div style={{ display: 'flex', alignItems: 'center', gap: '4px' }}>
                                  <span style={{ fontSize: '0.8rem', color: 'var(--color-text-dim)' }}>O/U:</span>
                                  <input
                                    type="number"
                                    step="0.5"
                                    placeholder={bet.bookmaker?.over_under ? bet.bookmaker.over_under.toString() : '160.0'}
                                    value={customOdds[gameKey]?.over_under || ''}
                                    onChange={(e) => handleCustomOddsChange(gameKey, 'over_under', e.target.value)}
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
                                <div style={{ display: 'flex', alignItems: 'center', gap: '4px' }}>
                                  <input
                                    type="text"
                                    placeholder={formatOddsPlaceholder(bet.bookmaker?.over_odds || 1.91, oddsFormat)}
                                    value={customOdds[gameKey]?.over_odds || ''}
                                    onChange={(e) => handleCustomOddsChange(gameKey, 'over_odds', e.target.value)}
                                    style={{
                                      width: oddsFormat === 'american' ? '60px' : '54px',
                                      background: 'rgba(255, 255, 255, 0.05)',
                                      border: '1px solid var(--border-card)',
                                      color: 'var(--color-text-main)',
                                      borderRadius: '6px',
                                      padding: '4px 6px',
                                      fontSize: '0.75rem',
                                      textAlign: 'center',
                                      outline: 'none'
                                    }}
                                  />
                                  <span style={{ color: 'var(--color-text-dim)' }}>/</span>
                                  <input
                                    type="text"
                                    placeholder={formatOddsPlaceholder(bet.bookmaker?.under_odds || 1.91, oddsFormat)}
                                    value={customOdds[gameKey]?.under_odds || ''}
                                    onChange={(e) => handleCustomOddsChange(gameKey, 'under_odds', e.target.value)}
                                    style={{
                                      width: oddsFormat === 'american' ? '60px' : '54px',
                                      background: 'rgba(255, 255, 255, 0.05)',
                                      border: '1px solid var(--border-card)',
                                      color: 'var(--color-text-main)',
                                      borderRadius: '6px',
                                      padding: '4px 6px',
                                      fontSize: '0.75rem',
                                      textAlign: 'center',
                                      outline: 'none'
                                    }}
                                  />
                                </div>
                              </div>
                              {bet.bookmaker?.is_fanduel ? (
                                <span style={{
                                  background: customOdds[gameKey]?.over_under || customOdds[gameKey]?.over_odds || customOdds[gameKey]?.under_odds
                                    ? 'rgba(245, 158, 11, 0.15)'
                                    : 'rgba(16, 185, 129, 0.15)',
                                  border: customOdds[gameKey]?.over_under || customOdds[gameKey]?.over_odds || customOdds[gameKey]?.under_odds
                                    ? '1px solid var(--neon-amber)'
                                    : '1px solid var(--neon-emerald)',
                                  color: customOdds[gameKey]?.over_under || customOdds[gameKey]?.over_odds || customOdds[gameKey]?.under_odds
                                    ? 'var(--neon-amber)'
                                    : 'var(--neon-emerald)',
                                  padding: '2px 6px',
                                  borderRadius: '4px',
                                  fontSize: '0.65rem',
                                  fontWeight: '600',
                                  display: 'inline-block',
                                  marginTop: '2px'
                                }}>
                                  {customOdds[gameKey]?.over_under || customOdds[gameKey]?.over_odds || customOdds[gameKey]?.under_odds ? '[FanDuel - Custom]' : '[FanDuel]'}
                                </span>
                              ) : (
                                <span style={{
                                  background: customOdds[gameKey]?.over_under || customOdds[gameKey]?.over_odds || customOdds[gameKey]?.under_odds
                                    ? 'rgba(245, 158, 11, 0.15)'
                                    : 'rgba(255, 255, 255, 0.05)',
                                  border: customOdds[gameKey]?.over_under || customOdds[gameKey]?.over_odds || customOdds[gameKey]?.under_odds
                                    ? '1px solid var(--neon-amber)'
                                    : '1px solid var(--border-card)',
                                  color: customOdds[gameKey]?.over_under || customOdds[gameKey]?.over_odds || customOdds[gameKey]?.under_odds
                                    ? 'var(--neon-amber)'
                                    : 'var(--color-text-muted)',
                                  padding: '2px 6px',
                                  borderRadius: '4px',
                                  fontSize: '0.65rem',
                                  fontWeight: '600',
                                  display: 'inline-block',
                                  marginTop: '2px'
                                }}>
                                  {customOdds[gameKey]?.over_under || customOdds[gameKey]?.over_odds || customOdds[gameKey]?.under_odds ? '[ELO - Custom]' : '[ELO]'}
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
                                {bettingMode === 'spread' 
                                  ? `BET ${isHomeBet ? bet.home_team_abbr : bet.away_team_abbr}`
                                  : `BET ${activeTeam.toUpperCase()}`}
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
                                Max Edge: {bettingMode === 'spread' 
                                 ? (Math.max(homeEdge, awayEdge) > 0 ? `+${(Math.max(homeEdge, awayEdge) * 100).toFixed(1)}%` : `${(Math.max(homeEdge, awayEdge) * 100).toFixed(1)}%`)
                                 : (Math.max(overEdge, underEdge) > 0 ? `+${(Math.max(overEdge, underEdge) * 100).toFixed(1)}%` : `${(Math.max(overEdge, underEdge) * 100).toFixed(1)}%`)}
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
                                    <span className="health-metric-label" style={{ fontSize: '0.65rem' }}>Missing NET</span>
                                    <span className="health-metric-value" style={{ fontSize: '0.85rem' }}>{(bet.home_health?.Missing_Net_Rating ?? bet.home_health?.Missing_Net_Rating_Pct ?? bet.home_health?.Missing_NET_Pct ?? bet.home_health?.Missing_BPM_Pct ?? 0).toFixed(1)}</span>
                                  </div>
                                  <div className="health-metric-box" style={{ flex: '1 1 70px', padding: '6px 10px', background: 'rgba(255, 255, 255, 0.02)', borderRadius: '8px' }}>
                                    <span className="health-metric-label" style={{ fontSize: '0.65rem' }}>Missing PIE</span>
                                    <span className="health-metric-value" style={{ fontSize: '0.85rem' }}>{(bet.home_health?.Missing_PIE ?? bet.home_health?.Missing_PIE_Pct ?? 0).toFixed(1)}%</span>
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
                                    <span className="health-metric-label" style={{ fontSize: '0.65rem' }}>Missing NET</span>
                                    <span className="health-metric-value" style={{ fontSize: '0.85rem' }}>{(bet.away_health?.Missing_Net_Rating ?? bet.away_health?.Missing_Net_Rating_Pct ?? bet.away_health?.Missing_NET_Pct ?? bet.away_health?.Missing_BPM_Pct ?? 0).toFixed(1)}</span>
                                  </div>
                                  <div className="health-metric-box" style={{ flex: '1 1 70px', padding: '6px 10px', background: 'rgba(255, 255, 255, 0.02)', borderRadius: '8px' }}>
                                    <span className="health-metric-label" style={{ fontSize: '0.65rem' }}>Missing PIE</span>
                                    <span className="health-metric-value" style={{ fontSize: '0.85rem' }}>{(bet.away_health?.Missing_PIE ?? bet.away_health?.Missing_PIE_Pct ?? 0).toFixed(1)}%</span>
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

      {/* Tracked Bets History Card */}
      <div className="glass-card" style={{ marginTop: '24px' }}>
        <div className="card-title" style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <span>Tracked Bets History</span>
          <span className="badge" style={{ background: 'rgba(99, 102, 241, 0.15)', borderColor: 'var(--neon-indigo)', color: 'var(--neon-indigo)' }}>
            {confirmedBets.length} Tracked Bets
          </span>
        </div>

        <div style={{ display: 'flex', gap: '12px', flexWrap: 'wrap', padding: '0 4px', marginBottom: '16px' }}>
          <div style={{
            background: 'rgba(255, 255, 255, 0.03)',
            border: '1px solid var(--border-card)',
            borderRadius: '8px',
            padding: '8px 14px',
            fontSize: '0.8rem',
            display: 'flex',
            alignItems: 'center',
            gap: '8px'
          }}>
            <span style={{ color: 'var(--color-text-dim)' }}>Spread / ML:</span>
            <span style={{ fontWeight: '700', color: 'var(--neon-indigo)' }}>{spreadWins}W - {spreadLosses}L</span>
            <span style={{ color: 'var(--neon-emerald)', fontWeight: '600' }}>({spreadWinRate}%)</span>
          </div>

          <div style={{
            background: 'rgba(255, 255, 255, 0.03)',
            border: '1px solid var(--border-card)',
            borderRadius: '8px',
            padding: '8px 14px',
            fontSize: '0.8rem',
            display: 'flex',
            alignItems: 'center',
            gap: '8px'
          }}>
            <span style={{ color: 'var(--color-text-dim)' }}>Over / Under:</span>
            <span style={{ fontWeight: '700', color: 'var(--neon-purple)' }}>{totalsWins}W - {totalsLosses}L</span>
            <span style={{ color: 'var(--neon-emerald)', fontWeight: '600' }}>({totalsWinRate}%)</span>
          </div>

          <div style={{
            background: 'rgba(255, 255, 255, 0.03)',
            border: '1px solid var(--border-card)',
            borderRadius: '8px',
            padding: '8px 14px',
            fontSize: '0.8rem',
            display: 'flex',
            alignItems: 'center',
            gap: '8px'
          }}>
            <span style={{ color: 'var(--color-text-dim)' }}>Combined Total:</span>
            <span style={{ fontWeight: '700', color: 'var(--color-text-main)' }}>{wins}W - {losses}L</span>
            <span style={{ color: 'var(--neon-emerald)', fontWeight: '600' }}>({winRate}%)</span>
          </div>
        </div>

        {confirmedBets.length === 0 ? (
          <div style={{ textAlign: 'center', padding: '30px', color: 'var(--color-text-muted)', fontSize: '0.9rem' }}>
            No tracked bets recorded yet. Confirm a bet from the Edge Finder above to start tracking.
          </div>
        ) : (
          <div className="table-container">
            <table className="custom-table">
              <thead>
                <tr>
                  <th>Date</th>
                  <th>Matchup</th>
                  <th>Bet Target</th>
                  <th>Type</th>
                  <th style={{ textAlign: 'right' }}>Wager</th>
                  <th style={{ textAlign: 'right' }}>Odds</th>
                  <th style={{ textAlign: 'center' }}>Outcome</th>
                  <th style={{ textAlign: 'right' }}>PnL</th>
                  <th style={{ textAlign: 'center', width: '160px' }}>Actions</th>
                </tr>
              </thead>
              <tbody>
                {confirmedBets.map((bet) => {
                  const isEditing = editingBetId === bet.id;
                  
                  // Calculate potential change for preview in editing
                  let previewPnL = 0;
                  if (isEditing) {
                    const tempWager = parseFloat(editWager) || 0;
                    const tempOdds = parseFloat(editOdds) || 0;
                    if (editOutcome === 'WON') {
                      previewPnL = tempWager * (tempOdds - 1.0);
                    } else if (editOutcome === 'LOST') {
                      previewPnL = -tempWager;
                    }
                  }

                  const isTotalBet = bet.recommended_side.trim().toUpperCase() === 'OVER' || bet.recommended_side.trim().toUpperCase() === 'UNDER';
                  
                  return (
                    <tr key={bet.id}>
                      <td>{bet.match_date}</td>
                      <td style={{ fontWeight: '600' }}>
                        {bet.home_team} vs {bet.away_team}
                      </td>
                      <td>
                        {isEditing ? (
                          <select
                            value={editSide}
                            onChange={(e) => setEditSide(e.target.value)}
                            className="select-input"
                            style={{ padding: '4px', fontSize: '0.8rem', background: 'var(--bg-card)', color: '#fff', border: '1px solid var(--border-card)', borderRadius: '4px' }}
                          >
                            <option value="OVER">OVER</option>
                            <option value="UNDER">UNDER</option>
                            <option value={bet.home_team}>{bet.home_team}</option>
                            <option value={bet.away_team}>{bet.away_team}</option>
                          </select>
                        ) : (
                          <span style={{
                            fontWeight: '700',
                            color: isTotalBet ? 'var(--neon-purple)' : 'var(--neon-indigo)'
                          }}>
                            {bet.recommended_side}
                          </span>
                        )}
                      </td>
                      <td>{bet.wager_type}</td>
                      <td style={{ textAlign: 'right', fontWeight: '700' }}>
                        {isEditing ? (
                          <input
                            type="number"
                            step="0.01"
                            value={editWager}
                            onChange={(e) => setEditWager(e.target.value)}
                            style={{ width: '70px', padding: '4px', textAlign: 'right', background: 'var(--bg-card)', color: '#fff', border: '1px solid var(--border-card)', borderRadius: '4px' }}
                          />
                        ) : (
                          `$${bet.wager_amount.toFixed(2)}`
                        )}
                      </td>
                      <td style={{ textAlign: 'right' }}>
                        {isEditing ? (
                          <input
                            type="number"
                            step="0.01"
                            value={editOdds}
                            onChange={(e) => setEditOdds(e.target.value)}
                            style={{ width: '60px', padding: '4px', textAlign: 'right', background: 'var(--bg-card)', color: '#fff', border: '1px solid var(--border-card)', borderRadius: '4px' }}
                          />
                        ) : (
                          bet.odds.toFixed(2)
                        )}
                      </td>
                      <td style={{ textAlign: 'center' }}>
                        {isEditing ? (
                          <select
                            value={editOutcome}
                            onChange={(e) => setEditOutcome(e.target.value)}
                            className="select-input"
                            style={{ padding: '4px', fontSize: '0.8rem', background: 'var(--bg-card)', color: '#fff', border: '1px solid var(--border-card)', borderRadius: '4px' }}
                          >
                            <option value="PENDING">PENDING</option>
                            <option value="WON">WON</option>
                            <option value="LOST">LOST</option>
                            <option value="PUSH">PUSH</option>
                          </select>
                        ) : (
                          <span className="badge" style={{
                            background: bet.outcome === null
                              ? 'rgba(245, 158, 11, 0.15)'
                              : bet.outcome?.toLowerCase() === 'won'
                                ? 'rgba(16, 185, 129, 0.15)'
                                : bet.outcome?.toLowerCase() === 'push'
                                  ? 'rgba(255, 255, 255, 0.1)'
                                  : 'rgba(244, 63, 94, 0.15)',
                            borderColor: bet.outcome === null
                              ? 'var(--neon-amber)'
                              : bet.outcome?.toLowerCase() === 'won'
                                ? 'var(--neon-emerald)'
                                : bet.outcome?.toLowerCase() === 'push'
                                  ? 'var(--color-text-dim)'
                                  : 'var(--neon-rose)',
                            color: bet.outcome === null
                              ? 'var(--neon-amber)'
                              : bet.outcome?.toLowerCase() === 'won'
                                ? 'var(--neon-emerald)'
                                : bet.outcome?.toLowerCase() === 'push'
                                  ? 'var(--color-text-dim)'
                                  : 'var(--neon-rose)',
                            fontSize: '0.7rem',
                            padding: '2px 6px'
                          }}>
                            {(bet.outcome || 'PENDING').toUpperCase()}
                          </span>
                        )}
                      </td>
                      <td style={{ textAlign: 'right', fontWeight: '700' }}>
                        {isEditing ? (
                          <input
                            type="number"
                            step="0.01"
                            value={editBankrollChange}
                            onChange={(e) => setEditBankrollChange(e.target.value)}
                            style={{ width: '80px', padding: '4px', textAlign: 'right', background: 'var(--bg-card)', color: '#fff', border: '1px solid var(--border-card)', borderRadius: '4px' }}
                          />
                        ) : (
                          bet.outcome === null ? (
                            <span style={{ color: 'var(--color-text-dim)' }}>—</span>
                          ) : (
                            <span style={{
                              color: bet.bankroll_change >= 0 ? 'var(--neon-emerald)' : 'var(--neon-rose)'
                            }}>
                              {bet.bankroll_change >= 0 ? '+' : ''}${bet.bankroll_change.toFixed(2)}
                            </span>
                          )
                        )}
                      </td>
                      <td style={{ textAlign: 'center' }}>
                        {isEditing ? (
                          <div style={{ display: 'flex', gap: '6px', justifyContent: 'center' }}>
                            <button
                              onClick={() => handleSaveEditBet(bet.id)}
                              className="select-input"
                              style={{
                                padding: '2px 8px',
                                fontSize: '0.7rem',
                                cursor: 'pointer',
                                background: 'rgba(16, 185, 129, 0.15)',
                                border: '1px solid var(--neon-emerald)',
                                borderRadius: '6px',
                                color: 'var(--neon-emerald)',
                                fontWeight: '600'
                              }}
                            >
                              Save
                            </button>
                            <button
                              onClick={() => setEditingBetId(null)}
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
                              Cancel
                            </button>
                          </div>
                        ) : (
                          <div style={{ display: 'flex', gap: '6px', justifyContent: 'center' }}>
                            <button
                              onClick={() => {
                                setEditingBetId(bet.id);
                                setEditSide(bet.recommended_side);
                                setEditWager(bet.wager_amount.toString());
                                setEditOdds(bet.odds.toString());
                                setEditOutcome(bet.outcome ? bet.outcome.toUpperCase() : 'PENDING');
                                setEditBankrollChange(bet.bankroll_change.toString());
                              }}
                              className="select-input"
                              style={{
                                padding: '2px 8px',
                                fontSize: '0.7rem',
                                cursor: 'pointer',
                                background: 'rgba(99, 102, 241, 0.15)',
                                border: '1px solid var(--neon-indigo)',
                                borderRadius: '6px',
                                color: 'var(--neon-indigo)'
                              }}
                            >
                              Edit
                            </button>
                            <button
                              onClick={() => {
                                if (window.confirm("Are you sure you want to delete/reset this confirmed bet? This will remove it from your bankroll history.")) {
                                  handleDeleteConfirmedBet(bet.id);
                                }
                              }}
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
                              Delete
                            </button>
                          </div>
                        )}
                      </td>
                    </tr>
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
