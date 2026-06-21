import React, { useState, useEffect } from 'react';

// Custom lightweight SVG Line Chart for Bankroll History over time (react-19 compatible, no dependencies)
function BankrollChart({ history, initialBankroll }) {
  if (!history || history.length === 0) {
    return <div className="text-muted" style={{ padding: '24px', textAlign: 'center' }}>No history data to display.</div>;
  }
  const [hoveredPoint, setHoveredPoint] = useState(null);
  
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
    for (let i = 0; i < tickCount; i++) {
      const idx = Math.floor((i / (tickCount - 1)) * (pointsCount - 1));
      xTicks.push(points[idx]);
    }
  }

  return (
    <div style={{ position: 'relative', width: '100%' }}>
      <svg viewBox={`0 0 ${width} ${height}`} style={{ width: '100%', height: 'auto', overflow: 'visible' }}>
        <defs>
          <linearGradient id="chart-glow" x1="0" y1="0" x2="0" y2="1">
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
        {areaD && <path d={areaD} fill="url(#chart-glow)" />}
        
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
            top: `${(hoveredPoint.y / height) * 100 - 110}%`,
            transform: 'translateX(-50%)',
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

export default function SimulationBacktester() {
  const [season, setSeason] = useState('2025');
  const [initialBankroll, setInitialBankroll] = useState(1000);
  const [minEdge, setMinEdge] = useState(0); // in percent, e.g. 0 = 0%
  const [wagerType, setWagerType] = useState('flat');
  const [flatWagerPct, setFlatWagerPct] = useState(2); // 2 = 2% of initial bankroll
  const [marketSource, setMarketSource] = useState('bookie');

  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const seasonsList = ['2018', '2019', '2020', '2021', '2022', '2023', '2024', '2025', '2026'];

  const fetchSimulation = async () => {
    setLoading(true);
    setError(null);
    try {
      // Map minEdge and flatWagerPct from percent to ratio for the backend API
      const minEdgeRatio = (minEdge / 100).toFixed(4);
      const flatWagerPctRatio = (flatWagerPct / 100).toFixed(4);
      
      const url = `/api/simulation/run?season=${season}&initial_bankroll=${initialBankroll}&min_edge=${minEdgeRatio}&wager_type=${wagerType}&flat_wager_pct=${flatWagerPctRatio}&market_source=${marketSource}`;
      const response = await fetch(url);
      if (!response.ok) {
        throw new Error(`Server returned status: ${response.status}`);
      }
      const json = await response.json();
      if (json.error) {
        throw new Error(json.error);
      }
      setData(json);
    } catch (err) {
      setError(err.message || 'Failed to fetch backtest simulation data.');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchSimulation();
  }, [season]); // Refetch automatically on season change; for betting config we allow manual re-run triggers

  const metrics = data?.metrics;
  const standings = data?.standings || [];
  const games = data?.games || [];
  const bettingMetrics = data?.betting_metrics;
  const bankrollHistory = data?.bankroll_history || [];

  // Filter games where bets were placed
  const valueBets = games.filter(g => g.bet_placed);

  return (
    <div className="sim-dashboard-grid">
      {/* Parameter Control Card */}
      <div className="glass-card" style={{ marginBottom: '8px' }}>
        <div className="card-title">
          <span>Simulation Backtesting & Betting Parameters</span>
          <span className="badge">Betting Strategy Backtester</span>
        </div>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: '20px', alignItems: 'end' }}>
          
          <div className="control-group">
            <label className="control-label" htmlFor="sim-season">Season</label>
            <select
              id="sim-season"
              className="select-input"
              value={season}
              onChange={(e) => setSeason(e.target.value)}
              disabled={loading}
            >
              {seasonsList.map(yr => (
                <option key={yr} value={yr}>{yr} Season</option>
              ))}
            </select>
          </div>

          <div className="control-group">
            <label className="control-label" htmlFor="sim-market">Market Odds Source</label>
            <select
              id="sim-market"
              className="select-input"
              value={marketSource}
              onChange={(e) => setMarketSource(e.target.value)}
              disabled={loading}
            >
              <option value="bookie">Traditional Bookmakers</option>
              <option value="polymarket">Polymarket YES Contract Price</option>
            </select>
          </div>

          <div className="control-group">
            <label className="control-label" htmlFor="sim-bankroll">Starting Bankroll ($)</label>
            <input
              id="sim-bankroll"
              type="number"
              className="date-input"
              value={initialBankroll}
              onChange={(e) => setInitialBankroll(Math.max(10, Number(e.target.value)))}
              disabled={loading}
              min="10"
            />
          </div>

          <div className="control-group">
            <label className="control-label" htmlFor="sim-edge">Min Required Edge ({minEdge}%)</label>
            <input
              id="sim-edge"
              type="range"
              min="0"
              max="15"
              step="1"
              value={minEdge}
              onChange={(e) => setMinEdge(Number(e.target.value))}
              disabled={loading}
              style={{ width: '100%', height: '38px' }}
            />
          </div>

          <div className="control-group">
            <label className="control-label" htmlFor="sim-wager-type">Bet Sizing Strategy</label>
            <select
              id="sim-wager-type"
              className="select-input"
              value={wagerType}
              onChange={(e) => setWagerType(e.target.value)}
              disabled={loading}
            >
              <option value="flat">Flat Betting ({flatWagerPct}% of Initial Bankroll)</option>
              <option value="kelly">Kelly Criterion (Quarter-Kelly, 15% Cap)</option>
            </select>
          </div>

          {wagerType === 'flat' && (
            <div className="control-group">
              <label className="control-label" htmlFor="sim-flat-pct">Flat Bet Size ({flatWagerPct}% = ${Math.round(initialBankroll * flatWagerPct / 100)})</label>
              <input
                id="sim-flat-pct"
                type="number"
                min="0.5"
                max="25"
                step="0.5"
                className="date-input"
                value={flatWagerPct}
                onChange={(e) => setFlatWagerPct(Math.max(0.5, Number(e.target.value)))}
                disabled={loading}
              />
            </div>
          )}

          <div className="control-group" style={{ gridColumn: wagerType === 'kelly' ? 'span 1' : 'span 1' }}>
            <button
              onClick={fetchSimulation}
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
                filter: 'drop-shadow(0 2px 4px rgba(99, 102, 241, 0.3))'
              }}
              disabled={loading}
            >
              {loading ? 'Simulating...' : 'Reset & Re-Run'}
            </button>
          </div>
        </div>
      </div>

      {error && <div className="error-alert">{error}</div>}

      {/* Betting Overview Statistics Grid */}
      {bettingMetrics && (
        <div className="metrics-grid" style={{ gridTemplateColumns: 'repeat(auto-fit, minmax(220px, 1fr))' }}>
          <div className="metric-card">
            <span className="metric-card-label">Ending Bankroll</span>
            <span className={`metric-card-value ${bettingMetrics.net_profit >= 0 ? 'emerald' : 'rose'}`}>
              ${bettingMetrics.final_bankroll.toLocaleString()}
            </span>
            <span className="metric-card-sub">
              Starting: ${bettingMetrics.initial_bankroll.toLocaleString()}
            </span>
          </div>

          <div className="metric-card">
            <span className="metric-card-label">Net Profit / Loss</span>
            <span className={`metric-card-value ${bettingMetrics.net_profit >= 0 ? 'emerald' : 'rose'}`}>
              {bettingMetrics.net_profit >= 0 ? '+' : ''}${bettingMetrics.net_profit.toLocaleString()}
            </span>
            <span className="metric-card-sub">
              Growth: {((bettingMetrics.net_profit / bettingMetrics.initial_bankroll) * 100).toFixed(1)}%
            </span>
          </div>

          <div className="metric-card">
            <span className="metric-card-label">Return on Investment (ROI)</span>
            <span className={`metric-card-value ${bettingMetrics.roi >= 0 ? 'emerald' : 'rose'}`}>
              {bettingMetrics.roi >= 0 ? '+' : ''}{bettingMetrics.roi}%
            </span>
            <span className="metric-card-sub">
              Wagered: ${bettingMetrics.total_wagered.toLocaleString()}
            </span>
          </div>

          <div className="metric-card">
            <span className="metric-card-label">Bet Record & Win Rate</span>
            <span className="metric-card-value emerald">
              {bettingMetrics.won_bets}W - {bettingMetrics.total_bets - bettingMetrics.won_bets}L
            </span>
            <span className="metric-card-sub">
              Win Rate: {bettingMetrics.win_rate}% on {bettingMetrics.total_bets} wagers
            </span>
          </div>
        </div>
      )}

      {/* Stands and Bankroll Chart Row */}
      <div style={{ display: 'grid', gridTemplateColumns: '1.6fr 1.4fr', gap: '28px' }}>
        
        {/* Bankroll Chart */}
        <div className="glass-card chart-section">
          <div className="card-title">
            <span>Bankroll Growth History</span>
            <span style={{ fontSize: '0.8rem', color: 'var(--color-text-muted)' }}>
              Hover coordinates to view historical balance
            </span>
          </div>
          {loading ? (
            <div style={{ height: '300px', display: 'flex', alignItems: 'center', justifyContent: 'center', color: 'var(--color-text-muted)' }}>
              <div className="sim-spinner" style={{ marginRight: '10px' }}></div>
              Calculating simulation path...
            </div>
          ) : (
            <BankrollChart history={bankrollHistory} initialBankroll={initialBankroll} />
          )}
        </div>

        {/* Standings Comparison Table */}
        <div className="glass-card">
          <div className="card-title">
            <span>Simulated vs. Actual standins</span>
            <span className="badge" style={{ borderColor: 'var(--neon-indigo)', color: 'var(--neon-indigo)' }}>
              {season} Season
            </span>
          </div>
          {loading ? (
            <div style={{ padding: '60px', textAlign: 'center', color: 'var(--color-text-muted)' }}>
              Compiling standings...
            </div>
          ) : (
            <div className="table-container" style={{ marginHeight: '300px', overflowY: 'auto' }}>
              <table className="custom-table" style={{ fontSize: '0.8rem' }}>
                <thead>
                  <tr>
                    <th>Team</th>
                    <th style={{ textAlign: 'center' }}>Actual Wins</th>
                    <th style={{ textAlign: 'center' }}>Simulated Wins</th>
                    <th style={{ textAlign: 'center' }}>Win Diff</th>
                    <th>Expectation Status</th>
                  </tr>
                </thead>
                <tbody>
                  {standings.map((std, idx) => {
                    const diff = std.actual_wins - std.simulated_wins;
                    const diffStr = diff >= 0 ? `+${diff.toFixed(1)}` : `${diff.toFixed(1)}`;
                    
                    let statusClass = 'steady';
                    let statusLabel = 'Met expectations';
                    if (diff >= 3.0) {
                      statusClass = 'overperforming';
                      statusLabel = 'Overperformed';
                    } else if (diff <= -3.0) {
                      statusClass = 'underperforming';
                      statusLabel = 'Underperformed';
                    }
                    
                    return (
                      <tr key={idx}>
                        <td style={{ fontWeight: '700' }}>{std.team}</td>
                        <td style={{ textAlign: 'center', fontWeight: '600' }}>{std.actual_wins}W - {std.actual_losses}L</td>
                        <td style={{ textAlign: 'center' }}>{std.simulated_wins}W - {std.simulated_losses}L</td>
                        <td style={{
                          textAlign: 'center',
                          fontWeight: '700',
                          color: diff >= 0 ? 'var(--neon-emerald)' : 'var(--neon-rose)'
                        }}>
                          {diffStr}
                        </td>
                        <td>
                          <span className={`perf-badge ${statusClass}`}>
                            {statusLabel}
                          </span>
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

      {/* Model vs Market Predictive Performance Metrics */}
      {metrics && (
        <div className="glass-card">
          <div className="card-title">
            <span>Model vs. Market Predictor Evaluation</span>
            <span className="badge">Quality Metrics Comparison</span>
          </div>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))', gap: '20px' }}>
            
            {/* Accuracy */}
            <div className="glass-card" style={{ padding: '16px', background: 'rgba(255, 255, 255, 0.015)' }}>
              <span className="control-label" style={{ display: 'block', marginBottom: '12px' }}>Win/Loss Accuracy (%)</span>
              <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
                <div>
                  <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.75rem', marginBottom: '2px' }}>
                    <span>XGBoost Model Prediction</span>
                    <span style={{ fontWeight: '700', color: 'var(--neon-indigo)' }}>{metrics.model.accuracy}%</span>
                  </div>
                  <div className="meter-track"><div className="meter-fill" style={{ width: `${metrics.model.accuracy}%`, background: 'var(--neon-indigo)' }}></div></div>
                </div>
                <div>
                  <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.75rem', marginBottom: '2px' }}>
                    <span>Traditional Bookmakers</span>
                    <span style={{ fontWeight: '700', color: 'var(--neon-purple)' }}>{metrics.bookie.accuracy || 'N/A'}%</span>
                  </div>
                  <div className="meter-track"><div className="meter-fill" style={{ width: `${metrics.bookie.accuracy || 0}%`, background: 'var(--neon-purple)' }}></div></div>
                </div>
                {metrics.polymarket.accuracy && (
                  <div>
                    <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.75rem', marginBottom: '2px' }}>
                      <span>Polymarket Consensus</span>
                      <span style={{ fontWeight: '700', color: 'var(--neon-emerald)' }}>{metrics.polymarket.accuracy}%</span>
                    </div>
                    <div className="meter-track"><div className="meter-fill" style={{ width: `${metrics.polymarket.accuracy}%`, background: 'var(--neon-emerald)' }}></div></div>
                  </div>
                )}
              </div>
            </div>

            {/* Brier Score */}
            <div className="glass-card" style={{ padding: '16px', background: 'rgba(255, 255, 255, 0.015)' }}>
              <span className="control-label" style={{ display: 'block', marginBottom: '12px' }}>Brier Score (Lower is better)</span>
              <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
                <div>
                  <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.75rem', marginBottom: '2px' }}>
                    <span>XGBoost Model</span>
                    <span style={{ fontWeight: '700', color: 'var(--neon-indigo)' }}>{metrics.model.brier_score}</span>
                  </div>
                  <div className="meter-track"><div className="meter-fill" style={{ width: `${(1 - metrics.model.brier_score) * 100}%`, background: 'var(--neon-indigo)' }}></div></div>
                </div>
                <div>
                  <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.75rem', marginBottom: '2px' }}>
                    <span>Traditional Bookmakers</span>
                    <span style={{ fontWeight: '700', color: 'var(--neon-purple)' }}>{metrics.bookie.brier_score || 'N/A'}</span>
                  </div>
                  <div className="meter-track"><div className="meter-fill" style={{ width: `${(1 - (metrics.bookie.brier_score || 1)) * 100}%`, background: 'var(--neon-purple)' }}></div></div>
                </div>
                {metrics.polymarket.brier_score && (
                  <div>
                    <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.75rem', marginBottom: '2px' }}>
                      <span>Polymarket</span>
                      <span style={{ fontWeight: '700', color: 'var(--neon-emerald)' }}>{metrics.polymarket.brier_score}</span>
                    </div>
                    <div className="meter-track"><div className="meter-fill" style={{ width: `${(1 - metrics.polymarket.brier_score) * 100}%`, background: 'var(--neon-emerald)' }}></div></div>
                  </div>
                )}
              </div>
            </div>

            {/* Log Loss */}
            <div className="glass-card" style={{ padding: '16px', background: 'rgba(255, 255, 255, 0.015)' }}>
              <span className="control-label" style={{ display: 'block', marginBottom: '12px' }}>Log Loss (Lower is better)</span>
              <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
                <div>
                  <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.75rem', marginBottom: '2px' }}>
                    <span>XGBoost Model</span>
                    <span style={{ fontWeight: '700', color: 'var(--neon-indigo)' }}>{metrics.model.log_loss}</span>
                  </div>
                  <div className="meter-track"><div className="meter-fill" style={{ width: `${Math.max(10, (1.2 - metrics.model.log_loss) * 80)}%`, background: 'var(--neon-indigo)' }}></div></div>
                </div>
                <div>
                  <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.75rem', marginBottom: '2px' }}>
                    <span>Traditional Bookmakers</span>
                    <span style={{ fontWeight: '700', color: 'var(--neon-purple)' }}>{metrics.bookie.log_loss || 'N/A'}</span>
                  </div>
                  <div className="meter-track"><div className="meter-fill" style={{ width: `${Math.max(10, (1.2 - (metrics.bookie.log_loss || 1.2)) * 80)}%`, background: 'var(--neon-purple)' }}></div></div>
                </div>
                {metrics.polymarket.log_loss && (
                  <div>
                    <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.75rem', marginBottom: '2px' }}>
                      <span>Polymarket</span>
                      <span style={{ fontWeight: '700', color: 'var(--neon-emerald)' }}>{metrics.polymarket.log_loss}</span>
                    </div>
                    <div className="meter-track"><div className="meter-fill" style={{ width: `${Math.max(10, (1.2 - metrics.polymarket.log_loss) * 80)}%`, background: 'var(--neon-emerald)' }}></div></div>
                  </div>
                )}
              </div>
            </div>

          </div>
        </div>
      )}

      {/* Value Bets Log Table */}
      <div className="glass-card">
        <div className="card-title">
          <span>Simulation Value Bets Log</span>
          <span className="badge" style={{ borderColor: 'var(--neon-purple)', color: 'var(--neon-purple)', background: 'rgba(168, 85, 247, 0.08)' }}>
            {valueBets.length} Bets Placed
          </span>
        </div>

        {loading ? (
          <div style={{ padding: '40px', textAlign: 'center', color: 'var(--color-text-muted)' }}>
            Loading bet logs...
          </div>
        ) : valueBets.length === 0 ? (
          <div style={{ padding: '40px', textAlign: 'center', color: 'var(--color-text-muted)' }}>
            No bets placed. Try decreasing the minimum edge threshold parameter.
          </div>
        ) : (
          <div className="table-container">
            <table className="custom-table">
              <thead>
                <tr>
                  <th>Date</th>
                  <th>Matchup</th>
                  <th>Bet Selection</th>
                  <th style={{ textAlign: 'right' }}>Model Prob</th>
                  <th style={{ textAlign: 'right' }}>Market Prob</th>
                  <th style={{ textAlign: 'right' }}>Edge</th>
                  <th style={{ textAlign: 'right' }}>Wager Amount</th>
                  <th style={{ textAlign: 'right' }}>Winnings / Loss</th>
                </tr>
              </thead>
              <tbody>
                {valueBets.map((bet, idx) => {
                  const homeProbPercent = (bet.model_prob_home * 100).toFixed(1);
                  const marketProbPercent = marketSource === 'polymarket' 
                    ? (bet.poly_prob_home * 100).toFixed(1)
                    : (bet.bookie_prob_home * 100).toFixed(1);
                  
                  return (
                    <tr key={idx}>
                      <td>{bet.date}</td>
                      <td>
                        <span style={{ fontWeight: '500' }}>{bet.home_team}</span>
                        <span style={{ color: 'var(--color-text-dim)', margin: '0 6px' }}>vs</span>
                        <span>{bet.away_team}</span>
                        <span style={{ color: 'var(--color-text-muted)', fontSize: '0.75rem', marginLeft: '8px' }}>
                          ({bet.home_score} - {bet.away_score})
                        </span>
                      </td>
                      <td>
                        <span style={{
                          fontWeight: '700',
                          color: bet.bet_team === bet.home_team ? 'var(--neon-emerald)' : 'var(--neon-purple)',
                          marginRight: '6px'
                        }}>
                          {bet.bet_team}
                        </span>
                        <span style={{ color: 'var(--color-text-dim)', fontSize: '0.8rem' }}>
                          (@{bet.bet_odds.toFixed(2)})
                        </span>
                      </td>
                      <td style={{ textAlign: 'right', fontWeight: '600' }}>
                        {bet.bet_team === bet.home_team ? homeProbPercent : (100 - homeProbPercent).toFixed(1)}%
                      </td>
                      <td style={{ textAlign: 'right' }}>
                        {bet.bet_team === bet.home_team ? marketProbPercent : (100 - marketProbPercent).toFixed(1)}%
                      </td>
                      <td style={{ textAlign: 'right', color: 'var(--neon-indigo)', fontWeight: '700' }}>
                        +{(bet.bet_edge * 100).toFixed(1)}%
                      </td>
                      <td style={{ textAlign: 'right', fontWeight: '600' }}>
                        ${bet.bet_wager.toFixed(2)}
                      </td>
                      <td style={{
                        textAlign: 'right',
                        fontWeight: '700',
                        color: bet.bet_win ? 'var(--neon-emerald)' : 'var(--neon-rose)'
                      }}>
                        {bet.bet_win ? '+' : ''}${bet.bet_payout.toFixed(2)}
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
