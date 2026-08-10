import React, { useState, useEffect } from 'react';

const ALL_FRANCHISES = [
  'Atlanta Dream',
  'Chicago Sky',
  'Connecticut Sun',
  'Dallas Wings',
  'Golden State Valkyries',
  'Indiana Fever',
  'Las Vegas Aces',
  'Los Angeles Sparks',
  'Minnesota Lynx',
  'New York Liberty',
  'Phoenix Mercury',
  'Portland Fire',
  'Seattle Storm',
  'Toronto Tempo',
  'Washington Mystics'
];

export default function TotalsAnalyticsCard() {
  const [activeTab, setActiveTab] = useState('league'); // 'league' | 'h2h'

  // League Totals State
  const [leagueSeason, setLeagueSeason] = useState('2026');
  const [leagueWindow, setLeagueWindow] = useState('all'); // 'all', '5', '10'
  const [leagueData, setLeagueData] = useState(null);
  const [leagueLoading, setLeagueLoading] = useState(false);
  const [leagueError, setLeagueError] = useState(null);
  const [sortField, setSortField] = useState('avg_total');
  const [sortDirection, setSortDirection] = useState('desc');

  // Modal State for Team Game History
  const [modalTeam, setModalTeam] = useState(null);
  const [modalData, setModalData] = useState(null);
  const [modalLoading, setModalLoading] = useState(false);
  const [modalError, setModalError] = useState(null);

  // Hover states for table interactive UX
  const [hoverRowIndex, setHoverRowIndex] = useState(null);
  const [hoverBadgeIndex, setHoverBadgeIndex] = useState(null);

  // H2H State
  const [teamA, setTeamA] = useState('Toronto Tempo');
  const [teamB, setTeamB] = useState('Indiana Fever');
  const [h2hWindow, setH2hWindow] = useState('10'); // 'all', '5', '10'
  const [h2hSeason, setH2hSeason] = useState('all');
  const [h2hData, setH2hData] = useState(null);
  const [h2hLoading, setH2hLoading] = useState(false);
  const [h2hError, setH2hError] = useState(null);

  // Fetch League Totals Data
  useEffect(() => {
    if (activeTab !== 'league') return;
    setLeagueLoading(true);
    setLeagueError(null);

    fetch(`/api/team_totals?season=${leagueSeason}&window=${leagueWindow}`)
      .then(res => {
        if (!res.ok) throw new Error('Failed to fetch league totals data');
        return res.json();
      })
      .then(data => {
        setLeagueData(data);
        setLeagueLoading(false);
      })
      .catch(err => {
        setLeagueError(err.message);
        setLeagueLoading(false);
      });
  }, [activeTab, leagueSeason, leagueWindow]);

  // Fetch Team Game History for Modal
  useEffect(() => {
    if (!modalTeam) {
      setModalData(null);
      setModalError(null);
      return;
    }
    setModalLoading(true);
    setModalError(null);

    const teamParam = modalTeam.team_name || modalTeam.team_abbr;
    fetch(`/api/team_game_history?team=${encodeURIComponent(teamParam)}&season=${leagueSeason}&window=${leagueWindow}`)
      .then(res => {
        if (!res.ok) throw new Error('Failed to fetch team game history');
        return res.json();
      })
      .then(data => {
        setModalData(data);
        setModalLoading(false);
      })
      .catch(err => {
        setModalError(err.message);
        setModalLoading(false);
      });
  }, [modalTeam, leagueSeason, leagueWindow]);

  // Fetch H2H Analytics Data
  useEffect(() => {
    if (activeTab !== 'h2h') return;
    setH2hLoading(true);
    setH2hError(null);

    const query = new URLSearchParams({
      team_a: teamA,
      team_b: teamB,
      window: h2hWindow,
      season: h2hSeason
    });

    fetch(`/api/h2h_analytics?${query.toString()}`)
      .then(res => {
        if (!res.ok) throw new Error('Failed to fetch H2H analytics data');
        return res.json();
      })
      .then(data => {
        setH2hData(data);
        setH2hLoading(false);
      })
      .catch(err => {
        setH2hError(err.message);
        setH2hLoading(false);
      });
  }, [activeTab, teamA, teamB, h2hWindow, h2hSeason]);

  // Handle Sort
  const handleSort = (field) => {
    if (sortField === field) {
      setSortDirection(prev => (prev === 'asc' ? 'desc' : 'asc'));
    } else {
      setSortField(field);
      setSortDirection('desc');
    }
  };

  const getSortedTeams = () => {
    if (!leagueData || !leagueData.teams) return [];
    return [...leagueData.teams].sort((a, b) => {
      let valA = a[sortField];
      let valB = b[sortField];
      if (typeof valA === 'string') {
        valA = valA.toLowerCase();
        valB = valB.toLowerCase();
      }
      if (valA < valB) return sortDirection === 'asc' ? -1 : 1;
      if (valA > valB) return sortDirection === 'asc' ? 1 : -1;
      return 0;
    });
  };

  return (
    <div className="card shadow-lg mb-4 glassmorphic-card" style={{ background: 'rgba(15, 23, 42, 0.75)', border: '1px solid rgba(255, 255, 255, 0.1)', backdropFilter: 'blur(12px)' }}>
      <div className="card-header d-flex flex-wrap justify-content-between align-items-center gap-3" style={{ borderBottom: '1px solid rgba(255, 255, 255, 0.1)', background: 'rgba(30, 41, 59, 0.5)' }}>
        <div className="d-flex align-items-center gap-2">
          <span style={{ fontSize: '1.5rem' }}>📊</span>
          <h4 className="m-0 font-weight-bold text-white">Totals & Head-to-Head Analytics</h4>
        </div>

        {/* Mode Selector Tabs */}
        <div className="btn-group" role="group">
          <button
            className={`btn btn-sm ${activeTab === 'league' ? 'btn-primary font-weight-bold' : 'btn-outline-secondary text-white'}`}
            onClick={() => setActiveTab('league')}
            style={{ borderRadius: '8px 0 0 8px', padding: '8px 16px' }}
          >
            🏀 League Team Totals
          </button>
          <button
            className={`btn btn-sm ${activeTab === 'h2h' ? 'btn-primary font-weight-bold' : 'btn-outline-secondary text-white'}`}
            onClick={() => setActiveTab('h2h')}
            style={{ borderRadius: '0 8px 8px 0', padding: '8px 16px' }}
          >
            ⚔️ Head-to-Head Explorer
          </button>
        </div>
      </div>

      <div className="card-body text-white">
        {/* LEAGUE TEAM TOTALS TAB */}
        {activeTab === 'league' && (
          <div>
            {/* Filter Toolbar */}
            <div className="row g-3 align-items-center mb-4 p-3 rounded" style={{ background: 'rgba(30, 41, 59, 0.4)', border: '1px solid rgba(255, 255, 255, 0.05)' }}>
              <div className="col-12 col-md-4">
                <label className="text-muted small font-weight-bold mb-1">SELECT SEASON</label>
                <select
                  className="form-select bg-dark text-white border-secondary"
                  value={leagueSeason}
                  onChange={(e) => setLeagueSeason(e.target.value)}
                >
                  <option value="2026">2026 Season (Current)</option>
                  <option value="2025">2025 Season</option>
                  <option value="2024">2024 Season</option>
                  <option value="2023">2023 Season</option>
                  <option value="2022">2022 Season</option>
                  <option value="all">All Seasons (Historical)</option>
                </select>
              </div>

              <div className="col-12 col-md-4">
                <label className="text-muted small font-weight-bold mb-1">GAME WINDOW</label>
                <select
                  className="form-select bg-dark text-white border-secondary"
                  value={leagueWindow}
                  onChange={(e) => setLeagueWindow(e.target.value)}
                >
                  <option value="all">Full Season / Window</option>
                  <option value="10">Last 10 Games</option>
                  <option value="5">Last 5 Games</option>
                </select>
              </div>

              <div className="col-12 col-md-4 text-md-end">
                <span className="badge bg-indigo-soft text-indigo p-2" style={{ background: 'rgba(99, 102, 241, 0.15)', color: '#818cf8', border: '1px solid rgba(99, 102, 241, 0.3)' }}>
                  15 Franchises Evaluated
                </span>
              </div>
            </div>

            {/* Error or Loading State */}
            {leagueLoading && <div className="text-center py-5"><div className="spinner-border text-primary" role="status"></div><p className="mt-2 text-muted">Loading team totals analytics...</p></div>}
            {leagueError && <div className="alert alert-danger">{leagueError}</div>}

            {/* League Data Table */}
            {!leagueLoading && !leagueError && leagueData && (
              <div className="table-responsive">
                <table className="table table-dark table-hover align-middle" style={{ borderRadius: '8px', overflow: 'hidden', border: '1px solid rgba(255, 255, 255, 0.05)' }}>
                  <thead style={{ background: 'rgba(30, 41, 59, 0.9)' }}>
                    <tr>
                      <th onClick={() => handleSort('team_name')} style={{ cursor: 'pointer' }}>Franchise {sortField === 'team_name' ? (sortDirection === 'asc' ? '▲' : '▼') : ''}</th>
                      <th onClick={() => handleSort('gp')} style={{ cursor: 'pointer', textAlign: 'center' }}>GP</th>
                      <th onClick={() => handleSort('pts_for')} style={{ cursor: 'pointer', textAlign: 'center' }}>Offense (Pts For)</th>
                      <th onClick={() => handleSort('pts_against')} style={{ cursor: 'pointer', textAlign: 'center' }}>Defense (Pts Against)</th>
                      <th onClick={() => handleSort('avg_total')} style={{ cursor: 'pointer', textAlign: 'center' }}>Avg Combined Total</th>
                      <th onClick={() => handleSort('avg_line')} style={{ cursor: 'pointer', textAlign: 'center' }}>Avg Market Line</th>
                      <th onClick={() => handleSort('diff')} style={{ cursor: 'pointer', textAlign: 'center' }}>Line Diff (+/-)</th>
                      <th onClick={() => handleSort('over_pct')} style={{ cursor: 'pointer', textAlign: 'center' }}>OVER Hit Rate %</th>
                    </tr>
                  </thead>
                  <tbody>
                    {getSortedTeams().map((team, idx) => {
                      const isHighOver = team.over_pct >= 75;
                      const isLowOver = team.over_pct < 50;
                      const hideBadge = team.team_abbr && (team.team_abbr === team.team_name || team.team_name.startsWith(team.team_abbr));
                      const isRowHovered = hoverRowIndex === idx;
                      const isBadgeHovered = hoverBadgeIndex === idx;

                      return (
                        <tr
                          key={team.team_abbr || team.team_name}
                          onClick={() => setModalTeam(team)}
                          onMouseEnter={() => setHoverRowIndex(idx)}
                          onMouseLeave={() => setHoverRowIndex(null)}
                          style={{
                            background: isRowHovered ? 'rgba(51, 65, 85, 0.5)' : (idx % 2 === 0 ? 'rgba(15, 23, 42, 0.4)' : 'rgba(30, 41, 59, 0.2)'),
                            cursor: 'pointer',
                            transition: 'background-color 0.15s ease-in-out'
                          }}
                        >
                          <td>
                            <strong className="text-white">{team.team_name}</strong>
                            {!hideBadge && (
                              <span className="badge bg-secondary ms-2 text-uppercase">{team.team_abbr}</span>
                            )}
                          </td>
                          <td style={{ textAlign: 'center' }}>{team.gp}</td>
                          <td style={{ textAlign: 'center', color: '#60a5fa', fontWeight: '600' }}>{team.pts_for}</td>
                          <td style={{ textAlign: 'center', color: '#f87171', fontWeight: '600' }}>{team.pts_against}</td>
                          <td style={{ textAlign: 'center', fontWeight: 'bold', fontSize: '1.05rem', color: '#f59e0b' }}>
                            {team.avg_total} pts
                          </td>
                          <td style={{ textAlign: 'center', color: '#9ca3af' }}>{team.avg_line} pts</td>
                          <td style={{ textAlign: 'center', fontWeight: '600', color: team.diff > 0 ? '#34d399' : '#f87171' }}>
                            {team.diff > 0 ? `+${team.diff}` : team.diff} pts
                          </td>
                          <td style={{ textAlign: 'center' }}>
                            <span
                              className="badge p-2"
                              onClick={(e) => {
                                e.stopPropagation();
                                setModalTeam(team);
                              }}
                              onMouseEnter={() => setHoverBadgeIndex(idx)}
                              onMouseLeave={() => setHoverBadgeIndex(null)}
                              style={{
                                background: isHighOver ? 'rgba(16, 185, 129, 0.2)' : isLowOver ? 'rgba(239, 68, 68, 0.2)' : 'rgba(245, 158, 11, 0.2)',
                                color: isHighOver ? '#34d399' : isLowOver ? '#f87171' : '#fbbf24',
                                border: `1px solid ${isHighOver ? '#10b981' : isLowOver ? '#ef4444' : '#f59e0b'}`,
                                cursor: 'pointer',
                                transition: 'all 0.15s ease-in-out',
                                transform: isBadgeHovered ? 'scale(1.06)' : 'scale(1)',
                                boxShadow: isBadgeHovered ? `0 0 10px ${isHighOver ? '#10b981' : isLowOver ? '#ef4444' : '#f59e0b'}` : 'none'
                              }}
                            >
                              {team.over_pct}% ({team.over_hits}/{team.gp})
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
        )}

        {/* HEAD-TO-HEAD EXPLORER TAB */}
        {activeTab === 'h2h' && (
          <div>
            {/* H2H Selectors Toolbar */}
            <div className="row g-3 align-items-center mb-4 p-3 rounded" style={{ background: 'rgba(30, 41, 59, 0.4)', border: '1px solid rgba(255, 255, 255, 0.05)' }}>
              <div className="col-12 col-md-3">
                <label className="text-muted small font-weight-bold mb-1">TEAM A</label>
                <select
                  className="form-select bg-dark text-white border-secondary"
                  value={teamA}
                  onChange={(e) => setTeamA(e.target.value)}
                >
                  {ALL_FRANCHISES.map(t => (
                    <option key={t} value={t}>{t}</option>
                  ))}
                </select>
              </div>

              <div className="col-12 col-md-1 text-center font-weight-bold text-muted">
                VS
              </div>

              <div className="col-12 col-md-3">
                <label className="text-muted small font-weight-bold mb-1">TEAM B</label>
                <select
                  className="form-select bg-dark text-white border-secondary"
                  value={teamB}
                  onChange={(e) => setTeamB(e.target.value)}
                >
                  {ALL_FRANCHISES.map(t => (
                    <option key={t} value={t}>{t}</option>
                  ))}
                </select>
              </div>

              <div className="col-12 col-md-3">
                <label className="text-muted small font-weight-bold mb-1">H2H WINDOW</label>
                <select
                  className="form-select bg-dark text-white border-secondary"
                  value={h2hWindow}
                  onChange={(e) => setH2hWindow(e.target.value)}
                >
                  <option value="10">Last 10 H2H Meetings</option>
                  <option value="5">Last 5 H2H Meetings</option>
                  <option value="all">All H2H Meetings</option>
                </select>
              </div>

              <div className="col-12 col-md-2">
                <label className="text-muted small font-weight-bold mb-1">SEASON</label>
                <select
                  className="form-select bg-dark text-white border-secondary"
                  value={h2hSeason}
                  onChange={(e) => setH2hSeason(e.target.value)}
                >
                  <option value="all">All Seasons</option>
                  <option value="2026">2026 Season</option>
                  <option value="2025">2025 Season</option>
                  <option value="2024">2024 Season</option>
                </select>
              </div>
            </div>

            {/* Error or Loading State */}
            {h2hLoading && <div className="text-center py-5"><div className="spinner-border text-primary" role="status"></div><p className="mt-2 text-muted">Fetching head-to-head match analytics...</p></div>}
            {h2hError && <div className="alert alert-danger">{h2hError}</div>}

            {/* H2H Stat Cards & Game Log */}
            {!h2hLoading && !h2hError && h2hData && (
              <div>
                {/* Summary Stat Cards */}
                <div className="row g-3 mb-4">
                  <div className="col-12 col-sm-6 col-lg-3">
                    <div className="p-3 rounded text-center" style={{ background: 'rgba(30, 41, 59, 0.6)', border: '1px solid rgba(255, 255, 255, 0.05)' }}>
                      <span className="text-muted small font-weight-bold d-block mb-1">H2H MATCH RECORD</span>
                      <h4 className="m-0 text-white font-weight-bold">
                        {h2hData.team_a_wins} - {h2hData.team_b_wins}
                      </h4>
                      <small className="text-muted">({h2hData.total_games} Total Meetings)</small>
                    </div>
                  </div>

                  <div className="col-12 col-sm-6 col-lg-3">
                    <div className="p-3 rounded text-center" style={{ background: 'rgba(30, 41, 59, 0.6)', border: '1px solid rgba(255, 255, 255, 0.05)' }}>
                      <span className="text-muted small font-weight-bold d-block mb-1">AVG COMBINED SCORE</span>
                      <h4 className="m-0 text-warning font-weight-bold">{h2hData.avg_total_pts} pts</h4>
                      <small className="text-muted">vs Line Avg: {h2hData.avg_line} pts ({h2hData.line_diff > 0 ? `+${h2hData.line_diff}` : h2hData.line_diff})</small>
                    </div>
                  </div>

                  <div className="col-12 col-sm-6 col-lg-3">
                    <div className="p-3 rounded text-center" style={{ background: 'rgba(30, 41, 59, 0.6)', border: '1px solid rgba(255, 255, 255, 0.05)' }}>
                      <span className="text-muted small font-weight-bold d-block mb-1">OVER HIT RATE</span>
                      <h4 className="m-0 text-emerald font-weight-bold" style={{ color: h2hData.over_pct >= 60 ? '#34d399' : '#f87171' }}>
                        {h2hData.over_pct}%
                      </h4>
                      <small className="text-muted">({h2hData.over_hits} / {h2hData.total_games} games OVER)</small>
                    </div>
                  </div>

                  <div className="col-12 col-sm-6 col-lg-3">
                    <div className="p-3 rounded text-center" style={{ background: 'rgba(30, 41, 59, 0.6)', border: '1px solid rgba(255, 255, 255, 0.05)' }}>
                      <span className="text-muted small font-weight-bold d-block mb-1">AVG SCORE SPLIT</span>
                      <h5 className="m-0 text-white font-weight-bold">
                        {h2hData.avg_team_a_pts} - {h2hData.avg_team_b_pts}
                      </h5>
                      <small className="text-muted">({h2hData.team_a} vs {h2hData.team_b})</small>
                    </div>
                  </div>
                </div>

                {/* H2H Game Log Table */}
                <h5 className="font-weight-bold mb-3 text-white">Past Head-to-Head Games</h5>
                {h2hData.matches.length === 0 ? (
                  <div className="alert alert-info">No head-to-head matches found for the selected filters.</div>
                ) : (
                  <div className="table-responsive">
                    <table className="table table-dark table-hover align-middle" style={{ borderRadius: '8px', overflow: 'hidden', border: '1px solid rgba(255, 255, 255, 0.05)' }}>
                      <thead style={{ background: 'rgba(30, 41, 59, 0.9)' }}>
                        <tr>
                          <th>Date</th>
                          <th>Matchup (Home vs Away)</th>
                          <th style={{ textAlign: 'center' }}>Final Score</th>
                          <th style={{ textAlign: 'center' }}>Combined Total</th>
                          <th style={{ textAlign: 'center' }}>Market O/U Line</th>
                          <th style={{ textAlign: 'center' }}>O/U Outcome</th>
                        </tr>
                      </thead>
                      <tbody>
                        {h2hData.matches.map((m, idx) => (
                          <tr key={idx} style={{ background: idx % 2 === 0 ? 'rgba(15, 23, 42, 0.4)' : 'rgba(30, 41, 59, 0.2)' }}>
                            <td>{m.date} <span className="badge bg-secondary ms-1">{m.season}</span></td>
                            <td>
                              <strong>{m.home_team}</strong> vs <strong>{m.away_team}</strong>
                            </td>
                            <td style={{ textAlign: 'center', fontWeight: 'bold' }}>
                              {m.home_score} - {m.away_score}
                            </td>
                            <td style={{ textAlign: 'center', color: '#f59e0b', fontWeight: 'bold' }}>
                              {m.total_score} pts
                            </td>
                            <td style={{ textAlign: 'center', color: '#9ca3af' }}>
                              {m.over_under} pts
                            </td>
                            <td style={{ textAlign: 'center' }}>
                              <span
                                className="badge p-2"
                                style={{
                                  background: m.is_over ? 'rgba(16, 185, 129, 0.2)' : 'rgba(239, 68, 68, 0.2)',
                                  color: m.is_over ? '#34d399' : '#f87171',
                                  border: `1px solid ${m.is_over ? '#10b981' : '#ef4444'}`
                                }}
                              >
                                {m.is_over ? 'OVER' : 'UNDER'}
                              </span>
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                )}
              </div>
            )}
          </div>
        )}
      </div>

      {/* TEAM GAME HISTORY MODAL */}
      {modalTeam && (
        <div
          className="modal-backdrop-custom"
          onClick={(e) => {
            if (e.target === e.currentTarget) setModalTeam(null);
          }}
          style={{
            position: 'fixed',
            top: 0,
            left: 0,
            right: 0,
            bottom: 0,
            backgroundColor: 'rgba(0, 0, 0, 0.75)',
            backdropFilter: 'blur(8px)',
            zIndex: 1050,
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            padding: '20px'
          }}
        >
          <div
            className="modal-content-custom"
            style={{
              background: 'rgba(15, 23, 42, 0.95)',
              border: '1px solid rgba(255, 255, 255, 0.15)',
              borderRadius: '16px',
              boxShadow: '0 25px 50px -12px rgba(0, 0, 0, 0.7)',
              width: '100%',
              maxWidth: '950px',
              maxHeight: '90vh',
              display: 'flex',
              flexDirection: 'column',
              overflow: 'hidden',
              color: '#fff'
            }}
          >
            {/* Modal Header */}
            <div
              className="p-3 p-md-4 d-flex justify-content-between align-items-center"
              style={{
                borderBottom: '1px solid rgba(255, 255, 255, 0.1)',
                background: 'rgba(30, 41, 59, 0.8)'
              }}
            >
              <div className="d-flex align-items-center gap-3 flex-wrap">
                <div className="d-flex align-items-center gap-2">
                  <span style={{ fontSize: '1.5rem' }}>🏀</span>
                  <h4 className="m-0 font-weight-bold text-white">
                    {modalTeam.team_name}
                  </h4>
                  {modalTeam.team_abbr && modalTeam.team_abbr !== modalTeam.team_name && !modalTeam.team_name.startsWith(modalTeam.team_abbr) && (
                    <span className="badge bg-secondary text-uppercase" style={{ fontSize: '0.85rem' }}>
                      {modalTeam.team_abbr}
                    </span>
                  )}
                </div>
                <div className="d-flex gap-2">
                  <span
                    className="badge"
                    style={{
                      background: 'rgba(99, 102, 241, 0.2)',
                      color: '#818cf8',
                      border: '1px solid rgba(99, 102, 241, 0.4)',
                      padding: '6px 12px',
                      fontSize: '0.8rem'
                    }}
                  >
                    {leagueSeason === 'all' ? 'All Seasons' : `${leagueSeason} Season`}
                  </span>
                  <span
                    className="badge"
                    style={{
                      background: 'rgba(168, 85, 247, 0.2)',
                      color: '#c084fc',
                      border: '1px solid rgba(168, 85, 247, 0.4)',
                      padding: '6px 12px',
                      fontSize: '0.8rem'
                    }}
                  >
                    {leagueWindow === 'all' ? 'Full Window' : `Last ${leagueWindow} Games`}
                  </span>
                </div>
              </div>

              <button
                type="button"
                className="btn-close btn-close-white"
                onClick={() => setModalTeam(null)}
                aria-label="Close"
                style={{ cursor: 'pointer' }}
              />
            </div>

            {/* Modal Body */}
            <div className="p-3 p-md-4" style={{ overflowY: 'auto', flex: 1 }}>
              {modalLoading && (
                <div className="text-center py-5">
                  <div className="spinner-border text-primary" role="status"></div>
                  <p className="mt-2 text-muted">Fetching game history for {modalTeam.team_name}...</p>
                </div>
              )}

              {modalError && (
                <div className="alert alert-danger mb-0">
                  {modalError}
                </div>
              )}

              {!modalLoading && !modalError && modalData && (
                <div>
                  {/* Summary Stat Bar */}
                  <div className="row g-2 mb-4">
                    <div className="col-6 col-md-2">
                      <div className="p-3 rounded text-center" style={{ background: 'rgba(30, 41, 59, 0.6)', border: '1px solid rgba(255, 255, 255, 0.08)' }}>
                        <span className="text-muted small d-block font-weight-bold mb-1">TOTAL GAMES</span>
                        <span className="h4 m-0 font-weight-bold text-white">{modalData.total_games}</span>
                      </div>
                    </div>

                    <div className="col-6 col-md-3">
                      <div className="p-3 rounded text-center" style={{ background: 'rgba(30, 41, 59, 0.6)', border: '1px solid rgba(255, 255, 255, 0.08)' }}>
                        <span className="text-muted small d-block font-weight-bold mb-1">O / U / P RECORD</span>
                        <span className="h5 m-0 font-weight-bold">
                          <span style={{ color: '#34d399' }}>{modalData.over_hits} OVER</span> - <span style={{ color: '#f87171' }}>{modalData.under_hits} UNDER</span>
                          {modalData.push_hits > 0 && <span style={{ color: '#fbbf24' }}> - {modalData.push_hits} PUSH</span>}
                        </span>
                      </div>
                    </div>

                    <div className="col-6 col-md-2">
                      <div className="p-3 rounded text-center" style={{ background: 'rgba(30, 41, 59, 0.6)', border: '1px solid rgba(255, 255, 255, 0.08)' }}>
                        <span className="text-muted small d-block font-weight-bold mb-1">OVER HIT %</span>
                        <span className="h4 m-0 font-weight-bold" style={{ color: modalData.over_pct >= 55 ? '#34d399' : modalData.over_pct < 45 ? '#f87171' : '#fbbf24' }}>
                          {modalData.over_pct}%
                        </span>
                      </div>
                    </div>

                    <div className="col-6 col-md-2">
                      <div className="p-3 rounded text-center" style={{ background: 'rgba(30, 41, 59, 0.6)', border: '1px solid rgba(255, 255, 255, 0.08)' }}>
                        <span className="text-muted small d-block font-weight-bold mb-1">AVG TOTAL</span>
                        <span className="h5 m-0 font-weight-bold text-warning">{modalData.avg_total_pts} pts</span>
                      </div>
                    </div>

                    <div className="col-6 col-md-3">
                      <div className="p-3 rounded text-center" style={{ background: 'rgba(30, 41, 59, 0.6)', border: '1px solid rgba(255, 255, 255, 0.08)' }}>
                        <span className="text-muted small d-block font-weight-bold mb-1">AVG LINE / DIFF</span>
                        <span className="h5 m-0 font-weight-bold text-white">
                          {modalData.avg_line} pts{' '}
                          <small style={{ color: modalData.line_diff > 0 ? '#34d399' : modalData.line_diff < 0 ? '#f87171' : '#9ca3af', fontWeight: 'bold' }}>
                            ({modalData.line_diff > 0 ? `+${modalData.line_diff}` : modalData.line_diff})
                          </small>
                        </span>
                      </div>
                    </div>
                  </div>

                  {/* Scrollable Game History Table */}
                  <div className="table-responsive" style={{ maxHeight: '420px', overflowY: 'auto' }}>
                    <table className="table table-dark table-hover align-middle mb-0" style={{ borderRadius: '8px', overflow: 'hidden', border: '1px solid rgba(255, 255, 255, 0.08)' }}>
                      <thead style={{ position: 'sticky', top: 0, zIndex: 2, background: 'rgba(30, 41, 59, 0.95)' }}>
                        <tr>
                          <th>Date / Season</th>
                          <th>Matchup</th>
                          <th style={{ textAlign: 'center' }}>Score</th>
                          <th style={{ textAlign: 'center' }}>Combined Total</th>
                          <th style={{ textAlign: 'center' }}>Market Line</th>
                          <th style={{ textAlign: 'center' }}>Line Diff</th>
                          <th style={{ textAlign: 'center' }}>Outcome</th>
                        </tr>
                      </thead>
                      <tbody>
                        {modalData.games && modalData.games.length > 0 ? (
                          modalData.games.map((g, idx) => {
                            const isOver = g.outcome === 'OVER';
                            const isUnder = g.outcome === 'UNDER';
                            return (
                              <tr key={idx} style={{ background: idx % 2 === 0 ? 'rgba(15, 23, 42, 0.4)' : 'rgba(30, 41, 59, 0.2)' }}>
                                <td>
                                  <span className="text-white font-weight-bold">{g.date}</span>
                                  <span className="badge bg-secondary ms-2">{g.season}</span>
                                </td>
                                <td>
                                  <strong>{g.home_team}</strong> vs <strong>{g.away_team}</strong>
                                  <span
                                    className="badge ms-2"
                                    style={{
                                      background: g.is_home ? 'rgba(16, 185, 129, 0.15)' : 'rgba(59, 130, 246, 0.15)',
                                      color: g.is_home ? '#34d399' : '#60a5fa',
                                      border: `1px solid ${g.is_home ? '#10b981' : '#3b82f6'}`,
                                      fontSize: '0.7rem'
                                    }}
                                  >
                                    {g.is_home ? 'HOME' : 'AWAY'}
                                  </span>
                                </td>
                                <td style={{ textAlign: 'center', fontWeight: 'bold' }}>
                                  {g.home_score} - {g.away_score}
                                </td>
                                <td style={{ textAlign: 'center', color: '#f59e0b', fontWeight: 'bold' }}>
                                  {g.total_score} pts
                                </td>
                                <td style={{ textAlign: 'center', color: '#9ca3af' }}>
                                  {g.over_under} pts
                                </td>
                                <td style={{ textAlign: 'center', fontWeight: '600', color: g.line_diff > 0 ? '#34d399' : g.line_diff < 0 ? '#f87171' : '#9ca3af' }}>
                                  {g.line_diff > 0 ? `+${g.line_diff}` : g.line_diff} pts
                                </td>
                                <td style={{ textAlign: 'center' }}>
                                  <span
                                    className="badge p-2"
                                    style={{
                                      background: isOver ? 'rgba(16, 185, 129, 0.2)' : isUnder ? 'rgba(239, 68, 68, 0.2)' : 'rgba(245, 158, 11, 0.2)',
                                      color: isOver ? '#34d399' : isUnder ? '#f87171' : '#fbbf24',
                                      border: `1px solid ${isOver ? '#10b981' : isUnder ? '#ef4444' : '#f59e0b'}`
                                    }}
                                  >
                                    {g.outcome}
                                  </span>
                                </td>
                              </tr>
                            );
                          })
                        ) : (
                          <tr>
                            <td colSpan="7" className="text-center py-4 text-muted">
                              No game history found for this team and selected filters.
                            </td>
                          </tr>
                        )}
                      </tbody>
                    </table>
                  </div>
                </div>
              )}
            </div>

            {/* Modal Footer */}
            <div
              className="p-3 d-flex justify-content-end"
              style={{
                borderTop: '1px solid rgba(255, 255, 255, 0.1)',
                background: 'rgba(30, 41, 59, 0.8)'
              }}
            >
              <button
                type="button"
                className="btn btn-secondary px-4 font-weight-bold"
                onClick={() => setModalTeam(null)}
              >
                Close
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

