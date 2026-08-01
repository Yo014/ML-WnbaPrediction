import React from 'react';
import { formatOddsPlaceholder } from './UpcomingBets';

export default function BettingOddsCard({ odds, restDiff, customOdds = {}, onCustomOddsChange, onResetCustomOdds }) {
  const defaultHomeOdds = odds?.bookie_home_odds ?? 1.90;
  const defaultAwayOdds = odds?.bookie_away_odds ?? 1.90;
  const defaultOverOdds = odds?.bookie_over_odds ?? 1.90;
  const defaultUnderOdds = odds?.bookie_under_odds ?? 1.90;
  const impliedProb = odds?.implied_prob_home ?? 50.0;
  const defaultClosingSpread = odds?.closing_spread ?? 0.0;
  const defaultOpeningSpread = odds?.opening_spread ?? 0.0;
  const defaultOverUnder = odds?.over_under ?? 160.0;
  const rest = restDiff ?? 0;

  const hasCustomOverrides = Boolean(
    customOdds.homeOdds || customOdds.awayOdds || customOdds.closingSpread || customOdds.overUnder || customOdds.overOdds || customOdds.underOdds
  );

  const oddsFormat = localStorage.getItem('wnba_odds_format') || 'american';

  const formattedImplied = `${impliedProb.toFixed(1)}%`;
  const formattedOpening = defaultOpeningSpread > 0 ? `+${defaultOpeningSpread.toFixed(1)}` : defaultOpeningSpread.toFixed(1);
  const formattedRest = `${rest > 0 ? '+' : ''}${rest} days`;

  return (
    <div className="glass-card">
      <div className="card-title" style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <span>Market & Custom Odds Setup</span>
        {hasCustomOverrides && (
          <button
            onClick={onResetCustomOdds}
            style={{
              background: 'rgba(239, 68, 68, 0.15)',
              border: '1px solid #ef4444',
              color: '#ef4444',
              borderRadius: '6px',
              padding: '2px 8px',
              fontSize: '0.7rem',
              cursor: 'pointer',
              fontWeight: '600'
            }}
          >
            Reset to Market/ELO
          </button>
        )}
      </div>
      <div className="detail-section">
        <div className="info-grid">
          {/* Custom Moneyline Home / Away Odds */}
          <div className="info-item" style={{ gridColumn: 'span 2' }}>
            <span className="info-label">Custom Moneyline Odds (Home / Away)</span>
            <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginTop: '4px' }}>
              <input
                type="text"
                placeholder={formatOddsPlaceholder(defaultHomeOdds, oddsFormat)}
                value={customOdds.homeOdds || ''}
                onChange={(e) => onCustomOddsChange && onCustomOddsChange('homeOdds', e.target.value)}
                style={{
                  width: '100px',
                  background: 'rgba(255, 255, 255, 0.05)',
                  border: customOdds.homeOdds ? '1px solid var(--neon-amber)' : '1px solid var(--border-card)',
                  color: 'var(--color-text-main)',
                  borderRadius: '6px',
                  padding: '6px 8px',
                  fontSize: '0.85rem',
                  textAlign: 'center',
                  outline: 'none'
                }}
              />
              <span style={{ color: 'var(--color-text-dim)' }}>/</span>
              <input
                type="text"
                placeholder={formatOddsPlaceholder(defaultAwayOdds, oddsFormat)}
                value={customOdds.awayOdds || ''}
                onChange={(e) => onCustomOddsChange && onCustomOddsChange('awayOdds', e.target.value)}
                style={{
                  width: '100px',
                  background: 'rgba(255, 255, 255, 0.05)',
                  border: customOdds.awayOdds ? '1px solid var(--neon-amber)' : '1px solid var(--border-card)',
                  color: 'var(--color-text-main)',
                  borderRadius: '6px',
                  padding: '6px 8px',
                  fontSize: '0.85rem',
                  textAlign: 'center',
                  outline: 'none'
                }}
              />
            </div>
          </div>

          {/* Custom Over / Under Odds */}
          <div className="info-item" style={{ gridColumn: 'span 2' }}>
            <span className="info-label">Custom Over / Under Odds (Over / Under)</span>
            <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginTop: '4px' }}>
              <input
                type="text"
                placeholder={formatOddsPlaceholder(defaultOverOdds, oddsFormat)}
                value={customOdds.overOdds || ''}
                onChange={(e) => onCustomOddsChange && onCustomOddsChange('overOdds', e.target.value)}
                style={{
                  width: '100px',
                  background: 'rgba(255, 255, 255, 0.05)',
                  border: customOdds.overOdds ? '1px solid var(--neon-amber)' : '1px solid var(--border-card)',
                  color: 'var(--color-text-main)',
                  borderRadius: '6px',
                  padding: '6px 8px',
                  fontSize: '0.85rem',
                  textAlign: 'center',
                  outline: 'none'
                }}
              />
              <span style={{ color: 'var(--color-text-dim)' }}>/</span>
              <input
                type="text"
                placeholder={formatOddsPlaceholder(defaultUnderOdds, oddsFormat)}
                value={customOdds.underOdds || ''}
                onChange={(e) => onCustomOddsChange && onCustomOddsChange('underOdds', e.target.value)}
                style={{
                  width: '100px',
                  background: 'rgba(255, 255, 255, 0.05)',
                  border: customOdds.underOdds ? '1px solid var(--neon-amber)' : '1px solid var(--border-card)',
                  color: 'var(--color-text-main)',
                  borderRadius: '6px',
                  padding: '6px 8px',
                  fontSize: '0.85rem',
                  textAlign: 'center',
                  outline: 'none'
                }}
              />
            </div>
          </div>

          {/* Market Closing Spread Line */}
          <div className="info-item">
            <span className="info-label">Closing Spread Line</span>
            <input
              type="number"
              step="0.5"
              placeholder={defaultClosingSpread > 0 ? `+${defaultClosingSpread.toFixed(1)}` : defaultClosingSpread.toFixed(1)}
              value={customOdds.closingSpread || ''}
              onChange={(e) => onCustomOddsChange && onCustomOddsChange('closingSpread', e.target.value)}
              style={{
                width: '100%',
                background: 'rgba(255, 255, 255, 0.05)',
                border: customOdds.closingSpread ? '1px solid var(--neon-amber)' : '1px solid var(--border-card)',
                color: 'var(--color-text-main)',
                borderRadius: '6px',
                padding: '6px 8px',
                fontSize: '0.85rem',
                textAlign: 'center',
                outline: 'none',
                marginTop: '4px'
              }}
            />
          </div>

          {/* Over / Under Total Score Line */}
          <div className="info-item">
            <span className="info-label">Over / Under Total Line</span>
            <input
              type="number"
              step="0.5"
              placeholder={defaultOverUnder.toFixed(1)}
              value={customOdds.overUnder || ''}
              onChange={(e) => onCustomOddsChange && onCustomOddsChange('overUnder', e.target.value)}
              style={{
                width: '100%',
                background: 'rgba(255, 255, 255, 0.05)',
                border: customOdds.overUnder ? '1px solid var(--neon-amber)' : '1px solid var(--border-card)',
                color: 'var(--color-text-main)',
                borderRadius: '6px',
                padding: '6px 8px',
                fontSize: '0.85rem',
                textAlign: 'center',
                outline: 'none',
                marginTop: '4px'
              }}
            />
          </div>

          <div className="info-item">
            <span className="info-label">Bookie Implied Home Prob</span>
            <span className="info-value">{formattedImplied}</span>
          </div>

          <div className="info-item">
            <span className="info-label">Rest Days Difference</span>
            <span className="info-value">{formattedRest}</span>
          </div>
        </div>
      </div>
    </div>
  );
}
