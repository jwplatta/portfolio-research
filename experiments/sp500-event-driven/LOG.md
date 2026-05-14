# SP500 Event Driven Log

## Gap Fade

### v0
- Baseline overnight gap fade.
- Raw custom signal, demeaned cross-section, liquidity constraint, equal-weight long/short.

### v1
- Tighten event selection.
- Larger absolute gap threshold and lower allowed opening participation.

### v2
- Add low-volatility conditioning.
- Remove high-volatility names to reduce noisy reversal trades.

### v3
- Add weak-trend conditioning.
- Avoid strongly trending stocks where fading gaps is less reliable.

### v4
- Add inverse-vol weighting.
- Same filtered candidate set as v3, but size down volatile names.

### v5
- Add portfolio-level volatility targeting.
- Same setup as v4 with a 18% annualized volatility target.

## Volume Shock Continuation

### v6
- Baseline volume shock continuation.
- Five-day move times abnormal volume, liquidity constraint, inverse-vol weighting.

### v7
- Tighten event selection.
- Require larger return moves and stronger volume shocks.

### v8
- Extend event horizon and rebalance weekly.
- Shift toward medium-horizon continuation with lower turnover.

### v9
- Residualize returns against SPY and sector ETFs.
- Focus on stock-specific continuation rather than market or sector drift.

### v10
- Add low-volatility conditioning.
- Combine residual continuation with a lower-volatility preference.

### v11
- Add benchmark trend regime scaling.
- Reduce gross exposure when the broad market is below long-term trend.

### v12
- Residual move z-score.
- Replace raw residual move magnitude with a time-series z-score of the residual move.

### v13
- Abnormal move relative to realized volatility plus 20-day trend confirmation.
- Normalize the residual move by realized vol and only keep names with positive short-term residual trend.

### v14
- Stronger market and sector neutralization.
- Add a Barra-like factor model and explicitly neutralize the signal to market and sector exposures.

### v15
- Holding-period optimization: 3-day hold.
- Keep the improved v10-style signal but rebalance every 3 days.

### v16
- Holding-period optimization: 10-day hold.
- Same signal as v10 but with slower exit cadence.

### v17
- Staggered exits.
- Blend 0-day, 5-day, and 10-day position sleeves to smooth exits.

### v18
- Entry timing delay.
- Delay entry by one extra day beyond the standard execution lag.

### v19
- Combine the v16 10-day hold with residual move z-score.
- Test whether the normalized event magnitude improves the best horizon result.

### v20
- Combine the v16 10-day hold with an extra one-day entry delay.
- Test whether waiting one more day improves fill quality and persistence.

### v21
- Combine the v16 10-day hold with both residual move z-score and entry delay.
- Test the strongest combined version of the two follow-up ideas.

### v22
- Branch off v21 with an 8-day hold.
- Test whether a slightly faster exit improves Sharpe.

### v23
- Branch off v21 with a 12-day hold.
- Test whether a slightly slower exit improves persistence.

### v24
- Branch off v21 with a 15-day hold.
- Test whether the continuation edge extends further than 10-12 days.

### v25
- Branch off v21 with a 40-day residual move z-score window.
- Test whether a shorter normalization horizon better captures recent regime shifts.

### v26
- Branch off v21 with tighter liquidity breadth.
- Reduce the eligible universe from top 175 to top 150 by liquidity.

sharpe	0.6868233449909870
ann_return	0.23802259680299700
ann_vol	0.42537370076624400
max_drawdown		-0.475744700548549
max_drawdown_duration		643
max_drawdown_start		2021-06-11 00:00:00
max_drawdown_end		2023-12-29 00:00:00
avg_daily_turnover		1.423029483276830
benchmark_ann_return		0.11777405247721000
benchmark_sharpe		0.7063929581682670
benchmark_corr		0.43289202322558800
information_ratio		0.42861576274806800
