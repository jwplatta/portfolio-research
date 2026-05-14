# XSection Mean Reversion with Cointegration

You typically use cointegration by first grouping economically similar equities (same sector, business model, region, etc.), then estimating stable equilibrium relationships between them. For each stock, regress its log price on a small basket of peers and compute the residual:
\epsilon_t = y_t - X_t \beta
If the residual is stationary (ADF test on residuals passes), you treat deviations from equilibrium as temporary dislocations. When the residual z-score becomes large, you bet on reversion: long underperformers, short outperformers.

In a medium-frequency setup, you usually recompute hedge ratios and cointegration tests on rolling windows (e.g. 6–12 months), then hold positions for days to weeks. The main signal is the normalized residual:
z_t = \frac{\epsilon_t - \mu}{\sigma}
You rank the universe cross-sectionally by z-score and build a market/sector-neutral portfolio that buys the most negative residuals and shorts the most positive ones.

In practice, the hard parts are not the cointegration tests themselves. They are: maintaining stable relationships through regime changes, avoiding crowded residuals, controlling sector/factor exposures, and estimating whether the residual actually mean reverts fast enough after costs. Most production stat arb systems eventually evolve into “factor-neutral residual mean reversion” rather than pure pair trading.