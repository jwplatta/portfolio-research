# Sleeve Attribution Observations — Equal Volatility Barbell Portfolio

## Overview

The sleeve attribution results suggest that the final portfolio is meaningfully diversified across multiple alpha engines rather than being dominated by a single strategy. Although the distance-pairs mean reversion (DPMR) sleeves remain important contributors, the portfolio-level return, variance, and turnover contributions are distributed relatively evenly across the sleeve set.

The attribution profile supports the conclusion that the portfolio behaves more like a diversified multi-sleeve statistical arbitrage portfolio than a narrowly concentrated factor bet.

---

# Key Observations

## 1. Return Contribution Is Broadly Diversified

Return contribution is distributed relatively evenly across sleeves:

- Largest sleeve contribution: ~16%
- Smallest sleeve contribution: ~9%
- Most sleeves contribute between ~10–14%

This is a healthy diversification profile for an 8-sleeve portfolio and suggests that portfolio performance is not driven by a single isolated strategy.

This also reduces concerns that the overall Sharpe ratio is being artificially inflated by one dominant sleeve.

---

# 2. DPMR Sleeves Are Important but Not Overwhelmingly Dominant

Distance-pairs mean reversion sleeves contribute materially to overall returns, but they do not completely dominate the portfolio economically.

The strongest DPMR sleeve:

- `dist_mr_k3_z60__r10__cond__none`

contributes:

- ~16% of total return contribution
- ~13.5% of variance contribution

This sleeve appears to function as a strong core alpha engine with favorable risk-adjusted characteristics.

Importantly, multiple DPMR sleeves contribute positively across different volatility and conditioning regimes, supporting the idea that the family is internally diversified rather than redundant.

---

# 3. The Vol-Contraction DPMR Sleeve Is an Extremely Efficient Diversifier

One of the most notable findings is:

- `dist_mr_k3_z20__r21__cond__vol_contraction_10_60`

This sleeve contributes:

- ~13.8% of portfolio returns
- only ~7.3% of portfolio variance

This implies a very high return-to-risk efficiency ratio.

Economically, this sleeve appears to act as a stabilizing relative-value sleeve that performs particularly well during calmer volatility regimes while adding relatively little incremental portfolio risk.

This is one of the strongest portfolio construction sleeves in the entire portfolio.

---

# 4. Residual Gap Reversal Is Operationally Expensive but Productive

The event-driven sleeve:

- `resid_gap_reversion__r10__breadth_40`

accounts for a disproportionately large share of turnover contribution.

However, it also contributes materially to portfolio returns while maintaining only moderate variance contribution.

Interpretation:

- the sleeve is expensive to trade,
- but it is not simply adding noise or unnecessary turnover,
- it appears to provide genuinely orthogonal alpha.

This suggests that the sleeve is likely sensitive to implementation quality and transaction costs, but still economically valuable within the broader portfolio.

---

# 5. Residual Mean Reversion Sleeves Behave Like Stable Portfolio Anchors

The residual mean reversion sleeves show:

- moderate return contribution,
- relatively higher variance contribution,
- lower turnover contribution.

These sleeves appear less "efficient" from a pure standalone alpha perspective than the strongest DPMR sleeves.

However, their profile is consistent with stable institutional-style relative-value sleeves that serve as:

- structural portfolio anchors,
- diversification stabilizers,
- lower-turnover core exposures.

This behavior is economically plausible and desirable within a multi-sleeve portfolio framework.

---

# 6. Crisis Protection Appears to Come More from Relative-Value Convergence Than Explicit Crash Sleeves

An interesting observation is that:

- the panic DPMR sleeve,
- and the MON crash momentum sleeve,

both experienced losses during the portfolio’s worst drawdown window.

Meanwhile, several core relative-value sleeves remained positive contributors during the same period.

This suggests that the portfolio’s drawdown resilience may come less from explicit crash or momentum overlays and more from:

- diversified relative-value convergence behavior,
- cross-sleeve diversification,
- regime specialization among DPMR sleeves.

This is an important distinction because it indicates that portfolio resilience emerges structurally from diversification rather than from a single dedicated hedge sleeve.

---

# 7. The Portfolio Appears Structurally Coherent

Overall, the attribution profile supports the interpretation that the final portfolio consists of:

- multiple complementary alpha engines,
- diversified regime exposures,
- heterogeneous turnover profiles,
- both fast-reacting and slow-reacting sleeves,
- and several economically distinct relative-value mechanisms.

The final portfolio therefore appears substantially more robust and diversified than a simple collection of highly correlated mean reversion strategies.

---

# Final Interpretation

The sleeve attribution results support the conclusion that the portfolio is:

- diversified across multiple statistical arbitrage mechanisms,
- not dominated by a single alpha sleeve,
- robust across several conditioning regimes,
- and internally balanced across turnover, variance, and return contribution.

The attribution profile strengthens the interpretation that the portfolio represents a coherent multi-sleeve statistical arbitrage framework rather than an overfit single-factor strategy.