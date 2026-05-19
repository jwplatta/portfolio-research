# Core Signal Formulas

This note summarizes the generalized core signals used in the sweep scripts:

- [scripts/signal_sweep_monoton.py](/Users/jplatta/repos/portfolio-research/scripts/signal_sweep_monoton.py)
- [scripts/signal_sweep_mean_reversion.py](/Users/jplatta/repos/portfolio-research/scripts/signal_sweep_mean_reversion.py)
- [scripts/signal_sweep_resid_mean_reversion.py](/Users/jplatta/repos/portfolio-research/scripts/signal_sweep_resid_mean_reversion.py)
- [scripts/signal_sweep_distance_pairs.py](/Users/jplatta/repos/portfolio-research/scripts/signal_sweep_distance_pairs.py)
- [scripts/signal_sweep_sector_relative.py](/Users/jplatta/repos/portfolio-research/scripts/signal_sweep_sector_relative.py)
- [scripts/signal_sweep_event.py](/Users/jplatta/repos/portfolio-research/scripts/signal_sweep_event.py)

The formulas below are intentionally generalized. They capture the signal families rather than every parameterization and variant in the sweeps. Where the code has a detail that is easy to misread from a high-level formula, I call it out explicitly.

## Notation

Let:

- $i$ index stocks and $t$ index dates.
- $r_{i,t}$ be the stock's active return at date $t$.
- $\tilde{r}_{i,t}$ be the stock's residual return at date $t$.
- $w$ be a lookback window.
- $\bar{x}^{(w)}_{i,t} = \frac{1}{w}\sum_{k=0}^{w-1} x_{i,t-k}$ be a rolling mean.
- $\sigma^{(w)}_{i,t}$ be a rolling standard deviation.
- $\operatorname{sign}(\cdot)$ be the sign function.

In the scripts, the raw signal is then ranked cross-sectionally and converted into long/short positions. The formulas here describe the pre-ranking signal.

## Monotonic Momentum

The core monotonic momentum signal measures how consistently daily returns have shared the same sign as the rolling mean return, then weights that consistency by the magnitude of the rolling mean:

$$
\mu^{(w)}_{i,t} = \bar{r}^{(w)}_{i,t}
$$

$$
h^{(w)}_{i,t} =
\frac{1}{w}
\sum_{k=0}^{w-1}
\mathbf{1}
\left(
\operatorname{sign}(r_{i,t-k}) = \operatorname{sign}(\mu^{(w)}_{i,t-k})
\right)
$$

$$
s^{\mathrm{mon}}_{i,t} = h^{(w)}_{i,t}\, \left| \mu^{(w)}_{i,t} \right|
$$

Interpretation:

- High values mean the stock has been moving in one direction persistently, not just on average.
- This is stronger than plain momentum because it rewards path consistency, not only cumulative return.

Clarifying notes:

- This formula matches the exact `monoton_{w}d` signal family tested in [signal_sweep_monoton.py](/Users/jplatta/repos/portfolio-research/scripts/signal_sweep_monoton.py).
- The exact tested windows for this core family are `$w \in \{20, 40, 60, 120, 252\}$`.
- In code, the sign-agreement term is formed date by date as `sign(r)` versus `sign(rolling_mean(r))`, and then that indicator is itself averaged over the same window.
- The same script also tests nearby variants built from the same idea: `monoton_raw_{w}d`, `monoton_signed_{w}d`, and `monoton_skip_{w}d`.

## Mean Reversion with Active Returns

The basic active-return mean reversion signal is the negative rolling average of active returns:

$$
s^{\mathrm{mr}}_{i,t} = -\bar{r}^{(w)}_{i,t}
$$

Interpretation:

- Recent underperformers get positive signal values.
- Recent outperformers get negative signal values.

This is the canonical short-horizon cross-sectional reversal formulation used in [signal_sweep_mean_reversion.py](/Users/jplatta/repos/portfolio-research/scripts/signal_sweep_mean_reversion.py).

Clarifying notes:

- This formula matches the exact `mr_{w}d` signals tested in the script.
- The exact tested windows for this core family are `$w \in \{2, 3, 5, 10, 15, 20\}$`.
- Here `$r_{i,t}$` is specifically the script's active return series, `cache["_active_returns"]`.
- The same script also tests the related reversal families `zscore_rev_{fast}_{slow}` and `cumret_spread_{short}_{long}`, detailed below.

### Active-Return Z-Score Reversal

The `zscore_rev_{fast}_{slow}` family compares a short-horizon active-return mean to a longer-horizon active-return baseline, scaled by the long-horizon volatility. Here $f$ denotes the fast window and $s$ denotes the slow window.

$$
\mu^{(s)}_{i,t} = \bar{r}^{(s)}_{i,t}
$$

$$
\sigma^{(s)}_{i,t} = \operatorname{std}\!\left(r_{i,t}, r_{i,t-1}, \ldots, r_{i,t-s+1}\right)
$$

$$
s^{\mathrm{zrev}}_{i,t} = -\frac{\bar{r}^{(f)}_{i,t} - \mu^{(s)}_{i,t}}{\sigma^{(s)}_{i,t}}
$$

Interpretation:

- This fades names whose recent short-horizon active return has moved too far above their longer-run active-return mean.
- It is a standardized reversal signal, so the same raw move counts less when the stock's active returns are already volatile.

Clarifying notes:

- This matches the exact implementation in [signal_sweep_mean_reversion.py](/Users/jplatta/repos/portfolio-research/scripts/signal_sweep_mean_reversion.py):
  `r = cache["_active_returns"]`, `mu = r.rolling(slow).mean()`, `sigma = r.rolling(slow).std().clip(lower=1e-8)`, and `signal = -(r.rolling(fast).mean() - mu) / sigma`.
- The exact tested parameter pairs are `zscore_rev_5_60`, `zscore_rev_10_120`, `zscore_rev_5_252`, and `zscore_rev_20_252`.
- Unlike some other scripts, this z-score is not clipped before ranking.

### Active-Return Cumret-Spread Reversal

The `cumret_spread_{short}_{long}` family compares short-horizon and long-horizon active-return averages, then reverses that spread. Here $u$ denotes the short window and $v$ denotes the long window.

$$
s^{\mathrm{cs}}_{i,t} = -\left(\bar{r}^{(u)}_{i,t} - \bar{r}^{(v)}_{i,t}\right)
$$

Interpretation:

- This fades names whose short-run active performance is rich relative to their own longer-run active-performance baseline.
- Relative to the z-score version, this keeps the same mean-reversion idea but does not volatility-standardize it.

Clarifying notes:

- This matches the exact implementation in [signal_sweep_mean_reversion.py](/Users/jplatta/repos/portfolio-research/scripts/signal_sweep_mean_reversion.py):
  `r = cache["_active_returns"]` and `signal = -(r.rolling(short).mean() - r.rolling(long).mean())`.
- The exact tested parameter pairs are `cumret_spread_5_60`, `cumret_spread_10_120`, and `cumret_spread_20_252`.

## Mean Reversion with Residual Returns

Residual mean reversion uses the same reversal logic, but on residual returns rather than active returns:

$$
s^{\mathrm{res}}_{i,t} = -\bar{\tilde{r}}^{(w)}_{i,t}
$$

Interpretation:

- The signal fades recent idiosyncratic moves after removing common-factor structure.
- In [signal_sweep_resid_mean_reversion.py](/Users/jplatta/repos/portfolio-research/scripts/signal_sweep_resid_mean_reversion.py), this is applied to two residual engines: factor-model residuals and ETF-factor time-series residuals.

The trading formula is the same in both cases. The only difference is how $\tilde{r}_{i,t}$ is constructed upstream.

Clarifying notes:

- This formula matches the exact `factor_model_resid_mr_{w}d` and `etf_factor_resid_mr_{w}d` signals tested in [signal_sweep_resid_mean_reversion.py](/Users/jplatta/repos/portfolio-research/scripts/signal_sweep_resid_mean_reversion.py).
- The exact tested windows are `$w \in \{2, 3, 5, 10, 15, 20\}$` for the factor-model residual branch and `$w \in \{2, 5, 10\}$` for the ETF-factor residual branch.
- The factor-model branch uses `add_factor_model(factors=["market", "sector"], sector_map=...)` before `residualize_returns()`.
- The ETF-factor branch uses the study's `factors=FACTORS` data and then applies `residualize_returns()` directly.
- The same script also tests residual z-score reversal and residual volatility signals, which are not written out here because they are not the core rolling-mean reversion formula.

## Distance Pairs Mean Reversion

For each stock $i$, define a partner set $P(i)$ consisting of the historically closest names under the script's distance metric. Build normalized price paths $p_{i,t}$, then compare each stock to the average normalized path of its partners:

$$
p_{i,t} = \prod_{\tau \le t} (1 + r_{i,\tau})
$$

$$
n_{i,t} = \frac{p_{i,t}}{p_{i,0}}
$$

$$
d_{i,t} = n_{i,t} - \frac{1}{|P(i)|}\sum_{j \in P(i)} n_{j,t}
$$

The signal is a z-scored reversal of that divergence:

$$
z_{i,t} = \frac{d_{i,t} - \bar{d}^{(w)}_{i,t}}{\sigma^{(w)}(d_{i,t})}
$$

$$
s^{\text{pairs-mr}}_{i,t} = -z_{i,t}
$$

Interpretation:

- If a stock has risen too far above its closest peers, the signal turns negative.
- If it has lagged too far below them, the signal turns positive.

This is the core idea implemented in [signal_sweep_distance_pairs.py](/Users/jplatta/repos/portfolio-research/scripts/signal_sweep_distance_pairs.py).

Clarifying notes:

- This section is intentionally split into two layers because that is how the script works: partner selection is done once from a historical distance matrix, and the live trading signal is then computed from active-return-derived cumulative paths.
- The tested distance proxy is not a rolling pair spread estimated on the fly. The script computes cumulative log-return paths over the full sample, standardizes those paths cross-section by ticker, defines distance as $1 - \operatorname{corr}$, and chooses the $k$ nearest partners with $k \in \{1, 3\}$.
- After partner selection, the actual tested trading signal uses $p_{i,t} = \prod_{\tau \le t}(1 + r_{i,\tau})$ with $r_{i,t}$ equal to active returns.
- The tested z-score windows are `$w \in \{10, 20, 60\}$`, and the z-score is clipped to `$[-2, 2]$` before ranking.

## Sector-Relative Mean Reversion

Let $g(i)$ denote the sector ETF assigned to stock $i$, and let $r^{\text{sector}}_{g(i),t}$ be that ETF's return. Define the stock's sector-relative return as:

$$
r^{\text{rel}}_{i,t} = r_{i,t} - r^{\text{sector}}_{g(i),t}
$$

Then sector-relative mean reversion is:

$$
s^{\text{sector-mr}}_{i,t} = -\overline{r^{\text{rel}}}^{(w)}_{i,t}
$$

Interpretation:

- This fades moves relative to the stock's own sector rather than relative to the full market.
- It tries to isolate within-sector dislocations.

This is the main mean-reversion family in [signal_sweep_sector_relative.py](/Users/jplatta/repos/portfolio-research/scripts/signal_sweep_sector_relative.py).

Clarifying notes:

- This formula matches the exact `sector_rel_mr_{w}d` signals tested in the script.
- The exact tested windows for this core family are $w \in \{5, 20, 60\}$.
- More precisely, the tested relative-return series is $r^{\text{rel}}_{i,t} = r^{\text{active}}_{i,t} - r^{\text{sector ETF}}_{g(i),t}$.
- So this is not raw stock return minus sector return. It is active return minus sector ETF return.
- The same script also tests `sector_rel_mom_{w}d` and `sector_rel_zscore_5_60`.

## Gap Reversion

The simplest event-style gap reversion signal fades the most recent one-day shock:

$$
s^{\text{gap-rev}}_{i,t} = -r_{i,t-1}
$$

Residual gap reversion uses the same form on residual returns:

$$
s^{\text{resid-gap-rev}}_{i,t} = -\tilde{r}_{i,t-1}
$$

Interpretation:

- A large positive move yesterday produces a negative signal today.
- A large negative move yesterday produces a positive signal today.

This is the core gap-reversion family in [signal_sweep_event.py](/Users/jplatta/repos/portfolio-research/scripts/signal_sweep_event.py).

Clarifying notes:

- These formulas match the exact `gap_reversion` and `resid_gap_reversion` signals tested in the script.
- The word "gap" in this script is a naming shorthand. The exact tested formula is just the negative one-period lag of the return series: active version `-cache["_active_returns"].shift(1)` and residual version `-cache["residual_returns"].shift(1)`.
- In the event script, the residual version comes from `Study(..., factors=FACTORS).residualize_returns()`, so this is the ETF-factor residualization path rather than the factor-model branch used elsewhere.
- So this is not an explicit overnight open-versus-close gap construction.
- The same script also tests gap-accumulation variants `gap_accum_{w}d` and `resid_gap_accum_{w}d`, with the exact tested formula
  $$
  s_{i,t} = -\max\{\rho_{i,t}, \rho_{i,t-1}, \ldots, \rho_{i,t-w+1}\},
  $$
  where $\rho$ is either active returns or residual returns.

## Summary

At a high level, the six core families reduce to:

- Monotonic momentum: reward persistent same-direction movement.
- Active-return mean reversion: fade recent active winners and losers.
- Residual-return mean reversion: fade recent idiosyncratic winners and losers.
- Distance pairs mean reversion: fade divergence from a stock's closest peer set.
- Sector-relative mean reversion: fade divergence versus the stock's sector benchmark.
- Gap reversion: fade the most recent one-day shock.
