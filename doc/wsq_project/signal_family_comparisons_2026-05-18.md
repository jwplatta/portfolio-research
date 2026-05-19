****# Signal Family Comparisons: 2026-05-18

This note pulls examples directly from the rerun sweep CSVs under [results/2026-05-18](/Users/jplatta/repos/portfolio-research/results/2026-05-18).

Scope for this pass:

- base (`none`) vs best risk-scaled version
- comparisons across variants within the same family

Conditioning-filter comparisons are intentionally deferred for now.

`Gross` below refers to `gross_sharpe`, since that is the gross metric available in the sweep CSVs.

## Monotonic Momentum

| Role | Signal name | Net Sharpe | Gross | Turnover | Max drawdown | Longest drawdown duration | Bench corr | Read |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Best scaled core | `monoton_120d__r21__disp_60_q30` | 0.769 | 0.910 | 0.040 | -0.101 | 392 | 0.253 | Best overall monotonic sleeve; low-dispersion scaling improved the core 120d signal. |
| Best conditioned version | `monoton_120d__r21__disp_60_q30__cond__none` | 0.769 | 0.910 | 0.040 | -0.101 | 392 | 0.253 | No conditioning filter improved on the already-best scaled monotonic sleeve. |
| Base core | `monoton_120d__r21__none` | 0.606 | 0.763 | 0.051 | -0.147 | 712 | 0.262 | Clean base reference for the strongest monotonic signal. |
| Best skip variant | `monoton_skip_60d__r21__none` | 0.515 | 0.710 | 0.055 | -0.096 | 706 | 0.290 | Skip-week logic worked, but not as well as the plain 120d monotonicity signal. |
| Best raw variant | `monoton_raw_60d__r21__none` | 0.177 | 0.392 | 0.049 | -0.089 | 1221 | -0.132 | Removing the magnitude weighting hurt badly. |

## Mean Reversion

| Role | Signal name | Net Sharpe | Gross | Turnover | Max drawdown | Longest drawdown duration | Bench corr | Read |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Best scaled core | `mr_5d__r10__trend_20_100` | 0.708 | 0.949 | 0.052 | -0.067 | 905 | 0.154 | Best overall active-return reversal; trend scaling clearly helped. |
| Best conditioned version | `zscore_rev_5_60__r10__trend_20_100__cond__residual_dispersion_high_20_q75` | 1.279 | 1.426 | 0.017 | -0.032 | 418 | 0.143 | Conditioning helped a lot; the biggest winner in the family was the z-score branch gated to high residual-dispersion regimes. |
| Base core | `mr_5d__r10__none` | 0.557 | 0.927 | 0.117 | -0.129 | 790 | 0.176 | Strong base version of the plain 5-day reversal. |
| Best z-score variant | `zscore_rev_5_60__r10__none` | 0.372 | 0.834 | 0.115 | -0.117 | 1065 | 0.189 | Standardization helped relative to weaker variants, but still lagged plain 5d MR. |
| Best cumret-spread variant | `cumret_spread_5_60__r10__none` | 0.197 | 0.575 | 0.109 | -0.191 | 905 | 0.144 | The raw short-vs-long spread version was notably weaker. |

## Residual Mean Reversion

| Role | Signal name | Net Sharpe | Gross | Turnover | Max drawdown | Longest drawdown duration | Bench corr | Read |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Best scaled core | `etf_factor_resid_mr_5d__r10__trend_20_100` | 0.918 | 1.267 | 0.051 | -0.044 | 553 | 0.138 | Strongest residual sleeve overall; trend scaling helped a lot. |
| Best conditioned version | `factor_model_resid_mr_5d__r10__trend_20_100__cond__residual_dispersion_high_20_q75` | 1.083 | 1.192 | 0.015 | -0.045 | 532 | 0.092 | Conditioning helped here too, and it flipped the family winner to the factor-model branch under high residual-dispersion gating. |
| Base core | `etf_factor_resid_mr_5d__r10__none` | 0.694 | 1.219 | 0.119 | -0.079 | 389 | 0.138 | Best unscaled residual-reversion reference point. |
| ETF-factor 2d variant | `etf_factor_resid_mr_2d__r10__none` | 0.458 | 0.977 | 0.120 | -0.117 | 1095 | 0.171 | Shorter residual reversal still worked, but trailed the 5d version. |
| Best factor-model variant | `factor_model_resid_mr_5d__r10__none` | 0.179 | 0.687 | 0.109 | -0.092 | 1256 | 0.145 | The factor-model residual branch lagged the ETF-factor branch by a wide margin. |

## Distance Pairs Mean Reversion

| Role | Signal name | Net Sharpe | Gross | Turnover | Max drawdown | Longest drawdown duration | Bench corr | Read |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Best overall | `dist_mr_k3_z60__r10` | 1.078 | 1.549 | 0.129 | -0.097 | 229 | -0.179 | Best result in the family; no separate risk-scaler axis was swept here. |
| Best conditioned version | `dist_mr_k3_z60__r10__cond__none` | 1.078 | 1.549 | 0.129 | -0.097 | 229 | -0.179 | No conditioning filter improved on the top distance-pairs sleeve; the ungated version remained best. |
| Best 20-day window | `dist_mr_k3_z20__r21` | 0.755 | 1.116 | 0.069 | -0.100 | 857 | 0.047 | Medium z-window was solid, but still clearly behind the 60-day version. |
| 1-partner comparison | `dist_mr_k1_z20__r21` | 0.668 | 1.124 | 0.068 | -0.071 | 394 | 0.009 | Using only one partner was weaker than using three. |
| Short-window comparison | `dist_mr_k3_z10__r10` | 0.654 | 1.397 | 0.133 | -0.066 | 481 | 0.094 | Very short z-windows were weaker and more turnover-heavy. |

## Sector-Relative

| Role | Signal name | Net Sharpe | Gross | Turnover | Max drawdown | Longest drawdown duration | Bench corr | Read |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Best scaled core | `sector_rel_zscore_5_60__r10__trend_20_100` | 0.584 | 1.008 | 0.051 | -0.050 | 465 | 0.129 | Best overall sector-relative sleeve; trend scaling made a big difference. |
| Best conditioned version | `sector_rel_mr_5d__r10__trend_20_100__cond__residual_dispersion_high_20_q75` | 0.886 | 1.014 | 0.020 | -0.059 | 371 | 0.019 | Conditioning helped materially and shifted the family winner from the z-score branch to the plain 5d sector-relative MR branch. |
| Base z-score core | `sector_rel_zscore_5_60__r10__none` | 0.248 | 0.900 | 0.119 | -0.152 | 891 | 0.154 | Best unscaled sector-relative signal. |
| Base 5d MR variant | `sector_rel_mr_5d__r10__none` | 0.227 | 0.694 | 0.120 | -0.160 | 905 | 0.147 | Plain sector-relative MR was close, but weaker than the z-score version. |
| Momentum contrast | `sector_rel_mom_60d__r10__none` | -0.595 | -0.408 | 0.065 | -0.461 | 1986 | -0.188 | Useful negative example: sector-relative momentum failed badly here. |

## Event / Gap Reversion

| Role | Signal name | Net Sharpe | Gross | Turnover | Max drawdown | Longest drawdown duration | Bench corr | Read |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Best base / overall | `resid_gap_reversion__r10__none` | 0.626 | 1.178 | 0.111 | -0.105 | 1231 | 0.128 | Best event sleeve overall; no non-`none` scaler improved on it. |
| Best conditioned version | `resid_gap_reversion__r10__trend_50_200__cond__vol_contraction_10_60` | 0.825 | 1.093 | 0.036 | -0.044 | 803 | 0.043 | Conditioning helped substantially; vol-contraction gating improved both Sharpe and drawdown while cutting turnover. |
| Best non-`none` scaler | `resid_gap_reversion__r10__trend_50_200` | 0.618 | 1.007 | 0.052 | -0.096 | 1274 | 0.123 | Good contrast row showing the scaler reduced turnover, but not enough to beat the base sleeve. |
| Active gap base | `gap_reversion__r10__none` | 0.335 | 0.711 | 0.109 | -0.143 | 1161 | 0.165 | The active-return version worked, but much worse than the residual version. |
| Gap-accum comparison | `gap_accum_2d__r10__none` | -0.046 | 0.280 | 0.110 | -0.283 | 1991 | 0.058 | Multi-day accumulation was a useful negative example; the effect was in the immediate one-day shock. |
