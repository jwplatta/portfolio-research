# Barbell Robustness Notes

These notes focus on the `Barbell Portfolio` branch of [portfolio_sweep_target_10_extensions.csv](/Users/jplatta/repos/portfolio-research/results/2026-05-18/portfolio_sweep_target_10_extensions.csv), using the `equal`-weight rows only.

The goal is to understand which extension patterns were most robust when we remove weighting optimization and just ask: what sleeve combinations improve the barbell on a plain equal-weight basis?

## Core Result

The strongest equal-weight extensions are not broad mixes of many themes. The best results come from pairing:

- one monotonic momentum sleeve
- one residual-gap reversion sleeve

That `MON + EV` pairing is the cleanest and strongest improvement over the base barbell.

## Base Versus Best Equal-Weight Extension

| Portfolio | Net Sharpe | Max DD | Turnover |
|---|---:|---:|---:|
| `Barbell Portfolio` | 1.883 | -0.0281 | 0.0470 |
| `Barbell Portfolio + monoton_120d__r21__crash_10_5pct + resid_gap_reversion__r10__breadth_40` | 2.039 | -0.0235 | 0.0502 |

The best equal-weight extension improves Sharpe by about `+0.156` while also slightly improving drawdown.

## Best Equal-Weight Barbell Rows

| Net Sharpe | Max DD | Turnover | Portfolio | Extension sleeve | Ext count |
|---|---:|---:|---|---|---:|
| 2.039 | -0.0235 | 0.0502 | `Barbell Portfolio + monoton_120d__r21__crash_10_5pct + resid_gap_reversion__r10__breadth_40` | `monoton_120d__r21__crash_10_5pct + resid_gap_reversion__r10__breadth_40` | 2 |
| 2.038 | -0.0235 | 0.0490 | `Barbell Portfolio + monoton_120d__r21__crash_10_5pct + resid_gap_reversion__r10__breadth_60` | `monoton_120d__r21__crash_10_5pct + resid_gap_reversion__r10__breadth_60` | 2 |
| 2.024 | -0.0215 | 0.0485 | `Barbell Portfolio + monoton_120d__r21__crash_10_5pct + resid_gap_reversion__r10__trend_50_200 + resid_gap_reversion__r10__breadth_60` | `monoton_120d__r21__crash_10_5pct + resid_gap_reversion__r10__trend_50_200 + resid_gap_reversion__r10__breadth_60` | 3 |
| 2.016 | -0.0215 | 0.0497 | `Barbell Portfolio + monoton_120d__r21__crash_10_5pct + resid_gap_reversion__r10__trend_50_200 + resid_gap_reversion__r10__breadth_40` | `monoton_120d__r21__crash_10_5pct + resid_gap_reversion__r10__trend_50_200 + resid_gap_reversion__r10__breadth_40` | 3 |
| 2.015 | -0.0252 | 0.0515 | `Barbell Portfolio + monoton_120d__r21__crash_10_5pct + resid_gap_reversion__r10__none` | `monoton_120d__r21__crash_10_5pct + resid_gap_reversion__r10__none` | 2 |
| 2.013 | -0.0228 | 0.0474 | `Barbell Portfolio + monoton_120d__r21__vol_20_60 + resid_gap_reversion__r10__breadth_60` | `monoton_120d__r21__vol_20_60 + resid_gap_reversion__r10__breadth_60` | 2 |
| 2.012 | -0.0228 | 0.0485 | `Barbell Portfolio + monoton_120d__r21__vol_20_60 + resid_gap_reversion__r10__breadth_40` | `monoton_120d__r21__vol_20_60 + resid_gap_reversion__r10__breadth_40` | 2 |
| 2.012 | -0.0264 | 0.0453 | `Barbell Portfolio + monoton_120d__r21__crash_10_5pct + resid_gap_reversion__r10__trend_50_200` | `monoton_120d__r21__crash_10_5pct + resid_gap_reversion__r10__trend_50_200` | 2 |
| 2.009 | -0.0224 | 0.0534 | `Barbell Portfolio + monoton_120d__r21__crash_10_5pct + resid_gap_reversion__r10__breadth_40 + resid_gap_reversion__r10__breadth_60` | `monoton_120d__r21__crash_10_5pct + resid_gap_reversion__r10__breadth_40 + resid_gap_reversion__r10__breadth_60` | 3 |

## Best Equal-Weight Representative By Extension Family

| Family | Best equal-weight portfolio | Net Sharpe | Max DD | Turnover |
|---|---|---:|---:|---:|
| `MON + EV` | `Barbell Portfolio + monoton_120d__r21__crash_10_5pct + resid_gap_reversion__r10__breadth_40` | 2.039 | -0.0235 | 0.0502 |
| `MON + EV + EV` | `Barbell Portfolio + monoton_120d__r21__crash_10_5pct + resid_gap_reversion__r10__trend_50_200 + resid_gap_reversion__r10__breadth_60` | 2.024 | -0.0215 | 0.0485 |
| `MON` | `Barbell Portfolio + monoton_120d__r21__crash_10_5pct` | 1.989 | -0.0320 | 0.0462 |
| `MON + EV + SR` | `Barbell Portfolio + monoton_120d__r21__crash_10_5pct + resid_gap_reversion__r10__breadth_40 + sector_rel_zscore_5_60__r10__trend_20_100` | 1.972 | -0.0249 | 0.0493 |
| `MON + MON + EV` | `Barbell Portfolio + monoton_120d__r21__vol_20_60 + monoton_120d__r21__crash_10_5pct + resid_gap_reversion__r10__breadth_40` | 1.964 | -0.0293 | 0.0483 |
| `EV` | `Barbell Portfolio + resid_gap_reversion__r10__breadth_60` | 1.940 | -0.0283 | 0.0501 |
| `EV + EV` | `Barbell Portfolio + resid_gap_reversion__r10__trend_50_200 + resid_gap_reversion__r10__breadth_60` | 1.933 | -0.0258 | 0.0495 |
| `MON + SR` | `Barbell Portfolio + monoton_120d__r21__crash_10_5pct + sector_rel_zscore_5_60__r10__trend_20_100` | 1.922 | -0.0295 | 0.0460 |
| `MON + MON` | `Barbell Portfolio + monoton_120d__r21__vol_20_60 + monoton_120d__r21__crash_10_5pct` | 1.914 | -0.0393 | 0.0446 |
| `SR` | `Barbell Portfolio + sector_rel_zscore_5_60__r10__trend_20_100` | 1.817 | -0.0293 | 0.0466 |

## What These Differences Suggest

The best equal-weight extension is not “more sleeves.” It is a very specific cross-family pairing: one monotonic momentum sleeve plus one residual-gap sleeve. That pairing beats:

- adding only momentum
- adding only event reversal
- adding momentum plus sector-relative
- adding multiple momentum sleeves

The residual-gap sleeve choice is also fairly stable. The top results use:

- `resid_gap_reversion__r10__breadth_40`
- `resid_gap_reversion__r10__breadth_60`

Those two are nearly interchangeable at the top, which is a useful robustness sign. The best row is `breadth_40`, but `breadth_60` is effectively tied.

The simpler `MON + EV` construction also beats the more crowded `MON + EV + EV` variants. A second residual-gap sleeve can still work, but it does not improve on the cleanest two-sleeve pair. That suggests the second event sleeve is mostly refining the same theme instead of introducing a meaningfully distinct source of return.

Sector-relative sleeves help, but they do not define the top of the equal-weight barbell results. `MON + EV + SR` is still strong, but it comes in below the best `MON + EV` portfolios. This suggests sector-relative is a useful secondary diversifier rather than the key ingredient in the strongest barbell extension.

Similarly, adding a second momentum sleeve does not help as much as pairing momentum with event reversal. `MON + MON + EV` and `MON + MON` both lag the top `MON + EV` rows, and the pure `MON + MON` version comes with worse drawdowns.

## Report-Friendly Interpretation

If the report needs one concise robustness message, it is this:

> Within the equal-weight barbell extension sweep, the most reliable improvement came from pairing one monotonic momentum sleeve with one breadth-gated residual-gap sleeve. That combination beat single-family add-ons and more crowded multi-sleeve variants, while remaining stable across nearby parameter choices such as `breadth_40` versus `breadth_60`.
