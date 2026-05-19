# Distance-Pairs Sleeves In Best Buildout

In the current best buildout portfolio, the distance-pairs sleeves are:

- `dist_mr_k3_z20__r21__cond__vol_contraction_10_60`
- `dist_mr_k3_z60__r5__cond__vol_expansion_10_60`
- `dist_mr_k3_z60__r10__cond__none`
- `dist_mr_k3_z10__r10__cond__panic_10d_minus5`

They all come from the same base model in [signal_sweep_distance_pairs.py](/Users/jplatta/repos/portfolio-research/scripts/signal_sweep_distance_pairs.py), where:

- `k` = number of nearest historical partners
- `z` = rolling z-score window on the pair spread
- `r` = rebalance frequency in trading days
- `cond` = conditioning filter added later in [signal_sweep_conditioning_best_strategies.py](/Users/jplatta/repos/portfolio-research/scripts/signal_sweep_conditioning_best_strategies.py)

## Sleeve Parameters

| Sleeve | k | z window | Rebalance | Conditioning filter | What changes vs the others |
|---|---:|---:|---:|---|---|
| `dist_mr_k3_z20__r21__cond__vol_contraction_10_60` | 3 | 20 | 21 | `vol_contraction_10_60` | Medium-speed spread normalization, slowest rebalance, only active in lower-volatility contraction regimes |
| `dist_mr_k3_z60__r5__cond__vol_expansion_10_60` | 3 | 60 | 5 | `vol_expansion_10_60` | Slowest and smoothest z-score window, fastest trading of the four, only active in volatility expansion regimes |
| `dist_mr_k3_z60__r10__cond__none` | 3 | 60 | 10 | `none` | Same slow 60-day spread normalization, medium rebalance, no conditioning gate |
| `dist_mr_k3_z10__r10__cond__panic_10d_minus5` | 3 | 10 | 10 | `panic_10d_minus5` | Fastest and most reactive spread normalization, medium rebalance, only active after panic-type market drops |

## Common Base Model Settings

| Parameter | Value | Source |
|---|---:|---|
| Pair count `k` sweep | `1, 3` | `make_signals()` |
| Z windows tested | `10, 20, 60` | `make_signals()` |
| Rebalance periods tested | `1, 5, 10, 21` | `REBALANCE_PERIODS` |
| Chosen pair count in buildout | `3` for all four | sleeve names |
| Long / short book size | `20 / 20` | `N_LONG`, `N_SHORT` |
| Liquidity filter | top `300` names | `add_tradeable_constraint(qs.liquidity(top_n=300))` |
| Base risk scaling | `equity_curve_regime_scale` | always applied in sweep |
| Cost assumption | `10 bps` | `COST_BPS` |

## Signal Construction

For all of them, the base signal is:

1. Build historical pair partners using normalized log-price distance.
2. For each stock, compute spread versus the mean of its `k` nearest partners.
3. Z-score that spread over a rolling `z` window.
4. Trade the negative z-score, clipped to `[-2, 2]`.

In shorthand:

$$
s_{i,t} = -\mathrm{clip}\left(\frac{\mathrm{spread}_{i,t} - \mu_{i,t}^{(z)}}{\sigma_{i,t}^{(z)}}, -2, 2\right)
$$

where `spread` is stock `i` minus the average normalized path of its selected partners.

## Practical Differences

- `z=10` is the most reactive and short-horizon.
- `z=20` is a middle-ground version.
- `z=60` is the slowest and smoothest.
- `r=5` updates positions much more often than `r=21`.
- `cond=none` is always on.
- `vol_contraction`, `vol_expansion`, and `panic` are regime gates that decide when the sleeve is allowed to trade.

One useful point for the report: all four sleeves use `k=3`, so the diversification within this cluster is coming from:

- spread horizon (`z`)
- rebalance speed (`r`)
- conditioning regime (`cond`)

not from changing the number of pair partners.
