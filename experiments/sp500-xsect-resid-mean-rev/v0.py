"""
Residual mean reversion baseline.
"""

import numpy as np
import pandas as pd
from common import emit_metrics

import qstudy as qs
from qstudy import Study
from qstudy.constants import SP500

start_date = "2015-01-01"
end_date = "2023-12-31"
universe = qs.download(SP500, start_date, end_date)
factors = qs.download(["SPY", "XLK"], start_date, end_date)
benchmark = qs.download("SPY", start_date, end_date)


def demean_signal(signal, **cache):
    """Cross-sectionally demean the signal each day."""
    return signal.sub(signal.mean(axis=1), axis=0)


def equity_curve_regime_scale(positions, **cache):
    returns = cache["returns"]
    mask = cache.get("_tradeable_mask")

    if mask is None:
        mask = cache.get("_liquidity_mask")
    if mask is not None:
        returns = returns.where(mask)

    raw_ret = (positions.shift(1) * returns).sum(axis=1)
    equity = (1 + raw_ret).cumprod()
    equity_ma = equity.rolling(20).mean()
    scale = pd.Series(np.where(equity > equity_ma, 1.0, 0.25), index=equity.index)
    return positions.mul(scale.shift(1), axis=0)


def proportional_positions(signal, **cache):
    signal_z = signal.sub(signal.mean(axis=1), axis=0)
    signal_z = signal_z.div(signal_z.std(axis=1), axis=0)
    signal_z = signal_z.clip(-3, 3)
    signal_z = signal_z.sub(signal_z.mean(axis=1), axis=0)
    return signal_z.div(signal_z.abs().sum(axis=1), axis=0)


def mr_signal(**cache):
    return -cache["residual_returns"].rolling(5).mean().shift(1)


study = (
    Study(
        universe=universe,
        benchmark=benchmark,
        factors=factors,
        name="residual_mean_reversion",
    )
    .residualize_returns()
    .base_signal(mr_signal)
    .transform_signal(demean_signal)
    .add_vol_filter(vol_window=5, quantile=0.6)
    .add_volume_zscore_filter(window=30, min_zscore_quantile=0.8)
    .add_momentum_context_filter(window=60, max_abs_quantile=0.7)
    .add_tradeable_constraint(qs.liquidity(top_n=250, window=60))
    .build_positions(proportional_positions)
    .scale_risk(equity_curve_regime_scale)
    .rebalance(every=1)
    .run()
)

if __name__ == "__main__":
    emit_metrics(study)
