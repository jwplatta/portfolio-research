"""Parameter sweep for short-horizon sector-relative momentum.

Grid:
  window       : [5, 10, 20]      -- return horizon in trading days
  skip         : [0, 3, 5]        -- skip period (days excluded before window)
  residualize  : [False, True]    -- strip SPY beta before signal
  vol_normalize: [False, True]    -- divide signal by rolling realized vol
  weighting    : ["equal", "inv_vol", "proportional"]

Total combinations: 3 x 3 x 2 x 2 x 3 = 108
"""

from __future__ import annotations

import itertools
import json
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from common import build_study

HERE = Path(__file__).resolve().parent
CSV_PATH = HERE / "results.csv"

PARAM_GRID = {
    "window": [5, 10, 20],
    "skip": [0, 3, 5],
    "residualize": [False, True],
    "vol_normalize": [False, True],
    "weighting": ["equal", "inv_vol", "proportional"],
}


def combo_name(params: dict) -> str:
    return (
        f"w{params['window']}"
        f"_s{params['skip']}"
        f"_res{int(params['residualize'])}"
        f"_vnorm{int(params['vol_normalize'])}"
        f"_{params['weighting']}"
    )


if __name__ == "__main__":
    keys = list(PARAM_GRID.keys())
    combos = list(itertools.product(*PARAM_GRID.values()))

    rows = []
    for combo in tqdm(combos, desc="param sweep"):
        params = dict(zip(keys, combo))
        name = f"sp500_market_neutral_short_horizon_momentum_{combo_name(params)}"
        study = build_study(name, **params)
        metrics = study.metrics_dict()
        metrics["version"] = combo_name(params)
        rows.append(metrics)
        print(json.dumps({**params, **{k: metrics[k] for k in ["sharpe", "ann_return", "max_drawdown"]}}, default=str))

    df = pd.DataFrame(rows).set_index("version")
    df.transpose().to_csv(CSV_PATH)
    print(f"\nTop 10 by Sharpe:")
    print(df.sort_values("sharpe", ascending=False)[["sharpe", "ann_return", "max_drawdown", "information_ratio"]].head(10).round(4).to_string())
    print(f"\nWrote CSV: {CSV_PATH}")
