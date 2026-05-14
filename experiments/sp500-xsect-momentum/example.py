import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
import yfinance as yf
from statsmodels.graphics.tsaplots import plot_acf

start_date = "2015-01-01"
end_date = "2023-12-31"

universe = None  # Get stickers from constants.py for the SP500 and sectors

data = yf.download(universe, start=start_date, end=end_date)
close_df = data["Close"].dropna(axis=1)
volume_df = data["Volume"].dropna(axis=1)
position_cnt = 25
asset_rank = 250
returns_df = close_df.pct_change().fillna(0)

# NOTE: signals
base_signal = -returns_df.rolling(5).mean().shift(1)

# NOTE: volume spike filter
volm_win = 10
volm_z = (volume_df - volume_df.rolling(volm_win).mean()) / volume_df.rolling(volm_win).std()

# NOTE: medium term momentum filter
long_mom = returns_df.rolling(15).mean()

signal_df = base_signal.where(
    volm_z.gt(volm_z.quantile(0.65, axis=1), axis=0)
    & long_mom.abs().lt(long_mom.quantile(0.75, axis=1), axis=0)
)

dollar_vol_df = (close_df * volume_df).dropna(axis=1)
avg_dollar_vol_df = dollar_vol_df.rolling(60).mean()

rank_df = avg_dollar_vol_df.rank(axis=1, ascending=False)
rank_mask = rank_df <= asset_rank

filtered_returns = returns_df.where(rank_mask)
filtered_signal_df = signal_df.where(rank_mask)

signal_rank = filtered_signal_df.rank(axis=1, ascending=False, na_option="bottom")

long = signal_rank <= position_cnt
short_cutoff = signal_rank.count(axis=1) - (position_cnt - 1)
short = signal_rank.ge(short_cutoff.values[:, None])

positions = long.astype(int) - short.astype(int)
positions = positions.div(positions.abs().sum(axis=1), axis=0)

pnl = positions.shift(1) * filtered_returns
port_ret_mr = pnl.sum(axis=1)
sharpe = port_ret_mr.mean() / port_ret_mr.std() * np.sqrt(252)
print(sharpe)

# cumulative equity curve
cum_ret = (1 + port_ret_mr).cumprod()
# running peak
running_max = cum_ret.cummax()
# drawdown series
drawdown_mr = cum_ret / running_max - 1
# max drawdown
max_dd = drawdown.min()

print("Max Drawdown:", max_dd)
