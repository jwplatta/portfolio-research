"""Example script for factor data loading — debug before adding to notebook."""
from __future__ import annotations

import io
import zipfile
import requests
import numpy as np
import pandas as pd
import statsmodels.api as sm
from functools import cache

import qstudy as qs
from qstudy.constants import SP500

START_DATE = '2015-01-01'
IS_END     = '2023-12-31'
OOS_END    = '2026-04-30'
OOS_SPLIT  = pd.Timestamp('2024-01-01')
FACTORS    = ['SPY', 'XLK', 'XLF', 'XLE', 'XLV', 'XLI', 'XLY', 'XLP', 'XLU', 'XLRE', 'XLB']

@cache
def load_universe():
    return qs.download(SP500, START_DATE, IS_END)

@cache
def load_benchmark():
    return qs.download(['SPY'], START_DATE, IS_END)

@cache
def load_oos_universe():
    return qs.download(SP500, START_DATE, OOS_END)

@cache
def load_oos_benchmark():
    return qs.download(['SPY'], START_DATE, OOS_END)


# ── 1. Ken French data ────────────────────────────────────────────────────────

def fetch_french_daily(dataset_name: str) -> pd.DataFrame:
    url = f'https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/{dataset_name}_CSV.zip'
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(r.content)) as z:
        fname = [n for n in z.namelist() if n.lower().endswith('.csv')][0]
        raw = z.read(fname).decode('latin-1')

    lines = raw.splitlines()
    # Find the header line (comma-separated, not starting with a digit) just before the data
    data_start = next(i for i, l in enumerate(lines) if l.strip() and l.strip()[0].isdigit())
    header_idx = data_start - 1
    end = next(
        (i for i in range(data_start, len(lines)) if lines[i].strip() and not lines[i].strip()[0].isdigit()),
        len(lines)
    )
    data_str = '\n'.join(lines[header_idx:end])

    df = pd.read_csv(io.StringIO(data_str), index_col=0)
    df.index = pd.to_datetime(df.index.astype(str).str.strip(), format='%Y%m%d')
    df.index.name = 'date'
    df.columns = [c.strip() for c in df.columns]
    df = df.apply(pd.to_numeric, errors='coerce') / 100.0
    return df

print('Fetching FF3...')
ff3_raw = fetch_french_daily('F-F_Research_Data_Factors_daily')
print(f'  FF3 columns: {ff3_raw.columns.tolist()}')
print(f'  FF3 shape: {ff3_raw.shape}, tail:\n{ff3_raw.tail(3)}')

print('\nFetching UMD...')
mom_raw = fetch_french_daily('F-F_Momentum_Factor_daily')
print(f'  UMD columns: {mom_raw.columns.tolist()}')
mom_raw.columns = [mom_raw.columns[0]]  # keep as-is, rename after join
mom_raw = mom_raw.rename(columns={mom_raw.columns[0]: 'UMD'})

ff_factors = ff3_raw.join(mom_raw, how='inner')
ff_factors = ff_factors[ff_factors.index >= '2014-01-01']
print(f'\nCombined FF factors shape: {ff_factors.shape}')
print(ff_factors.tail(3))


# ── 2. BAB ────────────────────────────────────────────────────────────────────

def build_bab(returns: pd.DataFrame, benchmark: pd.Series, window: int = 60) -> pd.Series:
    """Long low-beta tercile, short high-beta tercile. Beta estimated with rolling window."""
    bench = benchmark.reindex(returns.index).fillna(0.0)
    mean_r = returns.rolling(window).mean()
    mean_b = bench.rolling(window).mean()
    cov = returns.mul(bench, axis=0).rolling(window).mean().sub(mean_r.mul(mean_b, axis=0))
    var_b = bench.rolling(window).var().replace(0.0, float('nan'))
    # shift(1): use yesterday's beta estimate to avoid lookahead
    beta = cov.div(var_b, axis=0).shift(1)

    dates, vals = [], []
    for date in returns.index:
        b_t = beta.loc[date]
        r_t = returns.loc[date]
        # Keep only tickers with valid beta and valid return
        valid = b_t.notna() & r_t.notna()
        b_t = b_t[valid]
        r_t = r_t[valid]
        if len(r_t) < 30:
            continue
        lo = b_t <= b_t.quantile(1/3)
        hi = b_t >= b_t.quantile(2/3)
        dates.append(date)
        vals.append(r_t[lo].mean() - r_t[hi].mean())

    return pd.Series(vals, index=pd.DatetimeIndex(dates), name='BAB')

print('\nBuilding BAB (IS)...')
bab_is = build_bab(load_universe().returns, load_benchmark().returns['SPY'])
print(f'  BAB IS shape: {bab_is.shape}, tail:\n{bab_is.tail(3)}')

print('\nBuilding BAB (OOS)...')
bab_oos = build_bab(load_oos_universe().returns, load_oos_benchmark().returns['SPY'])
bab_full = pd.concat([bab_is[bab_is.index < OOS_SPLIT], bab_oos[bab_oos.index >= OOS_SPLIT]])

factors_5f = ff_factors.join(bab_full.rename('BAB'), how='left')
print(f'\nFinal factors_5f shape: {factors_5f.shape}')
print(factors_5f.tail(3))
print('\nDone.')
