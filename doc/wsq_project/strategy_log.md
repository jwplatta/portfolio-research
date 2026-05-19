## Conditioning Sweep Source Index

The current conditioning shortlist in `scripts/signal_sweep_conditioning_best_strategies.py` is sourced from these base signal sweeps:

- `scripts/signal_sweep_monoton.py`: monotonic momentum sleeves
- `scripts/signal_sweep_mean_reversion.py`: raw mean-reversion sleeves
- `scripts/signal_sweep_resid_mean_reversion.py`: all residual mean-reversion sleeves, including the legacy `resid_raw_mr_*` family that used to live under the residual-xsection sweep
- `scripts/signal_sweep_distance_pairs.py`: distance-pairs mean-reversion sleeves
- `scripts/signal_sweep_sector_relative.py`: sector-relative mean-reversion sleeves
- `scripts/signal_sweep_event.py`: residual-gap event sleeves

Residual family note:
- `RESIDUAL-XSECTION-MR` is no longer a separate contributor to the conditioning sweep.
- Its surviving `resid_raw_mr_*` ideas now belong to `scripts/signal_sweep_resid_mean_reversion.py`.

## Monotonicity Momentum
================================================================================
                          MONOTON Signal Sweep Results
================================================================================
                                        name  rebalance         scaler  net_sharpe  avg_daily_turnover  benchmark_corr  max_drawdown  max_drawdown_duration
0             monoton_120d__r21__disp_60_q30         21    disp_60_q30      0.7648              0.0398          0.2535       -0.1015                    392
1             monoton_120d__r21__disp_60_q20         21    disp_60_q20      0.7358              0.0412          0.2540       -0.1015                    392
2               monoton_120d__r21__vol_20_60         21      vol_20_60      0.7263              0.0344          0.2100       -0.0845                    510
3           monoton_120d__r21__crash_10_5pct         21  crash_10_5pct      0.7130              0.0494          0.2545       -0.1471                    572
4              monoton_120d__r21__vol_20_100         21     vol_20_100      0.6051              0.0360          0.2181       -0.1280                    874
5             monoton_120d__r21__disp_40_q25         21    disp_40_q25      0.6024              0.0414          0.2485       -0.1015                    282
6                    monoton_120d__r21__none         21           none      0.5994              0.0507          0.2627       -0.1471                    712
7            monoton_120d__r21__crash_5_3pct         21   crash_5_3pct      0.5992              0.0484          0.2623       -0.1600                    712
8              monoton_120d__r5__disp_60_q30          5    disp_60_q30      0.5961              0.1158          0.2257       -0.0912                    703
9            monoton_120d__r21__crash_5_5pct         21   crash_5_5pct      0.5870              0.0501          0.2698       -0.1600                    712
10             monoton_120d__r5__disp_60_q20          5    disp_60_q20      0.5506              0.1221          0.2230       -0.0912                    699

## Mean Reversion
================================================================================
                      MEAN-REVERSION Signal Sweep Results
================================================================================
                                    name  rebalance   scaler_config  net_sharpe  avg_daily_turnover  benchmark_corr
0               mr_5d__r10__trend_20_100         10    trend_20_100      0.7083              0.0521          0.1549
1             mr_5d__r10__trend_20_100_h         10  trend_20_100_h      0.6716              0.0738          0.1706
2     zscore_rev_5_60__r10__trend_20_100         10    trend_20_100      0.6428              0.0479          0.1597
3               mr_5d__r10__trend_50_200         10    trend_50_200      0.6368              0.0547          0.1600
4                 mr_15d__r21__vol_10_60         21       vol_10_60      0.6148              0.0395         -0.0096
5    zscore_rev_5_252__r10__trend_20_100         10    trend_20_100      0.5892              0.0424          0.1645
6     zscore_rev_5_60__r10__trend_50_200         10    trend_50_200      0.5712              0.0528          0.1710
7                       mr_5d__r10__none         10            none      0.5580              0.1173          0.1764
8   zscore_rev_5_60__r10__trend_20_100_h         10  trend_20_100_h      0.5507              0.0702          0.1778
9                  mr_5d__r21__vol_20_60         21       vol_20_60      0.5394              0.0380          0.0000
10               mr_5d__r10__corr_20_q75         10     corr_20_q75      0.5291              0.1060          0.1193
11                 mr_5d__r21__vol_10_60         21       vol_10_60      0.5230              0.0397         -0.0130
================================================================================
Saved to signal_sweep_mean-reversion.csv


## Residual Mean Reversion
================================================================================
                   RESID-MEAN-REVERSION Signal Sweep Results
================================================================================
The conditioning shortlist now uses the merged `scripts/signal_sweep_resid_mean_reversion.py`
results. Current top residual sleeves to carry into the conditioning sweep are:

- `resid_raw_mr_w5__r10__trend_20_100`
- `resid_raw_mr_w5__r10__trend_20_100_h`
- `resid_raw_mr_w2__r10__trend_50_200`
- `resid_raw_mr_w5__r10__trend_50_200`
- `resid_raw_mr_w2__r10__trend_20_100`
- `resid_mr_5d__r10__trend_20_100`
- `resid_raw_mr_w5__r10__none`
- `resid_raw_mr_w5__r21__vol_10_60`
- `resid_raw_mr_w5__r5__trend_20_100`
- `resid_mr_2d__r10__trend_50_200`
================================================================================

## Residual Legacy Note

The old `RESIDUAL-XSECTION-MR` bucket is deprecated for conditioning purposes.
Its surviving `resid_raw_mr_*` sleeves are now maintained under residual mean reversion.

================================================================================
                     DISTANCE-PAIRS-MR Signal Sweep Results
================================================================================
                   name  rebalance  net_sharpe  avg_daily_turnover  benchmark_corr  max_drawdown  max_drawdown_duration
0   dist_mr_k3_z60__r10         10      1.0761              0.1291         -0.1795       -0.0971                    229
1    dist_mr_k3_z60__r5          5      0.7863              0.2096         -0.1053       -0.0990                    284
2   dist_mr_k3_z20__r21         21      0.7474              0.0688          0.0486       -0.0997                    865
3   dist_mr_k3_z20__r10         10      0.6847              0.1439          0.1969       -0.0701                    462
4   dist_mr_k3_z10__r10         10      0.6482              0.1331          0.0945       -0.0691                    480
5   dist_mr_k1_z20__r21         21      0.6466              0.0675          0.0095       -0.0711                    394
6   dist_mr_k3_z60__r21         21      0.5713              0.0695          0.4199       -0.3389                    259
7   dist_mr_k1_z60__r10         10      0.5101              0.1255         -0.3896       -0.1916                    602
8   dist_mr_k3_z10__r21         21      0.4808              0.0648          0.0445       -0.0547                    335
9   dist_mr_k1_z60__r21         21      0.4432              0.0646         -0.2539       -0.1201                    555
10  dist_mr_k1_z10__r10         10      0.4260              0.1366          0.0820       -0.0618                    723


===============================================================================
                      SECTOR-RELATIVE Signal Sweep Results
================================================================================
                                            name  rebalance   scaler_config  net_sharpe  avg_daily_turnover  benchmark_corr  max_drawdown  max_drawdown_duration
0      sector_rel_zscore_5_60__r10__trend_20_100         10    trend_20_100      0.5808              0.0508          0.1294       -0.0509                    465
1            sector_rel_mr_5d__r10__trend_20_100         10    trend_20_100      0.4941              0.0520          0.1358       -0.0812                    905
2    sector_rel_zscore_5_60__r10__trend_20_100_h         10  trend_20_100_h      0.4540              0.0735          0.1462       -0.0833                    891
3            sector_rel_mr_5d__r21__trend_50_200         21    trend_50_200      0.4015              0.0278          0.0318       -0.0778                   1279
4               sector_rel_mr_5d__r21__vol_10_60         21       vol_10_60      0.3990              0.0425          0.0205       -0.0978                    742
5      sector_rel_zscore_5_60__r10__trend_50_200         10    trend_50_200      0.3875              0.0558          0.1509       -0.0764                    891
6          sector_rel_mr_5d__r10__trend_20_100_h         10  trend_20_100_h      0.3849              0.0741          0.1471       -0.1079                    905
7            sector_rel_mr_5d__r21__trend_20_100         21    trend_20_100      0.3647              0.0258          0.0730       -0.0838                   1279


================================================================================
                           EVENT Signal Sweep Results
================================================================================
                             name  rebalance   scaler_config  net_sharpe  avg_daily_turnover  benchmark_corr  max_drawdown  max_drawdown_duration
0            resid_gap__r10__none         10            none      0.6303              0.1105          0.1285       -0.1078                   1231
1    resid_gap__r10__trend_50_200         10    trend_50_200      0.6168              0.0515          0.1239       -0.0982                   1274
2      resid_gap__r10__breadth_40         10      breadth_40      0.6016              0.0967          0.1006       -0.0710                    702
3      resid_gap__r10__breadth_60         10      breadth_60      0.5908              0.0841          0.0634       -0.0710                    664
4      resid_gap__r10__breadth_50         10      breadth_50      0.5671              0.0887          0.0668       -0.0710                    703
5       resid_gap__r10__vol_10_60         10       vol_10_60      0.5649              0.0824          0.0807       -0.0788                   1119
6    resid_gap__r10__trend_20_100         10    trend_20_100      0.5431              0.0495          0.1149       -0.0981                   1274
7  resid_gap__r10__trend_20_100_h         10  trend_20_100_h      0.5431              0.0495          0.1149       -0.0981                   1274


================================================================================
                       VOL-TREND-V2 Signal Sweep Results
================================================================================
                              name  rebalance  net_sharpe  avg_daily_turnover  benchmark_corr  max_drawdown  max_drawdown_duration
0           iret_vol_5_10_120__r10         10      0.6253              0.1192          0.1162       -0.0890                    694
1   ivol_explosion_10_120_p95__r21         21      0.5604              0.0001          0.2041       -0.0500                    309
2           iret_vol_5_10_120__r21         21      0.4681              0.0593          0.0572       -0.1314                    584



```python
TOP_STRATEGIES: list[dict[str, object]] = [
    {"group": "monoton", "label": "monoton_120d__r21__disp_60_q30", "signal": "monoton_120d", "rebalance": 21, "risk_scaling": "disp_60_q30"},
    {"group": "monoton", "label": "monoton_120d__r21__disp_60_q20", "signal": "monoton_120d", "rebalance": 21, "risk_scaling": "disp_60_q20"},
    {"group": "monoton", "label": "monoton_120d__r21__vol_20_60", "signal": "monoton_120d", "rebalance": 21, "risk_scaling": "vol_20_60"},
    {"group": "monoton", "label": "monoton_120d__r21__crash_10_5pct", "signal": "monoton_120d", "rebalance": 21, "risk_scaling": "crash_10_5pct"},
    {"group": "monoton", "label": "monoton_120d__r21__none", "signal": "monoton_120d", "rebalance": 21, "risk_scaling": "none"},
    {"group": "mean-reversion", "label": "mr_5d__r10__trend_20_100", "signal": "mr_5d", "rebalance": 10, "risk_scaling": "trend_20_100"},
    {"group": "mean-reversion", "label": "mr_5d__r10__trend_20_100_h", "signal": "mr_5d", "rebalance": 10, "risk_scaling": "trend_20_100_h"},
    {"group": "mean-reversion", "label": "zscore_rev_5_60__r10__trend_20_100", "signal": "zscore_rev_5_60", "rebalance": 10, "risk_scaling": "trend_20_100"},
    {"group": "mean-reversion", "label": "mr_5d__r10__trend_50_200", "signal": "mr_5d", "rebalance": 10, "risk_scaling": "trend_50_200"},
    {"group": "mean-reversion", "label": "mr_15d__r21__vol_10_60", "signal": "mr_15d", "rebalance": 21, "risk_scaling": "vol_10_60"},
    {"group": "resid-mean-reversion", "label": "factor_model_resid_mr_5d__r10__trend_20_100", "signal": "factor_model_resid_mr_5d", "rebalance": 10, "risk_scaling": "trend_20_100"},
    {"group": "resid-mean-reversion", "label": "etf_factor_resid_mr_5d__r10__trend_20_100", "signal": "etf_factor_resid_mr_5d", "rebalance": 10, "risk_scaling": "trend_20_100"},
    {"group": "resid-mean-reversion", "label": "etf_factor_resid_mr_5d__r10__trend_20_100_h", "signal": "etf_factor_resid_mr_5d", "rebalance": 10, "risk_scaling": "trend_20_100_h"},
    {"group": "resid-mean-reversion", "label": "etf_factor_resid_mr_2d__r10__trend_50_200", "signal": "etf_factor_resid_mr_2d", "rebalance": 10, "risk_scaling": "trend_50_200"},
    {"group": "resid-mean-reversion", "label": "etf_factor_resid_mr_5d__r10__trend_50_200", "signal": "etf_factor_resid_mr_5d", "rebalance": 10, "risk_scaling": "trend_50_200"},
    {"group": "resid-mean-reversion", "label": "etf_factor_resid_mr_2d__r10__trend_20_100", "signal": "etf_factor_resid_mr_2d", "rebalance": 10, "risk_scaling": "trend_20_100"},
    {"group": "resid-mean-reversion", "label": "etf_factor_resid_mr_5d__r10__none", "signal": "etf_factor_resid_mr_5d", "rebalance": 10, "risk_scaling": "none"},
    {"group": "resid-mean-reversion", "label": "etf_factor_resid_mr_5d__r21__vol_10_60", "signal": "etf_factor_resid_mr_5d", "rebalance": 21, "risk_scaling": "vol_10_60"},
    {"group": "resid-mean-reversion", "label": "etf_factor_resid_mr_5d__r5__trend_20_100", "signal": "etf_factor_resid_mr_5d", "rebalance": 5, "risk_scaling": "trend_20_100"},
    {"group": "resid-mean-reversion", "label": "factor_model_resid_mr_2d__r10__trend_50_200", "signal": "factor_model_resid_mr_2d", "rebalance": 10, "risk_scaling": "trend_50_200"},
    {"group": "resid-mean-reversion", "label": "factor_model_resid_mr_2d__r10__trend_plus_vol", "signal": "factor_model_resid_mr_2d", "rebalance": 10, "risk_scaling": "trend_plus_vol"},
    {"group": "resid-mean-reversion", "label": "factor_model_resid_mr_5d__r10__trend_plus_vol", "signal": "factor_model_resid_mr_5d", "rebalance": 10, "risk_scaling": "trend_plus_vol"},
    {"group": "resid-mean-reversion", "label": "factor_model_resid_mr_2d__r10__trend_20_100", "signal": "factor_model_resid_mr_2d", "rebalance": 10, "risk_scaling": "trend_20_100"},
    {"group": "distance-pairs-mr", "label": "dist_mr_k3_z60__r10", "signal": "dist_mr_k3_z60", "rebalance": 10, "risk_scaling": "none"},
    {"group": "distance-pairs-mr", "label": "dist_mr_k3_z60__r5", "signal": "dist_mr_k3_z60", "rebalance": 5, "risk_scaling": "none"},
    {"group": "distance-pairs-mr", "label": "dist_mr_k3_z20__r21", "signal": "dist_mr_k3_z20", "rebalance": 21, "risk_scaling": "none"},
    {"group": "distance-pairs-mr", "label": "dist_mr_k3_z20__r10", "signal": "dist_mr_k3_z20", "rebalance": 10, "risk_scaling": "none"},
    {"group": "distance-pairs-mr", "label": "dist_mr_k3_z10__r10", "signal": "dist_mr_k3_z10", "rebalance": 10, "risk_scaling": "none"},
    {"group": "sector-relative", "label": "sector_rel_zscore_5_60__r10__trend_20_100", "signal": "sector_rel_zscore_5_60", "rebalance": 10, "risk_scaling": "trend_20_100"},
    {"group": "sector-relative", "label": "sector_rel_mr_5d__r10__trend_20_100", "signal": "sector_rel_mr_5d", "rebalance": 10, "risk_scaling": "trend_20_100"},
    {"group": "sector-relative", "label": "sector_rel_zscore_5_60__r10__trend_20_100_h", "signal": "sector_rel_zscore_5_60", "rebalance": 10, "risk_scaling": "trend_20_100_h"},
    {"group": "sector-relative", "label": "sector_rel_mr_5d__r21__trend_50_200", "signal": "sector_rel_mr_5d", "rebalance": 21, "risk_scaling": "trend_50_200"},
    {"group": "sector-relative", "label": "sector_rel_mr_5d__r21__vol_10_60", "signal": "sector_rel_mr_5d", "rebalance": 21, "risk_scaling": "vol_10_60"},
    {"group": "event", "label": "resid_gap__r10__none", "signal": "resid_gap", "rebalance": 10, "risk_scaling": "none"},
    {"group": "event", "label": "resid_gap__r10__trend_50_200", "signal": "resid_gap", "rebalance": 10, "risk_scaling": "trend_50_200"},
    {"group": "event", "label": "resid_gap__r10__breadth_40", "signal": "resid_gap", "rebalance": 10, "risk_scaling": "breadth_40"},
    {"group": "event", "label": "resid_gap__r10__breadth_60", "signal": "resid_gap", "rebalance": 10, "risk_scaling": "breadth_60"},
    {"group": "event", "label": "resid_gap__r10__vol_10_60", "signal": "resid_gap", "rebalance": 10, "risk_scaling": "vol_10_60"},
]
```