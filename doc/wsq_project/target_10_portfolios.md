1. “Low Turnover Core”

* resid_mr_5d__r10__trend_20_100__cond__residual_dispersion_high_20_q75
* resid_mr_2d__r10__trend_plus_vol__cond__dispersion_high_60_q75
* mr_5d__r10__trend_20_100__cond__breadth_weak_40
* dist_mr_k3_z20__r21__cond__vol_contraction_10_60

2. “Crisis / Panic Diversifier”

* dist_mr_k3_z10__r10__cond__panic_10d_minus5
* mr_5d__r10__trend_20_100__cond__market_trend_down_20_100
* resid_raw_mr_w2__r10__cond__market_trend_down_20_100
* dist_mr_k3_z60__r5__cond__vol_expansion_10_60

3. “Residual Heavy”

* resid_mr_5d__r10__trend_20_100__cond__residual_dispersion_high_20_q75
* resid_mr_2d__r10__trend_50_200__cond__dispersion_high_60_q75
* resid_raw_mr_w5__r10__cond__market_trend_down_20_100
* resid_raw_mr_w5__r21__cond__vol_contraction_10_60

4. “Fast + Slow Horizon Blend”

* dist_mr_k3_z60__r5__cond__vol_expansion_10_60
* dist_mr_k3_z60__r10__cond__none
* dist_mr_k3_z20__r21__cond__vol_contraction_10_60
* resid_raw_mr_w5__r21__cond__vol_contraction_10_60

5. “High Convexity Portfolio”

* dist_mr_k3_z60__r5__cond__vol_expansion_10_60
* dist_mr_k3_z10__r10__cond__panic_10d_minus5
* dist_mr_k3_z60__r10__cond__none
* resid_mr_2d__r10__trend_plus_vol__cond__dispersion_high_60_q75

6. “Conservative Institutional Style”

* resid_mr_5d__r10__trend_20_100__cond__residual_dispersion_high_20_q75
* mr_5d__r10__trend_50_200__cond__breadth_weak_40
* dist_mr_k3_z20__r21__cond__vol_contraction_10_60
* dist_mr_k3_z60__r10__cond__breadth_weak_40

7. “Conditioning Orthogonalization”

* dist_mr_k3_z60__r10__cond__volume_spike_20_q80
* dist_mr_k3_z20__r21__cond__vol_contraction_10_60
* dist_mr_k3_z10__r10__cond__breadth_weak_40
* dist_mr_k3_z10__r10__cond__panic_10d_minus5

8. “Negative Beta Lean”

* dist_mr_k3_z60__r10__cond__none
* resid_mr_2d__r10__trend_plus_vol__cond__dispersion_high_60_q75
* resid_raw_mr_w2__r10__cond__dispersion_high_60_q75
* dist_mr_k3_z60__r5__cond__vol_expansion_10_60

9. “Balanced Multi-Engine”

* dist_mr_k3_z20__r21__cond__vol_contraction_10_60
* mr_5d__r10__trend_20_100__cond__breadth_weak_40
* resid_mr_5d__r10__trend_plus_vol__cond__residual_dispersion_high_20_q75
* resid_raw_mr_w2__r10__cond__dispersion_high_60_q75
* dist_mr_k3_z10__r10__cond__panic_10d_minus5
* dist_mr_k3_z60__r10__cond__volume_spike_20_q80


10. “Barbell Portfolio”
    Defensive side:

* resid_mr_5d__r10__trend_20_100__cond__residual_dispersion_high_20_q75
* mr_5d__r10__trend_50_200__cond__breadth_weak_40
* dist_mr_k3_z20__r21__cond__vol_contraction_10_60

Aggressive side:

* dist_mr_k3_z60__r5__cond__vol_expansion_10_60
* dist_mr_k3_z60__r10__cond__none
* dist_mr_k3_z10__r10__cond__panic_10d_minus5

## Extensions

After running each of htese portfolios and getting a baseline for each try adding these distinct strategy families. So there should be 10 results for the baseline, and then 15 * 10 for the extensions.

1. Persistent / monotonic momentum
{"group": "monoton", "label": "monoton_120d__r21__disp_60_q30", "signal": "monoton_120d", "rebalance": 21, "legacy_scaler": "disp_60_q30"},
{"group": "monoton", "label": "monoton_120d__r21__disp_60_q20", "signal": "monoton_120d", "rebalance": 21, "legacy_scaler": "disp_60_q20"},
{"group": "monoton", "label": "monoton_120d__r21__vol_20_60", "signal": "monoton_120d", "rebalance": 21, "legacy_scaler": "vol_20_60"},
{"group": "monoton", "label": "monoton_120d__r21__crash_10_5pct", "signal": "monoton_120d", "rebalance": 21, "legacy_scaler": "crash_10_5pct"},
{"group": "monoton", "label": "monoton_120d__r21__none", "signal": "monoton_120d", "rebalance": 21, "legacy_scaler": "none"},

1. Event / post-shock reaction

{"group": "event", "label": "resid_gap__r10__none", "signal": "resid_gap", "rebalance": 10, "legacy_scaler": "none"},
{"group": "event", "label": "resid_gap__r10__trend_50_200", "signal": "resid_gap", "rebalance": 10, "legacy_scaler": "trend_50_200"},
{"group": "event", "label": "resid_gap__r10__breadth_40", "signal": "resid_gap", "rebalance": 10, "legacy_scaler": "breadth_40"},
{"group": "event", "label": "resid_gap__r10__breadth_60", "signal": "resid_gap", "rebalance": 10, "legacy_scaler": "breadth_60"},
{"group": "event", "label": "resid_gap__r10__vol_10_60", "signal": "resid_gap", "rebalance": 10, "legacy_scaler": "vol_10_60"},

3. Sector-relative momentum / dispersion rotation
{"group": "sector-relative", "label": "sector_rel_zscore_5_60__r10__trend_20_100", "signal": "sector_rel_zscore_5_60", "rebalance": 10, "legacy_scaler": "trend_20_100"},
{"group": "sector-relative", "label": "sector_rel_mr_5d__r10__trend_20_100", "signal": "sector_rel_mr_5d", "rebalance": 10, "legacy_scaler": "trend_20_100"},
{"group": "sector-relative", "label": "sector_rel_zscore_5_60__r10__trend_20_100_h", "signal": "sector_rel_zscore_5_60", "rebalance": 10, "legacy_scaler": "trend_20_100_h"},
{"group": "sector-relative", "label": "sector_rel_mr_5d__r21__trend_50_200", "signal": "sector_rel_mr_5d", "rebalance": 21, "legacy_scaler": "trend_50_200"},
{"group": "sector-relative", "label": "sector_rel_mr_5d__r21__vol_10_60", "signal": "sector_rel_mr_5d", "rebalance": 21, "legacy_scaler": "vol_10_60"},

