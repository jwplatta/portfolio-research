# v0 Event Driven Strategies

## Overnight Gap Continuation/Reversal

This strategy studies how stocks behave after large overnight moves between the previous close and next open. The core idea is that some overnight gaps represent temporary overreactions that mean-revert intraday, while others reflect genuine new information that continues trending throughout the day or over several days. The key challenge is distinguishing between the two regimes. Useful conditioning variables include overnight gap size, relative volume at the open, premarket activity, market regime, and whether the move aligns with the broader market or sector. Mechanically, this is attractive because overnight gaps often concentrate information shocks, earnings reactions, macro news, and liquidity imbalances into a single observable event.

## Volume Shock Continuation

This strategy focuses on unusually large increases in trading volume as a proxy for institutional participation, information arrival, or forced flow. The intuition is that price moves accompanied by abnormally high volume are more likely to represent persistent repricing rather than noise. For example, a stock with a large positive return and a 5-sigma volume spike may continue trending over the following days because large participants cannot fully establish positions immediately. Typical signals involve volume z-scores, relative volume ratios, and combinations of return magnitude with abnormal participation. The strategy can be implemented as either continuation or reversal depending on horizon and conditioning.