import pandas as pd
import numpy as np


def vol_backtest(prices, target_vol=0.15, lookback=20, max_leverage=2.0):
    base_returns = np.log(prices / prices.shift(1))
    realised_vol = base_returns.rolling(window=lookback).std() * np.sqrt(252)
    weights = target_vol / realised_vol
    valid_weights = weights.clip(upper=max_leverage)
    weight_used = valid_weights.shift(1)

    daily_vol_returns = base_returns * weight_used
    total_vol_returns = daily_vol_returns.cumsum().apply(np.exp)
    total_base_returns = base_returns.cumsum().apply(np.exp)

    results = pd.DataFrame({
        "base_returns": base_returns,
        "total_base_returns": total_base_returns,
        "daily_vol_returns": daily_vol_returns,
        "total_vol_returns": total_vol_returns
    }).dropna()

    return results
