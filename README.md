# Financial Machine Learning: A Beginner's Attempt

A hands-on implementation study of the framework in Marcos López de Prado's *Advances in Financial Machine Learning* (AFML, 2018). It takes raw tick data all the way to a walk-forward-validated, bet-sized strategy, and critically reviews where each technique worked, where it didn't, and where my implementation cut corners.

**Data:** Binance SOL/USDT aggregated trades, October 2023 (3.1M ticks), from [Binance Data Vision](https://data.binance.vision/).

📄 **Full write-up:** [`AFML Implementation Report.pdf`](./AFML%20Implementation%20Report.pdf)
📓 **Code:** [`Beginner's Attempt on Financial ML (Ver1).ipynb`](./Beginner's%20Attempt%20on%20Financial%20ML%20(Ver1).ipynb)

## Pipeline

| Stage | AFML technique | What I found |
|---|---|---|
| 1. Bars | Dollar bars + order-flow imbalance (OFI) | 3.1M ticks → 4,549 dollar bars; per-bar OFI from the `is_buyer_maker` flag |
| 2. Distribution check | Dollar bars vs time bars | Dollar-bar returns have excess kurtosis closer to zero (−0.09 vs −0.65 for 1-min bars) |
| 3. Features | Fractional differentiation (FFD) | **Trade-off:** no *d* made cumulative OFI stationary without destroying its memory, so I kept memory and accepted mild non-stationarity |
| 4. Labels | Triple-barrier method + sample-uniqueness weights | Balanced labels (52% up / 46% down / 2% timeout); mean uniqueness 0.048 reflects heavy label overlap |
| 5. Validation | Purged K-fold CV with embargo | 5-fold accuracy 0.525 ± 0.051: a weak signal that varies across regimes |
| 6. Meta-labeling | Primary RF + meta-model, isotonic calibration | Calibration was essential: win rate rises from 58% to 65–67% at high confidence, keeping only 4–20% of trades |
| 7. Bet sizing | Linear, thresholded, regime-aware and continuous sizing | Dynamic sizing gave a similar Sharpe with much smaller drawdown and per-trade volatility |
| 8. Overfitting defence | PSR, Deflated Sharpe Ratio, walk-forward with costs | In-sample Sharpe 2.71 → **walk-forward 1.27** |

## Known limitations

I document these on purpose. They are the main thing this project taught me.

- **The P&L encoding is simplified and partly wrong.** Trade outcomes come from prediction correctness, not realised returns, so a correct short can book as a loss. Treat every Sharpe, drawdown, PSR and DSR figure as a *relative diagnostic*, not real strategy economics.
- **Purging is approximate.** I used a fixed embargo window, not purging by true label end times (t₁), and did not implement Combinatorial Purged CV. Some leakage is possible.
- **Thresholds were selected after observation.** I compared several confidence thresholds and picked the best, which is the data snooping the DSR is meant to penalise. The DSR's trial count is my own estimate.
- **The sample is narrow:** one asset, one month. The OFI predictability regression ran on only a small live sample.

## Next steps

- Fix the P&L to use realised bar returns, then rerun all the metrics
- Use full t₁-based purging and add CPCV
- Apply FFD to per-bar (non-cumulative) OFI
- Extend to more assets and a longer period

## Running it

```bash
pip install numpy pandas scipy statsmodels scikit-learn matplotlib requests pyarrow
jupyter notebook "Beginner's Attempt on Financial ML (Ver1).ipynb"
```

The notebook downloads the monthly aggTrades file from Binance Data Vision and caches it as Parquet, so it needs internet access on the first run.

## Reference

M. López de Prado, *Advances in Financial Machine Learning*, Wiley, 2018.
