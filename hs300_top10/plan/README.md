# HS300 Top‑10 Weekly Strategy

This repository implements a weekly long‑only strategy for the HS300 index using **XGBoost** as the prediction model.

## Project Structure
```
hs300_top10/
│   README.md                # (this file)
│   requirements.txt         # Python dependencies
│
├─ data/                      # Data download & loading
│   │   download_data.py
│   │   loader.py
│
├─ features/                  # Feature engineering & label creation
│   │   engineer.py
│   │   labeler.py
│
├─ model/                     # Training, rolling training and prediction
│   │   trainer.py
│   │   rolling_trainer.py
│   │   predictor.py
│
├─ strategy/                  # vnpy strategy implementation
│   │   hs300_top10_strategy.py
│
├─ backtest/                  # Back‑testing entry and evaluation
│   │   run_backtest.py
│   │   evaluation.py
│
└─ tests/                     # Unit tests (not included here)
```

## Usage
1. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```
2. **Download data** (cached locally)
   ```bash
   python -m hs300_top10.data.download_data
   ```
3. **Run weekly rolling training** (can be scheduled weekly)
   ```bash
   python -m hs300_top10.model.rolling_trainer
   ```
4. **Back‑test the strategy**
   ```bash
   python -m hs300_top10.backtest.run_backtest
   ```
5. **Deploy in live trading** – load `hs300_top10.strategy.HS300Top10Strategy` into your vnpy CTA engine.

---

**Key points**
- Features are computed **up to the Monday close** of each week.
- The entry price is the **Tuesday open** (the second trading day of the week).
- Labels are 1 when the max price from that Tuesday open to the Friday close is ≥ **5%** above the Tuesday open.
- Position management includes a hard stop‑loss at **‑3%**, a trailing‑take‑profit (activate at **+3%**, exit when price falls **2%** from the peak), and a forced exit after **4 trading days**.
- All back‑test calculations incorporate a **0.2%** round‑trip transaction cost and a **0.2%** slippage (0.002).

---

Feel free to adjust hyper‑parameters in `hs300_top10/model/trainer.py` or the strategy file.
