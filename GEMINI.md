# Project Instructions for Gemini CLI

This document contains team-shared architecture, conventions, workflows, and guidelines for the `vnpy` (HS300 Top-K) quantitative trading project. **These rules take absolute precedence over the general workflows.**

## 1. Quantitative Development & Iteration Guidelines

When developing, tuning, or backtesting trading strategies, you MUST strictly adhere to the following rules to prevent common statistical pitfalls:

*   **Survivorship Bias**: NEVER backtest using only the "current" constituents of an index (e.g., today's HS300). You MUST use Point-in-Time (PIT) historical snapshot data (e.g., via `BaoStock`) to filter the trading universe. The model must only see stocks that were *actually* in the index on day T.
*   **Overfitting & Look-Ahead Bias**: Do NOT artificially inflate backtest performance by blindly increasing model complexity (`max_depth`, `n_estimators`) or piling on homogeneous factors. Feature engineering and label generation MUST be strictly backward-looking. Never use future data (like full-period means or unadjusted prices that leak future corporate actions).
*   **No Tuning on Test Set**: NEVER perform repeated grid search or hyperparameter selection on the Out-of-Sample (OOS) validation period. All parameters must be fixed in the In-Sample (IS) period and then tested "read-only" on the OOS window (`oos_validation`).
*   **Bear Market Deterioration & Regime Conflict**: Do not declare a strategy successful based solely on its performance in a bull market. You MUST stress-test across a full bull/bear cycle (e.g., the 2022-2023 bear market). If a strategy fails heavily in a bear market, do NOT mix conflicting labels within a single tree model (this confuses the tree splitting). Instead, resolve macroeconomic regime issues externally using **Ensemble/Meta-Models** (e.g., dynamically switching between Alpha and Beta models based on a macro indicator like MA60).

## 2. Dashboard & Reporting Requirements

When generating backtest reports or dashboards (e.g., `dashboard.html`, `strategy_overview.html`):
*   **Baseline Comparison**: You MUST include the HS300 index (`000300.SSE`) as the baseline. Charts (especially equity curves) must clearly plot the strategy's performance against the HS300 baseline.
*   **Full Cycle Representation**: Backtest data and charts must cover the full bull/bear cycle (e.g., 2022-2026), not just a single trend. Ensure metrics reflect performance across these different regimes.

## 3. Workflow: Feishu Notification

Upon completing a strategy iteration, a core task, or when significantly blocked and needing user input, you MUST send a summary notification via Feishu.
*   The notification should summarize:
    *   Task progress and optimization points.
    *   Core backtest metrics (Total Return, Annualized Return, Sharpe Ratio, Max Drawdown).
    *   Any pitfalls encountered and how they were avoided.
    *   (Highly Recommended) Attach a visual chart (like the equity curve comparison) by calling the Feishu API.
