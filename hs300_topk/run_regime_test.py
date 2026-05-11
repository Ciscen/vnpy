"""
hs300_topk/run_regime_test.py

Regime 过滤效果验证: 多区间对比 V1.4 与 V1.4R。
"""
from __future__ import annotations

import logging
import sys
import time
from datetime import datetime

import polars as pl

from vnpy.trader.constant import Interval
from vnpy.alpha.strategy import BacktestingEngine

from hs300_topk.data.loader import get_lab, discover_symbols
from hs300_topk.model.rolling_trainer import rolling_train
from hs300_topk.strategy.hs300_topk_strategy import HS300Top10Strategy
from hs300_topk.strategy.config import OPTIMIZED_V14, OPTIMIZED_V14R

LAB_PATH = "./lab/hs300"
CAPITAL = 100_000
WEEKLY_LABEL = "friday_close"
LAG_DAYS = 3

PERIODS = {
    "bear_2022_2023": {
        "data_start": "2016-04-30",
        "data_end": "2023-12-31",
        "bt_start": "2022-01-01",
        "bt_end": "2023-12-31",
    },
    "bull_2024_2026": {
        "data_start": "2016-04-30",
        "data_end": "2026-04-30",
        "bt_start": "2024-05-01",
        "bt_end": "2026-04-30",
    },
    "full_2022_2026": {
        "data_start": "2016-04-30",
        "data_end": "2026-04-30",
        "bt_start": "2022-01-01",
        "bt_end": "2026-04-30",
    },
}

CONFIGS = {
    "V1.4": OPTIMIZED_V14,
    "V1.4R": OPTIMIZED_V14R,
}


def run_period(period_name: str, period: dict) -> dict[str, dict]:
    t0 = time.time()
    print(f"\n{'='*70}")
    print(f"  {period_name}: {period['bt_start']} ~ {period['bt_end']}")
    print(f"{'='*70}")

    signal_df, _ = rolling_train(
        lab_path=LAB_PATH,
        data_start=period["data_start"],
        data_end=period["data_end"],
        backtest_start=period["bt_start"],
        backtest_end=period["bt_end"],
        weekly_label=WEEKLY_LABEL,
        lag_days=LAG_DAYS,
    )
    print(f"  信号: {signal_df.shape[0]} 行, 耗时 {time.time()-t0:.0f}s")

    logging.disable(logging.CRITICAL)
    results: dict[str, dict] = {}
    for ver, cfg in CONFIGS.items():
        lab = get_lab(LAB_PATH)
        vt_symbols = discover_symbols(LAB_PATH)
        needs_bench = cfg.use_market_filter or cfg.regime_filter
        if needs_bench and cfg.market_benchmark not in vt_symbols:
            vt_symbols = vt_symbols + [cfg.market_benchmark]
            lab.add_contract_setting(cfg.market_benchmark, 0, 0, 1, 0.01)
        engine = BacktestingEngine(lab)
        engine.set_parameters(
            vt_symbols=vt_symbols,
            interval=Interval.DAILY,
            start=datetime.fromisoformat(period["bt_start"]),
            end=datetime.fromisoformat(period["bt_end"]),
            capital=CAPITAL,
        )
        setting = {
            k: v for k, v in cfg.to_dict().items()
            if k not in ("version", "description")
            and not k.startswith("xgb_") and k != "train_years"
        }
        engine.add_strategy(HS300Top10Strategy, setting, signal_df)
        engine.load_data()
        engine.run_backtesting()
        engine.calculate_result()
        stats = engine.calculate_statistics()
        results[ver] = stats
    logging.disable(logging.NOTSET)
    return results


def print_results(period_name: str, results: dict[str, dict]) -> None:
    header = (
        f"{'版本':<8s} {'Sharpe':>8s} {'年化%':>8s} {'总收益%':>10s} "
        f"{'最大回撤%':>10s} {'收益回撤比':>10s} {'交易笔数':>8s}"
    )
    print(f"\n  [{period_name}]")
    print(header)
    print("-" * 80)
    for ver, stats in results.items():
        sh = stats.get("sharpe_ratio", 0)
        ar = stats.get("annual_return", 0)
        tr = stats.get("total_return", 0)
        md = stats.get("max_ddpercent", 0)
        rd = stats.get("return_drawdown_ratio", 0)
        tc = stats.get("total_trade_count", 0)
        print(
            f"{ver:<8s} {sh:>8.2f} {ar:>8.1f} {tr:>10.1f} "
            f"{md:>10.1f} {rd:>10.2f} {tc:>8.0f}"
        )
    print("-" * 80)


def main() -> None:
    t0 = time.time()
    all_results: dict[str, dict[str, dict]] = {}

    for pname, pconfig in PERIODS.items():
        all_results[pname] = run_period(pname, pconfig)

    print("\n" + "=" * 80)
    print("  Regime 过滤效果 — V1.4 vs V1.4R 多区间对比")
    print("=" * 80)
    for pname, results in all_results.items():
        print_results(pname, results)

    print(f"\n总耗时: {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
