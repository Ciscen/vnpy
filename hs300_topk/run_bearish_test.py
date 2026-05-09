"""
hs300_topk/run_bearish_test.py

熊市区间 (2022-01 ~ 2023-12) Lag-3 回测脚本。
独立训练新信号，再跑 V1.3 / V1.4 / V1.5 策略回测，验证 Lag-3 特征在下行市场中的稳健性。
"""
from __future__ import annotations

import logging
import time
from datetime import datetime

import polars as pl

from vnpy.trader.constant import Interval
from vnpy.alpha.strategy import BacktestingEngine

from hs300_topk.data.loader import get_lab, discover_symbols
from hs300_topk.model.rolling_trainer import rolling_train
from hs300_topk.strategy.hs300_topk_strategy import HS300Top10Strategy
from hs300_topk.strategy.config import OPTIMIZED_V13, OPTIMIZED_V14, OPTIMIZED_V15

LAB_PATH = "./lab/hs300"
DATA_START = "2016-04-30"
DATA_END = "2023-12-31"
BT_START = "2022-01-01"
BT_END = "2023-12-31"
CAPITAL = 100_000
WEEKLY_LABEL = "friday_close"
LAG_DAYS = 3


def main() -> None:
    t0 = time.time()
    print("=" * 60)
    print(f"  熊市区间 Lag-3 回测: {BT_START} ~ {BT_END}")
    print(f"  训练范围: {DATA_START} ~ {DATA_END}")
    print(f"  标签模式: {WEEKLY_LABEL}, lag_days={LAG_DAYS}")
    print("=" * 60)

    signal_df, _ = rolling_train(
        lab_path=LAB_PATH,
        data_start=DATA_START,
        data_end=DATA_END,
        backtest_start=BT_START,
        backtest_end=BT_END,
        weekly_label=WEEKLY_LABEL,
        lag_days=LAG_DAYS,
    )

    train_sec = time.time() - t0
    print(f"\n训练完成: {signal_df.shape[0]} 行信号, 耗时 {train_sec:.0f}s")
    print(f"信号范围: {signal_df['datetime'].min()} ~ {signal_df['datetime'].max()}")

    configs = {
        "V1.3": OPTIMIZED_V13,
        "V1.4": OPTIMIZED_V14,
        "V1.5": OPTIMIZED_V15,
    }

    logging.disable(logging.CRITICAL)
    results: dict[str, dict] = {}
    for ver, cfg in configs.items():
        lab = get_lab(LAB_PATH)
        vt_symbols = discover_symbols(LAB_PATH)
        engine = BacktestingEngine(lab)
        engine.set_parameters(
            vt_symbols=vt_symbols,
            interval=Interval.DAILY,
            start=datetime.fromisoformat(BT_START),
            end=datetime.fromisoformat(BT_END),
            capital=CAPITAL,
        )
        setting = {
            k: v
            for k, v in cfg.to_dict().items()
            if k not in ("version", "description")
            and not k.startswith("xgb_")
            and k != "train_years"
        }
        engine.add_strategy(HS300Top10Strategy, setting, signal_df)
        engine.load_data()
        engine.run_backtesting()
        engine.calculate_result()
        stats = engine.calculate_statistics()
        results[ver] = stats
    logging.disable(logging.NOTSET)

    print("\n" + "=" * 80)
    print("  熊市区间 (2022-01 ~ 2023-12) Lag-3 回测结果")
    print("=" * 80)
    header = (
        f"{'版本':<8s} {'Sharpe':>8s} {'年化%':>8s} {'总收益%':>10s} "
        f"{'最大回撤%':>10s} {'收益回撤比':>10s} {'交易笔数':>8s}"
    )
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
    print("=" * 80)
    print(f"\n总耗时: {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
