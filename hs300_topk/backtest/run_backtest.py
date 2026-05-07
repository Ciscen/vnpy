"""
hs300_topk/backtest/run_backtest.py

回测入口脚本（仅回测阶段）。
完整流水线（含数据下载）请使用 run_pipeline.py。

用法::

    python -m hs300_topk.backtest.run_backtest
"""
from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

from vnpy.trader.constant import Interval
from vnpy.alpha import AlphaLab
from vnpy.alpha.strategy import BacktestingEngine

from hs300_topk.data.loader import get_lab, discover_symbols
from hs300_topk.model.rolling_trainer import rolling_train
from hs300_topk.strategy.hs300_topk_strategy import HS300Top10Strategy
from hs300_topk.backtest.evaluation import print_metrics, show_charts, export_report
from hs300_topk.data.downloader import ensure_component_index
from hs300_topk.pipeline_config import PIPELINE

# ──────────────────────────────────────────────────
# 配置 — 来自 pipeline_config 统一管理
# ──────────────────────────────────────────────────
LAB_PATH = PIPELINE.lab_path
DATA_START = PIPELINE.data_start
DATA_END = PIPELINE.data_end
BACKTEST_START = PIPELINE.backtest_start
BACKTEST_END = PIPELINE.backtest_end
CAPITAL = PIPELINE.capital

REPORT_DIR = Path("hs300_topk") / "output"


def main() -> None:
    print("=" * 60)
    print("  HS300 Top-K 周度选股策略回测")
    print(f"  数据区间: {DATA_START} ~ {DATA_END}")
    print(f"  回测区间: {BACKTEST_START} ~ {BACKTEST_END}")
    print("=" * 60)

    lab = get_lab(LAB_PATH)
    vt_symbols = discover_symbols(LAB_PATH)

    if not vt_symbols:
        print("错误: lab 目录中无数据，请先运行 python -m hs300_topk.run_pipeline")
        sys.exit(1)

    ensure_component_index(lab, vt_symbols)

    # ── Step 1: 滚动训练 → 信号 ──
    print("\n" + "=" * 60)
    print("  Phase 1: 滚动训练")
    print("=" * 60)

    signal_df, _ = rolling_train(
        lab_path=LAB_PATH,
        data_start=DATA_START,
        data_end=DATA_END,
        backtest_start=BACKTEST_START,
        backtest_end=BACKTEST_END,
    )

    # ── Step 2: 回测 ──
    print("\n" + "=" * 60)
    print("  Phase 2: 策略回测")
    print("=" * 60)

    engine = BacktestingEngine(lab)

    engine.set_parameters(
        vt_symbols=vt_symbols,
        interval=Interval.DAILY,
        start=datetime.fromisoformat(BACKTEST_START),
        end=datetime.fromisoformat(BACKTEST_END),
        capital=CAPITAL,
    )

    setting = {
        "top_k": 10,
        "stop_loss_pct": 0.03,
        "tp_activate_pct": 0.03,
        "tp_trail_pct": 0.02,
        "max_hold_days": 4,
        "cash_ratio": 0.95,
        "min_volume": 100,
        "price_add": 0.002,
    }

    engine.add_strategy(HS300Top10Strategy, setting, signal_df)
    print("\n加载历史数据 ...")
    engine.load_data()

    print("开始回测 ...")
    engine.run_backtesting()

    print("计算逐日盈亏 ...")
    engine.calculate_result()

    # ── Step 3: 绩效 ──
    print("\n" + "=" * 60)
    print("  Phase 3: 绩效评估")
    print("=" * 60)

    stats = engine.calculate_statistics()
    print_metrics(stats)

    # ── 导出报告 ──
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    export_report(engine, stats, REPORT_DIR)

    # ── 图表 ──
    try:
        show_charts(engine, benchmark_symbol="000300.SSE")
    except Exception as e:
        print(f"[提示] 图表展示跳过: {e}")

    print("\n回测完成。")


if __name__ == "__main__":
    main()
