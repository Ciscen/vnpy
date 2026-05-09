"""
hs300_topk/run_oos_validation.py

严格 OOS（样本外）分窗验证脚本。

目的：检测 V1.0~V1.5 策略参数是否存在"面向测试集调参"问题。

方法：
  1. 调参期（IS）：2024-05-01 ~ 2025-04-30  → 选出最优版本
  2. 验证期（OOS）：2025-05-01 ~ 2026-04-30  → 仅跑调参期胜者
  3. 对比 IS 与 OOS 绩效衰减幅度，判断过拟合程度

用法::

    python -m hs300_topk.run_oos_validation --weekly-label friday_close
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from datetime import datetime
from pathlib import Path

import polars as pl

from vnpy.trader.constant import Interval
from vnpy.alpha.strategy import BacktestingEngine

from hs300_topk.data.loader import get_lab, discover_symbols
from hs300_topk.strategy.hs300_topk_strategy import HS300Top10Strategy
from hs300_topk.strategy.config import (
    StrategyConfig,
    BASELINE_V10,
    OPTIMIZED_V11,
    OPTIMIZED_V12,
    OPTIMIZED_V13,
    OPTIMIZED_V14,
    OPTIMIZED_V15,
)
from hs300_topk.pipeline_config import PIPELINE

ALL_CONFIGS: dict[str, StrategyConfig] = {
    "v1.0": BASELINE_V10,
    "v1.1": OPTIMIZED_V11,
    "v1.2": OPTIMIZED_V12,
    "v1.3": OPTIMIZED_V13,
    "v1.4": OPTIMIZED_V14,
    "v1.5": OPTIMIZED_V15,
}

METRIC_KEYS = [
    ("sharpe_ratio", "Sharpe"),
    ("annual_return", "年化%"),
    ("total_return", "总收益%"),
    ("max_ddpercent", "最大回撤%"),
    ("return_drawdown_ratio", "收益回撤比"),
    ("total_trade_count", "交易笔数"),
]


def _run_backtest(
    signal_df: pl.DataFrame,
    config: StrategyConfig,
    backtest_start: str,
    backtest_end: str,
    capital: int,
    lab_path: str,
) -> dict:
    """对指定窗口执行一次回测，返回统计字典。"""
    lab = get_lab(lab_path)
    vt_symbols = discover_symbols(lab_path)

    engine = BacktestingEngine(lab)
    engine.set_parameters(
        vt_symbols=vt_symbols,
        interval=Interval.DAILY,
        start=datetime.fromisoformat(backtest_start),
        end=datetime.fromisoformat(backtest_end),
        capital=capital,
    )

    setting = {
        k: v for k, v in config.to_dict().items()
        if k not in ("version", "description") and not k.startswith("xgb_")
        and k != "train_years"
    }
    engine.add_strategy(HS300Top10Strategy, setting, signal_df)
    engine.load_data()
    engine.run_backtesting()
    engine.calculate_result()
    return engine.calculate_statistics()


def _print_comparison_table(
    is_results: dict[str, dict],
    oos_result: tuple[str, dict] | None,
) -> str:
    """打印对比表格，返回文本形式供飞书使用。"""
    lines: list[str] = []

    header = f"  {'版本':<8s}" + "".join(f"  {label:>10s}" for _, label in METRIC_KEYS)
    sep = "-" * len(header)

    lines.append("=" * 60)
    lines.append("  调参期 (IS) 各版本绩效")
    lines.append("=" * 60)
    lines.append(header)
    lines.append(sep)

    for ver, stats in sorted(is_results.items()):
        row = f"  {ver:<8s}"
        for key, _ in METRIC_KEYS:
            val = stats.get(key, 0)
            row += f"  {val:>10.2f}"
        lines.append(row)

    lines.append(sep)

    if oos_result:
        ver, stats = oos_result
        lines.append("")
        lines.append("=" * 60)
        lines.append(f"  验证期 (OOS) — {ver}")
        lines.append("=" * 60)
        lines.append(header)
        lines.append(sep)
        row = f"  {ver:<8s}"
        for key, _ in METRIC_KEYS:
            val = stats.get(key, 0)
            row += f"  {val:>10.2f}"
        lines.append(row)
        lines.append(sep)

        is_stats = is_results[ver]
        lines.append("")
        lines.append("  IS → OOS 绩效衰减:")
        for key, label in METRIC_KEYS:
            is_val = is_stats.get(key, 0)
            oos_val = stats.get(key, 0)
            if is_val != 0:
                pct = (oos_val - is_val) / abs(is_val) * 100
                lines.append(f"    {label}: {is_val:.2f} → {oos_val:.2f} ({pct:+.1f}%)")
            else:
                lines.append(f"    {label}: {is_val:.2f} → {oos_val:.2f}")

    text = "\n".join(lines)
    print(text)
    return text


def main() -> None:
    parser = argparse.ArgumentParser(description="HS300 Top-K 严格 OOS 验证")
    parser.add_argument(
        "--weekly-label",
        choices=["high_touch", "friday_close"],
        default="friday_close",
    )
    parser.add_argument("--is-metric", default="sharpe_ratio",
                        help="调参期选优指标 (默认 sharpe_ratio)")
    args = parser.parse_args()

    pipe = PIPELINE
    cache_path = (
        pipe.signal_cache_weekly_realistic
        if args.weekly_label == "friday_close"
        else pipe.signal_cache
    )

    if not cache_path.exists():
        print(f"错误: 信号缓存不存在 ({cache_path})，请先运行 run_pipeline")
        sys.exit(1)

    signal_df = pl.read_parquet(cache_path)
    print(f"加载信号: {signal_df.shape[0]} 行, "
          f"{signal_df['datetime'].min()} ~ {signal_df['datetime'].max()}")

    is_start = pipe.oos_tuning_backtest_start
    is_end = pipe.oos_tuning_backtest_end
    oos_start = pipe.oos_validation_backtest_start
    oos_end = pipe.oos_validation_backtest_end

    print(f"\n调参期 (IS):  {is_start} ~ {is_end}")
    print(f"验证期 (OOS): {oos_start} ~ {oos_end}")

    # ── Phase 1: 在调参期上跑所有版本 ──
    print("\n" + "=" * 60)
    print("  Phase 1: 调参期全版本扫描")
    print("=" * 60)

    is_results: dict[str, dict] = {}
    for ver, cfg in ALL_CONFIGS.items():
        print(f"\n  >>> {ver}: {cfg.description}")
        stats = _run_backtest(
            signal_df, cfg,
            backtest_start=is_start,
            backtest_end=is_end,
            capital=pipe.capital,
            lab_path=pipe.lab_path,
        )
        is_results[ver] = stats
        sharpe = stats.get("sharpe_ratio", 0)
        annual = stats.get("annual_return", 0)
        print(f"      Sharpe={sharpe:.2f}, 年化={annual:.1f}%")

    # ── Phase 2: 选出调参期最优版本 ──
    best_ver = max(is_results, key=lambda v: is_results[v].get(args.is_metric, 0))
    best_is_sharpe = is_results[best_ver].get("sharpe_ratio", 0)
    print(f"\n调参期最优: {best_ver} (Sharpe={best_is_sharpe:.2f})")

    # ── Phase 3: 仅跑最优版本的验证期 ──
    print("\n" + "=" * 60)
    print(f"  Phase 2: 验证期 OOS — 仅 {best_ver}")
    print("=" * 60)

    oos_stats = _run_backtest(
        signal_df, ALL_CONFIGS[best_ver],
        backtest_start=oos_start,
        backtest_end=oos_end,
        capital=pipe.capital,
        lab_path=pipe.lab_path,
    )

    # ── Phase 4: 输出对比 ──
    report_text = _print_comparison_table(is_results, (best_ver, oos_stats))

    output_dir = Path("hs300_topk/output/oos_validation")
    output_dir.mkdir(parents=True, exist_ok=True)

    report = {
        "weekly_label": args.weekly_label,
        "is_window": f"{is_start} ~ {is_end}",
        "oos_window": f"{oos_start} ~ {oos_end}",
        "is_metric": args.is_metric,
        "is_best_version": best_ver,
        "is_results": {v: {k: float(s.get(k, 0)) for k, _ in METRIC_KEYS} for v, s in is_results.items()},
        "oos_results": {best_ver: {k: float(oos_stats.get(k, 0)) for k, _ in METRIC_KEYS}},
    }
    report_path = output_dir / "oos_report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n报告已保存: {report_path}")

    (output_dir / "oos_report.txt").write_text(report_text, encoding="utf-8")


if __name__ == "__main__":
    main()
