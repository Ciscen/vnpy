"""
hs300_topk/run_pipeline.py

统一调度脚本 — 一键完成「数据下载 → 滚动训练 → 策略回测 → 报告生成」。

用法::

    # 完整流水线（跳过已有缓存）
    python -m hs300_topk.run_pipeline

    # 强制重新下载数据
    python -m hs300_topk.run_pipeline --force-download

    # 只运行回测（跳过下载和训练，使用上次信号缓存）
    python -m hs300_topk.run_pipeline --backtest-only

    # 纯样本外回测窗（2025-05-01 ~ 2026-04-30），报告输出到 v1.4_oos/
    python -m hs300_topk.run_pipeline --backtest-only --config v1.4 --oos-validate

    # 周频保守标签训练 + 回测（信号缓存在 hs300_topk_weekly_realistic.parquet）
    python -m hs300_topk.run_pipeline --weekly-label friday_close --config v1.4
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import polars as pl

from vnpy.trader.constant import Interval
from vnpy.alpha.strategy import BacktestingEngine

from hs300_topk.data.loader import get_lab, discover_symbols
from hs300_topk.data.downloader import phase_download
from hs300_topk.model.rolling_trainer import rolling_train
from hs300_topk.strategy.hs300_topk_strategy import HS300Top10Strategy
from hs300_topk.strategy.config import (
    StrategyConfig, BASELINE_V10, OPTIMIZED_V11, OPTIMIZED_V12,
    OPTIMIZED_V13, OPTIMIZED_V14, OPTIMIZED_V15,
)
from hs300_topk.backtest.evaluation import (
    print_metrics,
    show_charts,
    export_report,
)
from hs300_topk.pipeline_config import PIPELINE, PipelineConfig

# ══════════════════════════════════════════════════════════
# 默认报告根目录（单测时可由 main 拼接 version 子目录）
# ══════════════════════════════════════════════════════════
REPORT_DIR = Path("hs300_topk") / "output"


def phase_train(
    pipe: PipelineConfig,
    *,
    weekly_label: str = "high_touch",
) -> pl.DataFrame:
    """执行滚动训练，返回信号 DataFrame。结果会缓存到磁盘。"""
    cache_path = (
        pipe.signal_cache_weekly_realistic
        if weekly_label == "friday_close"
        else pipe.signal_cache
    )

    print("\n" + "=" * 60)
    print("  Phase 2: 周频滚动训练")
    print("=" * 60)

    signal_df, _ = rolling_train(
        lab_path=pipe.lab_path,
        data_start=pipe.data_start,
        data_end=pipe.data_end,
        backtest_start=pipe.backtest_start,
        backtest_end=pipe.backtest_end,
        weekly_label=weekly_label,
    )

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    signal_df.write_parquet(cache_path)
    print(f"\n  信号已缓存: {cache_path}")

    return signal_df


def phase_train_or_load(
    pipe: PipelineConfig,
    *,
    skip_train: bool = False,
    weekly_label: str = "high_touch",
) -> pl.DataFrame:
    """加载缓存信号或执行训练"""
    cache_path = (
        pipe.signal_cache_weekly_realistic
        if weekly_label == "friday_close"
        else pipe.signal_cache
    )

    if skip_train and cache_path.exists():
        print("\n" + "=" * 60)
        print("  Phase 2: 加载周频缓存信号 (跳过训练)")
        print("=" * 60)
        signal_df = pl.read_parquet(cache_path)
        print(f"  缓存文件: {cache_path}")
        print(f"  信号: {signal_df.shape[0]} 行, "
              f"{signal_df['datetime'].min()} ~ {signal_df['datetime'].max()}")
        return signal_df

    return phase_train(pipe, weekly_label=weekly_label)


# ══════════════════════════════════════════════════════════
# Phase 3: 策略回测
# ══════════════════════════════════════════════════════════

def phase_backtest(
    signal_df: pl.DataFrame,
    pipe: PipelineConfig,
    config: StrategyConfig | None = None,
    output_dir: Path | None = None,
) -> dict:
    """执行回测并返回统计指标。

    Parameters
    ----------
    config : StrategyConfig | None
        策略配置。None 则使用 BASELINE_V10。
    output_dir : Path | None
        报告输出目录。None 则使用 REPORT_DIR / config.version。
    """
    if config is None:
        from hs300_topk.strategy.config import BASELINE_V10
        config = BASELINE_V10

    print("\n" + "=" * 60)
    print(f"  Phase 3: 策略回测 [{config.version}] {config.description}")
    print("=" * 60)

    lab = get_lab(pipe.lab_path)
    vt_symbols = discover_symbols(pipe.lab_path)

    if config.use_market_filter and config.market_benchmark not in vt_symbols:
        vt_symbols = vt_symbols + [config.market_benchmark]
        lab.add_contract_setting(config.market_benchmark, 0, 0, 1, 0.01)

    engine = BacktestingEngine(lab)
    engine.set_parameters(
        vt_symbols=vt_symbols,
        interval=Interval.DAILY,
        start=datetime.fromisoformat(pipe.backtest_start),
        end=datetime.fromisoformat(pipe.backtest_end),
        capital=pipe.capital,
    )

    setting = {
        k: v for k, v in config.to_dict().items()
        if k not in ("version", "description") and not k.startswith("xgb_")
        and k != "train_years"
    }

    engine.add_strategy(HS300Top10Strategy, setting, signal_df)

    print("\n  加载历史数据 ...")
    engine.load_data()

    print("  运行回测 ...")
    engine.run_backtesting()

    print("  计算逐日盈亏 ...")
    engine.calculate_result()

    print("\n" + "=" * 60)
    print("  Phase 4: 绩效评估")
    print("=" * 60)

    stats = engine.calculate_statistics()
    print_metrics(stats)

    report_dir = output_dir or REPORT_DIR / config.version
    report_dir.mkdir(parents=True, exist_ok=True)
    export_report(engine, stats, report_dir, version_label=f"[{config.version}] {config.description}")

    config.to_json(report_dir / "config.json")
    print(f"  [报告] 策略配置 -> {report_dir / 'config.json'}")

    return stats


# ══════════════════════════════════════════════════════════
# 主入口
# ══════════════════════════════════════════════════════════

def _compare_results(results: list[tuple[str, dict]], output_dir: Path) -> None:
    """生成多版本对比报告"""
    output_dir.mkdir(parents=True, exist_ok=True)

    compare_keys = [
        ("total_return", "总收益率 (%)"),
        ("annual_return", "年化收益率 (%)"),
        ("max_ddpercent", "最大回撤 (%)"),
        ("sharpe_ratio", "Sharpe Ratio"),
        ("return_drawdown_ratio", "收益回撤比"),
        ("total_trade_count", "总交易笔数"),
        ("total_commission", "总手续费"),
        ("total_net_pnl", "总净盈亏"),
    ]

    print("\n" + "=" * 70)
    print("  版本对比")
    print("=" * 70)
    header = f"  {'指标':<16s}" + "".join(f"  {name:>14s}" for name, _ in results)
    print(header)
    print("-" * 70)

    compare_data = {}
    for key, label in compare_keys:
        row = f"  {label:<16s}"
        for name, stats in results:
            val = stats.get(key, 0)
            if "率" in label or "回撤" in label or "Ratio" in label or "比" in label:
                row += f"  {val:>14.2f}"
            else:
                row += f"  {val:>14,.0f}"
        print(row)
        compare_data[label] = {name: stats.get(key, 0) for name, stats in results}

    print("=" * 70)

    compare_path = output_dir / "comparison.json"

    def _default_serializer(obj):
        if hasattr(obj, "item"):
            return obj.item()
        raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")

    compare_path.write_text(
        json.dumps(compare_data, indent=2, ensure_ascii=False, default=_default_serializer),
        encoding="utf-8",
    )
    print(f"\n  [对比] 详细结果 -> {compare_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="HS300 Top-K 统一调度脚本")
    parser.add_argument("--force-download", action="store_true",
                        help="强制重新下载全部数据（忽略缓存）")
    parser.add_argument("--backtest-only", action="store_true",
                        help="仅回测（使用上次训练的信号缓存）")
    parser.add_argument("--skip-download", action="store_true",
                        help="跳过数据下载（使用已有 lab 数据）")
    parser.add_argument("--config", choices=["v1.0", "v1.1", "v1.2", "v1.3", "v1.4", "v1.5", "compare"], default="v1.3",
                        help="策略配置版本 (默认 v1.3，compare=同时运行所有版本)")
    parser.add_argument("--config-file", type=str, default=None,
                        help="自定义配置文件路径 (JSON)")
    parser.add_argument("--oos-validate", action="store_true",
                        help="仅回测样本外窗口 "
                             f"({PIPELINE.oos_validation_backtest_start} ~ "
                             f"{PIPELINE.oos_validation_backtest_end})，"
                             "报告子目录附加 _oos")
    parser.add_argument(
        "--weekly-label",
        choices=["high_touch", "friday_close"],
        default="high_touch",
        help="周频训练标签：high_touch=周内high触及+5%%；"
             "friday_close=保守对照，周内最后收盘 vs 周二开盘 +3%%",
    )
    parser.add_argument("--filter-hs300", action="store_true",
                        help="回测时信号只保留当前 HS300 成分股（模拟生产选股限制）")
    args = parser.parse_args()

    config_map = {
        "v1.0": BASELINE_V10, "v1.1": OPTIMIZED_V11,
        "v1.2": OPTIMIZED_V12, "v1.3": OPTIMIZED_V13,
        "v1.4": OPTIMIZED_V14, "v1.5": OPTIMIZED_V15,
    }

    if args.config_file:
        config = StrategyConfig.from_json(args.config_file)
    elif args.config != "compare":
        config = config_map[args.config]
    else:
        config = None

    pipe = PIPELINE.with_oos_validation_window() if args.oos_validate else PIPELINE

    print("=" * 60)
    print("  HS300 Top-K 选股策略 — 统一调度")
    print(f"  数据区间: {pipe.data_start} ~ {pipe.data_end}")
    print(f"  回测区间: {pipe.backtest_start} ~ {pipe.backtest_end}")
    if args.oos_validate:
        print(f"  [OOS] 调参参考窗: {pipe.oos_tuning_backtest_start} ~ "
              f"{pipe.oos_tuning_backtest_end}（本跑仅为验证期）")
    print(f"  初始资金: {pipe.capital:,.0f}")
    if config:
        print(f"  策略版本: [{config.version}] {config.description}")
    else:
        print(f"  策略版本: 对比模式")
    print("=" * 60)

    # Phase 1: 下载
    if not args.backtest_only and not args.skip_download:
        phase_download(force=args.force_download)
    else:
        vt_symbols = discover_symbols(pipe.lab_path)
        if not vt_symbols:
            print("错误: lab 目录中无数据，请先运行不带 --skip-download 的完整流水线")
            sys.exit(1)
        from hs300_topk.data.downloader import ensure_component_index
        lab = get_lab(pipe.lab_path)
        ensure_component_index(lab, vt_symbols)
        print(f"\n  使用已有数据: {len(vt_symbols)} 只股票")

    # Phase 2: 训练
    signal_df = phase_train_or_load(
        pipe,
        skip_train=args.backtest_only,
        weekly_label=args.weekly_label,
    )

    # 可选: 过滤到当前 HS300 成分股（用于对比测试）
    if args.filter_hs300:
        import akshare as ak
        from hs300_topk.data.downloader import symbol_to_exchange
        hs300_df = ak.index_stock_cons(symbol="000300")
        hs300_vt = {
            f"{code}.{symbol_to_exchange(code).value}"
            for code in hs300_df["品种代码"]
        }
        before = signal_df["vt_symbol"].n_unique()
        signal_df = signal_df.filter(pl.col("vt_symbol").is_in(hs300_vt))
        after = signal_df["vt_symbol"].n_unique()
        print(f"\n  [HS300 过滤] 信号股票: {before} → {after} 只")

    # Phase 3: 回测 + 报告
    dashboard_path: Path | None = None
    report_subdir_suffix = "_oos" if args.oos_validate else ""
    if args.weekly_label == "friday_close":
        report_subdir_suffix += "_lbl_friday_close"

    if args.config == "compare":
        results = []
        for ver, cfg in config_map.items():
            stats = phase_backtest(signal_df, pipe, config=cfg)
            results.append((ver, stats))
        _compare_results(results, REPORT_DIR)
    else:
        out_dir = REPORT_DIR / f"{config.version}{report_subdir_suffix}"
        stats = phase_backtest(signal_df, pipe, config=config, output_dir=out_dir)
        dashboard_path = out_dir / "dashboard.html"

    print("\n" + "=" * 60)
    print("  全部完成!")
    print(f"  报告输出: {REPORT_DIR.resolve()}")
    print("=" * 60)

    if dashboard_path and dashboard_path.exists():
        import webbrowser
        url = dashboard_path.resolve().as_uri()
        print(f"\n  正在打开仪表盘: {dashboard_path}")
        webbrowser.open(url)


if __name__ == "__main__":
    main()
