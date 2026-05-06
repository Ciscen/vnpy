"""
hs300_top10/run_pipeline.py

统一调度脚本 — 一键完成「数据下载 → 滚动训练 → 策略回测 → 报告生成」。

用法::

    # 完整流水线（跳过已有缓存）
    python -m hs300_top10.run_pipeline

    # 强制重新下载数据
    python -m hs300_top10.run_pipeline --force-download

    # 只运行回测（跳过下载和训练，使用上次信号缓存）
    python -m hs300_top10.run_pipeline --backtest-only
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

from hs300_top10.data.loader import get_lab, discover_symbols
from hs300_top10.data.downloader import phase_download
from hs300_top10.model.rolling_trainer import rolling_train, rolling_train_daily
from hs300_top10.strategy.hs300_top10_strategy import HS300Top10Strategy
from hs300_top10.strategy.config import (
    StrategyConfig, BASELINE_V10, OPTIMIZED_V11, OPTIMIZED_V12,
    OPTIMIZED_V13,
    OPTIMIZED_V20,
    OPTIMIZED_V21,
    OPTIMIZED_V22,
)
from hs300_top10.backtest.evaluation import (
    print_metrics,
    show_charts,
    export_report,
)
from hs300_top10.pipeline_config import PIPELINE

# ══════════════════════════════════════════════════════════
# 全局配置 — 来自 pipeline_config 统一管理
# ══════════════════════════════════════════════════════════
LAB_PATH = PIPELINE.lab_path
DATA_START = PIPELINE.data_start
DATA_END = PIPELINE.data_end
BACKTEST_START = PIPELINE.backtest_start
BACKTEST_END = PIPELINE.backtest_end
CAPITAL = PIPELINE.capital
SIGNAL_CACHE = PIPELINE.signal_cache
SIGNAL_CACHE_DAILY = PIPELINE.signal_cache_daily

# 报告输出目录
REPORT_DIR = Path("hs300_top10") / "output"


# ══════════════════════════════════════════════════════════
# Phase 2: 滚动训练
# ══════════════════════════════════════════════════════════

def phase_train(daily: bool = False) -> pl.DataFrame:
    """执行滚动训练，返回信号 DataFrame。结果会缓存到磁盘。"""
    mode_label = "日频" if daily else "周频"
    cache_path = SIGNAL_CACHE_DAILY if daily else SIGNAL_CACHE

    print("\n" + "=" * 60)
    print(f"  Phase 2: {mode_label}滚动训练")
    print("=" * 60)

    if daily:
        signal_df, _ = rolling_train_daily(
            lab_path=LAB_PATH,
            data_start=DATA_START,
            data_end=DATA_END,
            backtest_start=BACKTEST_START,
            backtest_end=BACKTEST_END,
        )
    else:
        signal_df, _ = rolling_train(
            lab_path=LAB_PATH,
            data_start=DATA_START,
            data_end=DATA_END,
            backtest_start=BACKTEST_START,
            backtest_end=BACKTEST_END,
        )

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    signal_df.write_parquet(cache_path)
    print(f"\n  信号已缓存: {cache_path}")

    return signal_df


def phase_train_or_load(skip_train: bool = False, daily: bool = False) -> pl.DataFrame:
    """加载缓存信号或执行训练"""
    cache_path = SIGNAL_CACHE_DAILY if daily else SIGNAL_CACHE

    if skip_train and cache_path.exists():
        mode_label = "日频" if daily else "周频"
        print("\n" + "=" * 60)
        print(f"  Phase 2: 加载{mode_label}缓存信号 (跳过训练)")
        print("=" * 60)
        signal_df = pl.read_parquet(cache_path)
        print(f"  信号: {signal_df.shape[0]} 行, "
              f"{signal_df['datetime'].min()} ~ {signal_df['datetime'].max()}")
        return signal_df

    return phase_train(daily=daily)


# ══════════════════════════════════════════════════════════
# Phase 3: 策略回测
# ══════════════════════════════════════════════════════════

def phase_backtest(
    signal_df: pl.DataFrame,
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
        from hs300_top10.strategy.config import BASELINE_V10
        config = BASELINE_V10

    print("\n" + "=" * 60)
    print(f"  Phase 3: 策略回测 [{config.version}] {config.description}")
    print("=" * 60)

    lab = get_lab(LAB_PATH)
    vt_symbols = discover_symbols(LAB_PATH)

    if config.use_market_filter and config.market_benchmark not in vt_symbols:
        vt_symbols = vt_symbols + [config.market_benchmark]
        lab.add_contract_setting(config.market_benchmark, 0, 0, 1, 0.01)

    engine = BacktestingEngine(lab)
    engine.set_parameters(
        vt_symbols=vt_symbols,
        interval=Interval.DAILY,
        start=datetime.fromisoformat(BACKTEST_START),
        end=datetime.fromisoformat(BACKTEST_END),
        capital=CAPITAL,
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
    parser = argparse.ArgumentParser(description="HS300 Top-10 统一调度脚本")
    parser.add_argument("--force-download", action="store_true",
                        help="强制重新下载全部数据（忽略缓存）")
    parser.add_argument("--backtest-only", action="store_true",
                        help="仅回测（使用上次训练的信号缓存）")
    parser.add_argument("--skip-download", action="store_true",
                        help="跳过数据下载（使用已有 lab 数据）")
    parser.add_argument("--config", choices=["v1.0", "v1.1", "v1.2", "v1.3", "v2.0", "v2.1", "v2.2", "compare"], default="v1.3",
                        help="策略配置版本 (默认 v1.3，compare=同时运行所有版本)")
    parser.add_argument("--config-file", type=str, default=None,
                        help="自定义配置文件路径 (JSON)")
    args = parser.parse_args()

    config_map = {
        "v1.0": BASELINE_V10, "v1.1": OPTIMIZED_V11,
        "v1.2": OPTIMIZED_V12, "v1.3": OPTIMIZED_V13,
        "v2.0": OPTIMIZED_V20,
        "v2.1": OPTIMIZED_V21,
        "v2.2": OPTIMIZED_V22,
    }

    if args.config_file:
        config = StrategyConfig.from_json(args.config_file)
    elif args.config != "compare":
        config = config_map[args.config]
    else:
        config = None

    print("=" * 60)
    print("  HS300 Top-10 选股策略 — 统一调度")
    print(f"  数据区间: {DATA_START} ~ {DATA_END}")
    print(f"  回测区间: {BACKTEST_START} ~ {BACKTEST_END}")
    print(f"  初始资金: {CAPITAL:,.0f}")
    if config:
        print(f"  策略版本: [{config.version}] {config.description}")
    else:
        print(f"  策略版本: 对比模式")
    print("=" * 60)

    # Phase 1: 下载
    if not args.backtest_only and not args.skip_download:
        phase_download(force=args.force_download)
    else:
        vt_symbols = discover_symbols(LAB_PATH)
        if not vt_symbols:
            print("错误: lab 目录中无数据，请先运行不带 --skip-download 的完整流水线")
            sys.exit(1)
        from hs300_top10.data.downloader import ensure_component_index
        lab = get_lab(LAB_PATH)
        ensure_component_index(lab, vt_symbols)
        print(f"\n  使用已有数据: {len(vt_symbols)} 只股票")

    # Phase 2: 训练
    use_daily = config.daily_signal if config else False
    signal_df = phase_train_or_load(skip_train=args.backtest_only, daily=use_daily)

    # Phase 3: 回测 + 报告
    dashboard_path: Path | None = None
    if args.config == "compare":
        results = []
        for ver, cfg in config_map.items():
            if cfg.daily_signal:
                ver_signal = phase_train_or_load(
                    skip_train=args.backtest_only, daily=True
                )
            else:
                ver_signal = signal_df
            stats = phase_backtest(ver_signal, config=cfg)
            results.append((ver, stats))
        _compare_results(results, REPORT_DIR)
    else:
        stats = phase_backtest(signal_df, config=config)
        dashboard_path = REPORT_DIR / config.version / "dashboard.html"

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
