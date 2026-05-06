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
import shelve
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import polars as pl

from vnpy.trader.constant import Interval, Exchange
from vnpy.trader.object import BarData
from vnpy.alpha import AlphaLab
from vnpy.alpha.strategy import BacktestingEngine

from hs300_top10.data.loader import get_lab, discover_symbols
from hs300_top10.model.rolling_trainer import rolling_train
from hs300_top10.strategy.hs300_top10_strategy import HS300Top10Strategy
from hs300_top10.strategy.config import (
    StrategyConfig, BASELINE_V10, OPTIMIZED_V11, OPTIMIZED_V12,
)
from hs300_top10.backtest.evaluation import (
    print_metrics,
    show_charts,
    export_report,
)

# ══════════════════════════════════════════════════════════
# 全局配置
# ══════════════════════════════════════════════════════════
LAB_PATH = "./lab/hs300"

# 数据区间：最近 10 年
DATA_START = "2016-04-30"
DATA_END = "2026-04-30"

# 回测区间：最后 2 年
BACKTEST_START = "2024-05-01"
BACKTEST_END = "2026-04-30"

CAPITAL = 10_000_000

# 下载配置
AK_START = "20160430"
AK_END = "20260430"
LONG_RATE = 0.001       # 买入佣金 0.1%
SHORT_RATE = 0.002      # 卖出佣金+印花税 0.2%
SIZE = 1
PRICETICK = 0.01
MAX_RETRIES = 3
SLEEP_BETWEEN = 1.5

# 信号缓存路径
SIGNAL_CACHE = Path(LAB_PATH) / "signal" / "hs300_top10.parquet"

# 报告输出目录
REPORT_DIR = Path("hs300_top10") / "output"


# ══════════════════════════════════════════════════════════
# Phase 1: 数据下载（增量，自动检查缓存）
# ══════════════════════════════════════════════════════════

def _symbol_to_exchange(code: str) -> Exchange:
    if code.startswith(("6", "5")):
        return Exchange.SSE
    elif code.startswith(("0", "3")):
        return Exchange.SZSE
    return Exchange.SSE


def _download_one(ak_symbol: str, start: str, end: str):
    """带重试的单只股票下载"""
    import akshare as ak

    for attempt in range(MAX_RETRIES):
        try:
            df = ak.stock_zh_a_daily(
                symbol=ak_symbol,
                start_date=start,
                end_date=end,
                adjust="qfq",
            )
            return df
        except Exception as e:
            if attempt < MAX_RETRIES - 1:
                wait = (attempt + 1) * 3
                print(f"    重试 {attempt+1}/{MAX_RETRIES} (等待{wait}s): {e}", flush=True)
                time.sleep(wait)
            else:
                print(f"    全部重试失败: {e}", flush=True)
                return None


def _get_parquet_max_date(parquet_file: Path) -> datetime | None:
    """读取 parquet 文件中的最大日期"""
    try:
        df = pl.read_parquet(parquet_file)
        if df.is_empty():
            return None
        return df["datetime"].max().replace(tzinfo=None)  # type: ignore
    except Exception:
        return None


def _ak_rows_to_bars(
    df,
    symbol: str,
    exchange: Exchange,
) -> list[BarData]:
    """将 akshare DataFrame 转换为 BarData 列表"""
    bars: list[BarData] = []
    for _, row in df.iterrows():
        raw_dt = row["date"]
        if isinstance(raw_dt, str):
            dt = datetime.strptime(raw_dt, "%Y-%m-%d")
        elif hasattr(raw_dt, "to_pydatetime"):
            dt = raw_dt.to_pydatetime().replace(tzinfo=None)
        else:
            dt = datetime(raw_dt.year, raw_dt.month, raw_dt.day)

        bar = BarData(
            symbol=symbol,
            exchange=exchange,
            datetime=dt,
            interval=Interval.DAILY,
            open_price=float(row["open"]),
            high_price=float(row["high"]),
            low_price=float(row["low"]),
            close_price=float(row["close"]),
            volume=float(row["volume"]),
            turnover=float(row.get("amount", row.get("turnover", 0))),
            open_interest=0,
            gateway_name="AKShare",
        )
        bars.append(bar)
    return bars


def _download_benchmark(lab: AlphaLab) -> None:
    """下载沪深 300 指数日线数据作为基准"""
    import akshare as ak

    benchmark_file = Path(LAB_PATH) / "daily" / "000300.SSE.parquet"
    target_end = datetime.strptime(AK_END, "%Y%m%d")

    existing_max = _get_parquet_max_date(benchmark_file) if benchmark_file.exists() else None
    if existing_max and existing_max >= target_end - timedelta(days=5):
        print("  沪深 300 指数数据已存在, 跳过", flush=True)
        return

    start_str = AK_START
    if existing_max:
        start_str = (existing_max + timedelta(days=1)).strftime("%Y%m%d")

    print(f"  下载沪深 300 指数 ({start_str} ~ {AK_END}) ...", flush=True)

    try:
        df = ak.stock_zh_index_daily(symbol="sh000300")
        if df is None or df.empty:
            print("  [警告] 沪深 300 指数数据为空", flush=True)
            return

        df = df.rename(columns={"turnover": "amount"})
        start_dt = datetime.strptime(AK_START, "%Y%m%d")
        end_dt = datetime.strptime(AK_END, "%Y%m%d")
        df["date"] = pd.to_datetime(df["date"])
        df = df[(df["date"] >= start_dt) & (df["date"] <= end_dt)]

        if existing_max:
            df = df[df["date"] > existing_max]

        if df.empty:
            print("  沪深 300 指数无新增数据", flush=True)
            return

        bars = _ak_rows_to_bars(df, "000300", Exchange.SSE)
        lab.save_bar_data(bars)
        lab.add_contract_setting("000300.SSE", 0, 0, 1, 0.01)
        print(f"  沪深 300 指数: {len(bars)} 根日线", flush=True)
    except Exception as e:
        print(f"  [警告] 沪深 300 指数下载失败: {e}", flush=True)


def phase_download(force: bool = False) -> list[str]:
    """下载沪深 300 成分股日线数据到 AlphaLab。

    智能增量更新：检查现有文件的最大日期，仅下载缺失的部分。

    Returns
    -------
    vt_symbols : list[str]
        所有可用股票代码
    """
    import akshare as ak

    print("\n" + "=" * 60)
    print("  Phase 1: 数据下载")
    print("=" * 60)

    lab = AlphaLab(LAB_PATH)
    daily_path = Path(LAB_PATH) / "daily"
    target_end = datetime.strptime(AK_END, "%Y%m%d")

    print("获取沪深 300 成分股列表 ...", flush=True)
    cons_df = ak.index_stock_cons(symbol="000300")
    symbols = list(cons_df["品种代码"])
    print(f"  成分股数量: {len(symbols)}", flush=True)

    total = len(symbols)
    success = 0
    skipped = 0
    incremental = 0
    failed = []

    for i, symbol in enumerate(symbols, 1):
        exchange = _symbol_to_exchange(symbol)
        vt_symbol = f"{symbol}.{exchange.value}"
        parquet_file = daily_path / f"{vt_symbol}.parquet"

        if parquet_file.exists() and not force:
            existing_max = _get_parquet_max_date(parquet_file)

            if existing_max and existing_max >= target_end - timedelta(days=5):
                skipped += 1
                success += 1
                if skipped <= 3 or skipped % 50 == 0:
                    print(f"  [{i}/{total}] {vt_symbol} 数据完整, 跳过 (累计跳过 {skipped})", flush=True)
                continue

            inc_start = (existing_max + timedelta(days=1)).strftime("%Y%m%d") if existing_max else AK_START
            ak_symbol = f"sh{symbol}" if symbol.startswith(("6", "5")) else f"sz{symbol}"
            df = _download_one(ak_symbol, inc_start, AK_END)

            if df is not None and not df.empty:
                bars = _ak_rows_to_bars(df, symbol, exchange)
                lab.save_bar_data(bars)
                lab.add_contract_setting(vt_symbol, LONG_RATE, SHORT_RATE, SIZE, PRICETICK)
                incremental += 1
                success += 1
                print(f"  [{i}/{total}] {vt_symbol}: 增量 {len(bars)} 根 "
                      f"({inc_start}~{AK_END})", flush=True)
                time.sleep(SLEEP_BETWEEN)
            else:
                success += 1
                skipped += 1
                if skipped <= 3 or skipped % 50 == 0:
                    print(f"  [{i}/{total}] {vt_symbol} 无新增数据, 跳过", flush=True)
            continue

        ak_symbol = f"sh{symbol}" if symbol.startswith(("6", "5")) else f"sz{symbol}"
        df = _download_one(ak_symbol, AK_START, AK_END)

        if df is None or df.empty:
            print(f"  [{i}/{total}] {vt_symbol} 无数据", flush=True)
            failed.append(vt_symbol)
            continue

        bars = _ak_rows_to_bars(df, symbol, exchange)
        lab.save_bar_data(bars)
        lab.add_contract_setting(vt_symbol, LONG_RATE, SHORT_RATE, SIZE, PRICETICK)

        success += 1
        print(f"  [{i}/{total}] {vt_symbol}: {len(bars)} 根日线", flush=True)
        time.sleep(SLEEP_BETWEEN)

    print(f"\n  下载完成: 成功 {success} (跳过 {skipped}, 增量更新 {incremental}), 失败 {len(failed)}")
    if failed:
        print(f"  失败列表: {failed[:20]}{'...' if len(failed) > 20 else ''}")

    _download_benchmark(lab)

    _ensure_component_index(lab, discover_symbols(LAB_PATH))

    return discover_symbols(LAB_PATH)


def _ensure_component_index(lab: AlphaLab, vt_symbols: list[str]) -> None:
    """确保 shelve 成分股索引覆盖整个数据区间。

    检查逻辑：如果 shelve 中最大日期 < 目标结束日期，则重建索引。
    """
    index_symbol = "HS300.SSE"
    db_path = str(lab.component_path.joinpath(index_symbol))

    target_end_str = "2026-12-31"
    needs_rebuild = False

    try:
        with shelve.open(db_path) as db:
            keys = list(db.keys())
            if not keys:
                needs_rebuild = True
            else:
                max_key = max(keys)
                if max_key < target_end_str:
                    print(f"  成分股索引仅覆盖到 {max_key}，需要扩展", flush=True)
                    needs_rebuild = True
    except Exception:
        needs_rebuild = True

    if not needs_rebuild:
        return

    print("  写入成分股索引 ...", flush=True)
    start_dt = datetime(2016, 1, 1)
    end_dt = datetime(2026, 12, 31)
    with shelve.open(db_path, flag="n") as db:
        cur = start_dt
        while cur <= end_dt:
            db[cur.strftime("%Y-%m-%d")] = vt_symbols
            cur += timedelta(days=1)
    print(f"  成分股索引: {index_symbol} -> {len(vt_symbols)} 只, "
          f"覆盖 {start_dt.date()} ~ {end_dt.date()}", flush=True)


# ══════════════════════════════════════════════════════════
# Phase 2: 滚动训练
# ══════════════════════════════════════════════════════════

def phase_train() -> pl.DataFrame:
    """执行滚动训练，返回信号 DataFrame。结果会缓存到磁盘。"""
    print("\n" + "=" * 60)
    print("  Phase 2: 滚动训练")
    print("=" * 60)

    signal_df, _ = rolling_train(
        lab_path=LAB_PATH,
        data_start=DATA_START,
        data_end=DATA_END,
        backtest_start=BACKTEST_START,
        backtest_end=BACKTEST_END,
    )

    # 缓存信号到磁盘
    SIGNAL_CACHE.parent.mkdir(parents=True, exist_ok=True)
    signal_df.write_parquet(SIGNAL_CACHE)
    print(f"\n  信号已缓存: {SIGNAL_CACHE}")

    return signal_df


def phase_train_or_load(skip_train: bool = False) -> pl.DataFrame:
    """加载缓存信号或执行训练"""
    if skip_train and SIGNAL_CACHE.exists():
        print("\n" + "=" * 60)
        print("  Phase 2: 加载缓存信号 (跳过训练)")
        print("=" * 60)
        signal_df = pl.read_parquet(SIGNAL_CACHE)
        print(f"  信号: {signal_df.shape[0]} 行, "
              f"{signal_df['datetime'].min()} ~ {signal_df['datetime'].max()}")
        return signal_df

    return phase_train()


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
    export_report(engine, stats, report_dir)

    config.to_json(report_dir / "config.json")
    print(f"  [报告] 策略配置 -> {report_dir / 'config.json'}")

    try:
        show_charts(engine, benchmark_symbol="000300.SSE")
    except Exception as e:
        print(f"\n  [提示] 图表展示跳过: {e}")

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
    parser.add_argument("--config", choices=["v1.0", "v1.1", "v1.2", "compare"], default="v1.2",
                        help="策略配置版本 (默认 v1.2，compare=同时运行所有版本)")
    parser.add_argument("--config-file", type=str, default=None,
                        help="自定义配置文件路径 (JSON)")
    args = parser.parse_args()

    config_map = {"v1.0": BASELINE_V10, "v1.1": OPTIMIZED_V11, "v1.2": OPTIMIZED_V12}

    if args.config_file:
        config = StrategyConfig.from_json(args.config_file)
    elif args.config != "compare":
        config = config_map[args.config]
    else:
        config = None

    print("=" * 60)
    print("  HS300 Top-10 周度选股策略 — 统一调度")
    print(f"  数据区间: {DATA_START} ~ {DATA_END}")
    print(f"  回测区间: {BACKTEST_START} ~ {BACKTEST_END}")
    print(f"  初始资金: {CAPITAL:,.0f}")
    if config:
        print(f"  策略版本: [{config.version}] {config.description}")
    else:
        print(f"  策略版本: 对比模式 (v1.0 vs v1.1)")
    print("=" * 60)

    # Phase 1: 下载
    if not args.backtest_only and not args.skip_download:
        phase_download(force=args.force_download)
    else:
        vt_symbols = discover_symbols(LAB_PATH)
        if not vt_symbols:
            print("错误: lab 目录中无数据，请先运行不带 --skip-download 的完整流水线")
            sys.exit(1)
        lab = get_lab(LAB_PATH)
        _ensure_component_index(lab, vt_symbols)
        print(f"\n  使用已有数据: {len(vt_symbols)} 只股票")

    # Phase 2: 训练
    signal_df = phase_train_or_load(skip_train=args.backtest_only)

    # Phase 3: 回测 + 报告
    if args.config == "compare":
        results = []
        for ver, cfg in config_map.items():
            stats = phase_backtest(signal_df, config=cfg)
            results.append((ver, stats))
        _compare_results(results, REPORT_DIR)
    else:
        stats = phase_backtest(signal_df, config=config)

    print("\n" + "=" * 60)
    print("  全部完成!")
    print(f"  报告输出: {REPORT_DIR.resolve()}")
    print("=" * 60)


if __name__ == "__main__":
    main()
