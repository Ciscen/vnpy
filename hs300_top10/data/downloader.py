"""
hs300_top10/data/downloader.py

沪深 300 成分股日线数据下载模块（增量更新 + 基准指数）。

可独立使用::

    python -m hs300_top10.data.downloader              # 增量下载
    python -m hs300_top10.data.downloader --force       # 强制全量

也被 run_pipeline.py Phase 1 调用。
"""
from __future__ import annotations

import shelve
import time
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import polars as pl

from vnpy.trader.constant import Interval, Exchange
from vnpy.trader.object import BarData
from vnpy.alpha import AlphaLab

from hs300_top10.data.loader import discover_symbols
from hs300_top10.pipeline_config import PIPELINE

# ──────────────────────────────────────────────────
# 下载参数
# ──────────────────────────────────────────────────
LONG_RATE = 0.001
SHORT_RATE = 0.002
SIZE = 1
PRICETICK = 0.01
MAX_RETRIES = 3
SLEEP_BETWEEN = 1.5


# ──────────────────────────────────────────────────
# 工具函数
# ──────────────────────────────────────────────────

def symbol_to_exchange(code: str) -> Exchange:
    """根据股票代码前缀判断交易所。"""
    if code.startswith(("6", "5")):
        return Exchange.SSE
    if code.startswith(("0", "3")):
        return Exchange.SZSE
    return Exchange.SSE


def download_one(ak_symbol: str, start: str, end: str):
    """带重试的单只股票下载（akshare 前复权日线）。"""
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


def get_parquet_max_date(parquet_file: Path) -> datetime | None:
    """读取 parquet 文件中的最大日期。"""
    try:
        df = pl.read_parquet(parquet_file)
        if df.is_empty():
            return None
        return df["datetime"].max().replace(tzinfo=None)  # type: ignore
    except Exception:
        return None


def ak_rows_to_bars(
    df,
    symbol: str,
    exchange: Exchange,
) -> list[BarData]:
    """将 akshare DataFrame 转换为 BarData 列表。"""
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


# ──────────────────────────────────────────────────
# 基准指数下载
# ──────────────────────────────────────────────────

def download_benchmark(
    lab: AlphaLab,
    lab_path: str,
    ak_start: str,
    ak_end: str,
) -> None:
    """下载沪深 300 指数日线数据作为基准。"""
    import akshare as ak

    benchmark_file = Path(lab_path) / "daily" / "000300.SSE.parquet"
    target_end = datetime.strptime(ak_end, "%Y%m%d")

    existing_max = get_parquet_max_date(benchmark_file) if benchmark_file.exists() else None
    if existing_max and existing_max >= target_end - timedelta(days=5):
        print("  沪深 300 指数数据已存在, 跳过", flush=True)
        return

    start_str = ak_start
    if existing_max:
        start_str = (existing_max + timedelta(days=1)).strftime("%Y%m%d")

    print(f"  下载沪深 300 指数 ({start_str} ~ {ak_end}) ...", flush=True)

    try:
        df = ak.stock_zh_index_daily(symbol="sh000300")
        if df is None or df.empty:
            print("  [警告] 沪深 300 指数数据为空", flush=True)
            return

        df = df.rename(columns={"turnover": "amount"})
        start_dt = datetime.strptime(ak_start, "%Y%m%d")
        end_dt = datetime.strptime(ak_end, "%Y%m%d")
        df["date"] = pd.to_datetime(df["date"])
        df = df[(df["date"] >= start_dt) & (df["date"] <= end_dt)]

        if existing_max:
            df = df[df["date"] > existing_max]

        if df.empty:
            print("  沪深 300 指数无新增数据", flush=True)
            return

        bars = ak_rows_to_bars(df, "000300", Exchange.SSE)
        lab.save_bar_data(bars)
        lab.add_contract_setting("000300.SSE", 0, 0, 1, 0.01)
        print(f"  沪深 300 指数: {len(bars)} 根日线", flush=True)
    except Exception as e:
        print(f"  [警告] 沪深 300 指数下载失败: {e}", flush=True)


# ──────────────────────────────────────────────────
# 成分股索引
# ──────────────────────────────────────────────────

def ensure_component_index(lab: AlphaLab, vt_symbols: list[str]) -> None:
    """确保 shelve 成分股索引覆盖整个数据区间。"""
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


# ══════════════════════════════════════════════════
# 主函数
# ══════════════════════════════════════════════════

def phase_download(
    force: bool = False,
    lab_path: str = PIPELINE.lab_path,
    data_start: str = PIPELINE.data_start,
    data_end: str = PIPELINE.data_end,
) -> list[str]:
    """下载沪深 300 成分股日线数据到 AlphaLab（增量更新）。

    Returns
    -------
    vt_symbols : list[str]
        所有可用股票代码
    """
    import akshare as ak

    ak_start = data_start.replace("-", "")
    ak_end = data_end.replace("-", "")

    print("\n" + "=" * 60)
    print("  Phase 1: 数据下载")
    print("=" * 60)

    lab = AlphaLab(lab_path)
    daily_path = Path(lab_path) / "daily"
    target_end = datetime.strptime(ak_end, "%Y%m%d")

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
        exchange = symbol_to_exchange(symbol)
        vt_symbol = f"{symbol}.{exchange.value}"
        parquet_file = daily_path / f"{vt_symbol}.parquet"

        if parquet_file.exists() and not force:
            existing_max = get_parquet_max_date(parquet_file)

            if existing_max and existing_max >= target_end - timedelta(days=5):
                skipped += 1
                success += 1
                if skipped <= 3 or skipped % 50 == 0:
                    print(f"  [{i}/{total}] {vt_symbol} 数据完整, 跳过 (累计跳过 {skipped})", flush=True)
                continue

            inc_start = (existing_max + timedelta(days=1)).strftime("%Y%m%d") if existing_max else ak_start
            ak_symbol = f"sh{symbol}" if symbol.startswith(("6", "5")) else f"sz{symbol}"
            df = download_one(ak_symbol, inc_start, ak_end)

            if df is not None and not df.empty:
                bars = ak_rows_to_bars(df, symbol, exchange)
                lab.save_bar_data(bars)
                lab.add_contract_setting(vt_symbol, LONG_RATE, SHORT_RATE, SIZE, PRICETICK)
                incremental += 1
                success += 1
                print(f"  [{i}/{total}] {vt_symbol}: 增量 {len(bars)} 根 "
                      f"({inc_start}~{ak_end})", flush=True)
                time.sleep(SLEEP_BETWEEN)
            else:
                success += 1
                skipped += 1
                if skipped <= 3 or skipped % 50 == 0:
                    print(f"  [{i}/{total}] {vt_symbol} 无新增数据, 跳过", flush=True)
            continue

        ak_symbol = f"sh{symbol}" if symbol.startswith(("6", "5")) else f"sz{symbol}"
        df = download_one(ak_symbol, ak_start, ak_end)

        if df is None or df.empty:
            print(f"  [{i}/{total}] {vt_symbol} 无数据", flush=True)
            failed.append(vt_symbol)
            continue

        bars = ak_rows_to_bars(df, symbol, exchange)
        lab.save_bar_data(bars)
        lab.add_contract_setting(vt_symbol, LONG_RATE, SHORT_RATE, SIZE, PRICETICK)

        success += 1
        print(f"  [{i}/{total}] {vt_symbol}: {len(bars)} 根日线", flush=True)
        time.sleep(SLEEP_BETWEEN)

    print(f"\n  下载完成: 成功 {success} (跳过 {skipped}, 增量更新 {incremental}), 失败 {len(failed)}")
    if failed:
        print(f"  失败列表: {failed[:20]}{'...' if len(failed) > 20 else ''}")

    download_benchmark(lab, lab_path, ak_start, ak_end)

    ensure_component_index(lab, discover_symbols(lab_path))

    return discover_symbols(lab_path)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="HS300 数据下载")
    parser.add_argument("--force", action="store_true", help="强制全量重新下载")
    args = parser.parse_args()

    symbols = phase_download(force=args.force)
    print(f"\n完成，共 {len(symbols)} 只股票可用")
