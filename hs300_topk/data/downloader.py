"""
hs300_topk/data/downloader.py

日线数据下载模块 — 确保训练宇宙的完整性。

下载范围:
  1. 当前 CSI800 (HS300+CSI500) 成分股（AKShare）
  2. 历史 HS300 成分但不在当前 CSI800 的股票（BaoStock 快照补充）
  3. 沪深 300 指数基准数据

这样可以确保 PIT（Point-in-Time）成分过滤时，所有历史成分都有数据可用，
从根本上消除因数据缺失导致的幸存者偏差。

可独立使用::

    python -m hs300_topk.data.downloader              # 增量下载
    python -m hs300_topk.data.downloader --force       # 强制全量

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

from hs300_topk.data.loader import discover_symbols
from hs300_topk.pipeline_config import PIPELINE

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
    """带重试的单只股票下载（首选 AKShare，失败后回退至 BaoStock 前复权日线）。"""
    import akshare as ak
    import baostock as bs

    for attempt in range(2):
        try:
            df = ak.stock_zh_a_daily(
                symbol=ak_symbol,
                start_date=start,
                end_date=end,
                adjust="qfq",
            )
            return df
        except Exception as e:
            if attempt < 1:
                wait = (attempt + 1) * 2
                time.sleep(wait)
            else:
                pass # 准备切 baostock

    # AKShare 失败，尝试 BaoStock 作为兜底
    bs_symbol = f"{ak_symbol[:2]}.{ak_symbol[2:]}"
    bs_start = f"{start[:4]}-{start[4:6]}-{start[6:]}" if len(start) == 8 else start
    bs_end = f"{end[:4]}-{end[4:6]}-{end[6:]}" if len(end) == 8 else end

    try:
        bs.login()
        rs = bs.query_history_k_data_plus(
            bs_symbol,
            "date,open,high,low,close,volume,amount",
            start_date=bs_start,
            end_date=bs_end,
            frequency="d",
            adjustflag="2"
        )
        if rs.error_code == '0' and len(rs.data) > 0:
            df = rs.get_data()
            # 确保列的数据类型正确
            for col in ["open", "high", "low", "close", "volume", "amount"]:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            bs.logout()
            return df
        bs.logout()
    except Exception as e:
        print(f"    BaoStock 兜底下载失败: {e}", flush=True)
        try:
            bs.logout()
        except:
            pass

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

def _bs_code_to_vt(bs_code: str) -> str:
    """将 BaoStock 代码 (sh.600000) 转为 vt_symbol (600000.SSE)。"""
    prefix, num = bs_code.split(".")
    exchange = "SSE" if prefix == "sh" else "SZSE"
    return f"{num}.{exchange}"


def _fetch_hs300_snapshots(
    start_year: int = 2016,
    end_year: int = 2026,
) -> dict[str, list[str]]:
    """通过 BaoStock 获取每半年的 HS300 成分股快照。

    Returns
    -------
    dict[date_str, list[vt_symbol]]
        每个调整节点对应的成分股列表。
    """
    import baostock as bs

    lg = bs.login()
    if lg.error_code != "0":
        raise RuntimeError(f"BaoStock login failed: {lg.error_msg}")

    snapshots: dict[str, list[str]] = {}
    query_dates = []
    for y in range(start_year, end_year + 1):
        query_dates.append(f"{y}-01-15")
        query_dates.append(f"{y}-07-15")

    for d in query_dates:
        rs = bs.query_hs300_stocks(date=d)
        codes: list[str] = []
        while rs.error_code == "0" and rs.next():
            row = rs.get_row_data()
            codes.append(_bs_code_to_vt(row[1]))
        if codes:
            snapshots[d] = sorted(codes)

    bs.logout()
    return snapshots


def _download_historical_hs300_members(
    lab: AlphaLab,
    daily_path: Path,
    ak_start: str,
    ak_end: str,
    target_end: datetime,
) -> None:
    """补充下载历史 HS300 成分但不在当前 CSI800 中的股票。

    用 BaoStock 历史快照获取所有曾在 HS300 中的股票，检查 lab 中是否已有数据，
    缺失的则用 AKShare 下载。这确保 PIT 过滤时不会因为数据缺失而丢掉样本。
    """
    print("\n  补充下载历史 HS300 成分 ...", flush=True)
    try:
        snapshots = _fetch_hs300_snapshots()
    except Exception as e:
        print(f"  ⚠ 获取历史快照失败 ({e})，跳过补充下载", flush=True)
        return

    all_historical: set[str] = set()
    for members in snapshots.values():
        all_historical.update(members)

    existing = set(f.stem for f in daily_path.glob("*.parquet"))
    missing = sorted(all_historical - existing)

    if not missing:
        print(f"  历史成分 {len(all_historical)} 只全部已覆盖", flush=True)
        return

    print(f"  历史成分累计 {len(all_historical)} 只, 已有 {len(existing & all_historical)} 只, "
          f"缺失 {len(missing)} 只", flush=True)

    downloaded = 0
    failed = 0
    for i, vt_symbol in enumerate(missing, 1):
        code = vt_symbol.split(".")[0]
        ak_symbol = f"sh{code}" if code.startswith(("6", "5")) else f"sz{code}"

        df = download_one(ak_symbol, ak_start, ak_end)
        if df is None or df.empty:
            failed += 1
            if failed <= 5 or failed % 20 == 0:
                print(f"    [{i}/{len(missing)}] {vt_symbol} 无数据", flush=True)
            continue

        exchange = symbol_to_exchange(code)
        bars = ak_rows_to_bars(df, code, exchange)
        lab.save_bar_data(bars)
        lab.add_contract_setting(vt_symbol, LONG_RATE, SHORT_RATE, SIZE, PRICETICK)

        downloaded += 1
        if downloaded <= 5 or downloaded % 20 == 0:
            print(f"    [{i}/{len(missing)}] {vt_symbol}: {len(bars)} 根日线", flush=True)
        time.sleep(SLEEP_BETWEEN)

    print(f"  补充完成: 新增 {downloaded} 只, 无数据 {failed} 只", flush=True)


def ensure_component_index(lab: AlphaLab, vt_symbols: list[str]) -> None:
    """使用 BaoStock 历史数据构建逐日成分股索引，消除幸存者偏差。

    首先尝试获取 BaoStock 历史数据，失败时降级为静态列表。
    """
    index_symbol = "HS300.SSE"
    db_path = str(lab.component_path.joinpath(index_symbol))
    marker_file = lab.component_path.joinpath(f"{index_symbol}.pit_version")

    PIT_VERSION = "baostock_v1"
    target_end_str = "2026-12-31"
    needs_rebuild = False

    try:
        current_ver = marker_file.read_text().strip() if marker_file.exists() else ""
        if current_ver != PIT_VERSION:
            print("  成分股索引需升级为 point-in-time 版本", flush=True)
            needs_rebuild = True
        else:
            with shelve.open(db_path) as db:
                keys = list(db.keys())
                if not keys:
                    needs_rebuild = True
                else:
                    max_key = max(keys)
                    if max_key < target_end_str:
                        needs_rebuild = True
    except Exception:
        needs_rebuild = True

    if not needs_rebuild:
        return

    print("  获取 BaoStock 历史成分股快照 ...", flush=True)
    try:
        snapshots = _fetch_hs300_snapshots()
    except Exception as e:
        print(f"  ⚠ BaoStock 获取失败 ({e})，降级为静态成分股列表", flush=True)
        start_dt = datetime(2016, 1, 1)
        end_dt = datetime(2026, 12, 31)
        with shelve.open(db_path, flag="n") as db:
            cur = start_dt
            while cur <= end_dt:
                db[cur.strftime("%Y-%m-%d")] = vt_symbols
                cur += timedelta(days=1)
        print(f"  成分股索引 (静态): {index_symbol} -> {len(vt_symbols)} 只", flush=True)
        return

    # 按时间排序快照节点
    sorted_dates = sorted(snapshots.keys())
    print(f"  获取到 {len(sorted_dates)} 个调整节点", flush=True)

    # 逐日写入：每天使用最近一次快照的成分股
    print("  写入逐日成分股索引 ...", flush=True)
    start_dt = datetime(2016, 1, 1)
    end_dt = datetime(2026, 12, 31)

    with shelve.open(db_path, flag="n") as db:
        snap_idx = 0
        cur = start_dt
        while cur <= end_dt:
            cur_str = cur.strftime("%Y-%m-%d")
            while (snap_idx + 1 < len(sorted_dates)
                   and sorted_dates[snap_idx + 1] <= cur_str):
                snap_idx += 1
            db[cur_str] = snapshots[sorted_dates[snap_idx]]
            cur += timedelta(days=1)
    marker_file.write_text(PIT_VERSION)

    all_ever = set()
    for codes in snapshots.values():
        all_ever.update(codes)
    print(f"  成分股索引 (point-in-time): {len(sorted_dates)} 个快照, "
          f"累计 {len(all_ever)} 只不同股票, "
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
    """下载 CSI800 (HS300+CSI500) 成分股日线数据到 AlphaLab（增量更新）。

    使用 CSI800 作为选股宇宙以减少幸存者偏差：训练时包含更广的股票池，
    预测时仍可限定在 HS300 内选股。

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

    print("获取 CSI800 选股宇宙 (HS300 + CSI500) ...", flush=True)
    hs300_df = ak.index_stock_cons(symbol="000300")
    hs300_codes = set(hs300_df["品种代码"])
    try:
        csi500_df = ak.index_stock_cons(symbol="000905")
        csi500_codes = set(csi500_df["品种代码"])
    except Exception as e:
        print(f"  ⚠ CSI500 获取失败 ({e})，仅使用 HS300", flush=True)
        csi500_codes = set()
    symbols = sorted(hs300_codes | csi500_codes)
    print(f"  选股宇宙: HS300({len(hs300_codes)}) + CSI500({len(csi500_codes)})"
          f" = {len(symbols)} 只（去重）", flush=True)

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

    # Phase 1b: 补充下载历史 HS300 成分股（消除幸存者偏差）
    _download_historical_hs300_members(lab, daily_path, ak_start, ak_end, target_end)

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
