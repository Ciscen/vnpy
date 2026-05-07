# hs300_topk/data/download_data.py
"""
hs300_topk/data/download_data.py

**[已废弃]** — 本文件功能已迁移至 run_pipeline.py Phase 1。
保留仅供参考，请勿在新代码中引用。

原始功能
--------
1. 使用 akshare 获取当前沪深 300 成分股列表。
2. 下载 10 年日线 OHLCV 数据。
3. 本地缓存（Parquet）+ 增量下载。
"""

import os
import json
import time
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Tuple, Set

import pandas as pd
import akshare as ak

# ----------------------------------------------------------------------
# 配置
# ----------------------------------------------------------------------
CACHE_DIR = Path(__file__).resolve().parents[1] / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# 数据文件（Parquet）以及成分股缓存（JSON）
DATA_CACHE_FILE = CACHE_DIR / "hs300_daily.parquet"
COMPONENT_CACHE_FILE = CACHE_DIR / "hs300_components.json"

# 下载区间（10 年: 2016-04-30 ~ 2026-04-30）
START_DATE = "20160430"
END_DATE   = "20260430"

# ----------------------------------------------------------------------
# 辅助工具函数
# ----------------------------------------------------------------------
def _load_component_cache() -> List[str]:
    """读取本地保存的当前 HS300 成分股列表（仅一次性获取）。"""
    if COMPONENT_CACHE_FILE.is_file():
        with open(COMPONENT_CACHE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return []

def _save_component_cache(symbols: List[str]) -> None:
    """将当前成分股列表写入 JSON 缓存。"""
    with open(COMPONENT_CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(symbols, f, ensure_ascii=False, indent=2)

def _get_current_components() -> List[str]:
    """使用 akshare 拉取当前沪深 300 成分股（不含历史动态）。"""
    df = ak.index_stock_cons(symbol="000300")
    # akshare 返回的列名为 "symbol"（如 "600000.SH"）
    symbols = df["symbol"].tolist()
    return symbols

def _trading_days(start: str, end: str) -> List[str]:
    """返回区间内所有交易日（排除周末），格式为 YYYY-MM-DD。"""
    start_dt = datetime.strptime(start, "%Y%m%d")
    end_dt = datetime.strptime(end, "%Y%m%d")
    days = []
    cur = start_dt
    while cur <= end_dt:
        if cur.weekday() < 5:  # Monday‑Friday
            days.append(cur.strftime("%Y-%m-%d"))
        cur += timedelta(days=1)
    return days

def _download_daily_for_symbols(symbols: List[str], start: str, end: str) -> pd.DataFrame:
    """批量下载单只股票在给定区间的日线数据（前复权），返回 MultiIndex DataFrame。"""
    records = []
    for sym in symbols:
        raw_code = sym.split(".")[0]
        try:
            df = ak.stock_zh_a_hist(symbol=raw_code,
                                    period="daily",
                                    start_date=start,
                                    end_date=end,
                                    adjust="qfq")  # 前复权
            if df.empty:
                continue
            df["symbol"] = sym
            df = df.rename(columns={
                "open": "open",
                "high": "high",
                "low": "low",
                "close": "close",
                "volume": "volume",
                "turnover": "turnover",
            })
            records.append(df[["symbol", "date", "open", "high", "low", "close", "volume", "turnover"]])
        except Exception as e:
            print(f"[download] {sym} failed: {e}")
        # 防止频繁请求被封禁
        time.sleep(0.5)
    if not records:
        return pd.DataFrame()
    full = pd.concat(records, ignore_index=True)
    full["date"] = pd.to_datetime(full["date"])
    full.set_index(["symbol", "date"], inplace=True)
    full.sort_index(inplace=True)
    return full

# ----------------------------------------------------------------------
# 主函数
# ----------------------------------------------------------------------
def download_hs300_data(start: str = START_DATE,
                        end: str = END_DATE,
                        force_refresh: bool = False) -> pd.DataFrame:
    """下载并返回沪深 300 成分股的日线数据（含换手率）。

    - 首次执行会全量下载并缓存；
    - 再次执行仅增量下载缺失的日期/股票；
    - 成分股列表仅使用当前名单（暂时接受幸存者偏差）。
    """
    # 1️⃣ 加载已有缓存（若存在且不强制刷新）
    if DATA_CACHE_FILE.is_file() and not force_refresh:
        cached_df = pd.read_parquet(DATA_CACHE_FILE)
    else:
        cached_df = pd.DataFrame()

    # 2️⃣ 获取当前成分股（缓存）
    component_symbols = _load_component_cache()
    if force_refresh or not component_symbols:
        print("[download] Fetching current HS300 component list …")
        component_symbols = _get_current_components()
        _save_component_cache(component_symbols)

    # 3️⃣ 计算需要下载的 (symbol, date) 对
    # 所有交易日集合
    all_dates = set(_trading_days(start, end))
    required_pairs: Set[Tuple[str, str]] = set()
    for sym in component_symbols:
        for dt in all_dates:
            required_pairs.add((sym, dt))

    # 已有数据的 (symbol, date) 集合，需要将 Timestamp 转为字符串
    existing_pairs: Set[Tuple[str, str]] = set()
    if not cached_df.empty:
        for sym, dt in cached_df.index:
            existing_pairs.add((sym, dt.strftime("%Y-%m-%d")))

    missing_pairs = required_pairs - existing_pairs
    if not missing_pairs:
        print("[download] All required data already cached.")
        return cached_df

    # 按股票分组收集缺失的日期列表，后续按最小/最大区间下载
    missing_by_symbol: dict = {}
    for sym, dt in missing_pairs:
        missing_by_symbol.setdefault(sym, []).append(dt)

    downloaded_parts = []
    for sym, day_list in missing_by_symbol.items():
        # 为简化下载，取该股票缺失日期的最小/最大区间进行一次请求
        start_day = min(day_list).replace("-", "")
        end_day = max(day_list).replace("-", "")
        part = _download_daily_for_symbols([sym], start_day, end_day)
        if not part.empty:
            downloaded_parts.append(part)

    if downloaded_parts:
        new_data = pd.concat(downloaded_parts)
        combined = pd.concat([cached_df, new_data])
    else:
        combined = cached_df

    # 4️⃣ 去除可能出现的重复行（基于完整的 MultiIndex）
    if not combined.empty:
        combined = combined[~combined.index.duplicated(keep="last")]
        combined.sort_index(inplace=True)

    # 5️⃣ 持久化缓存
    combined.to_parquet(DATA_CACHE_FILE)
    print(f"[download] Cached rows: {combined.shape[0]}")
    return combined

# ----------------------------------------------------------------------
# 脚本直接运行入口
# ----------------------------------------------------------------------
if __name__ == "__main__":
    df = download_hs300_data()
    if not df.empty:
        print(f"Download completed – rows: {len(df)}")
        print(f"Date range: {df.index.get_level_values(1).min().date()} ~ {df.index.get_level_values(1).max().date()}")
    else:
        print("No data downloaded.")
