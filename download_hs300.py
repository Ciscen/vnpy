"""
下载沪深300成分股历史数据到 AlphaLab (增量下载, 已下载的自动跳过)
"""
import time
import sys
from datetime import datetime
from pathlib import Path

import akshare as ak

from vnpy.trader.object import BarData
from vnpy.trader.constant import Interval, Exchange
from vnpy.alpha import AlphaLab

LAB_PATH = "./lab/hs300"
START = "20150101"
END = "20241231"

LONG_RATE = 0.0003
SHORT_RATE = 0.0013
SIZE = 1
PRICETICK = 0.01

MAX_RETRIES = 3
SLEEP_BETWEEN = 1.5


def symbol_to_exchange(code: str) -> Exchange:
    if code.startswith(("6", "5")):
        return Exchange.SSE
    elif code.startswith(("0", "3")):
        return Exchange.SZSE
    return Exchange.SSE


def symbol_to_ak_prefix(code: str) -> str:
    if code.startswith(("6", "5")):
        return f"sh{code}"
    return f"sz{code}"


def download_one(ak_symbol: str) -> "pd.DataFrame | None":
    """带重试的下载"""
    for attempt in range(MAX_RETRIES):
        try:
            df = ak.stock_zh_a_daily(
                symbol=ak_symbol,
                start_date=START,
                end_date=END,
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


def main() -> None:
    print("获取沪深300成分股列表...", flush=True)
    cons_df = ak.index_stock_cons(symbol="000300")
    symbols = list(cons_df["品种代码"])
    print(f"成分股数量: {len(symbols)}", flush=True)

    lab = AlphaLab(LAB_PATH)
    daily_path = Path(LAB_PATH) / "daily"
    total = len(symbols)
    success = 0
    skipped = 0
    failed = []

    for i, symbol in enumerate(symbols, 1):
        exchange = symbol_to_exchange(symbol)
        vt_symbol = f"{symbol}.{exchange.value}"

        parquet_file = daily_path / f"{vt_symbol}.parquet"
        if parquet_file.exists():
            skipped += 1
            success += 1
            if skipped <= 5 or skipped % 20 == 0:
                print(f"  [{i}/{total}] {vt_symbol} 已存在, 跳过 (共跳过{skipped})", flush=True)
            continue

        ak_symbol = symbol_to_ak_prefix(symbol)
        df = download_one(ak_symbol)

        if df is None or df.empty:
            print(f"  [{i}/{total}] {vt_symbol} 无数据", flush=True)
            failed.append(vt_symbol)
            continue

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
                turnover=float(row["amount"]),
                open_interest=0,
                gateway_name="AKShare"
            )
            bars.append(bar)

        lab.save_bar_data(bars)
        lab.add_contract_setting(vt_symbol, LONG_RATE, SHORT_RATE, SIZE, PRICETICK)

        success += 1
        print(f"  [{i}/{total}] {vt_symbol}: {len(bars)} 根日线 ✓", flush=True)
        time.sleep(SLEEP_BETWEEN)

    print(f"\n{'=' * 50}", flush=True)
    print(f"下载完成!", flush=True)
    print(f"  成功: {success} 只 (其中跳过已有: {skipped})", flush=True)
    print(f"  失败: {len(failed)} 只", flush=True)
    if failed:
        print(f"  失败列表: {failed}", flush=True)
    print(f"  数据保存在: {LAB_PATH}", flush=True)
    print(f"{'=' * 50}", flush=True)


if __name__ == "__main__":
    main()
