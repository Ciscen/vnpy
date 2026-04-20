"""
用 AKShare 下载真实 A 股数据到 AlphaLab (完全免费, 无需 Token)

AKShare 是开源免费的金融数据接口库，无需注册、无需 Token。
官网: https://akshare.akfamily.xyz

使用方式:
  pip install akshare (已安装)
  python download_real_data.py
"""
import time
from datetime import datetime

import akshare as ak

from vnpy.trader.object import BarData
from vnpy.trader.constant import Interval, Exchange
from vnpy.alpha import AlphaLab

# ============ 配置区 (只需修改这里) ============

LAB_PATH = "./lab/real_stock"

STOCKS = {
    # (AKShare前缀+代码, VeighNa代码, 交易所, 名称)
    "sh600519": ("600519", Exchange.SSE, "贵州茅台"),
    "sz000858": ("000858", Exchange.SZSE, "五粮液"),
    "sh601318": ("601318", Exchange.SSE, "中国平安"),
    "sz000001": ("000001", Exchange.SZSE, "平安银行"),
    "sh600036": ("600036", Exchange.SSE, "招商银行"),
}

START = "20150101"
END = "20241231"

LONG_RATE = 0.0003     # 买入佣金率 (万三)
SHORT_RATE = 0.0013    # 卖出佣金率 (万三 + 千一印花税)
SIZE = 1               # 合约乘数 (股票=1)
PRICETICK = 0.01       # 最小价格变动

# ===============================================


def download() -> None:
    lab = AlphaLab(LAB_PATH)
    total = len(STOCKS)

    for i, (ak_symbol, (symbol, exchange, name)) in enumerate(STOCKS.items(), 1):
        print(f"[{i}/{total}] 下载 {symbol} ({name}) ...")

        try:
            df = ak.stock_zh_a_daily(
                symbol=ak_symbol,
                start_date=START,
                end_date=END,
                adjust="qfq",
            )
        except Exception as e:
            print(f"  错误: {e}")
            continue

        if df is None or df.empty:
            print(f"  警告: {symbol} 无数据，跳过")
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

        vt_symbol = f"{symbol}.{exchange.value}"
        lab.add_contract_setting(vt_symbol, LONG_RATE, SHORT_RATE, SIZE, PRICETICK)

        print(f"  已保存 {len(bars)} 根日线 -> {vt_symbol}")
        time.sleep(1)

    print(f"\n{'=' * 50}")
    print(f"全部完成! 数据保存在: {LAB_PATH}")
    print(f"{'=' * 50}")
    print("\n下一步:")
    print("  修改 run_alpha_demo.py，替换数据源为真实数据即可")
    print("  具体方法见 ALPHA_LEARNING_GUIDE.md 第六章")


if __name__ == "__main__":
    download()
