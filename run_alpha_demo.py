"""
VeighNa Alpha 量化研究完整演示 (自包含, 无需 API Key)

使用模拟的 A 股数据，跑通完整的量化研究流水线:
1. 生成模拟股票数据 -> 存入 AlphaLab
2. 构建 Alpha158 因子数据集
3. 训练 LightGBM 预测模型
4. 生成交易信号
5. 运行回测
6. 输出绩效统计
"""
import sys
import random
import shelve
from datetime import datetime, timedelta
from functools import partial
from pathlib import Path

import numpy as np
import polars as pl

from vnpy.trader.object import BarData
from vnpy.trader.constant import Interval, Exchange

from vnpy.alpha import AlphaLab, AlphaDataset, Segment, AlphaModel
from vnpy.alpha.dataset import process_drop_na, process_cs_norm
from vnpy.alpha.dataset.datasets.alpha_158 import Alpha158
from vnpy.alpha.model.models.lgb_model import LgbModel
from vnpy.alpha.strategy import BacktestingEngine

LAB_PATH = "./lab/demo"
NUM_STOCKS = 20
START_DATE = datetime(2015, 1, 5)
END_DATE = datetime(2023, 12, 31)


def generate_mock_bars(symbol: str, exchange: Exchange) -> list[BarData]:
    """Generate realistic-looking daily bar data for one stock."""
    bars: list[BarData] = []

    rng = np.random.default_rng(hash(symbol) % (2**31))
    price = rng.uniform(10, 100)
    volume_base = rng.uniform(1e6, 1e8)

    current = START_DATE
    while current <= END_DATE:
        if current.weekday() >= 5:
            current += timedelta(days=1)
            continue

        daily_return = rng.normal(0.0003, 0.02)
        price *= (1 + daily_return)
        price = max(price, 1.0)

        open_price = price * (1 + rng.normal(0, 0.005))
        high_price = price * (1 + abs(rng.normal(0, 0.01)))
        low_price = price * (1 - abs(rng.normal(0, 0.01)))
        close_price = price
        volume = volume_base * (1 + rng.normal(0, 0.3))
        volume = max(volume, 10000)
        turnover = volume * price

        bar = BarData(
            symbol=symbol,
            exchange=exchange,
            datetime=current,
            interval=Interval.DAILY,
            open_price=round(open_price, 2),
            high_price=round(max(open_price, high_price, close_price), 2),
            low_price=round(min(open_price, low_price, close_price), 2),
            close_price=round(close_price, 2),
            volume=round(volume),
            turnover=round(turnover, 2),
            open_interest=0,
            gateway_name="DEMO"
        )
        bars.append(bar)
        current += timedelta(days=1)

    return bars


def main() -> None:
    print("=" * 60)
    print("  VeighNa Alpha 量化研究演示")
    print("=" * 60)

    symbols = [f"{600000 + i:06d}" for i in range(NUM_STOCKS)]
    vt_symbols = [f"{s}.SSE" for s in symbols]

    # ── Step 1: 创建 Lab 并写入模拟数据 ──
    print("\n[Step 1/6] 生成模拟股票数据并存入 AlphaLab ...")
    lab = AlphaLab(LAB_PATH)

    for i, (symbol, vt_symbol) in enumerate(zip(symbols, vt_symbols)):
        bars = generate_mock_bars(symbol, Exchange.SSE)
        lab.save_bar_data(bars)
        lab.add_contract_setting(
            vt_symbol,
            long_rate=0.001,
            short_rate=0.001,
            size=1,
            pricetick=0.01
        )
        print(f"  [{i+1}/{NUM_STOCKS}] {vt_symbol}: {len(bars)} 根日线")

    index_symbol = "000001.SSE"
    db_path = str(lab.component_path.joinpath(index_symbol))
    with shelve.open(db_path) as db:
        current = START_DATE
        while current <= END_DATE:
            date_str = current.strftime("%Y-%m-%d")
            db[date_str] = vt_symbols
            current += timedelta(days=1)
    print(f"  指数成分: {index_symbol} -> {NUM_STOCKS} 只成分股")

    # ── Step 2: 构建因子数据集 ──
    print("\n[Step 2/6] 构建 Alpha158 因子数据集 ...")
    df = lab.load_bar_df(
        vt_symbols=vt_symbols,
        interval=Interval.DAILY,
        start="2015-01-01",
        end="2023-12-31",
        extended_days=100
    )
    if df is None:
        print("  ERROR: 加载数据失败")
        sys.exit(1)
    print(f"  原始数据: {df.shape[0]} 行 x {df.shape[1]} 列")

    dataset = Alpha158(
        df,
        train_period=("2016-01-01", "2019-12-31"),
        valid_period=("2020-01-01", "2021-06-30"),
        test_period=("2021-07-01", "2023-06-30"),
    )

    dataset.add_processor("learn", partial(process_drop_na, names=["label"]))
    dataset.add_processor("learn", partial(process_cs_norm, names=["label"], method="zscore"))

    filters = lab.load_component_filters(index_symbol, "2015-01-01", "2023-12-31")

    print("  计算 158 个因子特征 (多进程) ...")
    dataset.prepare_data(filters, max_workers=4)
    print(f"  特征矩阵: {dataset.raw_df.shape[0]} 行 x {dataset.raw_df.shape[1]} 列")

    print("  运行数据处理器 (清洗+标准化) ...")
    dataset.process_data()
    print("  数据集准备完成")

    lab.save_dataset("demo", dataset)

    # ── Step 3: 训练 LightGBM 模型 ──
    print("\n[Step 3/6] 训练 LightGBM 预测模型 ...")
    model: AlphaModel = LgbModel(seed=42)
    model.fit(dataset)
    print("  模型训练完成")
    lab.save_model("demo", model)

    # ── Step 4: 生成交易信号 ──
    print("\n[Step 4/6] 在测试集上生成交易信号 ...")
    predictions = model.predict(dataset, Segment.TEST)
    df_test = dataset.fetch_infer(Segment.TEST)
    df_test = df_test.with_columns(pl.Series(predictions).alias("signal"))
    signal = df_test.select(["datetime", "vt_symbol", "signal"])
    print(f"  信号数据: {signal.shape[0]} 行")
    lab.save_signal("demo", signal)

    # ── Step 5: 运行策略回测 ──
    print("\n[Step 5/6] 运行多股票组合回测 ...")

    import vnpy.alpha.strategy.strategies.equity_demo_strategy as strat_module
    EquityDemoStrategy = strat_module.EquityDemoStrategy

    engine = BacktestingEngine(lab)
    engine.set_parameters(
        vt_symbols=vt_symbols,
        interval=Interval.DAILY,
        start=datetime(2021, 7, 1),
        end=datetime(2023, 6, 30),
        capital=10_000_000,
    )

    setting = {"top_k": 10, "n_drop": 2, "hold_thresh": 3}
    engine.add_strategy(EquityDemoStrategy, setting, signal)

    engine.load_data()
    engine.run_backtesting()
    engine.calculate_result()

    # ── Step 6: 输出绩效统计 ──
    print("\n[Step 6/6] 绩效统计结果:")
    print("-" * 50)
    stats = engine.calculate_statistics()

    print("\n" + "=" * 60)
    print("  演示完成!")
    print(f"  所有数据保存在: {Path(LAB_PATH).resolve()}")
    print("=" * 60)


if __name__ == "__main__":
    main()
