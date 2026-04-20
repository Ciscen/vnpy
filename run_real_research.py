"""
用真实 A 股数据跑通完整量化研究流程

数据来源: download_real_data.py 下载的 5 只股票 10 年日线
流程: 因子计算 -> 模型训练 -> 信号生成 -> 回测 -> 绩效统计
"""
import sys
import shelve
from datetime import datetime, timedelta
from functools import partial
from pathlib import Path

import polars as pl

from vnpy.trader.constant import Interval, Exchange

from vnpy.alpha import AlphaLab, AlphaDataset, Segment, AlphaModel
from vnpy.alpha.dataset import process_drop_na, process_cs_norm
from vnpy.alpha.dataset.datasets.alpha_158 import Alpha158
from vnpy.alpha.model.models.lgb_model import LgbModel
from vnpy.alpha.strategy import BacktestingEngine

LAB_PATH = "./lab/real_stock"

VT_SYMBOLS = [
    "600519.SSE",     # 贵州茅台
    "000858.SZSE",    # 五粮液
    "601318.SSE",     # 中国平安
    "000001.SZSE",    # 平安银行
    "600036.SSE",     # 招商银行
]


def main() -> None:
    print("=" * 60)
    print("  真实数据量化研究")
    print("=" * 60)

    lab = AlphaLab(LAB_PATH)

    # ── 准备: 写入成分股索引 (回测需要) ──
    print("\n[准备] 写入成分股索引 ...")
    index_symbol = "MY_POOL.SSE"
    db_path = str(lab.component_path.joinpath(index_symbol))
    start_dt = datetime(2015, 1, 1)
    end_dt = datetime(2024, 12, 31)
    with shelve.open(db_path) as db:
        current = start_dt
        while current <= end_dt:
            db[current.strftime("%Y-%m-%d")] = VT_SYMBOLS
            current += timedelta(days=1)
    print(f"  成分股: {index_symbol} -> {len(VT_SYMBOLS)} 只")

    # ── Step 1: 加载数据 ──
    print("\n[Step 1/5] 加载真实 K 线数据 ...")
    df = lab.load_bar_df(
        vt_symbols=VT_SYMBOLS,
        interval=Interval.DAILY,
        start="2015-01-01",
        end="2024-12-31",
        extended_days=100
    )
    if df is None:
        print("  ERROR: 加载数据失败，请先运行 download_real_data.py")
        sys.exit(1)
    print(f"  原始数据: {df.shape[0]} 行 x {df.shape[1]} 列")

    # ── Step 2: 构建因子数据集 ──
    print("\n[Step 2/5] 构建 Alpha158 因子数据集 ...")
    dataset = Alpha158(
        df,
        train_period=("2016-01-01", "2020-12-31"),   # 5年训练
        valid_period=("2021-01-01", "2022-06-30"),   # 1.5年验证
        test_period=("2022-07-01", "2024-06-30"),    # 2年测试
    )

    dataset.add_processor("learn", partial(process_drop_na, names=["label"]))
    dataset.add_processor("learn", partial(process_cs_norm, names=["label"], method="zscore"))

    filters = lab.load_component_filters(index_symbol, "2015-01-01", "2024-12-31")

    print("  计算 158 个因子特征 ...")
    dataset.prepare_data(filters, max_workers=4)
    print(f"  特征矩阵: {dataset.raw_df.shape[0]} 行 x {dataset.raw_df.shape[1]} 列")

    print("  运行数据处理器 ...")
    dataset.process_data()
    print("  数据集准备完成")

    lab.save_dataset("real_research", dataset)

    # ── Step 3: 训练 LightGBM 模型 ──
    print("\n[Step 3/5] 训练 LightGBM 预测模型 ...")
    model: AlphaModel = LgbModel(
        learning_rate=0.05,
        num_leaves=31,
        num_boost_round=1000,
        early_stopping_rounds=50,
        seed=42,
    )
    model.fit(dataset)
    print("  模型训练完成")
    lab.save_model("real_research", model)

    # ── Step 4: 生成交易信号 ──
    print("\n[Step 4/5] 在测试集上生成交易信号 ...")
    predictions = model.predict(dataset, Segment.TEST)
    df_test = dataset.fetch_infer(Segment.TEST)
    df_test = df_test.with_columns(pl.Series(predictions).alias("signal"))
    signal = df_test.select(["datetime", "vt_symbol", "signal"])
    print(f"  信号数据: {signal.shape[0]} 行")
    lab.save_signal("real_research", signal)

    # ── Step 5: 回测 ──
    print("\n[Step 5/5] 运行策略回测 ...")

    import vnpy.alpha.strategy.strategies.equity_demo_strategy as strat_module
    EquityDemoStrategy = strat_module.EquityDemoStrategy

    engine = BacktestingEngine(lab)
    engine.set_parameters(
        vt_symbols=VT_SYMBOLS,
        interval=Interval.DAILY,
        start=datetime(2022, 7, 1),
        end=datetime(2024, 6, 30),
        capital=1_000_000,
    )

    setting = {
        "top_k": 3,
        "n_drop": 1,
        "min_days": 5,
        "cash_ratio": 0.95,
        "min_volume": 100,
    }
    engine.add_strategy(EquityDemoStrategy, setting, signal)

    engine.load_data()
    engine.run_backtesting()
    engine.calculate_result()

    # ── 绩效统计 ──
    print("\n绩效统计结果:")
    print("-" * 50)
    stats = engine.calculate_statistics()

    print("\n" + "=" * 60)
    print("  研究完成!")
    print(f"  数据保存在: {Path(LAB_PATH).resolve()}")
    print("=" * 60)


if __name__ == "__main__":
    main()
