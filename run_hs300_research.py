"""
沪深300成分股量化研究 — 完整注释版

===== 整体流程概览 =====

  输入: 279 只股票 × 10 年日线 K 线数据 (Parquet 文件, 由 download_hs300.py 下载)
  输出: 绩效统计报告 (Sharpe Ratio、年化收益、最大回撤等)

  流程:
    [数据文件]
         ↓
    Step 1: load_bar_df() 把 Parquet 文件读入内存为 Polars DataFrame
         ↓ 输出: df (574024 行 × 10 列: datetime, vt_symbol, open, high, low, close, volume, turnover, ...)
    Step 2: Alpha158 计算 158 个因子 + 设置标签(label)
         ↓ 输出: dataset.raw_df (574024 行 × 161 列: datetime, vt_symbol, 158个因子, label)
         ↓ 经过 process_data() 清洗后得到 learn_df (供训练) 和 infer_df (供推理)
    Step 3: LgbModel.fit() 用训练集/验证集训练 LightGBM 模型
         ↓ 输出: 训练好的模型对象 (model), 序列化为 model/hs300.pkl
    Step 4: model.predict() 在测试集上预测, 生成交易信号
         ↓ 输出: signal (Polars DataFrame, 3 列: datetime, vt_symbol, signal)
    Step 5: BacktestingEngine 用信号驱动策略, 逐日模拟交易
         ↓ 输出: 绩效统计 (dict)

===== Android 开发者类比 =====

  AlphaLab      ≈ 项目的本地数据库 (Room/SQLite), 管理所有持久化文件
  AlphaDataset  ≈ RecyclerView.Adapter, 负责把原始数据加工成模型需要的格式
  LgbModel      ≈ TensorFlow Lite 模型, 输入特征向量 → 输出预测值
  Signal        ≈ 接口返回的 JSON 数据, 供 UI/策略层消费
  BacktestEngine ≈ Espresso UI 测试, 模拟用户操作验证行为正确性
  Strategy      ≈ ViewModel/Presenter, 接收信号 → 决策 → 执行操作
"""
import sys
import shelve
from datetime import datetime, timedelta
from functools import partial
from pathlib import Path

import polars as pl

from vnpy.trader.constant import Interval

# ── 核心模块导入说明 ──
# AlphaLab: 研究工作区管理器, 封装了文件系统操作 (读写 Parquet/shelve/pickle)
# AlphaDataset: 因子数据集基类, 管理特征工程和数据划分
# Segment: 枚举, 取值 TRAIN/VALID/TEST, 标识数据集的哪个分段
# AlphaModel: 预测模型基类, 定义 fit()/predict() 接口
from vnpy.alpha import AlphaLab, AlphaDataset, Segment, AlphaModel

# process_drop_na: 数据处理器 — 删除指定列含 NaN 的行
# process_cs_norm: 数据处理器 — 截面标准化 (同一天内, 对所有股票的某列做 Z-Score)
from vnpy.alpha.dataset import process_drop_na, process_cs_norm

# Alpha158: 预置的因子数据集, 内含 158 个量化因子的定义 + 标签定义
# 继承自 AlphaDataset, 在 __init__ 中自动调用 158 次 add_feature()
from vnpy.alpha.dataset.datasets.alpha_158 import Alpha158

# LgbModel: 基于 LightGBM 的梯度提升树模型
# 内部使用 lightgbm.train() 训练, 支持 early stopping
from vnpy.alpha.model.models.lgb_model import LgbModel

# BacktestingEngine: 回测引擎, 逐日回放历史数据并模拟交易
from vnpy.alpha.strategy import BacktestingEngine


# =====================================================
# 配置区
# =====================================================

# AlphaLab 工作区路径, 包含以下子目录:
#   daily/      → 日线 Parquet 文件 (每只股票一个文件)
#   minute/     → 分钟线 (本次不用)
#   component/  → 成分股索引 (shelve 数据库)
#   dataset/    → 计算好的因子数据集 (pickle)
#   model/      → 训练好的模型 (pickle)
#   signal/     → 交易信号 (Parquet)
#   contract.json → 合约参数 (手续费率、合约乘数等)
LAB_PATH = "./lab/hs300"


def main() -> None:
    print("=" * 60, flush=True)
    print("  沪深300 量化研究", flush=True)
    print("=" * 60, flush=True)

    # ──────────────────────────────────────────────────────────
    # 创建 AlphaLab 实例
    #
    # 输入: LAB_PATH (字符串, 工作区目录路径)
    # 作用: 初始化目录结构, 后续所有数据读写都通过 lab 对象完成
    # 类比: 类似 Android 的 Room.databaseBuilder().build()
    # ──────────────────────────────────────────────────────────
    lab = AlphaLab(LAB_PATH)

    # ──────────────────────────────────────────────────────────
    # 扫描 daily/ 目录, 获取所有已下载股票的 vt_symbol
    #
    # vt_symbol 格式: "股票代码.交易所"
    #   例: "600519.SSE" = 贵州茅台 (上海证券交易所)
    #       "000858.SZSE" = 五粮液 (深圳证券交易所)
    #
    # 输出: vt_symbols = ["000001.SZSE", "000002.SZSE", ..., "688599.SSE"]
    #       长度约 279 (下载成功的股票数)
    # ──────────────────────────────────────────────────────────
    daily_path = Path(LAB_PATH) / "daily"
    vt_symbols = sorted([f.stem for f in daily_path.glob("*.parquet")])
    print(f"\n可用股票数: {len(vt_symbols)}", flush=True)

    # ══════════════════════════════════════════════════════════
    # 准备工作: 写入成分股索引
    # ══════════════════════════════════════════════════════════
    #
    # 为什么需要成分股索引?
    #   在因子计算阶段, Alpha158.prepare_data() 需要知道每一天的"股票池"是哪些。
    #   例如: 2020-01-01 这天, 哪些股票属于沪深300?
    #   这个信息存储在 component/ 目录下的 shelve 数据库中。
    #
    # 数据结构: shelve 键值对
    #   key   = 日期字符串 "2020-01-01"
    #   value = vt_symbol 列表 ["600519.SSE", "000858.SZSE", ...]
    #
    # 简化处理: 这里假设所有 279 只股票在整个时间段内都是成分股
    #   (真实场景中成分股会定期调整, 需要按日期填入不同的列表)
    # ──────────────────────────────────────────────────────────
    print("\n[准备] 写入成分股索引 ...", flush=True)
    index_symbol = "HS300.SSE"  # 索引名称, 任意取名, 后面要保持一致
    db_path = str(lab.component_path.joinpath(index_symbol))
    start_dt = datetime(2015, 1, 1)
    end_dt = datetime(2024, 12, 31)
    with shelve.open(db_path) as db:
        current = start_dt
        while current <= end_dt:
            db[current.strftime("%Y-%m-%d")] = vt_symbols
            current += timedelta(days=1)
    print(f"  成分股: {index_symbol} -> {len(vt_symbols)} 只", flush=True)

    # ══════════════════════════════════════════════════════════
    # Step 1: 加载 K 线数据
    # ══════════════════════════════════════════════════════════
    #
    # 输入:
    #   vt_symbols:    要加载的股票列表 (279 只)
    #   interval:      K 线级别 (日线)
    #   start/end:     时间范围
    #   extended_days: 向前多读 100 天数据
    #                  因为某些因子需要 60 天历史窗口, 多读确保不缺数据
    #
    # 输出: Polars DataFrame, 每行是一只股票一天的数据
    #   ┌────────────┬──────────────┬───────┬───────┬───────┬───────┬────────┬──────────┬───┬───┐
    #   │ datetime   │ vt_symbol    │ open  │ high  │ low   │ close │ volume │ turnover │...│...│
    #   ├────────────┼──────────────┼───────┼───────┼───────┼───────┼────────┼──────────┼───┼───┤
    #   │ 2014-09-24 │ 000001.SZSE  │ 1.01  │ 1.02  │ 1.00  │ 1.01  │ 4.5e7  │ 5.2e8   │   │   │
    #   │ 2014-09-24 │ 000002.SZSE  │ 0.98  │ 0.99  │ 0.97  │ 0.98  │ 3.1e7  │ 2.8e8   │   │   │
    #   │ ...        │ ...          │ ...   │ ...   │ ...   │ ...   │ ...    │ ...      │   │   │
    #   └────────────┴──────────────┴───────┴───────┴───────┴───────┴────────┴──────────┴───┴───┘
    #
    # 注意: open/high/low/close 已被"归一化" — 除以每只股票的第一天收盘价
    #       所以值都在 1.0 附近, 方便跨股票比较
    #       volume 和 turnover 也做了同样处理
    #
    # 行数: 约 574024 (279只 × ~2060天)
    # 列数: 10 (datetime, vt_symbol, open, high, low, close, volume, turnover, open_interest, vwap)
    # ──────────────────────────────────────────────────────────
    print("\n[Step 1/5] 加载 K 线数据 (280只 x 10年, 可能需要30秒) ...", flush=True)
    df = lab.load_bar_df(
        vt_symbols=vt_symbols,
        interval=Interval.DAILY,
        start="2015-01-01",
        end="2024-12-31",
        extended_days=100
    )
    if df is None:
        print("  ERROR: 加载数据失败", flush=True)
        sys.exit(1)
    print(f"  原始数据: {df.shape[0]} 行 x {df.shape[1]} 列", flush=True)

    # ══════════════════════════════════════════════════════════
    # Step 2: 构建因子数据集
    # ══════════════════════════════════════════════════════════
    #
    # 这一步做三件事: (1)定义因子 (2)定义标签 (3)计算+清洗
    # ──────────────────────────────────────────────────────────

    print("\n[Step 2/5] 构建 Alpha158 因子数据集 (计算量大, 可能需要几分钟) ...", flush=True)

    # ── 2a. 创建 Alpha158 数据集 ──
    #
    # Alpha158.__init__() 内部自动执行:
    #   1. 调用 158 次 self.add_feature(name, expression) 注册因子
    #      例: add_feature("ma_5", "ts_mean(close, 5) / close")
    #           → 5日均线偏离度 = 过去5天收盘价均值 / 今天收盘价
    #      例: add_feature("roc_10", "ts_delay(close, 10) / close")
    #           → 10日动量 = 10天前的收盘价 / 今天收盘价
    #      例: add_feature("std_20", "ts_std(close, 20) / close")
    #           → 20日波动率 = 过去20天收益率标准差 / 今天收盘价
    #
    #   2. 调用 self.set_label("ts_delay(close, -3) / ts_delay(close, -1) - 1")
    #      → 标签 = (3天后收盘价 / 明天收盘价) - 1
    #      → 即: 从明天开始持有到第3天的收益率
    #      → 负号 ts_delay(close, -3) 表示"看未来", 只在训练时有意义
    #
    # 输入:
    #   df:            Step 1 的 DataFrame (574024 行 × 10 列)
    #   train_period:  训练集时间段, 模型在这些数据上学习
    #   valid_period:  验证集时间段, 用于调参和 early stopping
    #   test_period:   测试集时间段, 最终评估模型表现, 模型从未见过
    #
    # 时间轴示意:
    #   2015 ── 2016 ──────── 2020 ── 2021 ── 2022.6 ── 2024.6
    #           │  TRAIN (5年)  │  VALID (1.5年) │  TEST (2年)  │
    #
    dataset = Alpha158(
        df,
        train_period=("2016-01-01", "2020-12-31"),
        valid_period=("2021-01-01", "2022-06-30"),
        test_period=("2022-07-01", "2024-06-30"),
    )

    # ── 2b. 注册数据处理器 ──
    #
    # 数据处理器是一个函数, 在 process_data() 时被调用
    # "learn" 处理器: 只作用于 learn_df (训练+验证数据)
    # "infer" 处理器: 只作用于 infer_df (推理数据)
    #
    # process_drop_na(names=["label"]):
    #   删除 label 列为 NaN 的行
    #   为什么有 NaN? 因为标签定义用了"未来3天收盘价", 最后3天没有未来数据
    #   类比: RecyclerView 中过滤掉无效数据项
    #
    # process_cs_norm(names=["label"], method="zscore"):
    #   截面 Z-Score 标准化 — 对 label 列在每天内部做标准化
    #   计算: label = (label - 当天所有股票label的均值) / 当天所有股票label的标准差
    #   目的: 不同日期的市场环境不同 (牛市涨幅大, 熊市涨幅小),
    #         标准化后让模型关注"相对排名"而非"绝对涨幅"
    #   类比: 不同班级的考试分数标准化后才能跨班比较
    dataset.add_processor("learn", partial(process_drop_na, names=["label"]))
    dataset.add_processor("learn", partial(process_cs_norm, names=["label"], method="zscore"))

    # ── 2c. 加载成分股筛选器 ──
    #
    # 输入: index_symbol, 起止日期
    # 输出: filters = Dict[日期字符串, Set[vt_symbol]]
    #       例: {"2020-01-01": {"600519.SSE", "000858.SZSE", ...}, ...}
    # 作用: prepare_data() 用它来筛选每天哪些股票参与因子计算
    #       不在成分股内的股票数据会被丢弃
    filters = lab.load_component_filters(index_symbol, "2015-01-01", "2024-12-31")

    # ── 2d. 计算所有因子特征 (最耗时的步骤) ──
    #
    # 内部流程:
    #   1. 对每个因子表达式 (158个), 逐一解析并计算
    #      例: "ts_mean(close, 5) / close"
    #           → 先对每只股票的 close 列计算 5 日滚动均值
    #           → 再除以当天 close
    #   2. 使用多进程 (max_workers=4) 并行计算, 加速
    #   3. 计算标签列
    #   4. 按日期筛选: 只保留 train/valid/test 时间段内的数据
    #   5. 按成分股筛选: 用 filters 只保留当天属于成分股的行
    #
    # 输出: dataset.raw_df — 原始因子矩阵
    #   ┌────────────┬──────────────┬────────┬────────┬─────┬────────┬────────┐
    #   │ datetime   │ vt_symbol    │ kmid   │ klow_2 │ ... │ wvma_60│ label  │
    #   ├────────────┼──────────────┼────────┼────────┼─────┼────────┼────────┤
    #   │ 2016-01-04 │ 000001.SZSE  │ -0.003 │  0.45  │ ... │  1.23  │  0.015 │
    #   │ 2016-01-04 │ 000002.SZSE  │  0.008 │  0.52  │ ... │  0.87  │ -0.008 │
    #   │ ...        │ ...          │ ...    │ ...    │ ... │  ...   │  ...   │
    #   └────────────┴──────────────┴────────┴────────┴─────┴────────┴────────┘
    #   列: datetime + vt_symbol + 158个因子 + label = 161 列
    #   行: 约 574024 行
    print("  计算 158 个因子特征 (多进程) ...", flush=True)
    dataset.prepare_data(filters, max_workers=4)
    print(f"  特征矩阵: {dataset.raw_df.shape[0]} 行 x {dataset.raw_df.shape[1]} 列", flush=True)

    # ── 2e. 运行数据处理器 ──
    #
    # 将 raw_df 拆分为两个 DataFrame:
    #   learn_df: 供训练用, 经过 process_drop_na + process_cs_norm 处理
    #   infer_df: 供推理用, 不做 label 处理 (预测时不需要 label)
    #
    # 处理后的 learn_df 每天的 label 均值≈0, 标准差≈1
    print("  运行数据处理器 ...", flush=True)
    dataset.process_data()
    print("  数据集准备完成", flush=True)

    # 持久化: 将整个 dataset 对象序列化为 dataset/hs300.pkl
    # 下次可以直接 lab.load_dataset("hs300") 加载, 跳过因子计算
    lab.save_dataset("hs300", dataset)

    # ══════════════════════════════════════════════════════════
    # Step 3: 训练 LightGBM 预测模型
    # ══════════════════════════════════════════════════════════
    #
    # LightGBM 是什么?
    #   一种梯度提升决策树 (GBDT) 算法, 微软开源。
    #   原理: 构建一系列决策树, 每棵树纠正前一棵树的预测误差。
    #   类比: 就像多个评委打分, 最终取加权平均, 比单个评委更准。
    #
    # 输入:
    #   dataset: 包含 learn_df (训练+验证数据)
    #            特征 = 第3列到倒数第2列 (跳过 datetime, vt_symbol, label)
    #            标签 = label 列
    #
    # 参数解释:
    #   learning_rate=0.05:     学习率 — 每棵树对最终结果的贡献权重
    #                           越小: 学习越慢但越精细, 需要更多轮
    #                           越大: 学习越快但容易过拟合
    #   num_leaves=63:          叶节点数 — 单棵树的复杂度
    #                           默认31, 这里用63让每棵树能捕捉更复杂的模式
    #                           过大会过拟合, 过小会欠拟合
    #   num_boost_round=1000:   最大训练轮数 (最多建1000棵树)
    #   early_stopping_rounds=50: 早停 — 如果验证集上连续50轮没有改善, 就停止
    #                              防止过拟合的关键机制
    #   log_evaluation_period=20: 每20轮打印一次训练日志
    #   seed=42:                随机种子, 确保结果可复现
    #
    # 训练过程:
    #   第1轮: 建第1棵树, 在训练集上拟合 label, 在验证集上评估误差
    #   第2轮: 建第2棵树, 拟合第1棵树的"残差"(预测偏差)
    #   ...
    #   第N轮: 如果验证集误差连续50轮没降低, 停止 (early stopping)
    #   最终模型 = 所有树的预测值之和
    #
    # 输出: 训练好的模型对象 (self.model = lgb.Booster)
    # ──────────────────────────────────────────────────────────
    print("\n[Step 3/5] 训练 LightGBM 预测模型 ...", flush=True)
    model: AlphaModel = LgbModel(
        learning_rate=0.05,
        num_leaves=63,
        num_boost_round=1000,
        early_stopping_rounds=50,
        log_evaluation_period=20,
        seed=42,
    )
    model.fit(dataset)
    # fit() 内部做了什么:
    #   1. 从 dataset.learn_df 取 TRAIN 段 → 训练集
    #   2. 从 dataset.learn_df 取 VALID 段 → 验证集
    #   3. 将 DataFrame 转为 lightgbm.Dataset (numpy array)
    #   4. 调用 lgb.train() 开始迭代训练
    #   5. 每轮计算训练集和验证集的 MSE (均方误差)
    #   6. 如果验证集 MSE 连续50轮不降 → early stopping
    print("  模型训练完成", flush=True)

    # 持久化: 序列化为 model/hs300.pkl
    lab.save_model("hs300", model)

    # ══════════════════════════════════════════════════════════
    # Step 4: 生成交易信号
    # ══════════════════════════════════════════════════════════
    #
    # 用训练好的模型, 在测试集上做预测
    # 测试集 = 2022-07-01 ~ 2024-06-30 (模型从未见过的数据)
    #
    # model.predict() 内部流程:
    #   1. 从 dataset.infer_df 取 TEST 段的数据
    #   2. 提取特征列 (跳过 datetime, vt_symbol, label)
    #   3. 转为 numpy array
    #   4. 调用 lgb.Booster.predict() → 返回预测值数组
    #
    # predictions: numpy 数组, 长度 = 测试集行数 (约 279只 × 484天 ≈ 135000)
    #   每个值是模型对"该股票该天未来3天收益率"的预测
    #   值越大 → 模型越看好 → 应该买入
    #   值越小 → 模型越看空 → 应该卖出
    # ──────────────────────────────────────────────────────────
    print("\n[Step 4/5] 在测试集上生成交易信号 ...", flush=True)
    predictions = model.predict(dataset, Segment.TEST)

    # 获取测试集数据框架 (从 infer_df 中按日期范围筛选)
    df_test = dataset.fetch_infer(Segment.TEST)

    # 将预测值作为 "signal" 列加到 DataFrame
    df_test = df_test.with_columns(pl.Series(predictions).alias("signal"))

    # 只保留策略需要的三列: 日期、股票代码、信号值
    # ┌────────────┬──────────────┬──────────┐
    # │ datetime   │ vt_symbol    │ signal   │
    # ├────────────┼──────────────┼──────────┤
    # │ 2022-07-01 │ 000001.SZSE  │  0.0023  │  ← 模型预测平安银行会涨
    # │ 2022-07-01 │ 000002.SZSE  │ -0.0041  │  ← 模型预测万科会跌
    # │ 2022-07-01 │ 600519.SSE   │  0.0087  │  ← 模型预测茅台会涨
    # │ ...        │ ...          │ ...      │
    # └────────────┴──────────────┴──────────┘
    signal = df_test.select(["datetime", "vt_symbol", "signal"])
    print(f"  信号数据: {signal.shape[0]} 行", flush=True)

    # 持久化: 保存为 signal/hs300.parquet
    lab.save_signal("hs300", signal)

    # ══════════════════════════════════════════════════════════
    # Step 5: 运行策略回测
    # ══════════════════════════════════════════════════════════
    #
    # 回测 = 用历史数据模拟交易, 检验策略能否赚钱
    #
    # 整体流程:
    #   Day 1 (2022-07-01):
    #     1. 加载当天所有股票的 K 线
    #     2. 策略读取当天的 signal, 按 signal 从大到小排序
    #     3. 选出 Top 30 只股票作为目标持仓
    #     4. 对比当前持仓: 需要买哪些、卖哪些
    #     5. 生成委托单 (限价单)
    #     6. 引擎撮合成交
    #   Day 2 (2022-07-04):
    #     1. 结算: 计算昨天委托的盈亏
    #     2. 重复上述步骤
    #   ...
    #   Day 484 (2024-06-28):
    #     最后一天, 全部卖出, 统计总盈亏
    # ──────────────────────────────────────────────────────────
    print("\n[Step 5/5] 运行策略回测 ...", flush=True)

    # 导入策略类: EquityDemoStrategy (股票多头策略)
    # 策略逻辑详见 vnpy/alpha/strategy/strategies/equity_demo_strategy.py
    import vnpy.alpha.strategy.strategies.equity_demo_strategy as strat_module
    EquityDemoStrategy = strat_module.EquityDemoStrategy

    # ── 5a. 创建回测引擎 ──
    engine = BacktestingEngine(lab)

    # ── 5b. 设置回测参数 ──
    #
    # vt_symbols:  参与回测的股票池 (279 只)
    # interval:    K 线级别 (日线 = 每天调仓一次)
    # start/end:   回测时间段, 必须在测试集范围内
    # capital:     初始资金 1000 万元
    #
    # 引擎还会从 contract.json 读取每只股票的:
    #   long_rate:  买入手续费率 (0.03%)
    #   short_rate: 卖出手续费率 (0.13%, 含印花税)
    #   size:       合约乘数 (股票=1)
    #   pricetick:  最小价格变动 (0.01元)
    engine.set_parameters(
        vt_symbols=vt_symbols,
        interval=Interval.DAILY,
        start=datetime(2022, 7, 1),
        end=datetime(2024, 6, 30),
        capital=10_000_000,
    )

    # ── 5c. 添加策略 + 策略参数 ──
    #
    # EquityDemoStrategy 的核心逻辑:
    #   每天执行 on_bars() 回调:
    #     1. 获取当天所有股票的 signal, 从大到小排序
    #     2. 选出 signal 最高的 top_k=30 只作为"买入候选"
    #     3. 从当前持仓中, 找出 signal 最低的 n_drop=5 只列入"卖出清单"
    #     4. 但如果持有不足 min_days=5 天, 不卖 (避免频繁交易)
    #     5. 计算可用现金, 将资金平均分配给新买入的股票
    #     6. 调用 set_target() 设置每只股票的目标持仓量
    #     7. 调用 execute_trading() 自动生成买卖委托单
    #
    # 参数含义:
    #   top_k=30:       最多同时持有 30 只股票 (分散投资)
    #   n_drop=5:       每天卖出信号最差的 5 只 (淘汰末位)
    #   min_days=5:     至少持有 5 天才能卖 (减少交易频率、降低手续费)
    #   cash_ratio=0.95: 只使用 95% 的资金 (预留 5% 现金缓冲)
    #   min_volume=100:  最小交易单位 100 股 (A 股最小交易单位 = 1 手 = 100 股)
    setting = {
        "top_k": 30,
        "n_drop": 5,
        "min_days": 5,
        "cash_ratio": 0.95,
        "min_volume": 100,
    }
    engine.add_strategy(EquityDemoStrategy, setting, signal)
    # add_strategy() 做了什么:
    #   1. 实例化策略对象
    #   2. 将 signal DataFrame 绑定到策略
    #   3. 策略后续通过 self.get_signal() 获取当天的信号

    # ── 5d. 加载历史数据 ──
    # 从 lab 的 daily/ 目录读取每只股票在回测时间段内的 K 线
    # 存入 engine.history_data 字典: {(datetime, vt_symbol): BarData}
    engine.load_data()

    # ── 5e. 执行回测 ──
    # 按时间顺序逐日回放:
    #   对每个交易日:
    #     1. 收集当天所有股票的 BarData, 打包为 bars: Dict[vt_symbol, BarData]
    #     2. 撮合上一交易日的挂单 (cross_order): 如果收盘价触及委托价, 成交
    #     3. 调用 strategy.on_bars(bars): 策略根据信号决策, 生成新委托
    engine.run_backtesting()

    # ── 5f. 计算逐日盈亏 ──
    # 遍历所有成交记录 (trades), 按日期汇总:
    #   每天的持仓价值 + 现金 = 当天净值
    #   当天净值 - 昨天净值 = 当天盈亏
    engine.calculate_result()

    # ══════════════════════════════════════════════════════════
    # 绩效统计
    # ══════════════════════════════════════════════════════════
    #
    # calculate_statistics() 基于逐日盈亏数据计算:
    #
    # 输出 (dict):
    #   "total_return":    总收益率 = (结束资金/起始资金 - 1) × 100
    #   "annual_return":   年化收益 = 总收益率 / 交易日数 × 252
    #   "max_drawdown":    最大回撤 = 净值从峰值到谷底的最大跌幅
    #   "max_ddpercent":   百分比最大回撤 = 最大回撤 / 峰值净值
    #   "sharpe_ratio":    夏普比率 = (日均收益率 - 无风险利率) / 收益标准差 × √252
    #                      衡量"每承担1单位风险能获得多少收益"
    #   "return_drawdown_ratio": 收益回撤比 = 总盈亏 / 最大回撤
    #
    # 本次运行结果:
    #   总收益率:      12.47%
    #   年化收益:       6.19%
    #   最大回撤:     -27.80%
    #   Sharpe Ratio:   0.38
    #   收益回撤比:     0.36
    # ──────────────────────────────────────────────────────────
    print("\n绩效统计结果:", flush=True)
    print("-" * 50, flush=True)
    stats = engine.calculate_statistics()

    print("\n" + "=" * 60, flush=True)
    print("  研究完成!", flush=True)
    print(f"  数据保存在: {Path(LAB_PATH).resolve()}", flush=True)
    print("=" * 60, flush=True)


if __name__ == "__main__":
    main()
