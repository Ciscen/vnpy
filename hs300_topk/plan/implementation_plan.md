# HS300 周度前十选股模型设计方案（更新版）

## 目标描述

基于 `demo` 目录中更新至 **2026‑04‑30** 的 HS300 成分股历史数据（最近 10 年），实现一个完整的端到端管线：

1. **标签定义**：以 **周一下午 14:00** 为基准时间，若在随后一周的任意交易日收盘价相对基准价上涨 **≥3%**，则记为正例（1），否则记为负例（0）。
2. **模型训练**：采用 **滚动（walk‑forward）训练**——每次使用前 8 年数据训练模型，后 2 年数据用于回测，期间每月（或每周）重新训练。
3. **选股逻辑**：预测每支股票在基准时点的 3% 上涨概率，取概率最高的 **10 只** 股票做等权多头。
4. **仓位管理**：持仓期间若收益达到 **+3%** 即止盈，若亏损达到 **‑5%** 即止损。
5. **代码组织**：所有新增代码统一放在项目根目录下的新文件夹 `hs300_topk` 中，保持与原项目结构相互独立。

---

## 需用户确认的关键决策

> [!IMPORTANT]
> 以下事项请确认后方可继续实现。
>
> 1. **模型族选择** – 我们提供三种候选模型：
>    - **Lasso 回归**（线性、解释性强、对特征稀疏化有帮助）
>    - **GradientBoostingClassifier**（基于决策树的集成模型，鲁棒性好，适合中等规模数据）
>    - **XGBoostClassifier**（高效的梯度提升实现，常在金融预测中表现最佳）
>    请确认您希望使用哪些模型（可全部使用作对比），以及是否需要安装 XGBoost（会自动加入依赖）。
> 2. **特征集合** – 当前计划使用 MA、EMA、RSI、MACD、布林带宽、日收益等技术指标。若需要加入其他特征（如市值、行业、基本面因子），请说明。
> 3. **滚动训练频率** – 默认每月滚动一次（使用前 8 年完整数据训练，随后一个月的交易日做预测），如需改为每周或更细粒度，请告知。
> 4. **回测指标** – 建议包括累计收益、年化 Sharpe、最大回撤、胜率、平均持仓天数。若需增删请说明。
> 5. **模型持久化路径** – 默认保存至 `hs300_topk/models/`，文件名依据模型自动命名（如 `lasso.pkl`）。如有其他需求请说明。

---

## 待解决的问题

> [!WARNING]
> - **数据获取**：需要从公开数据源（如 TuShare、Wind、Yahoo）下载 HS300 成分股最近 10 年（2016‑01‑01 至 2026‑04‑30）的日线 OHLCV，并保存为 `hs300_topk/data/hs300.csv`。请确认是否使用 TuShare（需 API token）或其他渠道。
> - **标签实现细节**：计算基准价为 **周一 14:00** 的最新可用收盘价（若当天无交易则取前一日收盘价），随后 5 个交易日内最高收盘价是否 ≥ 基准价 × 1.03。若满足则标记 1，否则 0。
> - **缺失值处理**：已决定直接剔除含缺失值的行。
> - **止盈/止损触发**：在策略中加入 `stop_profit`（3%）和 `stop_loss`（‑5%）的条件检查，实现自动平仓。
> - **代码目录**：所有新建文件将放在 `hs300_topk/` 目录下，包括 `data/`, `features/`, `model/`, `strategy/`, `backtest/` 子文件夹。

---

## 方案概览（文件结构与改动）

```
hs300_topk/
│   README.md               # 项目说明与使用方法
│   requirements.txt        # 依赖列表（pandas, scikit-learn, xgboost, tushare 等）
│
├─ data/
│   │   hs300.csv           # 最新 10 年日线数据（由脚本自动下载）
│   │   download_data.py    # 下载并保存数据的实用脚本
│   │   loader.py           # CSV 读取与预处理（前向填充后剔除缺失）
│
├─ features/
│   │   engineer.py         # 技术指标特征构造（同前计划）
│   │   labeler.py          # 依据 3% 阈值生成二分类标签
│
├─ model/
│   │   trainer.py          # 支持 Lasso、GB、XGB 的训练与持久化
│   │   predictor.py        # 预测并输出概率
│   │   rolling_trainer.py  # 滚动训练调度（每月/每周）
│
├─ strategy/
│   │   hs300_topk_strategy.py   # vnpy 策略实现（选股、止盈、止损）
│
├─ backtest/
│   │   run_backtest.py     # 入口脚本，使用 vnpy BacktestingEngine
│   │   evaluation.py       # 回测结果统计与可视化
│
└─ tests/
    │   test_loader.py
    │   test_features.py
    │   test_trainer.py
    │   test_strategy.py
```

### 关键模块说明

#### 1. `data/download_data.py`
- 利用 TuShare（或其他）API 拉取 2016‑01‑01~2026‑04‑30 的日线数据。
- 自动过滤停牌、除权除息，保存为 `hs300.csv`。
- 若用户无 TuShare token，可改用 `yfinance` 抓取指数成分并单独下载。

#### 2. `data/loader.py`
```python
def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH, parse_dates=["date"])
    # 保留必要列，设置 MultiIndex (symbol, date)
    df.set_index(["symbol", "date"], inplace=True)
    df.sort_index(inplace=True)
    # 删除含 NaN 的行（缺失值直接剔除）
    df.dropna(inplace=True)
    return df
```

#### 3. `features/engineer.py`
- 与之前计划相同，但增加 `lookback` 参数，可灵活调整。

#### 4. `features/labeler.py`
```python
def generate_labels(df: pd.DataFrame, horizon_days: int = 5, rise_thresh: float = 0.03) -> pd.Series:
    # 对每只股票，找到每周一 14:00 的最近收盘价
    # 计算后 horizon_days 天内最高收盘价
    # 若最高价 >= 基准价 * (1 + rise_thresh) -> 1 else 0
    labels = []
    for symbol, group in df.groupby(level=0):
        # 取交易日的星期信息
        grp = group.reset_index()
        grp['weekday'] = grp['date'].dt.weekday
        monday_afternoon = grp[(grp['weekday'] == 0) & (grp['date'].dt.time >= pd.to_timedelta('14:00:00'))]
        # 若当天无交易，则使用前一交易日收盘价
        for _, row in monday_afternoon.iterrows():
            base_price = row['close']
            start_idx = grp[grp['date'] > row['date']].index[0]
            horizon_slice = grp.loc[start_idx:start_idx + horizon_days]
            max_price = horizon_slice['close'].max()
            label = int(max_price >= base_price * (1 + rise_thresh))
            labels.append((symbol, row['date'], label))
    lbl_df = pd.DataFrame(labels, columns=['symbol', 'date', 'label'])
    lbl_df.set_index(['symbol', 'date'], inplace=True)
    return lbl_df['label']
```
> 注：实现细节将在代码中优化，以免遍历过慢。

#### 5. `model/trainer.py`
- 与之前相同，返回模型保存路径。
- 对每种模型提供默认超参数，可在配置文件中覆盖。

#### 6. `model/rolling_trainer.py`
- 按月滑动窗口划分训练/验证集。
- 读取 `loader.load_data()` → 划分前 8 年为训练集，后 2 年为回测集。
- 每个月结束时重新训练模型并保存，文件名带日期前缀（`lasso_2023-04.pkl`）。

#### 7. `strategy/hs300_topk_strategy.py`
- 在 `on_bar` 中检查当前时间是否 **周一 14:00**（使用 `bar.datetime`）
- 若满足，则加载最新模型（最近一次滚动训练生成的文件），计算特征并预测概率。
- 选取前 10，执行等权买入。
- 每根 K 线实时检查持仓收益率，若 >= 3% 止盈，若 <= -5% 止损，使用 `sell` 平仓。
- 持仓周期为 **一周**，周一 14:00 重新调仓，自动平掉上一周未止盈/止损的仓位。

#### 8. `backtest/run_backtest.py`
- 使用 vnpy `BacktestingEngine`，时间区间 `2018-01-01`（训练集后两年）至 `2025-12-31`（最近两年）
- 配置滑点、手续费等参数。
- 运行后调用 `evaluation.plot_equity_curve()`、`evaluation.print_metrics()`。

#### 9. `tests/`
- 为每个模块编写单元测试，确保数据加载、标签生成、特征无 NaN、模型训练成功、策略止盈止损逻辑正确。

---

## 验证计划

### 自动化测试
- **单元测试**：`loader`, `engineer`, `labeler`, `trainer`, `predictor`, `rolling_trainer`。
- **策略测试**：使用 vnpy 模拟行情，验证在满足 3% 涨幅时止盈、跌破‑5% 时止损。
- **回测完整性**：执行 `run_backtest.py`，检查返回的 `result` 包含累计收益、年化 Sharpe、最大回撤等指标，并生成权益曲线图。

### 人工检查
- 读取生成的 `hs300.csv`，确认数据覆盖 **2016‑01‑01 至 2026‑04‑30**。
- 手动查看若干周的基准价与最高价，确认标签逻辑符合预期。
- 观察回测报告，若累计收益为正且风险可控，即视为通过。

---

> 请在确认模型族、特征、滚动训练频率、回测指标及数据获取方式后批准此方案，我将根据确认生成任务清单并开始实现代码。
