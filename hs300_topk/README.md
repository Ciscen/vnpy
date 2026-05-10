# HS300 Top-K 周度选股策略与量化流水线

基于 vnpy Alpha 框架构建的端到端量化选股与回测流水线。策略使用 **XGBoost 树模型** 结合 **Alpha158 量价因子**，在 CSI800 (沪深300 + 中证500) 宇宙上滚动训练，并在实盘中从沪深300 (HS300) 成分股中选出概率最高的 Top-K 只股票进行等权配置。

为了彻底消除**幸存者偏差**，数据流严格引入了 BaoStock 的 Point-in-Time (PIT) 历史快照；为避免未来信息泄漏，所有因子和标签严格后视（Backward-looking），并在独立样本外（OOS）窗口进行验证。

---

## 🚀 核心策略方案 (V1.4 vs V1.5)

在经历了 2022-2026 年完整的牛熊周期回测与打磨后，项目沉淀了两个核心的生产级版本。详细的版本选型对比请参考：[版本评估与选型文档 (version_selection.md)](docs/version_selection.md)。

### 1. V1.5 Ensemble (全天候稳健基线 - 当前默认)
这是当前流水线默认推荐运行的版本，完美平衡了防守与进攻。
- **核心机制 (Double Model Ensemble)**：
  - 训练两个物理隔离的模型：**Alpha 模型**（预测相对基准的 `excess_return` 超额收益）和 **Beta 模型**（预测绝对收益的 `high_touch` 触及概率）。
  - **动态 Regime 切换**：通过沪深300指数的 **MA60 均线**作为宏观状态判别器。当大盘处于 MA60 之上（牛市）时使用 Beta 模型获取高弹性；当大盘低于 MA60（熊市）时平滑切换至 Alpha 模型进行防御。
- **效果表现 (全周期 2022-2026)**：
  - **年化收益率**：21.81% (总收益 95.07%)
  - **最大回撤**：-35.62% (在 22-23 年单边熊市中大幅跑赢基准)
  - **Sharpe 比率**：0.70

### 2. V1.4 + Lag-3 (激进纯多头基线)
作为纯多头牛市高弹性的标杆，证明了量价动量在上升趋势中的绝对获利能力。
- **核心特征**：
  - 在 Alpha158 基础上，拼接入场前 3 天的全量因子（Lag-3），特征维度扩充至 **632 维**。
  - 使用保守的 `friday_close` 标签（周五收盘相对周二开盘上涨 3%）。
  - **高集中度**：Top-5 持仓。
- **效果表现**：在牛市阶段（如 2024.05-2026.04）爆发力极强（年化高达 86.4%，Sharpe 2.68），但在熊市中存在超过 -50% 的单边敞口回撤。

---

## 📂 项目结构与导航

```text
hs300_topk/
├── README.md               # 本文档
├── pipeline_config.py      # 流水线全局配置（日期、资金、路径基准）
├── run_pipeline.py         # 🚀 统一调度入口（下载 -> 训练 -> 回测 -> 报告）
├── run_oos_validation.py   # 独立的样本外严格验证测试脚本
│
├── docs/                   # 核心文档库
│   ├── version_selection.md     # V1.4 vs V1.5 版本选型与废弃理由
│   └── v1.4_series_tuning.md    # 策略演进与网格调参记录
│
├── data/                   # 数据层
│   ├── downloader.py       # AKShare下载 + BaoStock PIT 快照补充与回退兜底
│   └── loader.py           # 数据加载与 AlphaLab 格式转换
│
├── features/               # 特征工程层
│   ├── engineer.py         # 继承 Qlib Alpha158 因子计算与滞后拼接
│   └── labeler.py          # 周度标签生成 (绝对收益、超额收益等)
│
├── model/                  # 模型与训练层
│   ├── rolling_trainer.py  # XGBoost 月度 Walk-Forward 滚动训练
│   └── predictor.py        # 模型预测与信号持久化
│
├── strategy/               # 策略执行层
│   ├── config.py           # 各版本超参数定义 (V1.4, V1.4R, V1.5 等)
│   └── hs300_topk_strategy.py   # VnPy AlphaStrategy 实盘执行逻辑 (风控/调仓)
│
├── live/                   # 🔴 实盘与仿真环境
│   ├── bot.py              # 实盘自动交易机器人
│   └── feishu.py           # 飞书消息卡片与文档同步模块
│
└── output/                 # 📊 回测报告产出目录（运行后生成）
```

---

## 📈 输出与仪表盘 (Dashboards)

每次执行 `run_pipeline.py` 后，`output/` 目录下会自动生成极具交互性的富文本报告。生成的产出因 `--config` 和 `--weekly-label` 参数不同而放在独立的子目录中（例如 `output/v1.5_ensemble/`）。

核心产物包括：
- **`dashboard.html` / `strategy_overview.html`**：高度集成的交互式策略仪表盘。内置了：
  - **基线比对的净值图 (Equity Curve)**：强制与 HS300 基准比对。
  - **盈亏分布与回撤分析**。
  - **核心指标表**（年化、Sharpe、胜率、回撤及收益回撤比等）。
- **`daily_pnl.csv`**：逐日盯市盈亏、手续费、资金曲线数据。
- **`trades.csv`**：包含每一次开平仓的触发原因（如 `stop_loss`, `trailing_tp`, `max_hold`, `signal_buy`）。
- **`stock_details/*.html`**：10只高频交易股票的带买卖点标注的 K线图。

---

## 🛠️ 如何使用 (Usage)

### 1. 运行核心 V1.5 Ensemble 流水线 (包含数据下载、双模型训练、全周期回测)

```bash
python -m hs300_topk.run_pipeline --config v1.5 --weekly-label ensemble --lag-days 3
```

> **注意**：首次运行会下载接近 900 只股票近 10 年的数据，并从 BaoStock 获取成分股快照，耗时较长。后续运行可加上 `--skip-download` 跳过下载阶段，或利用缓存。

### 2. 常用命令行参数

- `--backtest-only`: 仅执行回测和报告生成，跳过模型训练（直接复用 `lab/hs300/signal/` 下已经跑好的预测信号）。
- `--oos-validate`: 将回测区间强制定向到独立的样本外时间窗（如 25.05 - 26.04），并将输出目录后缀标记为 `_oos`，用于最终的公平校验。
- `--force-download`: 忽略本地 Parquet 缓存，强制从网络拉取全量数据。

### 3. 多版本对比测试
如果你想横向对比 V1.4, V1.4R, V1.5 等多种参数配置：
```bash
python -m hs300_topk.run_pipeline --config compare --weekly-label friday_close
```

---

## 🛡️ 架构与风控规范 (AI Rules)

本项目强制遵守极高的量化标准，所有的规范已固化在项目根目录的 `GEMINI.md` 以及 `.cursor/rules/quant_guidelines.mdc` 中：

1. **零幸存者偏差**：成分股过滤强制依赖于时点（Point-in-Time）快照。
2. **严禁未来函数**：任何因子和 Label 的预处理必须按时间截断。
3. **拒绝单边行情测谎**：回测时间轴必须覆盖完整的熊市与牛市。
4. **拒绝单一分布混淆**：宏观状态应对交给 Ensemble 解决，不滥用单一树模型去处理相互矛盾的目标。
