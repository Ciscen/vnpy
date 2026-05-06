# HS300 Top-10 周度选股策略

基于 vnpy Alpha 框架，使用 **XGBoost 二分类模型** 预测沪深 300 成分股未来一周上涨概率，每周选出概率最高的 10 只股票进行等权多头配置。

## 项目结构

```
hs300_top10/
│   README.md               # 本文件
│   requirements.txt        # Python 依赖
│   run_pipeline.py         # 统一调度脚本（推荐入口）
│
├─ data/                    # 数据层
│   │   download_data.py    # 独立数据下载脚本 (akshare, 备用)
│   │   loader.py           # AlphaLab 数据加载封装
│
├─ features/                # 特征工程
│   │   engineer.py         # HS300Top10Dataset (继承 Alpha158, 158 个因子)
│   │   labeler.py          # 周度二分类标签生成
│
├─ model/                   # 模型层
│   │   trainer.py          # XgbClassifierModel (XGBoost 二分类)
│   │   predictor.py        # 信号生成
│   │   rolling_trainer.py  # 月度滚动训练流水线
│
├─ strategy/                # 策略层
│   │   hs300_top10_strategy.py   # vnpy AlphaStrategy 实现
│
├─ backtest/                # 回测层
│   │   run_backtest.py     # 回测入口脚本（不含下载）
│   │   evaluation.py       # 绩效统计、可视化与报告导出
│
├─ output/                  # 回测报告输出（自动生成）
│   │   statistics.json     # 绩效统计指标
│   │   daily_pnl.csv       # 逐日盈亏明细
│   │   trades.csv          # 全部成交记录
│   │   equity_curve.html   # 权益曲线图
│   │   daily_pnl_chart.html# 月度盈亏分布图
│
└─ plan/                    # 设计文档
    │   README.md
    │   implementation_plan.md
    │   task.md
```

## 核心参数

| 参数 | 值 | 说明 |
|------|------|------|
| 数据区间 | 2016-04-30 ~ 2026-04-30 | 最近 10 年 |
| 回测区间 | 2024-05-01 ~ 2026-04-30 | 最后 2 年 |
| 特征基准 | 周一收盘 | Alpha158 因子截止周一 |
| 入场时机 | 周二开盘 | 周一下单，周二成交 |
| 标签阈值 | 5% | 周二~周五最高价 >= 周二开盘 x 1.05 |
| 硬止损 | -3% | 持仓亏损达 3% 立即平仓 |
| 追踪止盈 | +3% 激活, -2% 退出 | 浮盈 3% 后追踪，回撤 2% 退出 |
| 强制退出 | 4 个交易日 | 最长持仓周期 |
| 交易成本 | 0.2% 佣金 + 0.2% 滑点 | 单边合计约 0.4% |

## 快速开始

### 1. 安装依赖

```bash
pip install -r hs300_top10/requirements.txt
```

### 2. 一键运行（推荐）

```bash
python -m hs300_top10.run_pipeline
```

该命令自动执行完整流水线：
1. 下载沪深 300 成分股日线数据（自动检查缓存，已有则跳过）
2. 计算 Alpha158 因子特征 + 周度标签
3. 逐月滚动训练 XGBoost 模型
4. 生成信号 -> 策略回测
5. 输出绩效统计 + 导出报告文件

### 3. 命令行选项

```bash
# 强制重新下载全部数据
python -m hs300_top10.run_pipeline --force-download

# 跳过数据下载（使用已有 lab 数据）
python -m hs300_top10.run_pipeline --skip-download

# 仅回测（使用上次训练的信号缓存）
python -m hs300_top10.run_pipeline --backtest-only
```

### 4. 仅运行回测（不含下载）

```bash
python -m hs300_top10.backtest.run_backtest
```

## 回测报告

执行完成后，`hs300_top10/output/` 目录下会自动生成：

| 文件 | 内容 |
|------|------|
| `statistics.json` | 全部绩效指标（Sharpe、年化收益、最大回撤等） |
| `daily_pnl.csv` | 逐日盈亏、余额、回撤等明细数据 |
| `trades.csv` | 每笔成交记录（时间、股票、方向、价格、数量） |
| `equity_curve.html` | 交互式权益曲线图（Plotly，可缩放） |
| `daily_pnl_chart.html` | 月度盈亏柱状图 |

## 架构说明

### 信号驱动模式

```
日线数据 -> Alpha158 因子 -> 周度标签 -> XGBoost 滚动训练 -> 信号 DataFrame
                                                              |
                                          BacktestingEngine <- 策略消费信号
                                                              |
                                                          绩效统计 + 报告
```

- **离线阶段**: 模型训练和信号生成在回测前完成
- **在线阶段**: 策略在 `on_bars()` 中消费预计算信号，执行调仓和风控

### 滚动训练

采用 walk-forward 方法，避免未来信息泄露：
- 每月初用截止上月底的历史数据训练模型
- 用该模型预测当月的周一信号
- 训练窗口最长 8 年

## 自定义参数

策略参数可在 `run_pipeline.py` 的 `phase_backtest()` 中调整：

```python
setting = {
    "top_k": 10,            # 选股数量
    "stop_loss_pct": 0.03,  # 止损幅度
    "tp_activate_pct": 0.03,# 追踪止盈激活阈值
    "tp_trail_pct": 0.02,   # 追踪止盈回撤退出
    "max_hold_days": 4,     # 最大持仓天数
    "cash_ratio": 0.95,     # 现金使用比例
}
```

模型超参数可在 `rolling_trainer.py` 中的 XGBClassifier 初始化处调整。
