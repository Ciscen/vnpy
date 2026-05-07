"""
hs300_top10/strategy/config.py

策略配置体系：将所有可调参数集中管理，支持版本对比。
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path


@dataclass
class StrategyConfig:
    """策略全局配置，覆盖选股、风控、执行三个层面。"""

    # ── 版本标识 ──
    version: str = "v1.0"
    description: str = "基线版本"

    # ── 选股参数 ──
    top_k: int = 10
    min_signal_prob: float = 0.0       # 最低信号概率阈值（0=不过滤）
    dynamic_k: bool = False            # 是否根据信号强度动态调整 K
    dynamic_k_min: int = 3             # 动态 K 最小值
    dynamic_k_prob_threshold: float = 0.35  # 概率低于此值时缩减 K
    weight_by_signal: bool = False     # 是否按概率加权分配仓位

    # ── 风控参数 ──
    stop_loss_pct: float = 0.03        # 硬止损幅度
    tp_activate_pct: float = 0.03      # 追踪止盈激活阈值
    tp_trail_pct: float = 0.02         # 追踪止盈回撤退出阈值
    max_hold_days: int = 4             # 最大持仓交易日

    use_atr_stop: bool = False         # 是否使用 ATR 自适应止损
    atr_stop_multiplier: float = 2.0   # ATR 止损倍数
    atr_stop_min: float = 0.02         # ATR 止损下限
    atr_stop_max: float = 0.06         # ATR 止损上限

    use_market_filter: bool = False    # 是否使用市场状态过滤
    market_ma_period: int = 20         # 市场均线周期（日）
    market_benchmark: str = "000300.SSE"  # 基准指数

    # ── 执行参数 ──
    cash_ratio: float = 0.95           # 现金使用比例
    min_volume: int = 100              # 最小交易单位
    price_add: float = 0.002           # 滑点
    close_cost_rate: float = 0.002     # 估算卖出成本

    smooth_rebalance: bool = False     # 调仓平滑：已持仓 & 仍在信号中的不动
    max_replace_ratio: float = 1.0     # 单次最大换仓比例（1.0=全换）

    # ── V1.2 组合级风控 ──
    portfolio_daily_loss_limit: float = 0.0   # 单日最大亏损限制（0=禁用）
    cooldown_days: int = 0                    # 触发后冷却天数
    min_signal_spread: float = 0.0            # top1-topK 概率差距最小值

    # ── V1.3 风控增强 ──
    conditional_hold_extend: bool = False
    hold_extend_min_pnl: float = 0.03
    hold_extend_days: int = 2
    absolute_stop_cap: float = 0.0
    profit_lock_threshold: float = 0.0
    profit_lock_trail_pct: float = 0.015
    # 动量确认入场
    momentum_filter: bool = False
    momentum_lookback: int = 3
    momentum_min_return: float = -0.03
    # 个股冷却
    stock_cooldown_days: int = 0             # 止损后个股冷却天数（0=禁用）
    # 调仓周期
    rebalance_period: int = 1                # 1=每周, 2=双周

    # ── V2.0 信号驱动持仓 ──
    daily_signal: bool = False               # 是否使用日频信号模式
    pool_size: int = 30                      # 月度候选池大小
    signal_horizon: int = 3                  # 信号预测窗口（天）
    entry_threshold: float = 0.50            # 新建仓最低信号概率
    renew_threshold: float = 0.40            # 到期续仓最低信号概率
    max_renewals: int = 3                    # 最大续期次数

    # ── 模型参数 ──
    xgb_n_estimators: int = 500
    xgb_max_depth: int = 6
    xgb_learning_rate: float = 0.05
    xgb_subsample: float = 0.8
    xgb_colsample_bytree: float = 0.8
    xgb_early_stopping: int = 30
    train_years: int = 8

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self, path: str | Path) -> None:
        Path(path).write_text(
            json.dumps(self.to_dict(), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    @classmethod
    def from_json(cls, path: str | Path) -> StrategyConfig:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})

    def diff(self, other: StrategyConfig) -> dict:
        """返回两个配置之间的差异"""
        d1, d2 = self.to_dict(), other.to_dict()
        return {k: (d1[k], d2[k]) for k in d1 if d1[k] != d2[k]}


# ── 预设配置 ──

BASELINE_V10 = StrategyConfig(
    version="v1.0",
    description="基线版本：Alpha158 + XGBoost + 固定止损止盈",
)

OPTIMIZED_V11 = StrategyConfig(
    version="v1.1",
    description="V1.1: 调仓平滑 + ATR自适应止损 + 动态K + 概率加权",
    smooth_rebalance=True,
    max_replace_ratio=0.7,
    use_atr_stop=True,
    atr_stop_multiplier=2.0,
    atr_stop_min=0.02,
    atr_stop_max=0.05,
    use_market_filter=False,
    dynamic_k=True,
    dynamic_k_min=5,
    dynamic_k_prob_threshold=0.30,
    weight_by_signal=True,
    min_signal_prob=0.15,
    stop_loss_pct=0.04,
    tp_activate_pct=0.04,
    tp_trail_pct=0.02,
    max_hold_days=5,
)

OPTIMIZED_V12 = StrategyConfig(
    version="v1.2",
    description="V1.2: V1.1 + 集中持仓(top8) + 更低换手",
    smooth_rebalance=True,
    max_replace_ratio=0.7,
    use_atr_stop=True,
    atr_stop_multiplier=2.0,
    atr_stop_min=0.02,
    atr_stop_max=0.05,
    dynamic_k=True,
    dynamic_k_min=4,
    dynamic_k_prob_threshold=0.30,
    weight_by_signal=True,
    min_signal_prob=0.15,
    stop_loss_pct=0.04,
    tp_activate_pct=0.04,
    tp_trail_pct=0.02,
    max_hold_days=5,
    top_k=8,
)

OPTIMIZED_V13 = StrategyConfig(
    version="v1.3",
    description="V1.3: V1.2 + 个股止损冷却(10天)",
    smooth_rebalance=True,
    max_replace_ratio=0.7,
    use_atr_stop=True,
    atr_stop_multiplier=2.0,
    atr_stop_min=0.02,
    atr_stop_max=0.05,
    dynamic_k=True,
    dynamic_k_min=4,
    dynamic_k_prob_threshold=0.30,
    weight_by_signal=True,
    min_signal_prob=0.15,
    stop_loss_pct=0.04,
    tp_activate_pct=0.04,
    tp_trail_pct=0.02,
    max_hold_days=5,
    top_k=8,
    stock_cooldown_days=10,
)

OPTIMIZED_V14 = StrategyConfig(
    version="v1.4",
    description="V1.4: V1.3 + 集中持仓(top5)",
    smooth_rebalance=True,
    max_replace_ratio=0.7,
    use_atr_stop=True,
    atr_stop_multiplier=2.0,
    atr_stop_min=0.02,
    atr_stop_max=0.05,
    dynamic_k=True,
    dynamic_k_min=3,
    dynamic_k_prob_threshold=0.30,
    weight_by_signal=True,
    min_signal_prob=0.15,
    stop_loss_pct=0.04,
    tp_activate_pct=0.04,
    tp_trail_pct=0.02,
    max_hold_days=5,
    top_k=5,
    stock_cooldown_days=10,
)

OPTIMIZED_V20 = StrategyConfig(
    version="v2.0",
    description="V2.0: 月度选池 + 信号驱动持仓（日频标签3日/2%）",
    daily_signal=True,
    pool_size=30,
    signal_horizon=3,
    entry_threshold=0.50,
    renew_threshold=0.40,
    max_renewals=3,
    top_k=10,
    stop_loss_pct=0.03,
    tp_activate_pct=0.03,
    tp_trail_pct=0.02,
    max_hold_days=15,
    smooth_rebalance=False,
    use_atr_stop=True,
    atr_stop_multiplier=2.0,
    atr_stop_min=0.02,
    atr_stop_max=0.05,
    absolute_stop_cap=0.10,
    weight_by_signal=True,
)

OPTIMIZED_V21 = StrategyConfig(
    version="v2.1",
    description="V2.1: V2.0参数优化（收紧入场+延长持仓+快止损）",
    daily_signal=True,
    pool_size=20,
    signal_horizon=5,
    entry_threshold=0.60,
    renew_threshold=0.45,
    max_renewals=4,
    top_k=10,
    stop_loss_pct=0.04,
    tp_activate_pct=0.04,
    tp_trail_pct=0.02,
    max_hold_days=20,
    smooth_rebalance=False,
    use_atr_stop=True,
    atr_stop_multiplier=2.0,
    atr_stop_min=0.02,
    atr_stop_max=0.05,
    absolute_stop_cap=0.07,
    weight_by_signal=True,
)

OPTIMIZED_V22 = StrategyConfig(
    version="v2.2",
    description="V2.2: V2.1 + 保守组合风控（日亏5%清仓+2天冷却）",
    daily_signal=True,
    pool_size=20,
    signal_horizon=5,
    entry_threshold=0.60,
    renew_threshold=0.45,
    max_renewals=4,
    top_k=10,
    stop_loss_pct=0.04,
    tp_activate_pct=0.04,
    tp_trail_pct=0.02,
    max_hold_days=20,
    smooth_rebalance=False,
    use_atr_stop=True,
    atr_stop_multiplier=2.0,
    atr_stop_min=0.02,
    atr_stop_max=0.05,
    absolute_stop_cap=0.07,
    weight_by_signal=True,
    portfolio_daily_loss_limit=0.05,
    cooldown_days=2,
)
