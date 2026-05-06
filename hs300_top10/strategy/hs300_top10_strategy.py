"""
hs300_top10/strategy/hs300_top10_strategy.py

HS300 周度前十选股策略，继承 AlphaStrategy。

调仓逻辑
--------
- 周一（信号日）: 读取模型信号 → 选 top_k 只股票 → 平旧仓 + 开新仓
- 周二～周五: 仅做风控检查

风控规则
--------
- 硬止损: 持仓亏损 >= 3% → 平仓
- 追踪止盈: 浮盈 >= 3% 激活追踪，回撤 2% 退出
- 强制退出: 持仓满 4 个交易日后平仓

交易成本
--------
- 买入佣金 0.1%, 卖出佣金 0.1% + 印花税 0.1% (合计单边 ~0.2%)
- 滑点通过 price_add 控制 (默认 0.002 即 0.2%)
"""
from __future__ import annotations

from collections import defaultdict

import polars as pl

from vnpy.trader.object import BarData, TradeData
from vnpy.trader.constant import Direction
from vnpy.trader.utility import round_to

from vnpy.alpha import AlphaStrategy


class HS300Top10Strategy(AlphaStrategy):
    """HS300 周度前十选股策略"""

    # ── 策略参数（可通过 setting 覆盖） ──
    top_k: int = 10
    stop_loss_pct: float = 0.03          # 硬止损幅度（正值）
    tp_activate_pct: float = 0.03        # 追踪止盈激活阈值
    tp_trail_pct: float = 0.02           # 追踪止盈回撤退出阈值
    max_hold_days: int = 4               # 最大持仓交易日数
    cash_ratio: float = 0.95             # 现金使用比例
    min_volume: int = 100                # 最小交易单位（1 手）
    price_add: float = 0.002             # 委托价格偏移（用作滑点模拟）
    close_cost_rate: float = 0.002       # 估算卖出成本率（用于资金预估）

    def on_init(self) -> None:
        """策略初始化"""
        self.entry_prices: dict[str, float] = {}
        self.peak_prices: dict[str, float] = {}
        self.hold_days: dict[str, int] = {}
        self.tp_activated: dict[str, bool] = {}

        self.write_log("HS300Top10Strategy 初始化完成")

    def on_trade(self, trade: TradeData) -> None:
        """成交回调，更新持仓跟踪状态"""
        if trade.direction == Direction.LONG:
            self.entry_prices[trade.vt_symbol] = trade.price
            self.peak_prices[trade.vt_symbol] = trade.price
            self.hold_days[trade.vt_symbol] = 0
            self.tp_activated[trade.vt_symbol] = False
        elif trade.direction == Direction.SHORT:
            self._clear_tracking(trade.vt_symbol)

    def on_bars(self, bars: dict[str, BarData]) -> None:
        """K 线切片回调 — 每日执行一次"""
        dt = self._current_dt(bars)
        if dt is None:
            return

        weekday = dt.weekday()  # 0=Mon … 4=Fri

        pos_symbols = [s for s, p in self.pos_data.items() if p > 0]

        # ── 1. 更新持仓天数 & 峰值价格 ──
        for symbol in pos_symbols:
            self.hold_days[symbol] = self.hold_days.get(symbol, 0) + 1
            bar = bars.get(symbol)
            if bar and symbol in self.peak_prices:
                if bar.close_price > self.peak_prices[symbol]:
                    self.peak_prices[symbol] = bar.close_price

        # ── 2. 风控检查：止损 / 追踪止盈 / 超时 ──
        risk_sell: set[str] = set()
        for symbol in pos_symbols:
            bar = bars.get(symbol)
            if not bar or not bar.close_price:
                continue
            entry = self.entry_prices.get(symbol)
            if not entry:
                continue

            pnl_pct = (bar.close_price - entry) / entry

            if pnl_pct <= -self.stop_loss_pct:
                risk_sell.add(symbol)
                continue

            if pnl_pct >= self.tp_activate_pct:
                self.tp_activated[symbol] = True

            if self.tp_activated.get(symbol, False):
                peak = self.peak_prices.get(symbol, entry)
                dd = (bar.close_price - peak) / peak
                if dd <= -self.tp_trail_pct:
                    risk_sell.add(symbol)
                    continue

            if self.hold_days.get(symbol, 0) >= self.max_hold_days:
                risk_sell.add(symbol)

        for symbol in risk_sell:
            self.set_target(symbol, 0)

        # ── 3. 周一：读取信号 → 调仓 ──
        if weekday == 0:
            self._rebalance(bars, pos_symbols, risk_sell)

        # ── 4. 统一下单 ──
        self.execute_trading(bars, price_add=self.price_add)

    # ------------------------------------------------------------------
    # 内部方法
    # ------------------------------------------------------------------

    def _rebalance(
        self,
        bars: dict[str, BarData],
        pos_symbols: list[str],
        risk_sell: set[str],
    ) -> None:
        """周一调仓逻辑"""
        signal: pl.DataFrame = self.get_signal()
        if signal.is_empty():
            return

        signal = signal.sort("signal", descending=True)
        buy_candidates: list[str] = list(signal["vt_symbol"][: self.top_k])

        # 平掉不在新 top-k 且未被风控平仓的旧仓
        for symbol in pos_symbols:
            if symbol not in buy_candidates and symbol not in risk_sell:
                self.set_target(symbol, 0)

        # 预估可用资金（现金 + 待卖仓位回收）
        cash = self.get_cash_available()
        for symbol in pos_symbols:
            if self.get_target(symbol) == 0:
                bar = bars.get(symbol)
                if bar and bar.close_price:
                    cash += bar.close_price * self.get_pos(symbol) * (
                        1 - self.close_cost_rate
                    )

        # 需要新开仓的股票
        new_buys = [
            s for s in buy_candidates
            if self.get_pos(s) <= 0 or self.get_target(s) == 0
        ]

        if not new_buys:
            return

        buy_value = cash * self.cash_ratio / len(new_buys)

        for symbol in new_buys:
            bar = bars.get(symbol)
            if not bar or not bar.close_price:
                continue
            volume = round_to(buy_value / bar.close_price, self.min_volume)
            if volume > 0:
                self.set_target(symbol, volume)

    def _clear_tracking(self, vt_symbol: str) -> None:
        """清除单只股票的跟踪数据"""
        self.entry_prices.pop(vt_symbol, None)
        self.peak_prices.pop(vt_symbol, None)
        self.hold_days.pop(vt_symbol, None)
        self.tp_activated.pop(vt_symbol, None)

    @staticmethod
    def _current_dt(bars: dict[str, BarData]):
        """从 bars 中取出当前 datetime"""
        for bar in bars.values():
            return bar.datetime
        return None
