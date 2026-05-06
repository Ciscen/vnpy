"""
hs300_top10/strategy/hs300_top10_strategy.py

HS300 周度前十选股策略，继承 AlphaStrategy。

调仓逻辑
--------
- 周一（信号日）: 读取模型信号 → 选 top_k 只股票 → 平旧仓 + 开新仓
- 周二～周五: 仅做风控检查

风控规则
--------
- 硬止损: 持仓亏损 >= stop_loss_pct → 平仓
- 追踪止盈: 浮盈 >= tp_activate_pct 激活追踪，回撤 tp_trail_pct 退出
- ATR 自适应止损（可选）: 止损幅度 = atr_multiplier × ATR / entry_price
- 市场状态过滤（可选）: 指数低于 MA 时暂停开仓
- 强制退出: 持仓满 max_hold_days 后平仓

交易成本
--------
- 买入佣金 0.1%, 卖出佣金 0.1% + 印花税 0.1% (合计单边 ~0.2%)
- 滑点通过 price_add 控制 (默认 0.002 即 0.2%)
"""
from __future__ import annotations

import polars as pl

from vnpy.trader.object import BarData, TradeData
from vnpy.trader.constant import Direction
from vnpy.trader.utility import round_to

from vnpy.alpha import AlphaStrategy


class HS300Top10Strategy(AlphaStrategy):
    """HS300 周度前十选股策略"""

    # ── 策略参数（可通过 setting 覆盖） ──
    top_k: int = 10
    stop_loss_pct: float = 0.03
    tp_activate_pct: float = 0.03
    tp_trail_pct: float = 0.02
    max_hold_days: int = 4
    cash_ratio: float = 0.95
    min_volume: int = 100
    price_add: float = 0.002
    close_cost_rate: float = 0.002

    # V1.1 新增参数
    smooth_rebalance: bool = False
    max_replace_ratio: float = 1.0
    use_atr_stop: bool = False
    atr_stop_multiplier: float = 2.0
    atr_stop_min: float = 0.02
    atr_stop_max: float = 0.06
    use_market_filter: bool = False
    market_ma_period: int = 20
    market_benchmark: str = "000300.SSE"
    dynamic_k: bool = False
    dynamic_k_min: int = 3
    dynamic_k_prob_threshold: float = 0.35
    weight_by_signal: bool = False
    min_signal_prob: float = 0.0

    def on_init(self) -> None:
        """策略初始化"""
        self.entry_prices: dict[str, float] = {}
        self.peak_prices: dict[str, float] = {}
        self.hold_days: dict[str, int] = {}
        self.tp_activated: dict[str, bool] = {}

        self._atr_cache: dict[str, float] = {}
        self._price_history: dict[str, list[float]] = {}
        self._benchmark_closes: list[float] = []
        self._market_ok: bool = True

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

        weekday = dt.weekday()
        pos_symbols = [s for s, p in self.pos_data.items() if p > 0]

        # ── 0. 更新 ATR 和市场状态 ──
        if self.use_atr_stop:
            self._update_atr(bars)

        if self.use_market_filter:
            self._update_market_state(bars)

        # ── 1. 更新持仓天数 & 峰值价格 ──
        for symbol in pos_symbols:
            self.hold_days[symbol] = self.hold_days.get(symbol, 0) + 1
            bar = bars.get(symbol)
            if bar and symbol in self.peak_prices:
                if bar.close_price > self.peak_prices[symbol]:
                    self.peak_prices[symbol] = bar.close_price

        # ── 2. 风控检查 ──
        risk_sell: set[str] = set()
        for symbol in pos_symbols:
            bar = bars.get(symbol)
            if not bar or not bar.close_price:
                continue
            entry = self.entry_prices.get(symbol)
            if not entry:
                continue

            pnl_pct = (bar.close_price - entry) / entry

            effective_sl = self._get_stop_loss(symbol, entry)
            if pnl_pct <= -effective_sl:
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
    # 风控辅助
    # ------------------------------------------------------------------

    def _get_stop_loss(self, symbol: str, entry_price: float) -> float:
        """获取止损幅度（支持 ATR 自适应）"""
        if not self.use_atr_stop:
            return self.stop_loss_pct

        atr = self._atr_cache.get(symbol, 0)
        if atr <= 0 or entry_price <= 0:
            return self.stop_loss_pct

        atr_sl = self.atr_stop_multiplier * atr / entry_price
        return max(self.atr_stop_min, min(self.atr_stop_max, atr_sl))

    def _update_atr(self, bars: dict[str, BarData]) -> None:
        """更新 ATR 缓存（14 日 ATR）"""
        for symbol, bar in bars.items():
            history = self._price_history.setdefault(symbol, [])
            history.append((bar.high_price, bar.low_price, bar.close_price))
            if len(history) > 15:
                history.pop(0)

            if len(history) >= 2:
                trs = []
                for i in range(1, len(history)):
                    h, l, _ = history[i]
                    prev_c = history[i - 1][2]
                    tr = max(h - l, abs(h - prev_c), abs(l - prev_c))
                    trs.append(tr)
                self._atr_cache[symbol] = sum(trs) / len(trs)

    def _update_market_state(self, bars: dict[str, BarData]) -> None:
        """更新市场状态（指数是否在 MA 上方）"""
        bench_bar = bars.get(self.market_benchmark)
        if bench_bar:
            self._benchmark_closes.append(bench_bar.close_price)
            if len(self._benchmark_closes) > self.market_ma_period + 1:
                self._benchmark_closes.pop(0)
        elif self._benchmark_closes:
            pass

        if len(self._benchmark_closes) >= self.market_ma_period:
            ma = sum(self._benchmark_closes[-self.market_ma_period:]) / self.market_ma_period
            self._market_ok = self._benchmark_closes[-1] >= ma
        else:
            self._market_ok = True

    # ------------------------------------------------------------------
    # 调仓逻辑
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

        # 市场状态过滤：指数在 MA 以下时不开新仓
        if self.use_market_filter and not self._market_ok:
            for symbol in pos_symbols:
                if symbol not in risk_sell:
                    self.set_target(symbol, 0)
            return

        signal = signal.sort("signal", descending=True)

        # 信号概率过滤
        if self.min_signal_prob > 0:
            signal = signal.filter(pl.col("signal") >= self.min_signal_prob)

        # 动态 K 值
        effective_k = self.top_k
        if self.dynamic_k and not signal.is_empty():
            max_prob = signal["signal"].max()
            if max_prob < self.dynamic_k_prob_threshold:
                effective_k = max(self.dynamic_k_min, self.top_k // 2)

        buy_candidates: list[str] = list(signal["vt_symbol"][:effective_k])

        if self.smooth_rebalance:
            self._smooth_rebalance(bars, pos_symbols, risk_sell, buy_candidates, signal)
        else:
            self._full_rebalance(bars, pos_symbols, risk_sell, buy_candidates, signal)

    def _full_rebalance(
        self,
        bars: dict[str, BarData],
        pos_symbols: list[str],
        risk_sell: set[str],
        buy_candidates: list[str],
        signal: pl.DataFrame,
    ) -> None:
        """全量调仓（V1.0 逻辑）"""
        for symbol in pos_symbols:
            if symbol not in buy_candidates and symbol not in risk_sell:
                self.set_target(symbol, 0)

        cash = self.get_cash_available()
        for symbol in pos_symbols:
            if self.get_target(symbol) == 0:
                bar = bars.get(symbol)
                if bar and bar.close_price:
                    cash += bar.close_price * self.get_pos(symbol) * (1 - self.close_cost_rate)

        new_buys = [s for s in buy_candidates if self.get_pos(s) <= 0 or self.get_target(s) == 0]
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

    def _smooth_rebalance(
        self,
        bars: dict[str, BarData],
        pos_symbols: list[str],
        risk_sell: set[str],
        buy_candidates: list[str],
        signal: pl.DataFrame,
    ) -> None:
        """平滑调仓（V1.1）：已持仓且仍在信号中的股票保持不动"""
        keep_symbols = set()
        sell_symbols = []

        for symbol in pos_symbols:
            if symbol in risk_sell:
                continue
            if symbol in buy_candidates:
                keep_symbols.add(symbol)
            else:
                sell_symbols.append(symbol)

        max_sell = max(1, int(len(pos_symbols) * self.max_replace_ratio))
        actual_sell = sell_symbols[:max_sell]

        for symbol in actual_sell:
            self.set_target(symbol, 0)

        cash = self.get_cash_available()
        for symbol in actual_sell:
            bar = bars.get(symbol)
            if bar and bar.close_price:
                cash += bar.close_price * self.get_pos(symbol) * (1 - self.close_cost_rate)

        new_buys = [s for s in buy_candidates if s not in keep_symbols and self.get_pos(s) <= 0]
        if not new_buys:
            return

        if self.weight_by_signal:
            sig_map = {
                row["vt_symbol"]: row["signal"]
                for row in signal.iter_rows(named=True)
            }
            weights = [sig_map.get(s, 0.0) for s in new_buys]
            total_w = sum(weights) or 1.0
            for symbol, w in zip(new_buys, weights):
                bar = bars.get(symbol)
                if not bar or not bar.close_price:
                    continue
                alloc = cash * self.cash_ratio * (w / total_w)
                volume = round_to(alloc / bar.close_price, self.min_volume)
                if volume > 0:
                    self.set_target(symbol, volume)
        else:
            buy_value = cash * self.cash_ratio / len(new_buys)
            for symbol in new_buys:
                bar = bars.get(symbol)
                if not bar or not bar.close_price:
                    continue
                volume = round_to(buy_value / bar.close_price, self.min_volume)
                if volume > 0:
                    self.set_target(symbol, volume)

    # ------------------------------------------------------------------
    # 工具方法
    # ------------------------------------------------------------------

    def _clear_tracking(self, vt_symbol: str) -> None:
        self.entry_prices.pop(vt_symbol, None)
        self.peak_prices.pop(vt_symbol, None)
        self.hold_days.pop(vt_symbol, None)
        self.tp_activated.pop(vt_symbol, None)

    @staticmethod
    def _current_dt(bars: dict[str, BarData]):
        for bar in bars.values():
            return bar.datetime
        return None
