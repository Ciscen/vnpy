"""
hs300_topk/strategy/hs300_topk_strategy.py

HS300 选股策略，继承 AlphaStrategy，支持两种模式：

V1.x 周度模式 (daily_signal=False):
  - 周一（信号日）: 读取模型信号 → 选 top_k 只股票 → 平旧仓 + 开新仓
  - 周二～周五: 仅做风控检查

V2.0 信号驱动模式 (daily_signal=True):
  - 月度选池: 每月第一个交易日，全市场信号排序取 top pool_size 候选
  - 信号到期: 持仓 >= signal_horizon 天后查看当日信号，强则续期，弱则退出
  - 空位补仓: 有空仓位时从候选池中买入信号最强的股票
  - 风控: 止损/止盈/追踪每日执行
"""
from __future__ import annotations

import polars as pl

from vnpy.trader.object import BarData, TradeData
from vnpy.trader.constant import Direction
from vnpy.trader.utility import round_to

from vnpy.alpha import AlphaStrategy


class HS300Top10Strategy(AlphaStrategy):
    """HS300 选股策略（支持 V1.x 周度模式和 V2.0 信号驱动模式）"""

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

    # V1.2 新增参数
    portfolio_daily_loss_limit: float = 0.0
    cooldown_days: int = 0
    min_signal_spread: float = 0.0

    # V1.3 新增参数
    conditional_hold_extend: bool = False
    hold_extend_min_pnl: float = 0.03
    hold_extend_days: int = 2
    absolute_stop_cap: float = 0.0
    profit_lock_threshold: float = 0.0
    profit_lock_trail_pct: float = 0.015
    momentum_filter: bool = False
    momentum_lookback: int = 3
    momentum_min_return: float = -0.03
    rebalance_period: int = 1
    stock_cooldown_days: int = 0

    # V2.0 信号驱动持仓参数
    daily_signal: bool = False
    pool_size: int = 30
    signal_horizon: int = 3
    entry_threshold: float = 0.50
    renew_threshold: float = 0.40
    max_renewals: int = 3

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
        self._cooldown_remaining: int = 0
        self._prev_balance: float = 0.0

        self._pending_sell_reasons: dict[str, str] = {}
        self.trade_log: list[dict] = []

        # V1.3: 条件延仓追踪
        self._extended_symbols: dict[str, int] = {}
        self._week_counter: int = 0
        self._stock_cooldowns: dict[str, int] = {}

        # V2.0: 信号驱动持仓状态
        self._candidate_pool: list[str] = []
        self._pool_month: int = -1
        self._renewal_count: dict[str, int] = {}

        self.write_log("HS300Top10Strategy 初始化完成")

    def on_trade(self, trade: TradeData) -> None:
        """成交回调，更新持仓跟踪状态并记录交易日志"""
        entry_price = trade.price
        pnl_pct = 0.0
        hold = 0

        if trade.direction == Direction.LONG:
            reason = "signal_buy"
            entry_price = trade.price
            self.entry_prices[trade.vt_symbol] = trade.price
            self.peak_prices[trade.vt_symbol] = trade.price
            self.hold_days[trade.vt_symbol] = 0
            self.tp_activated[trade.vt_symbol] = False
        elif trade.direction == Direction.SHORT:
            reason = self._pending_sell_reasons.pop(trade.vt_symbol, "unknown")
            entry_price = self.entry_prices.get(trade.vt_symbol, trade.price)
            hold = self.hold_days.get(trade.vt_symbol, 0)
            pnl_pct = (trade.price - entry_price) / entry_price if entry_price else 0
            if self.stock_cooldown_days > 0 and "stop_loss" in reason:
                self._stock_cooldowns[trade.vt_symbol] = self.stock_cooldown_days
            self._clear_tracking(trade.vt_symbol)
        else:
            reason = "unknown"

        self.trade_log.append({
            "datetime": str(trade.datetime),
            "vt_symbol": trade.vt_symbol,
            "direction": trade.direction.value,
            "price": trade.price,
            "volume": trade.volume,
            "reason": reason,
            "entry_price": round(entry_price, 4),
            "pnl_pct": round(pnl_pct * 100, 2),
            "hold_days": hold,
        })

    def on_bars(self, bars: dict[str, BarData]) -> None:
        """K 线切片回调 — 每日执行一次"""
        dt = self._current_dt(bars)
        if dt is None:
            return

        if self.daily_signal:
            self._on_bars_v2(bars, dt)
        else:
            self._on_bars_v1(bars, dt)

    # ==================================================================
    # V1.x 周度模式
    # ==================================================================

    def _on_bars_v1(self, bars: dict[str, BarData], dt) -> None:
        """V1.x 周度模式的每日处理逻辑"""
        weekday = dt.weekday()
        pos_symbols = [s for s, p in self.pos_data.items() if p > 0]

        if self.use_atr_stop or self.momentum_filter:
            self._update_atr(bars)

        if self.use_market_filter:
            self._update_market_state(bars)

        self._update_hold_and_peak(bars, pos_symbols)

        if self._stock_cooldowns:
            expired = [s for s, d in self._stock_cooldowns.items() if d <= 1]
            for s in expired:
                del self._stock_cooldowns[s]
            for s in self._stock_cooldowns:
                self._stock_cooldowns[s] -= 1

        risk_sell = self._risk_check(bars, pos_symbols)

        for symbol in risk_sell:
            self.set_target(symbol, 0)

        if self.portfolio_daily_loss_limit > 0:
            self._portfolio_risk_check(bars, pos_symbols, risk_sell)

        if self._cooldown_remaining > 0:
            self._cooldown_remaining -= 1

        if weekday == 0:
            self._week_counter += 1
        is_rebalance_day = (
            weekday == 0
            and self._cooldown_remaining <= 0
            and self._week_counter % self.rebalance_period == 0
        )
        if is_rebalance_day:
            self._rebalance(bars, pos_symbols, risk_sell)

        self._cap_buy_targets(bars)
        self.execute_trading(bars, price_add=self.price_add)

    # ==================================================================
    # V2.0 信号驱动模式
    # ==================================================================

    def _on_bars_v2(self, bars: dict[str, BarData], dt) -> None:
        """V2.0 信号驱动持仓的每日处理逻辑"""
        pos_symbols = [s for s, p in self.pos_data.items() if p > 0]

        if self.use_atr_stop:
            self._update_atr(bars)
        else:
            for symbol, bar in bars.items():
                history = self._price_history.setdefault(symbol, [])
                history.append((bar.high_price, bar.low_price, bar.close_price))
                if len(history) > 15:
                    history.pop(0)

        self._update_hold_and_peak(bars, pos_symbols)

        # 清除所有旧目标，避免上一根 K 线的未成交目标残留
        for sym in list(self.target_data.keys()):
            self.target_data[sym] = 0.0

        # 保持现有持仓（将目标设为当前持仓量）
        for sym in pos_symbols:
            self.target_data[sym] = self.pos_data[sym]

        # ── Step 0: 月度选池 ──
        signal = self.get_signal()
        if dt.month != self._pool_month and not signal.is_empty():
            self._v2_refresh_pool(signal)
            self._pool_month = dt.month

        # ── Step 0.5: 冷却倒计时（在风控之前递减，新触发会覆盖） ──
        if self._cooldown_remaining > 0:
            self._cooldown_remaining -= 1

        # ── Step 1: 风控检查（止损/止盈，不含 max_hold） ──
        risk_sell = self._risk_check_v2(bars, pos_symbols)
        for symbol in risk_sell:
            self.set_target(symbol, 0)

        # ── Step 1.5: 组合级风控 ──
        if self.portfolio_daily_loss_limit > 0:
            self._portfolio_risk_check(bars, pos_symbols, risk_sell)

        # ── Step 2: 信号到期检查 ──
        remaining_pos = [s for s in pos_symbols if s not in risk_sell]
        signal_sell = self._v2_signal_expiry_check(remaining_pos, signal)
        for symbol in signal_sell:
            self.set_target(symbol, 0)

        # ── Step 3: 绝对持仓上限检查 ──
        all_sells = risk_sell | signal_sell
        for symbol in pos_symbols:
            if symbol in all_sells:
                continue
            hold = self.hold_days.get(symbol, 0)
            if hold >= self.max_hold_days:
                self._pending_sell_reasons[symbol] = f"max_hold({hold}d)"
                self.set_target(symbol, 0)
                all_sells.add(symbol)

        # ── Step 4: 空位补仓（冷却期跳过） ──
        if self._cooldown_remaining <= 0:
            self._v2_fill_slots(bars, pos_symbols, signal)

        self._cap_buy_targets(bars)
        self.execute_trading(bars, price_add=self.price_add)

    def _v2_refresh_pool(self, signal: pl.DataFrame) -> None:
        """月度选池：全市场信号排序 → top pool_size"""
        sorted_signal = signal.sort("signal", descending=True)
        self._candidate_pool = list(
            sorted_signal["vt_symbol"][:self.pool_size]
        )
        self.write_log(
            f"[V2.0] 月度选池刷新: {len(self._candidate_pool)} 只候选"
        )

    def _risk_check_v2(
        self, bars: dict[str, BarData], pos_symbols: list[str]
    ) -> set[str]:
        """V2.0 风控：止损/止盈/绝对止损（不含 max_hold，由信号到期处理）"""
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

            if self.absolute_stop_cap > 0 and pnl_pct <= -self.absolute_stop_cap:
                self._pending_sell_reasons[symbol] = f"abs_stop_cap({pnl_pct*100:.1f}%)"
                risk_sell.add(symbol)
                continue

            if pnl_pct <= -effective_sl:
                self._pending_sell_reasons[symbol] = f"stop_loss({pnl_pct*100:.1f}%)"
                risk_sell.add(symbol)
                continue

            if pnl_pct >= self.tp_activate_pct:
                self.tp_activated[symbol] = True

            if self.tp_activated.get(symbol, False):
                peak = self.peak_prices.get(symbol, entry)
                dd = (bar.close_price - peak) / peak
                trail = self.tp_trail_pct
                if self.profit_lock_threshold > 0 and pnl_pct >= self.profit_lock_threshold:
                    trail = self.profit_lock_trail_pct
                if dd <= -trail:
                    self._pending_sell_reasons[symbol] = (
                        f"trailing_tp(peak_dd={dd*100:.1f}%,trail={trail*100:.1f}%)"
                    )
                    risk_sell.add(symbol)

        return risk_sell

    def _v2_signal_expiry_check(
        self, pos_symbols: list[str], signal: pl.DataFrame
    ) -> set[str]:
        """信号到期检查：持仓 >= signal_horizon 天后根据当日信号决定续期或退出"""
        sell_set: set[str] = set()

        if signal.is_empty():
            return sell_set

        sig_map: dict[str, float] = {
            row["vt_symbol"]: row["signal"]
            for row in signal.iter_rows(named=True)
        }

        for symbol in pos_symbols:
            hold = self.hold_days.get(symbol, 0)
            if hold < self.signal_horizon:
                continue

            today_sig = sig_map.get(symbol, 0.0)
            renewals = self._renewal_count.get(symbol, 0)

            if today_sig >= self.renew_threshold and renewals < self.max_renewals:
                self.hold_days[symbol] = 0
                self._renewal_count[symbol] = renewals + 1
                self.write_log(
                    f"[V2.0] {symbol} 信号续期 #{renewals+1} "
                    f"(sig={today_sig:.3f})"
                )
            else:
                reason = (
                    f"signal_expired(hold={hold}d,sig={today_sig:.3f},"
                    f"renew={renewals}/{self.max_renewals})"
                )
                self._pending_sell_reasons[symbol] = reason
                sell_set.add(symbol)

        return sell_set

    def _v2_fill_slots(
        self,
        bars: dict[str, BarData],
        pos_symbols: list[str],
        signal: pl.DataFrame,
    ) -> None:
        """空位补仓：从候选池中买入信号最强的股票

        使用实际持仓数（pos > 0）计算空位，而非预期卖出后的估算值。
        因为卖单在下一根 K 线才撮合，如遇跌停可能无法成交。
        """
        actual_pos_count = len(pos_symbols)
        free_slots = self.top_k - actual_pos_count
        if free_slots <= 0 or signal.is_empty():
            return

        exclude_set = set(pos_symbols)
        pool_set = set(self._candidate_pool)

        pool_signal = signal.filter(
            pl.col("vt_symbol").is_in(pool_set - exclude_set)
            & (pl.col("signal") >= self.entry_threshold)
        ).sort("signal", descending=True)

        raw_candidates = list(pool_signal["vt_symbol"][:free_slots * 2])
        buy_candidates = []
        for sym in raw_candidates:
            if len(buy_candidates) >= free_slots:
                break
            if self._is_near_limit_up(sym, bars):
                continue
            buy_candidates.append(sym)

        if not buy_candidates:
            return

        cash = self.get_cash_available()
        for symbol, pos in self.pos_data.items():
            if pos > 0 and self.get_target(symbol) == 0:
                bar = bars.get(symbol)
                if bar and bar.close_price:
                    cash += bar.close_price * pos * (1 - self.close_cost_rate)

        if self.weight_by_signal:
            sig_map = {
                row["vt_symbol"]: row["signal"]
                for row in pool_signal.iter_rows(named=True)
            }
            weights = [sig_map.get(s, 0.0) for s in buy_candidates]
            total_w = sum(weights) or 1.0
            for symbol, w in zip(buy_candidates, weights):
                bar = bars.get(symbol)
                if not bar or not bar.close_price:
                    continue
                alloc = cash * self.cash_ratio * (w / total_w)
                volume = round_to(alloc / bar.close_price, self.min_volume)
                if volume > 0:
                    self.set_target(symbol, volume)
        else:
            buy_value = cash * self.cash_ratio / len(buy_candidates)
            for symbol in buy_candidates:
                bar = bars.get(symbol)
                if not bar or not bar.close_price:
                    continue
                volume = round_to(buy_value / bar.close_price, self.min_volume)
                if volume > 0:
                    self.set_target(symbol, volume)

    def _cap_buy_targets(self, bars: dict[str, BarData]) -> None:
        """防止 round_to 取整导致总买入额超过可用资金。

        在 execute_trading 之前调用。
        逐轮缩减：每轮检查总额是否超限，若超则先淘汰最超额的个股，
        然后重新分配剩余预算。最多迭代 3 轮以防止无限循环。
        """
        cash = self.get_cash_available()
        for symbol, pos in self.pos_data.items():
            if pos > 0 and self.get_target(symbol) == 0:
                bar = bars.get(symbol)
                if bar and bar.close_price:
                    cash += bar.close_price * pos * (1 - self.close_cost_rate)

        max_buy = cash * self.cash_ratio

        for _ in range(3):
            total_buy_cost = 0.0
            buy_entries: list[tuple[str, float, float]] = []
            for symbol, target in self.target_data.items():
                pos = self.get_pos(symbol)
                diff = target - pos
                if diff > 0:
                    bar = bars.get(symbol)
                    if bar and bar.close_price:
                        cost = bar.close_price * diff
                        total_buy_cost += cost
                        buy_entries.append((symbol, diff, cost))

            if total_buy_cost <= max_buy or total_buy_cost <= 0:
                return

            per_stock_limit = max_buy / max(len(buy_entries), 1)
            dropped = False
            for symbol, diff, cost in buy_entries:
                if cost > per_stock_limit * 1.5:
                    pos = self.get_pos(symbol)
                    self.set_target(symbol, pos)
                    dropped = True

            if not dropped:
                scale = max_buy / total_buy_cost
                for symbol, diff, cost in buy_entries:
                    pos = self.get_pos(symbol)
                    new_diff = round_to(diff * scale, self.min_volume)
                    if new_diff > 0:
                        self.set_target(symbol, pos + new_diff)
                    else:
                        self.set_target(symbol, pos)
                return

    # ==================================================================
    # 共享辅助方法
    # ==================================================================

    def _is_near_limit_up(self, symbol: str, bars: dict[str, BarData]) -> bool:
        """判断股票当日是否接近涨停（次日大概率一字涨停不可买入）

        创业板(300)/科创板(688): ±20% → 当日涨幅 >= 18% 视为接近涨停
        主板: ±10% → 当日涨幅 >= 9% 视为接近涨停
        """
        bar = bars.get(symbol)
        if not bar or not bar.close_price:
            return False

        history = self._price_history.get(symbol)
        if not history or len(history) < 2:
            return False

        prev_close = history[-2][2]
        if prev_close <= 0:
            return False

        change_pct = (bar.close_price - prev_close) / prev_close

        is_wide_limit = symbol.startswith("300") or symbol.startswith("688")
        threshold = 0.18 if is_wide_limit else 0.09

        return change_pct >= threshold

    def _update_hold_and_peak(
        self, bars: dict[str, BarData], pos_symbols: list[str]
    ) -> None:
        """更新持仓天数和峰值价格"""
        for symbol in pos_symbols:
            self.hold_days[symbol] = self.hold_days.get(symbol, 0) + 1
            bar = bars.get(symbol)
            if bar and symbol in self.peak_prices:
                if bar.close_price > self.peak_prices[symbol]:
                    self.peak_prices[symbol] = bar.close_price

    def _risk_check(
        self, bars: dict[str, BarData], pos_symbols: list[str]
    ) -> set[str]:
        """V1.x 风控检查（含 max_hold 和条件延仓）"""
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

            if self.absolute_stop_cap > 0 and pnl_pct <= -self.absolute_stop_cap:
                self._pending_sell_reasons[symbol] = f"abs_stop_cap({pnl_pct*100:.1f}%)"
                risk_sell.add(symbol)
                continue

            if pnl_pct <= -effective_sl:
                self._pending_sell_reasons[symbol] = f"stop_loss({pnl_pct*100:.1f}%)"
                risk_sell.add(symbol)
                continue

            if pnl_pct >= self.tp_activate_pct:
                self.tp_activated[symbol] = True

            if self.tp_activated.get(symbol, False):
                peak = self.peak_prices.get(symbol, entry)
                dd = (bar.close_price - peak) / peak
                trail = self.tp_trail_pct
                if self.profit_lock_threshold > 0 and pnl_pct >= self.profit_lock_threshold:
                    trail = self.profit_lock_trail_pct
                if dd <= -trail:
                    self._pending_sell_reasons[symbol] = (
                        f"trailing_tp(peak_dd={dd*100:.1f}%,trail={trail*100:.1f}%)"
                    )
                    risk_sell.add(symbol)
                    continue

            hold = self.hold_days.get(symbol, 0)
            max_days = self.max_hold_days
            extended = self._extended_symbols.get(symbol, 0)

            if self.conditional_hold_extend and hold >= max_days:
                if extended < self.hold_extend_days and pnl_pct >= self.hold_extend_min_pnl:
                    self._extended_symbols[symbol] = extended + 1
                    continue
                self._pending_sell_reasons[symbol] = (
                    f"max_hold({hold}d,ext={extended}d,pnl={pnl_pct*100:.1f}%)"
                )
                risk_sell.add(symbol)
            elif hold >= max_days:
                self._pending_sell_reasons[symbol] = f"max_hold({hold}d)"
                risk_sell.add(symbol)

        return risk_sell

    def _portfolio_risk_check(
        self,
        bars: dict[str, BarData],
        pos_symbols: list[str],
        risk_sell: set[str],
    ) -> None:
        """组合级风控：单日最大亏损限制"""
        current_balance = self._estimate_balance(bars)
        if self._prev_balance > 0:
            daily_ret = (current_balance - self._prev_balance) / self._prev_balance
            if daily_ret <= -self.portfolio_daily_loss_limit:
                for s in pos_symbols:
                    if s not in risk_sell:
                        self._pending_sell_reasons[s] = f"portfolio_stop({daily_ret*100:.1f}%)"
                        self.set_target(s, 0)
                self._cooldown_remaining = self.cooldown_days
        self._prev_balance = current_balance

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

        if self.use_market_filter and not self._market_ok:
            for symbol in pos_symbols:
                if symbol not in risk_sell:
                    self._pending_sell_reasons[symbol] = "market_filter"
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

        # V1.3: 动量确认入场 — 过滤近 N 日跌幅过大的股票
        if self.momentum_filter:
            filtered = []
            for sym in buy_candidates:
                history = self._price_history.get(sym)
                if history and len(history) >= self.momentum_lookback:
                    recent_close = history[-1][2]
                    past_close = history[-self.momentum_lookback][2]
                    ret = (recent_close - past_close) / past_close if past_close else 0
                    if ret >= self.momentum_min_return:
                        filtered.append(sym)
                else:
                    filtered.append(sym)
            buy_candidates = filtered

        if self._stock_cooldowns:
            buy_candidates = [s for s in buy_candidates if s not in self._stock_cooldowns]

        if self.min_signal_spread > 0 and len(buy_candidates) >= 2:
            top1_prob = signal["signal"][0]
            topk_prob = signal["signal"][min(effective_k - 1, signal.height - 1)]
            if top1_prob - topk_prob < self.min_signal_spread:
                effective_k = max(self.dynamic_k_min if self.dynamic_k else 3, effective_k // 2)
                buy_candidates = list(signal["vt_symbol"][:effective_k])

        buy_candidates = [s for s in buy_candidates if not self._is_near_limit_up(s, bars)]

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
                self._pending_sell_reasons[symbol] = "rebalance_out"
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
            self._pending_sell_reasons[symbol] = "rebalance_out"
            self.set_target(symbol, 0)

        cash = self.get_cash_available()
        for symbol in actual_sell:
            bar = bars.get(symbol)
            if bar and bar.close_price:
                cash += bar.close_price * self.get_pos(symbol) * (1 - self.close_cost_rate)

        free_slots = max(0, self.top_k - len(pos_symbols) + len(actual_sell))
        new_buys = [s for s in buy_candidates if s not in keep_symbols and self.get_pos(s) <= 0][:free_slots]
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

    def _estimate_balance(self, bars: dict[str, BarData]) -> float:
        """估算当前账户总价值（现金 + 持仓市值）"""
        holding_value = 0.0
        for symbol, pos in self.pos_data.items():
            if pos > 0:
                bar = bars.get(symbol)
                if bar and bar.close_price:
                    holding_value += bar.close_price * pos
        return self.get_cash_available() + holding_value

    def _clear_tracking(self, vt_symbol: str) -> None:
        self.entry_prices.pop(vt_symbol, None)
        self.peak_prices.pop(vt_symbol, None)
        self.hold_days.pop(vt_symbol, None)
        self.tp_activated.pop(vt_symbol, None)
        self._extended_symbols.pop(vt_symbol, None)
        self._renewal_count.pop(vt_symbol, None)

    @staticmethod
    def _current_dt(bars: dict[str, BarData]):
        for bar in bars.values():
            return bar.datetime
        return None
