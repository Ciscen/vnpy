"""
hs300_topk/live/portfolio.py

持仓管理模块 — 飞书文档表格读取 + 本地缓存降级 + 调仓差异计算。

飞书文档表格格式约定（两种均支持）:

方式 A — 可用资金/总资产作为列:
    | 股票代码 | 股票名称 | 持仓数量 | 成本价 | 买入日期 | 可用资金 | 总资产 | 备注 |
    | 300394   | 天孚通信 | 200      | 25.30  | 2026-04-28 | 85000 |      |      |

方式 B — 可用资金作为独立行:
    | 股票代码 | 股票名称 | 持仓数量 | 成本价 | 买入日期 | 备注 |
    | 300394   | 天孚通信 | 200      | 25.30  | 2026-04-28 |      |
    | 可用资金 | 85000    |          |        |            |      |

字段说明:
    - 成本价: 应填写 **持仓成本价（加权均价）**，而非单次买入价。
      若有多次买入同一只股票，需自行计算均价后更新。
      该字段用于显示浮盈浮亏，不影响调仓决策（决策由信号驱动）。
      若未填写，程序以昨收价替代。
    - 可用资金: 券商账户中当前可用于买入的资金余额。
      "现金" 和 "可用资金" 等价，统一按可用资金理解。
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)

STATE_DIR = Path(__file__).parent / "state"
CACHE_FILE = STATE_DIR / "portfolio_cache.json"


@dataclass
class Position:
    """单只股票持仓。"""
    symbol: str
    name: str
    shares: int
    cost: float
    entry_date: str

    @property
    def vt_symbol(self) -> str:
        """转换为 vnpy 格式 (如 300394.SZSE)。"""
        code = self.symbol.replace(" ", "")
        if "." in code:
            return code
        if code.startswith(("6", "5")):
            return f"{code}.SSE"
        if code.startswith(("0", "3")):
            return f"{code}.SZSE"
        return f"{code}.SSE"


@dataclass
class Portfolio:
    """账户持仓快照。"""
    cash: float = 0.0
    positions: list[Position] = field(default_factory=list)
    cooldowns: dict[str, int] = field(default_factory=dict)
    updated_at: str = ""

    @property
    def position_map(self) -> dict[str, Position]:
        """vt_symbol -> Position 映射。"""
        return {p.vt_symbol: p for p in self.positions}

    @property
    def total_value(self) -> float:
        """按成本价计算的账户总值。"""
        return self.cash + sum(p.shares * p.cost for p in self.positions)

    def total_market_value(self, prices: dict[str, float]) -> float:
        """按市价计算的账户总值。"""
        return self.cash + sum(
            p.shares * prices.get(p.vt_symbol, p.cost)
            for p in self.positions
        )


# ──────────────────────────────────────────────────
# 飞书文档表格 → Portfolio
# ──────────────────────────────────────────────────

def _parse_float(s: str) -> float:
    """容错解析浮点数。"""
    s = s.replace(",", "").replace("，", "").strip()
    try:
        return float(s)
    except (ValueError, TypeError):
        return 0.0


def _parse_int(s: str) -> int:
    s = s.replace(",", "").replace("，", "").strip()
    try:
        return int(float(s))
    except (ValueError, TypeError):
        return 0


def _is_summary_row(row: list[str]) -> bool:
    """判断是否为现金/总资产等汇总行（非股票行）。"""
    first = row[0].strip() if row else ""
    return any(kw in first for kw in ("现金", "资金", "资产", "合计")) or "cash" in first.lower()


def parse_table_to_portfolio(table: list[list[str]]) -> Portfolio:
    """将飞书文档内嵌表格的二维矩阵解析为 Portfolio。

    约定第一行为表头，后续行为持仓或汇总行。
    支持"可用资金"作为列或独立行两种格式。
    """
    if len(table) < 2:
        logger.warning("表格行数不足 (%d行)，返回空持仓", len(table))
        return Portfolio(updated_at=datetime.now().isoformat())

    header = [h.strip() for h in table[0]]
    logger.debug("表头: %s", header)
    col_map: dict[str, int] = {}
    keywords = {
        "代码": "code", "名称": "name", "数量": "shares",
        "成本": "cost", "买入": "entry_date", "日期": "entry_date",
        "可用资金": "cash", "现金": "cash", "备注": "note",
    }
    for idx, h in enumerate(header):
        for kw, field_name in keywords.items():
            if kw in h:
                col_map[field_name] = idx
                break

    if "code" not in col_map:
        logger.warning("未识别到'代码'列，将默认使用第0列; 表头=%s", header)

    positions: list[Position] = []
    cash = 0.0
    cash_found = False
    skipped_rows = 0

    cash_col_idx = col_map.get("cash", -1)

    for row_idx, row in enumerate(table[1:], start=2):
        if not row or all(not c.strip() for c in row):
            continue

        if _is_summary_row(row):
            first = row[0].strip()
            if "资金" in first or "现金" in first or "cash" in first.lower():
                for cell in row[1:]:
                    v = _parse_float(cell)
                    if v > 0:
                        cash = v
                        cash_found = True
                        logger.debug("行%d: 识别为资金行, cash=%.2f", row_idx, cash)
                        break
            continue

        if not cash_found and cash_col_idx != -1 and cash_col_idx < len(row):
            cv = _parse_float(row[cash_col_idx])
            if cv > 0:
                cash = cv
                cash_found = True
                logger.debug("行%d: 从列提取 cash=%.2f", row_idx, cash)

        code_idx = col_map.get("code", 0)
        code = row[code_idx].strip() if code_idx < len(row) else ""
        if not code or not re.match(r"^\d{6}", code):
            skipped_rows += 1
            continue

        name_idx = col_map.get("name", 1)
        shares_idx = col_map.get("shares", 2)
        cost_idx = col_map.get("cost", 3)
        date_idx = col_map.get("entry_date", 4)

        name = row[name_idx].strip() if name_idx < len(row) else ""
        shares = _parse_int(row[shares_idx]) if shares_idx < len(row) else 0
        cost = _parse_float(row[cost_idx]) if cost_idx < len(row) else 0.0
        entry_date = row[date_idx].strip() if date_idx < len(row) else ""

        if shares <= 0:
            logger.debug("行%d: 股票 %s 数量≤0，跳过", row_idx, code)
            skipped_rows += 1
            continue

        if cost <= 0:
            logger.warning("行%d: 股票 %s 成本价为0，将在计算时用昨收价替代",
                           row_idx, code)

        if shares % 100 != 0:
            logger.warning("行%d: 股票 %s 持仓 %d 股不是100整数倍，请确认",
                           row_idx, code, shares)

        positions.append(Position(
            symbol=code, name=name, shares=shares,
            cost=cost, entry_date=entry_date,
        ))

    if not cash_found:
        logger.warning("未找到可用资金字段，cash=0; 请检查飞书文档格式")
    if skipped_rows > 0:
        logger.debug("解析中跳过 %d 行（空行/非股票行/零持仓）", skipped_rows)

    logger.info("表格解析完成: %d 只持仓, cash=%.2f, 跳过%d行",
                len(positions), cash, skipped_rows)

    return Portfolio(
        cash=cash,
        positions=positions,
        updated_at=datetime.now().isoformat(),
    )


def load_portfolio_from_feishu(
    doc_id: str | None = None,
    table_index: int = 0,
) -> Portfolio:
    """从飞书文档读取持仓。

    Parameters
    ----------
    doc_id : str | None
        飞书文档/Wiki ID，默认从环境变量 FEISHU_DOC_ID 读取。
    table_index : int
        文档中第几个表格（0-based，对多维/电子表格忽略此参数）。
    """
    import os
    from hs300_topk.live.feishu import FeishuClient

    doc_id = doc_id or os.environ.get("FEISHU_DOC_ID", "")
    if not doc_id:
        raise RuntimeError("请设置环境变量 FEISHU_DOC_ID")

    logger.info("从飞书加载持仓 (doc_id=%s...)", doc_id[:8])
    client = FeishuClient.from_env()
    
    obj_token = doc_id
    obj_type = "docx"
    
    try:
        node_info = client.get_wiki_node_info(doc_id)
        obj_token = node_info.get("obj_token", doc_id)
        obj_type = node_info.get("obj_type", "docx")
        logger.debug("Wiki 节点: obj_type=%s, obj_token=%s...",
                      obj_type, obj_token[:8])
    except Exception as e:
        logger.debug("Wiki 查询失败，按 docx 处理: %s", e)

    if obj_type == "sheet":
        logger.info("  文档类型: 电子表格")
        matrix = client.read_spreadsheet_values(obj_token)
        tables = [matrix]
        table_index = 0
    else:
        logger.info("  文档类型: %s (内嵌表格)", obj_type)
        tables = client.read_doc_tables(obj_token)

    if not tables:
        raise RuntimeError(f"文档 {doc_id} 中未找到表格或数据")
    if table_index >= len(tables):
        raise RuntimeError(f"文档中只有 {len(tables)} 个表格，请求第 {table_index} 个")

    logger.info("  找到 %d 个表格，解析第 %d 个 (%d行 x %d列)",
                len(tables), table_index,
                len(tables[table_index]),
                len(tables[table_index][0]) if tables[table_index] else 0)
    portfolio = parse_table_to_portfolio(tables[table_index])
    logger.info("飞书持仓加载完成: %d 只股票, 现金 %.2f",
                len(portfolio.positions), portfolio.cash)

    save_portfolio_local(portfolio)
    return portfolio


# ──────────────────────────────────────────────────
# 本地缓存（降级方案）
# ──────────────────────────────────────────────────

def save_portfolio_local(portfolio: Portfolio, path: Path = CACHE_FILE) -> None:
    """将持仓快照缓存到本地 JSON。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "cash": portfolio.cash,
        "positions": [asdict(p) for p in portfolio.positions],
        "cooldowns": portfolio.cooldowns,
        "updated_at": portfolio.updated_at,
    }
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def load_portfolio_local(path: Path = CACHE_FILE) -> Portfolio:
    """从本地 JSON 缓存读取持仓。"""
    if not path.exists():
        logger.warning("本地持仓缓存不存在: %s, 返回空持仓", path)
        return Portfolio(updated_at=datetime.now().isoformat())
    data = json.loads(path.read_text(encoding="utf-8"))
    positions = [Position(**p) for p in data.get("positions", [])]
    return Portfolio(
        cash=data.get("cash", 0.0),
        positions=positions,
        cooldowns=data.get("cooldowns", {}),
        updated_at=data.get("updated_at", ""),
    )


def load_portfolio(doc_id: str | None = None) -> Portfolio:
    """加载持仓：优先飞书，降级到本地缓存。"""
    try:
        return load_portfolio_from_feishu(doc_id)
    except Exception as e:
        logger.warning("飞书持仓读取失败: %s", e)
        logger.info("  → 降级: 使用本地缓存 %s", CACHE_FILE)
        p = load_portfolio_local()
        if p.updated_at:
            logger.info("  本地缓存数据时间: %s", p.updated_at[:19])
        return p


# ──────────────────────────────────────────────────
# 调仓差异计算
# ──────────────────────────────────────────────────

@dataclass
class RebalanceAction:
    """一条调仓指令。"""
    symbol: str
    name: str
    action: str       # BUY / SELL / HOLD
    shares: int
    ref_price: float
    price_low: float        # 建议最低执行价
    price_high: float       # 建议最高执行价
    estimated_amount: float
    current_pnl_pct: float
    reason: str
    reason_text: str        # 人读友好的调仓理由
    signal_prob: float
    signal_rank: int
    fee: float = 0.0             # 预估手续费
    net_amount: float = 0.0      # 扣费后净额
    market_value: float = 0.0    # 持仓市值（HOLD/SELL 有效）
    weight_pct: float = 0.0      # 持仓占总市值的百分比
    hold_days: int = 0           # 持有天数


# ──────────────────────────────────────────────────
# A 股手续费计算
# ──────────────────────────────────────────────────

@dataclass(frozen=True)
class FeeSchedule:
    """A 股交易手续费结构。"""
    commission_rate: float = 0.00025    # 佣金费率（万 2.5，单边）
    commission_min: float = 5.0         # 最低佣金（元）
    stamp_duty_rate: float = 0.0005     # 印花税费率（卖出时收取，千分之 0.5）
    transfer_fee_rate: float = 0.00001  # 过户费费率（万 0.1，双边）


DEFAULT_FEES = FeeSchedule()


def calc_trade_fee(
    amount: float,
    is_sell: bool,
    fees: FeeSchedule = DEFAULT_FEES,
) -> float:
    """计算单笔交易的预估手续费。

    Parameters
    ----------
    amount : float
        交易金额（股数 x 价格）。
    is_sell : bool
        是否为卖出。
    fees : FeeSchedule
        费率结构。

    Returns
    -------
    float
        手续费合计。
    """
    commission = max(amount * fees.commission_rate, fees.commission_min)
    transfer = amount * fees.transfer_fee_rate
    stamp = amount * fees.stamp_duty_rate if is_sell else 0.0
    return round(commission + transfer + stamp, 2)


def _calc_hold_days(entry_date: str) -> int:
    """计算持有天数。"""
    if not entry_date:
        return 0
    try:
        entry = datetime.fromisoformat(entry_date).date()
        return (datetime.now().date() - entry).days
    except (ValueError, TypeError):
        return 0


def compute_rebalance(
    portfolio: Portfolio,
    signals: list[dict],
    prices: dict[str, float],
    top_k: int = 10,
    stock_cooldown_days: int = 10,
    slippage_pct: float = 1.5,
) -> tuple[list[RebalanceAction], list[dict]]:
    """根据当前持仓和信号 Top-K 计算买卖差异。

    Parameters
    ----------
    portfolio : Portfolio
        当前持仓。
    signals : list[dict]
        按 signal 降序排列的信号列表，每项含 vt_symbol, signal。
    prices : dict[str, float]
        vt_symbol -> 上一交易日收盘价。
    top_k : int
        目标持仓数量。
    stock_cooldown_days : int
        个股冷却天数。
    slippage_pct : float
        允许的滑点百分比（默认 1.5%），用于计算建议价格范围。

    Returns
    -------
    tuple[list[RebalanceAction], list[dict]]
        (调仓动作列表, 冷却中被跳过的股票列表)
    """
    slip = slippage_pct / 100.0
    total_mkt_value = portfolio.total_market_value(prices)
    logger.debug("compute_rebalance: total_mkt=%.2f, cash=%.2f, "
                 "positions=%d, signals=%d, top_k=%d",
                 total_mkt_value, portfolio.cash,
                 len(portfolio.positions), len(signals), top_k)

    skipped_cooldowns: list[dict] = []
    top_k_symbols: list[str] = []
    for sig in signals:
        sym = sig["vt_symbol"]
        if sym in portfolio.cooldowns and portfolio.cooldowns[sym] > 0:
            skipped_cooldowns.append({
                "vt_symbol": sym,
                "signal": sig["signal"],
                "remaining_days": portfolio.cooldowns[sym],
            })
            continue
        top_k_symbols.append(sym)
        if len(top_k_symbols) >= top_k:
            break

    signal_map = {s["vt_symbol"]: s for s in signals}
    signal_rank = {s["vt_symbol"]: i + 1 for i, s in enumerate(signals)}
    pos_map = portfolio.position_map

    actions: list[RebalanceAction] = []

    for vt_sym, pos in pos_map.items():
        price = prices.get(vt_sym, pos.cost)
        pnl_pct = (price / pos.cost - 1) * 100 if pos.cost > 0 else 0.0
        sig = signal_map.get(vt_sym, {})
        rank = signal_rank.get(vt_sym, 999)
        sig_val = sig.get("signal", 0.0)
        mkt_val = pos.shares * price
        weight = (mkt_val / total_mkt_value * 100) if total_mkt_value > 0 else 0.0
        days = _calc_hold_days(pos.entry_date)

        if vt_sym in top_k_symbols:
            reason_text = f"信号排名 #{rank}（{sig_val:.2f}），仍在 Top-{top_k} 中，继续持有"
            actions.append(RebalanceAction(
                symbol=vt_sym, name=pos.name, action="HOLD",
                shares=pos.shares, ref_price=price,
                price_low=round(price * (1 - slip), 2),
                price_high=round(price * (1 + slip), 2),
                estimated_amount=0.0, current_pnl_pct=round(pnl_pct, 2),
                reason="still_top_k", reason_text=reason_text,
                signal_prob=sig_val, signal_rank=rank,
                market_value=round(mkt_val, 2),
                weight_pct=round(weight, 1),
                hold_days=days,
            ))
        else:
            if rank <= top_k + 5:
                reason_text = f"信号排名 #{rank}（{sig_val:.2f}），滑出 Top-{top_k}，建议卖出"
            elif rank > 100:
                reason_text = f"信号极弱 #{rank}（{sig_val:.2f}），远离 Top-{top_k}，建议卖出"
            else:
                reason_text = f"信号排名 #{rank}（{sig_val:.2f}），不在 Top-{top_k} 内，建议卖出"
            sell_amount = round(pos.shares * price, 2)
            fee = calc_trade_fee(sell_amount, is_sell=True)
            actions.append(RebalanceAction(
                symbol=vt_sym, name=pos.name, action="SELL",
                shares=pos.shares, ref_price=price,
                price_low=round(price * (1 - slip), 2),
                price_high=round(price * (1 + slip), 2),
                estimated_amount=sell_amount,
                current_pnl_pct=round(pnl_pct, 2),
                reason="rebalance_out", reason_text=reason_text,
                signal_prob=sig_val, signal_rank=rank,
                fee=fee, net_amount=round(sell_amount - fee, 2),
                market_value=round(mkt_val, 2),
                weight_pct=round(weight, 1),
                hold_days=days,
            ))

    held_symbols = set(pos_map.keys())
    sell_proceeds = sum(a.net_amount for a in actions if a.action == "SELL")
    available_cash = portfolio.cash + sell_proceeds
    new_buy_count = top_k - sum(1 for a in actions if a.action == "HOLD")

    if new_buy_count > 0:
        est_buy_fee_per_stock = calc_trade_fee(
            available_cash / new_buy_count, is_sell=False,
        )
        budget_after_fees = available_cash - est_buy_fee_per_stock * new_buy_count
        per_stock_budget = budget_after_fees / new_buy_count

        total_buy_cost = 0.0
        for vt_sym in top_k_symbols:
            if vt_sym in held_symbols:
                continue
            price = prices.get(vt_sym, 0.0)
            if price <= 0:
                continue
            shares = int(per_stock_budget / price / 100) * 100
            if shares <= 0:
                continue

            est_amount = round(shares * price, 2)
            fee = calc_trade_fee(est_amount, is_sell=False)
            total_cost = est_amount + fee
            if total_buy_cost + total_cost > available_cash:
                shares = int((available_cash - total_buy_cost - fee) / price / 100) * 100
                if shares <= 0:
                    continue
                est_amount = round(shares * price, 2)
                fee = calc_trade_fee(est_amount, is_sell=False)
                total_cost = est_amount + fee

            total_buy_cost += total_cost

            sig = signal_map.get(vt_sym, {})
            rank = signal_rank.get(vt_sym, 999)
            sig_val = sig.get("signal", 0.0)
            name = sig.get("name", vt_sym[:6])
            target_weight = (est_amount / total_mkt_value * 100) if total_mkt_value > 0 else 0.0
            reason_text = f"信号排名 #{rank}（{sig_val:.2f}），新进入 Top-{top_k}，建议买入"
            actions.append(RebalanceAction(
                symbol=vt_sym, name=name, action="BUY",
                shares=shares, ref_price=price,
                price_low=round(price * (1 - slip), 2),
                price_high=round(price * (1 + slip), 2),
                estimated_amount=est_amount,
                current_pnl_pct=0.0,
                reason="new_entry", reason_text=reason_text,
                signal_prob=sig_val, signal_rank=rank,
                fee=fee, net_amount=round(est_amount + fee, 2),
                market_value=est_amount,
                weight_pct=round(target_weight, 1),
            ))

    actions.sort(key=lambda a: ({"BUY": 0, "SELL": 1, "HOLD": 2}.get(a.action, 3), a.signal_rank))
    return actions, skipped_cooldowns
