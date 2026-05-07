"""
hs300_top10/live/portfolio.py

持仓管理模块 — 飞书文档表格读取 + 本地缓存降级 + 调仓差异计算。

飞书文档表格格式约定:
    | 股票代码 | 股票名称 | 持仓数量 | 成本价 | 买入日期 | 备注 |
    |----------|----------|----------|--------|----------|------|
    | 300394   | 天孚通信 | 200      | 25.30  | 2026-04-28 | |

    最后一行（或单独行）记录现金:
    | 可用现金 | 85000 | ... |
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
        return self.cash + sum(p.shares * p.cost for p in self.positions)


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


def _is_cash_row(row: list[str]) -> bool:
    first = row[0].strip() if row else ""
    return "现金" in first or "cash" in first.lower()


def parse_table_to_portfolio(table: list[list[str]]) -> Portfolio:
    """将飞书文档内嵌表格的二维矩阵解析为 Portfolio。

    约定第一行为表头，后续行为持仓或现金行。
    """
    if len(table) < 2:
        return Portfolio(updated_at=datetime.now().isoformat())

    header = [h.strip() for h in table[0]]
    col_map: dict[str, int] = {}
    keywords = {
        "代码": "code", "名称": "name", "数量": "shares",
        "成本": "cost", "买入": "entry_date", "日期": "entry_date",
        "现金": "cash", "备注": "note",
    }
    for idx, h in enumerate(header):
        for kw, field_name in keywords.items():
            if kw in h:
                col_map[field_name] = idx
                break

    positions: list[Position] = []
    cash = 0.0

    for row in table[1:]:
        if not row or all(not c.strip() for c in row):
            continue

        if _is_cash_row(row):
            for cell in row[1:]:
                v = _parse_float(cell)
                if v > 0:
                    cash = v
                    break
            continue

        code_idx = col_map.get("code", 0)
        code = row[code_idx].strip() if code_idx < len(row) else ""
        if not code or not re.match(r"^\d{6}", code):
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
            continue

        positions.append(Position(
            symbol=code, name=name, shares=shares,
            cost=cost, entry_date=entry_date,
        ))

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
        飞书文档 ID，默认从环境变量 FEISHU_DOC_ID 读取。
    table_index : int
        文档中第几个表格（0-based）。
    """
    import os
    from hs300_top10.live.feishu import FeishuClient

    doc_id = doc_id or os.environ.get("FEISHU_DOC_ID", "")
    if not doc_id:
        raise RuntimeError("请设置环境变量 FEISHU_DOC_ID")

    client = FeishuClient.from_env()
    tables = client.read_doc_tables(doc_id)

    if not tables:
        raise RuntimeError(f"文档 {doc_id} 中未找到表格")
    if table_index >= len(tables):
        raise RuntimeError(f"文档中只有 {len(tables)} 个表格，请求第 {table_index} 个")

    portfolio = parse_table_to_portfolio(tables[table_index])
    logger.info(
        "飞书持仓加载成功: %d 只股票, 现金 %.2f",
        len(portfolio.positions), portfolio.cash,
    )

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
        logger.warning("飞书持仓读取失败 (%s)，使用本地缓存", e)
        return load_portfolio_local()


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
    estimated_amount: float
    current_pnl_pct: float
    reason: str
    signal_prob: float
    signal_rank: int


def compute_rebalance(
    portfolio: Portfolio,
    signals: list[dict],
    prices: dict[str, float],
    top_k: int = 10,
    stock_cooldown_days: int = 10,
) -> list[RebalanceAction]:
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

    Returns
    -------
    list[RebalanceAction]
    """
    top_k_symbols = []
    for i, sig in enumerate(signals[:top_k]):
        sym = sig["vt_symbol"]
        if sym in portfolio.cooldowns and portfolio.cooldowns[sym] > 0:
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

        if vt_sym in top_k_symbols:
            actions.append(RebalanceAction(
                symbol=vt_sym, name=pos.name, action="HOLD",
                shares=pos.shares, ref_price=price,
                estimated_amount=0.0, current_pnl_pct=round(pnl_pct, 2),
                reason="still_top_k", signal_prob=sig.get("signal", 0.0),
                signal_rank=rank,
            ))
        else:
            actions.append(RebalanceAction(
                symbol=vt_sym, name=pos.name, action="SELL",
                shares=pos.shares, ref_price=price,
                estimated_amount=round(pos.shares * price, 2),
                current_pnl_pct=round(pnl_pct, 2),
                reason="rebalance_out", signal_prob=sig.get("signal", 0.0),
                signal_rank=rank,
            ))

    held_symbols = set(pos_map.keys())
    sell_proceeds = sum(a.estimated_amount for a in actions if a.action == "SELL")
    available_cash = portfolio.cash + sell_proceeds
    new_buy_count = top_k - sum(1 for a in actions if a.action == "HOLD")

    if new_buy_count > 0:
        per_stock_budget = available_cash / new_buy_count
        for vt_sym in top_k_symbols:
            if vt_sym in held_symbols:
                continue
            price = prices.get(vt_sym, 0.0)
            if price <= 0:
                continue
            shares = int(per_stock_budget / price / 100) * 100
            if shares <= 0:
                continue
            sig = signal_map.get(vt_sym, {})
            rank = signal_rank.get(vt_sym, 999)
            name = sig.get("name", vt_sym[:6])
            actions.append(RebalanceAction(
                symbol=vt_sym, name=name, action="BUY",
                shares=shares, ref_price=price,
                estimated_amount=round(shares * price, 2),
                current_pnl_pct=0.0, reason="new_entry",
                signal_prob=sig.get("signal", 0.0), signal_rank=rank,
            ))

    actions.sort(key=lambda a: ({"BUY": 0, "SELL": 1, "HOLD": 2}.get(a.action, 3), a.signal_rank))
    return actions
