"""
hs300_topk/run_live.py

生产执行入口 — 每周一开盘前生成交易建议并推送飞书。
当前策略版本由 CONFIG 决定（可在代码中切换）。

用法::

    # 正常执行（自动判断是否为调仓日）
    python -m hs300_topk.run_live

    # 强制重训模型
    python -m hs300_topk.run_live --retrain

    # 只计算信号，不更新持仓状态
    python -m hs300_topk.run_live --dry-run

    # 指定日期（调试用）
    python -m hs300_topk.run_live --date 2026-05-05

    # 强制执行 + 跳过下载（手动补跑最常用）
    python -m hs300_topk.run_live --force-run --skip-download

环境变量::

    FEISHU_APP_ID        飞书应用 App ID
    FEISHU_APP_SECRET    飞书应用 App Secret
    FEISHU_DOC_ID        飞书持仓文档 ID
    FEISHU_CHAT_ID       飞书推送目标 chat_id
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
import uuid
from dataclasses import asdict
from datetime import date, datetime, timedelta
from pathlib import Path

import polars as pl

from hs300_topk.pipeline_config import PIPELINE_LIVE
from hs300_topk.strategy.config import OPTIMIZED_V14

LOG_FORMAT = "%(asctime)s [%(levelname)s] %(message)s"
LOG_DATEFMT = "%Y-%m-%d %H:%M:%S"

logging.basicConfig(level=logging.INFO, format=LOG_FORMAT, datefmt=LOG_DATEFMT)
logger = logging.getLogger("run_live")

SIGNAL_DIR = Path(__file__).parent / "live" / "signals"
LOG_DIR = Path(__file__).parent / "live" / "logs"
CONFIG = OPTIMIZED_V14


# ══════════════════════════════════════════════════
# 交易日判断
# ══════════════════════════════════════════════════

_TRADING_CAL_CACHE: set[date] | None = None


def _load_trading_calendar() -> set[date]:
    """加载 A 股交易日历（akshare），缓存到模块级变量。"""
    global _TRADING_CAL_CACHE
    if _TRADING_CAL_CACHE is not None:
        return _TRADING_CAL_CACHE
    try:
        import akshare as ak
        df = ak.tool_trade_date_hist_sina()
        _TRADING_CAL_CACHE = {
            d.date() if hasattr(d, "date") else d
            for d in df["trade_date"]
        }
        logger.debug("交易日历加载成功: %d 天", len(_TRADING_CAL_CACHE))
    except Exception as e:
        logger.warning("交易日历加载失败 (%s)，降级为工作日判断", e)
        _TRADING_CAL_CACHE = set()
    return _TRADING_CAL_CACHE


def is_trading_day(d: date) -> bool:
    """判断是否为 A 股交易日。优先查交易日历，降级为工作日判断。"""
    cal = _load_trading_calendar()
    if cal:
        return d in cal
    return d.weekday() < 5


def is_first_rebalance_of_month(d: date) -> bool:
    """判断是否为当月第一次调仓日。

    逻辑：找到当月第一个周一，如果该周一是交易日则为调仓日；
    否则顺延到该周第一个交易日。
    """
    first_of_month = d.replace(day=1)
    days_to_monday = (7 - first_of_month.weekday()) % 7
    first_monday = first_of_month + timedelta(days=days_to_monday)

    for offset in range(5):
        candidate = first_monday + timedelta(days=offset)
        if is_trading_day(candidate):
            return d == candidate
    return False


# ══════════════════════════════════════════════════
# 获取昨收价格
# ══════════════════════════════════════════════════

def fetch_prev_close_prices(
    lab_path: str, vt_symbols: list[str]
) -> dict[str, float]:
    """从 AlphaLab parquet 文件读取最新收盘价。"""
    prices: dict[str, float] = {}
    daily_path = Path(lab_path) / "daily"
    for sym in vt_symbols:
        pq = daily_path / f"{sym}.parquet"
        if not pq.exists():
            continue
        try:
            df = pl.read_parquet(pq)
            if df.is_empty():
                continue
            last_row = df.sort("datetime").tail(1)
            close_col = "close_price" if "close_price" in df.columns else "close"
            prices[sym] = float(last_row[close_col][0])
        except Exception:
            continue
    return prices


# ══════════════════════════════════════════════════
# 执行告警收集器
# ══════════════════════════════════════════════════

class IssueCollector:
    """收集执行过程中的告警和异常，附到最终飞书推送中。"""

    def __init__(self) -> None:
        self._items: list[str] = []

    def warn(self, msg: str) -> None:
        self._items.append(f"⚠️ {msg}")
        logger.warning(msg)

    def error(self, msg: str) -> None:
        self._items.append(f"❌ {msg}")
        logger.error(msg)

    @property
    def items(self) -> list[str]:
        return list(self._items)

    @property
    def has_issues(self) -> bool:
        return len(self._items) > 0


# ══════════════════════════════════════════════════
# 上次调仓对账
# ══════════════════════════════════════════════════

def compare_with_last_signal(
    portfolio: "Portfolio",
    today: date,
) -> dict | None:
    """对比上次信号建议 vs 当前实际持仓，发现未执行的操作。

    Returns
    -------
    dict | None
        对账结果，包含 matched/unexecuted/unexpected 三类条目；
        如果没有上次信号则返回 None。
    """
    signal_files = sorted(SIGNAL_DIR.glob("*.json"), reverse=True)
    prev_file = None
    for f in signal_files:
        try:
            sig_date = date.fromisoformat(f.stem)
            if sig_date < today:
                prev_file = f
                break
        except ValueError:
            continue

    if prev_file is None:
        logger.info("  无历史信号文件，跳过对账")
        return None

    prev_data = json.loads(prev_file.read_text(encoding="utf-8"))
    prev_date = prev_data.get("date", prev_file.stem)
    prev_actions = prev_data.get("actions", [])
    logger.info("  对账基准: %s (run_id=%s)",
                prev_date, prev_data.get("run_id", "?"))

    current_syms = {p.vt_symbol for p in portfolio.positions}
    current_map = {p.vt_symbol: p for p in portfolio.positions}

    items: list[dict] = []

    for a in prev_actions:
        sym = a["symbol"]
        name = a.get("name", sym[:6])
        action = a["action"]

        if action == "SELL":
            if sym in current_syms:
                items.append({
                    "type": "unexecuted",
                    "action": "SELL",
                    "symbol": sym,
                    "name": name,
                    "shares": a.get("shares", 0),
                    "detail": "建议卖出但仍在持仓中",
                })
            else:
                items.append({
                    "type": "matched",
                    "action": "SELL",
                    "symbol": sym,
                    "name": name,
                    "detail": "已卖出",
                })

        elif action == "BUY":
            if sym in current_syms:
                pos = current_map[sym]
                expected = a.get("shares", 0)
                actual = pos.shares
                if expected > 0 and abs(actual - expected) > expected * 0.2:
                    items.append({
                        "type": "partial",
                        "action": "BUY",
                        "symbol": sym,
                        "name": name,
                        "expected_shares": expected,
                        "actual_shares": actual,
                        "detail": f"建议买{expected}股，实际{actual}股",
                    })
                else:
                    items.append({
                        "type": "matched",
                        "action": "BUY",
                        "symbol": sym,
                        "name": name,
                        "detail": f"已买入{actual}股",
                    })
            else:
                items.append({
                    "type": "unexecuted",
                    "action": "BUY",
                    "symbol": sym,
                    "name": name,
                    "shares": a.get("shares", 0),
                    "detail": "建议买入但未出现在持仓中",
                })

        elif action == "HOLD":
            if sym not in current_syms:
                items.append({
                    "type": "unexpected",
                    "action": "HOLD→GONE",
                    "symbol": sym,
                    "name": name,
                    "detail": "应持有但已不在持仓（手动卖出？）",
                })

    prev_all_syms = {a["symbol"] for a in prev_actions}
    for sym in current_syms:
        if sym not in prev_all_syms:
            pos = current_map[sym]
            items.append({
                "type": "unexpected",
                "action": "NEW",
                "symbol": sym,
                "name": pos.name,
                "shares": pos.shares,
                "detail": "上次信号中不存在（手动买入？）",
            })

    matched = [i for i in items if i["type"] == "matched"]
    issues = [i for i in items if i["type"] != "matched"]

    result = {
        "prev_date": prev_date,
        "prev_run_id": prev_data.get("run_id", "?"),
        "total_actions": len(prev_actions),
        "matched_count": len(matched),
        "issue_count": len(issues),
        "items": items,
    }

    if issues:
        logger.info("  对账发现 %d 条差异:", len(issues))
        for i in issues:
            logger.info("    [%s] %s %s: %s",
                        i["type"], i["action"], i["name"], i["detail"])
    else:
        logger.info("  对账完全匹配 (%d/%d)", len(matched), len(prev_actions))

    return result


# ══════════════════════════════════════════════════
# 构建交易建议 JSON
# ══════════════════════════════════════════════════

def build_signal_json(
    signal_date: date,
    actions: list,
    portfolio,
    prices: dict[str, float],
    model_info: dict,
    skipped_cooldowns: list[dict] | None = None,
    run_id: str = "",
    issues: list[str] | None = None,
    last_rebalance_review: dict | None = None,
) -> dict:
    """构建完整的交易建议 JSON。"""
    action_dicts = []
    for a in actions:
        d = asdict(a)
        if a.action == "BUY":
            d["estimated_cost"] = a.estimated_amount
        elif a.action == "SELL":
            d["estimated_proceeds"] = a.estimated_amount
        action_dicts.append(d)

    buys = [a for a in actions if a.action == "BUY"]
    sells = [a for a in actions if a.action == "SELL"]
    holds = [a for a in actions if a.action == "HOLD"]
    turnover = sum(a.estimated_amount for a in buys) + sum(a.estimated_amount for a in sells)
    total_fees = sum(a.fee for a in actions)
    total_mkt = portfolio.total_market_value(prices)
    pos_value = total_mkt - portfolio.cash

    sell_net = sum(a.net_amount for a in sells)
    buy_net = sum(a.net_amount for a in buys)

    return {
        "run_id": run_id,
        "generated_at": datetime.now().isoformat(),
        "date": signal_date.isoformat(),
        "strategy": CONFIG.version,
        "portfolio_before": {
            "cash": portfolio.cash,
            "position_value": round(pos_value, 2),
            "total_market_value": round(total_mkt, 2),
            "position_count": len(portfolio.positions),
            "updated_at": portfolio.updated_at,
            "stale_hours": _calc_stale_hours(portfolio.updated_at),
        },
        "portfolio_after": {
            "expected_cash": round(portfolio.cash + sell_net - buy_net, 2),
            "expected_positions": len(holds) + len(buys),
            "expected_total": round(total_mkt - total_fees, 2),
        },
        "actions": action_dicts,
        "skipped_cooldowns": skipped_cooldowns or [],
        "summary": {
            "buys": len(buys),
            "sells": len(sells),
            "holds": len(holds),
            "estimated_turnover": round(turnover, 2),
            "total_fees": round(total_fees, 2),
        },
        "model_info": model_info,
        "issues": issues or [],
        "last_rebalance_review": last_rebalance_review,
    }


def _calc_stale_hours(updated_at: str) -> float:
    """计算持仓数据的陈旧小时数。"""
    if not updated_at:
        return 999
    try:
        updated = datetime.fromisoformat(updated_at)
        delta = datetime.now() - updated
        return round(delta.total_seconds() / 3600, 1)
    except (ValueError, TypeError):
        return 999


# ══════════════════════════════════════════════════
# 飞书通知
# ══════════════════════════════════════════════════

def notify_feishu(signal_json: dict) -> bool:
    """通过飞书推送交易建议卡片。返回是否推送成功。"""
    chat_id = os.environ.get("FEISHU_CHAT_ID", "")
    if not chat_id:
        logger.warning("FEISHU_CHAT_ID 未设置，跳过飞书推送")
        return False

    client = None
    try:
        from hs300_topk.live.feishu import FeishuClient, build_rebalance_card

        client = FeishuClient.from_env()
        card = build_rebalance_card(
            signal_date=signal_json["date"],
            actions=signal_json["actions"],
            summary=signal_json["summary"],
            model_info=signal_json["model_info"],
            portfolio_before=signal_json.get("portfolio_before"),
            portfolio_after=signal_json.get("portfolio_after"),
            skipped_cooldowns=signal_json.get("skipped_cooldowns"),
            issues=signal_json.get("issues"),
            last_rebalance_review=signal_json.get("last_rebalance_review"),
        )
        client.send_card_message(chat_id, card)
        logger.info("飞书卡片消息推送成功 → chat_id=%s", chat_id[:8] + "...")
        return True
    except Exception as e:
        logger.error("飞书卡片推送失败: %s", e)
        if client:
            try:
                client.send_text_message(
                    chat_id,
                    f"[HS300 {CONFIG.version}] {signal_json['date']} 调仓建议已生成，"
                    f"买{signal_json['summary']['buys']}卖{signal_json['summary']['sells']}"
                    f"持{signal_json['summary']['holds']}，详见本地日志。",
                )
                logger.info("降级为纯文本消息推送成功")
                return True
            except Exception as e2:
                logger.error("纯文本推送也失败: %s", e2)
        return False


# ══════════════════════════════════════════════════
# 主流程
# ══════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(description=f"HS300 {CONFIG.version} 生产执行")
    parser.add_argument("--date", type=str, default=None,
                        help="指定执行日期 (YYYY-MM-DD)，默认今天")
    parser.add_argument("--retrain", action="store_true",
                        help="强制重新训练模型")
    parser.add_argument("--dry-run", action="store_true",
                        help="只计算信号，不推送飞书")
    parser.add_argument("--skip-download", action="store_true",
                        help="跳过数据下载")
    parser.add_argument("--force-run", action="store_true",
                        help="忽略交易日/周一判断，强制执行")
    args = parser.parse_args()

    run_id = uuid.uuid4().hex[:12]
    t_start = time.monotonic()
    today = date.fromisoformat(args.date) if args.date else date.today()
    issues = IssueCollector()

    # 按日期拆分日志文件
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(
        LOG_DIR / f"live_{today.isoformat()}.log",
        encoding="utf-8",
    )
    file_handler.setFormatter(logging.Formatter(LOG_FORMAT, datefmt=LOG_DATEFMT))
    logging.getLogger().addHandler(file_handler)

    logger.info("═" * 60)
    logger.info("  HS300 %s 生产执行", CONFIG.version.upper())
    logger.info("  日期: %s (%s)", today, "周" + "一二三四五六日"[today.weekday()])
    logger.info("  运行ID: %s", run_id)
    logger.info("  参数: force_run=%s, retrain=%s, dry_run=%s, "
                "skip_download=%s",
                args.force_run, args.retrain, args.dry_run,
                args.skip_download)
    logger.info("  策略: %s (%s), top_k=%d, cooldown=%d天",
                CONFIG.version, CONFIG.description, CONFIG.top_k,
                CONFIG.stock_cooldown_days)
    logger.info("═" * 60)

    # ── 交易日判断 ──
    if not args.force_run:
        if not is_trading_day(today):
            logger.info("⏭ 非交易日，跳过 (可用 --force-run 强制执行)")
            return

        monday_of_week = today - timedelta(days=today.weekday())
        if monday_of_week == today:
            logger.info("✅ 周一且是交易日，正常执行")
        elif is_trading_day(monday_of_week):
            logger.info("⏭ 周一是交易日但今天不是周一，跳过")
            return
        else:
            first_trading_day = None
            for offset in range(5):
                candidate = monday_of_week + timedelta(days=offset)
                if is_trading_day(candidate):
                    first_trading_day = candidate
                    break
            if today != first_trading_day:
                logger.info("⏭ 非本周首个交易日（首交易日=%s），跳过",
                            first_trading_day)
                return
            logger.info("✅ 周一(%s)非交易日，顺延至今日执行", monday_of_week)
    else:
        logger.info("✅ --force-run 模式，跳过交易日判断")

    # ── 信号文件存在检查 ──
    signal_path = SIGNAL_DIR / f"{today.isoformat()}.json"
    if signal_path.exists():
        existing = json.loads(signal_path.read_text(encoding="utf-8"))
        prev_run = existing.get("run_id", "unknown")
        prev_time = existing.get("generated_at", "unknown")
        issues.warn(f"今日信号文件已存在 (run_id={prev_run}, "
                    f"生成于 {prev_time})，本次执行将覆盖")

    # ── Phase 1: 增量下载 ──
    resolved = PIPELINE_LIVE.resolve(ref_date=today)
    lab_path = PIPELINE_LIVE.lab_path

    if not args.skip_download:
        t1 = time.monotonic()
        logger.info("Phase 1/7: 增量下载最新数据 (data_end=%s) ...",
                     resolved.data_end)
        try:
            from hs300_topk.data.downloader import phase_download
            symbols = phase_download(
                lab_path=lab_path,
                data_start=resolved.data_start,
                data_end=resolved.data_end,
            )
            logger.info("Phase 1 完成: %d 只股票数据已更新 (%.1fs)",
                        len(symbols), time.monotonic() - t1)
        except Exception as e:
            issues.error(f"Phase 1 数据下载失败: {e}，降级使用已有本地数据")
    else:
        logger.info("Phase 1/7: 跳过下载 (--skip-download)")

    # ── Phase 2: 训练 / 加载信号 ──
    t2 = time.monotonic()
    need_retrain = args.retrain or is_first_rebalance_of_month(today)
    retrain_reason = ("--retrain 参数" if args.retrain
                      else "月初首个调仓日" if is_first_rebalance_of_month(today)
                      else "无需重训")
    logger.info("Phase 2/7: 信号生成 (need_retrain=%s, 原因: %s)",
                need_retrain, retrain_reason)

    train_cutoff = "cached"
    if need_retrain:
        logger.info("  开始训练模型 (train_years=%d) ...",
                     PIPELINE_LIVE.train_years)
        from hs300_topk.model.rolling_trainer import predict_live
        signal_df = predict_live(
            target_date=today,
            lab_path=lab_path,
            data_start=resolved.data_start,
            train_years=PIPELINE_LIVE.train_years,
        )
        train_cutoff = (today - timedelta(days=8)).isoformat()
        logger.info("  训练完成, train_cutoff=%s", train_cutoff)
    else:
        cache_path = PIPELINE_LIVE.signal_cache
        if cache_path.exists():
            full_signal = pl.read_parquet(cache_path)
            target_dt = datetime(today.year, today.month, today.day)
            signal_df = full_signal.filter(
                pl.col("datetime") == pl.lit(target_dt)
            ).select(["vt_symbol", "signal"]).sort("signal", descending=True)

            if signal_df.is_empty():
                nearest = full_signal.filter(
                    pl.col("datetime") <= pl.lit(target_dt)
                ).sort("datetime", descending=True)
                if not nearest.is_empty():
                    latest_dt = nearest["datetime"][0]
                    signal_df = full_signal.filter(
                        pl.col("datetime") == pl.lit(latest_dt)
                    ).select(["vt_symbol", "signal"]).sort("signal", descending=True)
                    logger.info("  目标日期 %s 无信号，回退到最近信号日 %s",
                                today, latest_dt.date())
            logger.info("  缓存加载: %s (%d 行)", cache_path.name,
                        full_signal.height)
        else:
            logger.info("  缓存不存在 (%s)，执行实时训练 ...", cache_path)
            from hs300_topk.model.rolling_trainer import predict_live
            signal_df = predict_live(
                target_date=today,
                lab_path=lab_path,
                data_start=resolved.data_start,
                train_years=PIPELINE_LIVE.train_years,
            )
            train_cutoff = (today - timedelta(days=8)).isoformat()

    if signal_df.is_empty():
        logger.error("❌ 未能生成有效信号，退出")
        sys.exit(1)

    signals = signal_df.to_dicts()
    logger.info("Phase 2 完成: %d 只股票信号 (%.1fs)",
                len(signals), time.monotonic() - t2)
    for i, s in enumerate(signals[:5]):
        logger.info("  #%d %s signal=%.4f", i + 1, s["vt_symbol"], s["signal"])

    # ── Phase 3: 加载当前持仓 ──
    t3 = time.monotonic()
    logger.info("Phase 3/7: 加载当前持仓 ...")
    from hs300_topk.live.portfolio import load_portfolio, compute_rebalance
    portfolio = load_portfolio()

    if not portfolio.positions and portfolio.cash <= 0:
        issues.warn("持仓为空且现金为0，请检查飞书文档是否正确")

    stale = _calc_stale_hours(portfolio.updated_at)
    logger.info("  持仓: %d 只, 现金: %.2f, 成本总值: %.2f",
                len(portfolio.positions), portfolio.cash, portfolio.total_value)
    if portfolio.positions:
        for p in portfolio.positions:
            logger.info("    %s %s %d股 @%.2f 入=%s",
                        p.vt_symbol, p.name, p.shares, p.cost, p.entry_date)
    if stale > 48:
        issues.warn(f"持仓数据已 {stale:.0f} 小时未更新，建议先更新飞书文档")

    # 与上次信号对比
    logger.info("  与上次信号对比 ...")
    rebalance_review = compare_with_last_signal(portfolio, today)
    if rebalance_review and rebalance_review["issue_count"] > 0:
        issues.warn(
            f"上次调仓({rebalance_review['prev_date']})有 "
            f"{rebalance_review['issue_count']} 条操作未按预期执行"
        )
    logger.info("Phase 3 完成 (%.1fs)", time.monotonic() - t3)

    # ── Phase 4: 获取昨收价格 ──
    t4 = time.monotonic()
    logger.info("Phase 4/7: 获取参考价格 ...")
    from hs300_topk.data.loader import discover_symbols
    vt_symbols = discover_symbols(lab_path)
    prices = fetch_prev_close_prices(lab_path, vt_symbols)
    logger.info("  获取 %d 只股票价格", len(prices))

    missing_prices = [
        p.vt_symbol for p in portfolio.positions
        if p.vt_symbol not in prices
    ]
    if missing_prices:
        issues.warn(f"{len(missing_prices)} 只持仓股票无法获取价格 "
                    f"(将用成本价替代): {missing_prices}")

    total_mkt = portfolio.total_market_value(prices)
    logger.info("  账户市值: %.2f (现金 %.2f + 持仓 %.2f)",
                total_mkt, portfolio.cash, total_mkt - portfolio.cash)
    logger.info("Phase 4 完成 (%.1fs)", time.monotonic() - t4)

    # ── Phase 5: 计算调仓差异 ──
    t5 = time.monotonic()
    logger.info("Phase 5/7: 计算调仓差异 (top_k=%d, cooldown=%d天, slippage=1.5%%) ...",
                CONFIG.top_k, CONFIG.stock_cooldown_days)
    actions, skipped_cooldowns = compute_rebalance(
        portfolio=portfolio,
        signals=signals,
        prices=prices,
        top_k=CONFIG.top_k,
        stock_cooldown_days=CONFIG.stock_cooldown_days,
    )

    buys = [a for a in actions if a.action == "BUY"]
    sells = [a for a in actions if a.action == "SELL"]
    holds = [a for a in actions if a.action == "HOLD"]
    total_fees = sum(a.fee for a in actions)
    logger.info("  结果: 买入 %d, 卖出 %d, 持有 %d, 预估手续费 %.2f",
                len(buys), len(sells), len(holds), total_fees)

    for a in actions:
        if a.action == "SELL":
            logger.info("    SELL  %s %s %d股 @%.2f (%.2f~%.2f) "
                        "盈亏%+.1f%% 回款%.0f 费%.0f | %s",
                        a.symbol[:6], a.name, a.shares,
                        a.ref_price, a.price_low, a.price_high,
                        a.current_pnl_pct, a.net_amount, a.fee,
                        a.reason_text)
        elif a.action == "BUY":
            logger.info("    BUY   %s %s %d股 @%.2f (%.2f~%.2f) "
                        "成本%.0f 费%.0f 信号%.3f #%d | %s",
                        a.symbol[:6], a.name, a.shares,
                        a.ref_price, a.price_low, a.price_high,
                        a.net_amount, a.fee, a.signal_prob, a.signal_rank,
                        a.reason_text)
        else:
            logger.info("    HOLD  %s %s %d股 @%.2f 盈亏%+.1f%% "
                        "市值%.0f 占比%.1f%% %d天 | %s",
                        a.symbol[:6], a.name, a.shares,
                        a.ref_price, a.current_pnl_pct,
                        a.market_value, a.weight_pct, a.hold_days,
                        a.reason_text)

    if skipped_cooldowns:
        logger.info("  冷却中跳过 %d 只:", len(skipped_cooldowns))
        for s in skipped_cooldowns:
            logger.info("    %s signal=%.3f 剩余%d天",
                        s["vt_symbol"], s["signal"], s["remaining_days"])

    buy_total = sum(a.net_amount for a in buys)
    available_for_buy = portfolio.cash + sum(a.net_amount for a in sells)
    if buy_total > available_for_buy * 1.01:
        issues.error(f"资金校验失败: 买入总额 {buy_total:.2f} > "
                     f"可用资金 {available_for_buy:.2f}")

    logger.info("Phase 5 完成 (%.1fs)", time.monotonic() - t5)

    # ── Phase 6: 输出交易建议 ──
    t6 = time.monotonic()
    model_info = {
        "train_cutoff": train_cutoff,
        "signal_date": today.isoformat(),
        "signal_count": len(signals),
    }
    signal_json = build_signal_json(
        today, actions, portfolio, prices, model_info, skipped_cooldowns,
        run_id=run_id,
        issues=issues.items,
        last_rebalance_review=rebalance_review,
    )

    SIGNAL_DIR.mkdir(parents=True, exist_ok=True)
    signal_path.write_text(
        json.dumps(signal_json, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    logger.info("Phase 6/7: 交易建议 → %s", signal_path)

    # ── Phase 7: 飞书通知 ──
    t7 = time.monotonic()
    if args.dry_run:
        logger.info("Phase 7/7: dry-run 模式，跳过推送")
    else:
        logger.info("Phase 7/7: 推送飞书 ...")
        success = notify_feishu(signal_json)
        if success:
            logger.info("Phase 7 完成 (%.1fs)", time.monotonic() - t7)
        else:
            issues.error("飞书推送失败，请检查日志")

    # ── 执行总结 ──
    elapsed = time.monotonic() - t_start
    logger.info("═" * 60)
    logger.info("  执行完成!")
    logger.info("  运行ID: %s", run_id)
    logger.info("  耗时: %.1f 秒", elapsed)
    logger.info("  信号文件: %s", signal_path)
    logger.info("  操作摘要: 买%d 卖%d 持%d, 预估费用 %.2f",
                len(buys), len(sells), len(holds), total_fees)
    if issues.has_issues:
        logger.info("  告警/异常: %d 条", len(issues.items))
        for item in issues.items:
            logger.info("    %s", item)
    else:
        logger.info("  告警/异常: 无")
    logger.info("═" * 60)


if __name__ == "__main__":
    main()
