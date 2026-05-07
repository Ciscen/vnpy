"""
hs300_top10/run_live.py

生产执行入口 — 每周一开盘前生成交易建议并推送飞书。

用法::

    # 正常执行（自动判断是否为调仓日）
    python -m hs300_top10.run_live

    # 强制重训模型
    python -m hs300_top10.run_live --retrain

    # 只计算信号，不更新持仓状态
    python -m hs300_top10.run_live --dry-run

    # 指定日期（调试用）
    python -m hs300_top10.run_live --date 2026-05-05

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
from dataclasses import asdict
from datetime import date, datetime, timedelta
from pathlib import Path

import polars as pl

from hs300_top10.pipeline_config import PIPELINE_LIVE
from hs300_top10.strategy.config import OPTIMIZED_V13

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("run_live")

SIGNAL_DIR = Path(__file__).parent / "live" / "signals"
LOG_DIR = Path(__file__).parent / "live" / "logs"
CONFIG = OPTIMIZED_V13


# ══════════════════════════════════════════════════
# 交易日判断
# ══════════════════════════════════════════════════

def is_trading_day(d: date) -> bool:
    """简单判断：工作日即交易日（不处理节假日）。

    生产环境可替换为 akshare 交易日历查询。
    """
    return d.weekday() < 5


def is_first_monday_of_month(d: date) -> bool:
    """判断是否为当月第一个周一。"""
    if d.weekday() != 0:
        return False
    return d.day <= 7


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
# 构建交易建议 JSON
# ══════════════════════════════════════════════════

def build_signal_json(
    signal_date: date,
    actions: list,
    portfolio,
    prices: dict[str, float],
    model_info: dict,
    skipped_cooldowns: list[dict] | None = None,
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
    }


def _calc_stale_hours(updated_at: str) -> float:
    """计算持仓数据的陈旧小时数。"""
    if not updated_at:
        return 999
    try:
        updated = datetime.fromisoformat(updated_at)
        delta = datetime.now() - updated
        return delta.total_seconds() / 3600
    except (ValueError, TypeError):
        return 999


# ══════════════════════════════════════════════════
# 飞书通知
# ══════════════════════════════════════════════════

def notify_feishu(signal_json: dict) -> None:
    """通过飞书推送交易建议卡片。"""
    chat_id = os.environ.get("FEISHU_CHAT_ID", "")
    if not chat_id:
        logger.warning("FEISHU_CHAT_ID 未设置，跳过飞书推送")
        return

    client = None
    try:
        from hs300_top10.live.feishu import FeishuClient, build_rebalance_card

        client = FeishuClient.from_env()
        card = build_rebalance_card(
            signal_date=signal_json["date"],
            actions=signal_json["actions"],
            summary=signal_json["summary"],
            model_info=signal_json["model_info"],
            portfolio_before=signal_json.get("portfolio_before"),
            portfolio_after=signal_json.get("portfolio_after"),
            skipped_cooldowns=signal_json.get("skipped_cooldowns"),
        )
        client.send_card_message(chat_id, card)
        logger.info("飞书卡片消息推送成功")
    except Exception as e:
        logger.error("飞书推送失败: %s", e)
        if client:
            try:
                client.send_text_message(
                    chat_id,
                    f"[HS300 V1.3] {signal_json['date']} 调仓建议已生成，"
                    f"买{signal_json['summary']['buys']}卖{signal_json['summary']['sells']}"
                    f"持{signal_json['summary']['holds']}，详见本地日志。",
                )
            except Exception:
                pass


# ══════════════════════════════════════════════════
# 主流程
# ══════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(description="HS300 V1.3 生产执行")
    parser.add_argument("--date", type=str, default=None,
                        help="指定执行日期 (YYYY-MM-DD)，默认今天")
    parser.add_argument("--retrain", action="store_true",
                        help="强制重新训练模型")
    parser.add_argument("--dry-run", action="store_true",
                        help="只计算信号，不更新持仓/不推送")
    parser.add_argument("--skip-download", action="store_true",
                        help="跳过数据下载")
    parser.add_argument("--force-run", action="store_true",
                        help="忽略交易日/周一判断，强制执行")
    args = parser.parse_args()

    today = date.fromisoformat(args.date) if args.date else date.today()
    logger.info("=" * 60)
    logger.info("  HS300 V1.3 生产执行 | %s (%s)", today, "周" + "一二三四五六日"[today.weekday()])
    logger.info("=" * 60)

    # ── 交易日判断 ──
    if not args.force_run:
        if not is_trading_day(today):
            logger.info("非交易日，跳过")
            return
        if today.weekday() != 0:
            logger.info("非周一，V1.3 不调仓，跳过")
            return

    # ── Phase 1: 增量下载 ──
    resolved = PIPELINE_LIVE.resolve(ref_date=today)
    lab_path = PIPELINE_LIVE.lab_path

    if not args.skip_download:
        logger.info("Phase 1: 增量下载最新数据 ...")
        try:
            from hs300_top10.data.downloader import phase_download
            phase_download(
                lab_path=lab_path,
                data_start=resolved.data_start,
                data_end=resolved.data_end,
            )
        except Exception as e:
            logger.error("数据下载失败: %s", e)
            logger.info("尝试使用已有数据继续 ...")
    else:
        logger.info("Phase 1: 跳过下载 (--skip-download)")

    # ── Phase 2: 训练 / 加载信号 ──
    need_retrain = args.retrain or is_first_monday_of_month(today)

    if need_retrain:
        logger.info("Phase 2: 月度重新训练模型 ...")
        from hs300_top10.model.rolling_trainer import predict_live
        signal_df = predict_live(
            target_date=today,
            lab_path=lab_path,
            data_start=resolved.data_start,
            train_years=PIPELINE_LIVE.train_years,
        )
        train_cutoff = (today - timedelta(days=8)).isoformat()
    else:
        logger.info("Phase 2: 使用缓存信号 ...")
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
                    logger.info("  使用最近信号日期: %s", latest_dt.date())
            train_cutoff = "cached"
        else:
            logger.info("  缓存不存在，执行实时训练 ...")
            from hs300_top10.model.rolling_trainer import predict_live
            signal_df = predict_live(
                target_date=today,
                lab_path=lab_path,
                data_start=resolved.data_start,
                train_years=PIPELINE_LIVE.train_years,
            )
            train_cutoff = (today - timedelta(days=8)).isoformat()

    if signal_df.is_empty():
        logger.error("未能生成有效信号，退出")
        sys.exit(1)

    signals = signal_df.to_dicts()
    logger.info("Phase 2 完成: %d 只股票信号", len(signals))
    logger.info("  Top-5: %s", [(s["vt_symbol"], f'{s["signal"]:.3f}') for s in signals[:5]])

    # ── Phase 3: 加载当前持仓 ──
    logger.info("Phase 3: 加载当前持仓 ...")
    from hs300_top10.live.portfolio import load_portfolio, compute_rebalance
    portfolio = load_portfolio()
    logger.info("  持仓: %d 只, 现金: %.2f, 总值: %.2f",
                len(portfolio.positions), portfolio.cash, portfolio.total_value)

    # ── Phase 4: 获取昨收价格 ──
    logger.info("Phase 4: 获取参考价格 ...")
    from hs300_top10.data.loader import discover_symbols
    vt_symbols = discover_symbols(lab_path)
    prices = fetch_prev_close_prices(lab_path, vt_symbols)
    logger.info("  获取 %d 只股票价格", len(prices))

    # ── Phase 5: 计算调仓差异 ──
    logger.info("Phase 5: 计算调仓差异 ...")
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
    logger.info("  买入 %d, 卖出 %d, 持有 %d", len(buys), len(sells), len(holds))

    for a in actions:
        pnl_str = f" ({'+' if a.current_pnl_pct >= 0 else ''}{a.current_pnl_pct:.1f}%)" if a.action != "BUY" else ""
        logger.info("    %s %-6s %s %d股 @%.2f (%.2f~%.2f)%s [信号:%.3f #%d 市值:%.0f 占比:%.1f%%]",
                     a.action, a.symbol[:6], a.name, a.shares,
                     a.ref_price, a.price_low, a.price_high,
                     pnl_str, a.signal_prob, a.signal_rank,
                     a.market_value, a.weight_pct)

    if skipped_cooldowns:
        logger.info("  冷却中跳过: %s", [(s["vt_symbol"], f'{s["signal"]:.3f}', f'{s["remaining_days"]}天') for s in skipped_cooldowns])

    # ── Phase 6: 输出交易建议 ──
    model_info = {
        "train_cutoff": train_cutoff,
        "signal_date": today.isoformat(),
        "signal_count": len(signals),
    }
    signal_json = build_signal_json(
        today, actions, portfolio, prices, model_info, skipped_cooldowns,
    )

    SIGNAL_DIR.mkdir(parents=True, exist_ok=True)
    signal_path = SIGNAL_DIR / f"{today.isoformat()}.json"
    signal_path.write_text(
        json.dumps(signal_json, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    logger.info("Phase 6: 交易建议 -> %s", signal_path)

    # ── Phase 7: 飞书通知 ──
    if not args.dry_run:
        logger.info("Phase 7: 推送飞书 ...")
        notify_feishu(signal_json)
    else:
        logger.info("Phase 7: dry-run 模式，跳过推送")

    logger.info("=" * 60)
    logger.info("  执行完成!")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
