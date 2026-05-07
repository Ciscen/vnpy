"""
hs300_topk/live/bot.py

飞书机器人服务 — 通过 WebSocket 长连接接收命令，触发策略操作。

支持命令:
    /rerun          重跑当日策略（--force-run --skip-download）
    /rerun full     重跑当日策略（含数据下载）
    /retrain        强制重训模型 + 重跑
    /ls             查看 hs300_topk 文件树（深度4）
    /fetch <path>   发送指定文件/文件夹（zip）
    /status         当前持仓概况 + 最新信号日期
    /log [date]     查看指定日期日志（默认今天）
    /signal [date]  查看指定日期交易建议摘要
    /health         系统健康检查
    /help           显示命令帮助

用法::

    source .env && .venv/bin/python -m hs300_topk.live.bot

环境变量::

    FEISHU_APP_ID        飞书应用 App ID
    FEISHU_APP_SECRET    飞书应用 App Secret
"""
from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import tempfile
import threading
from datetime import date, datetime
from pathlib import Path

import lark_oapi as lark
from lark_oapi.api.im.v1 import (
    CreateFileRequest,
    CreateFileRequestBody,
    CreateMessageRequest,
    CreateMessageRequestBody,
    CreateMessageReactionRequest,
    CreateMessageReactionRequestBody,
    Emoji,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("bot")

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
STRATEGY_ROOT = Path(__file__).resolve().parent.parent
SIGNAL_DIR = STRATEGY_ROOT / "live" / "signals"
LOG_DIR = STRATEGY_ROOT / "live" / "logs"

_rerun_lock = threading.Lock()


# ══════════════════════════════════════════════════
# 飞书 SDK 客户端
# ══════════════════════════════════════════════════

def _build_client() -> lark.Client:
    return (
        lark.Client.builder()
        .app_id(os.environ["FEISHU_APP_ID"])
        .app_secret(os.environ["FEISHU_APP_SECRET"])
        .log_level(lark.LogLevel.INFO)
        .build()
    )


CLIENT: lark.Client | None = None


def get_client() -> lark.Client:
    global CLIENT
    if CLIENT is None:
        CLIENT = _build_client()
    return CLIENT


# ══════════════════════════════════════════════════
# 消息发送工具
# ══════════════════════════════════════════════════

def _send_to_chat(chat_id: str, text: str) -> None:
    """向 chat 发送纯文本消息。"""
    body = CreateMessageRequestBody.builder() \
        .receive_id(chat_id) \
        .msg_type("text") \
        .content(json.dumps({"text": text})) \
        .build()
    req = CreateMessageRequest.builder() \
        .receive_id_type("chat_id") \
        .request_body(body) \
        .build()
    get_client().im.v1.message.create(req)


def _send_post(chat_id: str, title: str, content_lines: list[list[dict]]) -> None:
    """发送富文本 post 消息。"""
    post = {
        "zh_cn": {
            "title": title,
            "content": content_lines,
        }
    }
    body = CreateMessageRequestBody.builder() \
        .receive_id(chat_id) \
        .msg_type("post") \
        .content(json.dumps(post, ensure_ascii=False)) \
        .build()
    req = CreateMessageRequest.builder() \
        .receive_id_type("chat_id") \
        .request_body(body) \
        .build()
    get_client().im.v1.message.create(req)


def _send_file(chat_id: str, file_path: Path, file_name: str | None = None) -> None:
    """上传文件并发送到 chat。"""
    fname = file_name or file_path.name
    with open(file_path, "rb") as f:
        upload_body = CreateFileRequestBody.builder() \
            .file_type("stream") \
            .file_name(fname) \
            .file(f) \
            .build()
        upload_req = CreateFileRequest.builder() \
            .request_body(upload_body) \
            .build()
        upload_resp = get_client().im.v1.file.create(upload_req)

    if not upload_resp.success():
        logger.error("文件上传失败: %s", upload_resp.msg)
        _send_to_chat(chat_id, f"文件上传失败: {upload_resp.msg}")
        return

    file_key = upload_resp.data.file_key
    content = json.dumps({"file_key": file_key})
    body = CreateMessageRequestBody.builder() \
        .receive_id(chat_id) \
        .msg_type("file") \
        .content(content) \
        .build()
    req = CreateMessageRequest.builder() \
        .receive_id_type("chat_id") \
        .request_body(body) \
        .build()
    get_client().im.v1.message.create(req)


def _add_reaction(message_id: str, emoji: str = "THUMBSUP") -> None:
    """给消息添加表情回复。"""
    body = CreateMessageReactionRequestBody.builder() \
        .reaction_type(Emoji.builder().emoji_type(emoji).build()) \
        .build()
    req = CreateMessageReactionRequest.builder() \
        .message_id(message_id) \
        .request_body(body) \
        .build()
    try:
        get_client().im.v1.message_reaction.create(req)
    except Exception as e:
        logger.warning("添加表情失败: %s", e)


# ══════════════════════════════════════════════════
# 命令处理器
# ══════════════════════════════════════════════════

def cmd_help(chat_id: str, _args: str) -> None:
    content = [
        [{"tag": "text", "text": "🚀 执行类\n"}],
        [
            {"tag": "text", "text": "/rerun"},
            {"tag": "text", "text": "  重跑当日策略（跳过下载）\n"},
        ],
        [
            {"tag": "text", "text": "/rerun full"},
            {"tag": "text", "text": "  含数据下载的完整重跑\n"},
        ],
        [
            {"tag": "text", "text": "/retrain"},
            {"tag": "text", "text": "  强制重训模型 + 重跑\n"},
        ],
        [{"tag": "text", "text": "\n📂 文件类\n"}],
        [
            {"tag": "text", "text": "/ls"},
            {"tag": "text", "text": "  查看 hs300_topk 文件树（深度4）\n"},
        ],
        [
            {"tag": "text", "text": "/fetch <path>"},
            {"tag": "text", "text": "  发送文件/文件夹（zip），路径相对 hs300_topk/\n"},
        ],
        [{"tag": "text", "text": "\n📊 查询类\n"}],
        [
            {"tag": "text", "text": "/status"},
            {"tag": "text", "text": "  持仓概况 + 最新信号日期\n"},
        ],
        [
            {"tag": "text", "text": "/log [YYYY-MM-DD]"},
            {"tag": "text", "text": "  查看指定日期日志（默认今天）\n"},
        ],
        [
            {"tag": "text", "text": "/signal [YYYY-MM-DD]"},
            {"tag": "text", "text": "  查看指定日期交易建议摘要\n"},
        ],
        [
            {"tag": "text", "text": "/health"},
            {"tag": "text", "text": "  系统健康检查（数据/模型/飞书）\n"},
        ],
        [{"tag": "text", "text": "\n❓ 其他\n"}],
        [
            {"tag": "text", "text": "/help"},
            {"tag": "text", "text": "  显示此帮助"},
        ],
    ]
    _send_post(chat_id, "📋 HS300 Top-K 机器人命令", content)


def cmd_rerun(chat_id: str, args: str) -> None:
    if not _rerun_lock.acquire(blocking=False):
        _send_to_chat(chat_id, "⚠ 正在执行中，请等待上一次完成")
        return

    try:
        cmd_args = ["--force-run"]
        if args.strip() != "full":
            cmd_args.append("--skip-download")

        _send_to_chat(chat_id, f"🚀 开始执行策略 ({' '.join(cmd_args)}) ...")

        result = subprocess.run(
            [str(PROJECT_ROOT / ".venv" / "bin" / "python"),
             "-m", "hs300_topk.run_live"] + cmd_args,
            capture_output=True, text=True, timeout=600,
            cwd=str(PROJECT_ROOT),
            env={**os.environ},
        )

        output = result.stdout[-2000:] if len(result.stdout) > 2000 else result.stdout
        if result.returncode == 0:
            _send_to_chat(chat_id, f"✅ 执行完成\n\n{output}")
        else:
            error = result.stderr[-1000:] if result.stderr else "无错误输出"
            _send_to_chat(chat_id, f"❌ 执行失败 (exit={result.returncode})\n\n{error}")
    except subprocess.TimeoutExpired:
        _send_to_chat(chat_id, "❌ 执行超时（10分钟），请检查日志")
    except Exception as e:
        _send_to_chat(chat_id, f"❌ 执行异常: {e}")
    finally:
        _rerun_lock.release()


def cmd_retrain(chat_id: str, _args: str) -> None:
    if not _rerun_lock.acquire(blocking=False):
        _send_to_chat(chat_id, "⚠ 正在执行中，请等待上一次完成")
        return

    try:
        _send_to_chat(chat_id, "🔄 开始重训模型（可能需要几分钟）...")

        result = subprocess.run(
            [str(PROJECT_ROOT / ".venv" / "bin" / "python"),
             "-m", "hs300_topk.run_live", "--force-run", "--retrain"],
            capture_output=True, text=True, timeout=1200,
            cwd=str(PROJECT_ROOT),
            env={**os.environ},
        )

        output = result.stdout[-2000:] if len(result.stdout) > 2000 else result.stdout
        if result.returncode == 0:
            _send_to_chat(chat_id, f"✅ 重训+执行完成\n\n{output}")
        else:
            error = result.stderr[-1000:] if result.stderr else "无错误输出"
            _send_to_chat(chat_id, f"❌ 重训失败 (exit={result.returncode})\n\n{error}")
    except subprocess.TimeoutExpired:
        _send_to_chat(chat_id, "❌ 重训超时（20分钟），请检查日志")
    except Exception as e:
        _send_to_chat(chat_id, f"❌ 重训异常: {e}")
    finally:
        _rerun_lock.release()


def cmd_ls(chat_id: str, _args: str) -> None:
    lines: list[str] = []

    def _walk(path: Path, prefix: str, depth: int) -> None:
        if depth > 4:
            return
        entries = sorted(path.iterdir(), key=lambda p: (p.is_file(), p.name))
        dirs = [e for e in entries if e.is_dir() and e.name not in {
            "__pycache__", ".git", "node_modules", "stock_details",
        }]
        files = [e for e in entries if e.is_file() and not e.name.startswith(".")]
        items = dirs + files
        for i, item in enumerate(items):
            is_last = i == len(items) - 1
            connector = "└── " if is_last else "├── "
            if item.is_dir():
                lines.append(f"{prefix}{connector}{item.name}/")
                extension = "    " if is_last else "│   "
                _walk(item, prefix + extension, depth + 1)
            else:
                size = item.stat().st_size
                if size > 1024 * 1024:
                    size_str = f" ({size / 1024 / 1024:.1f}MB)"
                elif size > 1024:
                    size_str = f" ({size / 1024:.0f}KB)"
                else:
                    size_str = ""
                lines.append(f"{prefix}{connector}{item.name}{size_str}")

    lines.append("hs300_topk/")
    _walk(STRATEGY_ROOT, "", 1)

    content = [
        [{"tag": "text", "text": "📁 hs300_topk 文件结构\n\n"}],
        [{"tag": "text", "text": "\n".join(lines)}],
    ]
    _send_post(chat_id, "文件树", content)


def cmd_fetch(chat_id: str, args: str) -> None:
    rel_path = args.strip()
    if not rel_path:
        _send_to_chat(chat_id, "用法: /fetch <相对路径>\n例: /fetch live/signals/2026-05-07.json")
        return

    target = STRATEGY_ROOT / rel_path
    if not target.exists():
        _send_to_chat(chat_id, f"❌ 文件不存在: {rel_path}")
        return

    # 安全检查
    try:
        target.resolve().relative_to(STRATEGY_ROOT.resolve())
    except ValueError:
        _send_to_chat(chat_id, "❌ 路径越界，只能访问 hs300_topk 目录内的文件")
        return

    if target.is_file():
        _send_to_chat(chat_id, f"📤 发送文件: {rel_path}")
        _send_file(chat_id, target)
    elif target.is_dir():
        _send_to_chat(chat_id, f"📦 打包并发送文件夹: {rel_path}")
        with tempfile.TemporaryDirectory() as tmpdir:
            zip_base = Path(tmpdir) / target.name
            zip_path = shutil.make_archive(str(zip_base), "zip", str(target))
            _send_file(chat_id, Path(zip_path), f"{target.name}.zip")


def cmd_status(chat_id: str, _args: str) -> None:
    lines = ["📊 系统状态\n"]

    # 最新信号文件
    signal_files = sorted(SIGNAL_DIR.glob("*.json"), reverse=True)
    if signal_files:
        latest = signal_files[0]
        data = json.loads(latest.read_text(encoding="utf-8"))
        lines.append(f"最新信号: {data.get('date', '?')}")
        lines.append(f"  运行ID: {data.get('run_id', '?')}")
        lines.append(f"  生成时间: {data.get('generated_at', '?')[:19]}")
        summary = data.get("summary", {})
        lines.append(f"  操作: 买{summary.get('buys', 0)} 卖{summary.get('sells', 0)} 持{summary.get('holds', 0)}")
        pf = data.get("portfolio_before", {})
        lines.append(f"  账户总值: {pf.get('total_market_value', 0):,.0f}")
        lines.append(f"  可用资金: {pf.get('cash', 0):,.0f}")
        lines.append(f"  持仓数: {pf.get('position_count', 0)}")
    else:
        lines.append("最新信号: 无")

    # 本地持仓缓存
    cache = STRATEGY_ROOT / "live" / "state" / "portfolio_cache.json"
    if cache.exists():
        pf_data = json.loads(cache.read_text(encoding="utf-8"))
        lines.append(f"\n本地持仓缓存:")
        lines.append(f"  更新时间: {pf_data.get('updated_at', '?')[:19]}")
        lines.append(f"  现金: {pf_data.get('cash', 0):,.0f}")
        positions = pf_data.get("positions", [])
        lines.append(f"  持仓: {len(positions)} 只")
        for p in positions:
            lines.append(f"    {p['symbol']} {p['name']} {p['shares']}股 @{p['cost']}")

    _send_to_chat(chat_id, "\n".join(lines))


def cmd_log(chat_id: str, args: str) -> None:
    log_date = args.strip() or date.today().isoformat()
    log_file = LOG_DIR / f"live_{log_date}.log"

    if not log_file.exists():
        available = sorted(LOG_DIR.glob("live_*.log"), reverse=True)[:5]
        avail_str = ", ".join(f.stem.replace("live_", "") for f in available)
        _send_to_chat(chat_id, f"❌ 日志不存在: {log_date}\n可用日期: {avail_str}")
        return

    content = log_file.read_text(encoding="utf-8")
    if len(content) > 3500:
        content = "...(截断前部)\n" + content[-3500:]

    _send_post(chat_id, f"日志 {log_date}", [
        [{"tag": "text", "text": content}],
    ])


def cmd_signal(chat_id: str, args: str) -> None:
    sig_date = args.strip() or date.today().isoformat()
    sig_file = SIGNAL_DIR / f"{sig_date}.json"

    if not sig_file.exists():
        available = sorted(SIGNAL_DIR.glob("*.json"), reverse=True)[:5]
        avail_str = ", ".join(f.stem for f in available)
        _send_to_chat(chat_id, f"❌ 信号不存在: {sig_date}\n可用日期: {avail_str}")
        return

    data = json.loads(sig_file.read_text(encoding="utf-8"))
    lines = [f"📈 交易建议 {sig_date}\n"]

    summary = data.get("summary", {})
    lines.append(f"买{summary.get('buys', 0)} 卖{summary.get('sells', 0)} 持{summary.get('holds', 0)}")
    lines.append(f"预估手续费: {summary.get('total_fees', 0):,.0f}")
    lines.append(f"换手金额: {summary.get('estimated_turnover', 0):,.0f}\n")

    for a in data.get("actions", []):
        action = a["action"]
        symbol = a["symbol"][:6] if "." in a["symbol"] else a["symbol"]
        name = a.get("name", symbol)
        shares = a.get("shares", 0)
        price = a.get("ref_price", 0)
        reason = a.get("reason_text", "")

        if action == "BUY":
            lines.append(f"🟢 BUY  {name} {symbol} {shares}股 @{price:.2f}")
        elif action == "SELL":
            pnl = a.get("current_pnl_pct", 0)
            lines.append(f"🔴 SELL {name} {symbol} {shares}股 @{price:.2f} ({pnl:+.1f}%)")
        else:
            lines.append(f"⚪ HOLD {name} {symbol} {shares}股 @{price:.2f}")
        if reason:
            lines.append(f"   {reason}")

    issues = data.get("issues", [])
    if issues:
        lines.append(f"\n⚠ 告警 ({len(issues)} 条):")
        for issue in issues:
            lines.append(f"  {issue}")

    _send_to_chat(chat_id, "\n".join(lines))


def cmd_health(chat_id: str, _args: str) -> None:
    lines = ["🏥 系统健康检查\n"]
    all_ok = True

    # 1. 数据新鲜度
    from hs300_topk.pipeline_config import PIPELINE_LIVE
    lab_path = Path(PIPELINE_LIVE.lab_path)
    if not lab_path.is_absolute():
        lab_path = PROJECT_ROOT / lab_path
    daily_path = lab_path / "daily"
    if daily_path.exists():
        parquets = list(daily_path.glob("*.parquet"))
        if parquets:
            newest = max(parquets, key=lambda p: p.stat().st_mtime)
            age_hours = (datetime.now().timestamp() - newest.stat().st_mtime) / 3600
            status = "✅" if age_hours < 48 else "⚠"
            if age_hours >= 48:
                all_ok = False
            lines.append(f"{status} 数据: {len(parquets)} 只股票, 最新更新 {age_hours:.0f}小时前")
        else:
            lines.append("❌ 数据: 无 parquet 文件")
            all_ok = False
    else:
        lines.append(f"❌ 数据目录不存在: {daily_path}")
        all_ok = False

    # 2. 信号缓存
    cache = PIPELINE_LIVE.signal_cache
    if not cache.is_absolute():
        cache = PROJECT_ROOT / cache
    if cache.exists():
        cache_age = (datetime.now().timestamp() - cache.stat().st_mtime) / 3600 / 24
        status = "✅" if cache_age < 35 else "⚠"
        if cache_age >= 35:
            all_ok = False
        lines.append(f"{status} 信号缓存: {cache.name}, {cache_age:.0f}天前更新")
    else:
        lines.append("⚠ 信号缓存: 不存在（首次运行将自动训练）")

    # 3. 最新信号
    signal_files = sorted(SIGNAL_DIR.glob("*.json"), reverse=True)
    if signal_files:
        latest = signal_files[0]
        age_days = (date.today() - date.fromisoformat(latest.stem)).days
        status = "✅" if age_days <= 7 else "⚠"
        if age_days > 7:
            all_ok = False
        lines.append(f"{status} 最新信号: {latest.stem} ({age_days}天前)")
    else:
        lines.append("⚠ 最新信号: 无")

    # 4. 飞书连通性
    try:
        from hs300_topk.live.feishu import FeishuClient
        client = FeishuClient.from_env()
        _ = client.token
        lines.append("✅ 飞书 API: 连通")
    except Exception as e:
        lines.append(f"❌ 飞书 API: {e}")
        all_ok = False

    # 5. 环境变量
    required_envs = ["FEISHU_APP_ID", "FEISHU_APP_SECRET", "FEISHU_DOC_ID", "FEISHU_CHAT_ID"]
    missing = [k for k in required_envs if not os.environ.get(k)]
    if missing:
        lines.append(f"❌ 环境变量缺失: {', '.join(missing)}")
        all_ok = False
    else:
        lines.append("✅ 环境变量: 全部已设置")

    lines.append(f"\n{'✅ 整体健康' if all_ok else '⚠ 存在异常，请检查'}")
    _send_to_chat(chat_id, "\n".join(lines))


COMMANDS: dict[str, callable] = {
    "/help": cmd_help,
    "/rerun": cmd_rerun,
    "/retrain": cmd_retrain,
    "/ls": cmd_ls,
    "/fetch": cmd_fetch,
    "/status": cmd_status,
    "/log": cmd_log,
    "/signal": cmd_signal,
    "/health": cmd_health,
}


# ══════════════════════════════════════════════════
# 事件处理
# ══════════════════════════════════════════════════

def handle_message(data: lark.im.v1.P2ImMessageReceiveV1) -> None:
    """处理收到的消息事件。"""
    try:
        msg = data.event.message
        message_id = msg.message_id
        chat_id = msg.chat_id
        msg_type = msg.message_type

        # 只处理文本消息
        if msg_type != "text":
            return

        content = json.loads(msg.content)
        text = content.get("text", "").strip()
        if not text:
            return

        logger.info("收到消息: chat=%s text=%s", chat_id[:12], text[:50])

        # 第一时间表情回复确认已读
        _add_reaction(message_id, "EYES")

        # 解析命令
        parts = text.split(maxsplit=1)
        cmd = parts[0].lower()
        args = parts[1] if len(parts) > 1 else ""

        handler = COMMANDS.get(cmd)
        if handler:
            # 耗时命令在新线程执行，避免阻塞 WebSocket
            if cmd in ("/rerun", "/retrain"):
                threading.Thread(
                    target=handler, args=(chat_id, args),
                    daemon=True,
                ).start()
            else:
                handler(chat_id, args)
        else:
            _send_to_chat(chat_id, f"未知命令: {cmd}\n输入 /help 查看可用命令")

    except Exception as e:
        logger.error("处理消息异常: %s", e, exc_info=True)


# ══════════════════════════════════════════════════
# 主入口
# ══════════════════════════════════════════════════

def main() -> None:
    app_id = os.environ.get("FEISHU_APP_ID", "")
    app_secret = os.environ.get("FEISHU_APP_SECRET", "")
    if not app_id or not app_secret:
        logger.error("请设置 FEISHU_APP_ID 和 FEISHU_APP_SECRET")
        return

    event_handler = (
        lark.EventDispatcherHandler.builder("", "")
        .register_p2_im_message_receive_v1(handle_message)
        .build()
    )

    ws_client = lark.ws.Client(
        app_id, app_secret,
        event_handler=event_handler,
        log_level=lark.LogLevel.INFO,
    )

    logger.info("═" * 50)
    logger.info("  HS300 Top-K 飞书机器人启动")
    logger.info("  App ID: %s...", app_id[:8])
    logger.info("  策略目录: %s", STRATEGY_ROOT)
    logger.info("  等待消息 ...")
    logger.info("═" * 50)

    ws_client.start()


if __name__ == "__main__":
    main()
