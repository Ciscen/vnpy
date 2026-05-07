#!/bin/bash
# HS300 飞书机器人启动脚本
#
# 用法: ./hs300_topk/run_bot.sh
#
# 功能:
#   - 通过飞书 WebSocket 长连接接收命令
#   - 支持 /rerun /retrain /ls /fetch /status /log /signal /health /help
#   - 断线自动重连（lark-oapi SDK 内置）
#
# 环境变量 (.env):
#   FEISHU_APP_ID / FEISHU_APP_SECRET — 必需
#   FEISHU_DOC_ID / FEISHU_CHAT_ID   — 部分命令需要

set -e

cd "$(dirname "$0")/.."

if [ ! -f .env ]; then
    echo "❌ 错误: 未找到 .env 文件，请先创建"
    echo "   需要: FEISHU_APP_ID, FEISHU_APP_SECRET"
    exit 1
fi

source .env

echo "========================================"
echo "  HS300 Top-K 飞书机器人"
echo "  启动时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo "  App ID: ${FEISHU_APP_ID:0:8}..."
echo "========================================"

exec .venv/bin/python -m hs300_topk.live.bot
