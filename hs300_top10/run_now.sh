#!/bin/bash
# HS300 V1.3 手动执行脚本 — 适用于漏执行或需要手动触发的场景
# 用法: ./hs300_top10/run_now.sh

set -e

cd "$(dirname "$0")/.."
source .env

echo "========================================"
echo "  HS300 V1.3 手动执行"
echo "  日期: $(date '+%Y-%m-%d %H:%M:%S')"
echo "========================================"

.venv/bin/python -m hs300_top10.run_live --force-run 2>&1 | tee -a hs300_top10/live/logs/live.log

echo ""
echo "执行完成，日志已追加到 hs300_top10/live/logs/live.log"
