#!/bin/bash
# HS300 手动执行脚本 — 适用于漏执行或需要手动触发的场景
# 用法: ./hs300_top10/run_now.sh
#
# 特性:
#   - 自动加载 .env 环境变量
#   - --force-run: 跳过交易日/周一判断
#   - 幂等安全: 重复运行会覆盖信号文件，但不会重复推送飞书
#     （除非加 --force-push）
#   - 节假日运行安全: 使用最近交易日数据

set -e

cd "$(dirname "$0")/.."

if [ ! -f .env ]; then
    echo "❌ 错误: 未找到 .env 文件，请先创建（参考 hs300_top10/live/README.md）"
    exit 1
fi

source .env

# 确保日志目录存在
mkdir -p hs300_top10/live/logs

TODAY=$(date '+%Y-%m-%d')
WEEKDAY=$(date '+%u')  # 1=周一 ... 7=周日
WEEKDAY_CN=("" "一" "二" "三" "四" "五" "六" "日")

echo "========================================"
echo "  HS300 Top-10 手动执行"
echo "  日期: $TODAY (周${WEEKDAY_CN[$WEEKDAY]})"
echo "  时间: $(date '+%H:%M:%S')"
echo "========================================"

if [ "$WEEKDAY" -ge 6 ]; then
    echo "⚠ 今天是周末，将使用最近交易日的数据生成建议"
fi

.venv/bin/python -m hs300_top10.run_live --force-run 2>&1 | tee -a hs300_top10/live/logs/live.log

echo ""
echo "执行完成，日志已追加到 hs300_top10/live/logs/live.log"
echo "信号文件: hs300_top10/live/signals/${TODAY}.json"
