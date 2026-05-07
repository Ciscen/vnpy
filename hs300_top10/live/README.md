# HS300 Top-10 V1.3 生产部署指南

## 概述

每周一开盘前自动生成交易建议，通过飞书推送到你手机。

```
cron 定时触发 → 增量下载数据 → 加载/训练模型 → 生成信号
    → 读取飞书持仓 → 计算买卖差异 → 推送飞书卡片
    → 你手动交易 → 更新飞书持仓表格 → 闭环
```

## 1. 环境准备

```bash
# 服务器上 clone 项目并安装依赖
cd /path/to/vnpy
pip install -r requirements.txt

# 确保数据已下载（首次需要全量下载，约 30 分钟）
python -m hs300_top10.run_pipeline --config v1.3
```

## 2. 环境变量配置

创建 `/path/to/vnpy/.env` 文件（不入 git）:

```bash
# 飞书开放平台应用凭证
FEISHU_APP_ID=cli_xxxxxxxxxxxx
FEISHU_APP_SECRET=xxxxxxxxxxxxxxxxxxxxxxxx

# 飞书持仓文档 ID（从文档 URL 中获取）
# 例如 https://xxx.feishu.cn/docx/ABC123DEF456 中的 ABC123DEF456
FEISHU_DOC_ID=xxxxxxxxxxxxxxxx

# 飞书推送目标（群 chat_id 或个人 chat_id）
FEISHU_CHAT_ID=oc_xxxxxxxxxxxxxxxx
```

加载方式（二选一）:

```bash
# 方式 A: 在 crontab 中 source
source /path/to/vnpy/.env && python -m hs300_top10.run_live

# 方式 B: 在 .bashrc / .zshrc 中 export
export $(cat /path/to/vnpy/.env | xargs)
```

## 3. 飞书文档持仓表格

在飞书中创建一个文档，插入如下格式的表格:

| 股票代码 | 股票名称 | 持仓数量 | 成本价 | 买入日期 | 备注 |
|----------|----------|----------|--------|----------|------|
| 300394   | 天孚通信 | 200      | 25.30  | 2026-04-28 | |
| 600519   | 贵州茅台 | 100      | 1850.00| 2026-04-21 | |
| 可用现金 | 85000    |          |        |          | |

要求:
- 第一行为表头，包含"代码"、"名称"、"数量"、"成本"、"日期"等关键字
- 每行一只股票，代码为 6 位数字
- 最后一行用"可用现金"标识账户余额
- 手动交易后及时更新表格

## 4. Cron 调度配置

```bash
crontab -e
```

添加以下内容:

```cron
# HS300 V1.3 周度调仓 — 每个工作日 8:30 执行（非周一会自动跳过）
30 8 * * 1-5 cd /path/to/vnpy && source .env && .venv/bin/python -m hs300_top10.run_live >> hs300_top10/live/logs/live.log 2>&1

# 每月 1 号 7:00 强制重训模型（如 1 号非交易日，脚本内部会跳过）
0 7 1 * * cd /path/to/vnpy && source .env && .venv/bin/python -m hs300_top10.run_live --retrain >> hs300_top10/live/logs/retrain.log 2>&1
```

## 5. CLI 参数

```bash
# 正常执行（自动判断是否为周一交易日）
python -m hs300_top10.run_live

# 强制重训模型（不管是不是月初）
python -m hs300_top10.run_live --retrain

# 只计算信号，不推送飞书（调试用）
python -m hs300_top10.run_live --dry-run

# 指定日期执行（调试用）
python -m hs300_top10.run_live --date 2026-05-05 --force-run

# 跳过数据下载（使用已有数据）
python -m hs300_top10.run_live --skip-download

# 组合使用
python -m hs300_top10.run_live --date 2026-05-05 --force-run --skip-download --dry-run
```

## 6. 输出文件

```
hs300_top10/live/
  signals/
    2026-05-05.json      # 每次执行的交易建议（JSON）
  state/
    portfolio_cache.json  # 飞书持仓的本地快照（降级备用）
  logs/
    live.log              # 执行日志
    retrain.log           # 重训日志
```

## 7. 飞书应用权限

在飞书开放平台 (https://open.feishu.cn) 的应用后台，确保开通以下权限:

- `docx:document:readonly` — 读取文档内容（持仓表格）
- `im:message:send_as_bot` — 以机器人身份发送消息

并将机器人添加到目标群中。

## 8. 故障处理

| 情况 | 处理方式 |
|------|----------|
| 飞书 API 不可用 | 自动降级到本地 portfolio_cache.json |
| 信号缓存不存在 | 自动触发实时训练（耗时约 10 分钟） |
| 数据下载失败 | 使用已有数据继续执行 |
| 非交易日执行 | 自动跳过，不做任何操作 |
| 非周一执行 | 自动跳过（V1.3 为周频策略） |
