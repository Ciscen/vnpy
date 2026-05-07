# HS300 Top-K 生产部署指南

## 架构概览

```
cron 定时触发 → 增量下载数据 → 加载/训练模型 → 生成信号
    → 读取飞书持仓 → 计算买卖差异 → 推送飞书卡片
    → 你手动交易 → 更新飞书持仓表格 → 闭环
```

## Step 1: 服务器环境准备

**最低配置**: 2核 CPU / 4GB 内存 / 20GB 磁盘 / Python 3.10+

```bash
# 1.1 克隆项目
git clone <your-repo-url> /opt/vnpy
cd /opt/vnpy

# 1.2 创建虚拟环境（如果本地 .venv 已有且平台相同，可跳过此步直接复制）
python3 -m venv .venv
source .venv/bin/activate

# 1.3 安装 vnpy 核心 + alpha 扩展依赖
pip install -e ".[alpha]"

# 1.4 安装 HS300 策略额外依赖（akshare, xgboost, requests 等）
pip install -r hs300_topk/requirements.txt
```

## Step 2: 首次数据初始化

首次部署需要全量下载历史数据（约 30 分钟）：

```bash
cd /opt/vnpy
source .venv/bin/activate

# 全量下载 + 滚动训练 + 回测验证
python -m hs300_topk.run_pipeline --config v1.4

# 验证输出
ls hs300_topk/output/v1.4/
# 应该看到: dashboard.html, statistics.json, trades.csv 等
```

## Step 3: 飞书应用配置

### 3.1 创建飞书应用

1. 登录 [飞书开放平台](https://open.feishu.cn)
2. 创建企业自建应用
3. 记录 **App ID** 和 **App Secret**

### 3.2 配置应用权限

在应用后台 → 权限管理，开通以下权限并提交审批：

| 权限 | 用途 |
|------|------|
| `docx:document:readonly` | 读取持仓文档 |
| `wiki:wiki:readonly` | 读取知识库文档（如果用知识库） |
| `sheets:spreadsheet` | 读取电子表格（如果用电子表格） |
| `im:message:send_as_bot` | 发送消息 |

### 3.3 添加机器人

- 在应用后台 → 机器人，启用机器人功能
- 在飞书群中 → 设置 → 群机器人 → 添加你创建的应用
- 或者直接发送到个人（使用 `ou_` 开头的 open_id）

### 3.4 获取 Chat ID

```bash
# 方法 1: 通过 API 获取（需先配置好环境变量 APP_ID/SECRET）
curl -X POST 'https://open.feishu.cn/open-apis/auth/v3/tenant_access_token/internal' \
  -H 'Content-Type: application/json' \
  -d '{"app_id":"YOUR_APP_ID","app_secret":"YOUR_SECRET"}'

# 用返回的 token 查询群列表
curl 'https://open.feishu.cn/open-apis/im/v1/chats' \
  -H 'Authorization: Bearer YOUR_TOKEN'

# 方法 2: 推送到个人
# 使用飞书管理后台获取你的 open_id（格式: ou_xxxxxxxx）
```

## Step 4: 创建飞书持仓文档

在飞书中创建文档（普通文档或电子表格均可），插入表格：

| 股票代码 | 股票名称 | 持仓数量 | 成本价 | 买入日期 | 可用资金 | 备注 |
|----------|----------|----------|--------|----------|----------|------|
| 300394   | 天孚通信 | 200      | 25.30  | 2026-04-28 | 85000 | |
| 600519   | 贵州茅台 | 100      | 1850.00| 2026-04-21 |       | |

**字段说明**：
- **成本价**: 填写持仓加权均价（非单次买入价），用于计算浮盈浮亏
- **可用资金**: 券商账户中当前可用于买入的资金余额。可作为列（只填一行）或独立行
- **总资产**: 可选列，仅展示用，不影响计算

**获取文档 ID**: 从飞书文档 URL 中提取
```
https://xxx.feishu.cn/docx/ABC123DEF456  → DOC_ID = ABC123DEF456
https://xxx.feishu.cn/wiki/XYZ789        → DOC_ID = XYZ789 (知识库，自动识别)
```

## Step 5: 配置环境变量

```bash
# 创建环境变量文件（不入 git）
cat > /opt/vnpy/.env << 'EOF'
export FEISHU_APP_ID=cli_xxxxxxxxxxxx
export FEISHU_APP_SECRET=xxxxxxxxxxxxxxxxxxxxxxxx
export FEISHU_DOC_ID=xxxxxxxxxxxxxxxx
export FEISHU_CHAT_ID=oc_xxxxxxxxxxxxxxxx
EOF

chmod 600 /opt/vnpy/.env
```

## Step 6: 端到端测试

```bash
cd /opt/vnpy
source .venv/bin/activate
source .env

# 6.1 测试飞书文档读取
python -c "
from hs300_topk.live.portfolio import load_portfolio_from_feishu
p = load_portfolio_from_feishu()
print(f'持仓: {len(p.positions)} 只, 可用资金: {p.cash:,.2f}')
for pos in p.positions:
    print(f'  {pos.symbol} {pos.name} {pos.shares}股 @{pos.cost}')
"

# 6.2 测试完整流程（dry-run 模式，不推送飞书）
python -m hs300_topk.run_live --force-run --dry-run

# 6.3 测试飞书推送（实际发送卡片）
python -m hs300_topk.run_live --force-run

# 6.4 确认飞书收到消息后，配置 cron
```

## Step 7: 配置 Cron 定时任务

```bash
crontab -e
```

添加以下内容：

```cron
# ─── HS300 Top-K 周度调仓 ───────────────────────────────
# 每个工作日 8:30 执行（非周一/非交易日 → 自动跳过）
# 日志按日期自动拆分到 live_YYYY-MM-DD.log
30 8 * * 1-5 cd /opt/vnpy && source .env && .venv/bin/python -m hs300_topk.run_live 2>&1

# 每月 1 号 7:00 强制重训模型
0 7 1 * * cd /opt/vnpy && source .env && .venv/bin/python -m hs300_topk.run_live --retrain 2>&1
```

**关于 cron 调度说明**：
- 设为每个工作日 8:30 而非只周一，是为了容错：如果周一执行失败，周二会自动重试
- 非周一时脚本内部判断 `today.weekday() != 0` → 自动跳过，不会重复调仓
- 日志按日期自动写入 `hs300_topk/live/logs/live_YYYY-MM-DD.log`，无需手动轮换
- 月度重训 cron 加在 1 号 7:00，比日常任务早 1.5 小时，确保模型训练完成后再执行调仓
- 重训日期如果不是交易日或周一，信号会被缓存等到下个周一使用

## Step 8: 日志管理

日志按日期自动拆分，每天生成独立文件 `live_YYYY-MM-DD.log`：

```bash
# 日志目录（程序自动创建）
ls hs300_topk/live/logs/
# live_2026-05-05.log
# live_2026-05-12.log

# 定期清理（可选，保留最近 90 天）
find /opt/vnpy/hs300_topk/live/logs -name "live_*.log" -mtime +90 -delete
```

## CLI 参数速查

```bash
python -m hs300_topk.run_live [OPTIONS]

  --date YYYY-MM-DD    指定执行日期（默认今天）
  --retrain            强制重新训练模型
  --dry-run            只计算信号，不推送飞书
  --skip-download      跳过数据下载
  --force-run          忽略交易日/周一判断，强制执行
```

## 输出文件结构

```
hs300_topk/live/
  signals/
    2026-05-05.json        # 每次执行的交易建议（含持仓、价格、手续费）
  state/
    portfolio_cache.json   # 飞书持仓的本地快照（降级备用）
  logs/
    live_2026-05-05.log    # 按日期拆分的执行日志
```

## 故障处理

| 情况 | 处理方式 |
|------|----------|
| 飞书 API 不可用 | 自动降级到本地 portfolio_cache.json |
| 信号缓存不存在 | 自动触发实时训练（耗时约 7 分钟） |
| 数据下载失败 | 使用已有数据继续执行 |
| 非交易日执行 | 自动跳过，不做任何操作 |
| 非周一执行 | 自动跳过（周频策略） |
| 目标日期无行情数据 | 自动回退到 14 天内最近可用数据 |

## 操作闭环

```
  周一 8:30    程序自动运行
       ↓
  收到飞书通知  查看调仓建议（含价格区间、手续费、理由）
       ↓
  9:30 开盘后  在建议价格区间内下限价单
       ↓
  成交后      更新飞书持仓表格（股数、成本均价、可用资金）
       ↓
  下周一 8:30  程序读取最新持仓，生成新的调仓建议
```
