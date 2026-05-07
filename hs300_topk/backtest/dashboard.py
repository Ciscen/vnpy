"""
hs300_topk/backtest/dashboard.py

综合仪表盘 HTML 构建 + 收益归因分析。
"""
from __future__ import annotations

import json as _json

import polars as pl
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots

from vnpy.alpha.strategy.backtesting import BacktestingEngine

from hs300_topk.backtest.charts import (
    build_equity_chart,
    build_excess_return_chart,
    build_pnl_chart,
    build_stock_detail_chart,
    classify_reason,
    detect_direction_values,
    REASON_COLORS,
    REASON_LABELS,
)


def build_dashboard_html(
    engine: BacktestingEngine,
    stats: dict,
    trade_log_df: pl.DataFrame | None,
    *,
    version_label: str = "",
) -> str | None:
    """构建综合仪表盘：多 Tab 汇总 HTML 页面。

    Tab 页面:
      1. 策略总览: 关键指标 + 权益曲线 + 回撤
      2. 持仓汇总: 所有操作过的股票表格（累计收益、胜率、交易次数等）
      3. 交易明细: 完整交易日志（可排序/搜索）
      4. 个股K线: 逐只股票的 K线 + 买卖标记（下拉选择）
      5. 收益归因: 按股票、按月份、按卖出原因的收益分解
    """
    if trade_log_df is None or trade_log_df.is_empty():
        return None

    long_val, short_val = detect_direction_values(trade_log_df)
    daily_df = engine.daily_df

    # ── 数据准备 ──
    buys = trade_log_df.filter(pl.col("direction") == long_val).sort("datetime")
    sells = trade_log_df.filter(pl.col("direction") == short_val).sort("datetime")

    # 个股统计
    stock_stats = compute_stock_stats(buys, sells, long_val, short_val)
    stock_stats_json = _json.dumps(stock_stats, ensure_ascii=False)

    # 交易明细
    trades_json = trade_log_df.sort("datetime").write_json()

    # ── Plotly 图表 ──
    equity_fig = build_equity_chart(daily_df)
    equity_html = pio.to_html(equity_fig, include_plotlyjs=False, full_html=False, div_id="equity-chart")

    pnl_fig = build_pnl_chart(daily_df)
    pnl_html = pio.to_html(pnl_fig, include_plotlyjs=False, full_html=False, div_id="monthly-pnl-chart")

    excess_fig = build_excess_return_chart(engine, benchmark_symbol="000300.SSE")
    excess_html = ""
    if excess_fig is not None:
        excess_html = pio.to_html(excess_fig, include_plotlyjs=False, full_html=False, div_id="excess-chart")

    # 收益归因图表
    attribution_html = build_attribution_charts(sells, daily_df, long_val)

    # 个股 K 线数据（全部有交易的股票）
    all_traded_symbols = sorted(set(trade_log_df["vt_symbol"].unique().to_list()))
    stock_charts_data = prepare_stock_charts_data(engine, trade_log_df, all_traded_symbols, long_val, short_val)

    # ── 组装 HTML ──
    html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>HS300 Top-K 策略回测仪表盘</title>
<script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
<style>
* {{ margin: 0; padding: 0; box-sizing: border-box; }}
body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
       background: #f5f6fa; color: #2c3e50; }}
.header {{ background: linear-gradient(135deg, #2c3e50, #3498db); color: white;
           padding: 24px 32px; }}
.header h1 {{ font-size: 24px; font-weight: 600; }}
.header .subtitle {{ font-size: 14px; opacity: 0.85; margin-top: 4px; }}
.tab-bar {{ display: flex; background: #fff; border-bottom: 2px solid #e0e0e0;
            padding: 0 16px; position: sticky; top: 0; z-index: 100;
            box-shadow: 0 2px 4px rgba(0,0,0,0.05); }}
.tab-btn {{ padding: 14px 24px; cursor: pointer; border: none; background: none;
            font-size: 14px; font-weight: 500; color: #666;
            border-bottom: 3px solid transparent; transition: all 0.2s; }}
.tab-btn:hover {{ color: #3498db; background: #f8f9fa; }}
.tab-btn.active {{ color: #3498db; border-bottom-color: #3498db; }}
.tab-content {{ display: none; padding: 24px; max-width: 1400px; margin: 0 auto; }}
.tab-content.active {{ display: block; }}
.card {{ background: #fff; border-radius: 8px; padding: 20px; margin-bottom: 20px;
         box-shadow: 0 1px 3px rgba(0,0,0,0.08); }}
.card h3 {{ font-size: 16px; color: #34495e; margin-bottom: 12px; padding-bottom: 8px;
            border-bottom: 1px solid #ecf0f1; }}
.metrics-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                  gap: 16px; margin-bottom: 20px; }}
.metric {{ background: #fff; border-radius: 8px; padding: 16px; text-align: center;
           box-shadow: 0 1px 3px rgba(0,0,0,0.08); }}
.metric .value {{ font-size: 28px; font-weight: 700; }}
.metric .label {{ font-size: 12px; color: #95a5a6; margin-top: 4px; }}
.positive {{ color: #27ae60; }}
.negative {{ color: #e74c3c; }}
table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
th {{ background: #f8f9fa; padding: 10px 12px; text-align: left; font-weight: 600;
      border-bottom: 2px solid #e0e0e0; cursor: pointer; user-select: none;
      position: sticky; top: 0; }}
th:hover {{ background: #ecf0f1; }}
td {{ padding: 8px 12px; border-bottom: 1px solid #f0f0f0; }}
tr:hover {{ background: #f8f9fa; }}
.table-wrap {{ max-height: 600px; overflow-y: auto; }}
.search-box {{ padding: 8px 12px; border: 1px solid #ddd; border-radius: 4px;
               width: 300px; margin-bottom: 12px; font-size: 14px; }}
.search-box:focus {{ outline: none; border-color: #3498db; }}
select.stock-select {{ padding: 8px 12px; border: 1px solid #ddd; border-radius: 4px;
                       font-size: 14px; min-width: 200px; }}
.pill {{ display: inline-block; padding: 2px 8px; border-radius: 10px;
         font-size: 11px; font-weight: 600; }}
.pill-profit {{ background: #e8f8f0; color: #27ae60; }}
.pill-loss {{ background: #fde8e8; color: #e74c3c; }}
</style>
</head>
<body>
<div class="header">
  <h1>HS300 Top-K 策略回测仪表盘</h1>
  <div class="subtitle">回测区间: {stats.get('start_date','')} ~ {stats.get('end_date','')} | 初始资金: {stats.get('capital',0):,.0f} | 策略版本: {version_label or '未知'}</div>
</div>

<div class="tab-bar">
  <button class="tab-btn active" onclick="switchTab('overview')">策略总览</button>
  <button class="tab-btn" onclick="switchTab('positions')">持仓汇总</button>
  <button class="tab-btn" onclick="switchTab('trades')">交易明细</button>
  <button class="tab-btn" onclick="switchTab('kline')">个股K线</button>
  <button class="tab-btn" onclick="switchTab('attribution')">收益归因</button>
</div>

<!-- ══════════ Tab 1: 策略总览 ══════════ -->
<div id="tab-overview" class="tab-content active">
  <div class="metrics-grid">
    <div class="metric">
      <div class="value {css_sign(stats.get('total_return',0))}">{stats.get('total_return',0):.2f}%</div>
      <div class="label">总收益率</div>
    </div>
    <div class="metric">
      <div class="value {css_sign(stats.get('annual_return',0))}">{stats.get('annual_return',0):.2f}%</div>
      <div class="label">年化收益率</div>
    </div>
    <div class="metric">
      <div class="value negative">{stats.get('max_ddpercent',0):.2f}%</div>
      <div class="label">最大回撤</div>
    </div>
    <div class="metric">
      <div class="value">{stats.get('sharpe_ratio',0):.2f}</div>
      <div class="label">Sharpe Ratio</div>
    </div>
    <div class="metric">
      <div class="value">{stats.get('return_drawdown_ratio',0):.2f}</div>
      <div class="label">收益回撤比</div>
    </div>
    <div class="metric">
      <div class="value">{stats.get('total_trade_count',0)}</div>
      <div class="label">总成交笔数</div>
    </div>
    <div class="metric">
      <div class="value">{stats.get('profit_days',0)} / {stats.get('total_days',0)}</div>
      <div class="label">盈利天数 / 总天数</div>
    </div>
    <div class="metric">
      <div class="value">{stats.get('total_commission',0):,.0f}</div>
      <div class="label">总手续费</div>
    </div>
  </div>
  <div class="card">{equity_html}</div>
  {"<div class='card'>" + excess_html + "</div>" if excess_html else ""}
  <div class="card">{pnl_html}</div>
</div>

<!-- ══════════ Tab 2: 持仓汇总 ══════════ -->
<div id="tab-positions" class="tab-content">
  <div class="card">
    <h3>所有操作过的股票汇总（共 <span id="stock-count"></span> 只）</h3>
    <input type="text" class="search-box" id="stock-search" placeholder="搜索股票代码..." oninput="filterStockTable()">
    <div class="table-wrap">
      <table id="stock-table">
        <thead>
          <tr>
            <th onclick="sortTable('stock-table',0)">股票代码</th>
            <th onclick="sortTable('stock-table',1,'num')">买入次数</th>
            <th onclick="sortTable('stock-table',2,'num')">卖出次数</th>
            <th onclick="sortTable('stock-table',3,'num')">累计盈亏(%)</th>
            <th onclick="sortTable('stock-table',4,'num')">平均盈亏(%)</th>
            <th onclick="sortTable('stock-table',5,'num')">胜率(%)</th>
            <th onclick="sortTable('stock-table',6,'num')">最大盈利(%)</th>
            <th onclick="sortTable('stock-table',7,'num')">最大亏损(%)</th>
            <th onclick="sortTable('stock-table',8,'num')">平均持仓天数</th>
            <th onclick="sortTable('stock-table',9)">卖出原因分布</th>
          </tr>
        </thead>
        <tbody id="stock-tbody"></tbody>
      </table>
    </div>
  </div>
</div>

<!-- ══════════ Tab 3: 交易明细 ══════════ -->
<div id="tab-trades" class="tab-content">
  <div class="card">
    <h3>完整交易日志（共 <span id="trade-count"></span> 笔）</h3>
    <input type="text" class="search-box" id="trade-search" placeholder="搜索股票代码、日期或原因..." oninput="filterTradeTable()">
    <div class="table-wrap" style="max-height:700px">
      <table id="trade-table">
        <thead>
          <tr>
            <th onclick="sortTable('trade-table',0)">日期</th>
            <th onclick="sortTable('trade-table',1)">股票代码</th>
            <th onclick="sortTable('trade-table',2)">方向</th>
            <th onclick="sortTable('trade-table',3,'num')">价格</th>
            <th onclick="sortTable('trade-table',4,'num')">数量</th>
            <th onclick="sortTable('trade-table',5)">原因</th>
            <th onclick="sortTable('trade-table',6,'num')">入场价</th>
            <th onclick="sortTable('trade-table',7,'num')">盈亏(%)</th>
            <th onclick="sortTable('trade-table',8,'num')">持仓天数</th>
          </tr>
        </thead>
        <tbody id="trade-tbody"></tbody>
      </table>
    </div>
  </div>
</div>

<!-- ══════════ Tab 4: 个股K线 ══════════ -->
<div id="tab-kline" class="tab-content">
  <div class="card">
    <h3>个股交易详情</h3>
    <select class="stock-select" id="kline-select" onchange="renderStockChart()">
      <option value="">-- 选择股票 --</option>
      {"".join(f'<option value="{s}">{s}</option>' for s in all_traded_symbols)}
    </select>
    <div id="kline-container" style="margin-top:16px"></div>
  </div>
</div>

<!-- ══════════ Tab 5: 收益归因 ══════════ -->
<div id="tab-attribution" class="tab-content">
  {attribution_html}
</div>

<script>
// ── 数据 ──
const stockStats = {stock_stats_json};
const allTrades = {trades_json};
const stockCharts = {stock_charts_data};

// ── Tab 切换 ──
function switchTab(name) {{
  document.querySelectorAll('.tab-content').forEach(el => el.classList.remove('active'));
  document.querySelectorAll('.tab-btn').forEach(el => el.classList.remove('active'));
  document.getElementById('tab-' + name).classList.add('active');
  event.target.classList.add('active');
  if (name === 'kline') {{ setTimeout(() => {{ window.dispatchEvent(new Event('resize')); }}, 100); }}
}}

// ── 持仓汇总表 ──
function renderStockTable() {{
  const tbody = document.getElementById('stock-tbody');
  const rows = stockStats.map(s => {{
    const pnlClass = s.total_pnl_pct >= 0 ? 'pill-profit' : 'pill-loss';
    return `<tr>
      <td><strong>${{s.vt_symbol}}</strong></td>
      <td>${{s.buy_count}}</td>
      <td>${{s.sell_count}}</td>
      <td><span class="pill ${{pnlClass}}">${{s.total_pnl_pct.toFixed(1)}}%</span></td>
      <td>${{s.avg_pnl_pct.toFixed(1)}}%</td>
      <td>${{s.win_rate.toFixed(1)}}%</td>
      <td class="positive">${{s.max_profit.toFixed(1)}}%</td>
      <td class="negative">${{s.max_loss.toFixed(1)}}%</td>
      <td>${{s.avg_hold_days.toFixed(1)}}</td>
      <td style="font-size:11px">${{s.reason_dist}}</td>
    </tr>`;
  }});
  tbody.innerHTML = rows.join('');
  document.getElementById('stock-count').textContent = stockStats.length;
}}

// ── 交易明细表 ──
function renderTradeTable() {{
  const tbody = document.getElementById('trade-tbody');
  const rows = allTrades.map(t => {{
    const dir = t.direction === '{long_val}' ? '买入' : '卖出';
    const dirColor = t.direction === '{long_val}' ? '#27ae60' : '#e74c3c';
    const pnl = t.pnl_pct != null ? t.pnl_pct.toFixed(1) + '%' : '-';
    const pnlClass = (t.pnl_pct || 0) >= 0 ? 'positive' : 'negative';
    return `<tr>
      <td>${{t.datetime.substring(0,10)}}</td>
      <td><strong>${{t.vt_symbol}}</strong></td>
      <td style="color:${{dirColor}};font-weight:600">${{dir}}</td>
      <td>${{t.price.toFixed(2)}}</td>
      <td>${{Math.round(t.volume)}}</td>
      <td style="font-size:12px">${{t.reason || '-'}}</td>
      <td>${{t.entry_price ? t.entry_price.toFixed(2) : '-'}}</td>
      <td class="${{pnlClass}}">${{pnl}}</td>
      <td>${{t.hold_days != null ? t.hold_days : '-'}}</td>
    </tr>`;
  }});
  tbody.innerHTML = rows.join('');
  document.getElementById('trade-count').textContent = allTrades.length;
}}

// ── 搜索过滤 ──
function filterStockTable() {{
  const q = document.getElementById('stock-search').value.toLowerCase();
  document.querySelectorAll('#stock-tbody tr').forEach(tr => {{
    tr.style.display = tr.textContent.toLowerCase().includes(q) ? '' : 'none';
  }});
}}
function filterTradeTable() {{
  const q = document.getElementById('trade-search').value.toLowerCase();
  document.querySelectorAll('#trade-tbody tr').forEach(tr => {{
    tr.style.display = tr.textContent.toLowerCase().includes(q) ? '' : 'none';
  }});
}}

// ── 表格排序 ──
let sortState = {{}};
function sortTable(tableId, colIdx, type) {{
  const key = tableId + '_' + colIdx;
  sortState[key] = !sortState[key];
  const asc = sortState[key];
  const tbody = document.getElementById(tableId).querySelector('tbody');
  const rows = Array.from(tbody.querySelectorAll('tr'));
  rows.sort((a, b) => {{
    let va = a.cells[colIdx].textContent.replace(/[%,元]/g, '').trim();
    let vb = b.cells[colIdx].textContent.replace(/[%,元]/g, '').trim();
    if (type === 'num') {{
      va = parseFloat(va) || 0;
      vb = parseFloat(vb) || 0;
    }}
    if (va < vb) return asc ? -1 : 1;
    if (va > vb) return asc ? 1 : -1;
    return 0;
  }});
  rows.forEach(r => tbody.appendChild(r));
}}

// ── 个股 K 线 ──
function renderStockChart() {{
  const sym = document.getElementById('kline-select').value;
  const container = document.getElementById('kline-container');
  if (!sym || !stockCharts[sym]) {{ container.innerHTML = '<p>请选择一只股票</p>'; return; }}
  const d = stockCharts[sym];
  const traces = [];

  traces.push({{
    type: 'candlestick', x: d.dates, open: d.open, high: d.high, low: d.low, close: d.close,
    name: 'K线', increasing: {{line:{{color:'#e74c3c'}}}}, decreasing: {{line:{{color:'#27ae60'}}}},
    xaxis: 'x', yaxis: 'y'
  }});

  if (d.buy_dates.length) {{
    traces.push({{
      type: 'scatter', mode: 'markers', x: d.buy_dates, y: d.buy_prices,
      name: '买入', text: d.buy_texts,
      marker: {{symbol: 'triangle-up', size: 11, color: '#27ae60', line:{{width:1,color:'#000'}}}},
      hovertemplate: '%{{text}}<br>价格: %{{y:.2f}}<extra></extra>',
      xaxis: 'x', yaxis: 'y'
    }});
  }}
  if (d.sell_dates.length) {{
    traces.push({{
      type: 'scatter', mode: 'markers', x: d.sell_dates, y: d.sell_prices,
      name: '卖出', text: d.sell_texts,
      marker: {{symbol: 'triangle-down', size: 11, color: '#e74c3c', line:{{width:1,color:'#000'}}}},
      hovertemplate: '%{{text}}<br>价格: %{{y:.2f}}<extra></extra>',
      xaxis: 'x', yaxis: 'y'
    }});
  }}

  traces.push({{
    type: 'bar', x: d.dates, y: d.volume,
    marker: {{color: d.vol_colors, opacity: 0.5}},
    name: '成交量', xaxis: 'x2', yaxis: 'y2'
  }});

  const layout = {{
    height: 600, grid: {{rows: 2, columns: 1, pattern: 'independent', roworder: 'top to bottom'}},
    xaxis: {{rangeslider: {{visible: false}}, domain: [0,1]}},
    xaxis2: {{domain: [0,1]}},
    yaxis: {{title: '价格', domain: [0.35, 1]}},
    yaxis2: {{title: '成交量', domain: [0, 0.28]}},
    title: sym + ' 交易详情',
    plot_bgcolor: 'white', paper_bgcolor: 'white',
    showlegend: true, legend: {{x:0, y:1}},
    shapes: d.shapes || []
  }};
  Plotly.newPlot(container, traces, layout, {{responsive: true}});
}}

// ── 初始化 ──
renderStockTable();
renderTradeTable();
</script>
</body>
</html>"""
    return html


def css_sign(v: float) -> str:
    return "positive" if v >= 0 else "negative"


def compute_stock_stats(
    buys: pl.DataFrame,
    sells: pl.DataFrame,
    long_val: str,
    short_val: str,
) -> list[dict]:
    """计算每只股票的交易汇总统计"""
    all_symbols = sorted(set(
        buys["vt_symbol"].unique().to_list() + sells["vt_symbol"].unique().to_list()
    ))

    results = []
    for sym in all_symbols:
        sym_buys = buys.filter(pl.col("vt_symbol") == sym)
        sym_sells = sells.filter(pl.col("vt_symbol") == sym)

        buy_count = sym_buys.height
        sell_count = sym_sells.height

        if sell_count > 0 and "pnl_pct" in sym_sells.columns:
            pnls = sym_sells["pnl_pct"].to_list()
            pnls = [p for p in pnls if p is not None]
            total_pnl = sum(pnls) if pnls else 0.0
            avg_pnl = total_pnl / len(pnls) if pnls else 0.0
            wins = sum(1 for p in pnls if p > 0)
            win_rate = (wins / len(pnls) * 100) if pnls else 0.0
            max_profit = max(pnls) if pnls else 0.0
            max_loss = min(pnls) if pnls else 0.0
        else:
            total_pnl = avg_pnl = win_rate = max_profit = max_loss = 0.0

        if sell_count > 0 and "hold_days" in sym_sells.columns:
            hds = sym_sells["hold_days"].to_list()
            hds = [h for h in hds if h is not None]
            avg_hold = sum(hds) / len(hds) if hds else 0.0
        else:
            avg_hold = 0.0

        reason_dist = ""
        if sell_count > 0 and "reason" in sym_sells.columns:
            reasons = sym_sells["reason"].to_list()
            from collections import Counter
            rc = Counter(classify_reason(r) for r in reasons if r)
            reason_dist = ", ".join(
                f"{REASON_LABELS.get(k, k)}:{v}" for k, v in rc.most_common(3)
            )

        results.append({
            "vt_symbol": sym,
            "buy_count": buy_count,
            "sell_count": sell_count,
            "total_pnl_pct": round(total_pnl, 2),
            "avg_pnl_pct": round(avg_pnl, 2),
            "win_rate": round(win_rate, 1),
            "max_profit": round(max_profit, 2),
            "max_loss": round(max_loss, 2),
            "avg_hold_days": round(avg_hold, 1),
            "reason_dist": reason_dist,
        })

    results.sort(key=lambda x: x["total_pnl_pct"], reverse=True)
    return results


def build_attribution_charts(
    sells: pl.DataFrame,
    daily_df: pl.DataFrame,
    long_val: str,
) -> str:
    """构建收益归因分析的 HTML 片段"""
    html_parts = []

    if sells.is_empty() or "pnl_pct" not in sells.columns:
        return '<div class="card"><h3>无卖出数据</h3></div>'

    # 1. 按股票的累计盈亏 Top/Bottom 10
    stock_pnl = (
        sells
        .group_by("vt_symbol")
        .agg(
            pl.col("pnl_pct").sum().alias("total_pnl"),
            pl.col("pnl_pct").count().alias("count"),
        )
        .sort("total_pnl", descending=True)
    )

    top10 = stock_pnl.head(10)
    bottom10 = stock_pnl.tail(10).sort("total_pnl")
    combined = pl.concat([top10, bottom10.sort("total_pnl", descending=True)])

    fig1 = go.Figure()
    c_symbols = combined["vt_symbol"].to_list()
    c_pnl = combined["total_pnl"].to_list()
    colors = ["#27ae60" if v >= 0 else "#e74c3c" for v in c_pnl]
    fig1.add_trace(go.Bar(
        x=c_symbols, y=c_pnl,
        marker_color=colors, name="累计盈亏",
        text=[f"{v:.1f}%" for v in c_pnl],
        textposition="outside",
    ))
    fig1.update_layout(
        title="Top/Bottom 10 股票累计盈亏", height=400, width=1200,
        plot_bgcolor="white", yaxis_title="累计盈亏 (%)",
    )
    html_parts.append(f'<div class="card">{pio.to_html(fig1, include_plotlyjs=False, full_html=False)}</div>')

    # 2. 按月份收益
    monthly_pnl = (
        sells
        .with_columns(pl.col("datetime").str.slice(0, 7).alias("month"))
        .group_by("month")
        .agg(
            pl.col("pnl_pct").sum().alias("total_pnl"),
            pl.col("pnl_pct").count().alias("count"),
            (pl.col("pnl_pct") > 0).sum().alias("win_count"),
        )
        .sort("month")
    )
    monthly_pnl = monthly_pnl.with_columns(
        (pl.col("win_count") / pl.col("count") * 100).alias("win_rate")
    )

    fig2 = make_subplots(specs=[[{"secondary_y": True}]])
    m_months = monthly_pnl["month"].to_list()
    m_pnl = monthly_pnl["total_pnl"].to_list()
    m_wr = monthly_pnl["win_rate"].to_list()
    colors2 = ["#27ae60" if v >= 0 else "#e74c3c" for v in m_pnl]
    fig2.add_trace(go.Bar(
        x=m_months, y=m_pnl,
        marker_color=colors2, name="月度累计盈亏(%)",
    ), secondary_y=False)
    fig2.add_trace(go.Scatter(
        x=m_months, y=m_wr,
        mode="lines+markers", name="月度胜率(%)",
        line=dict(color="#3498db", width=2),
    ), secondary_y=True)
    fig2.update_layout(
        title="月度收益与胜率", height=400, width=1200,
        plot_bgcolor="white",
    )
    fig2.update_yaxes(title_text="盈亏 (%)", secondary_y=False)
    fig2.update_yaxes(title_text="胜率 (%)", secondary_y=True)
    html_parts.append(f'<div class="card">{pio.to_html(fig2, include_plotlyjs=False, full_html=False)}</div>')

    # 3. 按卖出原因的盈亏分布
    reason_pnl = (
        sells
        .with_columns(
            pl.col("reason").map_elements(classify_reason, return_dtype=pl.Utf8).alias("reason_cat"),
        )
        .group_by("reason_cat")
        .agg(
            pl.col("pnl_pct").sum().alias("total_pnl"),
            pl.col("pnl_pct").mean().alias("avg_pnl"),
            pl.col("pnl_pct").count().alias("count"),
            (pl.col("pnl_pct") > 0).sum().alias("win_count"),
        )
        .sort("total_pnl", descending=True)
    )

    fig3 = go.Figure()
    for row in reason_pnl.iter_rows(named=True):
        label = REASON_LABELS.get(row["reason_cat"], row["reason_cat"])
        color = REASON_COLORS.get(row["reason_cat"], "#999")
        fig3.add_trace(go.Bar(
            x=[label], y=[row["total_pnl"]], name=label,
            marker_color=color,
            text=[f'{row["total_pnl"]:.1f}% ({row["count"]}笔, 均{row["avg_pnl"]:.1f}%)'],
            textposition="outside",
        ))
    fig3.update_layout(
        title="按卖出原因的盈亏统计", height=400, width=1200,
        plot_bgcolor="white", yaxis_title="累计盈亏 (%)", showlegend=False,
    )
    html_parts.append(f'<div class="card">{pio.to_html(fig3, include_plotlyjs=False, full_html=False)}</div>')

    # 4. 持仓天数 vs 盈亏散点图
    if "hold_days" in sells.columns:
        s_hold = sells["hold_days"].to_list()
        s_pnl = sells["pnl_pct"].to_list()
        s_sym = sells["vt_symbol"].to_list()
        fig4 = go.Figure()
        fig4.add_trace(go.Scatter(
            x=s_hold, y=s_pnl,
            mode="markers", name="单笔交易",
            marker=dict(
                color=s_pnl,
                colorscale="RdYlGn", cmin=-10, cmax=10,
                size=6, opacity=0.6, colorbar=dict(title="盈亏%"),
            ),
            text=s_sym,
            hovertemplate="%{text}<br>持仓: %{x}天<br>盈亏: %{y:.1f}%<extra></extra>",
        ))
        fig4.add_hline(y=0, line_dash="dash", line_color="gray")
        fig4.update_layout(
            title="持仓天数 vs 盈亏", height=400, width=1200,
            plot_bgcolor="white", xaxis_title="持仓天数", yaxis_title="盈亏 (%)",
        )
        html_parts.append(f'<div class="card">{pio.to_html(fig4, include_plotlyjs=False, full_html=False)}</div>')

    return "\n".join(html_parts)


def prepare_stock_charts_data(
    engine: BacktestingEngine,
    trade_log_df: pl.DataFrame,
    symbols: list[str],
    long_val: str,
    short_val: str,
) -> str:
    """为所有交易过的股票准备 K 线 + 交易标记数据（JSON 格式）"""
    from vnpy.trader.object import BarData as _BarData

    data = {}
    for sym in symbols:
        try:
            bars: list[_BarData] = engine.lab.load_bar_data(
                sym, engine.interval, engine.start, engine.end
            )
            if not bars:
                continue

            dates = [str(b.datetime)[:10] for b in bars]
            opens = [b.open_price for b in bars]
            highs = [b.high_price for b in bars]
            lows = [b.low_price for b in bars]
            closes = [b.close_price for b in bars]
            volumes = [b.volume for b in bars]
            vol_colors = ["#e74c3c" if c >= o else "#27ae60" for o, c in zip(opens, closes)]

            sym_trades = trade_log_df.filter(pl.col("vt_symbol") == sym).sort("datetime")
            sym_buys = sym_trades.filter(pl.col("direction") == long_val)
            sym_sells = sym_trades.filter(pl.col("direction") == short_val)

            buy_dates = [r[:10] for r in sym_buys["datetime"].to_list()]
            buy_prices = sym_buys["price"].to_list()
            buy_texts = [f"买入 vol={v:.0f}" for v in sym_buys["volume"].to_list()]

            sell_dates = [r[:10] for r in sym_sells["datetime"].to_list()]
            sell_prices = sym_sells["price"].to_list()
            sell_texts = []
            for row in sym_sells.iter_rows(named=True):
                pnl = row.get("pnl_pct")
                hd = row.get("hold_days")
                sell_texts.append(
                    f"{row.get('reason','')}<br>盈亏:{pnl:.1f}% 持仓:{hd}天"
                    if pnl is not None else row.get('reason', '')
                )

            shapes = []
            for i, buy_row in enumerate(sym_buys.iter_rows(named=True)):
                bd = buy_row["datetime"][:10]
                matched = sym_sells.filter(pl.col("datetime") > buy_row["datetime"]).head(1)
                if matched.is_empty():
                    continue
                sr = matched.row(0, named=True)
                sd = sr["datetime"][:10]
                pnl_val = sr.get("pnl_pct", 0) or 0
                fc = "rgba(0,200,0,0.07)" if pnl_val >= 0 else "rgba(255,0,0,0.07)"
                shapes.append({
                    "type": "rect", "xref": "x", "yref": "paper",
                    "x0": bd, "x1": sd, "y0": 0, "y1": 1,
                    "fillcolor": fc, "line": {"width": 0}, "layer": "below"
                })

            data[sym] = {
                "dates": dates, "open": opens, "high": highs, "low": lows,
                "close": closes, "volume": volumes, "vol_colors": vol_colors,
                "buy_dates": buy_dates, "buy_prices": buy_prices, "buy_texts": buy_texts,
                "sell_dates": sell_dates, "sell_prices": sell_prices, "sell_texts": sell_texts,
                "shapes": shapes,
            }
        except Exception:
            continue

    return _json.dumps(data, ensure_ascii=False)


def export_stock_details(
    engine: BacktestingEngine,
    trade_log_df: pl.DataFrame,
    output_dir: str,
    top_n: int = 10,
) -> None:
    """导出交易最频繁的 top_n 只个股详情图"""
    out = Path(output_dir) / "stock_details"
    out.mkdir(parents=True, exist_ok=True)

    long_val, _ = detect_direction_values(trade_log_df)

    freq = (
        trade_log_df
        .filter(pl.col("direction") == long_val)
        .group_by("vt_symbol")
        .agg(pl.len().alias("trade_count"))
        .sort("trade_count", descending=True)
        .head(top_n)
    )

    symbols = freq["vt_symbol"].to_list()
    print(f"  [报告] 生成 {len(symbols)} 只高频交易股票详情图 ...")

    for sym in symbols:
        try:
            fig = build_stock_detail_chart(engine, trade_log_df, sym)
            if fig is not None:
                path = out / f"{sym.replace('.', '_')}.html"
                fig.write_html(str(path), include_plotlyjs="cdn")
                print(f"    -> {path}")
        except Exception as e:
            print(f"    [!] {sym} 详情图失败: {e}")
