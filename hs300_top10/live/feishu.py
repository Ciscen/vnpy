"""
hs300_top10/live/feishu.py

飞书开放平台 API 客户端 — 文档表格读取 + 消息卡片推送。

环境变量:
  FEISHU_APP_ID        应用 App ID
  FEISHU_APP_SECRET    应用 App Secret
  FEISHU_DOC_ID        持仓文档 ID
  FEISHU_CHAT_ID       推送目标群/个人 chat_id
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass

import requests

BASE_URL = "https://open.feishu.cn/open-apis"


@dataclass
class FeishuClient:
    """飞书 API 客户端，自动管理 tenant_access_token。"""

    app_id: str
    app_secret: str
    _token: str = ""
    _token_expires: float = 0.0

    @classmethod
    def from_env(cls) -> FeishuClient:
        """从环境变量创建客户端。"""
        app_id = os.environ.get("FEISHU_APP_ID", "")
        app_secret = os.environ.get("FEISHU_APP_SECRET", "")
        if not app_id or not app_secret:
            raise RuntimeError(
                "请设置环境变量 FEISHU_APP_ID 和 FEISHU_APP_SECRET"
            )
        return cls(app_id=app_id, app_secret=app_secret)

    @property
    def token(self) -> str:
        if time.time() >= self._token_expires:
            self._refresh_token()
        return self._token

    def _refresh_token(self) -> None:
        resp = requests.post(
            f"{BASE_URL}/auth/v3/tenant_access_token/internal",
            json={"app_id": self.app_id, "app_secret": self.app_secret},
            timeout=10,
        )
        try:
            resp.raise_for_status()
        except requests.exceptions.HTTPError as e:
            raise RuntimeError(f"HTTP Error: {e}, Response: {resp.text}")
        data = resp.json()
        if data.get("code") != 0:
            raise RuntimeError(f"飞书认证失败: {data}")
        self._token = data["tenant_access_token"]
        self._token_expires = time.time() + data.get("expire", 7200) - 60

    def _headers(self) -> dict:
        return {
            "Authorization": f"Bearer {self.token}",
            "Content-Type": "application/json; charset=utf-8",
        }

    # ──────────────────────────────────────────────────
    # 文档读取
    # ──────────────────────────────────────────────────

    def get_document_blocks(self, document_id: str) -> list[dict]:
        """获取文档所有 blocks（自动分页）。"""
        blocks: list[dict] = []
        page_token = ""
        while True:
            params: dict = {"page_size": 500}
            if page_token:
                params["page_token"] = page_token
            resp = requests.get(
                f"{BASE_URL}/docx/v1/documents/{document_id}/blocks",
                headers=self._headers(),
                params=params,
                timeout=15,
            )
            try:
                resp.raise_for_status()
            except requests.exceptions.HTTPError as e:
                raise RuntimeError(f"HTTP Error: {e}, Response: {resp.text}")
            data = resp.json()
            if data.get("code") != 0:
                raise RuntimeError(f"获取文档块失败: {data}")
            items = data.get("data", {}).get("items", [])
            blocks.extend(items)
            if not data.get("data", {}).get("has_more", False):
                break
            page_token = data["data"].get("page_token", "")
        return blocks

    def read_doc_tables(self, document_id: str) -> list[list[list[str]]]:
        """读取文档中所有内嵌表格，返回二维字符串矩阵列表。

        Returns
        -------
        list of tables, 每个 table 是 list[list[str]]（行 x 列）。
        """
        blocks = self.get_document_blocks(document_id)

        block_map: dict[str, dict] = {}
        for b in blocks:
            block_map[b["block_id"]] = b

        tables: list[list[list[str]]] = []
        for b in blocks:
            if b.get("block_type") == 31:  # table block
                table_data = self._parse_table_block(b, block_map)
                if table_data:
                    tables.append(table_data)
        return tables

    def _parse_table_block(
        self, table_block: dict, block_map: dict[str, dict]
    ) -> list[list[str]]:
        """解析 table block 为二维字符串矩阵。"""
        table_prop = table_block.get("table", {})
        cells = table_prop.get("cells", [])
        row_size = table_prop.get("property", {}).get("row_size", 0)
        col_size = table_prop.get("property", {}).get("column_size", 0)

        if not cells or not row_size or not col_size:
            return []

        matrix: list[list[str]] = []
        for row_idx in range(row_size):
            row: list[str] = []
            for col_idx in range(col_size):
                flat_idx = row_idx * col_size + col_idx
                if flat_idx < len(cells):
                    cell_block_id = cells[flat_idx]
                    cell_text = self._extract_block_text(cell_block_id, block_map)
                    row.append(cell_text.strip())
                else:
                    row.append("")
            matrix.append(row)
        return matrix

    def get_wiki_node_info(self, wiki_token: str) -> dict:
        """获取 Wiki 节点信息，包含 obj_token 和 obj_type。"""
        resp = requests.get(
            f"{BASE_URL}/wiki/v2/spaces/get_node?token={wiki_token}",
            headers=self._headers(),
            timeout=15,
        )
        try:
            resp.raise_for_status()
        except requests.exceptions.HTTPError as e:
            raise RuntimeError(f"HTTP Error: {e}, Response: {resp.text}")
        data = resp.json()
        if data.get("code") != 0:
            raise RuntimeError(f"获取 Wiki 节点失败: {data}")
        return data["data"]["node"]

    def read_spreadsheet_values(self, spreadsheet_token: str) -> list[list[str]]:
        """读取电子表格第一页的所有数据。"""
        # 1. 获取 metainfo
        meta_resp = requests.get(
            f"{BASE_URL}/sheets/v2/spreadsheets/{spreadsheet_token}/metainfo",
            headers=self._headers(),
            timeout=15,
        )
        try:
            meta_resp.raise_for_status()
        except requests.exceptions.HTTPError as e:
            raise RuntimeError(f"HTTP Error: {e}, Response: {meta_resp.text}")
        meta_data = meta_resp.json()
        if meta_data.get("code") != 0:
            raise RuntimeError(f"获取电子表格 metainfo 失败: {meta_data}")
            
        sheet_id = meta_data["data"]["sheets"][0]["sheetId"]
        
        # 2. 读取 values
        val_resp = requests.get(
            f"{BASE_URL}/sheets/v2/spreadsheets/{spreadsheet_token}/values/{sheet_id}?valueRenderOption=ToString&dateTimeRenderOption=FormattedString",
            headers=self._headers(),
            timeout=15,
        )
        try:
            val_resp.raise_for_status()
        except requests.exceptions.HTTPError as e:
            raise RuntimeError(f"HTTP Error: {e}, Response: {val_resp.text}")
        val_data = val_resp.json()
        if val_data.get("code") != 0:
            raise RuntimeError(f"获取电子表格数据失败: {val_data}")
            
        values = val_data["data"]["valueRange"].get("values", [])
        
        matrix: list[list[str]] = []
        for row in values:
            str_row: list[str] = []
            for cell in row:
                if cell is None:
                    str_row.append("")
                else:
                    str_row.append(str(cell).strip())
            matrix.append(str_row)
        return matrix

    def _extract_block_text(self, block_id: str, block_map: dict[str, dict]) -> str:
        """递归提取 block 的纯文本内容。"""
        block = block_map.get(block_id)
        if not block:
            return ""

        texts: list[str] = []

        for key in ("text", "heading1", "heading2", "heading3"):
            content = block.get(key, {})
            elements = content.get("elements", [])
            for elem in elements:
                tr = elem.get("text_run", {})
                if tr.get("content"):
                    texts.append(tr["content"])

        children = block.get("children", [])
        for child_id in children:
            texts.append(self._extract_block_text(child_id, block_map))

        return "".join(texts)

    # ──────────────────────────────────────────────────
    # 消息推送
    # ──────────────────────────────────────────────────

    def send_card_message(self, chat_id: str, card: dict) -> dict:
        """向群/个人发送交互式卡片消息。

        Parameters
        ----------
        chat_id : str
            飞书 chat_id（群或个人）。
        card : dict
            飞书卡片 JSON 结构。

        Returns
        -------
        dict
            API 响应。
        """
        body = {
            "receive_id": chat_id,
            "msg_type": "interactive",
            "content": json.dumps(card, ensure_ascii=False),
        }
        receive_id_type = "open_id" if chat_id.startswith("ou_") else "chat_id"
        resp = requests.post(
            f"{BASE_URL}/im/v1/messages",
            headers=self._headers(),
            params={"receive_id_type": receive_id_type},
            json=body,
            timeout=15,
        )
        try:
            resp.raise_for_status()
        except requests.exceptions.HTTPError as e:
            raise RuntimeError(f"HTTP Error: {e}, Response: {resp.text}")
        data = resp.json()
        if data.get("code") != 0:
            raise RuntimeError(f"发送消息失败: {data}")
        return data

    def send_text_message(self, chat_id: str, text: str) -> dict:
        """发送纯文本消息（简单场景/降级方案）。"""
        body = {
            "receive_id": chat_id,
            "msg_type": "text",
            "content": json.dumps({"text": text}, ensure_ascii=False),
        }
        receive_id_type = "open_id" if chat_id.startswith("ou_") else "chat_id"
        resp = requests.post(
            f"{BASE_URL}/im/v1/messages",
            headers=self._headers(),
            params={"receive_id_type": receive_id_type},
            json=body,
            timeout=15,
        )
        try:
            resp.raise_for_status()
        except requests.exceptions.HTTPError as e:
            raise RuntimeError(f"HTTP Error: {e}, Response: {resp.text}")
        return resp.json()


# ══════════════════════════════════════════════════
# 卡片模板构建
# ══════════════════════════════════════════════════

def _pnl_str(pnl: float) -> str:
    sign = "+" if pnl >= 0 else ""
    return f"{sign}{pnl:.1f}%"


def build_rebalance_card(
    signal_date: str,
    actions: list[dict],
    summary: dict,
    model_info: dict,
    portfolio_before: dict | None = None,
    portfolio_after: dict | None = None,
    skipped_cooldowns: list[dict] | None = None,
) -> dict:
    """根据交易建议构建飞书消息卡片（增强版）。

    Parameters
    ----------
    signal_date : str
        信号日期。
    actions : list[dict]
        交易动作列表。
    summary : dict
        汇总统计。
    model_info : dict
        模型信息。
    portfolio_before : dict | None
        操作前持仓概况。
    portfolio_after : dict | None
        操作后预期持仓概况。
    skipped_cooldowns : list[dict] | None
        冷却期中被跳过的高信号股票。
    """
    buys = [a for a in actions if a["action"] == "BUY"]
    sells = [a for a in actions if a["action"] == "SELL"]
    holds = [a for a in actions if a["action"] == "HOLD"]

    elements: list[dict] = []

    # ── 当前持仓概况 ──
    if portfolio_before:
        pos_lines = []
        pos_lines.append(f"可用资金: **{portfolio_before.get('cash', 0):,.0f}**")
        pos_lines.append(f"持仓市值: **{portfolio_before.get('position_value', 0):,.0f}**")
        pos_lines.append(f"账户总值: **{portfolio_before.get('total_market_value', 0):,.0f}**")
        pos_lines.append(f"持仓数量: **{portfolio_before.get('position_count', 0)}** 只")

        updated_at = portfolio_before.get("updated_at", "")
        if updated_at:
            pos_lines.append(f"数据同步: {updated_at[:19]}")

        stale_hours = portfolio_before.get("stale_hours", 0)
        if stale_hours > 24:
            pos_lines.append(f"⚠️ **持仓数据已 {stale_hours:.0f} 小时未更新，请先确认飞书文档是否最新**")

        elements.append({
            "tag": "markdown",
            "content": "\n".join(pos_lines),
        })
        elements.append({"tag": "hr"})

    # ── 当前持仓明细 ──
    current_positions = holds + sells
    if current_positions:
        detail_lines = ["**当前持仓明细**"]
        for a in sorted(current_positions, key=lambda x: x.get("market_value", 0), reverse=True):
            pnl = a.get("current_pnl_pct", 0)
            mkt = a.get("market_value", 0)
            wt = a.get("weight_pct", 0)
            days = a.get("hold_days", 0)
            status = "📈" if pnl >= 0 else "📉"
            day_str = f"{days}天" if days > 0 else ""
            detail_lines.append(
                f"{status} {a['name']} {a['symbol'][:6]} | "
                f"{a['shares']}股 @{a['ref_price']:.2f} | "
                f"{_pnl_str(pnl)} | {mkt:,.0f} ({wt:.1f}%) | {day_str}"
            )
        elements.append({"tag": "markdown", "content": "\n".join(detail_lines)})
        elements.append({"tag": "hr"})

    # ── 调仓操作 ──
    op_lines: list[str] = []

    if sells:
        op_lines.append(f"**🔴 卖出 ({len(sells)}只)**")
        for a in sells:
            pnl = a.get("current_pnl_pct", 0)
            low = a.get("price_low", a["ref_price"])
            high = a.get("price_high", a["ref_price"])
            proceeds = a.get("estimated_proceeds", a.get("estimated_amount", 0))
            fee = a.get("fee", 0)
            net = a.get("net_amount", proceeds)
            reason_text = a.get("reason_text", "")
            op_lines.append(
                f"  {a['name']} {a['symbol'][:6]} | "
                f"{a['shares']}股 | {_pnl_str(pnl)}"
            )
            op_lines.append(
                f"    建议区间 {low:.2f}~{high:.2f} | "
                f"回款 {net:,.0f} (费{fee:,.0f})"
            )
            if reason_text:
                op_lines.append(f"    💬 {reason_text}")
        op_lines.append("")

    if buys:
        op_lines.append(f"**🟢 买入 ({len(buys)}只)**")
        for a in buys:
            low = a.get("price_low", a["ref_price"])
            high = a.get("price_high", a["ref_price"])
            cost = a.get("estimated_cost", a.get("estimated_amount", 0))
            fee = a.get("fee", 0)
            net = a.get("net_amount", cost)
            reason_text = a.get("reason_text", "")
            op_lines.append(
                f"  {a['name']} {a['symbol'][:6]} | "
                f"{a['shares']}股 | 信号 {a.get('signal_prob', 0):.2f} (#{a.get('signal_rank', 0)})"
            )
            op_lines.append(
                f"    建议区间 {low:.2f}~{high:.2f} | "
                f"成本 {net:,.0f} (费{fee:,.0f})"
            )
            if reason_text:
                op_lines.append(f"    💬 {reason_text}")
        op_lines.append("")

    if holds:
        op_lines.append(f"**⚪ 继续持有 ({len(holds)}只)**")
        for a in holds:
            pnl = a.get("current_pnl_pct", 0)
            reason_text = a.get("reason_text", "")
            op_lines.append(
                f"  {a['name']} {a['symbol'][:6]} | "
                f"{a['shares']}股 | {_pnl_str(pnl)} | 信号 {a.get('signal_prob', 0):.2f} (#{a.get('signal_rank', 0)})"
            )
            if reason_text:
                op_lines.append(f"    💬 {reason_text}")
        op_lines.append("")

    if op_lines:
        elements.append({"tag": "markdown", "content": "\n".join(op_lines)})
        elements.append({"tag": "hr"})

    # ── 冷却中的股票提示 ──
    if skipped_cooldowns:
        cd_lines = ["**❄️ 冷却期跳过（信号高但近期止损过）**"]
        for cd in skipped_cooldowns[:5]:
            cd_lines.append(
                f"  {cd['vt_symbol'][:6]} | 信号 {cd['signal']:.2f} | "
                f"剩余冷却 {cd['remaining_days']}天"
            )
        elements.append({"tag": "markdown", "content": "\n".join(cd_lines)})
        elements.append({"tag": "hr"})

    # ── 操作后预期状态 ──
    if portfolio_after:
        after_lines = ["**操作后预期状态**"]
        after_lines.append(
            f"预期可用资金: **{portfolio_after.get('expected_cash', 0):,.0f}**"
        )
        after_lines.append(
            f"预期持仓数量: **{portfolio_after.get('expected_positions', 0)}** 只"
        )
        after_lines.append(
            f"预期总值: **{portfolio_after.get('expected_total', 0):,.0f}**"
        )
        elements.append({"tag": "markdown", "content": "\n".join(after_lines)})
        elements.append({"tag": "hr"})

    # ── 汇总 & 模型信息 ──
    footer_lines = []
    turnover = summary.get("estimated_turnover", 0)
    total_fees = summary.get("total_fees", 0)
    footer_lines.append(
        f"换手金额: {turnover:,.0f} | "
        f"预估手续费合计: **{total_fees:,.0f}** | "
        f"买{summary.get('buys', 0)} 卖{summary.get('sells', 0)} 持{summary.get('holds', 0)}"
    )
    footer_lines.append(f"模型训练截止: {model_info.get('train_cutoff', 'N/A')}")
    footer_lines.append(
        f"信号日期: {model_info.get('signal_date', 'N/A')} | "
        f"候选池: {model_info.get('signal_count', 0)} 只"
    )
    footer_lines.append("")
    footer_lines.append("💡 价格区间基于昨收 ±1.5%，请在区间内挂限价单")
    footer_lines.append("💰 手续费按佣金万2.5（最低5元）+ 印花税千0.5（卖出）+ 过户费万0.1 估算")
    footer_lines.append("⚠️ 操作完成后请及时更新飞书持仓表格（含股数、成本均价、可用资金），否则下周调仓建议会基于错误的持仓计算")
    elements.append({"tag": "markdown", "content": "\n".join(footer_lines)})

    card = {
        "header": {
            "template": "blue",
            "title": {
                "tag": "plain_text",
                "content": f"HS300 V1.3 周度调仓建议 | {signal_date}",
            },
        },
        "elements": elements,
    }
    return card
