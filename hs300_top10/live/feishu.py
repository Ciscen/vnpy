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
        resp.raise_for_status()
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
            resp.raise_for_status()
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
        resp = requests.post(
            f"{BASE_URL}/im/v1/messages",
            headers=self._headers(),
            params={"receive_id_type": "chat_id"},
            json=body,
            timeout=15,
        )
        resp.raise_for_status()
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
        resp = requests.post(
            f"{BASE_URL}/im/v1/messages",
            headers=self._headers(),
            params={"receive_id_type": "chat_id"},
            json=body,
            timeout=15,
        )
        resp.raise_for_status()
        return resp.json()


# ══════════════════════════════════════════════════
# 卡片模板构建
# ══════════════════════════════════════════════════

def build_rebalance_card(
    signal_date: str,
    actions: list[dict],
    summary: dict,
    model_info: dict,
) -> dict:
    """根据交易建议构建飞书消息卡片。

    Parameters
    ----------
    signal_date : str
        信号日期，如 "2026-05-05"。
    actions : list[dict]
        交易动作列表（来自 signal JSON 的 actions 字段）。
    summary : dict
        汇总统计。
    model_info : dict
        模型信息。
    """
    buys = [a for a in actions if a["action"] == "BUY"]
    sells = [a for a in actions if a["action"] == "SELL"]
    holds = [a for a in actions if a["action"] == "HOLD"]

    lines: list[str] = []

    if buys:
        lines.append(f"**买入 ({len(buys)}只)**")
        for a in buys:
            lines.append(
                f"  {a['name']} {a['symbol'][:6]} | "
                f"{a['shares']}股 | ~{a['ref_price']:.2f}"
            )
        lines.append("")

    if sells:
        lines.append(f"**卖出 ({len(sells)}只)**")
        for a in sells:
            pnl = a.get("current_pnl_pct", 0)
            sign = "+" if pnl >= 0 else ""
            lines.append(
                f"  {a['name']} {a['symbol'][:6]} | "
                f"{a['shares']}股 | {sign}{pnl:.1f}%"
            )
        lines.append("")

    if holds:
        lines.append(f"**继续持有 ({len(holds)}只)**")
        for a in holds:
            pnl = a.get("current_pnl_pct", 0)
            sign = "+" if pnl >= 0 else ""
            lines.append(
                f"  {a['name']} {a['symbol'][:6]} | "
                f"{a['shares']}股 | {sign}{pnl:.1f}%"
            )
        lines.append("")

    turnover = summary.get("estimated_turnover", 0)
    lines.append(f"预估换手: {turnover:,.0f}")
    lines.append(f"模型训练截止: {model_info.get('train_cutoff', 'N/A')}")

    content_md = "\n".join(lines)

    card = {
        "header": {
            "template": "blue",
            "title": {
                "tag": "plain_text",
                "content": f"HS300 V1.3 周度调仓建议 | {signal_date}",
            },
        },
        "elements": [
            {
                "tag": "markdown",
                "content": content_md,
            },
        ],
    }
    return card
