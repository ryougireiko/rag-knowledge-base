"""MCP v2 业务工具服务器。可通过 stdio 运行，或使用 Client 在进程内连接。
此处不编码任何真实企业政策：费率必须从取证材料中提取。
"""
from __future__ import annotations
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
from mcp.server import MCPServer

# 创建业务工具服务器实例
server = MCPServer('KnowledgeAssistantBusinessTools')


def budget_total(rate_per_night: str, nights: int) -> dict:
    """计算住宿预算总额，校验天数与金额的合法范围。"""
    if not (1 <= nights <= 60):
        raise ValueError('nights 必须为 1 到 60')
    try:
        rate = Decimal(str(rate_per_night))
    except (InvalidOperation, ValueError) as exc:
        raise ValueError('金额格式无效') from exc
    if not Decimal('0') <= rate <= Decimal('100000'):
        raise ValueError('金额超出允许范围')
    # 使用 Decimal 四舍五入到分，避免浮点误差
    return {'rate_per_night':str(rate), 'nights':nights,
            'total':str((rate * nights).quantize(Decimal('0.01'),rounding=ROUND_HALF_UP)),
            'warning':'输入标准必须来自已核实的政策来源，本工具不判断报销资格。'}


@server.tool()
def calculate_lodging_budget(rate_per_night: str, nights: int) -> dict:
    """根据来自已核实政策证据的费率，计算住宿金额上限。
    本工具不会查询或臆造公司政策。
    """
    return budget_total(rate_per_night,nights)


@server.tool()
def calculate_reimbursement_gap(actual_amount: str, allowed_amount: str) -> dict:
    """比较实际花费与允许额度；over_limit 为正值时表示超额部分。"""
    try:
        actual,allowed = Decimal(str(actual_amount)), Decimal(str(allowed_amount))
    except InvalidOperation as exc:
        raise ValueError('金额格式无效') from exc
    if actual < 0 or allowed < 0 or actual > 10000000 or allowed > 10000000:
        raise ValueError('金额超出允许范围')
    # 超额取实际与额度之差（不低于 0），并标记是否在限额内
    return {'over_limit':str(max(Decimal('0'),actual-allowed).quantize(Decimal('0.01'))),
            'within_limit':actual <= allowed}


if __name__ == '__main__':
    server.run()  # v2 默认使用 stdio；可用 MCP Inspector 进行独立测试