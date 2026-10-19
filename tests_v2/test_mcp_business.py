import pytest
pytest.importorskip("mcp", reason="MCP SDK v2 not installed in this environment")
from enterprise.mcp_business import budget_total


def test_budget():
    assert budget_total('400',3)['total']=='1200.00'
    with pytest.raises(ValueError):
        budget_total('400',0)
    with pytest.raises(ValueError):
        budget_total('-50',3)


@pytest.mark.asyncio
async def test_mcp_protocol_roundtrip():
    from mcp import Client
    from enterprise.mcp_business import server
    async with Client(server,raise_exceptions=True) as client:
        names={t.name for t in (await client.list_tools()).tools}
        assert 'calculate_lodging_budget' in names
        result=await client.call_tool('calculate_lodging_budget',{'rate_per_night':'400','nights':3})
        assert result.is_error is False
        assert '1200.00' in str(result.structured_content or result.content)
