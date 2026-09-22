__all__ = ["create_mcp_server"]


def create_mcp_server():
    from mechcal.mcp.server import create_mcp_server as factory

    return factory()
