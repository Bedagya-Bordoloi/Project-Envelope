"""
mcp/server.py

Exposes the four tools in bms_mcp/tools.py as a real MCP server, per the
PS's "MCP Server or custom agentic tools" allowance.

Two ways this gets used:

1. Standalone (`python bms_mcp/server.py`) - runs a real MCP server over
   stdio using the official `mcp` Python SDK. Requires `pip install mcp`.
   Lets any real MCP client (Claude Desktop, another agent, an MCP
   inspector) connect to Project Envelope's tools over the actual MCP
   protocol -- this is what makes Feature 8 a genuine "MCP Server", not
   just a phrase in the README.

2. In-process (used by agents/strategist.py via main.py) - the live
   control loop calls bms_mcp.tools.call_tool(...) directly using
   TOOL_SCHEMAS for Groq's function-calling, with zero MCP protocol/
   subprocess overhead. This is the path actually used during the
   closed-loop run, by design -- see this file's bottom note for why.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bms_mcp.tools import ToolContext, TOOL_SCHEMAS, call_tool  # noqa: E402

try:
    from mcp.server.fastmcp import FastMCP
    _HAS_MCP_SDK = True
except ImportError:
    _HAS_MCP_SDK = False


class _DummyBridge:
    """Standalone-mode stand-in so the server can boot without a live sim.
    Matches every attribute the real EnergyPlusBridge now exposes."""
    last_indoor_temp = 22.0
    last_outdoor_temp = 10.0
    last_humidity = 45.0
    last_setpoint = 22.0
    cumulative_kwh = 0.0
    step_counter = 0
    pending_setpoint = None

    def get_forward_weather(self, hours=3):
        return [10.0 + i * 0.2 for i in range(hours)]


def build_context(bridge=None, policy=None) -> ToolContext:
    return ToolContext(
        bridge=bridge or _DummyBridge(),
        policy=policy or {},
        carbon_profile="flat_medium",
    )


def main():
    if not _HAS_MCP_SDK:
        print(
            "The 'mcp' package isn't installed, so this can't run as a real "
            "MCP server over stdio.\n"
            "Install it with: pip install mcp\n"
            "Until then, main.py already talks to these tools directly via "
            "bms_mcp/tools.py's call_tool() + TOOL_SCHEMAS."
        )
        return

    app = FastMCP("project-envelope")
    ctx = build_context()

    @app.tool()
    def get_state() -> dict:
        """Get current sensed indoor/outdoor temperature, humidity, setpoint, and energy use."""
        return call_tool("get_state", ctx)

    @app.tool()
    def set_hvac(setpoint_c: float) -> dict:
        """Stage a new HVAC cooling setpoint for Sentinel Gate review."""
        return call_tool("set_hvac", ctx, setpoint_c=setpoint_c)

    @app.tool()
    def get_weather(hours: int = 3) -> dict:
        """Get the next N hours of outdoor temperature (simulation look-ahead)."""
        return call_tool("get_weather", ctx, hours=hours)

    @app.tool()
    def get_carbon_intensity() -> dict:
        """Get the current simulated grid carbon-intensity level."""
        return call_tool("get_carbon_intensity", ctx)

    app.run()


if __name__ == "__main__":
    main()