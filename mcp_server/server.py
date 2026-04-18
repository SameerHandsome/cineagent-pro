"""
mcp_server/server.py
────────────────────
Your LOCAL FastMCP server.  Run it separately:
    python mcp_server/server.py

Exposes proprietary tools the agents need:
  • Script tools  — parse text, extract structure
  • Casting tools — user-specific casting DB lookups
  • Budget tools  — baseline budget maths
  • Market tools  — seeded comps dataset (~500 films)
  • Tavily search — live web search (box office, union rates, actor news)
  • Playwright    — headless browser scraping (IMDb, Box Office Mojo)
  • Filesystem    — read/list uploaded files

Agents call these over HTTP (SSE transport).
"""
import glob as _glob
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from dotenv import load_dotenv

load_dotenv()

from fastmcp import FastMCP  # noqa: E402
from tavily import TavilyClient  # noqa: E402

from mcp_server.tools.budget_tools import register_budget_tools  # noqa: E402
from mcp_server.tools.casting_tools import register_casting_tools  # noqa: E402
from mcp_server.tools.market_tools import register_market_tools  # noqa: E402
from mcp_server.tools.script_tools import register_script_tools  # noqa: E402

port = int(os.getenv("MCP_SERVER_PORT", 8001))
mcp = FastMCP("cineagent")

register_script_tools(mcp)
register_casting_tools(mcp)
register_budget_tools(mcp)
register_market_tools(mcp)


# ── Filesystem tools ──────────────────────────────────────────────────────────

@mcp.tool()
def read_file(path: str) -> str:
    """Read a file from the uploads directory."""
    safe_path = os.path.join("./uploads", os.path.basename(path))
    if not os.path.exists(safe_path):
        return f"File not found: {path}"
    with open(safe_path, "r", encoding="utf-8") as f:
        return f.read()


@mcp.tool()
def list_uploads() -> list[str]:
    """List all files in the uploads directory."""
    return _glob.glob("./uploads/*")


# ── Tavily web search ─────────────────────────────────────────────────────────

@mcp.tool()
def tavily_search(query: str, max_results: int = 5) -> dict:
    """
    Search the web using Tavily.
    Use for: live box office data, current SAG/IATSE union rates,
    actor news, streaming acquisition deals, film industry trends.
    """
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        return {"error": "TAVILY_API_KEY not set in environment"}
    client = TavilyClient(api_key=api_key)
    results = client.search(query, max_results=max_results)
    return results


# ── Playwright browser scraping ───────────────────────────────────────────────

@mcp.tool()
async def browser_navigate_and_snapshot(url: str) -> str:
    """
    Navigate to a URL and return the page text content.
    Use for: scraping IMDb film pages, Box Office Mojo grosses,
    Deadline/Variety casting announcements.
    Returns up to 8,000 characters of body text.
    """
    try:
        from playwright.async_api import async_playwright
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            page = await browser.new_page()
            await page.goto(url, wait_until="domcontentloaded", timeout=20000)
            content = await page.inner_text("body")
            await browser.close()
            return content[:8000]
    except ImportError:
        return "Error: playwright not installed. Run: pip install playwright && playwright install chromium"
    except Exception as e:
        return f"Error navigating to {url}: {e}"


if __name__ == "__main__":
    mcp.run(transport="sse", host="0.0.0.0", port=port)
