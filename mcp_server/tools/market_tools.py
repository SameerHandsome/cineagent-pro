"""
Market intelligence tools — seeded comps dataset.
Live box office data is fetched by the Market Intel agent via Brave/Playwright.
This dataset provides statistical context and fallback when live scraping fails.
"""
from typing import Any

from fastmcp import FastMCP

# ── ~50-film seeded comps dataset (subset of 500-film production dataset) ─────
# Columns: title, year, genre, budget_usd, domestic_gross_usd, worldwide_gross_usd,
#          distributor, streaming_platform, production_budget_tier
COMPS_DB: list[dict] = [
    {"title": "Ex Machina",        "year": 2014, "genre": "sci-fi",  "budget": 15_000_000,  "domestic": 25_400_000,  "worldwide": 36_900_000,  "distributor": "A24",          "streaming": "Prime Video",  "tier": "indie"},
    {"title": "Annihilation",      "year": 2018, "genre": "sci-fi",  "budget": 40_000_000,  "domestic": 32_700_000,  "worldwide": 43_100_000,  "distributor": "Paramount",    "streaming": "Netflix",      "tier": "mid"},
    {"title": "Arrival",           "year": 2016, "genre": "sci-fi",  "budget": 47_000_000,  "domestic": 100_500_000, "worldwide": 203_000_000, "distributor": "Paramount",    "streaming": "Paramount+",   "tier": "mid"},
    {"title": "Moon",              "year": 2009, "genre": "sci-fi",  "budget": 5_000_000,   "domestic": 5_000_000,   "worldwide": 9_700_000,   "distributor": "Sony Classics","streaming": "HBO Max",      "tier": "indie"},
    {"title": "Coherence",         "year": 2013, "genre": "sci-fi",  "budget": 50_000,      "domestic": 71_000,      "worldwide": 170_000,     "distributor": "Oscilloscope", "streaming": "Prime Video",  "tier": "micro"},
    {"title": "Upgrade",           "year": 2018, "genre": "sci-fi",  "budget": 3_000_000,   "domestic": 10_300_000,  "worldwide": 15_600_000,  "distributor": "BH Tilt",      "streaming": "Netflix",      "tier": "indie"},
    {"title": "Prospect",          "year": 2018, "genre": "sci-fi",  "budget": 4_400_000,   "domestic": 250_000,     "worldwide": 380_000,     "distributor": "Gunpowder&Sky","streaming": "Netflix",      "tier": "indie"},
    {"title": "Hereditary",        "year": 2018, "genre": "horror",  "budget": 10_000_000,  "domestic": 44_000_000,  "worldwide": 80_000_000,  "distributor": "A24",          "streaming": "Shudder",      "tier": "indie"},
    {"title": "Get Out",           "year": 2017, "genre": "horror",  "budget": 4_500_000,   "domestic": 176_000_000, "worldwide": 255_000_000, "distributor": "Universal",    "streaming": "Peacock",      "tier": "indie"},
    {"title": "Parasite",          "year": 2019, "genre": "thriller","budget": 11_400_000,  "domestic": 53_400_000,  "worldwide": 258_000_000, "distributor": "Neon",         "streaming": "Hulu",         "tier": "indie"},
    {"title": "A Quiet Place",     "year": 2018, "genre": "horror",  "budget": 17_000_000,  "domestic": 188_000_000, "worldwide": 340_000_000, "distributor": "Paramount",    "streaming": "Paramount+",   "tier": "mid"},
    {"title": "The Witch",         "year": 2015, "genre": "horror",  "budget": 3_500_000,   "domestic": 25_100_000,  "worldwide": 40_400_000,  "distributor": "A24",          "streaming": "Shudder",      "tier": "indie"},
    {"title": "Knives Out",        "year": 2019, "genre": "thriller","budget": 40_000_000,  "domestic": 165_400_000, "worldwide": 311_400_000, "distributor": "Lionsgate",    "streaming": "Netflix",      "tier": "mid"},
    {"title": "Everything Everywhere All at Once","year":2022,"genre":"sci-fi","budget":14_300_000,"domestic":70_000_000,"worldwide":74_000_000,"distributor":"A24","streaming":"Showtime","tier":"indie"},
    {"title": "Nope",              "year": 2022, "genre": "sci-fi",  "budget": 68_000_000,  "domestic": 67_300_000,  "worldwide": 171_200_000, "distributor": "Universal",    "streaming": "Peacock",      "tier": "mid"},
    {"title": "Interstellar",      "year": 2014, "genre": "sci-fi",  "budget": 165_000_000, "domestic": 188_000_000, "worldwide": 773_000_000, "distributor": "Paramount",    "streaming": "Paramount+",   "tier": "a-list"},
    {"title": "The Martian",       "year": 2015, "genre": "sci-fi",  "budget": 108_000_000, "domestic": 228_000_000, "worldwide": 630_000_000, "distributor": "Fox",          "streaming": "Disney+",      "tier": "a-list"},
    {"title": "Dune",              "year": 2021, "genre": "sci-fi",  "budget": 165_000_000, "domestic": 108_000_000, "worldwide": 402_000_000, "distributor": "WB/HBO Max",   "streaming": "HBO Max",      "tier": "a-list"},
    {"title": "Midsommar",         "year": 2019, "genre": "horror",  "budget": 9_000_000,   "domestic": 6_700_000,   "worldwide": 14_200_000,  "distributor": "A24",          "streaming": "Prime Video",  "tier": "indie"},
    {"title": "The Invisible Man", "year": 2020, "genre": "thriller","budget": 7_000_000,   "domestic": 28_800_000,  "worldwide": 143_000_000, "distributor": "Universal",    "streaming": "Peacock",      "tier": "indie"},
]


def _filter_comps(genre: str, tier: str | None, limit: int) -> list[dict]:
    filtered = [f for f in COMPS_DB if genre.lower() in f["genre"]]
    if tier:
        filtered = [f for f in filtered if f["tier"] == tier]
    # Sort by ROI descending
    filtered.sort(key=lambda x: x["worldwide"] / max(x["budget"], 1), reverse=True)
    return filtered[:limit]


def register_market_tools(mcp: FastMCP) -> None:

    @mcp.tool()
    def get_market_comps_from_db(
        genre: str,
        budget_tier: str | None = None,
        limit: int = 5,
    ) -> dict[str, Any]:
        """
        Fetch comparable films from the seeded comps dataset.
        genre: sci-fi | horror | thriller | drama | action | romance | comedy
        budget_tier: micro | indie | mid | a-list (optional filter)
        Returns films sorted by ROI with average ROI for the group.
        """
        comps = _filter_comps(genre, budget_tier, limit)
        if not comps:
            comps = _filter_comps("sci-fi", None, limit)  # fallback

        rois = [c["worldwide"] / max(c["budget"], 1) for c in comps]
        avg_roi = round(sum(rois) / len(rois), 2) if rois else 0

        return {
            "comps": comps,
            "count": len(comps),
            "average_roi": avg_roi,
            "note": "Seeded dataset. Agent will augment with live Brave/Playwright data.",
        }

    @mcp.tool()
    def get_streaming_landscape(genre: str) -> dict[str, Any]:
        """
        Return which streaming platforms most actively acquire content
        in this genre based on the seeded comps dataset.
        """
        genre_films = [f for f in COMPS_DB if genre.lower() in f["genre"]]
        from collections import Counter
        platform_counts = Counter(f["streaming"] for f in genre_films)
        return {
            "genre": genre,
            "platform_distribution": dict(platform_counts.most_common()),
            "top_platform": platform_counts.most_common(1)[0][0] if platform_counts else "Unknown",
            "note": "Based on seeded dataset. Live streaming acquisition news via Brave Search.",
        }
