"""
Casting tools — searches an in-process casting preference database.
In production this would query Neon PostgreSQL or a JSON flat file
seeded with the user's casting history.  Here we use a seeded dataset
as a stand-in so the tool works without a live DB during local dev.
"""
from typing import Any

from fastmcp import FastMCP

# ── Seeded casting database ───────────────────────────────────────────────────
# Structure: genre → list of {actor, recent_films, budget_tier, note}
CASTING_DB: dict[str, list[dict]] = {
    "sci-fi": [
        {"actor": "Oscar Isaac", "recent_films": ["Moon Knight", "Dune"], "budget_tier": "mid", "note": "Strong sci-fi credibility"},
        {"actor": "Awkwafina", "recent_films": ["The Farewell", "Raya"], "budget_tier": "indie", "note": "Rising indie profile"},
        {"actor": "Mahershala Ali", "recent_films": ["Swan Song", "Blade"], "budget_tier": "mid", "note": "Award-winning gravitas"},
        {"actor": "Tessa Thompson", "recent_films": ["Thor", "Creed"], "budget_tier": "mid", "note": "Franchise-proven"},
    ],
    "horror": [
        {"actor": "Florence Pugh", "recent_films": ["Midsommar", "Don't Worry Darling"], "budget_tier": "mid", "note": "Horror pedigree"},
        {"actor": "Lupita Nyong'o", "recent_films": ["Us", "A Quiet Place Day One"], "budget_tier": "mid", "note": "Proven horror lead"},
        {"actor": "Bill Skarsgård", "recent_films": ["It", "Barbarian"], "budget_tier": "indie", "note": "Excellent villain range"},
    ],
    "thriller": [
        {"actor": "Ana de Armas", "recent_films": ["No Time to Die", "Knives Out"], "budget_tier": "mid", "note": "Rising A-list"},
        {"actor": "Jake Gyllenhaal", "recent_films": ["Road House", "Zodiac"], "budget_tier": "mid", "note": "Thriller specialist"},
    ],
    "drama": [
        {"actor": "Jessie Buckley", "recent_films": ["Women Talking", "I'm Thinking of Ending Things"], "budget_tier": "indie", "note": "Critical darling"},
        {"actor": "Paul Mescal", "recent_films": ["Aftersun", "All of Us Strangers"], "budget_tier": "indie", "note": "Cannes favourite"},
    ],
    "action": [
        {"actor": "Simu Liu", "recent_films": ["Shang-Chi", "Barbie"], "budget_tier": "mid", "note": "Action + comedy range"},
        {"actor": "Zazie Beetz", "recent_films": ["Atlanta", "Deadpool 2"], "budget_tier": "indie", "note": "Strong indie-to-blockbuster bridge"},
    ],
}


def register_casting_tools(mcp: FastMCP) -> None:

    @mcp.tool()
    def search_casting_db(genres: list[str], budget_tier: str, num_results: int = 4) -> dict[str, Any]:
        """
        Search the internal casting database for actors matching genre and budget tier.
        budget_tier: 'indie' | 'mid' | 'a-list'
        Returns ranked suggestions per genre with career notes.
        """
        results: list[dict] = []
        seen: set[str] = set()

        for genre in genres:
            candidates = CASTING_DB.get(genre, [])
            for c in candidates:
                if c["actor"] not in seen:
                    # Budget compatibility: indie fits indie, mid fits indie+mid, a-list fits all
                    tier_map = {"indie": ["indie"], "mid": ["indie", "mid"], "a-list": ["indie", "mid", "a-list"]}
                    if c["budget_tier"] in tier_map.get(budget_tier, ["indie", "mid"]):
                        results.append({**c, "matched_genre": genre})
                        seen.add(c["actor"])
                if len(results) >= num_results:
                    break

        return {
            "suggestions": results,
            "count": len(results),
            "budget_tier_filter": budget_tier,
        }

    @mcp.tool()
    def get_casting_preferences(user_id: str) -> dict[str, Any]:
        """
        Retrieve this user's stored casting preferences and past shortlists.
        In production this hits the PostgreSQL user_preferences table.
        Returns empty preferences for new users — safe default.
        """
        # Stub: production version queries DB by user_id
        return {
            "user_id": user_id,
            "preferred_genres": [],
            "shortlisted_actors": [],
            "avoided_actors": [],
            "note": "No saved preferences yet — populate via past sessions",
        }
