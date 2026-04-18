"""
Budget planning tools — baseline math using seeded union rate data.
The Budget Planner agent first calls these for the baseline,
then cross-checks with live Brave Search results for current rates.
"""
from typing import Any

from fastmcp import FastMCP

# ── Seeded SAG-AFTRA / IATSE baseline rates (2024 scale) ─────────────────────
# Source: Published SAG-AFTRA 2023-2026 Theatrical & TV contract
UNION_RATES: dict[str, dict] = {
    "sag_day_player":       {"rate": 1056,  "unit": "per day",   "notes": "2024 SAG-AFTRA scale"},
    "sag_weekly_player":    {"rate": 3664,  "unit": "per week",  "notes": "2024 SAG-AFTRA scale"},
    "sag_lead":             {"rate": 7500,  "unit": "per week",  "notes": "Negotiated minimum for leads"},
    "dga_director":         {"rate": 18000, "unit": "per week",  "notes": "DGA low-budget threshold"},
    "wga_screenplay":       {"rate": 85000, "unit": "flat",      "notes": "WGA minimum theatrical"},
    "iatse_grip":           {"rate": 54,    "unit": "per hour",  "notes": "IATSE local 80 scale"},
    "iatse_cinematographer": {"rate": 72,    "unit": "per hour",  "notes": "IATSE local 600 scale"},
    "iatse_production_designer": {"rate": 4800, "unit": "per week", "notes": "IATSE scale"},
}

# ── Budget tier multipliers ────────────────────────────────────────────────────
TIER_RANGES: dict[str, dict] = {
    "micro":    {"min_usd": 0,         "max_usd": 500_000,   "label": "Micro-budget"},
    "indie":    {"min_usd": 500_000,   "max_usd": 5_000_000, "label": "Independent"},
    "mid":      {"min_usd": 5_000_000, "max_usd": 30_000_000,"label": "Mid-level studio"},
    "a-list":   {"min_usd": 30_000_000,"max_usd": 200_000_000,"label": "Major studio"},
}

# ── Line-item multipliers by structural complexity ────────────────────────────
COMPLEXITY_MULTIPLIERS: dict[str, float] = {
    "simple": 1.0,
    "moderate": 1.4,
    "complex": 2.1,
}


def _budget_tier_from_total(total: float) -> str:
    for tier, r in TIER_RANGES.items():
        if r["min_usd"] <= total < r["max_usd"]:
            return tier
    return "a-list"


def register_budget_tools(mcp: FastMCP) -> None:

    @mcp.tool()
    def get_union_rate_from_db(role: str) -> dict[str, Any]:
        """
        Return the baseline union rate for a given role key.
        role: one of sag_day_player | sag_weekly_player | sag_lead |
              dga_director | wga_screenplay | iatse_grip |
              iatse_cinematographer | iatse_production_designer
        """
        rate = UNION_RATES.get(role)
        if not rate:
            return {"error": f"Unknown role key '{role}'. Valid keys: {list(UNION_RATES.keys())}"}
        return {"role": role, **rate}

    @mcp.tool()
    def calculate_budget_line(
        department: str,
        structural_complexity: str,
        shoot_days: str | int = 30,
        num_principal_cast: str | int = 3,
    ) -> dict[str, Any]:
        """
        Generate a rough line-item budget breakdown for a given department.
        department: above_the_line | production | post | marketing
        structural_complexity: simple | moderate | complex
        Returns estimated USD amounts with a tier label.
        """
        # Groq sometimes passes integers as strings — coerce defensively
        shoot_days = int(shoot_days)
        num_principal_cast = int(num_principal_cast)

        multiplier = COMPLEXITY_MULTIPLIERS.get(structural_complexity, 1.0)

        line_items: dict[str, float] = {}

        if department == "above_the_line":
            line_items = {
                "screenplay":         UNION_RATES["wga_screenplay"]["rate"] * multiplier,
                "director_fee":       UNION_RATES["dga_director"]["rate"] * (shoot_days / 5) * multiplier,
                "lead_cast":          UNION_RATES["sag_lead"]["rate"] * (shoot_days / 5) * num_principal_cast * multiplier,
                "supporting_cast":    UNION_RATES["sag_weekly_player"]["rate"] * (shoot_days / 5) * 4 * multiplier,
                "producer_fee":       25_000 * multiplier,
            }
        elif department == "production":
            line_items = {
                "cinematographer":    UNION_RATES["iatse_cinematographer"]["rate"] * 10 * shoot_days * multiplier,
                "grip_electric":      UNION_RATES["iatse_grip"]["rate"] * 8 * shoot_days * 8 * multiplier,
                "production_design":  UNION_RATES["iatse_production_designer"]["rate"] * (shoot_days / 5) * multiplier,
                "locations":          3_500 * shoot_days * multiplier,
                "equipment_rental":   8_000 * shoot_days * multiplier,
                "catering":           65 * 50 * shoot_days,
            }
        elif department == "post":
            line_items = {
                "editing":            75_000 * multiplier,
                "vfx":                150_000 * multiplier,
                "sound_mix":          45_000 * multiplier,
                "music":              60_000 * multiplier,
                "color_grade":        25_000 * multiplier,
            }
        elif department == "marketing":
            production_estimate = 500_000 * multiplier
            line_items = {
                "digital_marketing":  production_estimate * 0.12,
                "festival_strategy":  35_000 * multiplier,
                "pr_publicity":       40_000 * multiplier,
                "trailer_production": 50_000 * multiplier,
            }
        else:
            return {"error": f"Unknown department '{department}'. Valid: above_the_line, production, post, marketing"}

        total = sum(line_items.values())
        return {
            "department": department,
            "line_items": {k: round(v, 2) for k, v in line_items.items()},
            "department_total": round(total, 2),
            "budget_tier": _budget_tier_from_total(total * 4),
            "complexity_used": structural_complexity,
            "shoot_days": shoot_days,
        }
