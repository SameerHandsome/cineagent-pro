"""
Script analysis tools — parse loglines/treatments into structured data.
These run locally (fast, no network) so agents always have a baseline
even when remote scrapers are down.
"""
import re
from typing import Any

from fastmcp import FastMCP

GENRE_KEYWORDS: dict[str, list[str]] = {
    "sci-fi": ["space", "ai", "robot", "future", "alien", "tech", "cyber", "quantum"],
    "horror": ["haunted", "monster", "fear", "ghost", "death", "dark", "terror"],
    "thriller": ["chase", "spy", "assassin", "conspiracy", "betrayal", "suspect"],
    "drama": ["family", "loss", "relationship", "struggle", "identity", "redemption"],
    "comedy": ["funny", "laugh", "quirky", "absurd", "ridiculous", "awkward"],
    "action": ["explosion", "fight", "war", "battle", "chase", "mission", "weapon"],
    "romance": ["love", "heart", "kiss", "wedding", "relationship", "affair"],
}

TONE_KEYWORDS: dict[str, list[str]] = {
    "dark":   ["death", "grief", "despair", "brutal", "bleak", "violent", "fear"],
    "hopeful": ["redemption", "overcome", "triumph", "survive", "love", "hope"],
    "comedic": ["funny", "laugh", "quirky", "absurd", "awkward", "hilarious"],
    "tense":  ["suspense", "danger", "threat", "mystery", "uncover", "chase"],
}


def _detect_genres(text: str) -> list[str]:
    text_lower = text.lower()
    return [g for g, kw in GENRE_KEYWORDS.items() if any(k in text_lower for k in kw)] or ["drama"]


def _detect_tone(text: str) -> list[str]:
    text_lower = text.lower()
    return [t for t, kw in TONE_KEYWORDS.items() if any(k in text_lower for k in kw)] or ["dramatic"]


def _extract_characters(text: str) -> list[dict[str, str]]:
    """Heuristic: capitalised noun phrases that look like character descriptors."""
    # Pattern: "A [adjective] [noun]" or "The [noun]"
    pattern = r"\b(A|An|The)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\b"
    found = re.findall(pattern, text)
    chars = []
    for article, name in found:
        chars.append({"name": name, "article": article, "description": "Extracted from concept text"})
    if not chars:
        # fallback: pick capitalised words after verbs
        caps = re.findall(r'\b[A-Z][a-z]{3,}\b', text)
        chars = [{"name": c, "article": "", "description": "Heuristic extraction"} for c in set(caps)][:5]
    return chars[:6]


def _structural_complexity(text: str) -> str:
    words = len(text.split())
    if words < 60:
        return "simple"       # logline only
    elif words < 250:
        return "moderate"     # short treatment
    else:
        return "complex"      # full treatment or excerpt


def register_script_tools(mcp: FastMCP) -> None:

    @mcp.tool()
    def parse_screenplay(concept_text: str) -> dict[str, Any]:
        """
        Parse a logline, treatment, or script excerpt into structured fields.
        Returns genres, tone, structural complexity, word count.
        """
        return {
            "genres": _detect_genres(concept_text),
            "tone": _detect_tone(concept_text),
            "structural_complexity": _structural_complexity(concept_text),
            "word_count": len(concept_text.split()),
            "sentence_count": len(re.split(r'[.!?]+', concept_text)),
        }

    @mcp.tool()
    def extract_characters(concept_text: str) -> dict[str, Any]:
        """
        Extract character mentions from concept text.
        Returns a list of detected character names with heuristic descriptions.
        """
        characters = _extract_characters(concept_text)
        return {"characters": characters, "count": len(characters)}

    @mcp.tool()
    def analyze_themes(concept_text: str) -> dict[str, Any]:
        """
        Identify thematic elements that affect budget, casting, and marketing.
        """
        text_lower = concept_text.lower()
        themes: list[str] = []

        theme_map = {
            "identity":       ["who am i", "identity", "self", "becoming"],
            "survival":       ["survive", "survival", "escape", "life or death"],
            "redemption":     ["redemption", "second chance", "forgiveness", "atone"],
            "power":          ["power", "control", "authority", "domination"],
            "ai_consciousness": ["ai", "artificial intelligence", "sentient", "conscious machine"],
            "isolation":      ["alone", "isolated", "solitary", "abandoned"],
            "family":         ["family", "father", "mother", "sibling", "child"],
        }

        for theme, keywords in theme_map.items():
            if any(k in text_lower for k in keywords):
                themes.append(theme)

        return {"themes": themes or ["unspecified"], "primary_theme": themes[0] if themes else "unspecified"}

    @mcp.tool()
    def identify_key_scenes(concept_text: str) -> dict[str, Any]:
        """
        Identify scene types that drive budget (VFX, practical stunts, locations).
        """
        text_lower = concept_text.lower()
        scene_flags = {
            "vfx_heavy": any(k in text_lower for k in ["space", "explosion", "alien", "portal", "cgi", "digital"]),
            "practical_stunts": any(k in text_lower for k in ["chase", "fight", "crash", "stunt", "battle"]),
            "exotic_locations": any(k in text_lower for k in ["jungle", "arctic", "desert", "underwater", "foreign"]),
            "crowd_scenes": any(k in text_lower for k in ["crowd", "army", "masses", "thousands"]),
            "period_sets": any(k in text_lower for k in ["historical", "medieval", "period", "ancient", "1800s", "1900s"]),
        }
        budget_flags = [k for k, v in scene_flags.items() if v]
        return {
            "budget_flags": budget_flags,
            "has_expensive_elements": len(budget_flags) > 0,
            "estimated_vfx_percentage": min(len(budget_flags) * 12, 45),
        }
