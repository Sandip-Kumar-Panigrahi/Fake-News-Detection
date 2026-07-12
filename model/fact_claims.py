"""Rule-based checks for short factual claims (geography, politics, dates, etc.)."""

from __future__ import annotations

import re
from datetime import datetime
from typing import Any, Dict, Optional

# Indian states and union territories (lowercase keys)
_INDIAN_STATES_UTS = {
    "andhra pradesh",
    "arunachal pradesh",
    "assam",
    "bihar",
    "chhattisgarh",
    "goa",
    "gujarat",
    "haryana",
    "himachal pradesh",
    "jharkhand",
    "karnataka",
    "kerala",
    "madhya pradesh",
    "maharashtra",
    "manipur",
    "meghalaya",
    "mizoram",
    "nagaland",
    "odisha",
    "orissa",
    "punjab",
    "rajasthan",
    "sikkim",
    "tamil nadu",
    "telangana",
    "tripura",
    "uttar pradesh",
    "uttarakhand",
    "west bengal",
    "andaman and nicobar",
    "chandigarh",
    "dadra and nagar haveli",
    "daman and diu",
    "delhi",
    "new delhi",
    "jammu and kashmir",
    "ladakh",
    "lakshadweep",
    "puducherry",
    "pondicherry",
}

# Well-known cities that are not countries
_CITIES_NOT_COUNTRIES = {
    "mumbai",
    "bombay",
    "chennai",
    "madras",
    "kolkata",
    "calcutta",
    "bangalore",
    "bengaluru",
    "hyderabad",
    "pune",
    "ahmedabad",
    "jaipur",
    "lucknow",
    "patna",
    "bhubaneswar",
    "paris",
    "london",
    "tokyo",
    "beijing",
    "sydney",
    "new york",
    "los angeles",
    "chicago",
    "dubai",
    "singapore city",
}

_COUNTRIES = {
    "india",
    "china",
    "japan",
    "pakistan",
    "bangladesh",
    "nepal",
    "sri lanka",
    "usa",
    "us",
    "u.s.a",
    "united states",
    "america",
    "uk",
    "united kingdom",
    "britain",
    "england",
    "france",
    "germany",
    "italy",
    "spain",
    "russia",
    "brazil",
    "canada",
    "australia",
    "mexico",
    "indonesia",
    "nigeria",
    "south africa",
    "egypt",
    "saudi arabia",
    "uae",
    "united arab emirates",
}

_PLACE_ALIASES = {
    "orissa": "odisha",
    "bombay": "mumbai",
    "madras": "chennai",
    "calcutta": "kolkata",
    "bengaluru": "bangalore",
    "pondicherry": "puducherry",
}

_IS_A_COUNTRY = re.compile(
    r"^[\s\"']*(?P<place>[\w][\w\s\-]{0,48}?)[\s\"']*\s+is\s+a\s+country\.?\s*$",
    re.IGNORECASE,
)
_IS_A_STATE = re.compile(
    r"^[\s\"']*(?P<place>[\w][\w\s\-]{0,48}?)[\s\"']*\s+is\s+a\s+state\.?\s*$",
    re.IGNORECASE,
)
_IS_NOT_A_COUNTRY = re.compile(
    r"^[\s\"']*(?P<place>[\w][\w\s\-]{0,48}?)[\s\"']*\s+is\s+not\s+a\s+country\.?\s*$",
    re.IGNORECASE,
)

# IPL champions by season year (user claim vs actual winner)
_IPL_WINNERS: dict[int, str] = {
    2008: "rajasthan royals",
    2009: "deccan chargers",
    2010: "chennai super kings",
    2011: "chennai super kings",
    2012: "kolkata knight riders",
    2013: "mumbai indians",
    2014: "kolkata knight riders",
    2015: "mumbai indians",
    2016: "sunrisers hyderabad",
    2017: "mumbai indians",
    2018: "chennai super kings",
    2019: "mumbai indians",
    2020: "mumbai indians",
    2021: "chennai super kings",
    2022: "gujarat titans",
    2023: "chennai super kings",
    2024: "kolkata knight riders",
}

_TEAM_ALIASES: dict[str, tuple[str, ...]] = {
    "rcb": ("royal challengers", "bangalore", "bengaluru", "royal challengers bangalore"),
    "mi": ("mumbai indians", "mumbai"),
    "csk": ("chennai super kings", "chennai"),
    "kkr": ("kolkata knight riders", "kolkata"),
    "srh": ("sunrisers hyderabad", "hyderabad"),
    "rr": ("rajasthan royals", "rajasthan"),
    "dc": ("delhi capitals", "delhi", "delhi daredevils"),
    "gt": ("gujarat titans", "gujarat"),
    "lsg": ("lucknow super giants", "lucknow"),
    "pbks": ("punjab kings", "punjab", "kings xi punjab"),
}

_WIN_WORDS = re.compile(
    r"\b(won|win|wins|winning|champion|champions|trophy|title|championship)\b",
    re.IGNORECASE,
)
_ALL_OUT = re.compile(r"\b(all\s*out|all-out|bowled\s*out)\b", re.IGNORECASE)

_WEEKDAYS = (
    "monday",
    "tuesday",
    "wednesday",
    "thursday",
    "friday",
    "saturday",
    "sunday",
)
_TODAY_IS_WEEKDAY = re.compile(
    r"^[\s\"']*today\s+is\s+(?P<day>monday|tuesday|wednesday|thursday|friday|saturday|sunday)\s*\.?\s*$",
    re.IGNORECASE,
)
_PM_OF_INDIA = re.compile(
    r"(?P<name>[\w][\w\s\.\-]{0,40}?)\s+is\s+(?:the\s+)?(?:prime\s+minister|pm)\s+of\s+india",
    re.IGNORECASE,
)
_CM_OF_STATE = re.compile(
    r"(?:^|[\s,])(?P<name>[\w][\w\s\.\-]{0,50}?)\s*,?\s*(?:is\s+)?(?:the\s+)?(?:cm|chief\s+minister)\s+of\s+(?P<state>odisha|orissa|bihar|delhi|maharashtra|karnataka|tamil\s+nadu|west\s+bengal|gujarat|rajasthan|punjab|kerala|assam|telangana|andhra\s+pradesh|madhya\s+pradesh|uttar\s+pradesh|jharkhand|chhattisgarh|haryana|himachal\s+pradesh|uttarakhand|goa|manipur|meghalaya|mizoram|nagaland|sikkim|tripura|arunachal\s+pradesh)\b",
    re.IGNORECASE,
)

# India PM and select state CMs (verified public offices; update when governments change)
_INDIA_PM_NAMES = frozenset({"narendra modi", "modi", "narendra damodardas modi"})
_STATE_CM_NAMES: dict[str, frozenset[str]] = {
    "odisha": frozenset(
        {
            "mohan charan majhi",
            "mohan majhi",
            "mohancharan majhi",
            "majhi",
            "mohan charan",
        }
    ),
    "orissa": frozenset(
        {
            "mohan charan majhi",
            "mohan majhi",
            "mohancharan majhi",
            "majhi",
            "mohan charan",
        }
    ),
}


def _normalize_place(name: str) -> str:
    place = str(name or "").lower().strip().strip("\"'")
    place = re.sub(r"^(the|a|an)\s+", "", place)
    place = re.sub(r"\s+", " ", place).strip()
    return _PLACE_ALIASES.get(place, place)


def _place_kind(place: str) -> str:
    if place in _INDIAN_STATES_UTS:
        return "indian_state_or_ut"
    if place in _CITIES_NOT_COUNTRIES:
        return "city"
    if place in _COUNTRIES:
        return "country"
    return "unknown"


def _winner_matches_team(actual_winner: str, team_key: str) -> bool:
    w = actual_winner.lower()
    aliases = _TEAM_ALIASES.get(team_key, (team_key,))
    return any(alias in w for alias in aliases)


def _detect_ipl_teams(text: str) -> list[str]:
    found: list[str] = []
    if re.search(r"\b(rcb|royal challengers)\b", text, re.I):
        found.append("rcb")
    if re.search(r"\b(mi|mumbai indians)\b", text, re.I):
        found.append("mi")
    if re.search(r"\b(csk|chennai super kings)\b", text, re.I):
        found.append("csk")
    if re.search(r"\b(kkr|kolkata knight riders)\b", text, re.I):
        found.append("kkr")
    if re.search(r"\b(srh|sunrisers hyderabad)\b", text, re.I):
        found.append("srh")
    if re.search(r"\b(rr|rajasthan royals)\b", text, re.I):
        found.append("rr")
    if re.search(r"\b(dc|delhi capitals|delhi daredevils)\b", text, re.I):
        found.append("dc")
    if re.search(r"\b(gt|gujarat titans)\b", text, re.I):
        found.append("gt")
    return found


def _check_ipl_known_scores(cleaned: str) -> Optional[Dict[str, Any]]:
    """Verified IPL innings scores — do not mark as fake just because the total is low."""
    t = cleaned.lower()
    if "ipl" not in t and "match" not in t:
        return None
    if not re.search(r"\b(rcb|royal challengers)\b", t):
        return None

    # RCB 49 all out vs KKR — IPL 2017 (7 May 2017, Eden Gardens)
    if re.search(r"\b49\b", t) and (
        _ALL_OUT.search(t) or re.search(r"\b49\s*runs?\b", t)
    ):
        return {
            "prediction": "Real",
            "confidence": 98.0,
            "explanation": (
                "Correct: Royal Challengers Bangalore were bowled out for 49 runs in an IPL match "
                "(vs Kolkata Knight Riders, IPL 2017, Eden Gardens). It was one of the lowest IPL team totals."
            ),
            "needs_verification": False,
        }
    return None


def _check_ipl_claim(cleaned: str) -> Optional[Dict[str, Any]]:
    t = cleaned.lower()
    if "ipl" not in t or not _WIN_WORDS.search(t):
        return None

    year_m = re.search(r"\b(20[0-2]\d)\b", t)
    teams = _detect_ipl_teams(t)
    if not teams:
        return None

    if not year_m:
        if "rcb" in teams or "royal challengers" in t:
            return {
                "prediction": "Fake",
                "confidence": 100.0,
                "explanation": (
                    "Incorrect: Royal Challengers Bangalore (RCB) has never won an IPL trophy. "
                    "They have reached finals but not won the title."
                ),
                "needs_verification": False,
            }
        return None

    year = int(year_m.group(1))
    actual = _IPL_WINNERS.get(year)
    if not actual:
        return None

    actual_title = " ".join(w.title() for w in actual.split())
    for team in teams:
        if not _winner_matches_team(actual, team):
            team_label = team.upper() if len(team) <= 3 else team.replace("_", " ").title()
            return {
                "prediction": "Fake",
                "confidence": 100.0,
                "explanation": (
                    f"Incorrect: {team_label} did not win the IPL in {year}. "
                    f"The {year} IPL champion was {actual_title}."
                ),
                "needs_verification": False,
            }

    return {
        "prediction": "Real",
        "confidence": 98.0,
        "explanation": f"Correct: the {year} IPL champion was {actual_title}.",
        "needs_verification": False,
    }


def _normalize_person_name(name: str) -> str:
    n = re.sub(r"\s+", " ", str(name or "").lower().strip().strip("\"'"))
    n = re.sub(r"[^a-z\s\.]", "", n)
    return re.sub(r"\s+", " ", n).strip()


def _name_matches_any(person: str, valid: frozenset[str]) -> bool:
    p = _normalize_person_name(person)
    if not p:
        return False
    if p in valid:
        return True
    return any(v in p or p in v for v in valid)


def _check_today_weekday(cleaned: str) -> Optional[Dict[str, Any]]:
    m = _TODAY_IS_WEEKDAY.match(cleaned)
    if not m:
        return None
    claimed = m.group("day").lower()
    actual = datetime.now().strftime("%A").lower()
    display_actual = actual.title()
    if claimed == actual:
        return {
            "prediction": "Real",
            "confidence": 96.0,
            "explanation": (
                f"Correct: today ({datetime.now().strftime('%B %d, %Y')}) is {display_actual}."
            ),
            "needs_verification": False,
        }
    return {
        "prediction": "Fake",
        "confidence": 96.0,
        "explanation": (
            f"Incorrect: today is {display_actual}, not {claimed.title()} "
            f"({datetime.now().strftime('%B %d, %Y')})."
        ),
        "needs_verification": False,
    }


def _check_pm_india(cleaned: str) -> Optional[Dict[str, Any]]:
    m = _PM_OF_INDIA.search(cleaned)
    if not m:
        return None
    person = _normalize_person_name(m.group("name"))
    if _name_matches_any(person, _INDIA_PM_NAMES):
        return {
            "prediction": "Real",
            "confidence": 98.0,
            "explanation": (
                "Correct: Narendra Modi is the Prime Minister of India "
                "(not MS Dhoni or other celebrities)."
            ),
            "needs_verification": False,
        }
    display = person.title() if person else "That person"
    return {
        "prediction": "Fake",
        "confidence": 98.0,
        "explanation": (
            f"Incorrect: {display} is not the Prime Minister of India. "
            "The PM of India is Narendra Modi."
        ),
        "needs_verification": False,
    }


def _check_state_cm(cleaned: str) -> Optional[Dict[str, Any]]:
    m = _CM_OF_STATE.search(cleaned)
    if not m:
        return None
    state = _normalize_place(m.group("state"))
    person = _normalize_person_name(m.group("name") or "")
    valid = _STATE_CM_NAMES.get(state)
    if not valid:
        return None
    state_display = state.title()
    if _name_matches_any(person, valid):
        return {
            "prediction": "Real",
            "confidence": 96.0,
            "explanation": (
                f"Correct: Mohan Charan Majhi is the Chief Minister of {state_display} "
                "(as of 2024–2026)."
            ),
            "needs_verification": False,
        }
    if not person:
        return {
            "prediction": "Unclear",
            "confidence": 55.0,
            "explanation": (
                f"A Chief Minister of {state_display} was mentioned but the name was not clear. "
                f"Current CM of Odisha is Mohan Charan Majhi."
            ),
            "needs_verification": True,
        }
    return {
        "prediction": "Fake",
        "confidence": 92.0,
        "explanation": (
            f"Incorrect or outdated: {person.title()} is not the listed CM of {state_display}. "
            "Verify the current CM from official state government sources."
        ),
        "needs_verification": True,
    }


def check_factual_claim(text: str) -> Optional[Dict[str, Any]]:
    """
    Return a verdict for short, checkable factual claims; None if not applicable.
    """
    raw = str(text or "").strip()
    if not raw or len(raw) > 280:
        return None

    cleaned = re.sub(r"\s+", " ", raw).strip().strip("\"'")

    weekday = _check_today_weekday(cleaned)
    if weekday is not None:
        return weekday

    pm = _check_pm_india(cleaned)
    if pm is not None:
        return pm

    cm = _check_state_cm(cleaned)
    if cm is not None:
        return cm

    known_score = _check_ipl_known_scores(cleaned)
    if known_score is not None:
        return known_score

    ipl = _check_ipl_claim(cleaned)
    if ipl is not None:
        return ipl

    m = _IS_A_COUNTRY.match(cleaned)
    if m:
        place = _normalize_place(m.group("place"))
        kind = _place_kind(place)

        if kind == "indian_state_or_ut":
            display = place.title()
            return {
                "prediction": "Fake",
                "confidence": 100.0,
                "explanation": (
                    f"Incorrect: {display} is a state in India, not a country. "
                    f"India is the country; {display} is one of its states or union territories."
                ),
                "needs_verification": False,
            }

        if kind == "city":
            display = place.title()
            return {
                "prediction": "Fake",
                "confidence": 100.0,
                "explanation": (
                    f"Incorrect: {display} is a city, not a country. "
                    "Cities belong inside countries (for example, Paris is in France)."
                ),
                "needs_verification": False,
            }

        if kind == "country":
            display = place.title()
            return {
                "prediction": "Real",
                "confidence": 98.0,
                "explanation": (
                    f"Correct: {display} is recognized as a country."
                ),
                "needs_verification": False,
            }

    m = _IS_A_STATE.match(cleaned)
    if m:
        place = _normalize_place(m.group("place"))
        if _place_kind(place) == "country":
            display = place.title()
            return {
                "prediction": "Fake",
                "confidence": 100.0,
                "explanation": (
                    f"Incorrect: {display} is a country, not a state. "
                    "For example, India is a country; Odisha is a state within India."
                ),
                "needs_verification": False,
            }
        if _place_kind(place) == "indian_state_or_ut":
            display = place.title()
            return {
                "prediction": "Real",
                "confidence": 98.0,
                "explanation": (
                    f"Correct: {display} is a state (or union territory) in India."
                ),
                "needs_verification": False,
            }

    m = _IS_NOT_A_COUNTRY.match(cleaned)
    if m:
        place = _normalize_place(m.group("place"))
        kind = _place_kind(place)
        if kind == "country":
            return {
                "prediction": "Fake",
                "confidence": 100.0,
                "explanation": (
                    f"Incorrect: {place.title()} is a country, so saying it is not a country is false."
                ),
                "needs_verification": False,
            }
        if kind in {"indian_state_or_ut", "city"}:
            return {
                "prediction": "Real",
                "confidence": 98.0,
                "explanation": (
                    f"Correct: {place.title()} is not a country "
                    f"({'a state/UT in India' if kind == 'indian_state_or_ut' else 'a city'})."
                ),
                "needs_verification": False,
            }

    return None
