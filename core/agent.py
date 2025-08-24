# core/agent.py
from __future__ import annotations

import json
import os
from typing import Dict, List, TypedDict, Optional, Tuple

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langgraph.graph import StateGraph, START, END

from core.tools import property_valuation_tool

load_dotenv()

# ------------------------------
# 1) Model (kept for parity; not used directly in CLI loop)
# ------------------------------
llm = init_chat_model("google_genai:gemini-2.0-flash")

# ------------------------------
# 2) Static option sets + friendly hints
# ------------------------------
VALID_CATEGORIES = [
    "Higher Villa",
    "Multi-Story Building",
    "Apartment / Condominium",
    "MPH & Factory Building",
    "Fuel Station",
    "Coffee Washing Site",
    "Green House",
]

VALID_USE = ["Residential", "Commercial"]

VALID_TOWN_CLASSES = [
    "Finfinne Border A1",
    "Surrounding Finfine B1",
    "Surrounding Finfine B2",
    "Surrounding Finfine B3",
    "Major Cities C1",
    "Major Cities C2",
    "Secondary Major Cities D1",
    "Secondary Major Cities D2",
    "Tertiary Towns E1",
    "Tertiary Towns E2",
]

TOWN_CLASS_EXAMPLES = {
    "Finfinne Border A1": "Houses near the edge of Finfinne city, close to main roads",
    "Surrounding Finfine B1": "Neighborhoods just outside the city, mostly homes",
    "Surrounding Finfine B2": "Areas farther from the city, mix of houses and small shops",
    "Surrounding Finfine B3": "Countryside areas outside the city, mostly farms",
    "Major Cities C1": "Busy city center, many shops and offices",
    "Major Cities C2": "Residential areas inside major cities",
    "Secondary Major Cities D1": "Small towns near bigger cities",
    "Secondary Major Cities D2": "Smaller towns, fewer buildings and shops",
    "Tertiary Towns E1": "Very small towns, mostly houses and farmland",
    "Tertiary Towns E2": "Remote villages, few houses, mostly farmland",
}

# Helpful examples for generic material questions (fallbacks; exact component list comes from JSON)
MATERIAL_EXAMPLES = {
    "foundation": "Reinforced concrete; Stone; Mud block",
    "roof": "Corrugated iron; Tile; Concrete slab",
    "floor": "Ceramic; Wood; Concrete",
    "ceiling": "Gypsum board; Wood; Concrete",
    "metal work": "Aluminum; Steel; Iron",
    "sanitary": "Standard; Premium; Basic",
}

# ------------------------------
# 3) Load JSON data needed by the agent
# ------------------------------
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")

def _load_json(filename: str) -> dict:
    path = os.path.join(DATA_DIR, filename)
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

# Used for automatic plot-grade selection
PLOT_PRICES = _load_json("location_data.json")

# Used to know which material components to ask per building category
# Structure expected: { "<Category>": { "<component>": { "<material substring>": "<grade>", ... }, ... }, ... }
MATERIAL_MAPPINGS = _load_json("material_mappings.json")

# (You already load minimum_completion_stages.json inside calculation engine;
# no direct agent use is required to enforce thresholds.)

def get_material_components_for_category(category: str) -> List[str]:
    """
    Extract component names from material_mappings.json for the given category.
    Falls back to a common, sensible set if category not present.
    """
    cat_map = MATERIAL_MAPPINGS.get(category, {})
    comps = list(cat_map.keys())
    if comps:
        return comps
    # Fallback if mapping missing
    return ["foundation", "roof", "floor", "ceiling", "metal work", "sanitary"]

# ------------------------------
# 4) Category-specific specialized components & examples
#    (names MUST match keys expected by calculation_engine.* helpers)
# ------------------------------
CATEGORY_SPECIAL_SLOTS: Dict[str, List[str]] = {
    # Buildings valued via building rates (no special slots needed here)
    "Higher Villa": [],
    "Multi-Story Building": [],
    "Apartment / Condominium": [],
    "MPH & Factory Building": [],

    # Specialized categories with their own engines:
    "Fuel Station": [
        "site_preparation_area",  # sqm
        "forecourt_area",         # sqm
        "canopy_area",            # sqm
        "num_pump_islands",       # count
        "num_ugt_30m3",           # count
        "num_ugt_50m3",           # count
    ],
    "Coffee Washing Site": [
        "cherry_hopper_area",         # sqm
        "fermentation_tanks_area",    # sqm
        "washing_channels_length",    # m
        "coffee_drier_area",          # sqm
    ],
    "Green House": [
        "greenhouse_area",       # sqm
        "in_farm_road_km",       # km
        "borehole_depth",        # m
        "land_preparation_area", # sqm
    ],
}

SPECIAL_SLOT_EXAMPLES: Dict[str, str] = {
    "site_preparation_area": "e.g., 1500 (sqm)",
    "forecourt_area": "e.g., 800 (sqm)",
    "canopy_area": "e.g., 320 (sqm)",
    "num_pump_islands": "e.g., 4 (count)",
    "num_ugt_30m3": "e.g., 2 (count)",
    "num_ugt_50m3": "e.g., 1 (count)",
    "cherry_hopper_area": "e.g., 45 (sqm)",
    "fermentation_tanks_area": "e.g., 120 (sqm)",
    "washing_channels_length": "e.g., 60 (meters)",
    "coffee_drier_area": "e.g., 300 (sqm)",
    "greenhouse_area": "e.g., 5000 (sqm)",
    "in_farm_road_km": "e.g., 1.2 (km)",
    "borehole_depth": "e.g., 80 (meters)",
    "land_preparation_area": "e.g., 8000 (sqm)",
}

# ------------------------------
# 5) Conversation State
# ------------------------------
class ValuationState(TypedDict):
    messages: List[Dict]
    slots: Dict[str, object]
    asked: List[str]

def initial_state() -> ValuationState:
    return {"messages": [], "slots": {}, "asked": []}

# Base required slots (category-specific materials + special slots are added dynamically)
BASE_REQUIRED_SLOTS_IN_ORDER: List[str] = [
    "building_name",
    "building_category",
    "length_m",
    "width_m",
    "num_floors",
    "has_basement",
    "is_under_construction",
    "incomplete_components",  # only used when is_under_construction=True
    "plot_area_sqm",
    "prop_town",
    "gen_use",
    "mcf",
    "pef",
    "has_elevator",
    "elevator_stops",         # only used when has_elevator=True
]

def _boolify(text: str) -> Optional[bool]:
    t = text.strip().lower()
    if t in {"yes", "y", "true", "t", "1"}: return True
    if t in {"no", "n", "false", "f", "0"}: return False
    return None

def format_choices_with_examples(choices: List[str], examples_map: Dict[str, str]) -> str:
    lines = []
    for i, c in enumerate(choices, start=1):
        ex = examples_map.get(c)
        if ex:
            lines.append(f"{i}. {c} — {ex}")
        else:
            lines.append(f"{i}. {c}")
    return "\n".join(lines)

def current_required_slots(slots: Dict[str, object]) -> List[str]:
    """
    Build the final required slot list dynamically:
    - base slots
    - materials for the chosen category (from material_mappings.json)
    - category specialized slots (e.g., fuel station components)
    """
    req = list(BASE_REQUIRED_SLOTS_IN_ORDER)

    cat = slots.get("building_category")
    if isinstance(cat, str) and cat:
        # materials for this category
        for comp in get_material_components_for_category(cat):
            req.append(f"material__{comp}")  # double underscore to avoid collision with generic names
        # specialized slots for category
        for sp in CATEGORY_SPECIAL_SLOTS.get(cat, []):
            req.append(sp)

    # De-duplicate while preserving order
    seen = set()
    final: List[str] = []
    for r in req:
        if r not in seen:
            final.append(r)
            seen.add(r)
    return final

def missing_slots(slots: Dict[str, object]) -> List[str]:
    needed = [s for s in current_required_slots(slots) if s not in slots]

    # Skip elevator_stops if has_elevator = False
    if "has_elevator" in slots and slots.get("has_elevator") is False:
        needed = [s for s in needed if s != "elevator_stops"]

    # Skip incomplete_components if not under construction
    if "is_under_construction" in slots and slots.get("is_under_construction") is False:
        needed = [s for s in needed if s != "incomplete_components"]

    return needed

# ------------------------------
# 6) Plot grade auto-selection
# ------------------------------
def select_plot_grade(location: str, use_type: str, plot_area: float) -> str:
    """Choose plot grade automatically from location_data.json tables."""
    try:
        town_data = PLOT_PRICES.get(location, {})
        use_data = town_data.get(use_type, {})
        # grade table is typically: { "1st": { "0-200": rate, "201-500": rate, ... }, ... }
        for grade, ranges in use_data.items():
            for range_str in ranges.keys():
                start_str, end_str = range_str.split("-")
                start = float(start_str)
                end = float("inf") if end_str.lower() == "inf" else float(end_str)
                if start <= plot_area <= end:
                    return grade
    except Exception:
        pass
    return "Average"

# ------------------------------
# 7) LangGraph Nodes
# ------------------------------
def extract_info_node(state: ValuationState) -> ValuationState:
    if not state.get("messages"):
        return state
    last = state["messages"][-1]
    if last.get("role") != "user":
        return state

    asked_slots = state.get("asked", [])
    if asked_slots:
        last_asked = asked_slots[-1]
        content = last.get("content", "").strip()

        # Normalize booleans
        if last_asked in {"has_basement", "has_elevator", "is_under_construction"}:
            b = _boolify(content)
            if b is not None:
                state["slots"][last_asked] = b
                return state

        # Numbered choices handler
        expected = state["slots"].get("__expected_choices__")
        if expected:
            slot, choices = expected
            if content.isdigit():
                idx = int(content) - 1
                if 0 <= idx < len(choices):
                    state["slots"][slot] = choices[idx]
                    state["slots"].pop("__expected_choices__", None)
                    return state

        # Everything else: store raw text
        state["slots"][last_asked] = content
        return state

    return state

def ask_next_question_node(state: ValuationState) -> ValuationState:
    slots = state.get("slots", {})
    asked = set(state.get("asked", []))
    remaining = [s for s in missing_slots(slots) if s not in asked]
    if not remaining:
        return state

    s = remaining[0]

    # Choice lists
    options_map = {
        "building_category": VALID_CATEGORIES,
        "gen_use": VALID_USE,
        "prop_town": VALID_TOWN_CLASSES,
    }

    # Pretty labels for some fields
    label_map = {
        "length_m": "building length (meters)",
        "width_m": "building width (meters)",
        "num_floors": "number of floors",
        "has_basement": "Does the building have a basement?",
        "is_under_construction": "Is the building under construction?",
        "incomplete_components": "List incomplete components",
        "plot_area_sqm": "plot area (sqm)",
        "mcf": "Market Condition Factor (MCF)",
        "pef": "Property Enhancement Factor (PEF)",
        "has_elevator": "Is there an elevator?",
        "elevator_stops": "How many elevator stops?",
    }

    # Examples for generic numeric answers
    generic_examples = {
        "length_m": "e.g., 20",
        "width_m": "e.g., 15",
        "num_floors": "e.g., 3",
        "incomplete_components": "e.g., Foundation, Roof (or leave empty)",
        "plot_area_sqm": "e.g., 450",
        "mcf": "Reply 1.0 if unsure",
        "pef": "Reply 1.0 if unsure",
        "elevator_stops": "e.g., 5",
    }

    # If we’re asking a choice slot
    if s in options_map:
        choices = options_map[s]
        if s == "prop_town":
            body = format_choices_with_examples(choices, TOWN_CLASS_EXAMPLES)
        else:
            body = "\n".join(f"{i+1}. {c}" for i, c in enumerate(choices))
        q = f"Please select {s.replace('_', ' ')}:\n{body}\n(Reply with the number)"
        state["slots"]["__expected_choices__"] = (s, choices)

    # Material slot? (material__<component>)
    elif s.startswith("material__"):
        comp = s.split("__", 1)[1]
        ex = MATERIAL_EXAMPLES.get(comp.lower(), "describe the material clearly")
        q = f"Enter the selected material for {comp} (e.g., {ex}):"

    # Category-special slot?
    elif s in SPECIAL_SLOT_EXAMPLES:
        label = s.replace("_", " ")
        q = f"Enter {label} ({SPECIAL_SLOT_EXAMPLES[s]}):"

    # Generic scalar questions
    elif s in label_map:
        label = label_map[s]
        ex = generic_examples.get(s)
        if s in {"has_basement", "is_under_construction", "has_elevator"}:
            q = f"{label} (yes/no)"
        else:
            q = f"Enter the {label}{f' ({ex})' if ex else ''}:"

    # Fallback
    else:
        q = f"Please provide the value for: {s}"

    state["messages"].append({"role": "assistant", "content": q})
    state.setdefault("asked", []).append(s)
    return state

def _collect_selected_materials(slots: Dict[str, object], category: str) -> Dict[str, str]:
    comps = get_material_components_for_category(category)
    out: Dict[str, str] = {}
    for comp in comps:
        out[comp] = str(slots.get(f"material__{comp}", "")).strip()
    return out

def _collect_specialized_components(slots: Dict[str, object], category: str) -> Dict[str, float | int]:
    spec = {}
    for key in CATEGORY_SPECIAL_SLOTS.get(category, []):
        val = slots.get(key)
        if val is None or val == "":
            continue
        # try numeric conversion where reasonable
        try:
            if key.startswith("num_"):
                spec[key] = int(val)
            else:
                spec[key] = float(val)
        except Exception:
            # leave as string if conversion fails (but engine expects numbers)
            spec[key] = val
    return spec

def calculate_node(state: ValuationState) -> ValuationState:
    slots = state.get("slots", {})

    # Scalars / factors
    mcf = float(slots.get("mcf", 1.0) or 1.0)
    pef = float(slots.get("pef", 1.0) or 1.0)
    has_elevator = bool(slots.get("has_elevator", False))
    elevator_stops = int(slots.get("elevator_stops") or 0)

    category = str(slots.get("building_category"))

    # Materials per category
    selected_materials = _collect_selected_materials(slots, category)

    # Building core data
    building = {
        "name": str(slots.get("building_name", "Building 1")),
        "category": category,
        "length": float(slots.get("length_m")),
        "width": float(slots.get("width_m")),
        "num_floors": int(slots.get("num_floors")),
        "has_basement": bool(slots.get("has_basement", False)),
        "is_under_construction": bool(slots.get("is_under_construction", False)),
        "incomplete_components": [
            c.strip() for c in str(slots.get("incomplete_components", "")).split(",") if c.strip()
        ],
        "selected_materials": selected_materials,
        "confirmed_grade": None,  # grade is suggested automatically by the engine
        "specialized_components": _collect_specialized_components(slots, category),
    }

    # Property details (+ auto plot-grade)
    prop_town = str(slots.get("prop_town"))
    gen_use = str(slots.get("gen_use"))
    plot_area = float(slots.get("plot_area_sqm"))

    plot_grade = select_plot_grade(prop_town, gen_use, plot_area)

    property_details = {
        "plot_area": plot_area,
        "prop_town": prop_town,
        "gen_use": gen_use,
        "plot_grade": plot_grade,  # user never chooses; auto-picked above
    }

    special_items = {"has_elevator": has_elevator, "elevator_stops": elevator_stops}

    other_costs = {
        "fence_percent": 0.0,
        "septic_percent": 0.0,
        "external_works_percent": 0.0,
        "water_tank_cost": 0.0,
        "consultancy_percent": 0.0,
    }

    financial_factors = {"mcf": mcf, "pef": pef}

    payload = {
        "buildings": [building],
        "property_details": property_details,
        "special_items": special_items,
        "other_costs": other_costs,
        "financial_factors": financial_factors,
        "remarks": "Generated by LangGraph agent",
    }

    result_text: str = property_valuation_tool.invoke(payload)
    state["messages"].append({"role": "assistant", "content": result_text})
    return state

def should_calculate(state: ValuationState) -> str:
    remaining = missing_slots(state.get("slots", {}))
    return "CALC" if not remaining else "ASK"

def build_graph():
    builder = StateGraph(ValuationState)
    builder.add_node("extract_info", extract_info_node)
    builder.add_node("ask", ask_next_question_node)
    builder.add_node("calculate", calculate_node)
    builder.add_edge(START, "extract_info")
    builder.add_conditional_edges("extract_info", should_calculate, {"ASK": "ask", "CALC": "calculate"})
    builder.add_edge("ask", "extract_info")
    builder.add_edge("calculate", END)
    builder.config = {"recursion_limit": 100}
    return builder

# ------------------------------
# 8) CLI Runner
# ------------------------------
if __name__ == "__main__":
    print("\n🏗️ Property Valuation Agent (LangGraph)")
    print("Type 'quit' to exit.\n")

    state: ValuationState = initial_state()
    graph = build_graph().compile()

    # Kick off
    state = ask_next_question_node(state)
    print("Bot:", state["messages"][-1]["content"], "\n")

    while True:
        user = input("You: ")
        if user.strip().lower() in {"quit", "exit"}:
            print("👋 Goodbye!")
            break

        state["messages"].append({"role": "user", "content": user})
        state = extract_info_node(state)

        remaining = missing_slots(state.get("slots", {}))
        if remaining:
            state = ask_next_question_node(state)
        else:
            state = calculate_node(state)
            state["asked"] = []

        print("Bot:", state["messages"][-1]["content"], "\n")
