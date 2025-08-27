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
# 1) Model
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

MATERIAL_EXAMPLES = {
    "foundation": "Reinforced concrete; Stone; Mud block",
    "roof": "Corrugated iron; Tile; Concrete slab",
    "floor": "Ceramic; Wood; Concrete",
    "ceiling": "Gypsum board; Wood; Concrete",
    "metal work": "Aluminum; Steel; Iron",
    "sanitary": "Standard; Premium; Basic",
}

# ------------------------------
# 3) Load JSON data
# ------------------------------
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")

def _load_json(filename: str) -> dict:
    path = os.path.join(DATA_DIR, filename)
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

PLOT_PRICES = _load_json("location_data.json")
MATERIAL_MAPPINGS = _load_json("material_mappings.json")

def get_material_components_for_category(category: str) -> List[str]:
    cat_map = MATERIAL_MAPPINGS.get(category, {})
    comps = list(cat_map.keys())
    return comps if comps else ["foundation", "roof", "floor", "ceiling", "metal work", "sanitary"]

# ------------------------------
# 4) Slots configuration
# ------------------------------
BASE_REQUIRED_SLOTS_IN_ORDER: List[str] = [
    "building_name",
    "building_category",
    "num_sections",           # new
    "section_dimensions",     # new, store list of dicts [{length, width}]
    "num_floors",
    "has_basement",
    "is_under_construction",
    "incomplete_components",
    "plot_area_sqm",
    "prop_town",
    "gen_use",
    "mcf",
    "pef",
    "has_elevator",
    "elevator_stops",
]

SPECIAL_CATEGORY_BASE_SLOTS: List[str] = [
    "building_name",
    "building_category",
    "plot_area_sqm",
    "prop_town",
    "gen_use",
    "mcf",
    "pef",
    "has_elevator",
    "elevator_stops",
]

SPECIAL_CATEGORIES = ["Fuel Station", "Coffee Washing Site", "Green House"]

CATEGORY_SPECIAL_SLOTS: Dict[str, List[str]] = {
    "Higher Villa": [],
    "Multi-Story Building": [],
    "Apartment / Condominium": [],
    "MPH & Factory Building": [],
    "Fuel Station": ["site_preparation_area","forecourt_area","canopy_area","num_pump_islands","num_ugt_30m3","num_ugt_50m3"],
    "Coffee Washing Site": ["cherry_hopper_area","fermentation_tanks_area","washing_channels_length","coffee_drier_area"],
    "Green House": ["greenhouse_area","in_farm_road_km","borehole_depth","land_preparation_area"],
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
# 5) Conversation state
# ------------------------------
class ValuationState(TypedDict):
    messages: List[Dict]
    slots: Dict[str, object]
    asked: List[str]

def initial_state() -> ValuationState:
    return {"messages": [], "slots": {}, "asked": []}

def _boolify(text: str) -> Optional[bool]:
    t = text.strip().lower()
    if t in {"yes","y","true","t","1"}: return True
    if t in {"no","n","false","f","0"}: return False
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
    Determine which slots (questions) need to be asked for the given building category.

    - Multi-Story: ask floors, elevator, number of sections, dimensions per section
    - Higher Villa: ask number of sections, dimensions per section (no floors/elevator)
    - Apartment / Condominium: ask number of sections, only total area (skip floors/elevator/length/width)
    - Special categories (Fuel Station, Coffee Site, Green House): skip generic building questions
    """
    cat = slots.get("building_category")

    # Base slots: special categories skip generic questions
    if isinstance(cat, str) and cat in SPECIAL_CATEGORIES:
        req = list(SPECIAL_CATEGORY_BASE_SLOTS)
    else:
        req = list(BASE_REQUIRED_SLOTS_IN_ORDER)

        if cat == "Apartment / Condominium":
            # Skip length, width, floors, elevator questions
            req = [s for s in req if s not in {
                "num_floors",
                "has_elevator",
                "elevator_stops"
            }]
            # Keep sections (for total area calculation)
            if "num_sections" not in req:
                req.append("num_sections")
            if "section_dimensions" not in req:
                req.append("section_dimensions")  # only area per section
        elif cat == "Multi-Story Building":
            # Ask floors + elevator + sections
            if "num_floors" not in req:
                req.append("num_floors")
            if "has_elevator" not in req:
                req.append("has_elevator")
            if "num_sections" not in req:
                req.append("num_sections")
            if "section_dimensions" not in req:
                req.append("section_dimensions")
        else:
            # Higher Villa and others: sections only
            if "num_sections" not in req:
                req.append("num_sections")
            if "section_dimensions" not in req:
                req.append("section_dimensions")

    # Add materials
    if isinstance(cat, str) and cat:
        for comp in get_material_components_for_category(cat):
            req.append(f"material__{comp}")
        for sp in CATEGORY_SPECIAL_SLOTS.get(cat, []):
            req.append(sp)

    # Deduplicate while preserving order
    seen = set()
    final: List[str] = []
    for r in req:
        if r not in seen:
            final.append(r)
            seen.add(r)
    return final


def missing_slots(slots: Dict[str, object]) -> List[str]:
    needed = [s for s in current_required_slots(slots) if s not in slots]

    if "has_elevator" in slots and slots.get("has_elevator") is False:
        needed = [s for s in needed if s != "elevator_stops"]
    if "is_under_construction" in slots and slots.get("is_under_construction") is False:
        needed = [s for s in needed if s != "incomplete_components"]
    return needed


def ask_next_question_node(state: ValuationState) -> ValuationState:
    slots = state.get("slots", {})
    asked = set(state.get("asked", []))
    remaining = [s for s in missing_slots(slots) if s not in asked]
    if not remaining:
        return state

    s = remaining[0]
    cat = slots.get("building_category")

    # Choice lists
    options_map = {
        "building_category": VALID_CATEGORIES,
        "gen_use": VALID_USE,
        "prop_town": VALID_TOWN_CLASSES,
    }

    label_map = {
        "num_floors": "number of floors",
        "has_basement": "Does the building have a basement?",
        "is_under_construction": "Is the building under construction?",
        "incomplete_components": "List incomplete components",
        "plot_area_sqm": "plot area (sqm)",
        "mcf": "Market Condition Factor (MCF)",
        "pef": "Property Enhancement Factor (PEF)",
        "has_elevator": "Is there an elevator?",
        "elevator_stops": "How many elevator stops?",
        "num_sections": "How many sections does the building have?",
    }

    generic_examples = {
        "num_floors":"e.g., 3",
        "incomplete_components":"e.g., Foundation, Roof (or leave empty)",
        "plot_area_sqm":"e.g., 450",
        "mcf":"Reply 1.0 if unsure",
        "pef":"Reply 1.0 if unsure",
        "elevator_stops":"e.g., 5",
    }

    # Handling choice slots
    if s in options_map:
        choices = options_map[s]
        if s == "prop_town":
            body = format_choices_with_examples(choices, TOWN_CLASS_EXAMPLES)
        else:
            body = "\n".join(f"{i+1}. {c}" for i, c in enumerate(choices))
        q = f"Please select {s.replace('_',' ')}:\n{body}\n(Reply with the number)"
        state["slots"]["__expected_choices__"] = (s, choices)

    # Material slots
    elif s.startswith("material__"):
        comp = s.split("__",1)[1]
        ex = MATERIAL_EXAMPLES.get(comp.lower(), "describe the material clearly")
        q = f"Enter the selected material for {comp} (e.g., {ex}):"

    # Category special slots
    elif s in SPECIAL_SLOT_EXAMPLES:
        label = s.replace("_"," ")
        q = f"Enter {label} ({SPECIAL_SLOT_EXAMPLES[s]}):"

    # Sections handling
    elif s == "section_dimensions":
        # Generate sub-slots dynamically based on num_sections
        num_sections = int(slots.get("num_sections", 1))
        if "section_index" not in slots:
            slots["section_index"] = 0
            slots["section_dimensions"] = []

        idx = slots["section_index"]
        if idx < num_sections:
            # Ask length first, then width
            if cat == "Apartment / Condominium":
                q = f"Enter area of section {idx+1} in sqm (e.g., 50):"
                slots["awaiting_width"] = False  # Not used for condos
            else:
                if "awaiting_width" in slots and slots["awaiting_width"]:
                    q = f"Enter width of section {idx+1} in meters (e.g., 5):"
                    slots["awaiting_width"] = False
                else:
                    q = f"Enter length of section {idx+1} in meters (e.g., 10):"
                    slots["awaiting_width"] = True
        else:
            # Done
            state["slots"].pop("section_index", None)
            state["slots"].pop("awaiting_width", None)
            return ask_next_question_node(state)

    # Generic scalar slots
    elif s in label_map:
        label = label_map[s]
        ex = generic_examples.get(s)
        if s in {"has_basement","is_under_construction","has_elevator"}:
            q = f"{label} (yes/no)"
        else:
            q = f"Enter {label}{f' ({ex})' if ex else ''}:"

    else:
        q = f"Please provide the value for: {s}"

    state["messages"].append({"role":"assistant","content":q})
    state.setdefault("asked",[]).append(s)
    return state


def calculate_node(state: ValuationState) -> ValuationState:
    slots = state.get("slots", {})
    category = slots.get("building_category")

    # Handle multi-section buildings
    if category in {"Higher Villa","Multi-Story Building"}:
        total_area = sum(float(sec["length"])*float(sec["width"]) for sec in slots.get("section_dimensions",[]))
    elif category == "Apartment / Condominium":
        total_area = sum(float(sec.get("area",0)) for sec in slots.get("section_dimensions",[]))
    else:
        total_area = float(slots.get("plot_area_sqm",0))

    result_text = f"🏗️ Total building area calculated: {total_area} sqm"
    state["messages"].append({"role":"assistant","content":result_text})
    return state

# ------------------------------
# 8) CLI Runner (simplified)
# ------------------------------
if __name__=="__main__":
    print("\n🏗️ Property Valuation Agent (LangGraph)")
    print("Type 'quit' to exit.\n")

    state: ValuationState = initial_state()

    # Kickoff
    state = ask_next_question_node(state)
    print("Bot:", state["messages"][-1]["content"], "\n")

    while True:
        user = input("You: ")
        if user.strip().lower() in {"quit","exit"}:
            print("👋 Goodbye!")
            break

        state["messages"].append({"role":"user","content":user})
        state = ask_next_question_node(state)
        print("Bot:", state["messages"][-1]["content"], "\n")
