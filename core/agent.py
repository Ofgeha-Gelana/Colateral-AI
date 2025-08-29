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
VALID_COLLATERAL_TYPES = ["House", "Car"]

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
    "collateral_type",  # First question: House or Car
    "building_category",
    "prop_town",
    "gen_use",
    "plot_area_sqm",
    "length",
    "width",
    "num_floors",
    "has_basement",
    "is_under_construction",
    "incomplete_components",
    "has_elevator",
    "elevator_stops",
    "num_sections",
    "section_dimensions",
    "mcf",
    "pef",
]

SPECIAL_CATEGORY_BASE_SLOTS: List[str] = [
    "prop_town"  # Only property location is required for special categories
]

SPECIAL_CATEGORIES = ["Fuel Station", "Coffee Washing Site", "Green House"]

CATEGORY_SPECIAL_SLOTS: Dict[str, List[str]] = {
    "Higher Villa": [],
    "Multi-Story Building": [],
    "Apartment / Condominium": [],
    "MPH & Factory Building": ["height_meters", "has_basement"],
    "Fuel Station": ["site_preparation_area", "forecourt_area", "canopy_area", "num_pump_islands", "num_ugt_30m3",
                     "num_ugt_50m3"],
    "Coffee Washing Site": ["cherry_hopper_area", "fermentation_tanks_area", "washing_channels_length",
                            "coffee_drier_area"],
    "Green House": ["greenhouse_area", "in_farm_road_km", "borehole_depth", "land_preparation_area"],
}

SPECIAL_SLOT_EXAMPLES: Dict[str, str] = {
    "height_meters": "e.g., 3.5 (in meters, will determine if <=4m or >4m category applies)",
    "has_basement": "e.g., yes/no (whether the building has a basement)",
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
    # Handle common typos and variations
    if t in {"yes", "y", "true", "t", "1", "ye", "yea", "yep", "ok", "okay", "sure"}:
        return True
    if t in {"no", "n", "false", "f", "0", "nah", "nope", "not", "mo", "mo]", "nop"}:
        return False
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
    # If collateral type is Car, no other questions needed
    if slots.get("collateral_type", "").lower() == "car":
        return []
        
    cat = slots.get("building_category")
    if isinstance(cat, str) and cat in SPECIAL_CATEGORIES:
        req = list(SPECIAL_CATEGORY_BASE_SLOTS)
        # Add only the special slots for special categories
        for sp in CATEGORY_SPECIAL_SLOTS.get(cat, []):
            req.append(sp)
        # Skip elevator questions for special categories
        if "has_elevator" in req:
            req.remove("has_elevator")
        if "elevator_stops" in req:
            req.remove("elevator_stops")
    else:
        req = list(BASE_REQUIRED_SLOTS_IN_ORDER)
        # Only ask about floors and elevators for Multi-Story Building
        if cat != "Multi-Story Building":
            req = [s for s in req if s not in {"num_floors", "has_elevator", "elevator_stops"}]
        
        # For MPH & Factory, we'll get dimensions in sections, so remove length/width from base questions
        # Also, automatically set gen_use to Commercial for MPH & Factory
        if cat == "MPH & Factory Building":
            req = [s for s in req if s not in {"length", "width", "gen_use", "num_sections", "section_dimensions"}]
            slots["gen_use"] = "Commercial"  # Auto-set to Commercial
            # Add special slots for MPH & Factory Building
            for sp in CATEGORY_SPECIAL_SLOTS.get(cat, []):
                req.append(sp)
            
        # Add material components for non-special categories
        if cat:
            for comp in get_material_components_for_category(cat):
                req.append(f"material__{comp}")

    seen = set()
    final: List[str] = []
    for r in req:
        if r not in seen:
            final.append(r)
            seen.add(r)
    return final


def missing_slots(slots: Dict[str, object]) -> List[str]:
    # If collateral type is Car, no slots needed
    if slots.get("collateral_type", "").lower() == "car":
        return []
    
    needed = [s for s in current_required_slots(slots) if s not in slots]
    
    cat = slots.get("building_category")
    
    # Skip section-related slots for Apartment/Condominium
    if cat == "Apartment / Condominium":
        needed = [s for s in needed if s not in {"num_sections", "section_dimensions", "has_basement"}]
    
    # Skip section-related slots for MPH & Factory Building
    if cat == "MPH & Factory Building":
        needed = [s for s in needed if s not in {"num_sections", "section_dimensions"}]

    # Handle other conditional fields
    if "has_elevator" in slots and not slots["has_elevator"]:
        needed = [s for s in needed if s != "elevator_stops"]
    if "is_under_construction" in slots and not slots["is_under_construction"]:
        needed = [s for s in needed if s != "incomplete_components"]

    # Only process section_dimensions if it's still needed and not for Apartment/Condominium or MPH & Factory
    if "section_dimensions" in needed and cat not in ["Apartment / Condominium", "MPH & Factory Building"]:
        if "section_index" in slots and slots["section_index"] < int(slots.get("num_sections", 1)):
            needed.remove("section_dimensions")

    return needed


def ask_next_question_node(state: ValuationState) -> ValuationState:
    slots = state.get("slots", {})
    asked = set(state.get("asked", []))
    remaining = [s for s in missing_slots(slots) if s not in asked]
    

    # Get building category
    cat = slots.get("building_category")
    
    # Skip section dimensions for Apartment/Condominium and MPH & Factory Building
    if cat in ["Apartment / Condominium", "MPH & Factory Building"]:
        remaining = [s for s in remaining if s not in {"num_sections", "section_dimensions"}]
    
    # Handle collateral type selection
    if "collateral_type" in remaining and "collateral_type" not in asked:
        question = "Select collateral type (House or Car):"
        state["messages"].append({"role": "assistant", "content": question})
        state.setdefault("asked", []).append("collateral_type")
        return state
        
    # If Car is selected, show pending message and end the flow
    if slots.get("collateral_type", "").lower() == "car":
        state["messages"].append({"role": "assistant", 
                                "content": "🚗 Car collateral valuation is currently in development. Please check back later!"})
        # Mark all slots as asked to prevent further questions
        state["asked"] = list(current_required_slots(slots))
        return state

    # Handle the case where we're in the middle of collecting section dimensions
    if "section_index" in slots and slots["section_index"] < int(slots.get("num_sections", 1)):

        s = "section_dimensions"
    elif not remaining:
        return state
    else:
        s = remaining[0] if remaining else None

    if s is None:
        return state

    # For special categories, skip directly to their special slots after basic info
    if isinstance(cat, str) and cat in SPECIAL_CATEGORIES and s not in SPECIAL_CATEGORY_BASE_SLOTS + CATEGORY_SPECIAL_SLOTS.get(cat, []):
        next_special = next((slot for slot in CATEGORY_SPECIAL_SLOTS.get(cat, []) if slot not in slots), None)
        if next_special:
            s = next_special
    
    # For MPH & Factory Building, ensure height_meters is asked before materials
    if cat == "MPH & Factory Building" and "height_meters" not in slots and s.startswith("material__"):
        s = "height_meters"

    options_map = {
        "building_category": VALID_CATEGORIES,
        "gen_use": VALID_USE,
        "prop_town": VALID_TOWN_CLASSES,
    }

    label_map = {
        "building_name": "building name",
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
        "building_name": "e.g., Villa Sunshine",
        "num_floors": "e.g., 3",
        "incomplete_components": "e.g., Foundation, Roof (or leave empty)",
        "plot_area_sqm": "e.g., 450",
        "mcf": "Reply 1.0 if unsure",
        "pef": "Reply 1.0 if unsure",
        "elevator_stops": "e.g., 5",
    }

    if s in options_map:
        choices = options_map[s]
        if s == "prop_town":
            body = format_choices_with_examples(choices, TOWN_CLASS_EXAMPLES)
        else:
            body = "\n".join(f"{i + 1}. {c}" for i, c in enumerate(choices))
        q = f"Please select {s.replace('_', ' ')}:\n{body}\n(Reply with the number)"
        state["slots"]["__expected_choices__"] = (s, choices)
    elif s.startswith("material__"):
        comp = s.split("__", 1)[1]
        material_options = MATERIAL_EXAMPLES.get(comp.lower(), "Reinforced concrete; Stone; Mud block").split("; ")
        body = "\n".join(f"{i + 1}. {option}" for i, option in enumerate(material_options))
        q = f"Select material for {comp}:\n{body}\n(Reply with the number)"
        state["slots"]["__expected_choices__"] = (s, material_options)
    elif s in SPECIAL_SLOT_EXAMPLES:
        label = s.replace("_", " ")
        q = f"Enter {label} ({SPECIAL_SLOT_EXAMPLES[s]}):"
    elif s == "section_dimensions":
        idx = state["slots"].get("section_index", 0)
        cat = state["slots"].get("building_category")

        if cat == "Apartment / Condominium":
            q = f"Enter area of section {idx + 1} in sqm (e.g., 50):"
        else:
            if state["slots"].get("awaiting_width", False):
                q = f"Enter width of section {idx + 1} in meters (e.g., 5):"
            else:
                q = f"Enter length of section {idx + 1} in meters (e.g., 10):"
    elif s in label_map:
        label = label_map[s]
        ex = generic_examples.get(s)
        if s in {"has_basement", "is_under_construction", "has_elevator"}:
            q = f"{label} (yes/no)"
        else:
            q = f"Enter {label}{f' ({ex})' if ex else ''}:"
    else:
        q = f"Please provide the value for: {s}"

    state["messages"].append({"role": "assistant", "content": q})
    state.setdefault("asked", []).append(s)
    return state


# ------------------------------
# 6) Info extraction node
# ------------------------------
def extract_info_node(state: ValuationState) -> ValuationState:
    if not state.get("messages"):
        return state
    last = state["messages"][-1]
    if last.get("role") != "user":
        return state

    asked_slots = state.get("asked", [])
    if not asked_slots:
        return state
    last_asked = asked_slots[-1]
    content = last.get("content", "").strip()

    if last_asked in {"has_basement", "has_elevator", "is_under_construction"}:
        b = _boolify(content)
        if b is not None:
            state["slots"][last_asked] = b
        else:
            state["messages"].append(
                {"role": "assistant", "content": "I couldn't understand that. Please reply with 'yes' or 'no'."})
        return state

    expected = state["slots"].get("__expected_choices__")
    if expected:
        slot, choices = expected
        if content.isdigit():
            idx = int(content) - 1
            if 0 <= idx < len(choices):
                state["slots"][slot] = choices[idx]
                state["slots"].pop("__expected_choices__", None)
            else:
                state["messages"].append(
                    {"role": "assistant", "content": f"Please select a number between 1 and {len(choices)}."})
        else:
            state["messages"].append(
                {"role": "assistant", "content": "Please enter a valid number corresponding to your choice."})
        return state

    if last_asked == "building_category":
        # Handle the selected category
        if content.isdigit() and 0 < int(content) <= len(VALID_CATEGORIES):
            selected_category = VALID_CATEGORIES[int(content) - 1]
            state["slots"][last_asked] = selected_category
            
            # For Apartment/Condominium, set default values for sections
            if selected_category == "Apartment / Condominium":
                state["slots"]["num_sections"] = "1"
                state["slots"]["section_dimensions"] = [{"area": "100"}]
                # Mark these as asked so they're skipped
                state.setdefault("asked", []).extend(["num_sections", "section_dimensions"])
        else:
            state["messages"].append({"role": "assistant", "content": "Please select a valid number from the list."})
        return state

    if last_asked == "section_dimensions":
        idx = state["slots"].get("section_index", 0)
        cat = state["slots"].get("building_category")

        # Initialize section_dimensions if it doesn't exist
        if "section_dimensions" not in state["slots"]:
            state["slots"]["section_dimensions"] = []

        if cat == "Apartment / Condominium":
            try:
                area = float(content)
                while len(state["slots"]["section_dimensions"]) <= idx:
                    state["slots"]["section_dimensions"].append({})
                state["slots"]["section_dimensions"][idx]["area"] = area
                state["slots"]["section_index"] += 1

                num_sections = int(state["slots"].get("num_sections", 1))
                if state["slots"]["section_index"] >= num_sections:
                    state["slots"].pop("section_index", None)
                    state["slots"].pop("awaiting_width", None)
                    if "section_dimensions" not in state["asked"]:
                        state["asked"].append("section_dimensions")
                return state
            except ValueError:
                state["messages"].append({"role": "assistant", "content": "Please enter a valid number for the area."})
                return state
        else:
            if state["slots"].get("awaiting_width", False):
                try:
                    width = float(content)
                    while len(state["slots"]["section_dimensions"]) <= idx:
                        state["slots"]["section_dimensions"].append({})
                    state["slots"]["section_dimensions"][idx]["width"] = width
                    state["slots"]["section_index"] += 1
                    state["slots"]["awaiting_width"] = False

                    num_sections = int(state["slots"].get("num_sections", 1))
                    if state["slots"]["section_index"] >= num_sections:
                        state["slots"].pop("section_index", None)
                        state["slots"]["section_index"] = None
                        state["slots"].pop("awaiting_width", None)
                        if "section_dimensions" not in state["asked"]:
                            state["asked"].append("section_dimensions")
                    return state
                except ValueError:
                    state["messages"].append(
                        {"role": "assistant", "content": "Please enter a valid number for the width."})
                    return state
            else:
                try:
                    length = float(content)
                    while len(state["slots"]["section_dimensions"]) <= idx:
                        state["slots"]["section_dimensions"].append({})
                    state["slots"]["section_dimensions"][idx]["length"] = length
                    state["slots"]["awaiting_width"] = True
                    return state
                except ValueError:
                    state["messages"].append(
                        {"role": "assistant", "content": "Please enter a valid number for the length."})
                    return state

    state["slots"][last_asked] = content
    return state


# ------------------------------
# 7) Confirmation / summary nodes
# ------------------------------
def summary_confirmation_node(state: ValuationState) -> ValuationState:
    import streamlit as st
    
    slots = state.get("slots", {})
    messages = state.get("messages", [])
    category = slots.get("building_category", "")
    building_name = slots.get("building_name", "Unnamed Property")
    
    # Generate basic property details
    property_details = [
        ("Location", slots.get('prop_town', 'Not specified')),
        ("Property Use", slots.get('gen_use', 'Not specified')),
        ("Plot Area", f"{slots.get('plot_area_sqm', '0')} sqm"),
        ("Category", category)
    ]
    
    # Generate detailed information
    detailed_info = []
    if category in SPECIAL_CATEGORIES:
        special_params = CATEGORY_SPECIAL_SLOTS.get(category, [])
        for param in special_params:
            if param in slots:
                detailed_info.append((param.replace('_', ' ').title(), str(slots[param])))
    else:
        if slots.get("section_dimensions"):
            for i, sec in enumerate(slots["section_dimensions"], 1):
                if "length" in sec and "width" in sec:
                    detailed_info.append((f"Section {i} Dimensions", f"{sec['length']}m × {sec['width']}m"))
                elif "area" in sec:
                    detailed_info.append((f"Section {i} Area", f"{sec['area']} sqm"))
        
        if "num_floors" in slots:
            detailed_info.append(("Number of Floors", slots['num_floors']))
        if "has_elevator" in slots:
            detailed_info.append(("Has Elevator", 'Yes' if slots['has_elevator'] else 'No'))
        if "has_basement" in slots:
            detailed_info.append(("Has Basement", 'Yes' if slots['has_basement'] else 'No'))
    
    # Create a detailed summary message with all calculations
    message = f"""
    **Property Valuation Summary**
    
    **Property:** {building_name}
    **Category:** {category}
    
    **Estimated Value:** ETB {int(float(slots.get('forced_sale_value', 0))):,}
    
    **Property Details:**
    """
    
    # Add property details
    all_details = property_details + detailed_info
    for label, value in all_details:
        message += f"- **{label}:** {value}\n"
    
    # Add confirmation prompt
    message += """
    
    Please confirm if this information is correct by typing:
    - 'yes' to confirm
    - 'no' to start over
    """
    
    # Add the message to the chat
    messages.append({"role": "assistant", "content": message})
    state.setdefault("asked", []).append("_confirmation")
    return state


def process_confirmation_node(state: ValuationState) -> ValuationState:
    slots = state.get("slots", {})
    messages = state.get("messages", [])
    if not messages:
        return state
    user_response = messages[-1]["content"].strip().lower()
    if user_response in {"yes", "y", "proceed", "confirm"}:
        slots["_confirmed"] = True
        messages.append({"role": "assistant", "content": "✅ Proceeding with valuation..."})
    elif user_response in {"no", "n", "cancel", "stop"}:
        messages.append({"role": "assistant", "content": "❌ Valuation cancelled. Type 'quit' or start over."})
        slots["_confirmed"] = False
    else:
        messages.append({"role": "assistant", "content": "Please respond with 'yes' or 'no'."})
    return state


def should_calculate(state: ValuationState) -> str:
    slots = state.get("slots", {})
    
    # If car is selected, don't proceed with valuation
    if slots.get("collateral_type", "").lower() == "car":
        return "ASK"  # This will prevent further processing
    
    remaining = missing_slots(slots)

    if "section_dimensions" in remaining:
        return "ASK"

    if remaining:
        return "ASK"
    elif not slots.get("_confirmed", False):
        return "CONFIRM"
    return "CALC"


# ------------------------------
# 8) Material / specialized helpers
# ------------------------------
def _collect_selected_materials(slots: Dict[str, object], category: str) -> Dict[str, str]:
    comps = get_material_components_for_category(category)
    out: Dict[str, str] = {}
    for c in comps:
        out[c] = str(slots.get(f"material__{c}", "")).strip()
    return out


def _collect_specialized_components(slots: Dict[str, object], category: str) -> Dict[str, float | int]:
    spec = {}
    for key in CATEGORY_SPECIAL_SLOTS.get(category, []):
        val = slots.get(key)
        if val is None or val == "":
            continue
        try:
            if key.startswith("num_"):
                spec[key] = int(val)
            else:
                spec[key] = float(val)
        except Exception:
            spec[key] = val

    # Add calculated total area for relevant categories
    if category in {"Higher Villa", "Multi-Story Building", "MPH & Factory Building", "Apartment / Condominium"}:
        total_area = 0.0
        if category == "Apartment / Condominium":
            total_area = sum(float(sec.get("area", 0)) for sec in slots.get("section_dimensions", []))
        elif category == "MPH & Factory Building":
            # For MPH & Factory, use plot area as total building area
            total_area = float(slots.get("plot_area_sqm", 0) or 0)
        else:
            total_area = sum(
                float(sec.get("length", 0)) * float(sec.get("width", 0)) for sec in slots.get("section_dimensions", []))
        spec["total_building_area"] = total_area

    return spec


def select_plot_grade(location: str, use_type: str, plot_area: float) -> str:
    try:
        town_data = PLOT_PRICES.get(location, {})
        use_data = town_data.get(use_type, {})
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


def calculate_node(state: ValuationState) -> ValuationState:
    slots = state.get("slots", {})
    category = slots.get("building_category")

    # Set default values for Apartment/Condominium if not already set
    if category == "Apartment / Condominium":
        if "num_sections" not in slots:
            slots["num_sections"] = "1"
        if "section_dimensions" not in slots:
            slots["section_dimensions"] = [{"area": "100"}]  # Default area of 100 sqm

    # --- Prepare payload for property_valuation_tool ---

    # Scalars / factors
    mcf = float(slots.get("mcf", 1.0) or 1.0)
    pef = float(slots.get("pef", 1.0) or 1.0)
    has_elevator = bool(slots.get("has_elevator", False))
    elevator_stops = int(slots.get("elevator_stops") or 0)

    # Building core data
    specialized_components = _collect_specialized_components(slots, category)

    # Common building fields
    building = {
        "name": str(slots.get("building_name", "Building 1")),
        "category": category,
        "length": float(slots.get("length", 0)) if slots.get("length") else None,
        "width": float(slots.get("width", 0)) if slots.get("width") else None,
        "num_floors": int(slots.get("num_floors", 1)),
        "has_basement": bool(slots.get("has_basement", False)),
        "is_under_construction": bool(slots.get("is_under_construction", False)),
        "incomplete_components": [
            c.strip() for c in str(slots.get("incomplete_components", "")).split(",") if c.strip()
        ],
        "selected_materials": _collect_selected_materials(slots, category),
        "confirmed_grade": None,
        "specialized_components": specialized_components,
    }

    # For special categories, ensure we have the required fields
    if category in ["Fuel Station", "Coffee Washing Site", "Green House"]:
        # Add any additional required fields for special categories
        if category == "Fuel Station":
            building.update({
                "length": float(slots.get("length", 0)) or 0.0,
                "width": float(slots.get("width", 0)) or 0.0,
                "num_floors": 1,  # Fuel stations are typically single-story
            })
        elif category == "Coffee Washing Site":
            building.update({
                "length": float(slots.get("length", 0)) or 0.0,
                "width": float(slots.get("width", 0)) or 0.0,
                "num_floors": 1,  # Coffee washing sites are typically single-story
            })
        elif category == "Green House":
            building.update({
                "length": float(slots.get("length", 0)) or 0.0,
                "width": float(slots.get("width", 0)) or 0.0,
                "num_floors": 1,  # Green houses are typically single-story
            })

    # Property details (+ auto plot-grade)
    prop_town = str(slots.get("prop_town", "Unknown"))
    gen_use = str(slots.get("gen_use", "Commercial"))  # Default to Commercial if not specified
    
    try:
        plot_area = float(slots.get("plot_area_sqm", 0) or 0)
    except (TypeError, ValueError):
        plot_area = 0.0  # Default to 0 if not provided or invalid
        
    plot_grade = select_plot_grade(prop_town, gen_use, plot_area)

    property_details = {
        "plot_area": plot_area,
        "prop_town": prop_town,
        "gen_use": gen_use,
        "plot_grade": plot_grade,
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

    # --- Invoke the tool and get the result ---
    try:
        # The tool returns a formatted string, not JSON
        result_text = property_valuation_tool.invoke(payload)
        
        # Extract building name or use a default
        building_name = slots.get('building_name', 'the property')
        
        # Extract valuation amounts
        import re
        valuation_amount = "[Calculating...]"
        market_value = "[Not available]"
        
        market_value_match = re.search(r'Estimated Market Value.*?ETB\s*([\d,]+(?:\.[\d]+)?)', result_text)
        forced_value_match = re.search(r'Estimated Forced Sale Value.*?ETB\s*([\d,]+(?:\.[\d]+)?)', result_text)
        
        if market_value_match:
            market_value = f"ETB {market_value_match.group(1)}"
        if forced_value_match:
            valuation_amount = f"ETB {forced_value_match.group(1)}"
            
        # Clean up the result text
        clean_result = re.sub(r'<[^>]+>', '', result_text)
        clean_result = ' '.join(clean_result.split())
        
        # Get materials used
        materials = _collect_selected_materials(slots, category)
        
        # Create a summary with property details and valuation
        summary_text = f"""
📌 **PROPERTY DETAILS**
**Location:** {prop_town}
**Category:** {category}
**Property Use:** {gen_use}
**Plot Area:** {plot_area:,.2f} sqm

🏠 **PROPERTY VALUATION SUMMARY**

**Market Value:** {market_value}
**Forced Sale Value (70% of Market Value):** {valuation_amount}
"""
        
        # Add the final summary message
        state["messages"].append({
            "role": "assistant",
            "content": summary_text
        })
    except Exception as e:
        # If there's an error, show a friendly error message
        error_message = [
            "❌ **Valuation Failed** ❌",
            "The tool was unable to calculate the valuation due to an error.",
            "",
            f"Error details: {str(e)}",
            "",
            "Please check the input data and try again. If the problem persists, contact support."
        ]
        state["messages"].append({"role": "assistant", "content": "\n".join(error_message)})
        summary_text = "\n".join(error_message)
    state["messages"].append({"role": "assistant", "content": summary_text})
    return state
# ------------------------------
# 8) CLI Runner (simplified)
# ------------------------------
if __name__ == "__main__":
    print("\n🏗️ Property Valuation Agent (LangGraph)")
    print("Type 'quit' to exit.\n")
    state: ValuationState = initial_state()
    state = ask_next_question_node(state)
    print("Bot:", state["messages"][-1]["content"], "\n")

    while True:
        user = input("You: ")
        if user.strip().lower() in {"quit", "exit"}:
            print("👋 Goodbye!")
            break

        state["messages"].append({"role": "user", "content": user})
        state = extract_info_node(state)

        # Determine the next step based on the updated state
        next_step = should_calculate(state)

        if next_step == "ASK":
            state = ask_next_question_node(state)
        elif next_step == "CONFIRM":
            state = summary_confirmation_node(state)
        elif next_step == "CALC":
            state = calculate_node(state)

        print("Bot:", state["messages"][-1]["content"], "\n")