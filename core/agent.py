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
    "num_sections",
    "section_dimensions",
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
    # Removed non-essential fields for special categories
]

SPECIAL_CATEGORIES = ["Fuel Station", "Coffee Washing Site", "Green House"]

CATEGORY_SPECIAL_SLOTS: Dict[str, List[str]] = {
    "Higher Villa": [],
    "Multi-Story Building": [],
    "Apartment / Condominium": [],
    "MPH & Factory Building": [],
    "Fuel Station": ["site_preparation_area", "forecourt_area", "canopy_area", "num_pump_islands", "num_ugt_30m3",
                     "num_ugt_50m3"],
    "Coffee Washing Site": ["cherry_hopper_area", "fermentation_tanks_area", "washing_channels_length",
                            "coffee_drier_area"],
    "Green House": ["greenhouse_area", "in_farm_road_km", "borehole_depth", "land_preparation_area"],
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
    if t in {"yes", "y", "true", "t", "1"}:
        return True
    if t in {"no", "n", "false", "f", "0"}:
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
    cat = slots.get("building_category")
    if isinstance(cat, str) and cat in SPECIAL_CATEGORIES:
        req = list(SPECIAL_CATEGORY_BASE_SLOTS)
    else:
        req = list(BASE_REQUIRED_SLOTS_IN_ORDER)
        if cat == "Apartment / Condominium":
            req = [s for s in req if s not in {"num_floors", "has_elevator", "elevator_stops"}]
        elif cat in {"Higher Villa", "MPH & Factory Building"}:
            req = [s for s in req if s not in {"num_floors", "has_elevator", "elevator_stops"}]

    if isinstance(cat, str) and cat:
        for comp in get_material_components_for_category(cat):
            req.append(f"material__{comp}")
        for sp in CATEGORY_SPECIAL_SLOTS.get(cat, []):
            req.append(sp)

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

    if "section_dimensions" in needed:
        if "section_index" in slots and slots.get("section_index", 0) < int(slots.get("num_sections", 1)):
            needed.remove("section_dimensions")

    return needed


def ask_next_question_node(state: ValuationState) -> ValuationState:
    slots = state.get("slots", {})
    asked = set(state.get("asked", []))
    remaining = [s for s in missing_slots(slots) if s not in asked]

    # Handle the case where we're in the middle of collecting section dimensions
    if "section_index" in slots and slots["section_index"] < int(slots.get("num_sections", 1)):
        s = "section_dimensions"
    elif not remaining:
        return state
    else:
        s = remaining[0]

    cat = slots.get("building_category")
    
    # For special categories, skip directly to their special slots after basic info
    if isinstance(cat, str) and cat in SPECIAL_CATEGORIES and s not in SPECIAL_CATEGORY_BASE_SLOTS + CATEGORY_SPECIAL_SLOTS.get(cat, []):
        next_special = next((slot for slot in CATEGORY_SPECIAL_SLOTS.get(cat, []) if slot not in slots), None)
        if next_special:
            s = next_special

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
        ex = MATERIAL_EXAMPLES.get(comp.lower(), "describe the material clearly")
        q = f"Enter the selected material for {comp} (e.g., {ex}):"
    elif s in SPECIAL_SLOT_EXAMPLES:
        label = s.replace("_", " ")
        q = f"Enter {label} ({SPECIAL_SLOT_EXAMPLES[s]}):"
    elif s == "section_dimensions":
        if "section_dimensions" not in slots:
            slots["section_dimensions"] = []
        if "section_index" not in slots:
            slots["section_index"] = 0
            # Only initialize awaiting_width for non-Apartment/Condominium buildings
            if cat != "Apartment / Condominium":
                slots["awaiting_width"] = False

        idx = slots["section_index"]

        if cat == "Apartment / Condominium":
            q = f"Enter area of section {idx + 1} in sqm (e.g., 50):"
        else:
            if slots.get("awaiting_width", False):
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

    if last_asked == "section_dimensions":
        idx = state["slots"].get("section_index", 0)
        cat = state["slots"].get("building_category")

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
    
    # Create a simple list-style summary
    message = f"""
    **Property Valuation Summary**
    
    **Property:** {building_name}
    **Category:** {category}
    
    **Estimated Value:** ETB {int(float(slots.get('forced_sale_value', 0))):,}
    
    **Property Details:**
    """
    
    # Add property details in a simple list
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

    # --- Prepare payload for property_valuation_tool ---

    # Scalars / factors
    mcf = float(slots.get("mcf", 1.0) or 1.0)
    pef = float(slots.get("pef", 1.0) or 1.0)
    has_elevator = bool(slots.get("has_elevator", False))
    elevator_stops = int(slots.get("elevator_stops") or 0)

    # Building core data
    # FIX: Correctly call the helper function to get the full specialized_components dict
    specialized_components = _collect_specialized_components(slots, category)

    building = {
        "name": str(slots.get("building_name", "Building 1")),
        "category": category,
        "length": None,
        "width": None,
        "num_floors": int(slots.get("num_floors", 1)),
        "has_basement": bool(slots.get("has_basement", False)),
        "is_under_construction": bool(slots.get("is_under_construction", False)),
        "incomplete_components": [
            c.strip() for c in str(slots.get("incomplete_components", "")).split(",") if c.strip()
        ],
        "selected_materials": _collect_selected_materials(slots, category),
        "confirmed_grade": None,
        "specialized_components": specialized_components,  # This is the crucial fix
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
        
        # Extract valuation amount and clean result
        import re
        valuation_amount = "[Calculating...]"
        market_value_match = re.search(r'Estimated Market Value.*?ETB\s*([\d,]+(?:\.[\d]+)?)', result_text)
        if market_value_match:
            valuation_amount = f"ETB {market_value_match.group(1)}"
            
        clean_result = re.sub(r'<[^>]+>', '', result_text)
        clean_result = ' '.join(clean_result.split())
        
        # Start with a clean, compact header
        summary_html = f"""
        <div style="font-family: Arial, sans-serif; max-width: 700px; margin: 0 auto; font-size: 14px;">
            <div style="border-bottom: 1px solid #e2e8f0; padding-bottom: 10px; margin-bottom: 15px;">
                <div style="color: #2d3748; font-size: 20px; font-weight: 600;">
                    {building_name}
                </div>
                <div style="color: #718096; font-size: 13px;">
                    {prop_town} • {category}
                </div>
            </div>
            
            <!-- Valuation Card -->
            <div style="background: #f7fafc; border-radius: 8px; padding: 15px; margin-bottom: 15px;">
                <div style="color: #4a5568; font-size: 14px; margin-bottom: 5px;">
                    Estimated Forced Sale Value
                </div>
                <div style="color: #2d3748; font-size: 24px; font-weight: 700;">
                    {valuation_amount}
                </div>
            </div>
        
            <!-- Property Specs -->
            <div style="display: flex; gap: 10px; margin-bottom: 15px; font-size: 13px;">
                <div style="flex: 1; background: white; border: 1px solid #e2e8f0; border-radius: 6px; padding: 10px;">
                    <div style="color: #718096; margin-bottom: 5px; font-size: 12px;">Plot Area</div>
                    <div style="font-weight: 600;">{plot_area:,.2f} sqm</div>
                </div>
                <div style="flex: 1; background: white; border: 1px solid #e2e8f0; border-radius: 6px; padding: 10px;">
                    <div style="color: #718096; margin-bottom: 5px; font-size: 12px;">Use</div>
                    <div style="font-weight: 600;">{gen_use}</div>
                </div>
                <div style="flex: 1; background: white; border: 1px solid #e2e8f0; border-radius: 6px; padding: 10px;">
                    <div style="color: #718096; margin-bottom: 5px; font-size: 12px;">Floors</div>
                    <div style="font-weight: 600;">{slots.get('num_floors', 'N/A')}</div>
                </div>
            </div>
            
            <!-- Show Details Button -->
            <div>
                <button onclick="this.nextElementSibling.style.display='block';this.style.display='none'" 
                        style="background: none;
                               border: 1px solid #4299e1;
                               color: #4299e1;
                               padding: 8px 16px;
                               border-radius: 4px;
                               cursor: pointer;
                               font-size: 13px;">
                    Show Details
                </button>
                <div style="display: none; margin-top: 15px; padding: 15px; background: white; border: 1px solid #e2e8f0; border-radius: 6px;">
                    <div style="color: #4a5568; margin-bottom: 10px; font-weight: 600;">Valuation Details</div>
                    <div style="color: #4a5568; font-size: 13px; line-height: 1.5;">
                        {clean_result}
                    </div>
        """
        
        # Add materials section if available
        materials = _collect_selected_materials(slots, category)
        if materials:
            materials_html = """
                    <div style="margin-top: 15px; padding-top: 15px; border-top: 1px solid #e2e8f0;">
                        <div style="color: #4a5568; margin-bottom: 10px; font-weight: 600;">Construction Materials</div>
                        <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 10px;">
            """
            
            for component, material in materials.items():
                if material:
                    materials_html += f"""
                        <div style="padding: 8px; background: #f8fafc; border-radius: 4px; font-size: 13px;">
                            <div style="color: #718096; font-size: 12px;">{component.replace('_', ' ').title()}</div>
                            <div style="font-weight: 500;">{material}</div>
                        </div>
                    """
            
            summary_html += materials_html + """
                        </div>
                    </div>
            """
        
        # Close all open divs
        summary_html += """
                </div>
            </div>
        </div>
        """
        
        # Clean up whitespace
        summary_text = "\n".join(line.strip() for line in summary_html.split('\n') if line.strip())
    
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