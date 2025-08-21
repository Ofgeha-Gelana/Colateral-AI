"""
agent.py
LangGraph agent implementation for the Property Valuation Chatbot.
- Collects required inputs via slot-filling dialogue
- Calls the valuation tool when slots are complete
- Returns a formatted valuation report

Requires:
  - tools.py (with property_valuation_tool)
  - core/calculation_engine.py (run_full_valuation)
  - Python 3.10+, langchain, langgraph
  - A valid Google Gemini API key configured for LangChain (eg. GOOGLE_API_KEY env)
"""
from __future__ import annotations

import json
from typing import Dict, List, TypedDict, Optional

from langchain.chat_models import init_chat_model
from langgraph.graph import StateGraph, START, END

# If you prefer the prebuilt agent wrapper, you can still import it, but here we
# call the tool directly for determinism.
from .tools import property_valuation_tool
from dotenv import load_dotenv

load_dotenv()

# ------------------------------
# 1) Model & global config
# ------------------------------
llm = init_chat_model("google_genai:gemini-2.0-flash")

# Domain enumerations (kept here for prompting & validation)
VALID_CATEGORIES = [
    "Higher Villa",
    "Multi-Story Building",
    "Apartment / Condominium",
    "MPH & Factory Building",
    "Fuel Station",
    "Coffee Washing Site",
    "Green House",
]

VALID_GRADES = ["Excellent", "Good", "Average", "Economy", "Minimum"]
VALID_USE = ["Residential", "Commercial"]
VALID_PLOT_GRADES = ["1st", "2nd", "3rd", "4th"]
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

# ------------------------------
# 2) Conversation state
# ------------------------------
class ValuationState(TypedDict):
    messages: List[Dict]
    slots: Dict[str, object]
    asked: List[str]  # what we've already asked to avoid repetition


def initial_state() -> ValuationState:
    return {"messages": [], "slots": {}, "asked": []}


# Required slots for the base (single-building) MVP
REQUIRED_SLOTS_IN_ORDER = [
    "building_name",            # e.g., "Block A" (for reporting)
    "building_category",        # one of VALID_CATEGORIES
    "length_m",                 # float (m)
    "width_m",                  # float (m)
    "num_floors",               # int
    "has_basement",             # bool
    "confirmed_grade",          # one of VALID_GRADES
    "plot_area_sqm",            # float
    "prop_town",                # one of VALID_TOWN_CLASSES
    "gen_use",                  # one of VALID_USE
    "plot_grade",               # one of VALID_PLOT_GRADES
    # Optional financial factors (default 1.0)
    "mcf",                      # float (default 1.0)
    "pef",                      # float (default 1.0)
    # Optional extras
    "has_elevator",             # bool
    "elevator_stops",           # int (required if has_elevator True)
]

# For categories with special components, you can extend here (MVP keeps it simple)
CATEGORY_SPECIAL_SLOTS: Dict[str, List[str]] = {
    # "Fuel Station": ["pump_island", "ugt_30m3", "ugt_50m3", "steel_canopy_sqm"],
    # "Coffee Washing Site": ["cherry_hopper_sqm", "fermentation_tanks_sqm", ...],
    # "Green House": ["greenhouse_cover_sqm", "in_farm_road_km", ...],
}


# ------------------------------
# 3) Utility helpers
# ------------------------------

def _boolify(text: str) -> Optional[bool]:
    t = text.strip().lower()
    if t in {"yes", "y", "true", "t", "1"}: return True
    if t in {"no", "n", "false", "f", "0"}: return False
    return None


def missing_slots(slots: Dict[str, object]) -> List[str]:
    """Return the remaining required slots based on current slots and category rules."""
    needed = [s for s in REQUIRED_SLOTS_IN_ORDER if s not in slots]

    cat = slots.get("building_category")
    if isinstance(cat, str) and cat in CATEGORY_SPECIAL_SLOTS:
        for s in CATEGORY_SPECIAL_SLOTS[cat]:
            if s not in slots:
                needed.append(s)

    # If has_elevator is False, do not require elevator_stops
    if "has_elevator" in slots and slots.get("has_elevator") in (False, "False", 0):
        needed = [s for s in needed if s != "elevator_stops"]

    return needed


# ------------------------------
# 4) Nodes
# ------------------------------

def extract_info_node(state: ValuationState) -> ValuationState:
    """LLM-based extraction from the latest user message into structured slots.
    We ask the model to output strict JSON and then safely parse+merge.
    """
    if not state.get("messages"):
        return state

    # Get latest user message (assumes CLI pushes user messages)
    last = state["messages"][-1]
    if last.get("role") != "user":
        return state

    prompt = f"""
    You are an information extraction parser for a property valuation chatbot.
    Read the user message and extract ONLY the following fields. If a field is not present, output null.
    Return STRICT JSON with these keys exactly:
    {{
      "building_name": string or null,
      "building_category": one of {VALID_CATEGORIES} or null,
      "length_m": number or null,
      "width_m": number or null,
      "num_floors": integer or null,
      "has_basement": true/false or null,
      "confirmed_grade": one of {VALID_GRADES} or null,
      "plot_area_sqm": number or null,
      "prop_town": one of {VALID_TOWN_CLASSES} or null,
      "gen_use": one of {VALID_USE} or null,
      "plot_grade": one of {VALID_PLOT_GRADES} or null,
      "mcf": number or null,
      "pef": number or null,
      "has_elevator": true/false or null,
      "elevator_stops": integer or null
    }}
    User message: '{last.get("content", "")}'
    Output ONLY JSON. No commentary.
    """

    resp = llm.invoke([{"role": "system", "content": "Return JSON only."},
                       {"role": "user", "content": prompt}])

    text = resp.content.strip()
    # Guard for fenced code blocks
    if text.startswith("```)" ):
        # Very defensive clean-up
        text = text.strip().strip("`")
        if text.lower().startswith("json"):
            text = text[4:].strip()

    try:
        data = json.loads(text)
    except Exception:
        # If parsing fails, do nothing; ask a direct question next
        return state

    # Normalize some booleans
    if isinstance(data.get("has_basement"), str):
        b = _boolify(data["has_basement"]) ;
        if b is not None: data["has_basement"] = b
    if isinstance(data.get("has_elevator"), str):
        b = _boolify(data["has_elevator"]) ;
        if b is not None: data["has_elevator"] = b

    # Merge into state.slots (keep previous unless overridden by non-null)
    merged = dict(state.get("slots", {}))
    for k, v in data.items():
        if v is not None:
            merged[k] = v

    state["slots"] = merged
    return state


def ask_next_question_node(state: ValuationState) -> ValuationState:
    slots = state.get("slots", {})
    asked = set(state.get("asked", []))
    remaining = [s for s in missing_slots(slots) if s not in asked]

    if not remaining:
        return state

    s = remaining[0]

    # Map slot to options if predefined
    options_map = {
        "building_category": VALID_CATEGORIES,
        "confirmed_grade": VALID_GRADES,
        "gen_use": VALID_USE,
        "plot_grade": VALID_PLOT_GRADES,
        "prop_town": VALID_TOWN_CLASSES,
    }

    # Default prompt
    q: str

    if s in options_map:
        choices = options_map[s]
        q = f"Please select {s.replace('_', ' ')} from the following options:\n" + "\n".join(f"{i+1}. {c}" for i, c in enumerate(choices))
    elif s == "building_name":
        q = "What is the building name or identifier (e.g., 'Block A' or 'Villa 1')?"
    elif s == "length_m":
        q = "What is the building length in meters (per floor footprint)?"
    elif s == "width_m":
        q = "What is the building width in meters (per floor footprint)?"
    elif s == "num_floors":
        q = "How many floors does the building have (e.g., 1 for single storey, 3 for G+2)?"
    elif s == "has_basement":
        q = "Does the building have a basement? (yes/no)"
    elif s == "plot_area_sqm":
        q = "What is the plot area in square meters?"
    elif s == "mcf":
        q = "Market Condition Factor (MCF)? If unsure, reply 1.0"
    elif s == "pef":
        q = "Property Enhancement Factor (PEF)? If unsure, reply 1.0"
    elif s == "has_elevator":
        q = "Is there an elevator? (yes/no)"
    elif s == "elevator_stops":
        q = "How many stops does the elevator have? (e.g., 2, 4, 6)"
    else:
        q = f"Please provide the value for: {s}"

    # Append question
    state["messages"].append({"role": "assistant", "content": q})
    state.setdefault("asked", []).append(s)
    return state


def calculate_node(state: ValuationState) -> ValuationState:
    """Build the tool payload and call the valuation engine."""
    slots = state.get("slots", {})

    # Defaults for optional fields
    mcf = float(slots.get("mcf", 1.0) or 1.0)
    pef = float(slots.get("pef", 1.0) or 1.0)

    has_elevator = bool(slots.get("has_elevator", False))
    elevator_stops = int(slots.get("elevator_stops", 0) or 0)

    building = {
        "name": str(slots.get("building_name", "Building 1")),
        "category": str(slots.get("building_category")),
        "length": float(slots.get("length_m")),
        "width": float(slots.get("width_m")),
        "num_floors": int(slots.get("num_floors")),
        "has_basement": bool(slots.get("has_basement", False)),
        "is_under_construction": False,  # MVP; extend if you capture this slot
        "incomplete_components": [],
        "selected_materials": {},
        "confirmed_grade": str(slots.get("confirmed_grade")),
        "specialized_components": {},  # Extend for Fuel/Coffee/GreenHouse
    }

    property_details = {
        "plot_area": float(slots.get("plot_area_sqm")),
        "prop_town": str(slots.get("prop_town")),
        "gen_use": str(slots.get("gen_use")),
        "plot_grade": str(slots.get("plot_grade")),
    }

    special_items = {
        "has_elevator": has_elevator,
        "elevator_stops": elevator_stops,
    }

    other_costs = {
        # Keep zero by default; wire to new slots if needed
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

    # Call the tool directly for a deterministic execution
    result_text: str = property_valuation_tool.invoke(payload)

    state["messages"].append({"role": "assistant", "content": result_text})
    return state


# ------------------------------
# 5) Routing / edges
# ------------------------------

def should_calculate(state: ValuationState) -> str:
    """Conditional router: if all slots are filled -> CALC, else ASK again."""
    remaining = missing_slots(state.get("slots", {}))
    return "CALC" if not remaining else "ASK"


# ------------------------------
# 6) Build the graph
# ------------------------------

def build_graph():
    builder = StateGraph(ValuationState)

    # Nodes
    builder.add_node("extract_info", extract_info_node)
    builder.add_node("ask", ask_next_question_node)
    builder.add_node("calculate", calculate_node)

    # Edges
    builder.add_edge(START, "extract_info")
    builder.add_conditional_edges(
        "extract_info",
        should_calculate,
        {"ASK": "ask", "CALC": "calculate"},
    )
    builder.add_edge("ask", "extract_info")  # loop until complete
    builder.add_edge("calculate", END)

    return builder.compile()


graph = build_graph()


# ------------------------------
# 7) Simple CLI runner for local testing
# ------------------------------
if __name__ == "__main__":
    print("\n🏗️ Property Valuation Agent (LangGraph)\nType 'quit' to exit.\n")
    state: ValuationState = initial_state()

    while True:
        user = input("You: ")
        if user.strip().lower() in {"quit", "exit"}:
            print("👋 Goodbye!")
            break

        state["messages"].append({"role": "user", "content": user})
        state = graph.invoke(state)

        # Print the last assistant message
        print("Bot:", state["messages"][-1]["content"], "\n")
