from __future__ import annotations

import json
from typing import Dict, List, TypedDict, Optional

from langchain.chat_models import init_chat_model
from langgraph.graph import StateGraph, START, END

from .tools import property_valuation_tool
from dotenv import load_dotenv

load_dotenv()

# ------------------------------
# 1) Model & global config
# ------------------------------
llm = init_chat_model("google_genai:gemini-2.0-flash")

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
    asked: List[str]

def initial_state() -> ValuationState:
    return {"messages": [], "slots": {}, "asked": []}

REQUIRED_SLOTS_IN_ORDER = [
    "building_name", "building_category", "length_m", "width_m",
    "num_floors", "has_basement", "confirmed_grade",
    "plot_area_sqm", "prop_town", "gen_use", "plot_grade",
    "mcf", "pef", "has_elevator", "elevator_stops",
]

CATEGORY_SPECIAL_SLOTS: Dict[str, List[str]] = {}

# ------------------------------
# 3) Utility helpers
# ------------------------------
def _boolify(text: str) -> Optional[bool]:
    t = text.strip().lower()
    if t in {"yes", "y", "true", "t", "1"}: return True
    if t in {"no", "n", "false", "f", "0"}: return False
    return None

def missing_slots(slots: Dict[str, object]) -> List[str]:
    needed = [s for s in REQUIRED_SLOTS_IN_ORDER if s not in slots]
    cat = slots.get("building_category")
    if isinstance(cat, str) and cat in CATEGORY_SPECIAL_SLOTS:
        for s in CATEGORY_SPECIAL_SLOTS[cat]:
            if s not in slots:
                needed.append(s)
    # skip elevator stops if no elevator
    if "has_elevator" in slots and slots.get("has_elevator") is False:
        needed = [s for s in needed if s != "elevator_stops"]
    return needed

# ------------------------------
# 4) Nodes
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

        # Normalize yes/no to boolean immediately
        if last_asked in {"has_basement", "has_elevator"}:
            b = _boolify(last.get("content", "").strip())
            if b is not None:
                state["slots"][last_asked] = b
                return state

        # Free-text slots (just save input)
        if last_asked not in {"building_category", "confirmed_grade", "gen_use", "plot_grade", "prop_town"}:
            state["slots"][last_asked] = last.get("content", "").strip()
            return state

    # Numbered choices
    expected = state["slots"].get("__expected_choices__")
    if expected:
        slot, choices = expected
        msg = last.get("content", "").strip()
        if msg.isdigit():
            idx = int(msg) - 1
            if 0 <= idx < len(choices):
                state["slots"][slot] = choices[idx]
                state["slots"].pop("__expected_choices__", None)
                return state

    # Fallback: try LLM extraction
    prompt = f"""
You are an information extraction parser for a property valuation chatbot.
Return STRICT JSON with the following keys. If not present, use null:
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
    try:
        resp = llm.invoke([{"role": "system", "content": "Return JSON only."},
                           {"role": "user", "content": prompt}])
        text = resp.content.strip()
        data = json.loads(text)
    except Exception:
        # fallback: save raw input to last asked slot
        if asked_slots:
            last_asked = asked_slots[-1]
            state["slots"][last_asked] = last.get("content", "").strip()
        return state

    # Normalize booleans
    for bool_key in ["has_basement", "has_elevator"]:
        if isinstance(data.get(bool_key), str):
            b = _boolify(data[bool_key])
            if b is not None:
                data[bool_key] = b

    # Merge into state
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
    options_map = {
        "building_category": VALID_CATEGORIES,
        "confirmed_grade": VALID_GRADES,
        "gen_use": VALID_USE,
        "plot_grade": VALID_PLOT_GRADES,
        "prop_town": VALID_TOWN_CLASSES,
    }
    if s in options_map:
        choices = options_map[s]
        q = f"Please select {s.replace('_', ' ')}:\n" + \
            "\n".join(f"{i+1}. {c}" for i, c in enumerate(choices)) + \
            "\n(Reply with the number)"
        state["slots"]["__expected_choices__"] = (s, choices)
    elif s == "building_name":
        q = "What is the building name or identifier (e.g., 'Block A')?"
    elif s == "length_m":
        q = "What is the building length in meters (per floor footprint)?"
    elif s == "width_m":
        q = "What is the building width in meters (per floor footprint)?"
    elif s == "num_floors":
        q = "How many floors does the building have?"
    elif s == "has_basement":
        q = "Does the building have a basement? (yes/no)"
    elif s == "plot_area_sqm":
        q = "What is the plot area in square meters?"
    elif s == "mcf":
        q = "Market Condition Factor (MCF)? Reply 1.0 if unsure."
    elif s == "pef":
        q = "Property Enhancement Factor (PEF)? Reply 1.0 if unsure."
    elif s == "has_elevator":
        q = "Is there an elevator? (yes/no)"
    elif s == "elevator_stops":
        q = "How many stops does the elevator have?"
    else:
        q = f"Please provide the value for: {s}"

    state["messages"].append({"role": "assistant", "content": q})
    state.setdefault("asked", []).append(s)
    return state

def calculate_node(state: ValuationState) -> ValuationState:
    slots = state.get("slots", {})
    mcf = float(slots.get("mcf", 1.0) or 1.0)
    pef = float(slots.get("pef", 1.0) or 1.0)
    has_elevator = bool(slots.get("has_elevator", False))
    elevator_stops = int(slots.get("elevator_stops") or 0)

    building = {
        "name": str(slots.get("building_name", "Building 1")),
        "category": str(slots.get("building_category")),
        "length": float(slots.get("length_m")),
        "width": float(slots.get("width_m")),
        "num_floors": int(slots.get("num_floors")),
        "has_basement": bool(slots.get("has_basement", False)),
        "is_under_construction": False,
        "incomplete_components": [],
        "selected_materials": {},
        "confirmed_grade": str(slots.get("confirmed_grade")),
        "specialized_components": {},
    }

    property_details = {
        "plot_area": float(slots.get("plot_area_sqm")),
        "prop_town": str(slots.get("prop_town")),
        "gen_use": str(slots.get("gen_use")),
        "plot_grade": str(slots.get("plot_grade")),
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

# ------------------------------
# 5) Routing / edges
# ------------------------------
def should_calculate(state: ValuationState) -> str:
    remaining = missing_slots(state.get("slots", {}))
    return "CALC" if not remaining else "ASK"

# ------------------------------
# 6) Build the graph
# ------------------------------
def build_graph():
    builder = StateGraph(ValuationState)
    builder.add_node("extract_info", extract_info_node)
    builder.add_node("ask", ask_next_question_node)
    builder.add_node("calculate", calculate_node)
    builder.add_edge(START, "extract_info")
    builder.add_conditional_edges("extract_info", should_calculate, {"ASK": "ask", "CALC": "calculate"})
    builder.add_edge("ask", "extract_info")
    builder.add_edge("calculate", END)
    builder.config = {"recursion_limit": 100}  # prevent recursion errors
    return builder

# ------------------------------
# 7) CLI runner
# ------------------------------
if __name__ == "__main__":
    print("\n🏗️ Property Valuation Agent (LangGraph)\nType 'quit' to exit.\n")
    state: ValuationState = initial_state()
    builder = build_graph()
    graph = builder.compile()

    # 🔹 Ask the first question manually
    state = ask_next_question_node(state)
    print("Bot:", state["messages"][-1]["content"], "\n")

    while True:
        user = input("You: ")
        if user.strip().lower() in {"quit", "exit"}:
            print("👋 Goodbye!")
            break

        # Append user message
        state["messages"].append({"role": "user", "content": user})

        # Extract info from user input
        state = extract_info_node(state)

        # Check remaining slots
        remaining = missing_slots(state.get("slots", {}))

        if remaining:
            # Ask next missing question
            state = ask_next_question_node(state)
        else:
            # All slots filled → calculate valuation
            state = calculate_node(state)
            # Reset asked list to prevent further questions
            state["asked"] = []

        print("Bot:", state["messages"][-1]["content"], "\n")
