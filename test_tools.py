# test_tools.py

import json
from core.tools import property_valuation_tool
from pydantic import ValidationError

def test_valuation_tool():
    """Simulates a call to the property_valuation_tool with sample data."""
    print("Running test for property_valuation_tool...")

    # Sample data that matches the Pydantic schemas
    sample_data = {
        "buildings": [
            {
                "name": "Main Villa",
                "category": "Higher Villa",
                "length": 20.0,
                "width": 15.0,
                "num_floors": 1,
                "has_basement": True,
                "selected_materials": {
                    "Foundation": "Concrete slab",
                    "Roofing": "Corrugated iron sheet",
                    "Metal Work": "Steel frames",
                    "Floor": "Ceramic tiles",
                    "Ceiling": "Gypsum board",
                    "Sanitary": "Standard fixtures"
                }
            }
        ],
        "property_details": {
            "plot_area": 500.0,
            "prop_town": "Addis Ababa",
            "gen_use": "Residential",
            "plot_grade": "Good"
        },
        "special_items": {
            "has_elevator": False,
            "elevator_stops": 0
        },
        "other_costs": {
            "fence_percent": 3.0,
            "septic_percent": 1.0,
            "external_works_percent": 5.0,
            "water_tank_cost": 25000,
            "consultancy_percent": 2.5
        },
        "financial_factors": {
            "mcf": 1.1,
            "pef": 1.05
        },
        "remarks": "This is a test valuation."
    }

    try:
        # Use .invoke() to pass the dictionary as a single argument
        report = property_valuation_tool.invoke(sample_data)
        
        # Print the formatted report.
        print("\n--- TEST SUCCESSFUL ---\n")
        print(report)
        print("\n-----------------------\n")
    except ValidationError as e:
        print("\n--- VALIDATION ERROR ---\n")
        print(e)
    except Exception as e:
        print("\n--- RUNTIME ERROR ---\n")
        print(f"An unexpected error occurred: {e}")

# Run the test
if __name__ == "__main__":
    test_valuation_tool()