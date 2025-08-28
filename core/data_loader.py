import pandas as pd
import json
from pathlib import Path

# Path to the data directory
DATA_PATH = Path(__file__).parent / ".." / "data"

# A helper function to load a JSON file
def load_json_data(file_name: str):
    """Loads and returns data from a JSON file in the data directory."""
    with open(DATA_PATH / file_name, 'r', encoding='utf-8') as f:
        return json.load(f)

# --- Master Data Functions (Updated) ---

def get_branches_data():
    """Loads branches data from branches.json."""
    return load_json_data('branches.json')

def get_building_rates_data():
    """Loads building rates data from building_rates.json."""
    return load_json_data('building_rates.json')

def get_component_percentages():
    """Loads and processes component percentages from component_percentages.json."""
    data = load_json_data('component_percentages.json')
    df = pd.DataFrame(data)
    df.set_index('Building_Component', inplace=True)
    return df

def get_all_location_data():
    """Loads location data from location_data.json and converts string keys back to tuples."""
    raw_data = load_json_data('location_data.json')
    processed_data = {}
    for location, types in raw_data.items():
        processed_data[location] = {}
        for prop_type, tiers in types.items():
            processed_data[location][prop_type] = {}
            for tier, rates in tiers.items():
                processed_data[location][prop_type][tier] = {}
                for area_range, value in rates.items():
                    # Convert "0-200" back to (0, 200) tuple
                    start, end = area_range.split('-')
                    if end == 'inf':
                        end = float('inf')
                    else:
                        end = int(end)
                    processed_data[location][prop_type][tier][(int(start), end)] = value
    return processed_data

def get_materials_by_category(category: str):
    """Loads material data from material_mappings.json based on category."""
    data = load_json_data('material_mappings.json')
    if category in ["Higher Villa", "Multi-Story Building", "Apartment / Condominium"]:
        return data.get("villa_and_multi_story_materials", {})
    elif category == "MPH & Factory Building":
        return data.get("mph_factory_materials", {})
    return {}

def get_mapping_by_category(category: str):
    """Loads material mapping data from material_mappings.json based on category."""
    data = load_json_data('material_mappings.json')
    if category in ["Higher Villa", "Multi-Story Building", "Apartment / Condominium"]:
        return data.get("villa_and_multi_story_mapping", {})
    elif category == "MPH & Factory Building":
        return data.get("mph_factory_mapping", {})
    return {}

def get_fuel_station_rates():
    """Loads fuel station rates from fuel_station_rates.json."""
    return load_json_data('fuel_station_rates.json')

def get_coffee_site_rates():
    """Loads coffee site rates from coffee_site_rates.json."""
    return load_json_data('coffee_site_rates.json')

def get_minimum_completion_stages():
    """Loads minimum completion stages from minimum_completion_stages.json."""
    return load_json_data('minimum_completion_stages.json')

def get_elevator_rates():
    """Loads elevator rates from elevator_rates.json and converts string keys back to tuples."""
    raw_data = load_json_data('elevator_rates.json')
    processed_data = {}
    for key, value in raw_data.items():
        capacity, stops = key.split('_')
        processed_data[(int(capacity), int(stops))] = value
    return processed_data

def get_green_house_rates():
    """Loads green house rates from green_house_rates.json."""
    return load_json_data('green_house_rates.json')

def get_mph_factory_rates():
    """Loads MPH & Factory building rates based on height from mph_factory_rates.json."""
    return load_json_data('mph_factory_rates.json')