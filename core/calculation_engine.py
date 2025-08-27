# core/calculation_engine.py

from core.data_loader import (
    get_building_rates_data, get_component_percentages,
    get_mapping_by_category, get_fuel_station_rates, get_coffee_site_rates,
    get_all_location_data, get_minimum_completion_stages, get_elevator_rates,
    get_green_house_rates
)

# Load all data at the module level
building_rates_data = get_building_rates_data()
component_percentages = get_component_percentages()
fuel_station_rates = get_fuel_station_rates()
coffee_site_rates = get_coffee_site_rates()
all_location_data = get_all_location_data()
minimum_completion_stages = get_minimum_completion_stages()
elevator_rates = get_elevator_rates()
green_house_rates = get_green_house_rates()


def get_building_grade_rate(building_type: str, grade: str) -> float:
    for item in building_rates_data:
        if item['Building Type'] == building_type:
            try:
                rate = (item[f'{grade}_Min'] + item[f'{grade}_Max']) / 2
                return rate
            except KeyError:
                return (item['Average_Min'] + item['Average_Max']) / 2
    return 0


def suggest_grade_from_materials(selected_materials: dict, category: str) -> str:
    quality_scores = {'Excellent': 4, 'Good': 3, 'Average': 2, 'Economy': 1, 'Minimum': 0}
    material_grade_mapping = get_mapping_by_category(category)
    total_score = 0
    count = 0
    for component, material in selected_materials.items():
        if component in material_grade_mapping:
            for material_substring, grade in material_grade_mapping[component].items():
                if material_substring in material:
                    total_score += quality_scores.get(grade, 2)
                    count += 1
                    break
    if count == 0: return "Average"
    avg_score = total_score / count
    if avg_score >= 3.5: return "Excellent"
    if avg_score >= 2.5: return "Good"
    if avg_score >= 1.5: return "Average"
    if avg_score >= 0.5: return "Economy"
    return "Minimum"


def calculate_under_construction_value(full_value: float, building_type: str, grade: str,
                                       incomplete_components: list) -> tuple[float, float]:
    total_deduction_percent = 0
    grade_map = {'Excellent': 'Best', 'Good': 'Best', 'Average': 'Avg', 'Economy': 'Poor', 'Minimum': 'Poor'}
    if "Single Story" in building_type:
        type_key = "Single_Storey"
    elif "G+1" in building_type or "G+2" in building_type:
        type_key = "G1_G2"
    elif "G+3" in building_type or "G+4" in building_type:
        type_key = "G3_G4"
    else:
        type_key = "G1_G2"
    column_key = f"{type_key}_{grade_map.get(grade, 'Avg')}"
    for component in incomplete_components:
        if component in component_percentages.index:
            try:
                deduction = component_percentages.loc[component, column_key]
                total_deduction_percent += deduction
            except KeyError:
                pass
    completed_percent = 1.0 - total_deduction_percent
    return full_value * completed_percent, completed_percent


def calculate_location_value(town_category: str, use_type: str, plot_grade: str, plot_area: float) -> float:
    town_data = all_location_data.get(town_category, {})
    use_type_data = town_data.get(use_type, {})
    grade_table = use_type_data.get(plot_grade, {})

    if not grade_table:
        return 3000 * plot_area

    for (min_area, max_area), rate in grade_table.items():
        if min_area <= plot_area <= max_area:
            return rate * plot_area

    return 0


def calculate_location_value_limit(ccw: float, plot_area: float) -> float:
    if ccw == 0: return 0
    if plot_area <= 2000:
        return 3.0 * ccw
    elif 2001 <= plot_area <= 10000:
        return (3.5 * ccw) - (ccw * plot_area / 4000)
    else:
        return 1.0 * ccw


def calculate_apartment_cost(base_cost: float, floor_number: int) -> float:
    if floor_number == 1:
        deduction = 0.025
    elif floor_number > 1:
        deduction = 0.025 - (0.015 * (floor_number - 1))
    else:
        deduction = 0
    final_deduction = max(-0.10, deduction)
    return base_cost * (1 - final_deduction)


def calculate_fuel_station_value(components: dict) -> float:
    total_value = 0
    total_value += components.get("site_preparation_area", 0) * fuel_station_rates["site_preparation"]
    total_value += components.get("forecourt_area", 0) * fuel_station_rates["reinforced_concrete_forecourt"]
    total_value += components.get("canopy_area", 0) * fuel_station_rates["steel_canopy"]
    total_value += components.get("num_pump_islands", 0) * fuel_station_rates["pump_island"]
    total_value += components.get("num_ugt_30m3", 0) * fuel_station_rates["ugt_30m3"]
    total_value += components.get("num_ugt_50m3", 0) * fuel_station_rates["ugt_50m3"]
    return total_value


def calculate_coffee_site_value(components: dict) -> float:
    total_value = 0
    total_value += components.get("cherry_hopper_area", 0) * coffee_site_rates["cherry_hopper"]
    total_value += components.get("fermentation_tanks_area", 0) * coffee_site_rates["fermentation_tanks"]
    total_value += components.get("washing_channels_length", 0) * coffee_site_rates["washing_channels"]
    total_value += components.get("coffee_drier_area", 0) * coffee_site_rates["coffee_drier"]
    return total_value


def calculate_green_house_value(components: dict) -> float:
    """Calculates value based on Green House components."""
    total_value = 0
    total_value += components.get("greenhouse_area", 0) * green_house_rates["greenhouse_cover"]
    total_value += components.get("in_farm_road_km", 0) * green_house_rates["in_farm_road"]
    total_value += components.get("borehole_depth", 0) * green_house_rates["borehole"]
    total_value += components.get("land_preparation_area", 0) * green_house_rates["land_preparation"]
    return total_value


def run_full_valuation(valuation_data: dict) -> dict:
    total_building_cost = 0
    all_suggested_grades = {}
    validation_warnings = []

    for i, building in enumerate(valuation_data.get('buildings', [])):
        category = building.get('category', 'Multi-Story Building')
        building_cost = 0

        specialized_components = building.get('specialized_components', {})
        
        # Calculate building area if not provided in specialized_components
        if "total_building_area" in specialized_components:
            total_building_area = specialized_components["total_building_area"]
        else:
            # Fallback to length * width if total_building_area not provided
            length = building.get('length', 0)
            width = building.get('width', 0)
            total_building_area = length * width

        if category in ["Higher Villa", "Multi-Story Building", "MPH & Factory Building", "Apartment / Condominium"]:
            num_floors = building.get('num_floors', 1)  # Default to 1 floor if not specified
            area = total_building_area

            if category == "Higher Villa":
                building_type_for_rate = "Single Story Building (higher Villa)"
                policy_check_type = "Higher Villa"
            elif 1 <= num_floors <= 3:
                building_type_for_rate = "G+1 and G+2"
                policy_check_type = "G+1-3"
            elif num_floors >= 4:
                building_type_for_rate = "G+3 and G+4"
                policy_check_type = "G+4 & above"
            else:
                building_type_for_rate = "Single Story Building (higher Villa)"
                policy_check_type = "Higher Villa"

            suggested_grade = suggest_grade_from_materials(building.get('selected_materials', {}), category)
            all_suggested_grades[f"Building {i + 1} ({building.get('name')})"] = suggested_grade
            grade = building.get('confirmed_grade') or suggested_grade

            rate = get_building_grade_rate(building_type_for_rate, grade)
            full_replacement_cost = area * rate * (num_floors + 1 if category != "Apartment / Condominium" else 1)

            if building.get('has_basement', False):
                full_replacement_cost *= 1.25

            if building.get('is_under_construction', False):
                building_cost, completed_percent = calculate_under_construction_value(
                    full_replacement_cost,
                    building_type_for_rate,
                    grade,
                    building.get('incomplete_components', [])
                )
                min_completion = minimum_completion_stages.get(policy_check_type, 0)
                if completed_percent < min_completion:
                    validation_warnings.append(
                        f"Warning: Building '{building.get('name')}' is only {completed_percent:.0%} complete, "
                        f"which is below the required minimum of {min_completion:.0%} for a loan."
                    )
            else:
                building_cost = full_replacement_cost

            if category == "Apartment / Condominium":
                building_cost = calculate_apartment_cost(building_cost, num_floors)

        elif category == "Fuel Station":
            building_cost = calculate_fuel_station_value(building.get('specialized_components', {}))

        elif category == "Coffee Washing Site":
            building_cost = calculate_coffee_site_value(building.get('specialized_components', {}))

        elif category == "Green House":
            building_cost = calculate_green_house_value(building.get('specialized_components', {}))

        total_building_cost += building_cost

    # The rest of the function remains the same
    ccw = total_building_cost

    special_items_cost = 0
    if valuation_data.get('special_items', {}).get('has_elevator', False):
        stops = valuation_data['special_items'].get('elevator_stops', 0)
        closest_key = min(elevator_rates.keys(), key=lambda k: abs(k[1] - stops))
        special_items_cost += elevator_rates.get(closest_key, 0)

    ccw += special_items_cost

    property_details = valuation_data.get('property_details', {})
    plot_area = property_details.get('plot_area', 0)

    if valuation_data['buildings'][0]['category'] == "Apartment / Condominium":
        grade = valuation_data['buildings'][0].get('confirmed_grade', 'Average')
        plot_area_factor = 0.8 if grade in ["Excellent", "Good"] else 0.4
        plot_area *= plot_area_factor

    calculated_lv = calculate_location_value(
        property_details.get('prop_town', ''),
        property_details.get('gen_use', ''),
        property_details.get('plot_grade', ''),
        plot_area
    )
    lv_limit = calculate_location_value_limit(ccw, plot_area)
    final_location_value = min(calculated_lv, lv_limit)

    if valuation_data['buildings'][0]['category'] == "Apartment / Condominium":
        total_other_costs = 0
    else:
        other_costs_details = valuation_data.get('other_costs', {})
        fence_cost = ccw * (other_costs_details.get('fence_percent', 0) / 100)
        septic_cost = ccw * (other_costs_details.get('septic_percent', 0) / 100)
        external_cost = ccw * (other_costs_details.get('external_works_percent', 0) / 100)
        water_tank_cost = other_costs_details.get('water_tank_cost', 0)
        total_other_costs = fence_cost + septic_cost + external_cost + water_tank_cost

    financial_factors = valuation_data.get('financial_factors', {})
    mcf = financial_factors.get('mcf', 1.0)
    pef = financial_factors.get('pef', 1.0)

    sub_total = ccw + final_location_value + total_other_costs
    consultancy_fee = sub_total * (valuation_data.get('other_costs', {}).get('consultancy_percent', 0) / 100)
    total_market_value = (sub_total + consultancy_fee) * mcf * pef
    forced_value = total_market_value * 0.8

    return {
        "total_building_cost": ccw,
        "total_other_costs": total_other_costs,
        "calculated_location_value": calculated_lv,
        "location_value_limit": lv_limit,
        "final_applied_location_value": final_location_value,
        "estimated_market_value": total_market_value,
        "estimated_forced_value": forced_value,
        "suggested_grades": all_suggested_grades,
        "validation_warnings": validation_warnings,
        "remarks": valuation_data.get('remarks', '')
    }