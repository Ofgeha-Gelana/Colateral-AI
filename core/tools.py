"""
Property valuation tools for the chatbot.
LangChain tool integration for the core valuation engine.
"""

from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field
from langchain_core.tools import tool
from core.calculation_engine import run_full_valuation


class BuildingDetails(BaseModel):
    """Schema for individual building details."""
    name: str = Field(description="Name or identifier for the building")
    category: str = Field(description="Building category (e.g., 'Higher Villa', 'Multi-Story Building', 'Apartment / Condominium', 'MPH & Factory Building', 'Fuel Station', 'Coffee Washing Site', 'Green House')")
    length: Optional[float] = Field(default=0, description="Building length in meters")
    width: Optional[float] = Field(default=0, description="Building width in meters")
    num_floors: Optional[int] = Field(default=1, description="Number of floors")
    has_basement: Optional[bool] = Field(default=False, description="Whether the building has a basement")
    is_under_construction: Optional[bool] = Field(default=False, description="Whether the building is under construction")
    incomplete_components: Optional[List[str]] = Field(default_factory=list, description="List of incomplete building components if under construction")
    selected_materials: Optional[Dict[str, str]] = Field(default_factory=dict, description="Selected materials for building components (Foundation, Roofing, Metal Work, Floor, Ceiling, Sanitary)")
    confirmed_grade: Optional[str] = Field(default=None, description="Confirmed building grade (Excellent, Good, Average, Economy, Minimum) - if not provided, will be auto-suggested from materials")
    specialized_components: Optional[Dict[str, float]] = Field(default_factory=dict, description="Specialized components for fuel stations, coffee sites, green houses, etc.")


class PropertyDetails(BaseModel):
    """Schema for property location and plot details."""
    plot_area: float = Field(description="Plot area in square meters")
    prop_town: str = Field(description="Property town/location category")
    gen_use: str = Field(description="General use type of the property")
    plot_grade: str = Field(description="Plot grade classification")


class SpecialItems(BaseModel):
    """Schema for special items like elevators."""
    has_elevator: Optional[bool] = Field(default=False, description="Whether the property has an elevator")
    elevator_stops: Optional[int] = Field(default=0, description="Number of elevator stops")


class OtherCosts(BaseModel):
    """Schema for other construction costs."""
    fence_percent: Optional[float] = Field(default=0, description="Fence cost as percentage of CCW (Current Cost of Work)")
    septic_percent: Optional[float] = Field(default=0, description="Septic system cost as percentage of CCW")
    external_works_percent: Optional[float] = Field(default=0, description="External works cost as percentage of CCW")
    water_tank_cost: Optional[float] = Field(default=0, description="Water tank cost in absolute ETB value")
    consultancy_percent: Optional[float] = Field(default=0, description="Consultancy fee as percentage of subtotal")


class FinancialFactors(BaseModel):
    """Schema for financial adjustment factors."""
    mcf: Optional[float] = Field(default=1.0, description="Market Condition Factor (typically 0.8-1.2)")
    pef: Optional[float] = Field(default=1.0, description="Property Enhancement Factor (typically 0.9-1.1)")


class PropertyValuationInput(BaseModel):
    """Complete schema for property valuation input."""
    buildings: List[BuildingDetails] = Field(description="List of buildings to be valued")
    property_details: PropertyDetails = Field(description="Property location and plot details")
    special_items: Optional[SpecialItems] = Field(default_factory=SpecialItems, description="Special items like elevators")
    other_costs: Optional[OtherCosts] = Field(default_factory=OtherCosts, description="Other construction costs")
    financial_factors: Optional[FinancialFactors] = Field(default_factory=FinancialFactors, description="Financial adjustment factors")
    remarks: Optional[str] = Field(default="", description="Additional remarks or notes")


@tool(args_schema=PropertyValuationInput)
def property_valuation_tool(
    buildings: List[BuildingDetails],
    property_details: PropertyDetails,
    special_items: Optional[SpecialItems] = None,
    other_costs: Optional[OtherCosts] = None,
    financial_factors: Optional[FinancialFactors] = None,
    remarks: Optional[str] = ""
) -> str:
    """
    Performs comprehensive property valuation based on building details, location, and other factors.
    
    Calculates market value and forced sale value considering:
    - Building construction costs based on materials, grade, and area
    - Location value based on town category and plot characteristics
    - Additional costs like fencing, septic systems, and consultancy fees
    - Financial factors like market conditions and property enhancements
    
    Returns detailed valuation report with cost breakdowns.
    """
    
    # Convert Pydantic models to dictionaries for the calculation engine
    valuation_data = {
        "buildings": [building.model_dump() for building in buildings],
        "property_details": property_details.model_dump(),
        "special_items": special_items.model_dump() if special_items else {},
        "other_costs": other_costs.model_dump() if other_costs else {},
        "financial_factors": financial_factors.model_dump() if financial_factors else {},
        "remarks": remarks or ""
    }
    
    # Run the valuation calculation
    try:
        result = run_full_valuation(valuation_data)
    except Exception as e:
        return f"Error in valuation calculation: {str(e)}"
    
    # Format the result as a human-readable string
    report = f"""
## Property Valuation Report

### Cost Breakdown:
- **Total Building Cost (CCW)**: ETB {result['total_building_cost']:,.2f}
- **Other Costs**: ETB {result['total_other_costs']:,.2f}
- **Location Value Applied**: ETB {result['final_applied_location_value']:,.2f}
  - Calculated Location Value: ETB {result['calculated_location_value']:,.2f}
  - Location Value Limit: ETB {result['location_value_limit']:,.2f}

### Final Valuation:
- **Estimated Market Value**: ETB {result['estimated_market_value']:,.2f}
- **Estimated Forced Sale Value**: ETB {result['estimated_forced_value']:,.2f}

### Building Grades:
"""
    
    # Add building grades to report
    for building, grade in result['suggested_grades'].items():
        report += f"- {building}: {grade}\n"
    
    # Add warnings if any
    if result['validation_warnings']:
        report += "\n### Warnings:\n"
        for warning in result['validation_warnings']:
            report += f"- {warning}\n"
    
    # Add remarks if provided
    if result['remarks']:
        report += f"\n### Remarks:\n{result['remarks']}\n"
    
    return report.strip()
