# Colateral AI - Property Valuation Assistant

Intelligent property valuation system for accurate market and forced sale value estimates.

## Features

- Multiple property types (Residential, Commercial, Special)
- Accurate market value and forced sale value (70% of market value)
- Location-based valuation
- Special property support (Fuel Stations, Coffee Washing Sites, Green Houses)

## Quick Start

1. Clone and install:
```bash
git clone https://github.com/Ofgeha-Gelana/Colateral-AI.git
cd Colateral-AI
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

2. Run the app:
```bash
streamlit run app.py
```

## Example Output
```
📌 PROPERTY DETAILS
Location: Finfinne Border A1
Category: Fuel Station
Property Use: Commercial
Plot Area: 2,500.00 sqm

🏠 VALUATION SUMMARY
Market Value: ETB 15,750,000
Forced Sale Value: ETB 11,025,000
```

## Project Structure
- `core/`: Main application logic
- `data/`: Property data and mappings
- `app.py`: Web interface

## License
MIT
