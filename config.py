import os

# Path to the real prediction data file, now in Parquet format (smaller
# and faster than the original CSV, and small enough to fit GitHub's
# 100MB file limit for deployment). Convert your CSV once with
# convert_to_parquet.py before deploying, or set this to a .csv path
# and switch data.py back to read_csv for local-only use.
DATA_PATH = os.environ.get(
    "FLIGHT_DATA_PATH",
    r"C:\FlightDashboard\results (1)\model_predictions.parquet",
)
METRICS_PATH = os.environ.get(
    "FLIGHT_METRICS_PATH",
    r"C:\FlightDashboard\results (1)\model_performance_summary.csv",
)

# Allowed frontend origins for CORS. Add your deployed Vercel URL here
# once you have it (e.g. "https://your-app.vercel.app"), via the
# FLIGHT_ALLOWED_ORIGINS env var, comma-separated.
ALLOWED_ORIGINS = os.environ.get(
    "FLIGHT_ALLOWED_ORIGINS",
    "http://localhost:5173,http://127.0.0.1:5173",
).split(",")

# Same severity thresholds used throughout the Streamlit dashboard, kept
# here so both projects agree on what counts as "good/amber/bad".
SEVERITY_LOW = 5
SEVERITY_HIGH = 10
DELAYED_THRESHOLD_MIN = 15

MONTH_NAMES = {i: n for i, n in enumerate(
    ["January", "February", "March", "April", "May", "June", "July",
     "August", "September", "October", "November", "December"], start=1)}
DAY_NAMES = {1: "Monday", 2: "Tuesday", 3: "Wednesday", 4: "Thursday",
             5: "Friday", 6: "Saturday", 7: "Sunday"}
CAUSE_COLS = ["carrier_delay", "weather_delay", "nas_delay",
              "security_delay", "late_aircraft_delay"]
CAUSE_LABELS = {
    "carrier_delay": "Carrier", "weather_delay": "Weather",
    "nas_delay": "National Air System", "security_delay": "Security",
    "late_aircraft_delay": "Late Aircraft",
}
CARRIER_NAMES = {
    "AA": "American Airlines", "DL": "Delta Air Lines", "UA": "United Airlines",
    "WN": "Southwest Airlines", "B6": "JetBlue Airways", "AS": "Alaska Airlines",
    "NK": "Spirit Airlines", "F9": "Frontier Airlines", "HA": "Hawaiian Airlines",
    "G4": "Allegiant Air", "MQ": "Envoy Air", "OO": "SkyWest Airlines",
    "YX": "Republic Airways", "OH": "PSA Airlines", "YV": "Mesa Airlines",
    "9E": "Endeavor Air", "EV": "ExpressJet", "VX": "Virgin America",
    "US": "US Airways", "CO": "Continental Airlines", "NW": "Northwest Airlines",
    "FL": "AirTran Airways", "TW": "Trans World Airlines",
}
