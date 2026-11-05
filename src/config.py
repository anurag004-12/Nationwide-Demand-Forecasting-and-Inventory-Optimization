from pathlib import Path

# Folder paths
ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
OUTPUTS = ROOT / "outputs"
MODELS = ROOT / "models"

# Project settings
HORIZON = 28                   
STATE = "CA"
DEPARTMENT = "FOODS_3"
TOP_N_ITEMS = 50                  
QUANTILES = [0.05, 0.25, 0.5, 0.75, 0.95]