import sys
from pathlib import Path

# Add project root to sys.path for pytest
BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))
