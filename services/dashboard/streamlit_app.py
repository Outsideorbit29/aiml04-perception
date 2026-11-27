"""Streamlit Community Cloud entrypoint: the analyst dashboard in demo mode (F7.3).

Main file path on share.streamlit.io: services/dashboard/streamlit_app.py. Streamlit Cloud
installs services/dashboard/requirements.txt (next to this file). Demo mode generates synthetic
data at startup (fraud.demo_data), so no IEEE-CIS data, Kafka, Redis or scorer is needed.
Locally:  streamlit run services/dashboard/streamlit_app.py
"""

import os
import runpy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:  # `import fraud` from the repo root
    sys.path.insert(0, str(ROOT))

os.environ.setdefault("DASHBOARD_DEMO", "1")
runpy.run_path(str(Path(__file__).with_name("app.py")), run_name="__main__")
