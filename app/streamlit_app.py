"""
Entry point for hosting the dashboard, e.g. on Streamlit Community Cloud (main file:
app/streamlit_app.py). This folder has its own requirements.txt, so the host installs only what the
dashboard needs, not the monitor's PyTorch/OpenCV stack.

It runs dashboard.py in demo mode: the recorded sample logs in eval_results/, or a log the visitor
uploads. Nothing here opens a camera or runs a model.

    streamlit run app/streamlit_app.py      # try it locally
"""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("CROWD_DASHBOARD_DEMO", "1")
os.environ.setdefault("CROWD_LOG_REFRESH", "0")   # the samples don't change

import dashboard  # noqa: E402

dashboard.app()
