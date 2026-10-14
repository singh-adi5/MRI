"""Streamlit entry point. Run: streamlit run src/mri/app/workbench.py"""
import sys
from pathlib import Path

# Make `mri` importable when launched via `streamlit run` (e.g. on Streamlit Community Cloud).
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import streamlit as st  # noqa: E402

st.set_page_config(page_title="MRI: Marketplace Risk Intelligence", page_icon=":material/shield:", layout="wide")

from mri.app import data, theme  # noqa: E402
from mri.app.views import about, governance, investigate, model_lab, overview  # noqa: E402

theme.inject_css()
data.ensure_pipeline_outputs()

pages = {
    "overview": st.Page(overview.render, title="Overview", icon=":material/dashboard:", default=True),
    "investigate": st.Page(investigate.render, title="Investigate", icon=":material/search:", url_path="investigate"),
    "model": st.Page(model_lab.render, title="Model lab", icon=":material/science:", url_path="model"),
    "governance": st.Page(governance.render, title="Governance", icon=":material/verified_user:", url_path="governance"),
    "about": st.Page(about.render, title="About", icon=":material/info:", url_path="about"),
}
st.session_state["_pages"] = pages
st.navigation(list(pages.values()), position="top").run()
