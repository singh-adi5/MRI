"""About: what this is, how it works, and what it does not prove."""
from __future__ import annotations

import streamlit as st

from mri.app import theme

REPO = "https://github.com/singh-adi5/MRI"


def render() -> None:
    st.html(
        '<div class="crumb">Risk intelligence &rsaquo; About</div><div class="mri-h1">Marketplace Risk Intelligence</div>'
        '<p class="mri-lede">A fraud and risk pipeline for online marketplaces. It scores sellers and customers with four independent signals, '
        "fuses them into one explainable score, and gives an analyst the evidence behind each case.</p>"
        + theme.signal_legend()
        + '<div class="mri-prose">'
        "<h3>How a score is made</h3>"
        "<p>A rule engine flags obvious abuse and produces weak labels. Those labels train a Random Forest, and an Isolation Forest scores how unusual each entity is. "
        "A graph of shared devices, regions and transactions adds network risk, and time-window features add burst risk. "
        "Each signal is rescaled to 0 to 1 and combined with fixed weights of 40, 25, 20 and 15 percent. "
        "Bands start at 0.30 (Medium), 0.55 (High) and 0.75 (Critical).</p>"
        "<h3>What this demo does not prove</h3>"
        "<ul>"
        "<li><b>The data is synthetic.</b> It was generated to exercise the pipeline, so no result here says anything about real marketplace fraud.</li>"
        "<li><b>The labels come from the rules.</b> Perfect model metrics therefore measure agreement with the rules, not detection of unseen fraud.</li>"
        "<li><b>Device signals are not selective.</b> The generated data uses only five device IDs shared by well over a hundred accounts each, "
        "so almost every entity ends up in one graph cluster.</li>"
        "<li><b>The anomaly signal counts twice.</b> It forms 30% of the ML risk and a further 15% of the fused score.</li>"
        "<li><b>Weights are hand-set.</b> They are not learned or validated against outcomes.</li>"
        "<li><b>Scores shift slightly between builds.</b> The models do not train to identical results on every run.</li>"
        "</ul>"
        "<h3>Where to go next</h3>"
        f'<p>Source code, tests and the data dictionary are in the <a href="{REPO}" target="_blank" rel="noopener">GitHub repository</a>. '
        "The pipeline runs from raw CSVs with one command, and the continuous-integration workflow runs the tests plus Bandit and pip-audit security scans.</p>"
        "</div>"
    )
