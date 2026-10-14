"""Governance: pipeline health, coverage, drift and the model registry."""
from __future__ import annotations

import streamlit as st

from mri.app import data
from mri.app.theme import esc


def render() -> None:
    health = data.load_health()
    drift = data.load_drift()
    reg = data.load_registry()
    counts = health["entity_counts"]
    total = counts["features"]
    flagged = int(health["drift_summary"]["features_flagged_for_drift"])

    st.html(
        '<div class="mri-h1">Pipeline health and model governance</div>'
        f'<p class="mri-lede">Pipeline status: <b>{esc(health["pipeline_status"])}</b>. Last run {esc(health["generated_at_utc"][:19].replace("T", " "))} UTC. '
        f'{len(health["registered_models"])} model files are registered with hashes, and {flagged} of '
        f'{health["drift_summary"]["features_checked"]} features are flagged for drift.</p>'
    )

    st.html('<div class="mri-h2" style="margin-top:.6rem">Coverage by stage</div>'
            '<p class="mri-note">Entities that reached each stage. Entities missing from the graph and temporal stages are scored with zero for that signal.</p>')
    names = {"features": "Features", "v2_risk": "ML scoring", "graph": "Graph analysis", "temporal": "Temporal analysis", "fused_v4": "Fusion"}
    rows = []
    for k, label in names.items():
        c = counts[k]
        miss = total - c
        note = f'<span class="up">{miss} missing</span>' if miss else '<span style="color:var(--ink3)">complete</span>'
        rows.append(
            f'<div class="contrib" style="grid-template-columns:10rem 1fr 9rem"><div class="nm"><b>{label}</b></div>'
            f'<div class="track"><div class="seg" style="left:0;width:{c/total*100:.1f}%;background:#3A3A3C"></div></div>'
            f'<div class="eq num">{c} of {total}, {note}</div></div>'
        )
    st.html("".join(rows))

    st.html('<div class="mri-h2">Feature drift</div>')
    if health.get("baseline_created_now"):
        st.html('<div class="callout">The drift baseline was created during this pipeline run, so every shift reads 0% by construction. '
                "Drift only becomes meaningful when a later run compares new data against a stored baseline.</div>")
    show = drift.assign(shift=drift["mean_shift_pct"].abs()).sort_values("shift", ascending=False).head(10)
    st.dataframe(
        show[["feature_name", "baseline_mean", "current_mean", "mean_shift_pct", "drift_flag"]], hide_index=True, use_container_width=True,
        column_config={
            "feature_name": "Feature", "baseline_mean": st.column_config.NumberColumn("Baseline mean", format="%.3f"),
            "current_mean": st.column_config.NumberColumn("Current mean", format="%.3f"),
            "mean_shift_pct": st.column_config.NumberColumn("Mean shift %", format="%.2f"), "drift_flag": "Flagged",
        },
    )

    st.html('<div class="mri-h2">Model registry</div>'
            '<p class="mri-note">Each model file is registered with its size and SHA-256 hash, so a changed file is detectable.</p>')
    m = reg["model_registry"].copy()
    m["file"] = m["file_path"].map(lambda p: str(p).replace("\\", "/").split("/")[-1])
    m["size_kb"] = (m["file_size_bytes"] / 1024).round(1)
    m["sha256_short"] = m["sha256"].str[:16]
    st.dataframe(m[["model_name", "file", "size_kb", "sha256_short", "registered_ts"]], hide_index=True, use_container_width=True,
                 column_config={"model_name": "Model", "file": "File", "size_kb": "Size (KB)", "sha256_short": "SHA-256 (first 16)",
                                "registered_ts": "Registered (UTC)"})

    st.html('<div class="mri-h2">Run log</div>')
    runs = reg["pipeline_runs"][["run_ts", "status", "stage", "details"]]
    st.dataframe(runs, hide_index=True, use_container_width=True,
                 column_config={"run_ts": "Time (UTC)", "status": "Status", "stage": "Stage", "details": "Details"})
