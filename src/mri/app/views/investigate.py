"""Investigate: pick a case, see why it scored, and record an analyst decision."""
from __future__ import annotations

import json
import re
from datetime import datetime, timezone

import streamlit as st

from mri.app import data, theme
from mri.app.theme import esc

PROFILE_FEATURES = 6
_SAFE_NAME = re.compile(r"[^A-Za-z0-9_-]")


def _safe_name(value: str) -> str:
    """Keep a download filename to a known-safe character set (no path separators)."""
    return _SAFE_NAME.sub("_", str(value))[:64] or "entity"


def _percentile_rows(key: str, etype: str) -> str:
    feats = data.load_features()
    imp = data.load_importance()
    pop = feats[feats["entity_type"] == etype]
    me = feats.loc[key]
    rows = []
    for f in imp["feature"]:
        if f not in pop.columns or pop[f].std() == 0:
            continue
        v = float(me[f])
        pct = float((pop[f] <= v).mean() * 100)
        rows.append(
            f'<div class="pct"><div>{esc(f.replace("_", " "))}</div>'
            f'<div class="pt" title="{pct:.0f}th percentile among {esc(etype)}s"><i style="left:calc({pct:.1f}% - 1.5px)"></i></div>'
            f'<div class="pv num">{v:,.2f} (p{pct:.0f})</div></div>'
        )
        if len(rows) == PROFILE_FEATURES:
            break
    return "".join(rows)


def render() -> None:
    df = data.load_fused()

    st.html(
        '<div class="crumb">Risk intelligence &rsaquo; Case record</div>'
        '<div class="mri-h1">Case record</div>'
        '<p class="mri-lede">Filter the queue, open an entity, and read the evidence behind its score.</p>'
    )

    f1, f2, f3 = st.columns([1.3, 2.4, 1.3], vertical_alignment="bottom")
    etype = f1.segmented_control("Entity type", ["All", "Customer", "Seller"], default="All", key="f_type") or "All"
    bands = f2.pills("Risk band", data.BANDS, selection_mode="multi", default=data.BANDS, key="f_band") or data.BANDS
    q = f3.text_input("Find by ID", placeholder="e.g. 44")

    view = df[df["fused_risk_band"].astype(str).isin(bands)]
    if etype != "All":
        view = view[view["entity_type"].str.lower() == etype.lower()]
    if q.strip():
        view = view[view["entity_id"].str.contains(q.strip(), regex=False)]
    if view.empty:
        st.html('<p class="mri-note">No entities match these filters. Widen the risk band or clear the ID search.</p>')
        return

    table = view[["rank", "label_text", "fused_risk_band", "fused_risk_score", "c_ml", "c_graph", "c_temporal", "c_anomaly"]].copy()
    table["fused_risk_band"] = table["fused_risk_band"].astype(str)
    event = st.dataframe(
        table, hide_index=True, height=300, on_select="rerun", selection_mode="single-row", key="queue",
        column_config={
            "rank": st.column_config.NumberColumn("Rank", width="small"),
            "label_text": st.column_config.TextColumn("Entity"),
            "fused_risk_band": st.column_config.TextColumn("Band", width="small"),
            "fused_risk_score": st.column_config.ProgressColumn("Fused score", min_value=0, max_value=1, format="%.3f"),
            "c_ml": st.column_config.NumberColumn("ML", format="%.2f", width="small"),
            "c_graph": st.column_config.NumberColumn("Graph", format="%.2f", width="small"),
            "c_temporal": st.column_config.NumberColumn("Temporal", format="%.2f", width="small"),
            "c_anomaly": st.column_config.NumberColumn("Anomaly", format="%.2f", width="small"),
        },
    )
    selected = event.selection.rows if event and event.selection else []
    if selected:
        key = view.iloc[selected[0]]["key"]
        st.session_state["case_key"] = key
    else:
        key = st.session_state.get("case_key")
        if key not in set(view["key"]):
            key = view.iloc[0]["key"]
    r = df.set_index("key").loc[key]
    st.caption("Select a row to open its case file. The columns ML to Anomaly are each signal's contribution to the fused score.")

    # ---- case file
    st.html(
        f'<div class="mri-h2" style="margin-top:1.6rem">{esc(r["label_text"])} '
        f'<span style="vertical-align:middle;margin-left:.5rem">{theme.band_chip(str(r["fused_risk_band"]))}</span></div>'
        f'<p class="mri-note">Rank {int(r["rank"])} of {len(df)}. Fused score <b class="num">{r["fused_risk_score"]:.3f}</b>.</p>'
    )
    left, right = st.columns([1.35, 1], gap="large")

    with left:
        rows = []
        for s in theme.SIGNALS:
            norm, contrib = float(r[s["norm"]]), float(r[f"c_{s['key']}"])
            rows.append(
                '<div class="contrib">'
                f'<div class="nm"><b>{esc(s["name"])}</b><span>raw {float(r[s["raw"]]):.2f}</span></div>'
                f'<div class="track" title="{esc(s["name"])} scaled score {norm:.2f}">'
                f'<div class="seg" style="left:0;width:{norm*100:.1f}%;background:{s["color"]}"></div></div>'
                f'<div class="eq num">{norm:.2f} x {int(s["weight"]*100)}% = <b>{contrib:.3f}</b></div></div>'
            )
        rows.append(f'<div class="contrib total"><div>Fused score</div><div></div><div class="eq num">{r["fused_risk_score"]:.3f}</div></div>')
        st.html('<p class="mri-note" style="margin-bottom:.2rem">How the score adds up. The bar shows each signal scaled to 0 to 1.</p>' + "".join(rows))

        st.html(f'<div class="mri-h2" style="font-size:1.1rem;margin-top:1.4rem">Analyst summary</div><div class="quote">{esc(r["analyst_summary"])}</div>')
        codes = [c.strip().replace("_", " ") for c in str(r["top_reason_codes"]).split("|") if c.strip() and c.strip() != "nan"]
        if codes:
            st.html("".join(f'<span class="tag">{esc(c)}</span>' for c in codes))

    with right:
        st.html('<p class="mri-note" style="margin-bottom:.2rem">How unusual this entity is, ranked against '
                f'other {esc(str(r["entity_type"]).lower())}s on the features the model relies on most. The marker shows its percentile.</p>'
                + _percentile_rows(key, str(r["entity_type"])))
        st.html(
            '<div class="mri-h2" style="font-size:1.1rem;margin-top:1.4rem">Network evidence</div>'
            f'<dl class="dl"><dt>Connected accounts</dt><dd class="num">{r["graph_degree"]:.0f}</dd>'
            f'<dt>Cluster size</dt><dd class="num">{r["component_size"]:.0f}</dd>'
            f'<dt>Shared-device links</dt><dd class="num">{r["shared_device_links"]:.0f}</dd>'
            f'<dt>Transaction links</dt><dd class="num">{r["transaction_links"]:.0f}</dd></dl>'
            '<div class="mri-h2" style="font-size:1.1rem;margin-top:1.4rem">Timing evidence</div>'
            f'<dl class="dl"><dt>Order burst ratio</dt><dd class="num">{r["order_burst_ratio"]:.2f}</dd>'
            f'<dt>Order velocity per hour</dt><dd class="num">{r["order_velocity_per_hour"]:.2f}</dd>'
            f'<dt>Login velocity per hour</dt><dd class="num">{r["login_velocity_per_hour"]:.3f}</dd>'
            f'<dt>Failed-attempt density</dt><dd class="num">{r["failed_attempt_density"]:.2f}</dd></dl>'
        )

    # ---- analyst decision
    st.html('<div class="mri-h2">Record a decision</div>'
            '<p class="mri-note">Decisions are not stored on the server. Download the case note to keep it.</p>')
    a1, a2, a3 = st.columns(3)
    status = a1.selectbox("Case status", ["New", "In review", "Escalated", "Monitor", "Closed"])
    action = a2.selectbox("Recommended action", ["No action", "Monitor", "Request manual review", "Restrict", "Escalate"])
    priority = a3.selectbox("Priority", ["P3", "P2", "P1", "Critical"])
    notes = st.text_area("Analyst notes", value=f"Review started for {r['label_text']}.", height=110)
    payload = {
        "entity": r["label_text"], "fused_score": round(float(r["fused_risk_score"]), 3), "band": str(r["fused_risk_band"]),
        "status": status, "action": action, "priority": priority, "notes": notes,
        "recorded_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    st.download_button("Download case note", data=json.dumps(payload, indent=2), file_name=f"case_{_safe_name(key)}.json", mime="application/json")
