"""Model lab: what the models rely on, how far to trust them, and a live what-if scorer."""
from __future__ import annotations

import plotly.graph_objects as go
import streamlit as st

from mri.app import data, theme
from mri.app.theme import esc


def _delta(new: float, old: float) -> str:
    d = new - old
    if abs(d) < 0.0005:
        return '<span style="color:var(--ink3)">no change</span>'
    cls = "up" if d > 0 else "down"
    return f'<span class="{cls} num">{d:+.3f}</span>'


def render() -> None:
    df = data.load_fused()
    feats = data.load_features()
    imp = data.load_importance()
    rep = data.load_report()
    keyed = df.set_index("key")

    st.html('<div class="mri-h1">What the model relies on, and how far to trust it</div>'
            '<p class="mri-lede">The scorer below runs the trained Random Forest and Isolation Forest on any feature values you set, '
            "so you can see which inputs move a score.</p>")

    # ---- what-if scorer
    st.html('<div class="mri-h2" style="margin-top:.4rem">What-if scorer</div>')
    # Open on a case whose score responds to the sliders: the supervised model is saturated at ~1.0 for most
    # top-ranked entities, so those show no change. Pick the first High-band entity with headroom.
    room = df[(df["fused_risk_band"] == "High") & (df["supervised_score"] < 0.99) & (df["anomaly_score"] < 0.6)]
    fallback = (room.iloc[0] if len(room) else df.iloc[0])["key"]
    default_key = st.session_state.get("case_key", fallback)
    keys = df["key"].tolist()
    key = st.selectbox("Entity", keys, index=keys.index(default_key) if default_key in keys else 0,
                       format_func=lambda k: keyed.loc[k, "label_text"])
    row, frow = keyed.loc[key], feats.loc[key]
    etype = str(row["entity_type"])
    pop = feats[feats["entity_type"] == etype]
    movable = [f for f in imp["feature"] if f in pop.columns and pop[f].std() > 0][:6]

    values = {c: float(frow[c]) for c in data.load_models()["cols"]}
    st.html(f'<p class="mri-note">Move the {len(movable)} inputs the model relies on most for {esc(etype)}s. Other inputs stay as recorded.</p>')
    cols = st.columns(3)
    for i, f in enumerate(movable):
        lo, hi, cur = float(pop[f].min()), float(pop[f].max()), float(frow[f])
        is_int = bool((pop[f].dropna() % 1 == 0).all())
        step = 1.0 if is_int else max((hi - lo) / 100, 1e-6)
        values[f] = cols[i % 3].slider(f.replace("_", " "), min_value=lo, max_value=max(hi, lo + step), value=min(max(cur, lo), max(hi, lo + step)),
                                       step=step, key=f"wi_{key}_{f}")

    base = data.score_vector({c: float(frow[c]) for c in values})
    new = data.score_vector(values)
    g, t = float(row["graph_score_norm"]), float(row["temporal_score_norm"])
    fused_base = data.fused_from_ml(base["ml"], base["anomaly"], g, t, df)
    fused_new = data.fused_from_ml(new["ml"], new["anomaly"], g, t, df)
    lines = [
        ("Supervised probability", base["supervised"], new["supervised"]),
        ("Anomaly score", base["anomaly"], new["anomaly"]),
        ("ML risk (70% supervised, 30% anomaly)", base["ml"], new["ml"]),
        ("Fused score", fused_base, fused_new),
    ]
    body = "".join(
        f'<tr><td>{esc(n)}</td><td class="num">{a:.3f}</td><td class="num">{b:.3f}</td><td>{_delta(b, a)}</td></tr>' for n, a, b in lines
    )
    st.html(
        '<table class="cmp"><thead><tr><th>Measure</th><th>As recorded</th><th>What-if</th><th>Change</th></tr></thead>'
        f"<tbody>{body}</tbody></table>"
        f'<p class="mri-note" style="margin-top:.6rem">Band: {theme.band_chip(data.band_for(fused_base))} becomes {theme.band_chip(data.band_for(fused_new))}. '
        "Graph and temporal signals stay at their pipeline values, so only the ML and anomaly parts move.</p>"
    )

    # ---- importance
    top = imp.head(10).iloc[::-1]
    fig = go.Figure(go.Bar(x=top["importance"], y=top["feature"].str.replace("_", " "), orientation="h",
                           marker_color="#2F5BEA", text=top["importance"].map("{:.3f}".format), textposition="outside",
                           hovertemplate="%{y}: %{x:.3f}<extra></extra>"))
    fig.update_layout(height=360, margin=dict(l=0, r=40, t=10, b=0), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      font=dict(family="Instrument Sans, system-ui, sans-serif", size=13, color="#101828"),
                      xaxis=dict(showgrid=False, visible=False), yaxis=dict(showgrid=False, ticks=""), bargap=0.45)
    st.html('<div class="mri-h2">Inputs the Random Forest relies on</div>'
            '<p class="mri-note">Share of the forest\'s splits attributed to each feature, top 10 of '
            f'{int(rep["feature_count"])}.</p>')
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    # ---- trust
    ld = rep["label_distribution"]
    st.html(
        '<div class="mri-h2">How far to trust it</div>'
        f'<p class="mri-note">Trained on {int(rep["dataset_rows"])} entities and {int(rep["feature_count"])} features. '
        f'Labels: {int(ld["1"])} flagged by the V1 rules, {int(ld["0"])} not flagged.</p>'
        '<div class="callout"><b>Precision, recall, F1 and ROC AUC all read '
        f'{rep["precision"]:.2f}.</b> {esc(rep["warning"])} '
        "A perfect score against rule-derived labels shows the model reproduces those rules. It does not show that it finds fraud the rules miss.</div>"
    )
