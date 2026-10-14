"""Risk queue: the band mix, and every entity's score split into the four signals."""
from __future__ import annotations

import plotly.graph_objects as go
import streamlit as st

from mri.app import data, theme
from mri.app.theme import esc


def _cue(row) -> str:
    codes = str(row.get("top_reason_codes") or "")
    first = codes.split("|")[0].strip().replace("_", " ")
    return first or "no dominant cue"


def _donut(counts: dict, total: int) -> go.Figure:
    """Share of entities by band. Four slices, directly labelled, so colour is never the only cue."""
    order = [b for b in ("Critical", "High", "Medium", "Low") if counts.get(b)]
    fig = go.Figure(
        go.Pie(
            labels=[f"{theme.BAND_GLYPH[b]} {b}" for b in order],
            values=[counts[b] for b in order],
            marker=dict(colors=[theme.band_color(b) for b in order], line=dict(color="#FFFFFF", width=2)),
            hole=0.62,
            sort=False,
            direction="clockwise",
            textinfo="label+percent",
            textposition="outside",
            textfont=dict(family="IBM Plex Sans", size=13, color="#0F2A30"),
            hovertemplate="%{label}: %{value} entities (%{percent})<extra></extra>",
            automargin=True,
        )
    )
    fig.update_layout(
        showlegend=False,
        margin=dict(t=58, b=48, l=80, r=80),
        height=380,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        annotations=[
            dict(text=str(total), x=0.5, y=0.56, font=dict(family="IBM Plex Mono", size=30, color="#0F2A30"), showarrow=False),
            dict(text="entities", x=0.5, y=0.42, font=dict(family="IBM Plex Sans", size=12, color="#6E858D"), showarrow=False),
        ],
    )
    return fig


def render() -> None:
    df = data.load_fused()
    counts = {str(k): int(v) for k, v in df["fused_risk_band"].value_counts().to_dict().items()}
    n = len(df)
    sellers = int((df["entity_type"] == "seller").sum())
    customers = n - sellers

    st.html(
        '<div class="crumb">Risk intelligence &rsaquo; Risk queue</div>'
        '<div class="mri-h1">Risk queue</div>'
        f'<p class="mri-lede">{n} marketplace entities ({sellers} sellers, {customers} customers) scored by four independent '
        "signals and fused into one number. Every bar below is that number, cut into the four pieces that produced it. "
        "All data in this demo is synthetic.</p>"
    )

    tiles = "".join(
        f'<div class="kpi"><div class="v" style="color:{theme.band_ink(b)}">{counts.get(b, 0)}</div>'
        f'<div class="t">{theme.band_chip(b)}</div>'
        f'<div class="s">{counts.get(b, 0) / n * 100:.0f}% of queue</div></div>'
        for b in ("Critical", "High", "Medium", "Low")
    )
    st.html(f'<div class="kpis">{tiles}</div>')

    c1, c2 = st.columns([1, 1.25], gap="large")
    with c1:
        st.html('<div class="mri-h2" style="margin-top:1.4rem">Share of the queue</div>'
                '<p class="mri-note">Band thresholds are fixed at 0.30, 0.55 and 0.75.</p>')
        st.plotly_chart(_donut(counts, n), use_container_width=True, config={"displayModeBar": False})

    with c2:
        st.html('<div class="mri-h2" style="margin-top:1.4rem">What drives each band</div>'
                '<p class="mri-note">Mean contribution of each signal, per band. The bar length is the mean fused score.</p>')
        rows = []
        for b in ("Critical", "High", "Medium", "Low"):
            sub = df[df["fused_risk_band"].astype(str) == b]
            if sub.empty:
                continue
            contribs = {s["key"]: float(sub[f"c_{s['key']}"].mean()) for s in theme.SIGNALS}
            tip = ", ".join(f"{s['name']} {contribs[s['key']]:.3f}" for s in theme.SIGNALS)
            rows.append(
                '<div class="board-row" style="grid-template-columns:6rem 1fr 3.6rem;padding-left:0">'
                f'<div class="ent"><b>{esc(b)}</b><span>{len(sub)} entities</span></div>'
                f'<div>{theme.stacked_track(contribs, tip)}</div>'
                f'<div class="sc">{sub["fused_risk_score"].mean():.2f}</div></div>'
            )
        st.html(theme.signal_legend() + '<div class="board" style="border:0;background:transparent">' + "".join(rows) + "</div>")

    st.html('<div class="mri-h2">Case board</div>'
            '<p class="mri-note">Entities ranked by fused score. Each bar is the score itself, split by what each signal '
            "contributed. Dashed lines mark the Medium, High and Critical thresholds.</p>")
    with st.container(key="panel_board"):
        top_n = st.slider("Entities shown", min_value=10, max_value=60, value=25, step=5)
        _board(df, top_n)

    _tail(df)


def _board(df, top_n) -> None:
    rows = []
    for _, r in df.head(top_n).iterrows():
        contribs = {s["key"]: r[f"c_{s['key']}"] for s in theme.SIGNALS}
        tip = ", ".join(f"{s['name']} {contribs[s['key']]:.2f}" for s in theme.SIGNALS) + f" = {r['fused_risk_score']:.3f}"
        rows.append(
            '<div class="board-row">'
            f'<div class="rk">{int(r["rank"])}</div>'
            f'<div class="ent"><b>{esc(r["label_text"])}</b><span>{esc(_cue(r))}</span></div>'
            f'<div class="bd">{theme.band_chip(str(r["fused_risk_band"]))}</div>'
            f'<div class="br">{theme.stacked_track(contribs, tip)}</div>'
            f'<div class="sc">{r["fused_risk_score"]:.2f}</div></div>'
        )
    st.html(
        theme.signal_legend()
        + '<div class="board"><div class="board-head"><div>#</div><div>ENTITY</div><div class="h-band">BAND</div>'
        + f'<div class="h-bar">{theme.band_axis()}</div><div style="text-align:right">SCORE</div></div>'
        + "".join(rows) + "</div>"
    )


def _tail(df) -> None:
    st.html('<div class="mri-h2">Open a case</div>')
    c1, c2 = st.columns([3, 1], vertical_alignment="bottom")
    options = df["key"].tolist()
    labels = dict(zip(df["key"], df["label_text"]))
    ranks = dict(zip(df["key"], df["rank"]))
    choice = c1.selectbox("Entity", options, format_func=lambda k: f"{labels[k]} (rank {int(ranks[k])})",
                          label_visibility="collapsed")
    if c2.button("Open case record", use_container_width=True):
        st.session_state["case_key"] = choice
        st.switch_page(st.session_state["_pages"]["investigate"])

    st.html('<div class="mri-h2">How the score is built</div>'
            '<p class="mri-note">Each signal is rescaled to 0 to 1 across the population, multiplied by its weight, and added. '
            "Weights are set by hand in <code>mri.fusion.fuse</code>, not learned.</p>")
    cols = st.columns(4)
    for col, s in zip(cols, theme.SIGNALS):
        col.html(
            f'<div class="sig"><span class="dot" style="background:{s["color"]}"></span><b>{esc(s["name"])}</b>'
            f'<span class="w">{int(s["weight"] * 100)}%</span><p>{esc(s["blurb"])}</p></div>'
        )
