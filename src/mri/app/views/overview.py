"""Overview: how the fused score splits across the four signals, entity by entity."""
from __future__ import annotations

import streamlit as st

from mri.app import data, theme
from mri.app.theme import esc


def _cue(row) -> str:
    codes = str(row.get("top_reason_codes") or "")
    first = codes.split("|")[0].strip().replace("_", " ")
    return first or "no dominant cue"


def render() -> None:
    df = data.load_fused()
    counts = df["fused_risk_band"].value_counts().to_dict()
    n = len(df)
    sellers = int((df["entity_type"] == "seller").sum())
    customers = n - sellers

    st.html(
        f'<div class="mri-h1">{counts.get("Critical", 0)} Critical and {counts.get("High", 0)} High-risk entities in the queue</div>'
        f'<p class="mri-lede">{n} entities scored ({sellers} sellers, {customers} customers). Every score is the sum of four signals, '
        "so each case shows which signal pushed it up. All data in this demo is synthetic.</p>"
    )
    tiles = []
    for b, cls in (("Critical", " crit"), ("High", ""), ("Medium", ""), ("Low", "")):
        c = int(counts.get(b, 0))
        tiles.append(
            f'<div class="kpi{cls}"><div class="v num">{c}</div>'
            f'<div class="l">{theme.band_chip(b)}</div>'
            f'<div class="s num">{c / n * 100:.0f}% of entities</div></div>'
        )
    st.html('<div class="kpis">' + "".join(tiles) + "</div>")
    st.html(theme.band_strip({str(k): int(v) for k, v in counts.items()}))

    st.html('<div class="mri-h2">Case board</div>'
            '<p class="mri-note">Entities ranked by fused score. Each bar is the score itself, split by what each signal contributed. '
            "Dashed lines mark the Medium, High and Critical thresholds.</p>")
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
            f'<div class="rk num">{int(r["rank"])}</div>'
            f'<div class="ent"><b>{esc(r["label_text"])}</b><span>{esc(_cue(r))}</span></div>'
            f'<div class="bd">{theme.band_chip(str(r["fused_risk_band"]))}</div>'
            f'<div>{theme.stacked_track(contribs, tip)}</div>'
            f'<div class="sc num">{r["fused_risk_score"]:.2f}</div></div>'
        )
    st.html(
        theme.signal_legend()
        + '<div class="board"><div class="board-head"><div>#</div><div>Entity</div><div class="h-band">Band</div>'
        + f'<div>{theme.band_axis()}</div><div style="text-align:right">Score</div></div>'
        + "".join(rows) + "</div>"
    )



def _tail(df) -> None:
    st.html('<div class="mri-h2">Open a case</div>')
    c1, c2 = st.columns([3, 1], vertical_alignment="bottom")
    options = df["key"].tolist()
    labels = dict(zip(df["key"], df["label_text"]))
    choice = c1.selectbox("Entity", options, format_func=lambda k: f"{labels[k]} (rank {int(df.set_index('key').loc[k, 'rank'])})",
                          label_visibility="collapsed")
    if c2.button("Open case file", use_container_width=True):
        st.session_state["case_key"] = choice
        st.switch_page(st.session_state["_pages"]["investigate"])

    st.html('<div class="mri-h2">How the score is built</div>'
            '<p class="mri-note">Each signal is rescaled to 0 to 1 across the population, multiplied by its weight, and added. '
            "Weights are set by hand in <code>mri.fusion.fuse</code>, not learned.</p>")
    cols = st.columns(4)
    for col, s_ in zip(cols, theme.SIGNALS):
        col.html(
            f'<div class="sig"><span class="dot" style="background:{s_["color"]}"></span><b>{esc(s_["name"])}</b>'
            f'<span class="w num">{int(s_["weight"] * 100)}%</span><p>{esc(s_["blurb"])}</p></div>'
        )
