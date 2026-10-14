"""Design tokens, CSS and small HTML components for the workbench."""
from __future__ import annotations

import html

import streamlit as st

# Signal identity colours (validated: lightness band, chroma floor, CVD separation, contrast).
SIGNALS = [
    dict(key="ml", name="ML risk", weight=0.40, color="#2F5BEA", norm="ml_score_norm", raw="final_risk_score",
         blurb="Random Forest probability blended with the anomaly model. Learns from rule-labelled entities."),
    dict(key="graph", name="Graph", weight=0.25, color="#0B9A8F", norm="graph_score_norm", raw="graph_risk_score",
         blurb="Shared devices, shared regions and transaction links between accounts."),
    dict(key="temporal", name="Temporal", weight=0.20, color="#B7791F", norm="temporal_score_norm", raw="temporal_risk_score",
         blurb="Bursts of orders, refunds and logins inside short time windows."),
    dict(key="anomaly", name="Anomaly", weight=0.15, color="#B83280", norm="anomaly_score_norm", raw="anomaly_score",
         blurb="Isolation Forest: how unlike the rest of the population an entity behaves."),
]
WEIGHTS = {s["key"]: s["weight"] for s in SIGNALS}

BAND_GLYPH = {"Low": "○", "Medium": "◔", "High": "◑", "Critical": "●"}

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:opsz,wght@12..96,500;12..96,700&family=Instrument+Sans:wght@400;500;600&display=swap');
:root{
  --canvas:#F2F4F7; --ink:#101828; --ink2:#475467; --ink3:#5B6678; --rule:#D5DBE5; --navy:#1D2A44;
  --ml:#2F5BEA; --graph:#0B9A8F; --temporal:#B7791F; --anomaly:#B83280;
  --b-low:#C9D1DD; --b-medium:#8D9BB3; --b-high:#44567A; --b-critical:#B42318;
  --display:'Bricolage Grotesque','Instrument Sans',system-ui,sans-serif;
  --body:'Instrument Sans',system-ui,-apple-system,'Segoe UI',sans-serif;
}
html, body, .stApp, [data-testid="stAppViewContainer"]{background:var(--canvas); color:var(--ink); font-family:var(--body);}
[data-testid="stHeader"]{background:var(--canvas);}
.block-container, [data-testid="stMainBlockContainer"]{max-width:1180px; padding-top:4.8rem !important; padding-bottom:4rem;}
footer{visibility:hidden;}
h1,h2,h3,h4{font-family:var(--display); color:var(--ink); letter-spacing:-0.012em;}
p,li,label,span,div{font-family:var(--body);}
.num{font-variant-numeric:tabular-nums;}

/* headings */
.mri-h1{font-family:var(--display); font-weight:700; font-size:2.55rem; line-height:1.08; margin:.2rem 0 .5rem; max-width:22ch;}
.mri-h2{font-family:var(--display); font-weight:700; font-size:1.4rem; line-height:1.2; margin:2.4rem 0 .35rem;}
.mri-lede{color:var(--ink2); font-size:1.05rem; line-height:1.55; max-width:62ch; margin:0 0 1.4rem;}
.mri-note{color:var(--ink3); font-size:.85rem; line-height:1.5; max-width:70ch;}
.mri-rule{border:0; border-top:1px solid var(--rule); margin:1.6rem 0;}
.mri-prose{max-width:68ch; line-height:1.65; color:var(--ink); font-size:1rem;}
.mri-prose p{margin:0 0 .9rem;}
.mri-prose h3{font-size:1.15rem; margin:1.6rem 0 .4rem;}
.mri-prose ul{margin:.2rem 0 1rem 1.1rem; padding:0;} .mri-prose li{margin:.25rem 0;}
.mri-prose a{color:var(--ml);}

/* band chip: label + glyph + colour, never colour alone */
.chip{display:inline-flex; align-items:center; gap:.4rem; font-size:.8rem; font-weight:600; padding:.12rem .55rem .14rem; border-radius:999px; border:1.5px solid currentColor; white-space:nowrap;}
.chip-Low{color:#475467;} .chip-Medium{color:#3B4A63;} .chip-High{color:#2C3C5E; background:#E3E8F1;} .chip-Critical{color:#fff; background:var(--b-critical); border-color:var(--b-critical);}
.tag{display:inline-block; font-size:.78rem; padding:.1rem .5rem; border:1px solid var(--rule); border-radius:4px; color:var(--ink2); background:#fff; margin:0 .3rem .3rem 0;}

/* band strip */
.strip{display:flex; gap:2px; height:2.1rem; margin:.5rem 0 .3rem;}
.strip > div{min-width:3px;}
.strip-labels{display:flex; gap:2px; font-size:.82rem; color:var(--ink2);}
.strip-labels > div{overflow:visible; white-space:nowrap;}
.strip-labels b{color:var(--ink); font-weight:600;}

/* signal legend */
.legend{display:flex; flex-wrap:wrap; gap:.4rem 1.4rem; font-size:.88rem; color:var(--ink2); margin:.2rem 0 .6rem;}
.legend i{display:inline-block; width:.8rem; height:.8rem; border-radius:2px; margin-right:.4rem; vertical-align:-1px;}

/* case board */
.board{margin-top:.4rem;}
.board-head, .board-row{display:grid; grid-template-columns:2rem 11.5rem 6.6rem 1fr 3.4rem; gap:0 .9rem; align-items:center;}
.board-head{font-size:.78rem; color:var(--ink3); padding:0 0 .35rem; border-bottom:1px solid var(--rule);}
.board-row{padding:.5rem 0; border-bottom:1px solid var(--rule); font-size:.92rem;}
.board-row:hover{background:#E9EDF3;}
.board-row .rk{color:var(--ink3);} .board-row .ent b{font-weight:600;} .board-row .ent span{display:block; color:var(--ink3); font-size:.78rem; margin-top:.05rem;}
.board-row .sc{text-align:right; font-weight:600;}
.axis{position:relative; height:1rem;} .axis span{position:absolute; transform:translateX(-50%); font-size:.72rem; color:var(--ink3); top:0;}
.track{position:relative; height:.85rem; background:#E6EAF0; border-radius:2px;}
.track .seg{position:absolute; top:0; height:100%; border-right:2px solid var(--canvas);}
.track .tick{position:absolute; top:-3px; bottom:-3px; width:0; border-left:1px dashed #98A2B3;}
@media (max-width: 760px){ .board-head, .board-row{grid-template-columns:1.6rem 8rem 1fr 3rem;} .board-head .h-band, .board-row .bd{display:none;} }

/* contribution rows in the case file */
.contrib{display:grid; grid-template-columns:9.5rem 1fr 11.5rem; gap:.2rem .9rem; align-items:center; padding:.55rem 0; border-bottom:1px solid var(--rule); font-size:.92rem;}
.contrib .nm b{font-weight:600;} .contrib .nm span{display:block; color:var(--ink3); font-size:.78rem;}
.contrib .eq{color:var(--ink2); text-align:right;}
.contrib .track{height:.7rem;}
.contrib.total{border-bottom:0; font-weight:600;}

/* percentile rows */
.pct{display:grid; grid-template-columns:9rem 1fr 8.2rem; gap:.2rem .9rem; align-items:center; padding:.4rem 0; font-size:.9rem; border-bottom:1px solid var(--rule);}
.pct .pt{position:relative; height:.7rem; background:#E6EAF0; border-radius:2px;}
.pct .pt i{position:absolute; top:-3px; width:3px; height:calc(100% + 6px); background:var(--navy); border-radius:1px;}
.pct .pv{text-align:right; color:var(--ink2);}

/* definition grid */
.dl{display:grid; grid-template-columns:1fr auto; gap:.3rem 1rem; font-size:.92rem; max-width:26rem;}
.dl dt{color:var(--ink2);} .dl dd{margin:0; text-align:right; font-weight:600;}

/* quote */
.quote{border-left:3px solid var(--navy); padding:.2rem 0 .2rem 1rem; color:var(--ink); max-width:64ch; line-height:1.6; margin:.4rem 0 1rem;}

/* comparison table */
.cmp{width:100%; border-collapse:collapse; font-size:.92rem; max-width:44rem;}
.cmp th{font-weight:500; color:var(--ink3); font-size:.8rem; text-align:right; padding:.3rem .6rem; border-bottom:1px solid var(--rule);} .cmp th:first-child{text-align:left; padding-left:0;}
.cmp td{padding:.5rem .6rem; text-align:right; border-bottom:1px solid var(--rule);} .cmp td:first-child{text-align:left; padding-left:0; color:var(--ink2);}
.up{color:var(--b-critical); font-weight:600;} .down{color:#067647; font-weight:600;}

/* callout */
.callout{border:1px solid var(--rule); border-left:4px solid var(--temporal); background:#fff; padding:.8rem 1rem; max-width:70ch; font-size:.92rem; line-height:1.55; margin:.6rem 0 1rem;}

/* streamlit widget tuning */
[data-testid="stDataFrame"]{border:1px solid var(--rule); border-radius:4px;}
.stButton>button, .stDownloadButton>button{border-radius:4px; border:1.5px solid var(--navy); color:var(--navy); background:transparent; font-weight:600;}
.stButton>button:hover, .stDownloadButton>button:hover{background:var(--navy); color:#fff; border-color:var(--navy);}
.stButton>button:focus-visible, .stDownloadButton>button:focus-visible, button:focus-visible{outline:3px solid var(--ml); outline-offset:2px;}
@media (prefers-reduced-motion: reduce){ *{transition:none !important; animation:none !important;} }
</style>
"""


def inject_css() -> None:
    st.markdown(CSS, unsafe_allow_html=True)


def esc(x) -> str:
    return html.escape(str(x), quote=True)


def band_chip(band: str) -> str:
    return f'<span class="chip chip-{esc(band)}">{BAND_GLYPH.get(str(band), "")} {esc(band)}</span>'


def band_color(band: str) -> str:
    return {"Low": "#C9D1DD", "Medium": "#8D9BB3", "High": "#44567A", "Critical": "#B42318"}[str(band)]


def signal_legend(show_weights: bool = True) -> str:
    items = []
    for s in SIGNALS:
        suffix = " " + str(int(round(s["weight"] * 100))) + "%" if show_weights else ""
        items.append('<span><i style="background:' + s["color"] + '"></i>' + esc(s["name"]) + suffix + "</span>")
    return '<div class="legend" role="list">' + "".join(items) + "</div>"


def stacked_track(contribs: dict[str, float], title: str = "") -> str:
    """One bar on a 0-1 fused scale; segment length is weight x normalised signal."""
    left = 0.0
    segs = []
    for s in SIGNALS:
        w = max(contribs.get(s["key"], 0.0), 0.0) * 100
        segs.append(f'<div class="seg" style="left:{left:.3f}%;width:{w:.3f}%;background:{s["color"]}"></div>')
        left += w
    ticks = "".join(f'<div class="tick" style="left:{t*100:.0f}%"></div>' for t in (0.30, 0.55, 0.75))
    return f'<div class="track" title="{esc(title)}">{ticks}{"".join(segs)}</div>'


def band_axis() -> str:
    marks = {"Medium": 0.30, "High": 0.55, "Critical": 0.75}
    return '<div class="axis">' + "".join(f'<span style="left:{v*100:.0f}%">{k}</span>' for k, v in marks.items()) + "</div>"


def band_strip(counts: dict[str, int]) -> str:
    total = max(sum(counts.values()), 1)
    order = ["Critical", "High", "Medium", "Low"]
    bars = "".join(
        f'<div title="{esc(b)}: {counts.get(b, 0)}" style="width:{counts.get(b, 0)/total*100:.3f}%;background:{band_color(b)}"></div>'
        for b in order if counts.get(b, 0)
    )
    labels = "".join(
        f'<div style="width:{counts.get(b, 0)/total*100:.3f}%"><b class="num">{counts.get(b, 0)}</b> {esc(b)}</div>'
        for b in order if counts.get(b, 0)
    )
    return f'<div class="strip" role="img" aria-label="Entities by risk band">{bars}</div><div class="strip-labels">{labels}</div>'
