"""Design tokens, CSS and small HTML components for the workbench.

Every helper here escapes the values it interpolates (see `esc`); views must pass
user- or data-derived text through `esc` before putting it in markup.
"""
from __future__ import annotations

import html

import streamlit as st

# Signal identity colours. Validated for a light surface: OKLCH lightness band,
# chroma floor, colour-blind separation (deutan/protan) and 3:1 contrast.
SIGNALS = [
    dict(key="ml", name="ML risk", weight=0.40, color="#2F62D4", norm="ml_score_norm", raw="final_risk_score",
         blurb="Random Forest probability blended with the anomaly model. Learns from rule-labelled entities."),
    dict(key="graph", name="Graph", weight=0.25, color="#009384", norm="graph_score_norm", raw="graph_risk_score",
         blurb="Shared devices, shared regions and transaction links between accounts."),
    dict(key="temporal", name="Temporal", weight=0.20, color="#9E5A12", norm="temporal_score_norm", raw="temporal_risk_score",
         blurb="Bursts of orders, refunds and logins inside short time windows."),
    dict(key="anomaly", name="Anomaly", weight=0.15, color="#8E3AAE", norm="anomaly_score_norm", raw="anomaly_score",
         blurb="Isolation Forest: how unlike the rest of the population an entity behaves."),
]
WEIGHTS = {s["key"]: s["weight"] for s in SIGNALS}

# Band identity: a glyph as well as a colour, so the band never rests on colour alone.
BAND_GLYPH = {"Low": "○", "Medium": "●", "High": "▲", "Critical": "◆"}
BAND_STYLE = {
    "Low": {"bar": "#3F9A7E", "bg": "#E4F4EE", "ink": "#0B5B46"},
    "Medium": {"bar": "#B38A14", "bg": "#FBF3DC", "ink": "#6E5407"},
    "High": {"bar": "#C2620E", "bg": "#FDEEE0", "ink": "#8A4409"},
    "Critical": {"bar": "#C0272D", "bg": "#FBE9E9", "ink": "#9B1C21"},
}

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=IBM+Plex+Sans:wght@400;500;600;700&display=swap');
:root{
  --canvas:#EFF3F4; --surface:#FFFFFF; --rail:#06272F;
  --ink:#0F2A30; --ink2:#4C6670; --ink3:#6E858D;
  --hair:#D5DEE1; --soft:#E8EEEF; --brand:#00796B; --brand-dark:#005F54;
  --ml:#2F62D4; --graph:#009384; --temporal:#9E5A12; --anomaly:#8E3AAE;
  --b-low:#3F9A7E; --b-medium:#B38A14; --b-high:#C2620E; --b-critical:#C0272D;
  --font:'IBM Plex Sans',system-ui,-apple-system,'Segoe UI',sans-serif;
  --mono:'IBM Plex Mono',ui-monospace,'SFMono-Regular',monospace;
  --r-lg:10px; --r-md:8px; --r-sm:6px;
}
html, body, .stApp, [data-testid="stAppViewContainer"]{background:var(--canvas); color:var(--ink); font-family:var(--font);}
[data-testid="stHeader"]{background:transparent;}
.block-container, [data-testid="stMainBlockContainer"]{max-width:none !important; padding:2.6rem 2.4rem 4rem !important;}
footer{visibility:hidden;}
h1,h2,h3,h4{font-family:var(--font); color:var(--ink); letter-spacing:-.02em;}
p,li,label,span,div,button,input,select,textarea{font-family:var(--font);}
.num{font-family:var(--mono); font-variant-numeric:tabular-nums;}

/* ---- left rail (Streamlit sidebar) ---- */
[data-testid="stSidebar"]{background:var(--rail); border-right:0;}
[data-testid="stSidebar"] *{color:#A9C4C6;}
[data-testid="stSidebar"] [data-testid="stSidebarNav"] a span,
[data-testid="stSidebar"] [data-testid="stSidebarNav"] span{color:#A9C4C6 !important;}
[data-testid="stSidebarNav"] a{border-radius:var(--r-md); margin:2px 6px; padding:.45rem .7rem; border-left:3px solid transparent;}
[data-testid="stSidebarNav"] a:hover{background:rgba(255,255,255,.08);}
[data-testid="stSidebarNav"] a[aria-current="page"]{background:rgba(0,181,160,.16); border-left:3px solid #00B5A0;}
[data-testid="stSidebarNav"] a[aria-current="page"] span{color:#FFFFFF !important; font-weight:600;}
.rail-brand{display:flex; align-items:center; gap:.7rem; padding:.2rem .2rem 1rem;}
.rail-brand .mark{display:flex; gap:3px; align-items:flex-end; height:22px;}
.rail-brand .mark i{display:block; width:5px; border-radius:1px;}
.rail-brand b{font-family:var(--mono); font-size:.95rem; font-weight:600; letter-spacing:.14em; color:#FFFFFF !important;}
.rail-brand span{display:block; font-size:.7rem; color:#7FA4A6 !important; margin-top:2px;}
.rail-sec{font-family:var(--mono); font-size:.62rem; letter-spacing:.14em; color:#5E868A !important; padding:.6rem .2rem .3rem;}
.rail-card{background:rgba(255,255,255,.05); border-radius:var(--r-md); padding:.7rem .8rem; margin-top:.3rem;}
.rail-card .ok{display:flex; align-items:center; gap:.5rem; font-size:.8rem; color:#CFE2E1 !important;}
.rail-card .ok i{width:7px; height:7px; border-radius:50%; background:#35C77A; display:inline-block;}
.rail-card p{font-family:var(--mono); font-size:.68rem; color:#7FA4A6 !important; margin:.45rem 0 0; line-height:1.5;}
.rail-foot{font-size:.68rem; line-height:1.55; color:#5E868A !important; padding:1.2rem .2rem 0;}

/* ---- page furniture ---- */
.crumb{font-size:.78rem; color:var(--ink3); margin-bottom:.4rem;}
.mri-h1{font-size:1.7rem; font-weight:600; letter-spacing:-.02em; margin:0 0 .35rem;}
.mri-h2{font-size:1.05rem; font-weight:600; letter-spacing:-.02em; margin:2rem 0 .3rem;}
.mri-lede{color:var(--ink2); font-size:.92rem; line-height:1.6; max-width:78ch; margin:0 0 1rem;}
.mri-note{color:var(--ink3); font-size:.82rem; line-height:1.55; max-width:76ch;}
.mri-prose{max-width:70ch; line-height:1.65; color:var(--ink); font-size:.95rem;}
.mri-prose p{margin:0 0 .85rem;}
.mri-prose h3{font-size:1.05rem; font-weight:600; margin:1.6rem 0 .35rem;}
.mri-prose ul{margin:.2rem 0 1rem 1.1rem; padding:0;} .mri-prose li{margin:.3rem 0;}
.mri-prose a, .mri-note a{color:var(--brand);}

/* panel: st.container(key="panel_*") */
[class*="st-key-panel"]{background:var(--surface); border:1px solid var(--hair); border-radius:var(--r-lg); padding:1.3rem 1.5rem 1.1rem; margin:.3rem 0 .6rem;}

/* band chip: glyph + label + colour */
.chip{display:inline-flex; align-items:center; gap:.35rem; font-size:.72rem; font-weight:600; padding:.18rem .55rem; border-radius:4px; white-space:nowrap;}
.chip-Low{background:#E4F4EE; color:#0B5B46;}
.chip-Medium{background:#FBF3DC; color:#6E5407;}
.chip-High{background:#FDEEE0; color:#8A4409;}
.chip-Critical{background:#FBE9E9; color:#9B1C21;}
.tag{display:inline-block; font-size:.78rem; padding:.2rem .6rem; border-radius:999px; color:var(--ink2); background:var(--soft); margin:0 .35rem .35rem 0;}

/* KPI tiles */
.kpis{display:grid; grid-template-columns:repeat(4,minmax(0,1fr)); gap:.9rem; margin:.4rem 0 .2rem;}
.kpi{background:var(--surface); border:1px solid var(--hair); border-radius:var(--r-lg); padding:1rem 1.2rem .9rem;}
.kpi .v{font-family:var(--mono); font-size:1.9rem; font-weight:600; line-height:1.05;}
.kpi .s{color:var(--ink3); font-size:.74rem; margin-top:.5rem; font-family:var(--mono);}
.kpi .t{margin-top:.45rem;}
@media (max-width:900px){ .kpis{grid-template-columns:repeat(2,minmax(0,1fr));} }

/* band strip */
.strip{display:flex; gap:2px; height:.6rem; margin:.5rem 0 .35rem; border-radius:999px; overflow:hidden;}
.strip > div{min-width:3px;}
.strip-labels{display:flex; gap:2px; font-size:.76rem; color:var(--ink3);}
.strip-labels > div{overflow:visible; white-space:nowrap;}
.strip-labels b{color:var(--ink); font-weight:600; font-family:var(--mono);}

/* signal legend */
.legend{display:flex; flex-wrap:wrap; gap:.4rem 1.3rem; font-size:.8rem; color:var(--ink2); margin:.2rem 0 .6rem;}
.legend i{display:inline-block; width:.6rem; height:.6rem; border-radius:2px; margin-right:.4rem;}

/* case board */
.board{margin-top:.1rem; background:var(--surface); border:1px solid var(--hair); border-radius:var(--r-lg); overflow:hidden;}
.board-head, .board-row{display:grid; grid-template-columns:2.2rem 10.5rem 6rem 1fr 3.6rem; gap:0 1rem; align-items:center;}
.board-head{font-family:var(--mono); font-size:.66rem; letter-spacing:.08em; color:#5E757D; padding:.7rem 1.1rem; background:#F2F6F7; border-bottom:1px solid var(--hair);}
.board-row{padding:.72rem 1.1rem; border-bottom:1px solid #EDF1F2; font-size:.88rem; border-left:3px solid transparent;}
.board-row:last-child{border-bottom:0;}
.board-row:hover{background:#F2F8F7; border-left-color:var(--brand);}
.board-row .rk{font-family:var(--mono); font-size:.76rem; color:var(--ink3);}
.board-row .ent b{font-weight:600;}
.board-row .ent span{display:block; color:var(--ink3); font-size:.74rem; margin-top:.1rem;}
.board-row .sc{text-align:right; font-family:var(--mono); font-weight:600;}
.axis{position:relative; height:.9rem;} .axis span{position:absolute; transform:translateX(-50%); font-family:var(--mono); font-size:.66rem; color:var(--ink3); top:0;}
.track{position:relative; height:.65rem; background:var(--soft); border-radius:2px;}
.track .seg{position:absolute; top:0; height:100%; border-right:1px solid var(--surface);}
.track .tick{position:absolute; top:-3px; bottom:-3px; width:0; border-left:1px solid #B9C6CA;}
@media (max-width:900px){ .board-head, .board-row{grid-template-columns:1.8rem 1fr 3.4rem;} .board-head .h-band, .board-row .bd, .board-head .h-bar, .board-row .br{display:none;} }

/* contribution rows */
.contrib{display:grid; grid-template-columns:9rem 1fr 11rem; gap:.2rem .9rem; align-items:center; padding:.6rem 0; border-bottom:1px solid #EDF1F2; font-size:.88rem;}
.contrib .nm b{font-weight:600;} .contrib .nm span{display:block; color:var(--ink3); font-size:.74rem; font-family:var(--mono);}
.contrib .eq{color:var(--ink2); text-align:right; font-family:var(--mono);}
.contrib .track{height:.55rem;}
.contrib.total{border-bottom:0; font-weight:600;}

/* percentile rows */
.pct{display:grid; grid-template-columns:8.5rem 1fr 7.5rem; gap:.2rem .9rem; align-items:center; padding:.45rem 0; font-size:.85rem; border-bottom:1px solid #EDF1F2;}
.pct .pt{position:relative; height:.4rem; background:var(--soft); border-radius:999px;}
.pct .pt i{position:absolute; top:-5px; width:.9rem; height:.9rem; border-radius:50%; background:var(--brand); border:3px solid var(--surface); box-shadow:0 0 0 1px #B6C8CB;}
.pct .pv{text-align:right; color:var(--ink2); font-family:var(--mono); font-size:.78rem;}

/* definition grid */
.dl{display:grid; grid-template-columns:1fr auto; gap:.3rem 1rem; font-size:.88rem; max-width:26rem;}
.dl dt{color:var(--ink2);} .dl dd{margin:0; text-align:right; font-weight:600; font-family:var(--mono);}

/* quote + callout */
.quote{background:#F4F8F8; border-radius:var(--r-md); padding:.9rem 1.1rem; color:#2A4248; max-width:66ch; line-height:1.6; margin:.3rem 0 .9rem; font-size:.9rem;}
.callout{background:#FFF8E8; border:1px solid #EBDCB4; border-left:4px solid #B58100; border-radius:var(--r-md); padding:.9rem 1.1rem; max-width:76ch; font-size:.88rem; line-height:1.6; margin:.5rem 0 1rem; color:#3E4F45;}

/* comparison table */
.cmp{width:100%; border-collapse:collapse; font-size:.88rem; max-width:44rem;}
.cmp th{font-family:var(--mono); font-weight:500; color:#5E757D; font-size:.68rem; letter-spacing:.06em; text-align:right; padding:.3rem .6rem; border-bottom:1px solid var(--hair);} .cmp th:first-child{text-align:left; padding-left:0;}
.cmp td{padding:.55rem .6rem; text-align:right; border-bottom:1px solid #EDF1F2; font-family:var(--mono);} .cmp td:first-child{text-align:left; padding-left:0; color:var(--ink2); font-family:var(--font);}
.up{color:#9B1C21; font-weight:600;} .down{color:#0B5B46; font-weight:600;}

/* signal tiles */
.sig{background:var(--surface); border:1px solid var(--hair); border-radius:var(--r-lg); padding:.9rem 1.1rem; height:100%;}
.sig .dot{display:inline-block; width:.6rem; height:.6rem; border-radius:2px; margin-right:.45rem;}
.sig b{font-size:.92rem; font-weight:600;} .sig .w{color:var(--ink3); margin-left:.3rem; font-family:var(--mono); font-size:.8rem;}
.sig p{color:var(--ink3); font-size:.8rem; line-height:1.5; margin:.4rem 0 0;}

/* widgets */
[data-testid="stDataFrame"]{border:1px solid var(--hair); border-radius:var(--r-md); overflow:hidden;}
.stButton>button, .stDownloadButton>button{border-radius:var(--r-sm); border:0; color:#FFFFFF; background:var(--brand); font-weight:600; padding:.45rem 1.1rem;}
.stButton>button:hover, .stDownloadButton>button:hover{background:var(--brand-dark); color:#FFFFFF;}
.stButton>button:focus-visible, .stDownloadButton>button:focus-visible, button:focus-visible, a:focus-visible, input:focus-visible, select:focus-visible{outline:2px solid var(--brand); outline-offset:2px;}
[data-baseweb="select"] > div, [data-testid="stTextInput"] input{border-radius:var(--r-sm);}
@media (prefers-reduced-motion: reduce){ *{transition:none !important; animation:none !important;} }
</style>
"""


def inject_css() -> None:
    st.markdown(CSS, unsafe_allow_html=True)


def esc(x) -> str:
    """HTML-escape any value before it is interpolated into markup."""
    return html.escape(str(x), quote=True)


def rail_brand() -> str:
    bars = "".join(
        f'<i style="height:{h}px;background:{c}"></i>'
        for h, c in ((22, "#2F62D4"), (15, "#00B5A0"), (11, "#D08A2E"), (8, "#C06BD8"))
    )
    return (
        f'<div class="rail-brand"><div class="mark">{bars}</div>'
        "<div><b>MRI</b><span>Risk Intelligence</span></div></div>"
    )


def band_chip(band: str) -> str:
    b = str(band)
    return f'<span class="chip chip-{esc(b)}">{esc(BAND_GLYPH.get(b, ""))} {esc(b)}</span>'


def band_color(band: str) -> str:
    return BAND_STYLE.get(str(band), BAND_STYLE["Low"])["bar"]


def band_ink(band: str) -> str:
    return BAND_STYLE.get(str(band), BAND_STYLE["Low"])["ink"]


def signal_legend(show_weights: bool = True) -> str:
    items = []
    for s in SIGNALS:
        suffix = " " + str(int(round(s["weight"] * 100))) + "%" if show_weights else ""
        items.append('<span><i style="background:' + s["color"] + '"></i>' + esc(s["name"]) + suffix + "</span>")
    return '<div class="legend" role="list">' + "".join(items) + "</div>"


def stacked_track(contribs: dict, title: str = "") -> str:
    """One bar on a 0-1 fused scale; segment length is weight x normalised signal."""
    left = 0.0
    segs = []
    for s in SIGNALS:
        w = max(float(contribs.get(s["key"], 0.0)), 0.0) * 100
        segs.append(f'<div class="seg" style="left:{left:.3f}%;width:{w:.3f}%;background:{s["color"]}"></div>')
        left += w
    ticks = "".join(f'<div class="tick" style="left:{t*100:.0f}%"></div>' for t in (0.30, 0.55, 0.75))
    return f'<div class="track" title="{esc(title)}">{ticks}{"".join(segs)}</div>'


def band_axis() -> str:
    marks = {"0.30": 0.30, "0.55": 0.55, "0.75": 0.75}
    return '<div class="axis">' + "".join(f'<span style="left:{v*100:.0f}%">{k}</span>' for k, v in marks.items()) + "</div>"


def band_strip(counts: dict) -> str:
    total = max(sum(counts.values()), 1)
    order = ["Critical", "High", "Medium", "Low"]
    bars = "".join(
        f'<div title="{esc(b)}: {counts.get(b, 0)}" style="width:{counts.get(b, 0)/total*100:.3f}%;background:{band_color(b)}"></div>'
        for b in order if counts.get(b, 0)
    )
    labels = "".join(
        f'<div style="width:{counts.get(b, 0)/total*100:.3f}%"><b>{counts.get(b, 0)}</b> {esc(b)}</div>'
        for b in order if counts.get(b, 0)
    )
    return f'<div class="strip" role="img" aria-label="Entities by risk band">{bars}</div><div class="strip-labels">{labels}</div>'
