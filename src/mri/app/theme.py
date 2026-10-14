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
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
:root{
  /* surfaces and ink (neutral, Apple-like) */
  --canvas:#F5F5F7; --surface:#FFFFFF; --ink:#1D1D1F; --ink2:#515154; --ink3:#6E6E73;
  --hair:rgba(0,0,0,.09); --rule:rgba(0,0,0,.09); --fill:#EEEEF1; --accent:#0066CC; --navy:#1D1D1F;
  /* signal identity colours (validated) */
  --ml:#2F5BEA; --graph:#0B9A8F; --temporal:#B7791F; --anomaly:#B83280;
  --b-low:#D2D2D7; --b-medium:#98989D; --b-high:#3A3A3C; --b-critical:#C4271B;
  --r-lg:20px; --r-md:12px; --r-sm:8px;
  --shadow:0 1px 2px rgba(0,0,0,.04), 0 10px 32px rgba(0,0,0,.05);
  --font:-apple-system,BlinkMacSystemFont,'SF Pro Text','SF Pro Display','Inter','Segoe UI',system-ui,sans-serif;
  --display:var(--font); --body:var(--font);
}
html, body, .stApp, [data-testid="stAppViewContainer"]{background:var(--canvas); color:var(--ink); font-family:var(--font); -webkit-font-smoothing:antialiased;}
[data-testid="stHeader"]{background:rgba(245,245,247,.78); backdrop-filter:saturate(180%) blur(20px); -webkit-backdrop-filter:saturate(180%) blur(20px); border-bottom:1px solid var(--hair);}
.block-container, [data-testid="stMainBlockContainer"]{max-width:1120px; padding-top:5.2rem !important; padding-bottom:5rem;}
footer{visibility:hidden;}
h1,h2,h3,h4{font-family:var(--font); color:var(--ink); letter-spacing:-0.022em;}
p,li,label,span,div{font-family:var(--font);}
.num{font-variant-numeric:tabular-nums;}

/* type scale */
.mri-h1{font-weight:700; font-size:3rem; line-height:1.06; letter-spacing:-0.035em; margin:.4rem 0 .8rem; max-width:20ch;}
.mri-h2{font-weight:600; font-size:1.55rem; line-height:1.2; letter-spacing:-0.024em; margin:3rem 0 .4rem;}
.mri-lede{color:var(--ink3); font-size:1.2rem; line-height:1.5; letter-spacing:-0.011em; max-width:56ch; margin:0 0 1.8rem;}
.mri-note{color:var(--ink3); font-size:.88rem; line-height:1.5; max-width:70ch;}
.mri-rule{border:0; border-top:1px solid var(--hair); margin:1.6rem 0;}
.mri-prose{max-width:66ch; line-height:1.6; color:var(--ink); font-size:1.05rem;}
.mri-prose p{margin:0 0 .9rem;}
.mri-prose h3{font-size:1.2rem; font-weight:600; margin:1.8rem 0 .4rem;}
.mri-prose ul{margin:.2rem 0 1rem 1.1rem; padding:0;} .mri-prose li{margin:.3rem 0;}
.mri-prose a, .mri-note a{color:var(--accent); text-decoration:none;} .mri-prose a:hover{text-decoration:underline;}

/* panels: st.container(key="panel_*") */
[class*="st-key-panel"]{background:var(--surface); border-radius:var(--r-lg); box-shadow:var(--shadow); padding:1.6rem 1.8rem 1.4rem; margin:.4rem 0 1rem;}

/* band chip: label + glyph + colour, never colour alone */
.chip{display:inline-flex; align-items:center; gap:.35rem; font-size:.78rem; font-weight:600; padding:.18rem .6rem; border-radius:999px; white-space:nowrap; background:var(--fill); color:var(--ink2);}
.chip-Medium{background:#EDEDF0; color:#424245;} .chip-High{background:#3A3A3C; color:#fff;} .chip-Critical{background:var(--b-critical); color:#fff;}
.tag{display:inline-block; font-size:.8rem; padding:.2rem .65rem; border-radius:999px; color:var(--ink2); background:var(--fill); margin:0 .35rem .35rem 0;}

/* KPI tiles */
.kpis{display:grid; grid-template-columns:repeat(4,1fr); gap:1rem; margin:.4rem 0 1rem;}
.kpi{background:var(--surface); border-radius:var(--r-lg); box-shadow:var(--shadow); padding:1.2rem 1.4rem 1.1rem;}
.kpi .v{font-size:2.6rem; font-weight:700; letter-spacing:-0.035em; line-height:1.05;}
.kpi .l{display:flex; align-items:center; gap:.4rem; color:var(--ink3); font-size:.9rem; margin-top:.3rem;}
.kpi .s{color:var(--ink3); font-size:.8rem; margin-top:.15rem;}
.kpi.crit .v{color:var(--b-critical);}
@media (max-width:760px){ .kpis{grid-template-columns:repeat(2,1fr);} .mri-h1{font-size:2.2rem;} }

/* band strip */
.strip{display:flex; gap:3px; height:.7rem; margin:.6rem 0 .4rem; border-radius:999px; overflow:hidden;}
.strip > div{min-width:3px;}
.strip-labels{display:flex; gap:3px; font-size:.82rem; color:var(--ink3);}
.strip-labels > div{overflow:visible; white-space:nowrap;}
.strip-labels b{color:var(--ink); font-weight:600;}

/* signal legend */
.legend{display:flex; flex-wrap:wrap; gap:.4rem 1.4rem; font-size:.88rem; color:var(--ink3); margin:.2rem 0 .8rem;}
.legend i{display:inline-block; width:.7rem; height:.7rem; border-radius:50%; margin-right:.4rem; vertical-align:0;}

/* case board */
.board{margin-top:.2rem;}
.board-head, .board-row{display:grid; grid-template-columns:2rem 11.5rem 6.6rem 1fr 3.4rem; gap:0 .9rem; align-items:center;}
.board-head{font-size:.78rem; color:var(--ink3); padding:0 .5rem .45rem; border-bottom:1px solid var(--hair);}
.board-row{padding:.7rem .5rem; border-bottom:1px solid var(--hair); font-size:.95rem; border-radius:var(--r-sm); transition:background .18s ease;}
.board-row:last-child{border-bottom:0;}
.board-row:hover{background:#F5F5F7;}
.board-row .rk{color:var(--ink3);} .board-row .ent b{font-weight:600;} .board-row .ent span{display:block; color:var(--ink3); font-size:.8rem; margin-top:.05rem;}
.board-row .sc{text-align:right; font-weight:600;}
.axis{position:relative; height:1rem;} .axis span{position:absolute; transform:translateX(-50%); font-size:.72rem; color:var(--ink3); top:0;}
.track{position:relative; height:.6rem; background:var(--fill); border-radius:999px; overflow:visible;}
.track .seg{position:absolute; top:0; height:100%; border-right:2px solid var(--surface);}
.track .seg:first-of-type{border-radius:999px 0 0 999px;}
.track .tick{position:absolute; top:-4px; bottom:-4px; width:0; border-left:1px dashed #AEAEB2;}
@media (max-width: 760px){ .board-head, .board-row{grid-template-columns:1.6rem 8rem 1fr 3rem;} .board-head .h-band, .board-row .bd{display:none;} }

/* contribution rows in the case file */
.contrib{display:grid; grid-template-columns:9.5rem 1fr 11.5rem; gap:.2rem .9rem; align-items:center; padding:.7rem 0; border-bottom:1px solid var(--hair); font-size:.95rem;}
.contrib .nm b{font-weight:600;} .contrib .nm span{display:block; color:var(--ink3); font-size:.8rem;}
.contrib .eq{color:var(--ink2); text-align:right;}
.contrib .track{height:.55rem;}
.contrib.total{border-bottom:0; font-weight:600;}

/* percentile rows */
.pct{display:grid; grid-template-columns:9rem 1fr 8.2rem; gap:.2rem .9rem; align-items:center; padding:.55rem 0; font-size:.92rem; border-bottom:1px solid var(--hair);}
.pct .pt{position:relative; height:.5rem; background:var(--fill); border-radius:999px;}
.pct .pt i{position:absolute; top:-5px; width:.95rem; height:.95rem; margin-left:-.47rem; background:var(--ink); border:2px solid var(--surface); border-radius:50%; box-shadow:0 1px 3px rgba(0,0,0,.25);}
.pct .pv{text-align:right; color:var(--ink2);}

/* definition grid */
.dl{display:grid; grid-template-columns:1fr auto; gap:.3rem 1rem; font-size:.95rem; max-width:26rem;}
.dl dt{color:var(--ink3);} .dl dd{margin:0; text-align:right; font-weight:600;}

/* quote */
.quote{background:var(--fill); border-radius:var(--r-md); padding:1rem 1.2rem; color:var(--ink); max-width:64ch; line-height:1.6; margin:.4rem 0 1rem;}

/* comparison table */
.cmp{width:100%; border-collapse:collapse; font-size:.95rem; max-width:44rem;}
.cmp th{font-weight:500; color:var(--ink3); font-size:.8rem; text-align:right; padding:.3rem .6rem; border-bottom:1px solid var(--hair);} .cmp th:first-child{text-align:left; padding-left:0;}
.cmp td{padding:.65rem .6rem; text-align:right; border-bottom:1px solid var(--hair);} .cmp td:first-child{text-align:left; padding-left:0; color:var(--ink2);}
.up{color:var(--b-critical); font-weight:600;} .down{color:#067647; font-weight:600;}

/* callout */
.callout{background:#FFF7E6; border-radius:var(--r-md); padding:.9rem 1.1rem; max-width:70ch; font-size:.92rem; line-height:1.55; margin:.6rem 0 1rem; color:#4A3B12;}

/* signal tiles */
.sig{background:var(--surface); border-radius:var(--r-lg); box-shadow:var(--shadow); padding:1.1rem 1.2rem; height:100%;}
.sig .dot{display:inline-block; width:.7rem; height:.7rem; border-radius:50%; margin-right:.45rem;}
.sig b{font-size:1rem; font-weight:600;} .sig .w{color:var(--ink3); margin-left:.3rem;}
.sig p{color:var(--ink3); font-size:.88rem; line-height:1.5; margin:.45rem 0 0;}

/* streamlit widget tuning */
[data-testid="stDataFrame"]{border:0; border-radius:var(--r-md); overflow:hidden; box-shadow:var(--shadow);}
.stButton>button, .stDownloadButton>button{border-radius:999px; border:0; color:#fff; background:var(--accent); font-weight:500; padding:.45rem 1.2rem; transition:background .18s ease;}
.stButton>button:hover, .stDownloadButton>button:hover{background:#0A5CB5; color:#fff;}
.stButton>button:focus-visible, .stDownloadButton>button:focus-visible, button:focus-visible{outline:3px solid rgba(0,102,204,.45); outline-offset:2px;}
[data-baseweb="select"] > div, [data-testid="stTextInput"] input{border-radius:var(--r-md);}
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
    return {"Low": "#D2D2D7", "Medium": "#98989D", "High": "#3A3A3C", "Critical": "#C4271B"}[str(band)]


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
