# app.py v4.2 — Football Predictor Pro · Complete
# Fixes: BUG-1,2,3 + UX-1..5 + F-1..3
# Extra: top-5 scores, confidence badge, quick stats, xG trend, perf dashboard
import streamlit as st
import pandas as pd
import numpy as np
import re
import plotly.graph_objects as go
import plotly.express as px
from orchestrator import DataOrchestrator, get_xg_source
from catalog_builder import (
    load_or_refresh_catalog, get_regions, get_comp_by_label,
    get_season_options, get_train_defaults, get_predict_default_label,
)

st.set_page_config(
    page_title="Football Predictor Pro",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ──────────────────────────────────────────────────────────────
_CSS = """
<style>
.big-num{font-size:2.2rem;font-weight:900;text-align:center;line-height:1.1;}
.lbl{font-size:.78rem;color:#94a3b8;text-align:center;margin-top:2px;}
.pred-card{border-radius:14px;padding:1.1rem .8rem;text-align:center;
  font-weight:700;font-size:1rem;margin:.2rem 0;}
.winner{background:#14532d;color:#4ade80;border:2px solid #4ade80;}
.neutral{background:#1e293b;color:#94a3b8;border:2px solid #334155;}
.model-badge{background:#052e16;border:1px solid #16a34a;border-radius:8px;
  padding:.45rem .8rem;font-size:.82rem;color:#4ade80;margin-bottom:.5rem;}
.status-ok{background:#052e16;border:1px solid #16a34a;border-radius:10px;
  padding:.5rem 1rem;font-size:.85rem;color:#4ade80;}
.status-warn{background:#1c1917;border:1px solid #d97706;border-radius:10px;
  padding:.5rem 1rem;font-size:.85rem;color:#fbbf24;}
.status-run{background:#0c1a3a;border:1px solid #3b82f6;border-radius:10px;
  padding:.5rem 1rem;font-size:.85rem;color:#93c5fd;}
.match-card{background:#1e293b;border:1px solid #3b82f6;border-radius:12px;
  padding:.7rem 1.1rem;margin:.4rem 0;}
.section-hdr{font-size:.92rem;font-weight:700;color:#94a3b8;
  letter-spacing:.07em;text-transform:uppercase;
  margin:1rem 0 .4rem;border-left:3px solid #3b82f6;padding-left:.6rem;}
.xg-badge{display:inline-block;background:#1e3a5f;color:#93c5fd;
  font-size:.72rem;border-radius:20px;padding:2px 8px;margin-left:6px;font-weight:600;}
.conf-hi{color:#4ade80;font-weight:800;}
.conf-md{color:#fbbf24;font-weight:800;}
.conf-lo{color:#f87171;font-weight:800;}
.score-row{display:flex;justify-content:space-between;
  border-bottom:1px solid #1e293b;padding:.28rem 0;font-size:.88rem;}
.mkt-row{display:flex;justify-content:space-between;
  border-bottom:1px solid #1e293b;padding:.28rem 0;font-size:.88rem;}
.chip{display:inline-block;background:#1e293b;border:1px solid #334155;
  border-radius:20px;padding:3px 10px;font-size:.76rem;margin:2px;color:#94a3b8;}
</style>
"""
st.markdown(_CSS, unsafe_allow_html=True)

# ── SESSION STATE ────────────────────────────────────────────────────
for _k, _v in [
    ("orch",           None),
    ("auto_on",        False),
    ("search_history", []),
    ("pending_pred",   None),
]:
    if _k not in st.session_state:
        st.session_state[_k] = _v

if st.session_state.orch is None:
    st.session_state.orch = DataOrchestrator()

orch    = st.session_state.orch
catalog = orch.get_catalog()
regions = get_regions(catalog)
status  = orch.get_status()

# ── HELPERS ──────────────────────────────────────────────────────────
def friendly_comp(comp_id):
    if not comp_id:
        return comp_id
    parts = comp_id.rsplit("-", 1)
    slug  = parts[0]
    yr    = parts[1] if len(parts) > 1 else ""
    meta  = catalog.get("competitions", {}).get(slug, {})
    name  = meta.get("name", slug.replace("-", " ").title())
    stype = meta.get("season_type", "single")
    if stype == "split" and yr.isdigit():
        return f"{name} {yr}/{str(int(yr)+1)[2:]}"
    return f"{name} {yr}"

def conf_badge(label):
    cls = {"Alta": "conf-hi", "Media": "conf-md", "Baja": "conf-lo"}.get(label, "conf-md")
    return f"<span class='{cls}'>{label}</span>"

def odd_str(p):
    return f"{round(1/p,2)}" if p and p > 0 else "99"

def push_history(home, away, comp_id):
    entry  = {"home": home, "away": away, "comp_id": comp_id,
              "label": f"{home} vs {away}"}
    h      = [e for e in st.session_state.search_history if e["label"] != entry["label"]]
    h.insert(0, entry)
    st.session_state.search_history = h[:6]

def top5_scores(matrix, max_g=6):
    M    = np.array(matrix)[:max_g+1, :max_g+1]
    rows = []
    for i in range(max_g+1):
        for j in range(max_g+1):
            rows.append({"Marcador": f"{i}-{j}", "Prob": float(M[i][j])})
    return sorted(rows, key=lambda x: -x["Prob"])[:5]

def mkt_row(label, prob):
    od  = odd_str(prob)
    return (
        f"<div class='mkt-row'>"
        f"<span style='color:#94a3b8'>{label}</span>"
        f"<span><b>{prob:.1%}</b>"
        f"<span style='color:#facc15;font-size:.8rem'> ({od})</span>"
        f"</span></div>"
    )

def render_prediction(pred, home, away, hs, as_, compact=False):
    """Renderiza bloque completo de predicción. compact=True solo muestra 1X2."""
    # 1X2 cards
    vals = [pred["combined_H"], pred["combined_D"], pred["combined_A"]]
    fp   = pred["final_prediction"]
    c_h, c_d, c_a = st.columns(3)
    for col, code, lbl, prob, odd in [
        (c_h, "H", f"Local · {home[:14]}", pred["combined_H"], pred["odds_H"]),
        (c_d, "D", "Empate",               pred["combined_D"], pred["odds_D"]),
        (c_a, "A", f"Visita · {away[:14]}",pred["combined_A"], pred["odds_A"]),
    ]:
        css = "winner" if code == fp else "neutral"
        sfx = " ✓"    if code == fp else ""
        with col:
            st.markdown(
                f"<div class='pred-card {css}'>"
                f"{lbl}{sfx}"
                f"<br><span style='font-size:1.9rem'>{prob:.1%}</span>"
                f"<br><span style='color:#facc15'>Cuota {odd}</span>"
                f"</div>",
                unsafe_allow_html=True,
            )
    # KPIs rápidos
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Marcador prob.", pred["expected_score"])
    k2.metric("Over 2.5",       f"{pred['over_2_5']:.1%}")
    k3.metric("BTTS",           f"{pred['btts']:.1%}")
    k4.markdown(
        f"<div style='text-align:center'>"
        f"<div class='lbl'>Confianza</div>"
        f"<div style='font-size:1.4rem'>"
        f"{conf_badge(pred['confidence'])}"
        f"</div></div>",
        unsafe_allow_html=True,
    )
    if compact:
        return
    st.markdown("---")
    # Barra comparativa
    fig_b = go.Figure(go.Bar(
        x=[f"Local ({home[:10]})", "Empate", f"Visita ({away[:10]})"],
        y=[v * 100 for v in vals],
        text=[f"{v:.1%}" for v in vals],
        textposition="outside",
        marker_color=["#4ade80" if v == max(vals) else "#475569" for v in vals],
    ))
    fig_b.update_layout(
        yaxis=dict(range=[0, max(vals) * 140]),
        height=220, showlegend=False,
        plot_bgcolor="#0f172a", paper_bgcolor="#0f172a",
        font_color="white", margin=dict(t=8, b=5),
    )
    st.plotly_chart(fig_b, use_container_width=True)
    # Comparativa rápida equipos
    if hs and as_:
        st.markdown(
            "<div class='section-hdr'>Comparativa de equipos</div>",
            unsafe_allow_html=True,
        )
        df_cmp = pd.DataFrame([
            {"": "PPP",       home[:16]: hs.get("total_PPP", 0),    away[:16]: as_.get("total_PPP", 0)},
            {"": "GF/pj",     home[:16]: hs.get("avg_gf", 0),       away[:16]: as_.get("avg_gf", 0)},
            {"": "GC/pj",     home[:16]: hs.get("avg_gc", 0),       away[:16]: as_.get("avg_gc", 0)},
            {"": "Cs",        home[:16]: hs.get("clean_sheets", 0), away[:16]: as_.get("clean_sheets", 0)},
            {"": "λ Poisson", home[:16]: pred["lambda_home"],        away[:16]: pred["lambda_away"]},
        ])
        st.dataframe(df_cmp, use_container_width=True, hide_index=True)
    # Mercados
    st.markdown(
        "<div class='section-hdr'>Mercados</div>",
        unsafe_allow_html=True,
    )
    m1, m2 = st.columns(2)
    with m1:
        st.markdown("**Over / Under**")
        pairs = [
            ("Over 1.5", pred["over_1_5"], "Under 1.5", pred["under_1_5"]),
            ("Over 2.5", pred["over_2_5"], "Under 2.5", pred["under_2_5"]),
            ("Over 3.5", pred["over_3_5"], "Under 3.5", pred["under_3_5"]),
        ]
        for ol, op, ul, up in pairs:
            co, cu = st.columns(2)
            co.metric(ol, f"{op:.1%}", f"({odd_str(op)})")
            cu.metric(ul, f"{up:.1%}", f"({odd_str(up)})")
    with m2:
        st.markdown("**Otros mercados**")
        mkts = [
            ("BTTS",                   pred["btts"]),
            ("No BTTS",                pred["no_btts"]),
            (f"AH {home[:12]} −0.5", pred["asian_home"]),
            (f"AH {away[:12]} −0.5", pred["asian_away"]),
        ]
        for lbl_, p_ in mkts:
            st.markdown(mkt_row(lbl_, p_), unsafe_allow_html=True)
    # Top 5 marcadores
    st.markdown(
        "<div class='section-hdr'>Top 5 marcadores más probables</div>",
        unsafe_allow_html=True,
    )
    t5 = top5_scores(pred["score_matrix"])
    cols5 = st.columns(5)
    for i, row in enumerate(t5):
        with cols5[i]:
            st.markdown(
                f"<div style='text-align:center;background:#1e293b;"
                f"border-radius:8px;padding:.4rem .3rem;'>"
                f"<div style='font-size:1.2rem;font-weight:800'>{row['Marcador']}</div>"
                f"<div style='color:#94a3b8;font-size:.8rem'>{row['Prob']:.1%}</div>"
                f"</div>",
                unsafe_allow_html=True,
            )
    # Mapa de calor
    st.markdown(
        "<div class='section-hdr'>Mapa de calor de marcadores</div>",
        unsafe_allow_html=True,
    )
    M = np.array(pred["score_matrix"])[:7, :7]
    fig_h = px.imshow(
        M * 100,
        labels=dict(x=f"Goles {away}", y=f"Goles {home}", color="%"),
        x=[str(i) for i in range(7)],
        y=[str(i) for i in range(7)],
        color_continuous_scale="Greens",
        text_auto=".1f",
        aspect="auto",
    )
    fig_h.update_layout(
        plot_bgcolor="#0f172a", paper_bgcolor="#0f172a",
        font_color="white", height=340,
        margin=dict(t=30, b=10),
    )
    st.plotly_chart(fig_h, use_container_width=True)
    # xG
    if pred.get("has_xg") and pred.get("xg_data"):
        xd = pred["xg_data"]
        st.markdown(
            "<div class='section-hdr'>Expected Goals</div>",
            unsafe_allow_html=True,
        )
        x1, x2, x3, x4 = st.columns(4)
        x1.metric(f"xGF/pj {home[:12]}", xd.get("home_xgF_pg", "—"))
        x2.metric(f"Suerte {home[:12]}", xd.get("home_luck", "—"),
                  help="Positivo = ha tenido suerte, posible regresión a la media")
        x3.metric(f"xGF/pj {away[:12]}", xd.get("away_xgF_pg", "—"))
        x4.metric(f"Suerte {away[:12]}", xd.get("away_luck", "—"))
        src = orch.state.get("xg_source", "—")
        st.caption(
            f"xG source: {src} · Lambda = 50% goles reales + 50% xG esperados"
        )
