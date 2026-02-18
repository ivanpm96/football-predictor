
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from predictor import FootballSystem, COMPETITION_MAP

# ─── CONFIGURACIÓN ──────────────────────────────────────────────────────
st.set_page_config(
    page_title="⚽ Football Predictor",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .big-metric { font-size: 2.5rem; font-weight: 800; text-align: center; }
    .label-metric { font-size: 0.9rem; color: #888; text-align: center; }
    .pred-card { 
        border-radius: 12px; padding: 1.2rem; margin: 0.3rem 0;
        text-align: center; font-weight: 700; font-size: 1.1rem;
    }
    .winner { background: #1a472a; color: #4ade80; border: 2px solid #4ade80; }
    .neutral { background: #1e293b; color: #94a3b8; border: 2px solid #334155; }
</style>
""", unsafe_allow_html=True)

# ─── SESSION STATE ───────────────────────────────────────────────────────
if "system" not in st.session_state:
    st.session_state.system = FootballSystem()
if "trained" not in st.session_state:
    st.session_state.trained = False
if "last_config" not in st.session_state:
    st.session_state.last_config = None

# ─── SIDEBAR ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/d/d9/Football_Victoria_logo.svg/200px-Football_Victoria_logo.svg.png", width=80)
    st.title("⚙️ Configuración")
    st.markdown("---")

    comp_label = st.selectbox(
        "🏆 Competición",
        options=list(COMPETITION_MAP.keys()),
        index=0
    )

    years = st.multiselect(
        "📅 Temporadas (año inicial)",
        options=[2022, 2023, 2024, 2025],
        default=[2023, 2024],
        help="2024 = temporada 2024/25 | 2025 = temporada 2025/26"
    )

    force_dl = st.checkbox("🔄 Forzar re-descarga", value=False)

    st.markdown("---")
    train_btn = st.button("🚀 Cargar & Entrenar", use_container_width=True, type="primary")

    if st.session_state.trained:
        s = st.session_state.system.stats
        st.markdown("---")
        st.markdown("### 📊 Estado del modelo")
        st.metric("Partidos jugados", s["played"])
        st.metric("Pendientes", s["pending"])
        st.metric("ML Accuracy (CV)", f"{s['cv_score']:.1%}")
        st.metric("Ventaja local", f"{s['home_adv']}x")
        st.metric("Equipos", len(s["teams"]))

# ─── ENTRENAMIENTO ───────────────────────────────────────────────────────
if train_btn:
    if not years:
        st.error("Selecciona al menos una temporada.")
    else:
        config = (comp_label, tuple(sorted(years)))
        prog_bar = st.progress(0)
        status = st.empty()

        def prog_fn(msg, val):
            status.info(f"⏳ {msg}")
            prog_bar.progress(val)

        ok, msg = st.session_state.system.train(
            comp_label, years, force=force_dl, progress_fn=prog_fn
        )
        prog_bar.empty()
        status.empty()

        if ok:
            st.session_state.trained = True
            st.session_state.last_config = config
            st.success(f"✅ Modelo entrenado: {comp_label} | Temporadas: {years}")
            st.rerun()
        else:
            st.error(f"❌ {msg}")

# ─── HEADER PRINCIPAL ────────────────────────────────────────────────────
st.title("⚽ Football Match Predictor")
st.caption("Sistema de predicción basado en Poisson + Gradient Boosting | Datos: fixturedownload.com")

if not st.session_state.trained:
    st.info("👈 Configura la competición y temporadas en el panel lateral, luego haz clic en **Cargar & Entrenar**.")
    st.stop()

# ─── TABS ────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["🎯 Predicción manual", "📅 Próximos partidos", "📊 Análisis de equipos"])

system = st.session_state.system
teams  = system.get_teams()
comp_ids = system.comp_ids

# ════════════════════════════════════════════════════════════════════════
# TAB 1 — PREDICCIÓN MANUAL
# ════════════════════════════════════════════════════════════════════════
with tab1:
    st.subheader("Selecciona los equipos")
    col1, mid, col2 = st.columns([5, 1, 5])

    with col1:
        home = st.selectbox("🏠 Equipo Local", teams, index=0, key="home_sel")
    with mid:
        st.markdown("<br><br><p style='text-align:center;font-size:1.5rem;'>vs</p>", unsafe_allow_html=True)
    with col2:
        away_options = [t for t in teams if t != home]
        away = st.selectbox("✈️ Equipo Visitante", away_options, index=0, key="away_sel")

    comp_sel = st.selectbox("Temporada para predicción", comp_ids, index=len(comp_ids)-1)

    predict_btn = st.button("🔮 Predecir partido", type="primary", use_container_width=True)

    if predict_btn:
        with st.spinner("Calculando probabilidades..."):
            pred = system.predict(home, away, comp_sel)

        if pred is None:
            st.error("No se pudo predecir. Verifica los equipos.")
        else:
            st.markdown("---")
            # ── Header del partido ──
            c1, c2, c3 = st.columns([4,2,4])
            with c1:
                st.markdown(f"<div class='big-metric'>{home}</div><div class='label-metric'>Local</div>", unsafe_allow_html=True)
            with c2:
                st.markdown(f"<div style='text-align:center;font-size:2rem;padding-top:1rem;'>🆚</div>", unsafe_allow_html=True)
            with c3:
                st.markdown(f"<div class='big-metric'>{away}</div><div class='label-metric'>Visitante</div>", unsafe_allow_html=True)

            st.markdown(f"<div style='text-align:center;color:#888;margin:0.5rem 0;'>📊 Marcador más probable: <b>{pred['expected_score']}</b> | λ local: {pred['lambda_home']} | λ visitante: {pred['lambda_away']}</div>", unsafe_allow_html=True)
            st.markdown("---")

            # ── Tarjetas de probabilidad ──
            col_h, col_d, col_a = st.columns(3)
            results_map = {
                "H": ("🏠 Local gana", pred["combined_H"], col_h),
                "D": ("🤝 Empate",      pred["combined_D"], col_d),
                "A": ("✈️ Visita gana", pred["combined_A"], col_a),
            }
            for code, (label, prob, col) in results_map.items():
                css = "winner" if code == pred["final_prediction"] else "neutral"
                suffix = " ⭐" if code == pred["final_prediction"] else ""
                with col:
                    st.markdown(f"<div class='pred-card {css}'>{label}{suffix}<br><span style='font-size:2rem;'>{prob:.1%}</span></div>", unsafe_allow_html=True)

            st.markdown("")

            # ── Gráfico barras comparativo ──
            fig_bar = go.Figure()
            labels  = [f"🏠 {home}", "🤝 Empate", f"✈️ {away}"]
            p_vals  = [pred["combined_H"], pred["combined_D"], pred["combined_A"]]
            colors  = ["#4ade80" if v==max(p_vals) else "#64748b" for v in p_vals]

            fig_bar.add_trace(go.Bar(
                x=labels, y=[v*100 for v in p_vals],
                marker_color=colors,
                text=[f"{v:.1%}" for v in p_vals],
                textposition="outside",
            ))
            fig_bar.update_layout(
                title="Probabilidades combinadas (Poisson + ML)",
                yaxis_title="Probabilidad (%)",
                plot_bgcolor="#0f172a", paper_bgcolor="#0f172a",
                font_color="white", height=350,
                yaxis=dict(range=[0, max(p_vals)*100*1.3]),
            )
            st.plotly_chart(fig_bar, use_container_width=True)

            # ── Over / Under ──
            col_o, col_u = st.columns(2)
            with col_o:
                st.metric("📈 Over 2.5 goles", f"{pred['over_2_5']:.1%}")
            with col_u:
                st.metric("📉 Under 2.5 goles", f"{pred['under_2_5']:.1%}")

            # ── Mapa de calor de marcadores ──
            st.markdown("#### 🔥 Mapa de calor de marcadores")
            max_g = 6
            M = np.array(pred["score_matrix"])[:max_g+1, :max_g+1]
            fig_heat = px.imshow(
                M * 100,
                labels=dict(x=f"Goles {away}", y=f"Goles {home}", color="Prob %"),
                x=[str(i) for i in range(max_g+1)],
                y=[str(i) for i in range(max_g+1)],
                color_continuous_scale="Greens",
                text_auto=".1f",
                aspect="auto",
            )
            fig_heat.update_layout(
                plot_bgcolor="#0f172a", paper_bgcolor="#0f172a",
                font_color="white", height=380,
                title=f"Probabilidad (%) de cada marcador exacto",
            )
            st.plotly_chart(fig_heat, use_container_width=True)

            # ── Tabla detallada ──
            with st.expander("Ver detalle por modelo"):
                detail_df = pd.DataFrame({
                    "Modelo":       ["Poisson", "ML (Gradient Boost)", "Combinado (50/50)"],
                    f"🏠 {home}":   [f"{pred['poisson_H']:.1%}", f"{pred['ml_H']:.1%}" if pred['ml_H'] else "N/A", f"{pred['combined_H']:.1%}"],
                    "🤝 Empate":    [f"{pred['poisson_D']:.1%}", f"{pred['ml_D']:.1%}" if pred['ml_D'] else "N/A", f"{pred['combined_D']:.1%}"],
                    f"✈️ {away}":   [f"{pred['poisson_A']:.1%}", f"{pred['ml_A']:.1%}" if pred['ml_A'] else "N/A", f"{pred['combined_A']:.1%}"],
                })
                st.dataframe(detail_df, use_container_width=True, hide_index=True)


# ════════════════════════════════════════════════════════════════════════
# TAB 2 — PRÓXIMOS PARTIDOS
# ════════════════════════════════════════════════════════════════════════
with tab2:
    st.subheader("Predicciones automáticas de próximos partidos")
    comp_pending = st.selectbox("Temporada", comp_ids, index=len(comp_ids)-1, key="pending_comp")

    if st.button("🔮 Predecir todos los pendientes", use_container_width=True):
        with st.spinner("Prediciendo partidos pendientes..."):
            df_pend = system.predict_pending(comp_pending)

        if df_pend.empty:
            st.warning("No hay partidos pendientes en esta temporada.")
        else:
            st.success(f"✅ {len(df_pend)} partidos predichos")

            # Formatear
            show_cols = ["round","date","home_team","away_team",
                         "combined_H","combined_D","combined_A",
                         "final_prediction","expected_score","over_2_5"]

            for c in show_cols:
                if c not in df_pend.columns:
                    df_pend[c] = "N/A"

            df_show = df_pend[show_cols].copy()
            df_show.columns = ["Jornada","Fecha","Local","Visitante",
                                "P(Local)","P(Empate)","P(Visita)",
                                "Predicción","Marcador","Over 2.5"]

            for c in ["P(Local)","P(Empate)","P(Visita)","Over 2.5"]:
                df_show[c] = df_show[c].apply(lambda x: f"{x:.1%}" if isinstance(x,(float,int)) else x)

            st.dataframe(df_show, use_container_width=True, hide_index=True)

            csv = df_pend.to_csv(index=False).encode("utf-8")
            st.download_button("⬇️ Descargar CSV", csv, f"predicciones_{comp_pending}.csv","text/csv")


# ════════════════════════════════════════════════════════════════════════
# TAB 3 — ANÁLISIS DE EQUIPO
# ════════════════════════════════════════════════════════════════════════
with tab3:
    st.subheader("Estadísticas históricas por equipo")
    team_sel = st.selectbox("Selecciona equipo", teams, key="team_analysis")

    df = system.df_proc
    if df is not None:
        played = df[df["played"]].copy()
        team_df = played[(played["home_team"] == team_sel) | (played["away_team"] == team_sel)].copy()
        team_df = team_df.sort_values("date")

        # Goles por partido
        gf_list, gc_list, dates_list, result_list = [], [], [], []
        for _, row in team_df.iterrows():
            if row["home_team"] == team_sel:
                gf_list.append(row["home_score"]); gc_list.append(row["away_score"])
                result_list.append("W" if row["result"]=="H" else ("D" if row["result"]=="D" else "L"))
            else:
                gf_list.append(row["away_score"]); gc_list.append(row["home_score"])
                result_list.append("W" if row["result"]=="A" else ("D" if row["result"]=="D" else "L"))
            dates_list.append(row["date"])

        team_stats = pd.DataFrame({"date":dates_list,"gf":gf_list,"gc":gc_list,"result":result_list})

        # Métricas
        total = len(team_stats)
        wins  = sum(r=="W" for r in result_list)
        draws = sum(r=="D" for r in result_list)
        loss  = sum(r=="L" for r in result_list)

        m1,m2,m3,m4,m5 = st.columns(5)
        m1.metric("Partidos",total)
        m2.metric("✅ Victorias",wins)
        m3.metric("🤝 Empates",draws)
        m4.metric("❌ Derrotas",loss)
        m5.metric("Goles/partido", f"{np.mean(gf_list):.2f}")

        # Gráfico forma reciente
        fig_form = go.Figure()
        color_map = {"W":"#4ade80","D":"#facc15","L":"#f87171"}
        fig_form.add_trace(go.Bar(
            x=list(range(1,len(gf_list)+1)), y=gf_list,
            name="Goles a favor",
            marker_color=[color_map[r] for r in result_list],
        ))
        fig_form.add_trace(go.Bar(
            x=list(range(1,len(gc_list)+1)), y=[-g for g in gc_list],
            name="Goles en contra",
            marker_color="#475569",
        ))
        fig_form.update_layout(
            title=f"Historial de goles — {team_sel} (verde=Victoria, amarillo=Empate, rojo=Derrota)",
            barmode="relative", height=380,
            plot_bgcolor="#0f172a", paper_bgcolor="#0f172a",
            font_color="white",
            yaxis_title="Goles",
        )
        st.plotly_chart(fig_form, use_container_width=True)
