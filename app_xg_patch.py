
# ======================================================================
# PATCH app.py — Agregar Tab xG y mostrar datos xG en prediccion/stats
# ======================================================================
# 
# PASO 1: Cambiar la lista de tabs (linea tab1,...= st.tabs([...]))
# ──────────────────────────────────────────────────────────────────────
# ANTES:
#   tab1,tab2,tab3,tab4,tab5,tab6,tab7,tab8=st.tabs([...,'Catalogo'])
#
# DESPUES (agregar tab xG entre Backtesting y Catalogo):
#   tab1,tab2,tab3,tab4,tab5,tab6,tab7,tab8,tab9=st.tabs([
#       'Prediccion','Tabla','Estadisticas','H2H',
#       'Proximos partidos','Datos Crudos','Backtesting','xG Analisis','Catalogo'])
#
# PASO 2: Agregar al final del TAB 1 (Prediccion), dentro del bloque "if pred:"
# ──────────────────────────────────────────────────────────────────────
# Despues del bloque de Forma reciente, agregar:
TAB1_XG_BLOCK = """
            # ── xG Panel (solo si el sistema tiene xG) ──────────────────
            if pred.get('has_xg') and pred.get('xg_data'):
                st.markdown("---")
                st.markdown("<div class=\'section-hdr\'>Expected Goals (xG)</div>",unsafe_allow_html=True)
                xd = pred['xg_data']
                x1,x2,x3,x4,x5,x6,x7,x8 = st.columns(8)
                x1.metric(f"xGF/pj {home}", xd.get('home_xgF_pg','—'))
                x2.metric(f"xGA/pj {home}", xd.get('home_xgA_pg','—'))
                x3.metric(f"Suerte {home}", xd.get('home_luck','—'),
                    help="(Goles - xG) - (xGC - GC). Positivo = ha tenido suerte, puede regresar.")
                x4.metric(f"Overperf Atq {home}", xd.get('home_overperf_att','—'),
                    help="Goles - xG. Si > 0, marca mas de lo esperado (posible regresion).")
                x5.metric(f"xGF/pj {away}", xd.get('away_xgF_pg','—'))
                x6.metric(f"xGA/pj {away}", xd.get('away_xgA_pg','—'))
                x7.metric(f"Suerte {away}", xd.get('away_luck','—'))
                x8.metric(f"Overperf Atq {away}", xd.get('away_overperf_att','—'))
                st.caption(
                    "Lambda Poisson hibrido = 50% goles reales + 50% xG. "
                    "xG source: understat.com"
                )
"""

# PASO 3: Agregar TAB 8 completo (tab8 = xG Analisis)
# ──────────────────────────────────────────────────────────────────────
TAB8_XG = """
# ── TAB 8: xG ANALISIS ────────────────────────────────────────────────
with tab8:
    if not system.has_xg:
        st.warning("xG no disponible para esta liga. Understat cubre solo: EPL, Bundesliga, La Liga, Serie A, Ligue 1.")
    else:
        pred_lbl = comp_info.get('predict_label','')
        st.subheader(f"xG Analysis — {comp_label} {pred_lbl}")
        st.caption("Fuente: understat.com | Lambda Poisson = 50% goles + 50% xG")

        # xG Tabla de posiciones
        st.markdown("#### Tabla de rendimiento esperado (xG)")
        df_xg_t = system.xg_league_table()
        if df_xg_t.empty:
            st.info("Sin datos xG suficientes.")
        else:
            st.dataframe(df_xg_t, use_container_width=True, hide_index=True)
            st.caption("**OVP_Atq** > 0: el equipo marca MAS de lo esperado (riesgo de regresion). "
                       "**Suerte** alta = posibles resultados inflados.")

            # Chart xG_Diff vs GD
            fig_xg = go.Figure()
            fig_xg.add_trace(go.Bar(
                name='xG Diff', x=df_xg_t['Equipo'], y=df_xg_t['xG_Diff'],
                marker_color='#3b82f6'))
            fig_xg.add_trace(go.Bar(
                name='GD Real', x=df_xg_t['Equipo'], y=df_xg_t['GA'].apply(lambda x: -x)+df_xg_t['GF'],
                marker_color='#94a3b8', opacity=0.6))
            fig_xg.update_layout(barmode='group', height=320,
                plot_bgcolor='#0f172a', paper_bgcolor='#0f172a', font_color='white',
                legend=dict(orientation='h'), margin=dict(t=20,b=10))
            st.plotly_chart(fig_xg, use_container_width=True)

            # Luck index scatter
            st.markdown("#### Indice de suerte (luck index)")
            st.caption("Eje X: overperformance ataque (Goles - xG). Eje Y: overperformance defensa (xGC - GC). "
                       "Cuadrante superior derecha = muy afortunado, pronostico a la baja.")
            fig_luck = go.Figure(go.Scatter(
                x=df_xg_t['OVP_Atq'],
                y=df_xg_t['OVP_Def'],
                mode='markers+text',
                text=df_xg_t['Equipo'],
                textposition='top center',
                marker=dict(size=10, color=df_xg_t['Suerte'],
                    colorscale='RdYlGn', showscale=True,
                    colorbar=dict(title='Suerte')),
            ))
            fig_luck.add_hline(y=0, line_dash='dash', line_color='#475569')
            fig_luck.add_vline(x=0, line_dash='dash', line_color='#475569')
            fig_luck.update_layout(height=500,
                plot_bgcolor='#0f172a', paper_bgcolor='#0f172a', font_color='white',
                xaxis_title='Overperf. Ataque (Goles - xG)',
                yaxis_title='Overperf. Defensa (xGC - GC)',
                margin=dict(t=20))
            st.plotly_chart(fig_luck, use_container_width=True)

        st.markdown("---")
        st.markdown("#### xG de un equipo especifico")
        team_xg = st.selectbox("Equipo", teams, key="xg_team")
        xg_s = system.xg_team_summary(team_xg)
        if xg_s:
            m1,m2,m3,m4,m5,m6 = st.columns(6)
            m1.metric("xGF total",    xg_s.get('xGF','—'))
            m2.metric("xGA total",    xg_s.get('xGA','—'))
            m3.metric("xGF/pj",       xg_s.get('xGF_per_game','—'))
            m4.metric("xGA/pj",       xg_s.get('xGA_per_game','—'))
            m5.metric("Overperf Atq", xg_s.get('overperf_attack','—'),
                help="Goles reales - xG. Positivo = ha marcado por encima de la calidad de sus tiros.")
            m6.metric("Indice Suerte",xg_s.get('luck_index','—'),
                help="Suerte total. Valores altos indican resultados inflados por fortuna.")
        else:
            st.info("Sin datos xG para este equipo.")
"""
