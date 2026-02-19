# app.py v4 — Football Predictor Pro · Autónomo · Búsqueda Global
import streamlit as st, pandas as pd, numpy as np, re
import plotly.graph_objects as go, plotly.express as px
from streamlit_autorefresh import st_autorefresh
from orchestrator import DataOrchestrator, get_xg_source
from catalog_builder import (
    load_or_refresh_catalog, get_regions, get_comp_by_label,
    get_season_options, get_train_defaults, get_predict_default_label,
)

st.set_page_config(page_title='Football Predictor Pro',page_icon='⚽',
                   layout='wide',initial_sidebar_state='expanded')

st.markdown('''<style>
.big-num{font-size:2.2rem;font-weight:900;text-align:center;line-height:1.1;}
.lbl{font-size:.78rem;color:#94a3b8;text-align:center;margin-top:2px;}
.pred-card{border-radius:14px;padding:1.1rem .8rem;text-align:center;
  font-weight:700;font-size:1rem;margin:.2rem 0;}
.winner{background:#14532d;color:#4ade80;border:2px solid #4ade80;}
.neutral{background:#1e293b;color:#94a3b8;border:2px solid #334155;}
.status-ok{background:#052e16;border:1px solid #16a34a;border-radius:10px;
  padding:.5rem 1rem;font-size:.85rem;color:#4ade80;}
.status-warn{background:#1c1917;border:1px solid #d97706;border-radius:10px;
  padding:.5rem 1rem;font-size:.85rem;color:#fbbf24;}
.status-run{background:#0c1a3a;border:1px solid #3b82f6;border-radius:10px;
  padding:.5rem 1rem;font-size:.85rem;color:#93c5fd;}
.search-result{background:#1e293b;border-radius:10px;padding:.6rem .9rem;
  margin:.3rem 0;cursor:pointer;border:1px solid #334155;}
.xg-badge{display:inline-block;background:#1e3a5f;color:#93c5fd;
  font-size:.72rem;border-radius:20px;padding:2px 8px;margin-left:6px;font-weight:600;}
.section-hdr{font-size:.95rem;font-weight:700;color:#94a3b8;letter-spacing:.07em;
  text-transform:uppercase;margin:1rem 0 .4rem;border-left:3px solid #3b82f6;padding-left:.6rem;}
</style>''',unsafe_allow_html=True)

# ── SESSION STATE ─────────────────────────────────────────────────────
if 'orch' not in st.session_state:
    st.session_state.orch = DataOrchestrator()
if 'auto_on' not in st.session_state: st.session_state.auto_on = False
if 'search_result' not in st.session_state: st.session_state.search_result = None
if 'deep_pb' not in st.session_state: st.session_state.deep_pb = None

orch    = st.session_state.orch
catalog = orch.get_catalog()
regions = get_regions(catalog)
status  = orch.get_status()

if st.session_state.auto_on:
    st_autorefresh(interval=5*60*1000,key='ar')

# ════════ SIDEBAR ══════════════════════════════════════════════════════
with st.sidebar:
    st.markdown('## ⚽ Football Predictor Pro')
    st.caption(f"v4 · {catalog.get('total_comps',0)} ligas · Autónomo")
    st.markdown('---')

    # ── STATUS BOX ──────────────────────────────────────────────────
    s = status
    if s['deep_running']:
        st.markdown(
            f"<div class='status-run'>⚡ Sync Deep en curso...<br>"
            f"{s.get('deep_message','')}<br>"
            f"<b>{s.get('deep_progress',0):.0%}</b></div>",
            unsafe_allow_html=True)
    elif s['system_trained']:
        xg_icon = {'understat':'🟢 Understat','fbref':'🔵 FBref','none':'⚪ Sin xG'}.get(str(s.get('xg_source','none')),'⚪')
        st.markdown(
            f"<div class='status-ok'>✅ {s.get('current_league','—')}"
            f" {s.get('current_season','')}"
            f" | {s.get('total_played',0):,} partidos"
            f" | {xg_icon}</div>",unsafe_allow_html=True)
    else:
        st.markdown(
            "<div class='status-warn'>⚠️ Sin datos cargados. Selecciona una liga o usa Sync Deep.</div>",
            unsafe_allow_html=True)

    if s['search_built']:
        st.caption(f"🔍 {s.get('search_teams',0):,} equipos indexados")
    st.markdown('---')

    # ── SYNC BUTTONS ────────────────────────────────────────────────
    st.markdown('### 🔄 Sincronización')
    c1,c2 = st.columns(2)
    with c1:
        sync_min_btn = st.button('⚡ Sync\nMínimo',use_container_width=True,
                                  help='Descarga solo la liga seleccionada + xG')
    with c2:
        sync_deep_bg = st.button('🌍 Sync\nDeep',use_container_width=True,
                                  help='Descarga TODAS las ligas en background (~20 min)')
    force_dl = st.checkbox('Forzar re-descarga',key='force_dl')

    if sync_deep_bg:
        if not orch._deep_running:
            ok = orch.sync_deep_background(force=force_dl)
            if ok: st.info('Sync Deep iniciado en background. Puedes seguir usando la app.')
        else:
            st.warning('Ya hay un Sync Deep en curso.')
    st.markdown('---')

    # ── SELECCIÓN DE LIGA ───────────────────────────────────────────
    st.markdown('### 🏆 Liga para entrenar')
    region_list  = list(regions.keys())
    region       = st.selectbox('Región',region_list,key='sb_region')
    comp_labels  = regions.get(region,[])
    comp_label   = st.selectbox('Competición',comp_labels,key='sb_comp')
    comp_info    = get_comp_by_label(catalog,comp_label)
    slug         = comp_info.get('slug','')
    season_opts  = get_season_options(comp_info)
    train_defs   = get_train_defaults(comp_info)
    predict_def  = get_predict_default_label(comp_info)
    stype        = comp_info.get('season_type','single')
    st.caption(f'Tipo: {stype} | xG: {get_xg_source(slug) or "Sin cobertura"}')

    train_labels = st.multiselect(
        'Temporadas de entrenamiento',
        options=list(season_opts.keys()),
        default=[l for l in train_defs if l in season_opts],key='tr_lbl')
    predict_lbl  = st.selectbox(
        'Temporada activa',
        options=list(season_opts.keys()),
        index=list(season_opts.keys()).index(predict_def) if predict_def in season_opts else 0,
        key='pred_lbl')

    if sync_min_btn:
        if not train_labels:
            st.error('Selecciona al menos una temporada de entrenamiento.')
        else:
            train_yrs = [season_opts[l] for l in train_labels if l in season_opts]
            pred_yr   = season_opts.get(predict_lbl,0)
            pb = st.progress(0.); sm = st.empty()
            def pf(msg,v): sm.caption(msg); pb.progress(float(v))
            ok,msg = orch.sync_minimum(slug,train_yrs,pred_yr,force=force_dl,progress_fn=pf)
            pb.empty(); sm.empty()
            if ok:
                st.success(f'Listo: {msg}')
                st.rerun()
            else:
                st.error(f'Error: {msg}')

    st.markdown('---')
    st.markdown('### ⚙️ Auto-update')
    auto_on = st.toggle('Activar',value=st.session_state.auto_on)
    if auto_on != st.session_state.auto_on:
        st.session_state.auto_on=auto_on; st.rerun()
    if s['system_trained']:
        if st.button('Chequear nuevos resultados',use_container_width=True):
            from auto_updater import AutoUpdater
            upd = AutoUpdater(orch.system,orch.system.comp_ids)
            r   = upd.check_all()
            nm  = r.get('new_matches',0)
            orch.search.build(orch.system.df_proc,catalog)
            st.success(f'{nm} nuevos partidos' if nm>0 else '✅ Al día')

# ════════ CABECERA ═════════════════════════════════════════════════════
st.title('⚽ Football Predictor Pro')
st.caption('Búsqueda global · xG automático · Multi-liga · Tiempo real')

teams = orch.get_teams()

tabs = st.tabs([
    '🔍 Buscar & Analizar',
    '⚡ Predicción Manual',
    '📊 Tabla',
    '👤 Estadísticas',
    '⚔️ H2H',
    '📅 Próximos',
    '📁 Datos Crudos',
    '🔬 Backtesting',
    '📈 xG Análisis',
    '🌐 Catálogo',
])
tab_search,tab_pred,tab_table,tab_stats,tab_h2h,tab_next,tab_raw,tab_bt,tab_xg,tab_cat = tabs

# ══════════════════════════════════════════════════════════════════════
# TAB 0: BÚSQUEDA GLOBAL & ANALIZAR
# ══════════════════════════════════════════════════════════════════════
with tab_search:
    st.markdown('<div class="section-hdr">Búsqueda Universal</div>',unsafe_allow_html=True)
    st.caption('Escribe el nombre de un equipo, una liga, o un partido (Ejemplo: "Bayern vs Dortmund")')

    query = st.text_input('',placeholder='🔍  Bayern Munich, Chelsea vs Arsenal, Premier League...',
                           key='global_search',label_visibility='collapsed')

    # ── Deep Sync Progress (si está corriendo) ───────────────────────
    if orch._deep_running:
        s2 = orch.get_status()
        st.markdown(
            f"<div class='status-run'>⚡ Sync Deep en curso: {s2.get('deep_message','')} "
            f"— {s2.get('deep_progress',0):.0%}</div>",unsafe_allow_html=True)
        st.progress(float(s2.get('deep_progress',0.0)))

    if not orch.search.built and not orch._deep_running:
        st.info('Para usar la búsqueda global: Sync Deep (todas las ligas) o Sync Mínimo (liga actual).')

    if query and orch.search.built:
        # ── Detectar si es búsqueda de partido (contiene 'vs' o '-') ─
        is_match_query = bool(re.search(r'\bvs\.?\b|\s+-\s+',query,re.I))

        if is_match_query:
            parts = re.split(r'\s+vs\.?\s+|\s+-\s+',query.strip(),maxsplit=1,flags=re.I)
            home_q = parts[0].strip(); away_q = parts[1].strip() if len(parts)>1 else ''
            match_info = orch.search.find_match(home_q,away_q)

            if match_info:
                st.success(f"Partido encontrado: **{match_info['home']}** vs **{match_info['away']}** — `{match_info['comp_id']}`")
                col_a,col_b = st.columns(2)
                with col_a: st.metric('Local',match_info['home'])
                with col_b: st.metric('Visitante',match_info['away'])
                st.metric('Competición',match_info['comp_id'])
                st.metric('Estado','Pendiente' if not match_info['played'] else 'Jugado')

                if orch.system.trained and match_info['comp_id'].startswith(orch.system.stats.get('comp_slug','__')):
                    if st.button('⚡ Analizar este partido ahora',type='primary',use_container_width=True):
                        with st.spinner('Calculando...'):
                            analysis = orch.analyze_match(match_info['home'],match_info['away'])
                        if analysis:
                            pred = analysis['prediction']
                            st.markdown('---')
                            st.markdown('<div class="section-hdr">Pronóstico 1X2</div>',unsafe_allow_html=True)
                            c_h,c_d,c_a = st.columns(3)
                            for col,code,lbl,prob,odd in [
                                (c_h,'H',f'Local {match_info["home"]}',pred['combined_H'],pred['odds_H']),
                                (c_d,'D','Empate',pred['combined_D'],pred['odds_D']),
                                (c_a,'A',f'Visita {match_info["away"]}',pred['combined_A'],pred['odds_A'])]:
                                css='winner' if code==pred['final_prediction'] else 'neutral'
                                sfx=' ✓' if code==pred['final_prediction'] else ''
                                with col:
                                    st.markdown(
                                        f"<div class='pred-card {css}'>{lbl}{sfx}"
                                        f"<br><span style='font-size:1.9rem'>{prob:.1%}</span>"
                                        f"<br><span style='color:#facc15'>Cuota: {odd}</span></div>",
                                        unsafe_allow_html=True)
                            st.markdown('---')
                            c1,c2,c3,c4 = st.columns(4)
                            c1.metric('Marcador prob.',pred['expected_score'])
                            c2.metric('Over 2.5',f"{pred['over_2_5']:.1%}")
                            c3.metric('BTTS',f"{pred['btts']:.1%}")
                            c4.metric('Confianza',pred['confidence'])
                            if pred.get('has_xg') and pred.get('xg_data'):
                                xd = pred['xg_data']
                                st.markdown('<div class="section-hdr">Expected Goals (xG)</div>',unsafe_allow_html=True)
                                xc1,xc2,xc3,xc4 = st.columns(4)
                                xc1.metric(f"xGF/pj {match_info['home']}",xd.get('home_xgF_pg','—'))
                                xc2.metric(f"Suerte {match_info['home']}",xd.get('home_luck','—'))
                                xc3.metric(f"xGF/pj {match_info['away']}",xd.get('away_xgF_pg','—'))
                                xc4.metric(f"Suerte {match_info['away']}",xd.get('away_luck','—'))
                else:
                    slug_q = match_info['comp_id'].rsplit('-',1)[0]
                    yr_q   = int(match_info['comp_id'].rsplit('-',1)[-1])
                    ci = get_comp_by_label(catalog, comp_label)
                    opts = get_season_options(ci)
                    if not orch.system.trained or not match_info['comp_id'].startswith(orch.system.stats.get('comp_slug','__')):
                        st.warning(f"Liga `{slug_q}` no está entrenada. Click para cargar:")
                        if st.button(f'Cargar {slug_q}-{yr_q} ahora',use_container_width=True):
                            train_yrs2 = [yr_q-2,yr_q-1,yr_q]
                            pb2 = st.progress(0.); sm2 = st.empty()
                            def pf2(msg,v): sm2.caption(msg); pb2.progress(float(v))
                            ok2,msg2 = orch.on_demand_load(slug_q,train_yrs2,yr_q,progress_fn=pf2)
                            pb2.empty(); sm2.empty()
                            if ok2: st.success('Listo'); st.rerun()
                            else:   st.error(msg2)
            else:
                st.warning('Partido no encontrado en los datos cargados. Intenta Sync Deep o carga la liga manualmente.')
                h_hits = orch.search.search_teams(home_q,top_n=3)
                a_hits = orch.search.search_teams(away_q,top_n=3)
                if h_hits: st.write('Equipos similares para home:',
                    [f"{h['team']} ({h['comp_id']})" for h in h_hits])
                if a_hits: st.write('Equipos similares para away:',
                    [f"{a['team']} ({a['comp_id']})" for a in a_hits])
        else:
            # ── Búsqueda de equipo ─────────────────────────────────
            team_hits = orch.search.search_teams(query,top_n=10)
            if team_hits:
                st.markdown(f'**{len(team_hits)} resultados para:** `{query}`')
                for hit in team_hits:
                    score_pct = int(hit['score']*100)
                    st.markdown(
                        f"<div class='search-result'>⚽ <b>{hit['team']}</b> "
                        f"— <code>{hit['comp_id']}</code> "
                        f"<span style='color:#64748b'>({score_pct}% match)</span></div>",
                        unsafe_allow_html=True)
                # Próximos partidos del primer resultado
                best_team = team_hits[0]['team']
                df_up = orch.search.upcoming_matches(best_team,limit=5)
                if not df_up.empty:
                    st.markdown(f'**Próximos fixtures de {best_team}:**')
                    st.dataframe(df_up,use_container_width=True,hide_index=True)
                df_rc = orch.search.search_matches(query,top_n=5)
                if not df_rc.empty:
                    st.markdown(f'**Últimos partidos de {best_team}:**')
                    st.dataframe(df_rc,use_container_width=True,hide_index=True)
            else:
                st.info('Sin resultados. Intenta un nombre diferente o haz Sync Deep primero.')

    elif query and not orch.search.built:
        st.warning('Índice no disponible. Haz Sync Mínimo o Sync Deep para activar la búsqueda.')

# ══════════════════════════════════════════════════════════════════════
# TAB 1: PREDICCIÓN MANUAL
# ══════════════════════════════════════════════════════════════════════
with tab_pred:
    if not orch.system.trained:
        st.info('Carga una liga primero desde el sidebar.')
    else:
        st.markdown('<div class="section-hdr">Predicción manual</div>',unsafe_allow_html=True)
        c1,cm,c2 = st.columns([5,1,5])
        with c1: home = st.selectbox('Local',teams,key='pm_h')
        with cm: st.markdown('<br><br><div style="text-align:center">vs</div>',unsafe_allow_html=True)
        with c2: away = st.selectbox('Visitante',[t for t in teams if t!=home],key='pm_a')
        if st.button('Predecir',type='primary',use_container_width=True,key='pm_btn'):
            with st.spinner('Calculando...'):
                analysis = orch.analyze_match(home,away)
            if analysis:
                pred = analysis['prediction']
                hs   = analysis['home_stats']
                as_  = analysis['away_stats']
                st.markdown('---')
                c1,cm,c2 = st.columns([4,2,4])
                with c1:
                    st.markdown(f"<div class='big-num'>{home}</div><div class='lbl'>Local</div>",unsafe_allow_html=True)
                    if hs: st.caption(f"PPP:{hs.get('total_PPP',0)} | GF/pj:{hs.get('avg_gf',0)} | Racha:{hs.get('streak','—')}")
                with cm:
                    st.markdown('<br><div style="text-align:center;font-size:1.3rem">VS</div>',unsafe_allow_html=True)
                    st.caption(f"λ {pred['lambda_home']} vs {pred['lambda_away']}")
                    if pred.get('has_xg'):
                        src = orch.state.get('xg_source','?')
                        st.markdown(f"<span class='xg-badge'>xG:{src}</span>",unsafe_allow_html=True)
                with c2:
                    st.markdown(f"<div class='big-num'>{away}</div><div class='lbl'>Visitante</div>",unsafe_allow_html=True)
                    if as_: st.caption(f"PPP:{as_.get('total_PPP',0)} | GF/pj:{as_.get('avg_gf',0)} | Racha:{as_.get('streak','—')}")
                st.markdown('---')
                c_h,c_d,c_a = st.columns(3)
                for col,code,lbl,prob,odd in [
                    (c_h,'H',f'Local {home}',pred['combined_H'],pred['odds_H']),
                    (c_d,'D','Empate',pred['combined_D'],pred['odds_D']),
                    (c_a,'A',f'Visita {away}',pred['combined_A'],pred['odds_A'])]:
                    css='winner' if code==pred['final_prediction'] else 'neutral'
                    sfx=' ✓' if code==pred['final_prediction'] else ''
                    with col:
                        st.markdown(
                            f"<div class='pred-card {css}'>{lbl}{sfx}"
                            f"<br><span style='font-size:1.9rem'>{prob:.1%}</span>"
                            f"<br><span style='color:#facc15'>Cuota: {odd}</span></div>",
                            unsafe_allow_html=True)
                vals=[pred['combined_H'],pred['combined_D'],pred['combined_A']]
                fig_b=go.Figure(go.Bar(
                    x=[f'Local {home}','Empate',f'Visita {away}'],
                    y=[v*100 for v in vals],text=[f'{v:.1%}' for v in vals],textposition='outside',
                    marker_color=['#4ade80' if v==max(vals) else '#475569' for v in vals]))
                fig_b.update_layout(yaxis=dict(range=[0,max(vals)*138]),height=240,
                    plot_bgcolor='#0f172a',paper_bgcolor='#0f172a',font_color='white',
                    showlegend=False,margin=dict(t=8,b=5))
                st.plotly_chart(fig_b,use_container_width=True)
                st.markdown('<div class="section-hdr">Mercados</div>',unsafe_allow_html=True)
                mc1,mc2 = st.columns(2)
                with mc1:
                    st.markdown('**Over / Under**')
                    for ol,op,ul,up in [('Over 1.5',pred['over_1_5'],'Under 1.5',pred['under_1_5']),
                                        ('Over 2.5',pred['over_2_5'],'Under 2.5',pred['under_2_5']),
                                        ('Over 3.5',pred['over_3_5'],'Under 3.5',pred['under_3_5'])]:
                        co,cu=st.columns(2)
                        co.metric(ol,f'{op:.1%}',f'({round(1/op,2) if op>0 else 99})')
                        cu.metric(ul,f'{up:.1%}',f'({round(1/up,2) if up>0 else 99})')
                with mc2:
                    st.markdown('**Otros**')
                    for lbl_,p_ in [('BTTS',pred['btts']),('No BTTS',pred['no_btts']),
                                    (f'Handi {home} -0.5',pred['asian_home']),
                                    (f'Handi {away} -0.5',pred['asian_away'])]:
                        od=round(1/p_,2) if p_>0 else 99
                        st.markdown(
                            f"<div style='display:flex;justify-content:space-between;border-bottom:1px solid #1e293b;padding:.3rem 0'>"
                            f"<span style='color:#94a3b8'>{lbl_}</span>"
                            f"<span><b>{p_:.1%}</b> <span style='color:#facc15;font-size:.8rem'>({od})</span></span></div>",
                            unsafe_allow_html=True)
                st.markdown('<div class="section-hdr">Mapa de calor</div>',unsafe_allow_html=True)
                M=np.array(pred['score_matrix'])[:7,:7]
                fig_h=px.imshow(M*100,labels=dict(x=f'Goles {away}',y=f'Goles {home}',color='%'),
                    x=[str(i) for i in range(7)],y=[str(i) for i in range(7)],
                    color_continuous_scale='Greens',text_auto='.1f',aspect='auto')
                fig_h.update_layout(plot_bgcolor='#0f172a',paper_bgcolor='#0f172a',
                    font_color='white',height=340,margin=dict(t=30,b=10))
                st.plotly_chart(fig_h,use_container_width=True)
                if pred.get('has_xg') and pred.get('xg_data'):
                    xd = pred['xg_data']
                    st.markdown('<div class="section-hdr">Expected Goals</div>',unsafe_allow_html=True)
                    x1,x2,x3,x4 = st.columns(4)
                    x1.metric(f'xGF/pj {home}',xd.get('home_xgF_pg','—'))
                    x2.metric(f'Suerte {home}',xd.get('home_luck','—'))
                    x3.metric(f'xGF/pj {away}',xd.get('away_xgF_pg','—'))
                    x4.metric(f'Suerte {away}',xd.get('away_luck','—'))
                    st.caption(f'xG source: {orch.state.get("xg_source","—")} | Lambda hibrido = 50% goles + 50% xG')

# ══════════════════════════════════════════════════════════════════════
# TAB 2: TABLA
# ══════════════════════════════════════════════════════════════════════
with tab_table:
    if not orch.system.trained: st.info('Carga una liga primero.')
    else:
        pred_lbl = comp_info.get('predict_label','')
        st.subheader(f'Tabla — {comp_label} {pred_lbl}')
        df_t = orch.system.league_table()
        if df_t.empty: st.warning('Sin datos.')
        else:
            st.dataframe(df_t,use_container_width=True)
            fig_p=px.bar(df_t.head(12),x='Equipo',y='PTS',color='PPP',color_continuous_scale='Blues',height=320)
            fig_p.update_layout(plot_bgcolor='#0f172a',paper_bgcolor='#0f172a',font_color='white')
            st.plotly_chart(fig_p,use_container_width=True)

# ══════════════════════════════════════════════════════════════════════
# TAB 3: ESTADÍSTICAS
# ══════════════════════════════════════════════════════════════════════
with tab_stats:
    if not orch.system.trained: st.info('Carga una liga primero.')
    else:
        st.subheader('Estadísticas de equipo')
        team_s = st.selectbox('Equipo',teams,key='t3')
        std_   = orch.system.team_full_stats(team_s)
        if std_:
            m1,m2,m3,m4,m5,m6,m7,m8 = st.columns(8)
            m1.metric('PJ',std_['total_PJ']); m2.metric('PG',std_['total_PG'])
            m3.metric('PE',std_['total_PE']); m4.metric('PP',std_['total_PP'])
            m5.metric('GF',std_['total_GF']); m6.metric('GC',std_['total_GC'])
            m7.metric('GD',f"{std_['total_GD']:+d}"); m8.metric('PPP',std_['total_PPP'])
            df_sp = pd.DataFrame([
                {'Cond.':'Local','PJ':std_['home_PJ'],'PG':std_['home_PG'],
                 'PE':std_['home_PE'],'PP':std_['home_PP'],
                 'GF':std_['home_GF'],'GC':std_['home_GC'],
                 'GD':std_['home_GD'],'PPP':std_['home_PPP']},
                {'Cond.':'Visita','PJ':std_['away_PJ'],'PG':std_['away_PG'],
                 'PE':std_['away_PE'],'PP':std_['away_PP'],
                 'GF':std_['away_GF'],'GC':std_['away_GC'],
                 'GD':std_['away_GD'],'PPP':std_['away_PPP']},
            ])
            st.dataframe(df_sp,use_container_width=True,hide_index=True)
            if std_.get('has_xg'):
                st.markdown('<div class="section-hdr">Expected Goals</div>',unsafe_allow_html=True)
                xq1,xq2,xq3,xq4,xq5,xq6 = st.columns(6)
                xq1.metric('xGF',std_.get('xGF','—'))
                xq2.metric('xGA',std_.get('xGA','—'))
                xq3.metric('xGF/pj',std_.get('xGF_per_game','—'))
                xq4.metric('xGA/pj',std_.get('xGA_per_game','—'))
                xq5.metric('Overperf Atq',std_.get('xg_overperf_att','—'),
                    help='Goles - xG. Positivo = sobrerendimiento (posible regresión)')
                xq6.metric('Luck Index',std_.get('luck_index','—'))
            sp1,sp2,sp3,sp4 = st.columns(4)
            sp1.metric('GF/pj',std_['avg_gf']); sp2.metric('GC/pj',std_['avg_gc'])
            sp3.metric('Portería 0',std_['clean_sheets']); sp4.metric('BTTS',std_['btts_count'])
            st.markdown(f'**Racha:** {std_.get("streak","—")}')

# ══════════════════════════════════════════════════════════════════════
# TAB 4: H2H
# ══════════════════════════════════════════════════════════════════════
with tab_h2h:
    if not orch.system.trained: st.info('Carga una liga primero.')
    else:
        st.subheader('Head-to-Head')
        h1,hm_,h2 = st.columns([5,1,5])
        with h1: hh = st.selectbox('Equipo A',teams,key='h2h_h')
        with hm_: st.markdown('<br><br><div style="text-align:center">VS</div>',unsafe_allow_html=True)
        with h2:  ha = st.selectbox('Equipo B',[t for t in teams if t!=hh],key='h2h_a')
        if st.button('Ver H2H',use_container_width=True):
            h2h_d = orch.system.h2h(hh,ha)
            s_    = h2h_d['summary']
            df_   = h2h_d['matches']
            if df_.empty: st.info('Sin enfrentamientos previos.')
            else:
                sm1,sm2,sm3,sm4,sm5 = st.columns(5)
                sm1.metric('Total',s_['total'])
                sm2.metric(f'Vic {hh}',s_.get(f'{hh}_wins',0))
                sm3.metric('Empates',s_['draws'])
                sm4.metric(f'Vic {ha}',s_.get(f'{ha}_wins',0))
                sm5.metric('Goles/pj',s_['avg_goals'])
                fig_p=go.Figure(go.Pie(
                    labels=[f'{hh}','Empate',f'{ha}'],
                    values=[s_.get(f'{hh}_wins',0),s_['draws'],s_.get(f'{ha}_wins',0)],
                    marker_colors=['#4ade80','#facc15','#f87171'],
                    textinfo='label+percent',hole=0.4))
                fig_p.update_layout(height=260,plot_bgcolor='#0f172a',
                    paper_bgcolor='#0f172a',font_color='white')
                st.plotly_chart(fig_p,use_container_width=True)
                df_s = df_[['competition_id','date','home_team','home_score','away_score','away_team']].copy()
                df_s['date'] = df_s['date'].dt.strftime('%Y-%m-%d')
                df_s['Score'] = df_s['home_score'].astype(int).astype(str)+'-'+df_s['away_score'].astype(int).astype(str)
                st.dataframe(df_s[['date','competition_id','home_team','Score','away_team']].rename(
                    columns={'date':'Fecha','competition_id':'Liga','home_team':'Local','away_team':'Visitante'}),
                    use_container_width=True,hide_index=True)

# ══════════════════════════════════════════════════════════════════════
# TAB 5: PRÓXIMOS PARTIDOS
# ══════════════════════════════════════════════════════════════════════
with tab_next:
    if not orch.system.trained: st.info('Carga una liga primero.')
    else:
        pred_lbl = comp_info.get('predict_label','')
        pred_id  = orch.system.stats.get('predict_id','')
        st.subheader(f'Pronósticos — {comp_label} {pred_lbl}')
        st.caption(f'Feed: {pred_id}')
        if st.button('Predecir todos los fixtures pendientes',
                     use_container_width=True,type='primary'):
            with st.spinner('Calculando...'):
                df_p = orch.system.predict_pending()
            if df_p.empty:
                st.warning('No hay partidos pendientes.')
            else:
                st.success(f'{len(df_p)} partidos pronosticados')
                show=['round','date','home_team','away_team','final_prediction',
                      'confidence','expected_score','combined_H','combined_D',
                      'combined_A','over_2_5','btts']
                df_s = df_p[[c for c in show if c in df_p.columns]].copy()
                df_s = df_s.rename(columns={'round':'Jornada','date':'Fecha',
                    'home_team':'Local','away_team':'Visitante','final_prediction':'Pred',
                    'confidence':'Conf','expected_score':'Marcador',
                    'combined_H':'P(L)','combined_D':'P(E)','combined_A':'P(V)',
                    'over_2_5':'O2.5','btts':'BTTS'})
                for c in ['P(L)','P(E)','P(V)','O2.5','BTTS']:
                    if c in df_s.columns:
                        df_s[c] = df_s[c].apply(lambda x:f'{x:.1%}' if isinstance(x,(float,int)) else x)
                st.dataframe(df_s,use_container_width=True,hide_index=True)
                st.download_button('Descargar CSV',df_p.to_csv(index=False).encode(),
                    f'pred_{pred_id}.csv','text/csv')

# ══════════════════════════════════════════════════════════════════════
# TAB 6: DATOS CRUDOS
# ══════════════════════════════════════════════════════════════════════
with tab_raw:
    if not orch.system.trained: st.info('Carga una liga primero.')
    else:
        st.subheader('Datos crudos')
        fc1,fc2,fc3 = st.columns(3)
        with fc1: tr_ = st.selectbox('Equipo',['(Todos)']+teams,key='raw_t')
        with fc2: rr_ = st.selectbox('Resultado',['(Todos)','H','D','A'],key='raw_r')
        with fc3: lr_ = st.slider('Max.',50,1000,300,50)
        df_r = orch.system.raw_matches(
            team=None if tr_=='(Todos)' else tr_,
            result=None if rr_=='(Todos)' else rr_).head(lr_)
        st.markdown(f'**{len(df_r)} partidos**')
        st.dataframe(df_r,use_container_width=True,hide_index=True)
        if not df_r.empty:
            st.download_button('Descargar',df_r.to_csv(index=False).encode(),'datos_crudos.csv','text/csv')

# ══════════════════════════════════════════════════════════════════════
# TAB 7: BACKTESTING
# ══════════════════════════════════════════════════════════════════════
with tab_bt:
    if not orch.system.trained: st.info('Carga una liga primero.')
    else:
        st.subheader('Backtesting del modelo')
        n_bt = st.slider('Últimos N',50,300,100,25)
        if st.button('Ejecutar backtest',use_container_width=True):
            with st.spinner('Evaluando...'): df_bt = orch.system.backtest(n_bt)
            if df_bt.empty: st.warning('Insuficientes datos.')
            else:
                acc = df_bt['correct'].mean()
                st.markdown(f'### Precisión: **{acc:.1%}** ({df_bt["correct"].sum()}/{len(df_bt)})')
                fig_bt=px.histogram(df_bt,x='conf',color='correct',nbins=20,height=270,
                    color_discrete_map={True:'#4ade80',False:'#f87171'},
                    title='Confianza vs Acierto')
                fig_bt.update_layout(plot_bgcolor='#0f172a',paper_bgcolor='#0f172a',font_color='white')
                st.plotly_chart(fig_bt,use_container_width=True)
                st.dataframe(df_bt.rename(columns={'date':'Fecha','home':'Local',
                    'away':'Visitante','real':'Real','pred':'Predicho',
                    'correct':'Correcto','conf':'Confianza'}),
                    use_container_width=True,hide_index=True)

# ══════════════════════════════════════════════════════════════════════
# TAB 8: xG ANÁLISIS
# ══════════════════════════════════════════════════════════════════════
with tab_xg:
    if not orch.system.trained: st.info('Carga una liga primero.')
    elif not orch.system.has_xg:
        xg_src = get_xg_source(slug)
        if xg_src:
            st.warning(f'xG disponible via {xg_src} para esta liga. Re-entrena para activarlo.')
        else:
            from fbref_scraper import coverage_summary
            st.warning('Esta liga no tiene cobertura xG. Ligas con xG disponible:')
            try:
                df_cov = coverage_summary()
                st.dataframe(df_cov[df_cov['xG_source']!='None'],use_container_width=True)
            except: st.info('coverage_summary no disponible.')
    else:
        pred_lbl = comp_info.get('predict_label','')
        src_label = orch.state.get('xg_source','?')
        st.subheader(f'xG Analysis — {comp_label} {pred_lbl}')
        st.caption(f'Fuente: {src_label} | Lambda = 50% goles + 50% xG')
        df_xg_t = orch.system.xg_league_table()
        if df_xg_t.empty: st.info('Sin datos xG suficientes.')
        else:
            st.markdown('#### Tabla de rendimiento esperado')
            st.dataframe(df_xg_t,use_container_width=True,hide_index=True)
            fig_xg=go.Figure()
            fig_xg.add_trace(go.Bar(name='xG Diff',x=df_xg_t['Equipo'],
                y=df_xg_t['xG_Diff'],marker_color='#3b82f6'))
            gd_real = df_xg_t['GF']-df_xg_t['GA']
            fig_xg.add_trace(go.Bar(name='GD Real',x=df_xg_t['Equipo'],
                y=gd_real,marker_color='#94a3b8',opacity=0.6))
            fig_xg.update_layout(barmode='group',height=300,
                plot_bgcolor='#0f172a',paper_bgcolor='#0f172a',font_color='white',
                legend=dict(orientation='h'),margin=dict(t=20,b=10))
            st.plotly_chart(fig_xg,use_container_width=True)
            st.markdown('#### Luck Index — ¿Quién ha tenido suerte?')
            st.caption('Cuadrante sup-derecha = resultados inflados por suerte → posible caída')
            fig_luck=go.Figure(go.Scatter(
                x=df_xg_t['OVP_Atq'],y=df_xg_t['OVP_Def'],
                mode='markers+text',text=df_xg_t['Equipo'],textposition='top center',
                marker=dict(size=11,color=df_xg_t['Suerte'],
                    colorscale='RdYlGn',showscale=True,
                    colorbar=dict(title='Suerte'))))
            fig_luck.add_hline(y=0,line_dash='dash',line_color='#475569')
            fig_luck.add_vline(x=0,line_dash='dash',line_color='#475569')
            fig_luck.update_layout(height=480,
                plot_bgcolor='#0f172a',paper_bgcolor='#0f172a',font_color='white',
                xaxis_title='Overperf Ataque (Goles-xG)',
                yaxis_title='Overperf Defensa (xGC-GC)',margin=dict(t=20))
            st.plotly_chart(fig_luck,use_container_width=True)
        st.markdown('---')
        st.markdown('#### xG por equipo')
        team_xg = st.selectbox('Equipo',teams,key='xg_t')
        xg_s = orch.system.xg_team_summary(team_xg)
        if xg_s:
            x1,x2,x3,x4,x5,x6 = st.columns(6)
            x1.metric('xGF',xg_s.get('xGF','—'))
            x2.metric('xGA',xg_s.get('xGA','—'))
            x3.metric('xGF/pj',xg_s.get('xGF_per_game','—'))
            x4.metric('xGA/pj',xg_s.get('xGA_per_game','—'))
            x5.metric('Overperf Atq',xg_s.get('overperf_attack','—'))
            x6.metric('Luck Index',xg_s.get('luck_index','—'))

# ══════════════════════════════════════════════════════════════════════
# TAB 9: CATÁLOGO
# ══════════════════════════════════════════════════════════════════════
with tab_cat:
    st.subheader('Catálogo de competiciones')
    st.caption(f"Actualizado: {catalog.get('last_updated','—')} | {catalog.get('total_comps',0)} competiciones")
    if orch.fd_universe:
        st.caption(f"FD Universe: {len(orch.fd_universe)} ligas descubiertas en último Sync Deep")
    search_cat = st.text_input('Buscar...',placeholder='bundesliga, serie, mls...',key='cat_q')
    try:
        from fbref_scraper import coverage_summary
        df_cov = coverage_summary()
        cov_map = {row['Liga']:row['xG_source'] for _,row in df_cov.iterrows()}
    except: cov_map = {}
    for reg,labels in regions.items():
        filtered = [l for l in labels if not search_cat or search_cat.lower() in l.lower()]
        if not filtered: continue
        with st.expander(f'**{reg}** ({len(filtered)} ligas)',expanded=bool(search_cat)):
            rows=[]
            for lbl in filtered:
                ci = get_comp_by_label(catalog,lbl)
                s_opts = get_season_options(ci)
                xg_src = cov_map.get(ci.get('name',''),'—')
                rows.append({'Competición':lbl,'Tipo':ci.get('season_type',''),
                    'xG':xg_src,'Temporadas':' | '.join(list(s_opts.keys())),
                    'Activa':ci.get('predict_label','')})
            st.dataframe(pd.DataFrame(rows),use_container_width=True,hide_index=True)