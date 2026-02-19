# app.py v4.3 — Indicadores de fuentes: FD + xG + SoccerStats + NormNames
import streamlit as st, pandas as pd, numpy as np, re
import plotly.graph_objects as go, plotly.express as px
from orchestrator import DataOrchestrator, get_xg_source
from catalog_builder import (
    load_or_refresh_catalog, get_regions, get_comp_by_label,
    get_season_options, get_train_defaults, get_predict_default_label,
)

st.set_page_config(page_title='Football Predictor Pro', page_icon='⚽',
                   layout='wide', initial_sidebar_state='expanded')

_CSS = '''<style>
.big-num{font-size:2.2rem;font-weight:900;text-align:center;line-height:1.1;}
.lbl{font-size:.78rem;color:#94a3b8;text-align:center;margin-top:2px;}
.pred-card{border-radius:14px;padding:1.1rem .8rem;text-align:center;font-weight:700;font-size:1rem;margin:.2rem 0;}
.winner{background:#14532d;color:#4ade80;border:2px solid #4ade80;}
.neutral{background:#1e293b;color:#94a3b8;border:2px solid #334155;}
.model-badge{background:#052e16;border:1px solid #16a34a;border-radius:8px;padding:.45rem .8rem;font-size:.82rem;color:#4ade80;margin-bottom:.5rem;}
.status-warn{background:#1c1917;border:1px solid #d97706;border-radius:10px;padding:.5rem 1rem;font-size:.85rem;color:#fbbf24;}
.match-card{background:#1e293b;border:1px solid #3b82f6;border-radius:12px;padding:.7rem 1.1rem;margin:.4rem 0;}
.section-hdr{font-size:.92rem;font-weight:700;color:#94a3b8;letter-spacing:.07em;text-transform:uppercase;margin:1rem 0 .4rem;border-left:3px solid #3b82f6;padding-left:.6rem;}
.xg-badge{display:inline-block;background:#1e3a5f;color:#93c5fd;font-size:.72rem;border-radius:20px;padding:2px 8px;margin-left:6px;font-weight:600;}
.mkt-row{display:flex;justify-content:space-between;border-bottom:1px solid #1e293b;padding:.28rem 0;font-size:.88rem;}
.src-badge{display:inline-block;border-radius:12px;padding:2px 9px;font-size:.73rem;margin:2px;font-weight:600;}
.score-cell{text-align:center;background:#1e293b;border-radius:8px;padding:.4rem .3rem;}
.src-panel{background:#0d1117;border:1px solid #21262d;border-radius:10px;padding:.5rem .8rem;margin-bottom:.6rem;}
</style>'''
st.markdown(_CSS, unsafe_allow_html=True)

for _k,_v in [('orch',None),('auto_on',False),('search_history',[]),('pending_pred',None)]:
    if _k not in st.session_state: st.session_state[_k]=_v
if st.session_state.orch is None: st.session_state.orch=DataOrchestrator()
orch=st.session_state.orch; catalog=orch.get_catalog(); regions=get_regions(catalog)

def friendly_comp(comp_id):
    if not comp_id: return comp_id or ''
    parts=comp_id.rsplit('-',1); slug=parts[0]; yr=parts[1] if len(parts)>1 else ''
    meta=catalog.get('competitions',{}).get(slug,{})
    name=meta.get('name',slug.replace('-',' ').title())
    stype=meta.get('season_type','single')
    if stype=='split' and yr.isdigit(): return f'{name} {yr}/{str(int(yr)+1)[2:]}'
    return f'{name} {yr}'

def odd_str(p): return f'{round(1/p,2)}' if p and p>0 else '99'
def conf_badge(label):
    cls={'Alta':'color:#4ade80','Media':'color:#fbbf24','Baja':'color:#f87171'}.get(label,'color:#fbbf24')
    return f"<span style='{cls};font-weight:800'>{label}</span>"

def push_history(home,away,comp_id):
    entry={'home':home,'away':away,'comp_id':comp_id,'label':f'{home} vs {away}'}
    h=[e for e in st.session_state.search_history if e['label']!=entry['label']]
    h.insert(0,entry); st.session_state.search_history=h[:6]

def top5_scores(matrix,max_g=6):
    M=np.array(matrix)[:max_g+1,:max_g+1]; rows=[]
    for i in range(max_g+1):
        for j in range(max_g+1): rows.append({'Marcador':f'{i}-{j}','Prob':float(M[i][j])})
    return sorted(rows,key=lambda x:-x['Prob'])[:5]

def mkt_row_html(label,prob):
    od=odd_str(prob)
    return (f"<div class='mkt-row'>"
            f"<span style='color:#94a3b8'>{label}</span>"
            f"<span><b>{prob:.1%}</b>"
            f"<span style='color:#facc15;font-size:.8rem'> ({od})</span>"
            f"</span></div>")

def sources_panel(pred=None):
    """Muestra indicadores LED de todas las fuentes de datos activas."""
    xg_ok = orch.state.get('has_xg', False)
    ss_ok = orch.state.get('has_ss', False)
    xg_nm = orch.state.get('xg_source','—') or '—'
    ss_n  = orch.state.get('ss_teams', 0)
    sources = [
        ('🗂 FixtureDownload',  True,    '#16a34a', 'Histórico de partidos y fixtures'),
        (f'📊 xG · {xg_nm}',   xg_ok,   '#3b82f6', 'Expected Goals por equipo'),
        (f'⚽ SoccerStats',     ss_ok,   '#8b5cf6', f'{ss_n} equipos · Over2.5% · BTTS% · Timing'),
        ('🔤 Name Normalizer',  True,    '#94a3b8', 'Alias entre fuentes'),
    ]
    badges = ''
    for lbl, active, color, tip in sources:
        bg  = '#0d1f0d' if active else '#1c1917'
        bdr = color if active else '#334155'
        txt = '#4ade80' if active else '#475569'
        ico = '✅' if active else '⚪'
        badges += (
            f"<span class='src-badge' style='background:{bg};border:1px solid {bdr};color:{txt}' title='{tip}'>"
            f"{ico} {lbl}</span>")
    st.markdown(
        f"<div class='src-panel'><span style='font-size:.72rem;color:#64748b;text-transform:uppercase;letter-spacing:.07em'>Fuentes de datos</span><br>"
        f"{badges}</div>", unsafe_allow_html=True)
    if ss_ok:
        st.caption(f'SoccerStats: {ss_n} equipos · λ = 40% hist + 40% xG + 20% SS')

def render_prediction(pred,home,away,hs,as_,compact=False):
    # [PARCHE 2] Fuentes activas siempre visibles
    sources_panel(pred)
    vals=[pred['combined_H'],pred['combined_D'],pred['combined_A']]
    fp=pred['final_prediction']
    c_h,c_d,c_a=st.columns(3)
    for col,code,lbl,prob,odd in [
        (c_h,'H',f'Local {home[:14]}',pred['combined_H'],pred['odds_H']),
        (c_d,'D','Empate',pred['combined_D'],pred['odds_D']),
        (c_a,'A',f'Visita {away[:14]}',pred['combined_A'],pred['odds_A'])]:
        css='winner' if code==fp else 'neutral'
        sfx=' ✓' if code==fp else ''
        with col:
            st.markdown(
                f"<div class='pred-card {css}'>"
                f"{lbl}{sfx}"
                f"<br><span style='font-size:1.9rem'>{prob:.1%}</span>"
                f"<br><span style='color:#facc15'>Cuota {odd}</span>"
                f"</div>",unsafe_allow_html=True)
    k1,k2,k3,k4=st.columns(4)
    k1.metric('Marcador',pred['expected_score'])
    k2.metric('Over 2.5',f"{pred['over_2_5']:.1%}")
    k3.metric('BTTS',f"{pred['btts']:.1%}")
    k4.markdown(
        f"<div style='text-align:center'>"
        f"<div style='font-size:.78rem;color:#94a3b8'>Confianza</div>"
        f"<div style='font-size:1.3rem'>{conf_badge(pred['confidence'])}</div>"
        f"</div>",unsafe_allow_html=True)
    if compact: return
    st.markdown('---')
    fig_b=go.Figure(go.Bar(
        x=[f'Local ({home[:10]})','Empate',f'Visita ({away[:10]})'],
        y=[v*100 for v in vals],text=[f'{v:.1%}' for v in vals],textposition='outside',
        marker_color=['#4ade80' if v==max(vals) else '#475569' for v in vals]))
    fig_b.update_layout(yaxis=dict(range=[0,max(vals)*140]),height=220,
        showlegend=False,plot_bgcolor='#0f172a',paper_bgcolor='#0f172a',
        font_color='white',margin=dict(t=8,b=5))
    st.plotly_chart(fig_b,use_container_width=True)
    if hs and as_:
        st.markdown("<div class='section-hdr'>Comparativa</div>",unsafe_allow_html=True)
        df_cmp=pd.DataFrame([
            {'':'PPP',       home[:16]:hs.get('total_PPP',0),   away[:16]:as_.get('total_PPP',0)},
            {'':'GF/pj',     home[:16]:hs.get('avg_gf',0),      away[:16]:as_.get('avg_gf',0)},
            {'':'GC/pj',     home[:16]:hs.get('avg_gc',0),      away[:16]:as_.get('avg_gc',0)},
            {'':'λ Poisson', home[:16]:pred['lambda_home'],      away[:16]:pred['lambda_away']},
        ])
        st.dataframe(df_cmp,use_container_width=True,hide_index=True)
    st.markdown("<div class='section-hdr'>Mercados</div>",unsafe_allow_html=True)
    m1,m2=st.columns(2)
    with m1:
        st.markdown('**Over / Under**')
        for ol,op,ul,up in [('Over 1.5',pred['over_1_5'],'Under 1.5',pred['under_1_5']),
            ('Over 2.5',pred['over_2_5'],'Under 2.5',pred['under_2_5']),
            ('Over 3.5',pred['over_3_5'],'Under 3.5',pred['under_3_5'])]:
            co,cu=st.columns(2)
            co.metric(ol,f'{op:.1%}',f'({odd_str(op)})')
            cu.metric(ul,f'{up:.1%}',f'({odd_str(up)})')
    with m2:
        st.markdown('**Otros mercados**')
        for lbl_,p_ in [('BTTS',pred['btts']),('No BTTS',pred['no_btts']),
                         (f'AH {home[:11]} -0.5',pred['asian_home']),
                         (f'AH {away[:11]} -0.5',pred['asian_away'])]:
            st.markdown(mkt_row_html(lbl_,p_),unsafe_allow_html=True)
    st.markdown("<div class='section-hdr'>Top 5 marcadores</div>",unsafe_allow_html=True)
    t5=top5_scores(pred['score_matrix']); cols5=st.columns(5)
    for i,row in enumerate(t5):
        with cols5[i]:
            st.markdown(
                f"<div class='score-cell'>"
                f"<div style='font-size:1.2rem;font-weight:800'>{row['Marcador']}</div>"
                f"<div style='color:#94a3b8;font-size:.8rem'>{row['Prob']:.1%}</div>"
                f"</div>",unsafe_allow_html=True)
    st.markdown("<div class='section-hdr'>Mapa de calor</div>",unsafe_allow_html=True)
    M=np.array(pred['score_matrix'])[:7,:7]
    fig_h=px.imshow(M*100,labels=dict(x=f'Goles {away}',y=f'Goles {home}',color='%'),
        x=[str(i) for i in range(7)],y=[str(i) for i in range(7)],
        color_continuous_scale='Greens',text_auto='.1f',aspect='auto')
    fig_h.update_layout(plot_bgcolor='#0f172a',paper_bgcolor='#0f172a',
        font_color='white',height=340,margin=dict(t=30,b=10))
    st.plotly_chart(fig_h,use_container_width=True)
    if pred.get('has_xg') and pred.get('xg_data'):
        xd=pred['xg_data']
        st.markdown("<div class='section-hdr'>Expected Goals</div>",unsafe_allow_html=True)
        x1,x2,x3,x4=st.columns(4)
        x1.metric(f'xGF/pj {home[:12]}',xd.get('home_xgF_pg','—'))
        x2.metric(f'Suerte {home[:12]}',xd.get('home_luck','—'),help='Positivo=suerte positiva')
        x3.metric(f'xGF/pj {away[:12]}',xd.get('away_xgF_pg','—'))
        x4.metric(f'Suerte {away[:12]}',xd.get('away_luck','—'))
        src=orch.state.get('xg_source','—')
        st.caption(f'xG:{src} · λ=50% goles + 50% xG')
    # [NUEVO] SoccerStats data en predicción
    if orch.state.get('has_ss') and pred.get('ss_data'):
        sd=pred['ss_data']
        st.markdown("<div class='section-hdr'>SoccerStats</div>",unsafe_allow_html=True)
        s1,s2,s3,s4=st.columns(4)
        s1.metric(f'Over2.5% {home[:10]}',f"{sd.get('home_over25',0):.0f}%")
        s2.metric(f'BTTS% {home[:10]}',   f"{sd.get('home_btts',0):.0f}%")
        s3.metric(f'Over2.5% {away[:10]}',f"{sd.get('away_over25',0):.0f}%")
        s4.metric(f'BTTS% {away[:10]}',   f"{sd.get('away_btts',0):.0f}%")
        st.caption('Fuente: soccerstats.com · Actualizado 24h · λ=40% hist+40% xG+20% SS')

# ════ SIDEBAR ════
with st.sidebar:
    st.markdown('## ⚽ Football Predictor Pro')
    st.caption(f"v4.3 · {catalog.get('total_comps',0)} ligas")
    if orch.system.trained:
        ss   =orch.system.stats
        c_meta=catalog.get('competitions',{}).get(ss.get('comp_slug',''),{})
        cname =c_meta.get('name',ss.get('comp_slug','—'))
        stype_=c_meta.get('season_type','single')
        yr_   =str(ss.get('predict_season',''))
        s_lbl =(f"{yr_}/{str(int(yr_)+1)[2:]}" if stype_=='split' and yr_.isdigit() else yr_)
        xg_ok =orch.state.get('has_xg',False)
        ss_ok =orch.state.get('has_ss',False)
        xg_nm =orch.state.get('xg_source','—') or '—'
        ss_n  =orch.state.get('ss_teams',0)
        cv_v  =ss.get('cv_score',0)
        played=ss.get('played_current',0)
        # Función inline para generar chip de fuente
        def _src_chip(label, active, color):
            bg  = '#0d1f0d' if active else '#1c1917'
            bdr = color    if active else '#334155'
            txt = '#4ade80' if active else '#475569'
            ico = '✅' if active else '⚪'
            return (f"<span style='background:{bg};border:1px solid {bdr};"
                    f"color:{txt};border-radius:12px;padding:1px 7px;"
                    f"font-size:.71rem;margin:2px;display:inline-block'>"
                    f"{ico} {label}</span>")
        chips = (
            _src_chip('FD',              True,   '#16a34a') +
            _src_chip(f'xG:{xg_nm}',    xg_ok,  '#3b82f6') +
            _src_chip(f'SS:{ss_n}eq',   ss_ok,  '#8b5cf6') +
            _src_chip('NormNames',       True,   '#94a3b8')
        )
        st.markdown(
            f"<div class='model-badge'>"
            f"🎯 <b>Modelo activo</b><br>"
            f"{cname} {s_lbl}&nbsp;"
            f"<span style='color:#86efac'>CV:{cv_v:.0%} · {played}pj</span><br>"
            f"<div style='margin-top:.35rem'>{chips}</div>"
            f"</div>",unsafe_allow_html=True)
        ls=orch.state.get('last_sync_time','')
        if ls: st.caption(f'Última sync: {ls}')
    else:
        st.markdown("<div class='status-warn'>⚠️ Sin modelo. Sync Mínimo para empezar.</div>",unsafe_allow_html=True)
    st.markdown('---')
    st.markdown('### 🔄 Sincronización')
    c1,c2=st.columns(2)
    with c1: sync_min_btn=st.button('⚡ Sync Mínimo',use_container_width=True)
    with c2: sync_deep_bg=st.button('🌍 Sync Deep',use_container_width=True)
    force_dl=st.checkbox('Forzar re-descarga',key='force_dl')
    if sync_deep_bg:
        if not orch._deep_running: orch.sync_deep_background(force=force_dl); st.info('🌍 Sync Deep iniciado.')
        else: st.warning('Ya hay un Sync Deep en curso.')
    st.markdown('---')
    st.markdown('### 🏆 Liga para entrenar')
    region_list=list(regions.keys())
    region=st.selectbox('Región',region_list,key='sb_region')
    comp_labels=regions.get(region,[])
    comp_label=st.selectbox('Competición',comp_labels,key='sb_comp')
    comp_info=get_comp_by_label(catalog,comp_label)
    slug=comp_info.get('slug','')
    season_opts=get_season_options(comp_info)
    train_defs=get_train_defaults(comp_info)
    predict_def=get_predict_default_label(comp_info)
    stype_sel=comp_info.get('season_type','single')
    xg_src_sel=get_xg_source(slug)
    xg_lbl=f'🟢 {xg_src_sel}' if xg_src_sel else '⚪ Sin cobertura'
    st.caption(f'Tipo: {stype_sel} · xG: {xg_lbl}')
    train_labels=st.multiselect('Temporadas de entrenamiento',
        options=list(season_opts.keys()),
        default=[l for l in train_defs if l in season_opts],key='tr_lbl')
    if train_labels:
        n_teams=comp_info.get('num_teams',18)
        n_match=len(train_labels)*(n_teams*(n_teams-1))
        t_est=max(10,round(len(train_labels)*4.5))
        st.caption(f'≈ {n_match:,} partidos · ~{t_est}s')
    predict_lbl=st.selectbox('Temporada activa',options=list(season_opts.keys()),
        index=list(season_opts.keys()).index(predict_def) if predict_def in season_opts else 0,
        key='pred_lbl')
    if sync_min_btn:
        if not train_labels: st.error('Selecciona al menos una temporada.')
        else:
            train_yrs=[season_opts[l] for l in train_labels if l in season_opts]
            pred_yr=season_opts.get(predict_lbl,0)
            pb=st.progress(0.0); sm=st.empty()
            def _pf(msg,v): sm.caption(msg); pb.progress(float(v))
            ok,msg=orch.sync_minimum(slug,train_yrs,pred_yr,force=force_dl,progress_fn=_pf)
            pb.empty(); sm.empty()
            if ok: st.success(f'Listo: {msg}'); st.rerun()
            else:  st.error(f'Error: {msg}')
    st.markdown('---')
    st.markdown('### ⚙️ Auto-update')
    auto_on=st.toggle('Activar',value=st.session_state.auto_on)
    if auto_on!=st.session_state.auto_on: st.session_state.auto_on=auto_on; st.rerun()
    if orch.system.trained:
        if st.button('Chequear nuevos resultados',use_container_width=True):
            try:
                from auto_updater import AutoUpdater
                upd=AutoUpdater(orch.system,orch.system.comp_ids)
                r=upd.check_all()
                nm_=r.get('new_matches',0)
                orch.search.build(orch.system.df_proc,catalog)
                st.success(f'{nm_} nuevos partidos' if nm_>0 else '✅ Al día')
            except Exception as e: st.error(str(e))

st.title('⚽ Football Predictor Pro')
st.caption('Búsqueda global · FD + xG + SoccerStats + NormNames · Multi-liga')
teams=orch.get_teams()
(tab_search,tab_pred,tab_table,tab_stats,tab_h2h,tab_next,tab_raw,tab_bt,tab_xg,tab_cat)=st.tabs([
    '🔍 Buscar & Analizar','⚡ Predicción Manual','📊 Tabla','👤 Estadísticas',
    '⚔️ H2H','📅 Próximos','📁 Datos Crudos','🔬 Backtesting','📈 xG Análisis','🌐 Catálogo'])

with tab_search:
    st.markdown("<div class='section-hdr'>Búsqueda Universal</div>",unsafe_allow_html=True)
    st.caption('Equipo, liga o partido · Ej: Bayern vs Dortmund · Chelsea · Premier League')
    # [PARCHE 3] — Panel de fuentes visible siempre en el tab principal
    if orch.system.trained: sources_panel()
    if st.session_state.search_history:
        chip_cols=st.columns(len(st.session_state.search_history))
        for i,h_e in enumerate(st.session_state.search_history):
            with chip_cols[i]:
                if st.button(f"↩ {h_e['home'][:8]} vs {h_e['away'][:8]}",key=f'chip_{i}',use_container_width=True):
                    st.session_state['prefill_search']=h_e['label']; st.rerun()
    prefill=st.session_state.pop('prefill_search','')
    query=st.text_input('',value=prefill,placeholder='🔍  Bayern Munich, Chelsea vs Arsenal ...',
                         key='global_search',label_visibility='collapsed')
    query=query.strip("\"' ").strip() if query else ''
    if orch._deep_running:
        s2=orch.get_status(); prog=float(s2.get('deep_progress',0.0))
        st.markdown(f"<div class='status-run'>⚡ Sync Deep: {s2.get('deep_message','')} — {prog:.0%}</div>",
            unsafe_allow_html=True)
        st.progress(prog)
    if not orch.search.built and not query:
        st.info('💡 Haz Sync Mínimo en el sidebar para indexar la liga activa, o Sync Deep para todas las ligas.')
    if query and orch.search.built:
        is_match_q=bool(re.search(r'\bvs\.?\b|\s+-\s+',query,re.I))
        if is_match_q:
            parts=re.split(r'\s+vs\.?\s+|\s+-\s+',query.strip(),maxsplit=1,flags=re.I)
            home_q=parts[0].strip()
            away_q=parts[1].strip() if len(parts)>1 else ''
            mi=orch.search.find_match(home_q,away_q)
            if mi:
                comp_fr=friendly_comp(mi['comp_id'])
                st.success(f"Partido encontrado: **{mi['home']}** vs **{mi['away']}**")
                # [PARCHE 3] caption inline con fuentes activas
                xg_ok_=orch.state.get('has_xg',False)
                ss_ok_=orch.state.get('has_ss',False)
                src_icons=(['🗂 FD']+([f"📊 xG:{orch.state.get('xg_source','?')}"] if xg_ok_ else [])
                           +([f"⚽ SS:{orch.state.get('ss_teams',0)}eq"] if ss_ok_ else [])+['🔤 NN'])
                st.caption('Fuentes: '+' · '.join(src_icons))
                st.markdown(
                    f"<div class='match-card'>"
                    f"<b>{mi['home']}</b>"
                    f"<span style='color:#64748b;padding:0 .7rem'>vs</span>"
                    f"<b>{mi['away']}</b>"
                    f"<span style='color:#3b82f6;margin-left:1rem'>{comp_fr}</span>"
                    f"<span style='color:#94a3b8;margin-left:.6rem;font-size:.82rem'>"
                    f"{'🕓 Pendiente' if not mi['played'] else '✅ Jugado'}"
                    f"</span></div>",unsafe_allow_html=True)
                is_same=(orch.system.trained and
                    mi['comp_id'].startswith(orch.system.stats.get('comp_slug','__')))
                if is_same:
                    if st.button('⚡ Analizar este partido ahora',type='primary',use_container_width=True):
                        push_history(mi['home'],mi['away'],mi['comp_id'])
                        with st.spinner('Calculando Poisson + GBM + xG + SS...'):
                            analysis=orch.analyze_match(mi['home'],mi['away'])
                        if analysis:
                            st.session_state.pending_pred={'home':mi['home'],'away':mi['away'],'analysis':analysis}
                            pred=analysis['prediction']; hs=analysis.get('home_stats'); as_=analysis.get('away_stats')
                            st.markdown(f"#### {mi['home']} vs {mi['away']} · {comp_fr}")
                            render_prediction(pred,mi['home'],mi['away'],hs,as_,compact=True)
                            st.info('💡 Tab Predicción Manual para análisis completo (heatmap, todos los mercados).')
                else:
                    slug_q=mi['comp_id'].rsplit('-',1)[0]
                    yr_q=int(mi['comp_id'].rsplit('-',1)[-1])
                    st.warning(f"Liga {friendly_comp(mi['comp_id'])} no entrenada.")
                    if st.button(f"📥 Cargar {friendly_comp(mi['comp_id'])} ahora",use_container_width=True):
                        pb2=st.progress(0.0); sm2=st.empty()
                        def _pf2(msg,v): sm2.caption(msg); pb2.progress(float(v))
                        ok2,msg2=orch.on_demand_load(slug_q,[yr_q-2,yr_q-1,yr_q],yr_q,progress_fn=_pf2)
                        pb2.empty(); sm2.empty()
                        if ok2: st.success('Listo'); st.rerun()
                        else: st.error(msg2)
            else:
                st.warning('Partido no encontrado. Prueba Sync Deep o revisa los nombres.')
                for lbl_,q_ in [('Local',home_q),('Visitante',away_q)]:
                    hits=orch.search.search_teams(q_,top_n=3)
                    if hits: st.caption(f"Similares {lbl_}: {', '.join(h['team'] for h in hits)}")
        else:
            team_hits=orch.search.search_teams(query,top_n=10)
            if team_hits:
                st.markdown(f'**{len(team_hits)} resultados para:** `{query}`')
                for hit in team_hits:
                    cfr=friendly_comp(hit['comp_id'])
                    st.markdown(
                        f"<div style='background:#1e293b;border:1px solid #334155;border-radius:8px;padding:.45rem .8rem;margin:.2rem 0;'>"
                        f"⚽ <b>{hit['team']}</b> — <span style='color:#3b82f6'>{cfr}</span>"
                        f"<span style='color:#475569;font-size:.78rem'> ({int(hit['score']*100)}%)</span>"
                        f"</div>",unsafe_allow_html=True)
                best=team_hits[0]['team']
                df_up=orch.search.upcoming_matches(best,limit=5)
                df_rc=orch.search.search_matches(query,top_n=5)
                if not df_up.empty: st.markdown(f'**Próximos — {best}:**'); st.dataframe(df_up,use_container_width=True,hide_index=True)
                if not df_rc.empty: st.markdown(f'**Últimos — {best}:**'); st.dataframe(df_rc,use_container_width=True,hide_index=True)
            else: st.info('Sin resultados. Prueba otro nombre o Sync Deep.')
    elif query and not orch.search.built: st.warning('Índice no construido. Haz Sync Mínimo primero.')

with tab_pred:
    if not orch.system.trained: st.info('Carga una liga desde el sidebar.')
    else:
        st.markdown("<div class='section-hdr'>Predicción Manual</div>",unsafe_allow_html=True)
        pp=st.session_state.get('pending_pred')
        def_home=pp['home'] if pp and pp.get('home') in teams else (teams[0] if teams else '')
        def_away=pp['away'] if pp and pp.get('away') in teams else (teams[1] if len(teams)>1 else '')
        if pp: st.info(f"⚡ Cargado desde búsqueda: **{pp['home']}** vs **{pp['away']}**")
        c1,cm,c2=st.columns([5,1,5])
        with c1: home=st.selectbox('Local',teams,index=teams.index(def_home) if def_home in teams else 0,key='pm_h')
        with cm: st.markdown('<br><br><div style="text-align:center">vs</div>',unsafe_allow_html=True)
        with c2:
            away_opts=[t for t in teams if t!=home]
            def_a2=def_away if def_away in away_opts else (away_opts[0] if away_opts else '')
            away=st.selectbox('Visitante',away_opts,index=away_opts.index(def_a2) if def_a2 in away_opts else 0,key='pm_a')
        auto_run=pp is not None
        if st.button('Predecir',type='primary',use_container_width=True,key='pm_btn') or auto_run:
            if auto_run and pp and 'analysis' in pp:
                analysis=pp['analysis']; st.session_state.pending_pred=None
            else:
                with st.spinner('Calculando...'): analysis=orch.analyze_match(home,away)
            if analysis:
                pred=analysis['prediction']; hs=analysis.get('home_stats'); as_=analysis.get('away_stats')
                push_history(home,away,orch.system.predict_comp_id)
                h1,hm,h2=st.columns([4,2,4])
                with h1:
                    st.markdown(f"<div class='big-num'>{home}</div>"
                                f"<div class='lbl'>Local · λ {pred['lambda_home']}</div>",unsafe_allow_html=True)
                    if hs: st.caption(f"PPP {hs.get('total_PPP',0)} · GF {hs.get('avg_gf',0)}/pj · Racha {hs.get('streak','—')}")
                with hm:
                    xg_flag=f"<span class='xg-badge'>xG·{orch.state.get('xg_source','?')}</span>" if pred.get('has_xg') else ''
                    ss_flag="<span class='xg-badge' style='background:#1a0a3f;color:#c4b5fd'>SS</span>" if orch.state.get('has_ss') else ''
                    st.markdown(f"<br><div style='text-align:center'>"
                                f"<div style='font-size:1.3rem'>VS</div>"
                                f"{xg_flag}{ss_flag}</div>",unsafe_allow_html=True)
                with h2:
                    st.markdown(f"<div class='big-num'>{away}</div>"
                                f"<div class='lbl'>Visitante · λ {pred['lambda_away']}</div>",unsafe_allow_html=True)
                    if as_: st.caption(f"PPP {as_.get('total_PPP',0)} · GF {as_.get('avg_gf',0)}/pj · Racha {as_.get('streak','—')}")
                st.markdown('---')
                render_prediction(pred,home,away,hs,as_,compact=False)

with tab_table:
    if not orch.system.trained: st.info('Carga una liga primero.')
    else:
        comp_fn=friendly_comp(orch.system.predict_comp_id)
        st.subheader(f'Tabla — {comp_fn}')
        df_t=orch.system.league_table()
        if df_t.empty: st.warning('Sin datos suficientes.')
        else:
            if 'GF' in df_t and 'PJ' in df_t:
                played_r=df_t['PJ'].sum()//2
                total_g=df_t['GF'].sum()
                avg_g=round(total_g/played_r,2) if played_r else 0
                s1,s2,s3=st.columns(3)
                s1.metric('Partidos jugados',played_r); s2.metric('Goles totales',total_g); s3.metric('Goles/partido',avg_g)
            st.dataframe(df_t,use_container_width=True)
            fig_p=px.bar(df_t.head(12),x='Equipo',y='PTS',color='PPP',color_continuous_scale='Blues',height=300)
            fig_p.update_layout(plot_bgcolor='#0f172a',paper_bgcolor='#0f172a',font_color='white')
            st.plotly_chart(fig_p,use_container_width=True)

with tab_stats:
    if not orch.system.trained: st.info('Carga una liga primero.')
    else:
        st.subheader(f'Estadísticas — {friendly_comp(orch.system.predict_comp_id)}')
        team_s=st.selectbox('Equipo',teams,key='t3')
        std_=orch.system.team_full_stats(team_s)
        if std_:
            m1,m2,m3,m4,m5,m6,m7,m8=st.columns(8)
            m1.metric('PJ',std_['total_PJ']); m2.metric('PG',std_['total_PG'])
            m3.metric('PE',std_['total_PE']); m4.metric('PP',std_['total_PP'])
            m5.metric('GF',std_['total_GF']); m6.metric('GC',std_['total_GC'])
            m7.metric('GD',f"{std_['total_GD']:+d}"); m8.metric('PPP',std_['total_PPP'])
            df_sp=pd.DataFrame([
                {'Cond.':'Local','PJ':std_['home_PJ'],'PG':std_['home_PG'],'PE':std_['home_PE'],'PP':std_['home_PP'],'GF':std_['home_GF'],'GC':std_['home_GC'],'GD':std_['home_GD'],'PPP':std_['home_PPP']},
                {'Cond.':'Visita','PJ':std_['away_PJ'],'PG':std_['away_PG'],'PE':std_['away_PE'],'PP':std_['away_PP'],'GF':std_['away_GF'],'GC':std_['away_GC'],'GD':std_['away_GD'],'PPP':std_['away_PPP']},
            ])
            st.dataframe(df_sp,use_container_width=True,hide_index=True)
            if std_.get('has_xg'):
                st.markdown("<div class='section-hdr'>Expected Goals</div>",unsafe_allow_html=True)
                x1,x2,x3,x4,x5,x6=st.columns(6)
                x1.metric('xGF',std_.get('xGF','—')); x2.metric('xGA',std_.get('xGA','—'))
                x3.metric('xGF/pj',std_.get('xGF_per_game','—')); x4.metric('xGA/pj',std_.get('xGA_per_game','—'))
                x5.metric('Overperf Atq',std_.get('xg_overperf_att','—'))
                x6.metric('Luck Index',std_.get('luck_index','—'))
            sp1,sp2,sp3,sp4=st.columns(4)
            sp1.metric('GF/pj',std_['avg_gf']); sp2.metric('GC/pj',std_['avg_gc'])
            sp3.metric('Clean sheets',std_['clean_sheets']); sp4.metric('BTTS',std_['btts_count'])
            st.markdown(f"**Racha:** {std_.get('streak','—')} · **Últimos 5:** {' '.join(std_.get('last5',[]) or [])}")
            # [PARCHE 4] SoccerStats por equipo
            if orch.state.get('has_ss') and std_.get('ss_home_gf_pg') is not None:
                st.markdown("<div class='section-hdr'>SoccerStats</div>",unsafe_allow_html=True)
                ss1,ss2,ss3,ss4,ss5,ss6=st.columns(6)
                ss1.metric('GF/pj SS',std_.get('ss_home_gf_pg','—'))
                ss2.metric('GA/pj SS',std_.get('ss_home_ga_pg','—'))
                ss3.metric('Over2.5%',f"{std_.get('ss_home_over25_pct',0):.0f}%" if std_.get('ss_home_over25_pct') else '—')
                ss4.metric('BTTS%',   f"{std_.get('ss_home_btts_pct',0):.0f}%"   if std_.get('ss_home_btts_pct') else '—')
                ss5.metric('Goles tarde%',std_.get('late_goals_pct','—'))
                ss6.metric('Fuente','⚽ SS',help='soccerstats.com · cache 24h')
                st.caption(f'soccerstats.com · Caché: fd_cache/soccerstats/ · Actualizado cada 24h')
        else: st.info('Sin estadísticas para este equipo.')

with tab_h2h:
    if not orch.system.trained: st.info('Carga una liga primero.')
    else:
        st.subheader('Head-to-Head')
        h1,hm_,h2=st.columns([5,1,5])
        with h1: hh=st.selectbox('Equipo A',teams,key='h2h_h')
        with hm_: st.markdown('<br><br><div style="text-align:center">VS</div>',unsafe_allow_html=True)
        with h2: ha=st.selectbox('Equipo B',[t for t in teams if t!=hh],key='h2h_a')
        if st.button('Ver H2H',use_container_width=True):
            h2h_d=orch.system.h2h(hh,ha); s_=h2h_d['summary']; df_=h2h_d['matches']
            if df_.empty: st.info('Sin enfrentamientos previos.')
            else:
                sm1,sm2,sm3,sm4,sm5=st.columns(5)
                sm1.metric('Total',s_['total']); sm2.metric(f'Vic {hh[:10]}',s_.get(f'{hh}_wins',0))
                sm3.metric('Empates',s_['draws']); sm4.metric(f'Vic {ha[:10]}',s_.get(f'{ha}_wins',0))
                sm5.metric('Goles/pj',s_['avg_goals'])
                fig_p=go.Figure(go.Pie(labels=[f'{hh}','Empate',f'{ha}'],
                    values=[s_.get(f'{hh}_wins',0),s_['draws'],s_.get(f'{ha}_wins',0)],
                    marker_colors=['#4ade80','#facc15','#f87171'],textinfo='label+percent',hole=0.4))
                fig_p.update_layout(height=260,plot_bgcolor='#0f172a',paper_bgcolor='#0f172a',font_color='white')
                st.plotly_chart(fig_p,use_container_width=True)
                df_show=df_[['date','home_team','home_score','away_score','away_team']].copy()
                df_show['date']=df_show['date'].dt.strftime('%Y-%m-%d')
                df_show['Score']=df_show['home_score'].astype(int).astype(str)+'-'+df_show['away_score'].astype(int).astype(str)
                st.dataframe(df_show[['date','home_team','Score','away_team']].rename(
                    columns={'date':'Fecha','home_team':'Local','away_team':'Visitante'}),use_container_width=True,hide_index=True)

with tab_next:
    if not orch.system.trained: st.info('Carga una liga primero.')
    else:
        pred_id=orch.system.stats.get('predict_id','')
        st.subheader(f'Pronósticos — {friendly_comp(pred_id)}')
        if st.button('Predecir todos los fixtures pendientes',use_container_width=True,type='primary'):
            with st.spinner('Calculando...'): df_p=orch.system.predict_pending()
            if df_p.empty: st.warning('No hay partidos pendientes.')
            else:
                st.success(f'{len(df_p)} partidos pronosticados')
                show=['round','date','home_team','away_team','final_prediction','confidence',
                      'expected_score','combined_H','combined_D','combined_A','over_2_5','btts']
                df_s=df_p[[c for c in show if c in df_p.columns]].rename(columns={
                    'round':'Jorn','date':'Fecha','home_team':'Local','away_team':'Visitante',
                    'final_prediction':'Pred','confidence':'Conf','expected_score':'Marcador',
                    'combined_H':'P(L)','combined_D':'P(E)','combined_A':'P(V)',
                    'over_2_5':'O2.5','btts':'BTTS'})
                for c in ['P(L)','P(E)','P(V)','O2.5','BTTS']:
                    if c in df_s.columns: df_s[c]=df_s[c].apply(lambda x:f'{x:.1%}' if isinstance(x,(float,int)) else x)
                st.dataframe(df_s,use_container_width=True,hide_index=True)
                c1,c2=st.columns(2)
                with c1: st.download_button('⬇ CSV',df_p.to_csv(index=False).encode(),f'pred_{pred_id}.csv','text/csv')
                with c2: st.download_button('⬇ JSON',df_p.to_json(orient='records',indent=2).encode(),f'pred_{pred_id}.json','application/json')

with tab_raw:
    if not orch.system.trained: st.info('Carga una liga primero.')
    else:
        st.subheader('Datos crudos')
        fc1,fc2,fc3=st.columns(3)
        with fc1: tr_=st.selectbox('Equipo',['(Todos)']+teams,key='raw_t')
        with fc2: rr_=st.selectbox('Resultado',['(Todos)','H','D','A'],key='raw_r')
        with fc3: lr_=st.slider('Max.',50,1000,300,50)
        df_r=orch.system.raw_matches(team=None if tr_=='(Todos)' else tr_,
            result=None if rr_=='(Todos)' else rr_).head(lr_)
        st.markdown(f'**{len(df_r)} partidos**')
        st.dataframe(df_r,use_container_width=True,hide_index=True)
        if not df_r.empty: st.download_button('⬇ Descargar',df_r.to_csv(index=False).encode(),'datos_crudos.csv','text/csv')

with tab_bt:
    if not orch.system.trained: st.info('Carga una liga primero.')
    else:
        st.subheader('Backtesting')
        n_bt=st.slider('Últimos N partidos',50,300,100,25)
        if st.button('Ejecutar backtest',use_container_width=True):
            with st.spinner('Evaluando...'): df_bt=orch.system.backtest(n_bt)
            if df_bt.empty: st.warning('Insuficientes datos.')
            else:
                acc=df_bt['correct'].mean()
                prec_h=df_bt[df_bt['real']=='H']['correct'].mean() if (df_bt['real']=='H').any() else 0
                prec_d=df_bt[df_bt['real']=='D']['correct'].mean() if (df_bt['real']=='D').any() else 0
                prec_a=df_bt[df_bt['real']=='A']['correct'].mean() if (df_bt['real']=='A').any() else 0
                bt1,bt2,bt3,bt4=st.columns(4)
                bt1.metric('Precisión global',f'{acc:.1%}')
                bt2.metric('Precisión H',f'{prec_h:.1%}')
                bt3.metric('Precisión D',f'{prec_d:.1%}')
                bt4.metric('Precisión A',f'{prec_a:.1%}')
                st.caption(f'CV:{orch.system.ml.cv_score:.1%}')
                fig_bt=px.histogram(df_bt,x='conf',color='correct',nbins=20,height=270,
                    color_discrete_map={True:'#4ade80',False:'#f87171'},title='Confianza vs Acierto')
                fig_bt.update_layout(plot_bgcolor='#0f172a',paper_bgcolor='#0f172a',font_color='white')
                st.plotly_chart(fig_bt,use_container_width=True)
                if 'real' in df_bt and 'pred' in df_bt:
                    from sklearn.metrics import confusion_matrix
                    labels_cm=['H','D','A']
                    cm=confusion_matrix(df_bt['real'],df_bt['pred'],labels=labels_cm)
                    fig_cm=px.imshow(cm,x=labels_cm,y=labels_cm,text_auto=True,
                        color_continuous_scale='Blues',height=280,
                        labels=dict(x='Predicho',y='Real',color='Partidos'))
                    fig_cm.update_layout(plot_bgcolor='#0f172a',paper_bgcolor='#0f172a',font_color='white')
                    st.plotly_chart(fig_cm,use_container_width=True)
                st.dataframe(df_bt.rename(columns={'date':'Fecha','home':'Local','away':'Visitante',
                    'real':'Real','pred':'Predicho','correct':'Correcto','conf':'Confianza'}),
                    use_container_width=True,hide_index=True)

with tab_xg:
    if not orch.system.trained: st.info('Carga una liga primero.')
    elif not orch.system.has_xg:
        xg_avail=get_xg_source(slug)
        if xg_avail:
            st.info(f'xG disponible via {xg_avail}. Actívalo re-entrenando con Sync Mínimo (forzar re-descarga).')
        else: st.warning('Liga sin cobertura xG.')
    else:
        comp_fn2=friendly_comp(orch.system.predict_comp_id)
        src_lbl=orch.state.get('xg_source','?')
        st.subheader(f'xG Análisis — {comp_fn2}')
        st.caption(f'Fuente: {src_lbl} · λ=50% goles + 50% xG')
        df_xg_t=orch.system.xg_league_table()
        if not df_xg_t.empty:
            st.dataframe(df_xg_t,use_container_width=True,hide_index=True)
            fig_luck=go.Figure(go.Scatter(
                x=df_xg_t['OVP_Atq'],y=df_xg_t['OVP_Def'],
                mode='markers+text',text=df_xg_t['Equipo'],textposition='top center',
                marker=dict(size=11,color=df_xg_t['Suerte'],colorscale='RdYlGn',showscale=True)))
            fig_luck.add_hline(y=0,line_dash='dash',line_color='#475569')
            fig_luck.add_vline(x=0,line_dash='dash',line_color='#475569')
            fig_luck.update_layout(height=480,plot_bgcolor='#0f172a',paper_bgcolor='#0f172a',font_color='white')
            st.plotly_chart(fig_luck,use_container_width=True)

with tab_cat:
    st.subheader('Catálogo')
    sc=st.text_input('Buscar...',placeholder='bundesliga, serie, mls...',key='cat_q')
    try:
        from fbref_scraper import coverage_summary
        df_cov=coverage_summary()
        cov_map={row['Liga']:row['xG_source'] for _,row in df_cov.iterrows()}
    except: cov_map={}
    for reg,labels in regions.items():
        filtered=[l for l in labels if not sc or sc.lower() in l.lower()]
        if not filtered: continue
        with st.expander(f'**{reg}** ({len(filtered)} ligas)',expanded=bool(sc)):
            rows=[]
            for lbl in filtered:
                ci=get_comp_by_label(catalog,lbl)
                s_opt=get_season_options(ci)
                xg_s2=cov_map.get(ci.get('name',''),'—')
                rows.append({'Competición':lbl,'Tipo':ci.get('season_type',''),
                    'xG':xg_s2,'Temporadas':' | '.join(list(s_opt.keys())),'Activa':ci.get('predict_label','')})
            st.dataframe(pd.DataFrame(rows),use_container_width=True,hide_index=True)
