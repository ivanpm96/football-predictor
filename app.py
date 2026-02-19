# app.py v3 — Football Predictor Pro con catalogo dinamico y season labels
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from streamlit_autorefresh import st_autorefresh
from predictor import FootballSystem
from catalog_builder import (
    load_or_refresh_catalog, get_regions, get_comp_by_label,
    get_season_options, get_train_defaults, get_predict_default_label,
)
from auto_updater import AutoUpdater, load_log

st.set_page_config(page_title='Football Predictor Pro',page_icon='⚽',
                   layout='wide',initial_sidebar_state='expanded')

st.markdown('''<style>
.big-num{font-size:2.2rem;font-weight:900;text-align:center;line-height:1.1;}
.lbl{font-size:.78rem;color:#94a3b8;text-align:center;margin-top:2px;}
.pred-card{border-radius:14px;padding:1.1rem .8rem;text-align:center;font-weight:700;font-size:1rem;margin:.2rem 0;}
.winner{background:#14532d;color:#4ade80;border:2px solid #4ade80;}
.neutral{background:#1e293b;color:#94a3b8;border:2px solid #334155;}
.status-ok{background:#052e16;border:1px solid #16a34a;border-radius:10px;padding:.5rem 1rem;font-size:.85rem;color:#4ade80;}
.status-warn{background:#1c1917;border:1px solid #d97706;border-radius:10px;padding:.5rem 1rem;font-size:.85rem;color:#fbbf24;}
.market-row{display:flex;justify-content:space-between;padding:.32rem 0;border-bottom:1px solid #1e293b;font-size:.9rem;}
.mkt-label{color:#94a3b8;}.mkt-val{font-weight:700;color:#f1f5f9;}.mkt-odd{color:#facc15;font-family:monospace;font-size:.83rem;}
.section-hdr{font-size:1rem;font-weight:700;color:#94a3b8;letter-spacing:.07em;text-transform:uppercase;
  margin:1rem 0 .4rem;border-left:3px solid #3b82f6;padding-left:.6rem;}
.season-pill{display:inline-block;background:#1e3a5f;color:#93c5fd;font-size:.75rem;
  border-radius:20px;padding:2px 9px;margin:2px;font-weight:600;}
</style>''', unsafe_allow_html=True)

# SESSION STATE
for k,v in [('system',FootballSystem()),('trained',False),('catalog',None),
            ('updater',None),('auto_on',False)]:
    if k not in st.session_state: st.session_state[k]=v

if st.session_state.catalog is None:
    st.session_state.catalog = load_or_refresh_catalog(fast_mode=True)

catalog = st.session_state.catalog
regions = get_regions(catalog)

if st.session_state.auto_on:
    st_autorefresh(interval=5*60*1000, key='auto_refresh')

# ════ SIDEBAR ═════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown('## ⚽ Football Predictor Pro')
    st.caption(f"🌐 {catalog.get('total_comps',0)} competiciones disponibles")
    st.markdown('---')

    region_list = list(regions.keys())
    region      = st.selectbox('🌐 Region', region_list, key='sb_region')
    comp_labels = regions.get(region, [])
    comp_label  = st.selectbox('🏆 Competicion', comp_labels, key='sb_comp')
    comp_info   = get_comp_by_label(catalog, comp_label)
    slug        = comp_info.get('slug','')

    # Opciones de temporada con label correcto
    season_opts  = get_season_options(comp_info)   # {'2025/26':2025, '2024/25':2024, ...}
    train_defs   = get_train_defaults(comp_info)    # ['2018/19','2019/20',...,'2024/25']
    predict_def  = get_predict_default_label(comp_info) # '2025/26'
    stype        = comp_info.get('season_type','single')

    st.markdown('---')
    st.markdown('### Configuracion de temporadas')
    if stype=='split':
        st.caption('SPLIT: 2025/26 = empieza 2025, termina 2026')
    elif stype=='single':
        st.caption('SINGLE: cada temporada es un unico año calendar')
    else:
        st.caption('TOURNAMENT: evento cada 2-4 años')

    train_labels = st.multiselect(
        'Temporadas de entrenamiento',
        options=list(season_opts.keys()),
        default=[l for l in train_defs if l in season_opts],
        help='Datos historicos para entrenar el modelo'
    )
    predict_label_sel = st.selectbox(
        'Temporada activa (pronosticos)',
        options=list(season_opts.keys()),
        index=list(season_opts.keys()).index(predict_def) if predict_def in season_opts else len(season_opts)-1,
    )

    # Panel visual de lo que va a entrenar
    if train_labels and predict_label_sel:
        st.markdown(
            "<div style='background:#0f172a;border-radius:8px;padding:.6rem;margin:.4rem 0;'>"
            "<div style='color:#94a3b8;font-size:.78rem;margin-bottom:4px;'>Entrena con:</div>"
            + ''.join([f"<span class='season-pill'>{slug}-{season_opts[l]}</span>" for l in train_labels])
            + "<div style='color:#94a3b8;font-size:.78rem;margin:6px 0 4px;'>Pronostica:</div>"
            + f"<span class='season-pill' style='background:#14532d;color:#4ade80;'>"
            + f"{slug}-{season_opts[predict_label_sel]} ({predict_label_sel})</span>"
            + '</div>',
            unsafe_allow_html=True
        )

    st.markdown('---')
    train_btn = st.button('Cargar y Entrenar', use_container_width=True, type='primary')
    force_dl  = st.checkbox('Forzar re-descarga')

    st.markdown('---')
    st.markdown('### Auto-update')
    auto_on = st.toggle('Activar actualizacion automatica', value=st.session_state.auto_on)
    if auto_on != st.session_state.auto_on:
        st.session_state.auto_on=auto_on; st.rerun()
    interval_min = st.select_slider('Intervalo',options=[15,30,60,120,360],value=30,
        format_func=lambda x: f'{x}min' if x<60 else f'{x//60}h')

    if st.session_state.trained:
        col1,col2=st.columns(2)
        with col1:
            if st.button('Chequear',use_container_width=True):
                with st.spinner('Chequeando...'):
                    upd=st.session_state.updater
                    if not upd:
                        upd=AutoUpdater(st.session_state.system,st.session_state.system.comp_ids)
                        st.session_state.updater=upd
                    r=upd.check_all()
                nm=r.get('new_matches',0)
                st.success(f'{nm} nuevos' if nm>0 else 'Al dia')
        with col2:
            if st.button('Forzar',use_container_width=True):
                with st.spinner('Forzando...'):
                    upd=st.session_state.updater or AutoUpdater(st.session_state.system,st.session_state.system.comp_ids)
                    upd.force_update(); st.session_state.updater=upd
                st.success('Listo')

    st.markdown('---')
    if st.button('Actualizar catalogo de ligas', use_container_width=True):
        with st.spinner('Verificando FixtureDownload...'):
            pb=st.progress(0.); sm=st.empty()
            def cat_prog(msg,p): sm.caption(msg); pb.progress(float(p))
            new_cat=load_or_refresh_catalog(force=True,fast_mode=False,progress_fn=cat_prog)
            pb.empty(); sm.empty()
            st.session_state.catalog=new_cat
        st.success(f"{new_cat['total_comps']} competiciones actualizadas")
        st.rerun()

    if st.session_state.trained:
        s=st.session_state.system.stats
        st.markdown('---')
        st.markdown('### Modelo activo')
        c1,c2=st.columns(2)
        c1.metric('Temporadas',len(s.get('train_seasons',[])))
        c2.metric('Activa',comp_info.get('predict_label',''))
        c1.metric('Partidos',s.get('played',0))
        c2.metric('Pendientes',s.get('pending',0))
        c1.metric('ML Acc.',f"{s.get('cv_score',0):.1%}")
        c2.metric('Equipos',len(s.get('teams',[])))

# ════ ENTRENAMIENTO ═══════════════════════════════════════════════════
if train_btn:
    if not train_labels: st.error('Selecciona al menos una temporada de entrenamiento.')
    elif not slug: st.error('Competicion invalida.')
    else:
        # Convertir labels a ints usando season_opts
        train_years   = [season_opts[l] for l in train_labels if l in season_opts]
        predict_year  = season_opts.get(predict_label_sel, 0)
        pb=st.progress(0.); st_=st.empty()
        def pf(msg,v): st_.info(f'{msg}'); pb.progress(float(v))
        ok,msg=st.session_state.system.train(slug,train_years,predict_year,
                                             force=force_dl,progress_fn=pf)
        pb.empty(); st_.empty()
        if ok:
            st.session_state.trained=True
            if auto_on:
                upd=AutoUpdater(st.session_state.system,st.session_state.system.comp_ids)
                upd.start_background(interval_min); st.session_state.updater=upd
            s=st.session_state.system.stats
            tl=', '.join([season_opts.get(l,'') and f'{slug}-{season_opts[l]} ({l})' or l
                          for l in train_labels[:3]])
            tl += ' ...' if len(train_labels)>3 else ''
            st.success(f"Modelo listo | {comp_label} | Entrena: {tl} | "
                       f"Activa: {predict_label_sel} | {s['played']} partidos | "
                       f"Acc: {s['cv_score']:.1%}")
            st.rerun()
        else: st.error(f'Error: {msg}')

# ════ CABECERA ════════════════════════════════════════════════════════
st.title('⚽ Football Predictor Pro')
if not st.session_state.trained:
    st.info('Selecciona una liga y haz click en Cargar y Entrenar para comenzar.')
    st.markdown('### Competiciones disponibles')
    for reg, labels in regions.items():
        with st.expander(f'**{reg}** — {len(labels)} ligas'):
            for lbl in labels:
                ci=get_comp_by_label(catalog,lbl)
                s_opts=get_season_options(ci)
                lbl_str=', '.join(list(s_opts.keys()))
                stype_=ci.get('season_type','single')
                st.markdown(f'- **{lbl}** ({stype_}) &nbsp; `{lbl_str}`')
    st.stop()

log_data=load_log()
if st.session_state.auto_on:
    upd_obj=st.session_state.updater
    running=upd_obj._running if upd_obj else False
    icon='ON' if running else 'PAUSA'
    st.markdown(
        f"<div class='status-ok'>Auto-update {icon}"
        f" | Ultimo chequeo: {log_data.get('last_check','Nunca')}"
        f" | Proximo: {log_data.get('next_check','—')}"
        f" | Nuevos acumulados: {log_data.get('total_new_matches',0)}</div>",
        unsafe_allow_html=True)
else:
    s_=st.session_state.system.stats
    pred_lbl=comp_info.get('predict_label', str(s_.get('predict_season','')))
    st.markdown(
        f"<div class='status-warn'>{comp_label}"
        f" | Temporadas entrenadas: {len(s_.get('train_seasons',[]))}"
        f" | Activa: {pred_lbl}"
        f" | Pendientes: {s_.get('pending',0)}</div>",
        unsafe_allow_html=True)
st.markdown('')

system=st.session_state.system
teams=system.get_teams()

tab1,tab2,tab3,tab4,tab5,tab6,tab7,tab8=st.tabs([
    'Prediccion','Tabla','Estadisticas','H2H',
    'Proximos partidos','Datos Crudos','Backtesting','Catalogo'])

# ── TAB 1: PREDICCION ─────────────────────────────────────────────────
with tab1:
    st.markdown('<div class="section-hdr">Partido a predecir</div>',unsafe_allow_html=True)
    c1,cm,c2=st.columns([5,1,5])
    with c1: home=st.selectbox('Local',teams,key='pred_h')
    with cm: st.markdown('<br><br><div style="text-align:center;font-size:1.5rem;">vs</div>',unsafe_allow_html=True)
    with c2: away=st.selectbox('Visitante',[t for t in teams if t!=home],key='pred_a')

    if st.button('Predecir',type='primary',use_container_width=True,key='pred_btn'):
        with st.spinner('Calculando...'):
            pred=system.predict(home,away)
            hs=system.team_full_stats(home)
            as_=system.team_full_stats(away)
        if not pred: st.error('No se pudo calcular.')
        else:
            st.markdown('---')
            c1,cm,c2=st.columns([4,2,4])
            with c1:
                st.markdown(f"<div class='big-num'>{home}</div><div class='lbl'>Local</div>",unsafe_allow_html=True)
                if hs: st.caption(f"PPP:{hs.get('total_PPP',0)} | GF/pj:{hs.get('avg_gf',0)} | Racha:{hs.get('streak','—')}")
            with cm:
                st.markdown('<br><div style="text-align:center;font-size:1.3rem">VS</div>',unsafe_allow_html=True)
                st.caption(f"lambda {pred['lambda_home']} vs {pred['lambda_away']}")
            with c2:
                st.markdown(f"<div class='big-num'>{away}</div><div class='lbl'>Visitante</div>",unsafe_allow_html=True)
                if as_: st.caption(f"PPP:{as_.get('total_PPP',0)} | GF/pj:{as_.get('avg_gf',0)} | Racha:{as_.get('streak','—')}")

            st.caption(f"Confianza: {pred['confidence']} | Marcador probable: {pred['expected_score']}")
            st.markdown('---')

            st.markdown('<div class="section-hdr">1X2</div>',unsafe_allow_html=True)
            c_h,c_d,c_a=st.columns(3)
            for col,code,lbl,prob,odd in [
                (c_h,'H',f'Local {home}',pred['combined_H'],pred['odds_H']),
                (c_d,'D','Empate',pred['combined_D'],pred['odds_D']),
                (c_a,'A',f'Visita {away}',pred['combined_A'],pred['odds_A'])]:
                css='winner' if code==pred['final_prediction'] else 'neutral'
                sfx=' PRED' if code==pred['final_prediction'] else ''
                with col:
                    st.markdown(f"<div class='pred-card {css}'>{lbl}{sfx}"
                        f"<br><span style='font-size:1.9rem'>{prob:.1%}</span>"
                        f"<br><span style='font-size:.82rem;color:#facc15'>Cuota: {odd}</span></div>",
                        unsafe_allow_html=True)

            vals=[pred['combined_H'],pred['combined_D'],pred['combined_A']]
            fig_b=go.Figure(go.Bar(x=[f'Local {home}','Empate',f'Visita {away}'],y=[v*100 for v in vals],
                marker_color=['#4ade80' if v==max(vals) else '#475569' for v in vals],
                text=[f'{v:.1%}' for v in vals],textposition='outside'))
            fig_b.update_layout(yaxis=dict(range=[0,max(vals)*138]),height=260,
                plot_bgcolor='#0f172a',paper_bgcolor='#0f172a',font_color='white',
                showlegend=False,margin=dict(t=10,b=5))
            st.plotly_chart(fig_b,use_container_width=True)

            st.markdown('<div class="section-hdr">Mercados</div>',unsafe_allow_html=True)
            mc1,mc2=st.columns(2)
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
                    odd_=round(1/p_,2) if p_>0 else 99
                    st.markdown(f"<div class='market-row'><span class='mkt-label'>{lbl_}</span>"
                        f"<span><b class='mkt-val'>{p_:.1%}</b> <span class='mkt-odd'>({odd_})</span></span></div>",
                        unsafe_allow_html=True)

            st.markdown('<div class="section-hdr">Mapa de calor de marcadores</div>',unsafe_allow_html=True)
            M=np.array(pred['score_matrix'])[:7,:7]
            fig_h=px.imshow(M*100,labels=dict(x=f'Goles {away}',y=f'Goles {home}',color='%'),
                x=[str(i) for i in range(7)],y=[str(i) for i in range(7)],
                color_continuous_scale='Greens',text_auto='.1f',aspect='auto')
            fig_h.update_layout(plot_bgcolor='#0f172a',paper_bgcolor='#0f172a',
                font_color='white',height=350,margin=dict(t=30,b=10))
            st.plotly_chart(fig_h,use_container_width=True)

            st.markdown('<div class="section-hdr">Forma reciente (ultimos 5)</div>',unsafe_allow_html=True)
            fc1,fc2=st.columns(2)
            for col,tm_,std_ in [(fc1,home,hs),(fc2,away,as_)]:
                with col:
                    st.markdown(f'**{tm_}**')
                    if std_ and std_.get('last5_matches'):
                        rows_=[{'':{"W":'WIN',"D":'EMP',"L":'DER'}.get(m['_res'],''),
                            'Rival':m['away_team'] if m['home_team']==tm_ else m['home_team'],
                            'Marcador':f"{int(m['home_score' if m['home_team']==tm_ else 'away_score'])}-{int(m['away_score' if m['home_team']==tm_ else 'home_score'])}"}
                            for m in std_['last5_matches']]
                        st.dataframe(pd.DataFrame(rows_),use_container_width=True,hide_index=True)
                    else: st.info('Sin historial.')

# ── TAB 2: TABLA ──────────────────────────────────────────────────────
with tab2:
    pred_lbl=comp_info.get('predict_label','')
    st.subheader(f'Tabla de posiciones — {comp_label} {pred_lbl}')
    df_t=system.league_table()
    if df_t.empty: st.warning('Sin datos.')
    else:
        df_s=df_t.copy()
        df_s['Forma']=df_s['Forma'].apply(lambda x:''.join({'W':'W','D':'D','L':'L'}.get(c,'?') for c in x))
        st.dataframe(df_s,use_container_width=True)
        fig_p=px.bar(df_t.head(12),x='Equipo',y='PTS',color='PPP',color_continuous_scale='Blues',height=320)
        fig_p.update_layout(plot_bgcolor='#0f172a',paper_bgcolor='#0f172a',font_color='white')
        st.plotly_chart(fig_p,use_container_width=True)

# ── TAB 3: ESTADISTICAS ───────────────────────────────────────────────
with tab3:
    st.subheader('Estadisticas de equipo')
    team_s=st.selectbox('Equipo',teams,key='t3')
    std_=system.team_full_stats(team_s)
    if std_:
        m1,m2,m3,m4,m5,m6,m7,m8=st.columns(8)
        m1.metric('PJ',std_['total_PJ']); m2.metric('PG',std_['total_PG'])
        m3.metric('PE',std_['total_PE']); m4.metric('PP',std_['total_PP'])
        m5.metric('GF',std_['total_GF']); m6.metric('GC',std_['total_GC'])
        m7.metric('GD',f"{std_['total_GD']:+d}"); m8.metric('PPP',std_['total_PPP'])
        df_sp=pd.DataFrame([
            {'Cond.':'Local','PJ':std_['home_PJ'],'PG':std_['home_PG'],'PE':std_['home_PE'],
             'PP':std_['home_PP'],'GF':std_['home_GF'],'GC':std_['home_GC'],'GD':std_['home_GD'],'PPP':std_['home_PPP']},
            {'Cond.':'Visita','PJ':std_['away_PJ'],'PG':std_['away_PG'],'PE':std_['away_PE'],
             'PP':std_['away_PP'],'GF':std_['away_GF'],'GC':std_['away_GC'],'GD':std_['away_GD'],'PPP':std_['away_PPP']},
        ])
        st.dataframe(df_sp,use_container_width=True,hide_index=True)
        sp1,sp2,sp3,sp4=st.columns(4)
        sp1.metric('GF/pj',std_['avg_gf']); sp2.metric('GC/pj',std_['avg_gc'])
        sp3.metric('Porteria 0',std_['clean_sheets']); sp4.metric('BTTS',std_['btts_count'])
        st.markdown(f"**Racha:** {std_.get('streak','—')}")

# ── TAB 4: H2H ────────────────────────────────────────────────────────
with tab4:
    st.subheader('Head-to-Head')
    h1,hm_,h2=st.columns([5,1,5])
    with h1: hh=st.selectbox('Equipo A',teams,key='h2h_h')
    with hm_: st.markdown('<br><br><div style="text-align:center">VS</div>',unsafe_allow_html=True)
    with h2: ha=st.selectbox('Equipo B',[t for t in teams if t!=hh],key='h2h_a')
    if st.button('Ver H2H',use_container_width=True):
        h2h_d=system.h2h(hh,ha); s_=h2h_d['summary']; df_=h2h_d['matches']
        if df_.empty: st.info('Sin enfrentamientos previos.')
        else:
            sm1,sm2,sm3,sm4,sm5=st.columns(5)
            sm1.metric('Total',s_['total']); sm2.metric(f'Vic {hh}',s_.get(f'{hh}_wins',0))
            sm3.metric('Empates',s_['draws']); sm4.metric(f'Vic {ha}',s_.get(f'{ha}_wins',0))
            sm5.metric('Goles/pj',s_['avg_goals'])
            fig_pie=go.Figure(go.Pie(
                labels=[f'{hh}','Empate',f'{ha}'],
                values=[s_.get(f'{hh}_wins',0),s_['draws'],s_.get(f'{ha}_wins',0)],
                marker_colors=['#4ade80','#facc15','#f87171'],textinfo='label+percent',hole=0.4))
            fig_pie.update_layout(height=260,plot_bgcolor='#0f172a',paper_bgcolor='#0f172a',font_color='white')
            st.plotly_chart(fig_pie,use_container_width=True)
            df_s=df_[['competition_id','date','home_team','home_score','away_score','away_team']].copy()
            df_s['date']=df_s['date'].dt.strftime('%Y-%m-%d')
            df_s['Marcador']=df_s['home_score'].astype(int).astype(str)+'-'+df_s['away_score'].astype(int).astype(str)
            st.dataframe(df_s[['date','competition_id','home_team','Marcador','away_team']].rename(columns={
                'date':'Fecha','competition_id':'Liga','home_team':'Local','away_team':'Visitante'}),
                use_container_width=True,hide_index=True)

# ── TAB 5: PROXIMOS PARTIDOS ──────────────────────────────────────────
with tab5:
    pred_lbl=comp_info.get('predict_label','')
    pred_id =system.stats.get('predict_id','')
    st.subheader(f'Pronosticos — {comp_label} {pred_lbl}')
    st.caption(f'Feed: {pred_id} | Partidos con HomeTeamScore=null')
    if st.button('Predecir todos los fixtures pendientes',use_container_width=True,type='primary'):
        with st.spinner(f'Calculando pronosticos para {pred_id}...'):
            df_p=system.predict_pending()
        if df_p.empty:
            st.warning('No hay partidos pendientes. Puede que la temporada este completa o sin datos.')
        else:
            st.success(f'{len(df_p)} partidos pronosticados')
            show=['round','date','home_team','away_team','final_prediction','confidence',
                  'expected_score','combined_H','combined_D','combined_A','over_2_5','btts']
            df_s=df_p[[c for c in show if c in df_p.columns]].copy()
            df_s=df_s.rename(columns={'round':'Jornada','date':'Fecha','home_team':'Local',
                'away_team':'Visitante','final_prediction':'Pred','confidence':'Conf',
                'expected_score':'Marcador','combined_H':'P(L)','combined_D':'P(E)',
                'combined_A':'P(V)','over_2_5':'O2.5','btts':'BTTS'})
            for c in ['P(L)','P(E)','P(V)','O2.5','BTTS']:
                if c in df_s.columns:
                    df_s[c]=df_s[c].apply(lambda x:f'{x:.1%}' if isinstance(x,(float,int)) else x)
            st.dataframe(df_s,use_container_width=True,hide_index=True)
            st.download_button('Descargar CSV',df_p.to_csv(index=False).encode(),
                f'predicciones_{pred_id}.csv','text/csv')

# ── TAB 6: DATOS CRUDOS ───────────────────────────────────────────────
with tab6:
    st.subheader('Datos crudos')
    fc1,fc2,fc3=st.columns(3)
    with fc1: tr_=st.selectbox('Equipo',['(Todos)']+teams,key='raw_t')
    with fc2: rr_=st.selectbox('Resultado',['(Todos)','H','D','A'],key='raw_r')
    with fc3: lr_=st.slider('Max.',50,1000,300,step=50)
    df_r=system.raw_matches(
        team=None if tr_=='(Todos)' else tr_,
        result=None if rr_=='(Todos)' else rr_).head(lr_)
    st.markdown(f'**{len(df_r)} partidos**')
    st.dataframe(df_r,use_container_width=True,hide_index=True)
    if not df_r.empty:
        st.download_button('Descargar CSV',df_r.to_csv(index=False).encode(),'datos_crudos.csv','text/csv')

# ── TAB 7: BACKTESTING ────────────────────────────────────────────────
with tab7:
    st.subheader('Backtesting del modelo')
    n_bt=st.slider('Ultimos N partidos',50,300,100,25)
    if st.button('Ejecutar backtest',use_container_width=True):
        with st.spinner('Evaluando...'): df_bt=system.backtest(n_bt)
        if df_bt.empty: st.warning('Insuficientes datos.')
        else:
            acc=df_bt['correct'].mean()
            st.markdown(f'### Precision: **{acc:.1%}** ({df_bt["correct"].sum()}/{len(df_bt)})')
            fig_bt=px.histogram(df_bt,x='conf',color='correct',nbins=20,height=280,
                color_discrete_map={True:'#4ade80',False:'#f87171'},
                title='Confianza vs Acierto')
            fig_bt.update_layout(plot_bgcolor='#0f172a',paper_bgcolor='#0f172a',font_color='white')
            st.plotly_chart(fig_bt,use_container_width=True)
            st.dataframe(df_bt.rename(columns={'date':'Fecha','home':'Local','away':'Visitante',
                'real':'Real','pred':'Predicho','correct':'Correcto','conf':'Confianza'}),
                use_container_width=True,hide_index=True)

# ── TAB 8: CATALOGO ───────────────────────────────────────────────────
with tab8:
    st.subheader('Catalogo de competiciones')
    st.caption(f"Actualizado: {catalog.get('last_updated','—')} | Total: {catalog.get('total_comps',0)} competiciones")
    search=st.text_input('Buscar liga...',placeholder='bundesliga, serie, mls...')
    for reg, labels in regions.items():
        filtered=[l for l in labels if not search or search.lower() in l.lower()]
        if not filtered: continue
        with st.expander(f'**{reg}** ({len(filtered)} ligas)',expanded=bool(search)):
            rows=[]
            for lbl in filtered:
                ci=get_comp_by_label(catalog,lbl)
                s_opts=get_season_options(ci)
                rows.append({
                    'Competicion':lbl,
                    'Tipo':ci.get('season_type',''),
                    'Temporadas':' | '.join(list(s_opts.keys())),
                    'Activa':ci.get('predict_label',''),
                    'Entrena con':' | '.join(ci.get('train_labels',[])),
                })
            st.dataframe(pd.DataFrame(rows),use_container_width=True,hide_index=True)