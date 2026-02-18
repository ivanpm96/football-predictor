
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from predictor import FootballSystem, COMPETITION_MAP, REGIONS

st.set_page_config(page_title="⚽ Football Predictor Pro",page_icon="⚽",layout="wide",
                   initial_sidebar_state="expanded")

# ════ CSS ════════════════════════════════════════════════════════════════
st.markdown("""
<style>
.big-num    {font-size:2.2rem;font-weight:900;text-align:center;line-height:1.1;}
.lbl        {font-size:.78rem;color:#94a3b8;text-align:center;margin-top:2px;}
.pred-card  {border-radius:14px;padding:1.1rem .8rem;text-align:center;font-weight:700;
             font-size:1rem;margin:.2rem 0;}
.winner     {background:#14532d;color:#4ade80;border:2px solid #4ade80;}
.neutral    {background:#1e293b;color:#94a3b8;border:2px solid #334155;}
.badge-W    {background:#166534;color:#bbf7d0;padding:2px 8px;border-radius:20px;font-size:.8rem;}
.badge-D    {background:#713f12;color:#fef9c3;padding:2px 8px;border-radius:20px;font-size:.8rem;}
.badge-L    {background:#7f1d1d;color:#fecaca;padding:2px 8px;border-radius:20px;font-size:.8rem;}
.market-row {display:flex;justify-content:space-between;padding:.35rem 0;
             border-bottom:1px solid #1e293b;font-size:.92rem;}
.mkt-label  {color:#94a3b8;}
.mkt-val    {font-weight:700;color:#f1f5f9;}
.mkt-odd    {color:#facc15;font-family:monospace;font-size:.85rem;}
.section-hdr{font-size:1.1rem;font-weight:700;color:#94a3b8;letter-spacing:.08em;
             text-transform:uppercase;margin:1.2rem 0 .5rem 0;border-left:3px solid #3b82f6;
             padding-left:.6rem;}
</style>
""", unsafe_allow_html=True)

# ════ SESSION ════════════════════════════════════════════════════════════
for k,v in [("system",FootballSystem()),("trained",False)]:
    if k not in st.session_state: st.session_state[k]=v

# ════ SIDEBAR ════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## ⚽ Football Predictor Pro")
    st.caption("Motor: Poisson + Gradient Boosting | fixturedownload.com")
    st.markdown("---")
    region=st.selectbox("🌐 Región",list(REGIONS.keys()),index=0)
    comp_label=st.selectbox("🏆 Competición",REGIONS[region],index=0)
    avail_years=st.session_state.system.available_years(comp_label)
    default_y=avail_years[-2:] if len(avail_years)>=2 else avail_years
    years=st.multiselect("📅 Temporadas",avail_years,default=default_y,
                         help="Más temporadas → más datos históricos para entrenar")
    force_dl=st.checkbox("🔄 Forzar re-descarga",value=False)
    st.markdown("---")
    train_btn=st.button("🚀 Cargar & Entrenar",use_container_width=True,type="primary")

    if st.session_state.trained:
        s=st.session_state.system.stats
        st.markdown("---")
        st.markdown("### 📊 Modelo activo")
        st.markdown(f"**{s.get('comp_label','')}**")
        c1,c2=st.columns(2)
        c1.metric("Jugados",s["played"]); c2.metric("Pendientes",s["pending"])
        c1.metric("ML Acc.",f"{s['cv_score']:.1%}"); c2.metric("Ventaja local",f"{s['home_adv']}×")
        st.metric("Equipos en BD",len(s["teams"]))

# ════ ENTRENAMIENTO ═══════════════════════════════════════════════════════
if train_btn:
    if not years: st.error("Selecciona al menos una temporada.")
    else:
        pb=st.progress(0.); st_=st.empty()
        def pf(msg,v): st_.info(f"⏳ {msg}"); pb.progress(float(v))
        ok,msg=st.session_state.system.train(comp_label,years,force=force_dl,progress_fn=pf)
        pb.empty(); st_.empty()
        if ok:
            st.session_state.trained=True
            s=st.session_state.system.stats
            st.success(f"✅ **{comp_label}** | {s['played']} partidos | {len(s['teams'])} equipos | Acc: {s['cv_score']:.1%}")
            st.rerun()
        else: st.error(f"❌ {msg}")

# ════ HEADER ═════════════════════════════════════════════════════════════
st.title("⚽ Football Predictor Pro")
st.caption("Sistema profesional de análisis y predicción de fútbol")
if not st.session_state.trained:
    c1,c2,c3=st.columns(3)
    c1.info("**1️⃣ Región & Liga**\n\n27 competiciones de todo el mundo.")
    c2.info("**2️⃣ Temporadas**\n\nElige los años históricos para entrenar.")
    c3.info("**3️⃣ Cargar & Entrenar**\n\nEl sistema hace todo automáticamente.")
    st.stop()

system=st.session_state.system
teams=system.get_teams()
comp_ids=system.comp_ids

# ════ TABS ════════════════════════════════════════════════════════════════
tab1,tab2,tab3,tab4,tab5,tab6,tab7=st.tabs([
    "🎯 Predicción","🏟️ Tabla","📊 Estadísticas","⚔️ Head-to-Head",
    "🔬 Datos Crudos","🧪 Backtesting","📈 Modelo"])

# ════════════════════════════════════════════════════════════════════
# TAB 1 — PREDICCIÓN PROFESIONAL
# ════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown('<div class="section-hdr">Selección de partido</div>',unsafe_allow_html=True)
    c1,cm,c2=st.columns([5,1,5])
    with c1: home=st.selectbox("🏠 Local",teams,index=0,key="home1")
    with cm: st.markdown("<br><br><div style='text-align:center;font-size:1.5rem;'>vs</div>",unsafe_allow_html=True)
    with c2: away=st.selectbox("✈️ Visitante",[t for t in teams if t!=home],index=0,key="away1")
    comp_sel=st.selectbox("Temporada de referencia",comp_ids,index=len(comp_ids)-1)

    if st.button("🔮 Predecir partido",type="primary",use_container_width=True):
        with st.spinner("Calculando..."):
            pred=system.predict(home,away,comp_sel)
        if not pred:
            st.error("No se pudo calcular.")
        else:
            # ── Header partido ──
            st.markdown("---")
            c1,cm,c2=st.columns([4,2,4])
            hs=system.team_full_stats(home,comp_sel)
            as_=system.team_full_stats(away,comp_sel)
            with c1:
                st.markdown(f"<div class='big-num'>{home}</div><div class='lbl'>🏠 Local</div>",unsafe_allow_html=True)
                if hs: st.markdown(f"<div style='text-align:center;color:#94a3b8;font-size:.85rem;'>"
                    f"PPP: <b>{hs.get('total_PPP',0)}</b> | GF/pj: <b>{hs.get('avg_gf',0)}</b> | "
                    f"Racha: <b>{hs.get('streak','—')}</b></div>",unsafe_allow_html=True)
            with cm:
                st.markdown(f"<br><div style='text-align:center;font-size:1.4rem;'>🆚</div>"
                    f"<div style='text-align:center;color:#64748b;font-size:.85rem;'>{comp_sel}</div>",
                    unsafe_allow_html=True)
            with c2:
                st.markdown(f"<div class='big-num'>{away}</div><div class='lbl'>✈️ Visitante</div>",unsafe_allow_html=True)
                if as_: st.markdown(f"<div style='text-align:center;color:#94a3b8;font-size:.85rem;'>"
                    f"PPP: <b>{as_.get('total_PPP',0)}</b> | GF/pj: <b>{as_.get('avg_gf',0)}</b> | "
                    f"Racha: <b>{as_.get('streak','—')}</b></div>",unsafe_allow_html=True)

            st.markdown(f"<p style='text-align:center;color:#475569;margin:.4rem 0;'>"
                f"λ local: <b>{pred['lambda_home']}</b> &nbsp;|&nbsp; λ visita: <b>{pred['lambda_away']}</b>"
                f" &nbsp;|&nbsp; Confianza: <b>{pred['confidence']}</b></p>",unsafe_allow_html=True)
            st.markdown("---")

            # ── Probabilidades 1X2 ──
            st.markdown('<div class="section-hdr">Probabilidades 1X2</div>',unsafe_allow_html=True)
            c_h,c_d,c_a=st.columns(3)
            for col,code,lbl,prob,odd in [
                (c_h,"H",f"🏠 {home}",pred["combined_H"],pred["odds_H"]),
                (c_d,"D","🤝 Empate",  pred["combined_D"],pred["odds_D"]),
                (c_a,"A",f"✈️ {away}", pred["combined_A"],pred["odds_A"]),
            ]:
                css="winner" if code==pred["final_prediction"] else "neutral"
                sfx=" ⭐" if code==pred["final_prediction"] else ""
                with col:
                    st.markdown(f"<div class='pred-card {css}'>{lbl}{sfx}"
                        f"<br><span style='font-size:2rem;'>{prob:.1%}</span>"
                        f"<br><span style='font-size:.85rem;color:#facc15;'>Cuota implícita: {odd}</span></div>",
                        unsafe_allow_html=True)

            # ── Barra de probabilidades ──
            fig_b=go.Figure()
            vals=[pred["combined_H"],pred["combined_D"],pred["combined_A"]]
            lbls=[f"🏠 {home}","🤝 Empate",f"✈️ {away}"]
            fig_b.add_trace(go.Bar(x=lbls,y=[v*100 for v in vals],
                marker_color=["#4ade80" if v==max(vals) else "#475569" for v in vals],
                text=[f"{v:.1%}" for v in vals],textposition="outside"))
            fig_b.update_layout(yaxis=dict(range=[0,max(vals)*135]),height=300,
                plot_bgcolor="#0f172a",paper_bgcolor="#0f172a",font_color="white",showlegend=False,
                margin=dict(t=20,b=10))
            st.plotly_chart(fig_b,use_container_width=True)

            # ── TODOS LOS MERCADOS ──
            st.markdown('<div class="section-hdr">Todos los mercados</div>',unsafe_allow_html=True)
            mc1,mc2=st.columns(2)
            with mc1:
                st.markdown("**⚽ Goles (Totales)**")
                markets_goals=[
                    ("Over 1.5",pred["over_1_5"],"Under 1.5",pred["under_1_5"]),
                    ("Over 2.5",pred["over_2_5"],"Under 2.5",pred["under_2_5"]),
                    ("Over 3.5",pred["over_3_5"],"Under 3.5",pred["under_3_5"]),
                ]
                for ov_lbl,ov_p,un_lbl,un_p in markets_goals:
                    c_ov,c_un=st.columns(2)
                    c_ov.metric(ov_lbl,f"{ov_p:.1%}",f"Cuota {round(1/ov_p,2) if ov_p>0 else '∞'}")
                    c_un.metric(un_lbl,f"{un_p:.1%}",f"Cuota {round(1/un_p,2) if un_p>0 else '∞'}")
            with mc2:
                st.markdown("**⚽ Otros mercados**")
                other_markets=[
                    ("BTTS (Ambos anotan)",pred["btts"]),
                    ("No BTTS",pred["no_btts"]),
                    (f"Handicap Asiático {home} (-0.5)",pred["asian_home"]),
                    (f"Handicap Asiático {away} (-0.5)",pred["asian_away"]),
                ]
                for lbl,p in other_markets:
                    odd_val=round(1/p,2) if p>0 else 99
                    st.markdown(f"<div class='market-row'><span class='mkt-label'>{lbl}</span>"
                        f"<span><b class='mkt-val'>{p:.1%}</b> "
                        f"<span class='mkt-odd'>({odd_val})</span></span></div>",
                        unsafe_allow_html=True)

            # ── Mapa de calor ──
            st.markdown('<div class="section-hdr">Mapa de calor de marcadores</div>',unsafe_allow_html=True)
            M=np.array(pred["score_matrix"])[:7,:7]
            fig_h=px.imshow(M*100,labels=dict(x=f"Goles {away}",y=f"Goles {home}",color="%"),
                x=[str(i) for i in range(7)],y=[str(i) for i in range(7)],
                color_continuous_scale="Greens",text_auto=".1f",aspect="auto")
            fig_h.update_layout(plot_bgcolor="#0f172a",paper_bgcolor="#0f172a",
                font_color="white",height=370,title="Probabilidad (%) por marcador exacto",
                margin=dict(t=40,b=10))
            st.plotly_chart(fig_h,use_container_width=True)

            # ── Últimos 5 partidos de cada equipo ──
            st.markdown('<div class="section-hdr">Forma reciente</div>',unsafe_allow_html=True)
            fc1,fc2=st.columns(2)
            for col,team_name,stats_d in [(fc1,home,hs),(fc2,away,as_)]:
                with col:
                    st.markdown(f"**{team_name}** — últimos 5")
                    if stats_d and stats_d.get("last5_matches"):
                        rows_=[]
                        for m in stats_d["last5_matches"]:
                            is_h=m["home_team"]==team_name
                            opp=m["away_team"] if is_h else m["home_team"]
                            gf_=m["home_score"] if is_h else m["away_score"]
                            gc_=m["away_score"] if is_h else m["home_score"]
                            badge={"W":"🟢","D":"🟡","L":"🔴"}.get(m["_res"],"")
                            rows_.append({"":badge,"Rival":opp,"Resultado":f"{gf_:.0f}-{gc_:.0f}","R":m["_res"]})
                        st.dataframe(pd.DataFrame(rows_)[["","Rival","Resultado"]],
                            use_container_width=True,hide_index=True)
                    else:
                        st.info("Sin historial previo.")

            # ── Radar comparativo ──
            st.markdown('<div class="section-hdr">Comparativa de métricas</div>',unsafe_allow_html=True)
            if hs and as_:
                cats=["PPP","GF/pj","GC/pj(inv)","% Victorias","% CS"]
                def safe(d,k,mx=1): return min(d.get(k,0)/mx,1)
                h_vals=[safe(hs,"total_PPP",3),safe(hs,"avg_gf",3),
                        1-safe(hs,"avg_gc",3),safe(hs,"home_PG",hs.get("home_PJ",1)),
                        safe(hs,"clean_sheets",hs.get("total_PJ",1))]
                a_vals=[safe(as_,"total_PPP",3),safe(as_,"avg_gf",3),
                        1-safe(as_,"avg_gc",3),safe(as_,"away_PG",as_.get("away_PJ",1)),
                        safe(as_,"clean_sheets",as_.get("total_PJ",1))]
                fig_r=go.Figure()
                fig_r.add_trace(go.Scatterpolar(r=h_vals+[h_vals[0]],
                    theta=cats+[cats[0]],name=home,fill="toself",line_color="#4ade80"))
                fig_r.add_trace(go.Scatterpolar(r=a_vals+[a_vals[0]],
                    theta=cats+[cats[0]],name=away,fill="toself",line_color="#f87171"))
                fig_r.update_layout(polar=dict(radialaxis=dict(range=[0,1],showticklabels=False)),
                    legend=dict(x=0.85,y=1),height=380,
                    plot_bgcolor="#0f172a",paper_bgcolor="#0f172a",font_color="white",
                    title="Comparativa normalizada de rendimiento")
                st.plotly_chart(fig_r,use_container_width=True)

            with st.expander("🔬 Detalle por modelo"):
                def fmt(x): return f"{x:.1%}" if isinstance(x,(float,int)) else "N/A"
                st.dataframe(pd.DataFrame({"Modelo":["Poisson","ML (GBM)","Combinado"],
                    f"🏠{home}":[fmt(pred["poisson_H"]),fmt(pred["ml_H"]),fmt(pred["combined_H"])],
                    "🤝Empate":[fmt(pred["poisson_D"]),fmt(pred["ml_D"]),fmt(pred["combined_D"])],
                    f"✈️{away}":[fmt(pred["poisson_A"]),fmt(pred["ml_A"]),fmt(pred["combined_A"])]}),
                    use_container_width=True,hide_index=True)

# ════════════════════════════════════════════════════════════════════
# TAB 2 — TABLA DE POSICIONES
# ════════════════════════════════════════════════════════════════════
with tab2:
    st.subheader("🏟️ Tabla de posiciones")
    comp_t=st.selectbox("Temporada",comp_ids,index=len(comp_ids)-1,key="tab2_comp")
    df_table=system.league_table(comp_t)
    if df_table.empty:
        st.warning("No hay datos suficientes para generar la tabla.")
    else:
        def forma_icons(s):
            return "".join({"W":"🟢","D":"🟡","L":"🔴"}.get(c,"⚪") for c in s)
        df_show=df_table.copy()
        df_show["Forma"]=df_show["Forma"].apply(forma_icons)
        st.dataframe(df_show,use_container_width=True,
            column_config={"PTS":st.column_config.NumberColumn("PTS",help="Puntos totales"),
                           "PPP":st.column_config.NumberColumn("PPP",format="%.2f",help="Puntos por partido"),
                           "GD":st.column_config.NumberColumn("GD",help="Diferencia de goles")})
        fig_pts=px.bar(df_table.head(10),x="Equipo",y="PTS",color="PPP",
            color_continuous_scale="Greens",title="Top 10 — Puntos totales",height=360)
        fig_pts.update_layout(plot_bgcolor="#0f172a",paper_bgcolor="#0f172a",font_color="white")
        st.plotly_chart(fig_pts,use_container_width=True)

# ════════════════════════════════════════════════════════════════════
# TAB 3 — ESTADÍSTICAS COMPLETAS DE EQUIPO
# ════════════════════════════════════════════════════════════════════
with tab3:
    st.subheader("📊 Estadísticas completas de equipo")
    c1,c2=st.columns([3,1])
    with c1: team_s=st.selectbox("Equipo",teams,key="t3_team")
    with c2: comp_s=st.selectbox("Temporada",comp_ids,index=len(comp_ids)-1,key="t3_comp")
    st_data=system.team_full_stats(team_s,comp_s)
    if st_data:
        # ── Métricas globales ──
        st.markdown('<div class="section-hdr">Global</div>',unsafe_allow_html=True)
        m1,m2,m3,m4,m5,m6,m7,m8=st.columns(8)
        m1.metric("PJ",st_data["total_PJ"]); m2.metric("PG",st_data["total_PG"])
        m3.metric("PE",st_data["total_PE"]); m4.metric("PP",st_data["total_PP"])
        m5.metric("GF",st_data["total_GF"]); m6.metric("GC",st_data["total_GC"])
        m7.metric("GD",f"{st_data['total_GD']:+d}"); m8.metric("PPP",st_data["total_PPP"])

        # ── Home vs Away ──
        st.markdown('<div class="section-hdr">Local vs Visitante</div>',unsafe_allow_html=True)
        df_split=pd.DataFrame([
            {"Condición":"🏠 Local","PJ":st_data["home_PJ"],"PG":st_data["home_PG"],
             "PE":st_data["home_PE"],"PP":st_data["home_PP"],
             "GF":st_data["home_GF"],"GC":st_data["home_GC"],"GD":st_data["home_GD"],
             "PTS":st_data["home_PTS"],"PPP":st_data["home_PPP"]},
            {"Condición":"✈️ Visitante","PJ":st_data["away_PJ"],"PG":st_data["away_PG"],
             "PE":st_data["away_PE"],"PP":st_data["away_PP"],
             "GF":st_data["away_GF"],"GC":st_data["away_GC"],"GD":st_data["away_GD"],
             "PTS":st_data["away_PTS"],"PPP":st_data["away_PPP"]},
        ])
        st.dataframe(df_split,use_container_width=True,hide_index=True)

        # ── Stats especiales ──
        st.markdown('<div class="section-hdr">Métricas avanzadas</div>',unsafe_allow_html=True)
        sp1,sp2,sp3,sp4,sp5=st.columns(5)
        sp1.metric("⚽ Goles/pj",st_data["avg_gf"])
        sp2.metric("🚫 GC/pj",st_data["avg_gc"])
        sp3.metric("🔒 Porterías a 0",st_data["clean_sheets"])
        sp4.metric("🔕 Sin anotar",st_data["failed_to_score"])
        sp5.metric("⚽⚽ BTTS",st_data["btts_count"])

        # ── Forma ──
        st.markdown('<div class="section-hdr">Forma reciente</div>',unsafe_allow_html=True)
        forma="".join({"W":"🟢","D":"🟡","L":"🔴"}.get(r,"⚪") for r in st_data.get("last5",[]))
        st.markdown(f"Últimos 5: **{forma}** &nbsp;&nbsp; Racha actual: **{st_data.get('streak','—')}**")

        # ── Gráfico goles histórico ──
        pl=system.df_proc[system.df_proc["played"]&(system.df_proc["competition_id"]==comp_s)]
        tm=pl[(pl["home_team"]==team_s)|(pl["away_team"]==team_s)].sort_values("date")
        gf_l,gc_l,res_l,opp_l=[],[],[],[]
        for _,row in tm.iterrows():
            is_h=row["home_team"]==team_s
            gf_l.append(row["home_score"] if is_h else row["away_score"])
            gc_l.append(row["away_score"] if is_h else row["home_score"])
            wr="H" if is_h else "A"
            res_l.append("W" if row["result"]==wr else("D" if row["result"]=="D" else "L"))
            opp_l.append(row["away_team"] if is_h else row["home_team"])

        cm_={"W":"#4ade80","D":"#facc15","L":"#f87171"}
        fig_f=go.Figure()
        fig_f.add_trace(go.Bar(x=list(range(1,len(gf_l)+1)),y=gf_l,name="GF",
            marker_color=[cm_[r] for r in res_l],customdata=opp_l,
            hovertemplate="%{customdata}<br>GF:%{y}<extra></extra>"))
        fig_f.add_trace(go.Bar(x=list(range(1,len(gc_l)+1)),y=[-g for g in gc_l],
            name="GC",marker_color="#475569",customdata=opp_l,
            hovertemplate="%{customdata}<br>GC:%{y}<extra></extra>"))
        fig_f.update_layout(barmode="relative",height=360,
            title=f"Historial de goles — {team_s} (🟢Victoria 🟡Empate 🔴Derrota)",
            plot_bgcolor="#0f172a",paper_bgcolor="#0f172a",font_color="white",yaxis_title="Goles")
        st.plotly_chart(fig_f,use_container_width=True)

# ════════════════════════════════════════════════════════════════════
# TAB 4 — HEAD TO HEAD
# ════════════════════════════════════════════════════════════════════
with tab4:
    st.subheader("⚔️ Head-to-Head")
    h1,hm,h2=st.columns([5,1,5])
    with h1: h2h_home=st.selectbox("Equipo A",teams,key="h2h_h")
    with hm: st.markdown("<br><br><div style='text-align:center;'>⚔️</div>",unsafe_allow_html=True)
    with h2: h2h_away=st.selectbox("Equipo B",[t for t in teams if t!=h2h_home],key="h2h_a")
    comp_h2h=st.selectbox("Temporada/s",["(Todas)"]+comp_ids,key="h2h_comp")
    cid_h2h=None if comp_h2h=="(Todas)" else comp_h2h

    if st.button("🔍 Ver historial H2H",use_container_width=True):
        h2h_data=system.h2h(h2h_home,h2h_away,cid_h2h)
        s_=h2h_data["summary"]; df_h2h=h2h_data["matches"]
        if df_h2h.empty:
            st.info("No hay enfrentamientos entre estos equipos.")
        else:
            # Resumen
            st.markdown('<div class="section-hdr">Resumen H2H</div>',unsafe_allow_html=True)
            sm1,sm2,sm3,sm4,sm5=st.columns(5)
            sm1.metric("Partidos",s_["total"])
            sm2.metric(f"✅ {h2h_home}",s_.get(f"{h2h_home}_wins",0))
            sm3.metric("🤝 Empates",s_["draws"])
            sm4.metric(f"✅ {h2h_away}",s_.get(f"{h2h_away}_wins",0))
            sm5.metric("Goles/pj",s_["avg_goals"])

            # Pie chart
            fig_pie=go.Figure(go.Pie(
                labels=[f"🏠 {h2h_home}","🤝 Empate",f"✈️ {h2h_away}"],
                values=[s_.get(f"{h2h_home}_wins",0),s_["draws"],s_.get(f"{h2h_away}_wins",0)],
                marker_colors=["#4ade80","#facc15","#f87171"],
                textinfo="label+percent",hole=0.4))
            fig_pie.update_layout(height=320,plot_bgcolor="#0f172a",paper_bgcolor="#0f172a",font_color="white")
            st.plotly_chart(fig_pie,use_container_width=True)

            # Tabla de partidos
            st.markdown('<div class="section-hdr">Historial de partidos</div>',unsafe_allow_html=True)
            df_show=df_h2h[["competition_id","date","home_team","home_score","away_score","away_team","result"]].copy()
            df_show["date"]=df_show["date"].dt.strftime("%Y-%m-%d")
            df_show["Marcador"]=df_show["home_score"].astype(int).astype(str)+"-"+df_show["away_score"].astype(int).astype(str)
            res_map={"H":f"✅ {h2h_home}","D":"🤝 Empate","A":f"✅ {h2h_away}"}
            df_show["Ganador"]=df_show.apply(
                lambda r: (f"✅ {r['home_team']}" if r["result"]=="H" else
                           ("🤝 Empate" if r["result"]=="D" else f"✅ {r['away_team']}")),axis=1)
            st.dataframe(df_show[["date","competition_id","home_team","Marcador","away_team","Ganador"]].rename(columns={
                "date":"Fecha","competition_id":"Liga","home_team":"Local","away_team":"Visitante"}),
                use_container_width=True,hide_index=True)

# ════════════════════════════════════════════════════════════════════
# TAB 5 — DATOS CRUDOS
# ════════════════════════════════════════════════════════════════════
with tab5:
    st.subheader("🔬 Datos crudos y resultados históricos")
    fc1,fc2,fc3,fc4=st.columns(4)
    with fc1: comp_raw=st.selectbox("Temporada",["(Todas)"]+comp_ids,key="raw_comp")
    with fc2: team_raw=st.selectbox("Equipo",["(Todos)"]+teams,key="raw_team")
    with fc3: res_raw=st.selectbox("Resultado",["(Todos)","H","D","A"],key="raw_res")
    with fc4: limit_raw=st.slider("Máx. filas",50,1000,200,step=50)

    cid_r=None if comp_raw=="(Todas)" else comp_raw
    team_r=None if team_raw=="(Todos)" else team_raw
    res_r=None if res_raw=="(Todos)" else res_raw

    df_raw=system.raw_matches(cid_r,team_r,res_r)
    df_raw=df_raw.head(limit_raw)

    st.markdown(f"Mostrando **{len(df_raw)}** partidos")
    st.dataframe(df_raw,use_container_width=True,hide_index=True)

    if not df_raw.empty:
        # ── Estadísticas rápidas ──
        st.markdown('<div class="section-hdr">Estadísticas de la selección</div>',unsafe_allow_html=True)
        s1,s2,s3,s4,s5=st.columns(5)
        res_counts=df_raw["Res"].value_counts() if "Res" in df_raw.columns else pd.Series()
        s1.metric("Total",len(df_raw))
        s1.metric("🏠 Victorias local",res_counts.get("H",0))
        s2.metric("🤝 Empates",res_counts.get("D",0))
        s2.metric("✈️ Victorias visita",res_counts.get("A",0))

        csv=df_raw.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Descargar CSV",csv,"datos_crudos.csv","text/csv")

# ════════════════════════════════════════════════════════════════════
# TAB 6 — BACKTESTING
# ════════════════════════════════════════════════════════════════════
with tab6:
    st.subheader("🧪 Backtesting — Precisión histórica del modelo")
    n_bt=st.slider("Últimos N partidos para evaluar",50,300,100,step=25)
    if st.button("▶️ Ejecutar backtesting",use_container_width=True):
        with st.spinner("Evaluando..."):
            df_bt=system.backtest(n_bt)
        if df_bt.empty:
            st.warning("Datos insuficientes para backtesting.")
        else:
            acc=df_bt["correct"].mean()
            st.markdown(f"### Precisión global: **{acc:.1%}** ({df_bt['correct'].sum()}/{len(df_bt)} aciertos)")

            # Métricas por resultado
            for res,label in [("H","🏠 Local gana"),("D","🤝 Empate"),("A","✈️ Visita gana")]:
                sub=df_bt[df_bt["real"]==res]
                if len(sub):
                    st.metric(label,f"{(sub['correct'].mean()):.1%}",f"{len(sub)} partidos evaluados")

            # Gráfico confianza vs acierto
            fig_bt=px.histogram(df_bt,x="conf",color="correct",nbins=20,
                color_discrete_map={True:"#4ade80",False:"#f87171"},
                title="Distribución de confianza (verde=acierto, rojo=fallo)",
                labels={"conf":"Confianza del modelo","count":"Partidos"})
            fig_bt.update_layout(plot_bgcolor="#0f172a",paper_bgcolor="#0f172a",
                font_color="white",height=340)
            st.plotly_chart(fig_bt,use_container_width=True)

            # Tabla detalle
            st.dataframe(df_bt[["date","home","away","real","pred","correct","conf",
                                 "p_H","p_D","p_A"]].rename(columns={
                "date":"Fecha","home":"Local","away":"Visitante","real":"Real",
                "pred":"Predicho","correct":"✓","conf":"Confianza",
                "p_H":"P(H)","p_D":"P(D)","p_A":"P(A)"}),
                use_container_width=True,hide_index=True)

            csv_bt=df_bt.to_csv(index=False).encode("utf-8")
            st.download_button("⬇️ Descargar resultados CSV",csv_bt,"backtest_results.csv","text/csv")

# ════════════════════════════════════════════════════════════════════
# TAB 7 — MODELO / IMPORTANCIA DE VARIABLES
# ════════════════════════════════════════════════════════════════════
with tab7:
    st.subheader("📈 Modelo e importancia de variables")
    imp=system.ml.importances()
    if imp:
        df_imp=pd.DataFrame({"Feature":list(imp.keys()),"Importancia":list(imp.values())})
        df_imp=df_imp.sort_values("Importancia",ascending=True).tail(20)
        fig_i=go.Figure(go.Bar(x=df_imp["Importancia"]*100,y=df_imp["Feature"],
            orientation="h",marker_color="#3b82f6",
            text=[f"{v:.2f}%" for v in df_imp["Importancia"]*100],textposition="outside"))
        fig_i.update_layout(title="Top 20 variables más influyentes",height=550,
            xaxis_title="Importancia (%)",
            plot_bgcolor="#0f172a",paper_bgcolor="#0f172a",font_color="white")
        st.plotly_chart(fig_i,use_container_width=True)

        with st.expander("📖 Glosario de variables"):
            st.markdown("""
| Prefijo | Descripción |
|---|---|
| `h_` | Historial general del **local** (últimos 5 partidos) |
| `hh_` | Historial del local jugando **en casa** |
| `a_` | Historial general del **visitante** |
| `aa_` | Historial del visitante jugando **de visita** |
| `h2h_` | Head-to-head entre ambos (últimos 6 encuentros) |
| `_gf` | Goles a favor promedio |
| `_gc` | Goles en contra promedio |
| `_pts` | Puntos por partido |
| `_wins/_draws/_losses` | Tasa de victorias / empates / derrotas |
| `_hw/_d/_aw` | En H2H: victorias local / empates / victorias visita |
""")

        # Info modelo
        s=st.session_state.system.stats
        st.markdown("---")
        st.markdown("### 🤖 Parámetros del modelo")
        st.json({"Algoritmo":"Gradient Boosting Classifier",
                 "n_estimators":300,"max_depth":4,"learning_rate":0.05,"subsample":0.8,
                 "CV_folds":5,"CV_accuracy":s.get("cv_score","—"),
                 "Partidos entrenamiento":s.get("played","—"),
                 "Equipos":len(s.get("teams",[])),"Temporadas":s.get("seasons",[])})
