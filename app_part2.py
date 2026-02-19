
# ════════ SIDEBAR ═════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## ⚽ Football Predictor Pro")
    st.caption(f"v4.2 · {catalog.get('total_comps', 0)} ligas indexadas")

    # ── [BUG-2] MODELO ACTIVO ─────────────────────────────────────────
    if orch.system.trained:
        ss    = orch.system.stats
        c_meta = catalog.get("competitions", {}).get(ss.get("comp_slug", ""), {})
        cname  = c_meta.get("name", ss.get("comp_slug", "—"))
        stype_ = c_meta.get("season_type", "single")
        yr_    = str(ss.get("predict_season", ""))
        s_lbl  = (f"{yr_}/{str(int(yr_)+1)[2:]}" if stype_ == "split" and yr_.isdigit()
                  else yr_)
        xg_ic  = "🟢" if orch.state.get("has_xg") else "⚪"
        xg_nm  = orch.state.get("xg_source", "sin xG") or "sin xG"
        cv_v   = ss.get("cv_score", 0)
        played = ss.get("played_current", 0)
        st.markdown(
            f"<div class='model-badge'>"
            f"🎯 <b>Modelo activo</b><br>"
            f"{cname} {s_lbl}&nbsp;&nbsp;"
            f"<span style='color:#86efac'>"
            f"{xg_ic} xG:{xg_nm} · CV:{cv_v:.0%} · {played} pj"
            f"</span></div>",
            unsafe_allow_html=True,
        )
        last_sync = orch.state.get("last_sync_time", "")
        if last_sync:
            st.caption(f"Última sync: {last_sync}")
    else:
        st.markdown(
            "<div class='status-warn'>⚠️ Sin modelo cargado. Sync Mínimo para empezar.</div>",
            unsafe_allow_html=True,
        )
    st.markdown("---")

    # ── SYNC ──────────────────────────────────────────────────────────
    st.markdown("### 🔄 Sincronización")
    c1, c2 = st.columns(2)
    with c1:
        sync_min_btn = st.button(
            "⚡ Sync
Mínimo", use_container_width=True,
            help="Descarga la liga seleccionada + xG automático",
        )
    with c2:
        sync_deep_bg = st.button(
            "🌍 Sync
Deep", use_container_width=True,
            help="Todas las ligas del mundo en background",
        )
    force_dl = st.checkbox("Forzar re-descarga", key="force_dl")

    if sync_deep_bg:
        if not orch._deep_running:
            ok = orch.sync_deep_background(force=force_dl)
            if ok:
                st.info("🌍 Sync Deep iniciado. Progreso visible en Tab 🔍.")
        else:
            st.warning("Ya hay un Sync Deep en curso.")

    st.markdown("---")

    # ── LIGA PARA ENTRENAR ────────────────────────────────────────────
    st.markdown("### 🏆 Liga para entrenar")
    region_list = list(regions.keys())
    region      = st.selectbox("Región", region_list, key="sb_region")
    comp_labels = regions.get(region, [])
    comp_label  = st.selectbox("Competición", comp_labels, key="sb_comp")
    comp_info   = get_comp_by_label(catalog, comp_label)
    slug        = comp_info.get("slug", "")
    season_opts = get_season_options(comp_info)
    train_defs  = get_train_defaults(comp_info)
    predict_def = get_predict_default_label(comp_info)
    stype_sel   = comp_info.get("season_type", "single")
    xg_src_sel  = get_xg_source(slug)
    xg_lbl      = f"🟢 {xg_src_sel}" if xg_src_sel else "⚪ Sin cobertura"
    st.caption(f"Tipo: {stype_sel} · xG: {xg_lbl}")

    train_labels = st.multiselect(
        "Temporadas de entrenamiento",
        options=list(season_opts.keys()),
        default=[l for l in train_defs if l in season_opts],
        key="tr_lbl",
    )
    # [UX-4] Estimación de partidos y tiempo
    if train_labels:
        n_teams = comp_info.get("num_teams", 18)
        n_match = len(train_labels) * (n_teams * (n_teams - 1))
        t_est   = max(10, round(len(train_labels) * 4.5))
        st.caption(f"≈ {n_match:,} partidos · ~{t_est}s de carga")

    predict_lbl = st.selectbox(
        "Temporada activa",
        options=list(season_opts.keys()),
        index=(list(season_opts.keys()).index(predict_def)
               if predict_def in season_opts else 0),
        key="pred_lbl",
    )

    if sync_min_btn:
        if not train_labels:
            st.error("Selecciona al menos una temporada.")
        else:
            train_yrs = [season_opts[l] for l in train_labels if l in season_opts]
            pred_yr   = season_opts.get(predict_lbl, 0)
            pb  = st.progress(0.0)
            sm  = st.empty()
            def _pf(msg, v): sm.caption(msg); pb.progress(float(v))
            ok, msg = orch.sync_minimum(slug, train_yrs, pred_yr,
                                        force=force_dl, progress_fn=_pf)
            pb.empty(); sm.empty()
            if ok:
                st.success(f"Listo: {msg}"); st.rerun()
            else:
                st.error(f"Error: {msg}")

    st.markdown("---")

    # ── AUTO-UPDATE ───────────────────────────────────────────────────
    st.markdown("### ⚙️ Auto-update")
    auto_on = st.toggle("Activar", value=st.session_state.auto_on)
    if auto_on != st.session_state.auto_on:
        st.session_state.auto_on = auto_on
        st.rerun()
    if orch.system.trained:
        if st.button("Chequear nuevos resultados", use_container_width=True):
            try:
                from auto_updater import AutoUpdater
                upd = AutoUpdater(orch.system, orch.system.comp_ids)
                r   = upd.check_all()
                nm  = r.get("new_matches", 0)
                orch.search.build(orch.system.df_proc, catalog)
                st.success(f"{nm} nuevos partidos" if nm > 0 else "✅ Al día")
            except Exception as e:
                st.error(str(e))


# ════════ CABECERA ═════════════════════════════════════════════════════
st.title("⚽ Football Predictor Pro")
st.caption("Búsqueda global · xG automático · Multi-liga · Tiempo real")
teams = orch.get_teams()

_TAB_NAMES = [
    "🔍 Buscar & Analizar", "⚡ Predicción Manual", "📊 Tabla",
    "👤 Estadísticas",      "⚔️ H2H",              "📅 Próximos",
    "📁 Datos Crudos",      "🔬 Backtesting",       "📈 xG Análisis",
    "🌐 Catálogo",
]
(tab_search, tab_pred, tab_table, tab_stats,
 tab_h2h, tab_next, tab_raw, tab_bt, tab_xg, tab_cat) = st.tabs(_TAB_NAMES)


# ══════════════════════════════════════════════════════════════════════
# TAB 0: BUSCAR & ANALIZAR
# ══════════════════════════════════════════════════════════════════════
with tab_search:
    st.markdown(
        "<div class='section-hdr'>Búsqueda Universal</div>",
        unsafe_allow_html=True,
    )
    st.caption(
        'Equipo, liga o partido · Ej: "Bayern vs Dortmund" · "Chelsea" · "Premier League"'
    )

    # [F-2] Historial como chips
    if st.session_state.search_history:
        chips_html = " ".join(
            f"<span class='chip'>⚽ {e['home'][:9]} vs {e['away'][:9]}</span>"
            for e in st.session_state.search_history
        )
        st.markdown(f"<div>{chips_html}</div>", unsafe_allow_html=True)
        chip_cols = st.columns(len(st.session_state.search_history))
        for i, h_entry in enumerate(st.session_state.search_history):
            with chip_cols[i]:
                if st.button(
                    f"↩ {h_entry['home'][:8]} vs {h_entry['away'][:8]}",
                    key=f"chip_{i}", use_container_width=True,
                ):
                    st.session_state["prefill_search"] = h_entry["label"]
                    st.rerun()

    prefill = st.session_state.pop("prefill_search", "")
    query   = st.text_input(
        "", value=prefill,
        placeholder="🔍  Bayern Munich, Chelsea vs Arsenal, La Liga...",
        key="global_search", label_visibility="collapsed",
    )
    # [BUG-3] Limpiar comillas
    query = query.strip('"'').strip() if query else ""

    # Sync Deep progress
    if orch._deep_running:
        s2 = orch.get_status()
        prog = float(s2.get("deep_progress", 0.0))
        st.markdown(
            f"<div class='status-run'>⚡ Sync Deep en curso: "
            f"{s2.get('deep_message', '')} — {prog:.0%}</div>",
            unsafe_allow_html=True,
        )
        st.progress(prog)

    if not orch.search.built and not query:
        st.info(
            "💡 Haz **Sync Mínimo** en el sidebar para indexar la liga activa, "
            "o **Sync Deep** para todas las ligas del mundo."
        )

    # ── LÓGICA DE BÚSQUEDA ────────────────────────────────────────────
    if query and orch.search.built:
        is_match_q = bool(re.search(r"\bvs\.?\b|\s+-\s+", query, re.I))

        if is_match_q:
            parts  = re.split(r"\s+vs\.?\s+|\s+-\s+", query.strip(),
                               maxsplit=1, flags=re.I)
            home_q = parts[0].strip()
            away_q = parts[1].strip() if len(parts) > 1 else ""
            mi     = orch.search.find_match(home_q, away_q)

            if mi:
                comp_friendly = friendly_comp(mi["comp_id"])
                estado_lbl    = "🕓 Pendiente" if not mi["played"] else "✅ Jugado"
                st.success(
                    f"Partido encontrado: **{mi['home']}** vs **{mi['away']}**"
                )
                # [UX-2] Card compacta
                st.markdown(
                    f"<div class='match-card'>"
                    f"<b>{mi['home']}</b>"
                    f"<span style='color:#64748b;padding:0 .7rem'>vs</span>"
                    f"<b>{mi['away']}</b>"
                    f"<span style='color:#3b82f6;margin-left:1rem'>{comp_friendly}</span>"
                    f"<span style='color:#94a3b8;margin-left:.6rem;font-size:.82rem'>"
                    f"{estado_lbl}</span>"
                    f"</div>",
                    unsafe_allow_html=True,
                )
                is_same = (
                    orch.system.trained
                    and mi["comp_id"].startswith(
                        orch.system.stats.get("comp_slug", "__")
                    )
                )
                if is_same:
                    if st.button(
                        "⚡ Analizar este partido ahora",
                        type="primary", use_container_width=True,
                    ):
                        push_history(mi["home"], mi["away"], mi["comp_id"])
                        with st.spinner("Calculando Poisson + GBM + xG..."):
                            analysis = orch.analyze_match(mi["home"], mi["away"])
                        if analysis:
                            st.session_state.pending_pred = {
                                "home":     mi["home"],
                                "away":     mi["away"],
                                "analysis": analysis,
                            }
                            pred = analysis["prediction"]
                            hs   = analysis.get("home_stats")
                            as_  = analysis.get("away_stats")
                            st.markdown("---")
                            st.markdown(
                                f"#### {mi['home']} vs {mi['away']}"
                                f" · {comp_friendly}"
                            )
                            # Inline compact result
                            render_prediction(pred, mi["home"], mi["away"],
                                              hs, as_, compact=True)
                            st.info(
                                "💡 Abre el Tab **⚡ Predicción Manual** "
                                "para heatmap, todos los mercados y xG completo."
                            )
                else:
                    slug_q = mi["comp_id"].rsplit("-", 1)[0]
                    yr_q   = int(mi["comp_id"].rsplit("-", 1)[-1])
                    comp_friendly2 = friendly_comp(mi["comp_id"])
                    st.warning(
                        f"Liga **{comp_friendly2}** no está entrenada. Cárgala:"
                    )
                    if st.button(
                        f"📥 Cargar {comp_friendly2} ahora",
                        use_container_width=True,
                    ):
                        train_yrs2 = [yr_q - 2, yr_q - 1, yr_q]
                        pb2 = st.progress(0.0); sm2 = st.empty()
                        def _pf2(msg, v): sm2.caption(msg); pb2.progress(float(v))
                        ok2, msg2 = orch.on_demand_load(
                            slug_q, train_yrs2, yr_q, progress_fn=_pf2
                        )
                        pb2.empty(); sm2.empty()
                        if ok2:
                            st.success("Listo"); st.rerun()
                        else:
                            st.error(msg2)
            else:
                st.warning(
                    "Partido no encontrado en el índice. "
                    "Intenta Sync Deep o revisa los nombres."
                )
                for label, q in [("Local", home_q), ("Visitante", away_q)]:
                    hits = orch.search.search_teams(q, top_n=3)
                    if hits:
                        names = ", ".join(
                            f"{e['team']} ({friendly_comp(e['comp_id'])})"
                            for e in hits
                        )
                        st.caption(f"Similares {label}: {names}")

        else:
            team_hits = orch.search.search_teams(query, top_n=10)
            if team_hits:
                st.markdown(f"**{len(team_hits)} resultados para:** `{query}`")
                for hit in team_hits:
                    pct = int(hit["score"] * 100)
                    cfr = friendly_comp(hit["comp_id"])
                    st.markdown(
                        f"<div style='background:#1e293b;border:1px solid #334155;"
                        f"border-radius:8px;padding:.45rem .8rem;margin:.2rem 0;'>"
                        f"⚽ <b>{hit['team']}</b>"
                        f" — <span style='color:#3b82f6'>{cfr}</span>"
                        f"<span style='color:#475569;font-size:.78rem'> ({pct}%)</span>"
                        f"</div>",
                        unsafe_allow_html=True,
                    )
                best   = team_hits[0]["team"]
                df_up  = orch.search.upcoming_matches(best, limit=5)
                df_rc  = orch.search.search_matches(query, top_n=5)
                if not df_up.empty:
                    st.markdown(f"**Próximos fixtures — {best}:**")
                    st.dataframe(df_up, use_container_width=True, hide_index=True)
                if not df_rc.empty:
                    st.markdown(f"**Últimos resultados — {best}:**")
                    st.dataframe(df_rc, use_container_width=True, hide_index=True)
            else:
                st.info("Sin resultados. Prueba otro nombre o haz Sync Deep.")

    elif query and not orch.search.built:
        st.warning("Índice no construido. Haz Sync Mínimo o Sync Deep primero.")


# ══════════════════════════════════════════════════════════════════════
# TAB 1: PREDICCIÓN MANUAL  [F-1] pre-carga desde pending_pred
# ══════════════════════════════════════════════════════════════════════
with tab_pred:
    if not orch.system.trained:
        st.info("Carga una liga desde el sidebar para empezar.")
    else:
        st.markdown(
            "<div class='section-hdr'>Predicción Manual</div>",
            unsafe_allow_html=True,
        )
        pp       = st.session_state.get("pending_pred")
        def_home = (pp["home"] if pp and pp.get("home") in teams
                    else (teams[0] if teams else ""))
        def_away = (pp["away"] if pp and pp.get("away") in teams
                    else (teams[1] if len(teams) > 1 else ""))
        if pp:
            st.info(
                f"⚡ Cargado desde búsqueda: **{pp['home']}** vs **{pp['away']}**"
            )
        c1, cm, c2 = st.columns([5, 1, 5])
        with c1:
            home = st.selectbox(
                "Local", teams,
                index=teams.index(def_home) if def_home in teams else 0,
                key="pm_h",
            )
        with cm:
            st.markdown(
                "<br><br><div style='text-align:center'>vs</div>",
                unsafe_allow_html=True,
            )
        with c2:
            away_opts = [t for t in teams if t != home]
            def_a2    = def_away if def_away in away_opts else (away_opts[0] if away_opts else "")
            away = st.selectbox(
                "Visitante", away_opts,
                index=away_opts.index(def_a2) if def_a2 in away_opts else 0,
                key="pm_a",
            )

        auto_run = pp is not None
        if st.button("Predecir", type="primary", use_container_width=True,
                     key="pm_btn") or auto_run:
            if auto_run and pp and "analysis" in pp:
                analysis = pp["analysis"]
                st.session_state.pending_pred = None
            else:
                with st.spinner("Calculando Poisson + GBM + xG..."):
                    analysis = orch.analyze_match(home, away)

            if analysis:
                pred = analysis["prediction"]
                hs   = analysis.get("home_stats")
                as_  = analysis.get("away_stats")
                push_history(home, away, orch.system.predict_comp_id)
                # Header del partido
                h1, hm, h2 = st.columns([4, 2, 4])
                with h1:
                    st.markdown(
                        f"<div class='big-num'>{home}</div>"
                        f"<div class='lbl'>Local · λ {pred['lambda_home']}</div>",
                        unsafe_allow_html=True,
                    )
                    if hs:
                        st.caption(
                            f"PPP {hs.get('total_PPP',0)} · "
                            f"GF {hs.get('avg_gf',0)}/pj · "
                            f"Racha {hs.get('streak','—')}"
                        )
                with hm:
                    xg_flag = ""
                    if pred.get("has_xg"):
                        xg_flag = (
                            f"<span class='xg-badge'>"
                            f"xG·{orch.state.get('xg_source','?')}"
                            f"</span>"
                        )
                    st.markdown(
                        f"<br><div style='text-align:center'>"
                        f"<div style='font-size:1.3rem'>VS</div>"
                        f"{xg_flag}"
                        f"</div>",
                        unsafe_allow_html=True,
                    )
                with h2:
                    st.markdown(
                        f"<div class='big-num'>{away}</div>"
                        f"<div class='lbl'>Visitante · λ {pred['lambda_away']}</div>",
                        unsafe_allow_html=True,
                    )
                    if as_:
                        st.caption(
                            f"PPP {as_.get('total_PPP',0)} · "
                            f"GF {as_.get('avg_gf',0)}/pj · "
                            f"Racha {as_.get('streak','—')}"
                        )
                st.markdown("---")
                render_prediction(pred, home, away, hs, as_, compact=False)
