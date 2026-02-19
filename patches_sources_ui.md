# app.py — Parches de indicadores de fuentes de datos

## PARCHE 1: Sidebar badge — 3 fuentes con LED verde/gris
```python
# ── REEMPLAZAR el bloque del badge de modelo activo en el sidebar ──────
# BUSCAR esta línea:
#   st.markdown( f"<div class='model-badge'>🎯 <b>Modelo activo</b><br> ...
# REEMPLAZAR POR:

    if orch.system.trained:
        ss    = orch.system.stats
        c_meta = catalog.get("competitions",{}).get(ss.get("comp_slug",""),{})
        cname  = c_meta.get("name", ss.get("comp_slug","—"))
        stype_ = c_meta.get("season_type","single")
        yr_    = str(ss.get("predict_season",""))
        s_lbl  = (f"{yr_}/{str(int(yr_)+1)[2:]}" if stype_=="split" and yr_.isdigit() else yr_)

        # Estados de todas las fuentes de datos
        xg_ok  = orch.state.get("has_xg", False)
        ss_ok  = orch.state.get("has_ss", False)
        fd_ok  = orch.system.trained
        xg_nm  = orch.state.get("xg_source","—") or "—"
        ss_n   = orch.state.get("ss_teams", 0)
        cv_v   = ss.get("cv_score", 0)
        played = ss.get("played_current", 0)

        xg_badge = f"🟢 xG:{xg_nm}" if xg_ok else "⚪ sin xG"
        ss_badge = f"🟢 SS:{ss_n}eq" if ss_ok else "⚪ sin SS"

        st.markdown(
            f"<div class='model-badge'>"
            f"🎯 <b>Modelo activo</b><br>"
            f"{cname} {s_lbl}<br>"
            f"<span style='color:#86efac'>"
            f"CV:{cv_v:.0%} · {played}pj</span><br>"
            f"<div style='margin-top:.3rem;display:flex;gap:.4rem;flex-wrap:wrap;'>"
            f"<span style='background:#0f2d0f;border:1px solid #16a34a;"
            f"border-radius:12px;padding:1px 8px;font-size:.72rem'>🗂 FD ✅</span>"
            f"<span style='background:{'#0f2d0f' if xg_ok else '#1c1917'};"
            f"border:1px solid {'#16a34a' if xg_ok else '#475569'};"
            f"border-radius:12px;padding:1px 8px;font-size:.72rem'>{xg_badge}</span>"
            f"<span style='background:{'#0f2d0f' if ss_ok else '#1c1917'};"
            f"border:1px solid {'#16a34a' if ss_ok else '#475569'};"
            f"border-radius:12px;padding:1px 8px;font-size:.72rem'>{ss_badge}</span>"
            f"</div></div>",
            unsafe_allow_html=True,
        )
        ls = orch.state.get("last_sync_time","")
        if ls: st.caption(f"Última sync: {ls}")
```

## PARCHE 2: render_prediction() — bloque Fuentes activas
```python
# ── AGREGAR al inicio de render_prediction(), ANTES de los 3 botones ──
# (dentro de la función, primera cosa que se muestra)

    # ── Fuentes de datos usadas ──────────────────────────────────────
    fd_ok  = True  # siempre si hay predicción
    xg_ok  = pred.get("has_xg", False)
    ss_ok  = orch.state.get("has_ss", False)
    nm_ok  = True  # name_normalizer siempre activo

    src_items = [
        ("🗂 FixtureDownload", "#16a34a", True,   "Histórico de partidos"),
        ("📊 xG Understat/FBref", "#16a34a", xg_ok, "Expected Goals por equipo"),
        ("⚽ SoccerStats",     "#16a34a", ss_ok, f"Over2.5%, BTTS%, H/A, Timing"),
        ("🔤 Name Normalizer", "#3b82f6", nm_ok,  "Aliases entre fuentes"),
    ]
    badges_html = "".join(
        f"<span style=\"background:{'#0f2d0f' if active else '#1c1917'};"
        f"border:1px solid {color if active else '#334155'};"
        f"border-radius:12px;padding:2px 10px;font-size:.74rem;"
        f"margin:2px;color:{'#4ade80' if active else '#64748b'}\">"
        f"{'✅' if active else '⚪'} {label}"
        f"</span>"
        for label, color, active, _ in src_items
    )
    st.markdown(
        f"<div style=\"margin-bottom:.6rem\">"
        f"<span style=\"font-size:.75rem;color:#64748b;text-transform:uppercase;"
        f"letter-spacing:.06em\">Fuentes activas:</span><br>"
        f"{badges_html}"
        f"</div>",
        unsafe_allow_html=True,
    )
    if ss_ok:
        ss_n = orch.state.get("ss_teams", 0)
        xg_nm = orch.state.get("xg_source","—")
        st.caption(
            f"SoccerStats: {ss_n} equipos indexados · "
            f"xG source: {xg_nm} · "
            f"λ = 40% hist + 40% xG + 20% SS"
        )
```

## PARCHE 3: Tab 0 Buscar — caption con fuentes del partido
```python
# ── AGREGAR después de st.success("Partido encontrado...") en Tab 0 ──

                # Indicadores de fuentes disponibles para este partido
                xg_disp = orch.state.get("has_xg", False)
                ss_disp = orch.state.get("has_ss", False)
                icons = []
                icons.append("🗂 FD")
                if xg_disp: icons.append(f"📊 xG:{orch.state.get('xg_source','?')}")
                if ss_disp: icons.append(f"⚽ SoccerStats:{orch.state.get('ss_teams',0)}eq")
                icons.append("🔤 NormNames")
                st.caption("Fuentes: " + " · ".join(icons))
```

## PARCHE 4: Tab 3 Stats — sección SoccerStats por equipo
```python
# ── AGREGAR al final de Tab 3 (Estadísticas), después de la racha ──

            # ── Datos de SoccerStats (si disponibles) ─────────────────
            if orch.state.get("has_ss") and std_:
                st.markdown(
                    "<div class='section-hdr'>SoccerStats</div>",
                    unsafe_allow_html=True,
                )
                ss1,ss2,ss3,ss4,ss5,ss6 = st.columns(6)
                ss1.metric("GF/pj (SS)",  std_.get("ss_home_gf_pg","—"))
                ss2.metric("GA/pj (SS)",  std_.get("ss_home_ga_pg","—"))
                ss3.metric("Over2.5% (SS)",
                    f"{std_.get('ss_home_over25_pct',0):.0f}%" if std_.get("ss_home_over25_pct") else "—")
                ss4.metric("BTTS% (SS)",
                    f"{std_.get('ss_home_btts_pct',0):.0f}%" if std_.get("ss_home_btts_pct") else "—")
                ss5.metric("Goles tarde %", std_.get("late_goals_pct","—"))
                ss6.metric("Fuente", "SoccerStats")
                st.caption(
                    "Datos de soccerstats.com · Actualizado cada 24h · "
                    f"Caché: fd_cache/soccerstats/{orch.system.stats.get('comp_slug','')}*"
                )
```

