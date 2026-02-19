# Patch Notes — Integración Name Normalizer + SoccerStats

## orchestrator.py
```python

# ── PATCH orchestrator.py — agregar SoccerStats al sync_minimum ───────
# En la clase DataOrchestrator, dentro de sync_minimum(), agregar
# DESPUÉS del bloque de xG scraping existente:

    # ── SoccerStats enrichment (opcional, no bloquea) ─────────────────
    try:
        from soccerstats_scraper import SoccerStatsScraper
        ss_scraper = SoccerStatsScraper()
        ss_data    = ss_scraper.get_all(slug, force=force)
        if ss_data["stats"]:
            self.system.df_proc = ss_scraper.apply_to_df(
                self.system.df_proc, slug
            )
            self.state["has_ss"] = True
            self.state["ss_teams"] = len(ss_data["stats"])
            progress_fn("SoccerStats: datos de goles y timing aplicados", 0.78)
        else:
            self.state["has_ss"] = False
    except Exception as e:
        log.warning(f"SoccerStats enrich falló (no crítico): {e}")
        self.state["has_ss"] = False

```

## predictor.py
```python

# ── PATCH predictor.py — usar ss_over25_pct y ss_btts_pct como features ──
# En DataCollector._collect_team_stats() o _xg_stats(), agregar:

    def _ss_stats(self, df, team, venue):
        """Extrae features de SoccerStats para un equipo."""
        sfx  = "home" if venue == "home" else "away"
        mask = (df["home_team"] == team) if venue=="home" else (df["away_team"] == team)
        rows = df[mask & df[f"ss_{sfx}_gf_pg"].notna()]
        if rows.empty:
            return {}
        last = rows.sort_values("date").tail(5)
        return {
            "ss_gf_pg":     float(last[f"ss_{sfx}_gf_pg"].mean()),
            "ss_ga_pg":     float(last[f"ss_{sfx}_ga_pg"].mean()),
            "ss_over25_pct":float(last[f"ss_{sfx}_over25_pct"].mean()) if f"ss_{sfx}_over25_pct" in last else 0.0,
            "ss_btts_pct":  float(last[f"ss_{sfx}_btts_pct"].mean())   if f"ss_{sfx}_btts_pct"   in last else 0.0,
        }

# Usar en lambda Poisson híbrido (agregando tercer componente):
#   lambda_h = 0.40 * hist_gf + 0.40 * xg_gf + 0.20 * ss_gf_pg
#   (cuando todos los datos estén disponibles)

```
