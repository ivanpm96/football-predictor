# predictor_xg_patch.py
# INSTRUCCIONES: Reemplaza los metodos indicados en predictor.py
# Este archivo documenta los diffs exactos a aplicar.

# ════════════════════════════════════════════════════════════════════
# CAMBIO 1: En clase FeatureEngineer — metodo build()
# Agregar columnas xG a las features si estan disponibles
# ════════════════════════════════════════════════════════════════════

# AGREGAR este metodo a FeatureEngineer:
XG_FEAT_METHOD = '''
    def _xg_stats(self, matches, team, pfx):
        z = {f"{pfx}xgf": 0., f"{pfx}xgc": 0.,
             f"{pfx}xg_diff": 0., f"{pfx}overperf": 0.}
        has = matches[matches["has_xg"] == True] if "has_xg" in matches.columns else pd.DataFrame()
        if has.empty: return z
        xgf_col = "home_xg" if "home_xg" in has.columns else None
        xgc_col = "away_xg" if "away_xg" in has.columns else None
        gf_col  = "home_score"
        gc_col  = "away_score"
        xgf=xgc=gf=gc=0
        for _,m in has.iterrows():
            is_h = m["home_team"]==team
            xgf += (m["home_xg"] if is_h else m["away_xg"]) if xgf_col else 0
            xgc += (m["away_xg"] if is_h else m["home_xg"]) if xgc_col else 0
            gf  += m["home_score"] if is_h else m["away_score"]
            gc  += m["away_score"] if is_h else m["home_score"]
        n = len(has)
        return {
            f"{pfx}xgf":      round(xgf/n,3),
            f"{pfx}xgc":      round(xgc/n,3),
            f"{pfx}xg_diff":  round((xgf-xgc)/n,3),
            f"{pfx}overperf": round((gf-xgf)/n,3),
        }
'''

# ════════════════════════════════════════════════════════════════════
# CAMBIO 2: En clase PoissonModel — metodo fit()
# Usar lambda hibrido: 50% goles reales + 50% xG (si disponible)
# ════════════════════════════════════════════════════════════════════

POISSON_FIT_XG = '''
    def fit(self, df, xg_weight=0.5):
        """
        xg_weight: 0.0 = solo goles reales | 1.0 = solo xG | 0.5 = hibrido
        """
        pl   = df[df["played"]]
        has_xg = "home_xg" in pl.columns

        for t in sorted(set(pl["home_team"]) | set(pl["away_team"])):
            hm = pl[pl["home_team"]==t]
            am = pl[pl["away_team"]==t]

            # Goles reales
            g_scored   = pd.concat([hm["home_score"], am["away_score"]])
            g_conceded = pd.concat([hm["away_score"], am["home_score"]])

            real_att = g_scored.mean()   if len(g_scored)   else 1.0
            real_def = g_conceded.mean() if len(g_conceded) else 1.0

            if has_xg:
                hm_xg = hm[hm["has_xg"]==True]
                am_xg = am[am["has_xg"]==True]
                xg_scored   = pd.concat([hm_xg["home_xg"], am_xg["away_xg"]])
                xg_conceded = pd.concat([hm_xg["away_xg"], am_xg["home_xg"]])
                xg_att = xg_scored.mean()   if len(xg_scored)>3   else real_att
                xg_def = xg_conceded.mean() if len(xg_conceded)>3 else real_def
                # Lambda hibrido
                self.attack[t]  = max(0.3, real_att*(1-xg_weight) + xg_att*xg_weight)
                self.defense[t] = max(0.3, real_def*(1-xg_weight) + xg_def*xg_weight)
            else:
                self.attack[t]  = max(0.3, real_att)
                self.defense[t] = max(0.3, real_def)

        ah = pl["home_score"].mean()
        aa = pl["away_score"].mean()
        self.home_adv = round(ah/aa, 3) if aa else 1.0
'''

# ════════════════════════════════════════════════════════════════════
# CAMBIO 3: En clase FootballSystem — metodo train()
# Llamar al XGScraper despues de descargar datos FD
# ════════════════════════════════════════════════════════════════════

TRAIN_XG_BLOCK = '''
        # --- BLOQUE xG (agregar despues de self.df_proc = ...) ---
        from xg_scraper import XGScraper, FD_TO_UNDERSTAT
        self.has_xg = False
        if comp_slug in FD_TO_UNDERSTAT:
            if progress_fn: progress_fn("Descargando xG de Understat...", 0.55)
            xg_sc   = XGScraper()
            xg_data = xg_sc.fetch_many_seasons(comp_slug, all_years, force=force)
            if xg_data is not None:
                self.df_proc = xg_sc.merge_xg_into_results(self.df_proc, xg_data, comp_slug)
                self.has_xg  = True
                print(f"  xG integrado: {self.df_proc['has_xg'].sum()} partidos con xG")
        else:
            self.df_proc["home_xg"] = None
            self.df_proc["away_xg"] = None
            self.df_proc["has_xg"]  = False
        # ---- FIN BLOQUE xG ----
'''

# ════════════════════════════════════════════════════════════════════
# CAMBIO 4: En FCOLS de MLModel — agregar features xG
# ════════════════════════════════════════════════════════════════════

FCOLS_XG_ADDITION = """
    # Agregar al final de FCOLS:
    # 'h_xgf','h_xgc','h_xg_diff','h_overperf',
    # 'hh_xgf','hh_xgc',
    # 'a_xgf','a_xgc','a_xg_diff','a_overperf',
    # 'aa_xgf','aa_xgc',
"""