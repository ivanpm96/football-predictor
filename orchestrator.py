# orchestrator.py — Coordinador maestro autónomo
# FD Discovery + Understat + FBref + SearchEngine + On-demand loading
import os, json, threading, time, re
from datetime import datetime, timezone
import pandas as pd
from predictor import FootballSystem, DataCollector, DataProcessor
from search_engine import SearchEngine
from catalog_builder import load_or_refresh_catalog, season_label, SEASON_TYPE

STATE_FILE = './fd_cache/orchestrator_state.json'
os.makedirs('./fd_cache', exist_ok=True)

def _now(): return datetime.now(timezone.utc).isoformat()

# ── Prioridad de fuente xG: Understat > FBref > None ─────────────────
def get_xg_source(fd_slug):
    try:
        from xg_scraper import FD_TO_UNDERSTAT
        if fd_slug in FD_TO_UNDERSTAT: return 'understat'
    except: pass
    try:
        from fbref_scraper import FBREF_COMPS
        if fd_slug in FBREF_COMPS: return 'fbref'
    except: pass
    return None

def _apply_xg(df_proc, fd_slug, years, force=False):
    """
    Aplica xG al DataFrame según la mejor fuente disponible.
    Retorna (df_proc_con_xg, fuente_usada).
    """
    source = get_xg_source(fd_slug)
    if source == 'understat':
        try:
            from xg_scraper import XGScraper
            xg_sc  = XGScraper()
            xg_df  = xg_sc.fetch_many_seasons(fd_slug, years, force=force)
            if xg_df is not None:
                df_proc = xg_sc.merge_xg_into_results(df_proc, xg_df, fd_slug)
                return df_proc, 'understat'
        except Exception as e:
            print(f'  Understat xG error ({fd_slug}): {e}')
    if source == 'fbref':
        try:
            from fbref_scraper import FBrefScraper
            fb_sc  = FBrefScraper()
            xg_df  = fb_sc.fetch_many_seasons(fd_slug, years, force=force)
            if xg_df is not None:
                df_proc = fb_sc.merge_with_results(df_proc, xg_df, fd_slug)
                return df_proc, 'fbref'
        except Exception as e:
            print(f'  FBref xG error ({fd_slug}): {e}')
    if 'has_xg' not in df_proc.columns:
        df_proc['home_xg']=None; df_proc['away_xg']=None
        df_proc['xg_diff']=None; df_proc['has_xg']=False
    return df_proc, None

# ════════════════════════════════════════════════════════════════════════
class DataOrchestrator:
    def __init__(self):
        self.system    = FootballSystem()
        self.search    = SearchEngine()
        self.catalog   = None
        self.fd_universe = {}
        self._lock     = threading.Lock()
        self._deep_running = False
        self.state     = self._load_state()

    # ── Estado persistente ───────────────────────────────────────────
    def _load_state(self):
        if os.path.exists(STATE_FILE):
            try:
                with open(STATE_FILE,'r') as f: return json.load(f)
            except: pass
        return {'loaded_comps':[],'last_minimum_sync':None,'last_deep_sync':None,
                'total_played':0,'total_pending':0,'has_xg':False,
                'xg_source':None,'syncs_done':0,'deep_status':'idle',
                'deep_progress':0.0,'deep_message':'',
                'fd_universe_size':0,'xg_leagues':0}
    def _save_state(self):
        with open(STATE_FILE,'w') as f: json.dump(self.state,f,indent=2)

    # ── Catálogo ─────────────────────────────────────────────────────
    def get_catalog(self, force=False):
        if self.catalog is None or force:
            self.catalog = load_or_refresh_catalog(fast_mode=not force, force=force)
        return self.catalog

    # ── SYNC MINIMUM: solo liga activa ───────────────────────────────
    def sync_minimum(self, comp_slug, train_seasons, predict_season,
                     force=False, progress_fn=None):
        """
        Entrena el sistema en una liga específica.
        Descarga FD + xG (Understat si disponible, FBref como fallback).
        """
        ok, msg = self.system.train(comp_slug, train_seasons, predict_season,
                                    force=force, progress_fn=progress_fn)
        if ok:
            cat = self.get_catalog()
            self.search.build(self.system.df_proc, cat)
            self.state.update({
                'loaded_comps':   self.system.comp_ids,
                'last_minimum_sync': _now(),
                'total_played':   int(self.system.df_proc['played'].sum()),
                'total_pending':  self.system.stats.get('pending',0),
                'has_xg':         self.system.has_xg,
                'xg_source':      get_xg_source(comp_slug) if self.system.has_xg else None,
            })
            self._save_state()
        return ok, msg

    # ── SYNC DEEP: todas las ligas ────────────────────────────────────
    def sync_deep(self, force=False, delay=1.5, progress_fn=None):
        """
        Descarga TODAS las ligas de FixtureDownload +
        xG para todas las ligas con cobertura (Understat + FBref).
        Operación larga (~10-30 min). Se puede lanzar en background.
        """
        if self._deep_running:
            return False, 'Ya hay un Sync Deep en curso'
        self._deep_running = True
        self.state['deep_status'] = 'running'
        self._save_state()
        def _upd(msg, pct):
            self.state['deep_message']  = msg
            self.state['deep_progress'] = round(pct,3)
            self._save_state()
            if progress_fn: progress_fn(msg, pct)
        try:
            # 1. Descubrir ligas FD
            _upd('Descubriendo ligas en FixtureDownload...', 0.01)
            from fd_discovery import load_or_refresh_discovery, full_fd_download
            universe = load_or_refresh_discovery(force=force)
            self.fd_universe = universe
            _upd(f'{len(universe)} ligas descubiertas. Descargando fixtures...', 0.04)

            # 2. Descargar TODOS los fixtures FD
            def fd_prog(msg, pct): _upd(msg, 0.04 + pct * 0.55)
            df_all = full_fd_download(universe, force=force, delay=delay,
                                      progress_fn=fd_prog)
            if df_all.empty:
                self._deep_running = False
                self.state['deep_status'] = 'error'
                self._save_state()
                return False, 'FD download retornó vacío'
            _upd(f'FD OK: {len(df_all):,} filas. Descargando xG...', 0.60)

            # 3. xG Understat (5 grandes ligas)
            xg_count = 0
            try:
                from xg_scraper import XGScraper, FD_TO_UNDERSTAT
                xg_sc = XGScraper()
                n_us = len(FD_TO_UNDERSTAT)
                for i,(slug) in enumerate(FD_TO_UNDERSTAT.keys()):
                    if slug in universe:
                        years = universe[slug].get('seasons',[])
                        _upd(f'Understat {slug} ({i+1}/{n_us})...', 0.60+i/n_us*0.12)
                        xg_df = xg_sc.fetch_many_seasons(slug, years, force=force)
                        if xg_df is not None:
                            df_all = xg_sc.merge_xg_into_results(df_all, xg_df, slug)
                            xg_count += 1
            except Exception as e:
                print(f'  Understat block error: {e}')

            # 4. xG FBref (ligas sin Understat)
            try:
                from fbref_scraper import FBrefScraper, FBREF_COMPS, has_fbref_xg
                from xg_scraper import FD_TO_UNDERSTAT as US_SLUGS
                fb_sc  = FBrefScraper()
                fb_only = [s for s in FBREF_COMPS if s not in US_SLUGS and s in universe]
                n_fb   = len(fb_only)
                for i, slug in enumerate(fb_only):
                    years = universe[slug].get('seasons',[])
                    _upd(f'FBref {slug} ({i+1}/{n_fb})...', 0.72+i/max(1,n_fb)*0.12)
                    xg_df = fb_sc.fetch_many_seasons(slug, years, force=force)
                    if xg_df is not None:
                        df_all = fb_sc.merge_with_results(df_all, xg_df, slug)
                        xg_count += 1
            except Exception as e:
                print(f'  FBref block error: {e}')

            # 5. Guardar en sistema
            _upd('Asignando datos al sistema...', 0.86)
            self.system.df_proc = df_all
            self.system.has_xg  = df_all.get('has_xg', pd.Series(dtype=bool)).any() if 'has_xg' in df_all.columns else False

            # 6. Construir índice de búsqueda
            _upd('Construyendo índice de búsqueda...', 0.90)
            cat = self.get_catalog()
            n_teams = self.search.build(df_all, cat)

            # 7. Estado final
            played  = int(df_all['played'].sum())
            pending = int((~df_all['played']).sum())
            self.state.update({
                'last_deep_sync':   _now(),
                'total_played':     played,
                'total_pending':    pending,
                'syncs_done':       self.state.get('syncs_done',0)+1,
                'fd_universe_size': len(universe),
                'xg_leagues':       xg_count,
                'has_xg':           xg_count>0,
                'deep_status':      'done',
                'deep_progress':    1.0,
                'deep_message':     f'{len(universe)} ligas | {played:,} partidos | {xg_count} con xG | {n_teams:,} equipos indexados',
            })
            self._save_state()
            self._deep_running = False
            result_msg = self.state['deep_message']
            _upd(result_msg, 1.0)
            return True, result_msg
        except Exception as e:
            self._deep_running = False
            self.state['deep_status'] = 'error'
            self.state['deep_message'] = str(e)
            self._save_state()
            return False, str(e)

    def sync_deep_background(self, force=False, delay=1.5):
        """Lanza sync_deep en un thread daemon (no bloquea la UI)."""
        if self._deep_running: return False
        t = threading.Thread(target=self.sync_deep,
                             kwargs=dict(force=force, delay=delay),
                             daemon=True)
        t.start()
        return True

    # ── ON-DEMAND LOAD ────────────────────────────────────────────────
    def on_demand_load(self, comp_slug, seasons, predict_season,
                       force=False, progress_fn=None):
        """Carga on-demand una liga cuando el usuario busca un partido."""
        return self.sync_minimum(comp_slug, seasons, predict_season,
                                 force=force, progress_fn=progress_fn)

    def detect_league_for_teams(self, home_q, away_q):
        """
        Intenta detectar la liga de dos equipos.
        Busca en el índice de búsqueda y retorna el comp_id más probable.
        """
        if not self.search.built: return None
        h_hits = self.search.search_teams(home_q, top_n=3)
        a_hits = self.search.search_teams(away_q, top_n=3)
        if not h_hits or not a_hits: return None
        # Intersección de comp_ids
        h_comps = {e['comp_id'] for e in h_hits}
        a_comps = {e['comp_id'] for e in a_hits}
        common  = h_comps & a_comps
        if common:
            # Preferir la más reciente
            return sorted(common, reverse=True)[0]
        # Si no hay intersección, devolver el comp_id del home con mayor score
        return h_hits[0]['comp_id'] if h_hits else None

    # ── ANÁLISIS COMPLETO DE PARTIDO ──────────────────────────────────
    def analyze_match(self, home, away):
        """Análisis completo: predicción + stats + H2H + xG."""
        if not self.system.trained: return None
        pred   = self.system.predict(home, away)
        if not pred: return None
        return {
            'prediction':  pred,
            'home_stats':  self.system.team_full_stats(home),
            'away_stats':  self.system.team_full_stats(away),
            'h2h':         self.system.h2h(home, away),
            'league_table':self.system.league_table(),
        }

    # ── STATUS ────────────────────────────────────────────────────────
    def get_status(self):
        return {
            **self.state,
            'system_trained':  self.system.trained,
            'search_built':    self.search.built,
            'search_teams':    len(self.search.team_index),
            'catalog_size':    self.catalog.get('total_comps',0) if self.catalog else 0,
            'deep_running':    self._deep_running,
            'current_league':  self.system.stats.get('comp_slug','—') if self.system.trained else '—',
            'current_season':  self.system.stats.get('predict_season','—') if self.system.trained else '—',
        }
    def get_teams(self):
        return self.system.get_teams() if self.system.trained else []