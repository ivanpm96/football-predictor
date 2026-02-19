# fbref_scraper.py — xG de FBref para 30+ ligas
# Fuente: https://fbref.com/en/comps/{id}/{season}/schedule/{season}-{name}-Scores-and-Fixtures
# Tecnica: pd.read_html() en la tabla sched_* (no requiere Selenium)
import os, json, time, requests, re
import pandas as pd
from difflib import get_close_matches
from datetime import datetime

FBREF_BASE = 'https://fbref.com'
FBREF_DELAY = 4.0   # segundos entre requests (regla de robots.txt de FBref)
HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)',
    'Accept-Language': 'en-US,en;q=0.9',
}

# ── Catalogo de ligas FBref ───────────────────────────────────────────
# fd_slug : (fbref_comp_id, fbref_name_url, season_type)
FBREF_COMPS = {
    'epl':                  (9,  'Premier-League',     'split'),
    'championship':         (10, 'Championship',        'split'),
    'league-one':           (15, 'League-One',          'split'),
    'league-two':           (16, 'League-Two',          'split'),
    'bundesliga':           (20, 'Bundesliga',          'split'),
    '2-bundesliga':         (33, '2-Bundesliga',        'split'),
    'la-liga':              (12, 'La-Liga',             'split'),
    'la-liga2':             (17, 'Segunda-Division',    'split'),
    'serie-a':              (11, 'Serie-A',             'split'),
    'serie-b':              (18, 'Serie-B',             'split'),
    'ligue-1':              (13, 'Ligue-1',             'split'),
    'ligue-2':              (60, 'Ligue-2',             'split'),
    'eredivisie':           (23, 'Eredivisie',          'split'),
    'primeira-liga':        (32, 'Primeira-Liga',       'split'),
    'super-lig':            (26, 'Super-Lig',           'split'),
    'scottish-premiership': (40, 'Scottish-Premiership','split'),
    'jupiler-pro-league':   (37, 'Belgian-First-Division-A','split'),
    'mls':                  (22, 'Major-League-Soccer', 'single'),
    'brasileirao':          (24, 'Serie-A',             'single'),
    'primera-division-arg': (21, 'Primera-Division',    'single'),
    'j-league':             (25, 'J1-League',           'split'),
    'k-league-1':           (55, 'K-League-1',          'single'),
    'saudi-pro-league':     (70, 'Saudi-Professional-League','split'),
    'champions-league':     (8,  'Champions-League',   'split'),
    'europa-league':        (19, 'Europa-League',       'split'),
    'conference-league':    (882,'Conference-League',   'split'),
}

def _fbref_season_str(slug, year):
    """Convierte slug + year a formato de temporada FBref."""
    _, _, stype = FBREF_COMPS.get(slug, (None,None,'single'))
    if stype == 'split': return f'{year}-{year+1}'
    return str(year)

def _fbref_url(slug, year):
    cid, name, _ = FBREF_COMPS.get(slug, (None,None,'single'))
    if not cid: return None
    s = _fbref_season_str(slug, year)
    return f'{FBREF_BASE}/en/comps/{cid}/{s}/schedule/{s}-{name}-Scores-and-Fixtures'

def _norm(name):
    import unicodedata
    name = unicodedata.normalize('NFD', str(name))
    name = ''.join(c for c in name if unicodedata.category(c) != 'Mn')
    return re.sub(r'[^a-z0-9]', '', name.lower())

class FBrefScraper:
    def __init__(self, cache_dir='./fd_cache/fbref', delay=FBREF_DELAY):
        self.cache_dir = cache_dir
        self.delay     = delay
        os.makedirs(cache_dir, exist_ok=True)

    def _cache_path(self, slug, year):
        return os.path.join(self.cache_dir, f'fbref_{slug}_{year}.json')

    def _is_stale(self, path, year):
        if not os.path.exists(path): return True
        age = (time.time() - os.path.getmtime(path)) / 3600
        cy  = datetime.now().year
        ttl = 12 if year >= cy else 720   # 12h temporada activa, 30d historica
        return age >= ttl

    def fetch_league_season(self, fd_slug, year, force=False):
        """
        Descarga fixtures + xG de FBref para una liga+temporada.
        Retorna DataFrame con: date_str, home_team, away_team, home_xg, away_xg
        o None si la liga no esta en FBref o hay error.
        """
        if fd_slug not in FBREF_COMPS: return None
        cache_path = self._cache_path(fd_slug, year)
        if not force and not self._is_stale(cache_path, year):
            with open(cache_path,'r',encoding='utf-8') as f:
                return pd.DataFrame(json.load(f))

        url = _fbref_url(fd_slug, year)
        if not url: return None

        try:
            r = requests.get(url, headers=HEADERS, timeout=40)
            r.raise_for_status()
        except Exception as e:
            print(f'  FBREF ERROR {fd_slug}-{year}: {e}')
            if os.path.exists(cache_path):
                with open(cache_path,'r',encoding='utf-8') as f:
                    return pd.DataFrame(json.load(f))
            return None

        try:
            # FBref tiene la tabla de fixtures como la primera tabla con id sched_*
            # pd.read_html puede parsearla directamente
            tables = pd.read_html(r.text, attrs={'id': re.compile(r'sched_')})
            if not tables:
                # Fallback: cualquier tabla con columnas Home/Away/xG
                tables = pd.read_html(r.text)
            df = tables[0]
        except Exception as e:
            print(f'  FBREF PARSE ERROR {fd_slug}-{year}: {e}')
            return None

        # Normalizar columnas FBref -> nombres estandar
        df.columns = [str(c).strip() for c in df.columns]
        # FBref columnas tipicas: Wk, Day, Date, Time, Home, xG, Score, xG.1, Away, ...
        col_map = {}
        for c in df.columns:
            cl = c.lower().strip()
            if cl == 'home':    col_map[c] = 'home_team'
            elif cl == 'away':  col_map[c] = 'away_team'
            elif cl == 'date':  col_map[c] = 'date_str'
            elif cl == 'xg' and 'home_xg' not in col_map.values():
                col_map[c] = 'home_xg'
            elif cl in ['xg.1','xg'] and 'away_xg' not in col_map.values():
                col_map[c] = 'away_xg'
            elif cl == 'score': col_map[c] = 'score_raw'
        df = df.rename(columns=col_map)

        needed = ['home_team','away_team','date_str','home_xg','away_xg']
        for n in needed:
            if n not in df.columns: df[n] = None

        # Filtrar filas sin fecha o sin xG
        df = df[df['date_str'].notna() & df['home_team'].notna() & df['away_team'].notna()].copy()
        df['date_str'] = pd.to_datetime(df['date_str'], errors='coerce').dt.strftime('%Y-%m-%d')
        df = df[df['date_str'].notna()]
        df['home_xg'] = pd.to_numeric(df['home_xg'], errors='coerce')
        df['away_xg'] = pd.to_numeric(df['away_xg'], errors='coerce')
        # Solo filas con xG disponible
        df_xg = df[df['home_xg'].notna() & df['away_xg'].notna()].copy()
        if df_xg.empty:
            print(f'  FBREF WARNING {fd_slug}-{year}: 0 filas con xG')
            return None

        rows = df_xg[['date_str','home_team','away_team','home_xg','away_xg']].copy()
        rows['home_xg'] = rows['home_xg'].round(4)
        rows['away_xg'] = rows['away_xg'].round(4)
        rows['xg_diff'] = (rows['home_xg'] - rows['away_xg']).round(4)
        rows['source']  = 'fbref'
        records = rows.to_dict('records')
        with open(cache_path,'w',encoding='utf-8') as f:
            json.dump(records, f, ensure_ascii=False, indent=2)
        time.sleep(self.delay)
        print(f'  FBREF OK {fd_slug}-{year}: {len(rows)} partidos con xG')
        return rows

    def fetch_many_seasons(self, fd_slug, years, force=False, progress_fn=None):
        frames = []
        for i, year in enumerate(years):
            if progress_fn: progress_fn(f'FBref {fd_slug}-{year}', i/len(years))
            df = self.fetch_league_season(fd_slug, year, force=force)
            if df is not None and not df.empty:
                df['season_year'] = year
                frames.append(df)
        return pd.concat(frames, ignore_index=True) if frames else None

    def merge_with_results(self, df_results, xg_df, fd_slug):
        """Mismo merge que xg_scraper.py pero para datos FBref."""
        if xg_df is None or xg_df.empty:
            for c in ['home_xg','away_xg','xg_diff','has_xg','xg_source']:
                if c not in df_results.columns:
                    df_results[c] = None if c != 'has_xg' else False
            return df_results

        # Construir lookup {(date, norm_home, norm_away): (h_xg, a_xg)}
        xg_lookup = {}
        for _, row in xg_df.iterrows():
            key = (str(row['date_str'])[:10], _norm(str(row['home_team'])), _norm(str(row['away_team'])))
            xg_lookup[key] = (float(row['home_xg']), float(row['away_xg']))

        h_xgs, a_xgs, xg_diffs, matched = [], [], [], 0
        for _, row in df_results.iterrows():
            if not row.get('played', False):
                h_xgs.append(None); a_xgs.append(None); xg_diffs.append(None)
                continue
            date_s = str(row['date'])[:10] if row['date'] is not None else ''
            key    = (date_s, _norm(str(row['home_team'])), _norm(str(row['away_team'])))
            if key in xg_lookup:
                h, a = xg_lookup[key]
                h_xgs.append(h); a_xgs.append(a); xg_diffs.append(round(h-a,4))
                matched += 1
            else:
                h_xgs.append(None); a_xgs.append(None); xg_diffs.append(None)

        df_results = df_results.copy()
        df_results['home_xg']  = h_xgs
        df_results['away_xg']  = a_xgs
        df_results['xg_diff']  = xg_diffs
        df_results['has_xg']   = df_results['home_xg'].notna()
        df_results['xg_source']= 'fbref'
        total = int(df_results['played'].sum())
        pct   = round(matched/max(1,total)*100,1)
        print(f'  FBREF MERGE {fd_slug}: {matched}/{total} partidos ({pct}%)')
        return df_results

# ── Coverage check ────────────────────────────────────────────────────
def has_fbref_xg(fd_slug):
    return fd_slug in FBREF_COMPS

def coverage_summary():
    from catalog_builder import SLUG_META
    rows = []
    for slug, meta in SLUG_META.items():
        fb  = slug in FBREF_COMPS
        from xg_scraper import FD_TO_UNDERSTAT
        us  = slug in FD_TO_UNDERSTAT
        rows.append({'Liga':meta['name'],'Region':meta['region'],
                     'Understat': 'YES' if us else '—',
                     'FBref':     'YES' if fb else '—',
                     'xG_source': 'Understat' if us else ('FBref' if fb else 'None')})
    return pd.DataFrame(rows).sort_values(['xG_source','Region'])