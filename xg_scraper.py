# xg_scraper.py — Scraper de xG desde Understat.com
# Ligas cubiertas: EPL, Bundesliga, La Liga, Serie A, Ligue 1
# Tecnica: extrae JSON embebido en <script> sin necesitar Selenium
import re, json, codecs, os, time, requests
import pandas as pd
from difflib import get_close_matches
from datetime import datetime, timezone

HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
    'Accept-Language': 'en-US,en;q=0.9',
    'Accept': 'text/html,application/xhtml+xml',
}

# ── Mapa slug FD -> slug Understat ────────────────────────────────────
FD_TO_UNDERSTAT = {
    'epl':      'EPL',
    'bundesliga':'Bundesliga',
    'la-liga':  'La_liga',
    'serie-a':  'Serie_A',
    'ligue-1':  'Ligue_1',
}

# ── Correcciones manuales de nombres de equipos ───────────────────────
# {fd_slug: {understat_name: fd_name}}
TEAM_MAP = {
    'bundesliga': {
        'Borussia Dortmund':       'Dortmund',
        'Borussia Moenchengladbach':'Borussia M.Gladbach',
        'Bayer Leverkusen':        'Leverkusen',
        'RasenBallsport Leipzig':  'RB Leipzig',
        'FC Augsburg':             'Augsburg',
        'FC Bayern Muenchen':      'Bayern Munich',
        'Eintracht Frankfurt':     'Eintracht Frankfurt',
        'VfB Stuttgart':           'Stuttgart',
        'TSG Hoffenheim':          'Hoffenheim',
        'SC Freiburg':             'Freiburg',
        'Werder Bremen':           'Werder Bremen',
        'VfL Wolfsburg':           'Wolfsburg',
        'VfL Bochum':              'Bochum',
        'Hamburger SV':            'Hamburg',
        'Heidenheim':              'Heidenheim',
        'FC St. Pauli':            'St. Pauli',
    },
    'epl': {
        'Manchester United':       'Manchester United',
        'Manchester City':         'Manchester City',
        'Newcastle United':        'Newcastle United',
        'Tottenham':               'Tottenham',
        'Wolverhampton Wanderers': 'Wolves',
        'Brighton':                'Brighton',
        'Leicester':               'Leicester City',
        'Nottingham Forest':       'Nottingham Forest',
        'West Ham':                'West Ham',
    },
    'la-liga': {
        'Athletic Club':           'Athletic Club',
        'Atletico Madrid':         'Atletico de Madrid',
        'Betis':                   'Real Betis',
        'Celta Vigo':              'Celta de Vigo',
        'Deportivo Alaves':        'Alaves',
        'Leganes':                 'Leganes',
        'Espanyol':                'Espanyol',
        'Villarreal':              'Villarreal',
    },
    'serie-a': {
        'Internazionale':          'Inter',
        'Milan':                   'Milan',
        'Hellas Verona':           'Verona',
        'SPAL':                    'SPAL',
        'Monza':                   'Monza',
        'Venezia':                 'Venezia',
    },
    'ligue-1': {
        'Paris Saint Germain':     'Paris Saint-Germain',
        'Clermont Foot':           'Clermont',
        'Stade Rennais':           'Rennes',
        'Olympique de Marseille':  'Marseille',
        'Olympique Lyonnais':      'Lyon',
        'Girondins de Bordeaux':   'Bordeaux',
        'AS Monaco':               'Monaco',
        'OGC Nice':                'Nice',
        'Stade de Reims':          'Reims',
    },
}

# ── Normalizacion de nombre de equipo ─────────────────────────────────
def _norm(name):
    import unicodedata
    name = unicodedata.normalize('NFD', str(name))
    name = ''.join(c for c in name if unicodedata.category(c) != 'Mn')
    return name.lower().strip().replace('fc ','').replace(' fc','').replace('  ',' ')

def _resolve_team(understat_name, fd_teams, slug):
    """
    Resuelve un nombre de equipo Understat al nombre equivalente en FD.
    Jerarquia: mapa manual -> coincidencia exacta norm -> difflib fuzzy.
    """
    # 1. Mapa manual
    manual = TEAM_MAP.get(slug, {})
    if understat_name in manual:
        return manual[understat_name]
    # 2. Coincidencia exacta normalizada
    norm_us = _norm(understat_name)
    for fd in fd_teams:
        if _norm(fd) == norm_us:
            return fd
    # 3. Contiene
    for fd in fd_teams:
        if norm_us in _norm(fd) or _norm(fd) in norm_us:
            return fd
    # 4. Fuzzy
    matches = get_close_matches(norm_us, [_norm(f) for f in fd_teams], n=1, cutoff=0.6)
    if matches:
        norm_to_fd = {_norm(f): f for f in fd_teams}
        return norm_to_fd.get(matches[0])
    return None

# ── Scraper principal ─────────────────────────────────────────────────
class XGScraper:
    def __init__(self, cache_dir='./fd_cache/xg', delay=2.0):
        self.cache_dir = cache_dir
        self.delay     = delay
        os.makedirs(cache_dir, exist_ok=True)

    def _cache_path(self, slug, year):
        return os.path.join(self.cache_dir, f'xg_{slug}_{year}.json')

    def _is_stale(self, path, year):
        if not os.path.exists(path): return True
        age = (time.time() - os.path.getmtime(path)) / 3600
        cy  = datetime.now().year
        ttl = 12 if year >= cy else 168   # 12h temporada activa, 7d historico
        return age >= ttl

    def _extract_json(self, html, var_name='datesData'):
        """
        Extrae el JSON embebido en: var datesData = JSON.parse('...')
        Tecnica probada en Understat 2019-2026.
        """
        pattern = re.compile(r'var\s+' + var_name + r'\s*=\s*JSON\.parse\((.*)\);', re.DOTALL)
        match   = pattern.search(html)
        if not match:
            return None
        raw = match.group(1)
        # Decodificar escapes unicode (\u00e9 etc)
        try:
            decoded = codecs.getdecoder('unicode_escape')(raw)[0]
            # Quitar comillas externas que JSON.parse espera
            decoded = decoded[1:-1] if decoded.startswith('"') else decoded
            # Algunos años tienen comilla simple como delimitador
            if decoded.startswith("'"):
                decoded = decoded[1:-1]
            return json.loads(decoded)
        except Exception:
            # Fallback: intentar parsing directo
            try:
                clean = raw.strip().strip('"').strip("'")
                return json.loads(clean)
            except Exception:
                return None

    def fetch_league_season(self, fd_slug, year, force=False):
        """
        Descarga xG de una liga+temporada desde Understat.
        Retorna DataFrame con: date, home_team_us, away_team_us,
                               home_xg, away_xg, home_goals, away_goals, match_id
        Retorna None si la liga no esta cubierta o hay error.
        """
        us_slug = FD_TO_UNDERSTAT.get(fd_slug)
        if not us_slug:
            return None   # Liga no cubierta por Understat

        cache_path = self._cache_path(fd_slug, year)
        if not force and not self._is_stale(cache_path, year):
            with open(cache_path,'r',encoding='utf-8') as f:
                return pd.DataFrame(json.load(f))

        url = f'https://understat.com/league/{us_slug}/{year}'
        try:
            r = requests.get(url, headers=HEADERS, timeout=30)
            r.raise_for_status()
            data = self._extract_json(r.text, 'datesData')
            if not data:
                print(f'  XG WARNING: no se pudo extraer datesData de {url}')
                return None

            rows = []
            for match in data:
                if not match.get('isResult'): continue
                h_xg = match.get('xG', {}).get('h')
                a_xg = match.get('xG', {}).get('a')
                h_g  = match.get('goals', {}).get('h')
                a_g  = match.get('goals', {}).get('a')
                if h_xg is None or a_xg is None: continue
                rows.append({
                    'match_id':   match.get('id',''),
                    'date_str':   match.get('datetime','')[:10],
                    'home_team_us': match['h']['title'],
                    'away_team_us': match['a']['title'],
                    'home_xg':    round(float(h_xg), 4),
                    'away_xg':    round(float(a_xg), 4),
                    'home_goals': int(h_g) if h_g is not None else None,
                    'away_goals': int(a_g) if a_g is not None else None,
                    'xg_diff':    round(float(h_xg) - float(a_xg), 4),
                })

            if not rows:
                print(f'  XG WARNING: 0 partidos con resultado en {url}')
                return None

            with open(cache_path,'w',encoding='utf-8') as f:
                json.dump(rows, f, ensure_ascii=False, indent=2)
            time.sleep(self.delay)
            print(f'  XG OK {fd_slug}-{year}: {len(rows)} partidos')
            return pd.DataFrame(rows)

        except Exception as e:
            print(f'  XG ERROR {fd_slug}-{year}: {e}')
            # Intentar desde cache aunque este stale
            if os.path.exists(cache_path):
                with open(cache_path,'r',encoding='utf-8') as f:
                    return pd.DataFrame(json.load(f))
            return None

    def fetch_many_seasons(self, fd_slug, years, force=False):
        """Descarga multiples temporadas y las concatena."""
        frames = []
        for year in years:
            df = self.fetch_league_season(fd_slug, year, force=force)
            if df is not None and not df.empty:
                df['season_year'] = year
                frames.append(df)
        return pd.concat(frames, ignore_index=True) if frames else None

    def merge_xg_into_results(self, df_results, xg_df, fd_slug):
        """
        Une el df de FixtureDownload con el df de xG de Understat.
        Usa normalizacion de nombres + fuzzy matching.
        Columnas nuevas en df_results: home_xg, away_xg, xg_diff.
        """
        if xg_df is None or xg_df.empty:
            df_results['home_xg']  = None
            df_results['away_xg']  = None
            df_results['xg_diff']  = None
            df_results['has_xg']   = False
            return df_results

        # Construir lookup: (date, us_home, us_away) -> (home_xg, away_xg)
        xg_lookup = {}
        for _, row in xg_df.iterrows():
            key = (row['date_str'], _norm(row['home_team_us']), _norm(row['away_team_us']))
            xg_lookup[key] = (row['home_xg'], row['away_xg'])

        # Lista de equipos FD para resolver nombres
        fd_teams = list(set(df_results['home_team'].tolist() + df_results['away_team'].tolist()))
        team_cache = {}

        def get_us_name(fd_name):
            """Nombre FD -> nombre Understat (inverso del mapa)"""
            # Invertir TEAM_MAP para esta liga
            inv = {v:k for k,v in TEAM_MAP.get(fd_slug,{}).items()}
            return inv.get(fd_name, fd_name)

        # Asignar xG a cada partido
        home_xgs, away_xgs, xg_diffs = [], [], []
        matched = 0
        for _, row in df_results.iterrows():
            if not row.get('played', False):
                home_xgs.append(None); away_xgs.append(None); xg_diffs.append(None)
                continue
            date_s = str(row['date'])[:10] if row['date'] is not None else ''
            hn = get_us_name(row['home_team'])
            an = get_us_name(row['away_team'])
            key = (date_s, _norm(hn), _norm(an))
            if key in xg_lookup:
                h_x, a_x = xg_lookup[key]
                home_xgs.append(h_x); away_xgs.append(a_x)
                xg_diffs.append(round(h_x - a_x, 4))
                matched += 1
            else:
                home_xgs.append(None); away_xgs.append(None); xg_diffs.append(None)

        df_results = df_results.copy()
        df_results['home_xg']  = home_xgs
        df_results['away_xg']  = away_xgs
        df_results['xg_diff']  = xg_diffs
        df_results['has_xg']   = df_results['home_xg'].notna()
        total = df_results['played'].sum()
        pct = round(matched/max(1,total)*100, 1)
        print(f'  XG MERGE {fd_slug}: {matched}/{int(total)} partidos ({pct}%)')
        return df_results

# ── Helpers de analisis xG ────────────────────────────────────────────
def xg_team_summary(df, team, comp_id=None):
    """
    Resumen xG de un equipo: xGF, xGA, xG_diff, overperformance.
    overperformance > 0 significa que el equipo marco mas de lo esperado
    (posible regresion futura).
    """
    pl = df[df['played'] & df['has_xg']].copy()
    if comp_id: pl = pl[pl['competition_id'] == comp_id]
    hm = pl[pl['home_team'] == team]
    am = pl[pl['away_team'] == team]
    if len(hm) + len(am) == 0:
        return {}
    xgf_h = hm['home_xg'].sum();   xgf_a = am['away_xg'].sum()
    xgc_h = hm['away_xg'].sum();   xgc_a = am['home_xg'].sum()
    gf_h  = hm['home_score'].sum(); gf_a = am['away_score'].sum()
    gc_h  = hm['away_score'].sum(); gc_a = am['home_score'].sum()
    n     = len(hm) + len(am)
    xgf   = xgf_h + xgf_a
    xgc   = xgc_h + xgc_a
    gf    = gf_h + gf_a
    gc    = gc_h + gc_a
    return {
        'xGF':           round(xgf, 2),
        'xGA':           round(xgc, 2),
        'xGF_per_game':  round(xgf/n, 3),
        'xGA_per_game':  round(xgc/n, 3),
        'xG_diff':       round(xgf - xgc, 2),
        'xG_diff_per_game': round((xgf-xgc)/n, 3),
        'overperf_attack':  round(gf - xgf, 2),
        'overperf_defense': round(xgc - gc, 2),
        'luck_index':       round((gf - xgf) - (gc - xgc), 2),
        'n_xg_matches':     n,
    }

def xg_league_table(df, comp_id):
    """Tabla de posiciones basada en xG (rendimiento esperado)."""
    pl = df[df['played'] & df['has_xg'] & (df['competition_id']==comp_id)].copy()
    if pl.empty: return pd.DataFrame()
    rows = []
    for team in sorted(set(pl['home_team']) | set(pl['away_team'])):
        hm = pl[pl['home_team']==team]
        am = pl[pl['away_team']==team]
        n  = len(hm)+len(am)
        if not n: continue
        xgf = hm['home_xg'].sum()+am['away_xg'].sum()
        xgc = hm['away_xg'].sum()+am['home_xg'].sum()
        gf  = hm['home_score'].sum()+am['away_score'].sum()
        gc  = hm['away_score'].sum()+am['home_score'].sum()
        rows.append({'Equipo':team,'PJ':n,
            'xGF':round(xgf,1),'xGA':round(xgc,1),'xG_Diff':round(xgf-xgc,1),
            'xGF/pj':round(xgf/n,2),'xGA/pj':round(xgc/n,2),
            'GF':int(gf),'GA':int(gc),
            'OVP_Atq':round(gf-xgf,1),
            'OVP_Def':round(xgc-gc,1),
            'Suerte':round((gf-xgf)-(gc-xgc),1)})
    return pd.DataFrame(rows).sort_values('xG_Diff',ascending=False).reset_index(drop=True)