# search_engine.py — Motor de busqueda universal
# Busca equipos, ligas y partidos en tiempo real sobre el DataFrame cargado
import pandas as pd
import unicodedata, re
from difflib import get_close_matches, SequenceMatcher

def _norm(name):
    name = unicodedata.normalize('NFD', str(name))
    name = ''.join(c for c in name if unicodedata.category(c) != 'Mn')
    return re.sub(r'\s+', ' ', name.lower().strip())

def _similarity(a, b):
    return SequenceMatcher(None, _norm(a), _norm(b)).ratio()

class SearchEngine:
    """
    Indice de busqueda en memoria sobre el DataFrame principal.
    Se reconstruye cada vez que el sistema carga nuevos datos.
    """
    def __init__(self):
        self.df         = None
        self.team_index = {}   # norm_name -> [team_name, comp_id, slug, season]
        self.comp_index = {}   # comp_id -> {slug, name, label}
        self.built      = False

    def build(self, df_proc, catalog):
        """
        Construye el indice a partir del DataFrame procesado y el catalogo.
        Llamar despues de cada train() o update().
        """
        self.df = df_proc
        self.team_index = {}
        self.comp_index = {}

        comps = catalog.get('competitions', {})
        for comp_id in df_proc['competition_id'].unique():
            slug  = comp_id.rsplit('-', 1)[0]
            year  = comp_id.rsplit('-', 1)[-1]
            meta  = comps.get(slug, {})
            self.comp_index[comp_id] = {
                'slug':  slug,
                'name':  meta.get('name', slug),
                'label': meta.get('label', slug),
                'year':  year,
                'season_label': meta.get('current_label', year),
            }

        played = df_proc[df_proc['played']].copy()
        for comp_id in played['competition_id'].unique():
            sub = played[played['competition_id'] == comp_id]
            teams = set(sub['home_team'].tolist() + sub['away_team'].tolist())
            for team in teams:
                norm = _norm(team)
                if norm not in self.team_index:
                    self.team_index[norm] = []
                self.team_index[norm].append({
                    'team':    team,
                    'comp_id': comp_id,
                    'slug':    comp_id.rsplit('-',1)[0],
                })
        self.built = True
        return len(self.team_index)

    def search_teams(self, query, top_n=10, min_score=0.45):
        """
        Busca equipos por nombre en todos los datos cargados.
        Retorna lista de dicts ordenados por relevancia.
        """
        if not self.built or not query: return []
        q_norm = _norm(query)
        results = []
        seen    = set()
        # Coincidencia exacta / parcial primero
        for norm, entries in self.team_index.items():
            if q_norm in norm or norm in q_norm:
                for e in entries:
                    key = (e['team'], e['comp_id'])
                    if key not in seen:
                        seen.add(key)
                        results.append({**e, 'score': 1.0 if q_norm==norm else 0.85})
        # Fuzzy si hay pocos resultados
        if len(results) < top_n:
            all_norms  = list(self.team_index.keys())
            fuzzy_hits = get_close_matches(q_norm, all_norms, n=top_n*2, cutoff=min_score)
            for hn in fuzzy_hits:
                for e in self.team_index[hn]:
                    key = (e['team'], e['comp_id'])
                    if key not in seen:
                        seen.add(key)
                        score = _similarity(query, e['team'])
                        results.append({**e, 'score': round(score,3)})
        results.sort(key=lambda x: -x['score'])
        return results[:top_n]

    def search_matches(self, query, comp_id=None, top_n=20):
        """
        Busca partidos (jugados o pendientes) que involucren equipos
        que coincidan con la query.
        """
        if self.df is None or not query: return pd.DataFrame()
        team_hits = self.search_teams(query, top_n=5)
        if not team_hits: return pd.DataFrame()
        matched_teams = [h['team'] for h in team_hits]
        mask = (
            self.df['home_team'].isin(matched_teams) |
            self.df['away_team'].isin(matched_teams)
        )
        if comp_id: mask &= (self.df['competition_id'] == comp_id)
        df_m = self.df[mask].copy()
        df_m['date_fmt'] = df_m['date'].dt.strftime('%Y-%m-%d')
        df_m['score']    = df_m.apply(
            lambda r: f"{int(r['home_score'])}-{int(r['away_score'])}" if r['played'] else '—', axis=1)
        return df_m.sort_values('date', ascending=False).head(top_n)[
            ['competition_id','date_fmt','home_team','score','away_team','result','played']
        ].rename(columns={'competition_id':'Liga','date_fmt':'Fecha',
            'home_team':'Local','score':'Marcador','away_team':'Visitante',
            'result':'Res','played':'Jugado'})

    def upcoming_matches(self, team, comp_id=None, limit=10):
        """Partidos pendientes del equipo (futuros fixtures sin resultado)."""
        if self.df is None: return pd.DataFrame()
        mask = (
            (~self.df['played']) &
            (self.df['home_team'].str.lower().str.contains(team.lower(), na=False) |
             self.df['away_team'].str.lower().str.contains(team.lower(), na=False))
        )
        if comp_id: mask &= (self.df['competition_id'] == comp_id)
        df_u = self.df[mask].copy()
        df_u['date_fmt'] = df_u['date'].dt.strftime('%Y-%m-%d')
        return df_u.sort_values('date').head(limit)[
            ['competition_id','date_fmt','round','home_team','away_team']
        ].rename(columns={'competition_id':'Liga','date_fmt':'Fecha',
            'round':'Jornada','home_team':'Local','away_team':'Visitante'})

    def find_match(self, home_q, away_q, comp_id=None):
        """
        Encuentra el partido mas probable entre dos equipos dados.
        Busca en pendientes primero, luego en jugados.
        Retorna (home_exact, away_exact, comp_id, played) o None.
        """
        if self.df is None: return None
        home_hits = self.search_teams(home_q, top_n=3)
        away_hits = self.search_teams(away_q, top_n=3)
        if not home_hits or not away_hits: return None

        home_names = [h['team'] for h in home_hits]
        away_names = [h['team'] for h in away_hits]

        # Buscar partido pendiente primero
        for hn in home_names:
            for an in away_names:
                mask = (
                    (~self.df['played']) &
                    (self.df['home_team']==hn) & (self.df['away_team']==an)
                )
                if comp_id: mask &= (self.df['competition_id']==comp_id)
                if mask.any():
                    row = self.df[mask].iloc[0]
                    return {'home':hn,'away':an,'comp_id':row['competition_id'],'played':False}
        # Buscar partido jugado mas reciente
        for hn in home_names:
            for an in away_names:
                mask = (
                    self.df['played'] &
                    (self.df['home_team']==hn) & (self.df['away_team']==an)
                )
                if comp_id: mask &= (self.df['competition_id']==comp_id)
                if mask.any():
                    row = self.df[mask].sort_values('date',ascending=False).iloc[0]
                    return {'home':hn,'away':an,'comp_id':row['competition_id'],'played':True}
        return None

    def league_matches(self, comp_id, played_only=False, pending_only=False, limit=200):
        """Todos los partidos de una liga."""
        if self.df is None: return pd.DataFrame()
        mask = self.df['competition_id'] == comp_id
        if played_only:  mask &= self.df['played']
        if pending_only: mask &= ~self.df['played']
        df_l = self.df[mask].copy()
        df_l['date_fmt'] = df_l['date'].dt.strftime('%Y-%m-%d')
        df_l['score']    = df_l.apply(
            lambda r: f"{int(r['home_score'])}-{int(r['away_score'])}" if r['played'] else '—', axis=1)
        return df_l.sort_values('date', ascending=False).head(limit)[
            ['competition_id','round','date_fmt','home_team','score','away_team','result','played']
        ].rename(columns={'competition_id':'Liga','round':'Jornada','date_fmt':'Fecha',
            'home_team':'Local','score':'Marcador','away_team':'Visitante',
            'result':'Res','played':'Jugado'})