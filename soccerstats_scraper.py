# soccerstats_scraper.py
# Scraper responsable de soccerstats.com
# Datos extraídos: tabla con goles, home/away breakdown,
#                  timing de goles (franjas), BTTS%, Over2.5%
# Rate limit: 3-5s entre requests, caché 24h, User-Agent realista
import requests, time, re, json, logging
from pathlib import Path
from bs4 import BeautifulSoup
from datetime import datetime, timedelta

try:
    from name_normalizer import get_normalizer
    _norm = get_normalizer()
except ImportError:
    _norm = None

log = logging.getLogger("soccerstats")
CACHE_DIR = Path("fd_cache/soccerstats")
CACHE_DIR.mkdir(parents=True, exist_ok=True)

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
    "Accept":          "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Referer":         "https://www.soccerstats.com/",
}

# ── Mapa de ligas conocidas en soccerstats ────────────────────────────
LEAGUE_SLUGS = {
    "epl":              "england",
    "bundesliga":       "germany",
    "la-liga":          "spain",
    "serie-a":          "italy",
    "ligue-1":          "france",
    "eredivisie":       "netherlands",
    "primeira-liga":    "portugal",
    "super-lig":        "turkey",
    "brasileirao":      "brazil",
    "liga-profesional": "argentina",
    "mls":              "usa",
    "liga-mx":          "mexico",
    "scottish-prem":    "scotland",
    "belgian-pro":      "belgium",
    "russian-prem":     "russia",
    "ukrainian-prem":   "ukraine",
    "swiss-super":      "switzerland",
    "austrian-bund":    "austria",
    "greek-super":      "greece",
    "danish-super":     "denmark",
    "norwegian-elite":  "norway",
    "swedish-allsv":    "sweden",
    "chinese-super":    "china",
    "j-league":         "japan",
    "k-league":         "south-korea",
}

SESSION = requests.Session()
SESSION.headers.update(HEADERS)


# ── Helpers ───────────────────────────────────────────────────────────
def _cache_path(league_slug, table_type):
    return CACHE_DIR / f"{league_slug}_{table_type}.json"

def _is_fresh(path, hours=24):
    if not path.exists():
        return False
    mtime = datetime.fromtimestamp(path.stat().st_mtime)
    return datetime.now() - mtime < timedelta(hours=hours)

def _get(url, delay=4.0):
    """GET responsable con delay y manejo de errores."""
    time.sleep(delay)
    try:
        r = SESSION.get(url, timeout=20)
        r.raise_for_status()
        return r.text
    except requests.exceptions.HTTPError as e:
        log.warning(f"HTTP {e.response.status_code} para {url}")
        return None
    except requests.exceptions.RequestException as e:
        log.warning(f"Error de red: {e}")
        return None

def _parse_int(s):
    try:    return int(re.sub(r"[^\d]", "", str(s)))
    except: return 0

def _parse_float(s):
    try:    return float(re.sub(r"[^\d.]", "", str(s)))
    except: return 0.0

def _normalize_team(name):
    if _norm:
        return _norm.team(name.strip())
    return name.strip()


# ── Scraper principal ─────────────────────────────────────────────────
class SoccerStatsScraper:
    """
    Extrae estadísticas detalladas de soccerstats.com de forma responsable.

    Datos disponibles:
    - league_stats(slug)  → tabla general con GF, GA, Over2.5%, BTTS%
    - home_away(slug)     → desglose local vs visitante
    - goal_timing(slug)   → % goles por franja horaria (0-15, 15-30, ... 75-90)
    - team_detail(slug, team_name) → stats completas de un equipo
    """

    BASE = "https://www.soccerstats.com"

    def league_stats(self, slug, force=False):
        """
        Tabla general de la liga con columnas extra de goles.
        Returns: list[dict] o [] si no disponible.
        """
        ss_key = LEAGUE_SLUGS.get(slug)
        if not ss_key:
            log.info(f"soccerstats: liga '{slug}' no en mapa de ligas")
            return []

        cache = _cache_path(slug, "stats")
        if not force and _is_fresh(cache, hours=24):
            return json.loads(cache.read_text(encoding="utf-8"))

        url  = f"{self.BASE}/league.asp?league={ss_key}"
        html = _get(url, delay=4.0)
        if not html:
            return []

        rows = self._parse_league_table(html, slug)
        if rows:
            cache.write_text(json.dumps(rows, ensure_ascii=False, indent=2),
                             encoding="utf-8")
            log.info(f"soccerstats: {len(rows)} equipos guardados para '{slug}'")
        return rows

    def _parse_league_table(self, html, slug):
        soup = BeautifulSoup(html, "html.parser")
        rows = []
        # Buscar tabla principal (clase 'sortable' o primera tabla grande)
        tables = soup.find_all("table")
        target = None
        for t in tables:
            ths = [th.get_text(strip=True) for th in t.find_all("th")]
            if any(h in ths for h in ["Played","MP","W","D","L","GF","GA","Pts"]):
                target = t
                break
        if not target:
            log.warning(f"soccerstats: tabla no encontrada para '{slug}'")
            return []

        headers = [th.get_text(strip=True) for th in target.find_all("th")]
        # Mapeo flexible de columnas
        col_map = {}
        for i, h in enumerate(headers):
            h_low = h.lower()
            if h_low in ["team","club","name"]:       col_map["team"]   = i
            elif h_low in ["played","mp","pj","gp"]:  col_map["played"] = i
            elif h_low in ["w","won","wins"]:          col_map["wins"]   = i
            elif h_low in ["d","drawn","draws"]:       col_map["draws"]  = i
            elif h_low in ["l","lost","losses"]:       col_map["losses"] = i
            elif h_low in ["gf","for","goals for","f"]:col_map["gf"]    = i
            elif h_low in ["ga","against","goals against","a"]: col_map["ga"] = i
            elif h_low in ["pts","points","pt"]:       col_map["pts"]    = i
            elif "over" in h_low and "2.5" in h_low:  col_map["over25"] = i
            elif "btts" in h_low or "gg" in h_low:    col_map["btts"]   = i
            elif "cs" in h_low or "clean" in h_low:   col_map["cs"]     = i

        for tr in target.find_all("tr")[1:]:
            tds = tr.find_all(["td","th"])
            if len(tds) < 5:
                continue
            row = {}
            if "team" in col_map:
                row["team"]   = _normalize_team(tds[col_map["team"]].get_text(strip=True))
            if "played" in col_map:
                row["played"] = _parse_int(tds[col_map["played"]].get_text())
            if "wins"   in col_map:
                row["wins"]   = _parse_int(tds[col_map["wins"]].get_text())
            if "draws"  in col_map:
                row["draws"]  = _parse_int(tds[col_map["draws"]].get_text())
            if "losses" in col_map:
                row["losses"] = _parse_int(tds[col_map["losses"]].get_text())
            if "gf"     in col_map:
                row["gf"]     = _parse_int(tds[col_map["gf"]].get_text())
            if "ga"     in col_map:
                row["ga"]     = _parse_int(tds[col_map["ga"]].get_text())
            if "pts"    in col_map:
                row["pts"]    = _parse_int(tds[col_map["pts"]].get_text())
            if "over25" in col_map:
                row["over25_pct"] = _parse_float(tds[col_map["over25"]].get_text())
            if "btts"   in col_map:
                row["btts_pct"]   = _parse_float(tds[col_map["btts"]].get_text())
            if "cs"     in col_map:
                row["clean_sheets"] = _parse_int(tds[col_map["cs"]].get_text())
            # Calcular derivados si hay datos base
            if row.get("played",0) > 0:
                row["gf_per_game"] = round(row.get("gf",0)/row["played"],2)
                row["ga_per_game"] = round(row.get("ga",0)/row["played"],2)
            if row.get("team"):
                rows.append(row)
        return rows

    def home_away(self, slug, force=False):
        """
        Estadísticas separadas de local y visitante.
        Returns: dict{team: {"home":{...}, "away":{...}}}
        """
        ss_key = LEAGUE_SLUGS.get(slug)
        if not ss_key:
            return {}

        cache = _cache_path(slug, "homeaway")
        if not force and _is_fresh(cache, hours=24):
            return json.loads(cache.read_text(encoding="utf-8"))

        url  = f"{self.BASE}/homeaway.asp?league={ss_key}"
        html = _get(url, delay=4.0)
        if not html:
            return {}

        result = self._parse_home_away(html)
        if result:
            cache.write_text(json.dumps(result, ensure_ascii=False, indent=2),
                             encoding="utf-8")
        return result

    def _parse_home_away(self, html):
        soup   = BeautifulSoup(html, "html.parser")
        result = {}
        tables = soup.find_all("table")
        for t in tables:
            rows = t.find_all("tr")
            if len(rows) < 3:
                continue
            for tr in rows[1:]:
                tds = tr.find_all(["td","th"])
                if len(tds) < 8:
                    continue
                texts = [td.get_text(strip=True) for td in tds]
                try:
                    team = _normalize_team(texts[0])
                    result[team] = {
                        "home_pj":  _parse_int(texts[1]),
                        "home_wins":_parse_int(texts[2]),
                        "home_draws":_parse_int(texts[3]),
                        "home_losses":_parse_int(texts[4]),
                        "home_gf":  _parse_int(texts[5]),
                        "home_ga":  _parse_int(texts[6]),
                        "away_pj":  _parse_int(texts[7]) if len(texts)>7 else 0,
                        "away_wins":_parse_int(texts[8]) if len(texts)>8 else 0,
                        "away_draws":_parse_int(texts[9]) if len(texts)>9 else 0,
                        "away_losses":_parse_int(texts[10]) if len(texts)>10 else 0,
                        "away_gf":  _parse_int(texts[11]) if len(texts)>11 else 0,
                        "away_ga":  _parse_int(texts[12]) if len(texts)>12 else 0,
                    }
                    pj_h = result[team]["home_pj"]
                    pj_a = result[team]["away_pj"]
                    if pj_h > 0:
                        result[team]["home_gf_pg"] = round(result[team]["home_gf"]/pj_h,2)
                        result[team]["home_ga_pg"] = round(result[team]["home_ga"]/pj_h,2)
                    if pj_a > 0:
                        result[team]["away_gf_pg"] = round(result[team]["away_gf"]/pj_a,2)
                        result[team]["away_ga_pg"] = round(result[team]["away_ga"]/pj_a,2)
                except (IndexError, ValueError):
                    continue
        return result

    def goal_timing(self, slug, force=False):
        """
        % de goles marcados por franja: 0-15, 16-30, 31-45, 46-60, 61-75, 76-90+
        Returns: list[dict] con team + franjas
        """
        ss_key = LEAGUE_SLUGS.get(slug)
        if not ss_key:
            return []

        cache = _cache_path(slug, "timing")
        if not force and _is_fresh(cache, hours=24):
            return json.loads(cache.read_text(encoding="utf-8"))

        url  = f"{self.BASE}/timing.asp?league={ss_key}"
        html = _get(url, delay=4.0)
        if not html:
            return []

        rows = self._parse_timing(html)
        if rows:
            cache.write_text(json.dumps(rows, ensure_ascii=False, indent=2),
                             encoding="utf-8")
        return rows

    def _parse_timing(self, html):
        soup   = BeautifulSoup(html, "html.parser")
        result = []
        INTERVALS = ["0_15","16_30","31_45","46_60","61_75","76_90"]
        for t in soup.find_all("table"):
            rows = t.find_all("tr")
            for tr in rows[1:]:
                tds = tr.find_all(["td","th"])
                if len(tds) < 4:
                    continue
                texts = [td.get_text(strip=True) for td in tds]
                try:
                    team   = _normalize_team(texts[0])
                    row_d  = {"team": team}
                    for i, itv in enumerate(INTERVALS):
                        if i+1 < len(texts):
                            row_d[f"goals_pct_{itv}"] = _parse_float(texts[i+1])
                    # Late goals flag (>50% en 61-90)
                    late = row_d.get("goals_pct_61_75",0) + row_d.get("goals_pct_76_90",0)
                    row_d["late_goals_pct"] = round(late, 1)
                    if team:
                        result.append(row_d)
                except (IndexError, ValueError):
                    continue
        return result

    def get_all(self, slug, force=False):
        """
        Obtiene todos los datos disponibles para una liga.
        Returns: {"stats":[], "home_away":{}, "timing":[]}
        """
        log.info(f"SoccerStats: descargando datos para '{slug}'")
        data = {
            "slug":     slug,
            "fetched":  datetime.now().isoformat(),
            "stats":    self.league_stats(slug, force=force),
            "home_away":self.home_away(slug, force=force),
            "timing":   self.goal_timing(slug, force=force),
        }
        # Merge home_away en stats por nombre de equipo normalizado
        if data["stats"] and data["home_away"]:
            for row in data["stats"]:
                team = row.get("team","")
                ha   = data["home_away"].get(team, {})
                row.update(ha)
        log.info(
            f"SoccerStats '{slug}': "
            f"{len(data['stats'])} equipos, "
            f"{len(data['home_away'])} H/A, "
            f"{len(data['timing'])} timings"
        )
        return data

    def apply_to_df(self, df, slug, home_col="home_team", away_col="away_team"):
        """
        Enriquece un DataFrame de partidos con columnas de SoccerStats.
        Agrega: ss_home_gf_pg, ss_home_ga_pg, ss_away_gf_pg, ss_away_ga_pg,
                ss_home_over25, ss_away_over25, ss_home_btts, ss_away_btts
        """
        import pandas as pd
        data = self.get_all(slug)
        if not data["stats"]:
            df["ss_source"] = "none"
            return df

        stats_map = {r["team"]: r for r in data["stats"] if "team" in r}
        df = df.copy()
        df["ss_source"] = "soccerstats"

        for col, team_col, sfx in [(home_col,"home"), (away_col,"away")]:
            for src_key, dst_key in [
                ("home_gf_pg", f"ss_{sfx}_gf_pg"),
                ("home_ga_pg", f"ss_{sfx}_ga_pg"),
                ("away_gf_pg", f"ss_{sfx}_away_gf_pg"),
                ("over25_pct", f"ss_{sfx}_over25_pct"),
                ("btts_pct",   f"ss_{sfx}_btts_pct"),
            ]:
                df[dst_key] = df[col].map(
                    lambda n, sk=src_key: stats_map.get(n, {}).get(sk, None)
                )
        return df


# ── Función de cobertura ──────────────────────────────────────────────
def ss_coverage_summary():
    """Lista las ligas disponibles en SoccerStats."""
    import pandas as pd
    rows = [{"Liga": k, "ss_slug": v} for k, v in LEAGUE_SLUGS.items()]
    return pd.DataFrame(rows)
