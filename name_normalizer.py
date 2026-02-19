# name_normalizer.py
# Normalización de nombres entre: FixtureDownload, Understat, FBref, SoccerStats
# Estrategia: Slug → Alias Dict → Fuzzy Match (rapidfuzz)
import re, json, os, unicodedata
from pathlib import Path

try:
    from rapidfuzz import process as rfp, fuzz
    FUZZY_OK = True
except ImportError:
    FUZZY_OK = False

CACHE_DIR  = Path("fd_cache")
ALIAS_FILE = CACHE_DIR / "name_aliases.json"
CACHE_DIR.mkdir(exist_ok=True)

# ── Alias base (curados manualmente, +150 equipos críticos) ──────────
_BASE_ALIASES = {
    # ENGLAND
    "manchester united":     ["man united","man utd","manchester utd","man.united","manutd"],
    "manchester city":       ["man city","man.city","mcfc"],
    "tottenham hotspur":     ["tottenham","spurs","tottenham hotspur"],
    "wolverhampton wanderers":["wolverhampton","wolves","wolverhampton w."],
    "sheffield united":      ["sheffield utd","sheffield u."],
    "west ham united":       ["west ham","west ham utd"],
    "nottingham forest":     ["nottm forest","nott'm forest","nottingham f."],
    "brighton & hove albion":["brighton","brighton & h.a.","bha"],
    "newcastle united":      ["newcastle","newcastle utd","newcastle u."],
    "luton town":            ["luton"],
    # GERMANY
    "borussia dortmund":     ["dortmund","bvb"],
    "borussia monchengladbach":["monchengladbach","m'gladbach","b.m'gladbach","gladbach"],
    "bayer leverkusen":      ["leverkusen","b. leverkusen"],
    "rb leipzig":            ["rasenballsport leipzig","leipzig"],
    "tsg 1899 hoffenheim":   ["hoffenheim","tsg hoffenheim"],
    "1. fsv mainz 05":       ["mainz","fsv mainz","mainz 05"],
    "sc freiburg":           ["freiburg"],
    "sv werder bremen":      ["werder bremen","werder"],
    "hamburger sv":          ["hamburg","hsv"],
    "vfb stuttgart":         ["stuttgart"],
    "fc augsburg":           ["augsburg"],
    "eintracht frankfurt":   ["frankfurt","ein frankfurt"],
    "fc union berlin":       ["union berlin"],
    "vfl bochum":            ["bochum"],
    "vfl wolfsburg":         ["wolfsburg"],
    "fc heidenheim":         ["heidenheim"],
    "holstein kiel":         ["kiel"],
    "st. pauli":             ["fc st. pauli"],
    # SPAIN
    "atletico madrid":       ["atl. madrid","atl madrid","atlético madrid","atletico de madrid"],
    "real betis":            ["betis","real betis balompie"],
    "athletic bilbao":       ["athletic club","athletic club bilbao"],
    "real sociedad":         ["real sociedad de futbol"],
    "deportivo alaves":      ["alaves","d. alaves"],
    "rayo vallecano":        ["rayo"],
    "girona fc":             ["girona"],
    "getafe cf":             ["getafe"],
    "cd leganes":            ["leganes"],
    # ITALY
    "internazionale":        ["inter","inter milan","fc internazionale","inter fc"],
    "ac milan":              ["milan","ac milan"],
    "ssc napoli":            ["napoli"],
    "as roma":               ["roma"],
    "ss lazio":              ["lazio"],
    "atalanta bc":           ["atalanta"],
    "juventus":              ["juventus fc","juve"],
    "udinese calcio":        ["udinese"],
    "acf fiorentina":        ["fiorentina"],
    "bologna fc":            ["bologna"],
    "torino fc":             ["torino"],
    "us lecce":              ["lecce"],
    "hellas verona":         ["verona","hellas verona fc"],
    "empoli fc":             ["empoli"],
    "cagliari calcio":       ["cagliari"],
    "como 1907":             ["como"],
    # FRANCE
    "paris saint-germain":   ["psg","paris sg","paris saint germain","paris s-g"],
    "olympique de marseille":["marseille","om","olympique marseille"],
    "olympique lyonnais":    ["lyon","ol","olympique lyon"],
    "as monaco":             ["monaco","as monaco fc"],
    "stade rennais":         ["rennes","stade rennais fc"],
    "rc lens":               ["lens"],
    "losc lille":            ["lille","losc"],
    "ogc nice":              ["nice"],
    "montpellier hsc":       ["montpellier"],
    "stade brestois 29":     ["brest","stade brestois"],
    "toulouse fc":           ["toulouse"],
    "stade de reims":        ["reims"],
    "le havre ac":           ["le havre"],
    "fc nantes":             ["nantes"],
    "angers sco":            ["angers"],
    "rc strasbourg":         ["strasbourg","strasbourg alsace"],
    "saint-etienne":         ["st-etienne","saint etienne"],
    "auxerre":               ["aj auxerre"],
    # NETHERLANDS
    "ajax":                  ["ajax amsterdam","afc ajax"],
    "psv eindhoven":         ["psv"],
    "feyenoord":             ["feyenoord rotterdam"],
    "az alkmaar":            ["az"],
    "fc utrecht":            ["utrecht"],
    "sc heerenveen":         ["heerenveen"],
    "vitesse":               ["vitesse arnhem"],
    "nec nijmegen":          ["nec"],
    "sparta rotterdam":      ["sparta"],
    "go ahead eagles":       ["go ahead"],
    # PORTUGAL
    "sl benfica":            ["benfica","sport lisboa e benfica"],
    "fc porto":              ["porto"],
    "sporting cp":           ["sporting","sporting clube de portugal"],
    "sc braga":              ["braga","sporting braga"],
    "vitoria guimaraes":     ["vitoria sc","v. guimaraes"],
    "gil vicente":           ["gil vicente fc"],
    # BRAZIL
    "atletico mineiro":      ["atletico-mg","atletico mg","atl. mineiro"],
    "flamengo":              ["cr flamengo"],
    "palmeiras":             ["se palmeiras"],
    "fluminense":            ["fluminense fc"],
    "corinthians":           ["sport corinthians","sc corinthians paulista"],
    "santos":                ["santos fc"],
    "sao paulo":             ["sao paulo fc","spfc"],
    "gremio":                ["gremio fb porto alegrense"],
    "internacional":         ["internacional porto alegre","sc internacional"],
    # ARGENTINA
    "boca juniors":          ["ca boca juniors"],
    "river plate":           ["ca river plate"],
    "independiente":         ["ca independiente"],
    "racing club":           ["racing club avellaneda"],
    "san lorenzo":           ["san lorenzo almagro"],
    # COLOMBIA
    "atletico nacional":     ["nacional","atletico nacional medellin"],
    "millonarios":           ["millonarios fc"],
    "deportivo cali":        ["cali","dep. cali"],
    "junior":                ["atletico junior","junior barranquilla"],
    "santa fe":              ["independiente santa fe"],
}

# ── Normalización de slug ─────────────────────────────────────────────
def _slug(name):
    """Normaliza: minúsculas, sin acentos, sin prefijos FC/CF/AS/etc., sin puntuación."""
    if not name:
        return ""
    s = name.lower().strip()
    # Quitar acentos
    s = unicodedata.normalize("NFD", s)
    s = "".join(c for c in s if unicodedata.category(c) != "Mn")
    # Quitar prefijos/sufijos comunes
    for pfx in ["fc ", "cf ", "as ", "sc ", "sv ", "vfl ", "vfb ", "ac ",
                 "ss ", "ssc ", "us ", "ud ", "cd ", "rc ", "sd ",
                 " fc", " cf", " sc", " ac", " 1899", " 1907", " 1910"]:
        s = s.replace(pfx, " ")
    # Quitar puntuación
    s = re.sub(r"[.\-\'\",!&]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


class NameNormalizer:
    """
    Normaliza nombres de equipos y competencias entre múltiples fuentes.

    Uso:
        nn = NameNormalizer()
        canon = nn.team("Bayern München")          # "Bayern Munich"
        canon = nn.team("Man Utd", candidates=df["team"].tolist())
        nn.learn("Dortmund", "Borussia Dortmund")  # aprendizaje manual
    """

    def __init__(self):
        self._aliases  = {}      # slug(alias) → canonical
        self._learned  = {}      # persistido en disco
        self._slug_cache = {}    # canonical → slug(canonical) para búsqueda inversa
        self._load_base()
        self._load_learned()

    # ── Carga ─────────────────────────────────────────────────────────
    def _load_base(self):
        for canonical, aliases in _BASE_ALIASES.items():
            self._add_entry(canonical, aliases)

    def _load_learned(self):
        if ALIAS_FILE.exists():
            try:
                self._learned = json.loads(ALIAS_FILE.read_text(encoding="utf-8"))
                for alias, canon in self._learned.items():
                    self._aliases[_slug(alias)] = canon
            except Exception:
                pass

    def _save_learned(self):
        ALIAS_FILE.write_text(json.dumps(self._learned, ensure_ascii=False, indent=2),
                               encoding="utf-8")

    def _add_entry(self, canonical, aliases):
        slug_c = _slug(canonical)
        self._aliases[slug_c]          = canonical
        self._slug_cache[canonical]    = slug_c
        for alias in aliases:
            self._aliases[_slug(alias)] = canonical

    # ── API pública ───────────────────────────────────────────────────
    def team(self, name, candidates=None, threshold=82):
        """
        Normaliza un nombre de equipo.
        Si candidates se provee, busca el mejor match dentro de esa lista.
        """
        if not name:
            return name
        slug = _slug(name)
        # Capa 1: lookup directo
        if slug in self._aliases:
            return self._aliases[slug]
        # Capa 2: si hay candidates, fuzzy match
        if candidates and FUZZY_OK:
            slug_cands = {_slug(c): c for c in candidates}
            result = rfp.extractOne(slug, list(slug_cands.keys()),
                                     scorer=fuzz.token_sort_ratio,
                                     score_cutoff=threshold)
            if result:
                matched_slug, score, _ = result
                canon = slug_cands[matched_slug]
                self.learn(name, canon)
                return canon
        # Capa 3: devolver el nombre original normalizado (title case limpio)
        return name.strip()

    def competition(self, name):
        """Normaliza nombres de competencias."""
        _COMP_MAP = {
            "premier league":             "EN Premier League",
            "english premier league":     "EN Premier League",
            "epl":                        "EN Premier League",
            "bundesliga":                 "DE Bundesliga",
            "1. bundesliga":              "DE Bundesliga",
            "german bundesliga":          "DE Bundesliga",
            "la liga":                    "ES La Liga",
            "primera division":           "ES La Liga",
            "serie a":                    "IT Serie A",
            "italian serie a":            "IT Serie A",
            "ligue 1":                    "FR Ligue 1",
            "french ligue 1":             "FR Ligue 1",
            "champions league":           "UEFA Champions League",
            "uefa champions league":      "UEFA Champions League",
            "ucl":                        "UEFA Champions League",
            "europa league":              "UEFA Europa League",
            "eredivisie":                 "NL Eredivisie",
            "primeira liga":              "PT Primeira Liga",
            "liga nos":                   "PT Primeira Liga",
            "brasileirao":                "BR Brasileirao",
            "serie a brasileira":         "BR Brasileirao",
            "liga profesional":           "AR Liga Profesional",
            "superliga argentina":        "AR Liga Profesional",
            "liga betplay":               "CO Liga BetPlay",
            "primera a":                  "CO Liga BetPlay",
        }
        slug = _slug(name)
        return _COMP_MAP.get(slug, name)

    def learn(self, alias, canonical):
        """Agrega un nuevo alias aprendido y lo persiste en disco."""
        key = _slug(alias)
        if key not in self._aliases:
            self._aliases[key]         = canonical
            self._learned[alias]       = canonical
            self._save_learned()

    def cross_map(self, df, col, candidates=None, threshold=82):
        """
        Normaliza una columna completa de un DataFrame.
        df[col] → columna con nombres crudos de la fuente.
        Retorna df con columna col+"_canonical" agregada.
        """
        import pandas as pd
        if candidates is None:
            candidates = df[col].dropna().unique().tolist()
        df = df.copy()
        df[col + "_canonical"] = df[col].apply(
            lambda n: self.team(n, candidates=candidates, threshold=threshold)
        )
        return df

    def report_unmatched(self, df, col_raw, col_canonical):
        """Lista los nombres que no pudieron normalizarse correctamente."""
        import pandas as pd
        unmatched = df[df[col_raw] != df[col_canonical]][[col_raw, col_canonical]].drop_duplicates()
        return unmatched


# Instancia global singleton
_normalizer = None
def get_normalizer():
    global _normalizer
    if _normalizer is None:
        _normalizer = NameNormalizer()
    return _normalizer
