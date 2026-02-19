# fd_discovery.py — Auto-descubrimiento de TODAS las ligas de FixtureDownload
# Scraping de https://fixturedownload.com/sport/football
import re, requests, time, json, os
import pandas as pd
from bs4 import BeautifulSoup
from datetime import datetime, timezone
from catalog_builder import SEASON_TYPE, season_label, SLUG_META

FD_BASE    = 'https://fixturedownload.com'
FD_SPORT   = 'https://fixturedownload.com/sport/football'
HEADERS    = {'User-Agent':'Mozilla/5.0 (Windows NT 10.0; Win64; x64)',
              'Accept':'text/html,application/xhtml+xml'}
CACHE_DIR  = './fd_cache'
DISC_CACHE = './fd_cache/fd_discovery.json'
DISC_TTL_H = 24

# ── Descubrir todas las ligas desde la pagina de FD ───────────────────
def scrape_fd_leagues(delay=1.5):
    """
    Scraping de fixturedownload.com/sport/football.
    Extrae todos los slugs + anos disponibles de los links href.
    Retorna: dict[slug] = {name, seasons:[int], season_type, region}
    """
    try:
        r = requests.get(FD_SPORT, headers=HEADERS, timeout=30)
        r.raise_for_status()
    except Exception as e:
        print(f'FD-DISC ERROR: {e}')
        return {}

    soup  = BeautifulSoup(r.text, 'html.parser')
    found = {}

    # Patron: href='/results/epl-2025' o '/download/epl-2025' o '/feed/json/epl-2025'
    pattern = re.compile(r'(?:results|download|view-results)/([a-z0-9][a-z0-9\-]+)-(\d{4})(?:[^a-z]|$)')

    for a in soup.find_all('a', href=True):
        m = pattern.search(a['href'])
        if not m: continue
        slug = m.group(1)
        year = int(m.group(2))
        if slug not in found:
            # Intentar obtener el nombre del link o del texto adyacente
            name = a.get_text(strip=True) or slug.replace('-',' ').title()
            found[slug] = {'name':name, 'seasons':set(), 'raw_links':set()}
        found[slug]['seasons'].add(year)
        found[slug]['raw_links'].add(a['href'])

    # Tambien buscar en texto plano de la pagina por si el HTML es dinamico
    text_links = pattern.findall(r.text)
    for slug, year_s in text_links:
        year = int(year_s)
        if slug not in found:
            found[slug] = {'name':slug.replace('-',' ').title(),'seasons':set(),'raw_links':set()}
        found[slug]['seasons'].add(year)

    # Si la pagina devolvio poco (JS dinamico), hacer probe manual de slugs conocidos
    if len(found) < 5:
        print('FD-DISC: pagina dinamica detectada, usando probe de slugs conocidos...')
        found = _probe_known_slugs(delay=delay)

    # Construir resultado final
    result = {}
    for slug, info in found.items():
        seasons = sorted(info['seasons'])
        if not seasons: continue
        stype   = SEASON_TYPE.get(slug, 'single')
        meta    = SLUG_META.get(slug, {})
        result[slug] = {
            'name':    meta.get('name', info['name']),
            'emoji':   meta.get('emoji', '🌐'),
            'region':  meta.get('region', 'World'),
            'season_type': stype,
            'seasons': seasons,
            'current': max(seasons),
            'current_label': season_label(slug, max(seasons)),
        }
    print(f'FD-DISC: {len(result)} competiciones encontradas')
    return result

def _probe_known_slugs(delay=0.5):
    """Probe de slugs conocidos cuando el scraping directo falla (pagina JS)."""
    from catalog_builder import SLUG_META, load_or_refresh_catalog
    cat = load_or_refresh_catalog(fast_mode=True)
    result = {}
    for slug, comp in cat.get('competitions',{}).items():
        result[slug] = {
            'name': comp['name'], 'seasons':comp['seasons'],
            'raw_links': set()
        }
    return result

def full_fd_download(slugs_dict, force=False, delay=1.2, progress_fn=None):
    """
    Descarga TODAS las temporadas de TODAS las ligas descubiertas.
    Retorna DataFrame gigante con toda la data.
    """
    from predictor import DataCollector, DataProcessor
    col  = DataCollector(delay=delay)
    proc = DataProcessor()
    total_ids = sum(len(v['seasons']) for v in slugs_dict.values())
    done = 0; frames = []

    for slug, info in slugs_dict.items():
        for year in info['seasons']:
            cid = f'{slug}-{year}'
            done += 1
            if progress_fn:
                pct = done / total_ids
                progress_fn(f'Descargando {cid} ({done}/{total_ids})', pct * 0.85)
            data = col.download_json(cid, force=force)
            if data:
                frames.append(col._to_df(data, cid))

    if not frames: return pd.DataFrame()
    raw = pd.concat(frames, ignore_index=True)
    if progress_fn: progress_fn('Procesando...', 0.95)
    return proc.process(raw)

def load_or_refresh_discovery(force=False):
    os.makedirs(CACHE_DIR, exist_ok=True)
    if os.path.exists(DISC_CACHE) and not force:
        try:
            with open(DISC_CACHE,'r',encoding='utf-8') as f: d=json.load(f)
            ts  = datetime.fromisoformat(d.get('scraped_at','2000-01-01T00:00:00+00:00'))
            age = (datetime.now(timezone.utc)-ts).total_seconds()/3600
            if age < DISC_TTL_H: return d.get('leagues',{})
        except: pass
    leagues = scrape_fd_leagues()
    with open(DISC_CACHE,'w',encoding='utf-8') as f:
        json.dump({'scraped_at':datetime.now(timezone.utc).isoformat(),
                   'total':len(leagues),'leagues':leagues},f,ensure_ascii=False,indent=2)
    return leagues