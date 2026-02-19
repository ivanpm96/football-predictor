# catalog_builder.py v2  —  season_type: split / single / tournament
import os, json, time, requests
from datetime import datetime, timezone

CATALOG_FILE  = './fd_cache/competitions.json'
CATALOG_TTL_H = 168
HEADERS = {'User-Agent':'Mozilla/5.0','Accept':'application/json'}

SEASON_TYPE = {
    'epl':'split','championship':'split','league-one':'split','league-two':'split',
    'bundesliga':'split','2-bundesliga':'split','la-liga':'split','la-liga2':'split',
    'serie-a':'split','serie-b':'split','ligue-1':'split','ligue-2':'split',
    'eredivisie':'split','primeira-liga':'split','super-lig':'split',
    'scottish-premiership':'split','jupiler-pro-league':'split',
    'champions-league':'split','europa-league':'split','conference-league':'split',
    'j-league':'split','k-league-1':'split','saudi-pro-league':'split',
    'aleague-men':'split','aleague-women':'split','wsl':'split',
    'mls':'single','nwsl':'single','usl-championship':'single',
    'brasileirao':'single','primera-division-arg':'single',
    'copa-america':'tournament','afcon':'tournament','fifa-world-cup':'tournament',
    'uefa-euro':'tournament','nations-league':'tournament',
    'fifa-club-world-cup':'tournament','fifa-u-20-world-cup':'tournament',
}

def season_label(slug, year):
    if SEASON_TYPE.get(slug,'single') == 'split':
        return f"{year}/{str(year+1)[-2:]}"
    return str(year)

def season_labels_list(slug, years):
    return [season_label(slug, y) for y in years]

SLUG_META = {
    'epl':{'name':'Premier League','emoji':'EN','region':'England'},
    'championship':{'name':'Championship','emoji':'EN','region':'England'},
    'league-one':{'name':'League One','emoji':'EN','region':'England'},
    'league-two':{'name':'League Two','emoji':'EN','region':'England'},
    'bundesliga':{'name':'Bundesliga','emoji':'DE','region':'Germany'},
    '2-bundesliga':{'name':'2. Bundesliga','emoji':'DE','region':'Germany'},
    'la-liga':{'name':'La Liga','emoji':'ES','region':'Spain'},
    'la-liga2':{'name':'La Liga 2','emoji':'ES','region':'Spain'},
    'serie-a':{'name':'Serie A','emoji':'IT','region':'Italy'},
    'serie-b':{'name':'Serie B','emoji':'IT','region':'Italy'},
    'ligue-1':{'name':'Ligue 1','emoji':'FR','region':'France'},
    'ligue-2':{'name':'Ligue 2','emoji':'FR','region':'France'},
    'eredivisie':{'name':'Eredivisie','emoji':'NL','region':'Netherlands'},
    'primeira-liga':{'name':'Primeira Liga','emoji':'PT','region':'Portugal'},
    'super-lig':{'name':'Super Lig','emoji':'TR','region':'Turkey'},
    'scottish-premiership':{'name':'Scottish Premiership','emoji':'SC','region':'Scotland'},
    'jupiler-pro-league':{'name':'Jupiler Pro League','emoji':'BE','region':'Belgium'},
    'champions-league':{'name':'Champions League','emoji':'EU','region':'UEFA'},
    'europa-league':{'name':'Europa League','emoji':'EU','region':'UEFA'},
    'conference-league':{'name':'Conference League','emoji':'EU','region':'UEFA'},
    'uefa-euro':{'name':'UEFA Euro','emoji':'EU','region':'UEFA'},
    'nations-league':{'name':'Nations League','emoji':'EU','region':'UEFA'},
    'mls':{'name':'MLS','emoji':'US','region':'USA'},
    'nwsl':{'name':'NWSL','emoji':'US','region':'USA'},
    'aleague-men':{'name':'A-League Men','emoji':'AU','region':'Australia'},
    'copa-america':{'name':'Copa America','emoji':'SA','region':'CONMEBOL'},
    'brasileirao':{'name':'Brasileirao','emoji':'BR','region':'Brazil'},
    'primera-division-arg':{'name':'Liga Argentina','emoji':'AR','region':'Argentina'},
    'afcon':{'name':'AFCON','emoji':'AF','region':'Africa'},
    'j-league':{'name':'J1 League','emoji':'JP','region':'Japan'},
    'k-league-1':{'name':'K League 1','emoji':'KR','region':'Korea'},
    'saudi-pro-league':{'name':'Saudi Pro League','emoji':'SA','region':'Saudi Arabia'},
    'fifa-world-cup':{'name':'FIFA World Cup','emoji':'WC','region':'FIFA'},
    'fifa-club-world-cup':{'name':'FIFA Club WC','emoji':'WC','region':'FIFA'},
}

def _entry(slug, meta, seasons, current, train):
    stype = SEASON_TYPE.get(slug,'single')
    return {
        'name':meta['name'],'emoji':meta['emoji'],'region':meta['region'],
        'label':f"{meta['emoji']} {meta['name']}",
        'season_type':stype,'seasons':seasons,
        'season_labels':season_labels_list(slug,seasons),
        'current_season':current,'current_label':season_label(slug,current),
        'train_seasons':train,'train_labels':season_labels_list(slug,train),
        'predict_season':current,'predict_label':season_label(slug,current),
    }

def probe_season(slug, year, delay=0.4):
    url = f'https://fixturedownload.com/feed/json/{slug}-{year}'
    try:
        r = requests.head(url,headers=HEADERS,timeout=10,allow_redirects=True)
        if r.status_code==200: time.sleep(delay); return True
        r2 = requests.get(url,headers=HEADERS,timeout=10,stream=True)
        r2.raw.read(200); r2.close(); time.sleep(delay)
        return r2.status_code==200
    except: return False

def build_catalog(start_year=2015, end_year=None, delay=0.5, progress_fn=None):
    if end_year is None: end_year = datetime.now().year+1
    years = list(range(start_year, end_year+1))
    total = len(SLUG_META)*len(years); done = 0; catalog = {}
    for slug,meta in SLUG_META.items():
        valid = []
        for year in years:
            done += 1
            if progress_fn: progress_fn(f'{slug}-{year} ({done}/{total})',done/total)
            if probe_season(slug,year,delay): valid.append(year)
        if valid:
            cur = max(valid); tr = [y for y in valid if y<cur] or valid
            catalog[slug] = _entry(slug,meta,valid,cur,tr)
    regions = {}
    for slug,d in catalog.items():
        r=d['region']
        if r not in regions: regions[r]=[]
        regions[r].append(slug)
    return {'last_updated':datetime.now(timezone.utc).isoformat(),
            'total_comps':len(catalog),'regions':regions,'competitions':catalog}

def load_or_refresh_catalog(force=False, fast_mode=True, progress_fn=None):
    os.makedirs('./fd_cache',exist_ok=True)
    if os.path.exists(CATALOG_FILE) and not force:
        try:
            with open(CATALOG_FILE,'r',encoding='utf-8') as f: cat=json.load(f)
            ts  = datetime.fromisoformat(cat.get('last_updated','2000-01-01T00:00:00+00:00'))
            age = (datetime.now(timezone.utc)-ts).total_seconds()/3600
            if age < CATALOG_TTL_H: return cat
        except: pass
    if fast_mode and not force: return _fallback_catalog()
    cat = build_catalog(progress_fn=progress_fn)
    with open(CATALOG_FILE,'w',encoding='utf-8') as f: json.dump(cat,f,ensure_ascii=False,indent=2)
    return cat

def _fallback_catalog():
    M = SLUG_META; S = SEASON_TYPE
    r18=list(range(2018,2026)); r19=list(range(2019,2026))
    r20=list(range(2020,2026)); r17=list(range(2017,2026))
    C = {
        'epl':_entry('epl',M['epl'],r19,2025,list(range(2019,2025))),
        'championship':_entry('championship',M['championship'],r19,2025,list(range(2019,2025))),
        'league-one':_entry('league-one',M['league-one'],r20,2025,list(range(2020,2025))),
        'league-two':_entry('league-two',M['league-two'],r20,2025,list(range(2020,2025))),
        'bundesliga':_entry('bundesliga',M['bundesliga'],r18,2025,list(range(2018,2025))),
        '2-bundesliga':_entry('2-bundesliga',M['2-bundesliga'],r20,2025,list(range(2020,2025))),
        'la-liga':_entry('la-liga',M['la-liga'],r18,2025,list(range(2018,2025))),
        'la-liga2':_entry('la-liga2',M['la-liga2'],r20,2025,list(range(2020,2025))),
        'serie-a':_entry('serie-a',M['serie-a'],r17,2025,list(range(2017,2025))),
        'serie-b':_entry('serie-b',M['serie-b'],r20,2025,list(range(2020,2025))),
        'ligue-1':_entry('ligue-1',M['ligue-1'],r18,2025,list(range(2018,2025))),
        'ligue-2':_entry('ligue-2',M['ligue-2'],r20,2025,list(range(2020,2025))),
        'eredivisie':_entry('eredivisie',M['eredivisie'],[2022,2023,2024,2025],2025,[2022,2023,2024]),
        'primeira-liga':_entry('primeira-liga',M['primeira-liga'],[2020,2023,2024,2025],2025,[2020,2023,2024]),
        'super-lig':_entry('super-lig',M['super-lig'],r19,2025,list(range(2019,2025))),
        'scottish-premiership':_entry('scottish-premiership',M['scottish-premiership'],[2022,2023,2024,2025],2025,[2022,2023,2024]),
        'jupiler-pro-league':_entry('jupiler-pro-league',M['jupiler-pro-league'],[2022,2023,2024,2025],2025,[2022,2023,2024]),
        'champions-league':_entry('champions-league',M['champions-league'],r17,2025,list(range(2017,2025))),
        'europa-league':_entry('europa-league',M['europa-league'],r19,2025,list(range(2019,2025))),
        'conference-league':_entry('conference-league',M['conference-league'],[2023,2024,2025],2025,[2023,2024]),
        'uefa-euro':_entry('uefa-euro',M['uefa-euro'],[2016,2020,2024],2024,[2016,2020]),
        'nations-league':_entry('nations-league',M['nations-league'],[2020,2022,2024],2024,[2020,2022]),
        'mls':_entry('mls',M['mls'],list(range(2019,2027)),2026,list(range(2019,2026))),
        'nwsl':_entry('nwsl',M['nwsl'],[2022,2023,2024,2025],2025,[2022,2023,2024]),
        'aleague-men':_entry('aleague-men',M['aleague-men'],[2022,2023,2024,2025],2025,[2022,2023,2024]),
        'copa-america':_entry('copa-america',M['copa-america'],[2016,2019,2021,2024],2024,[2016,2019,2021]),
        'brasileirao':_entry('brasileirao',M['brasileirao'],r19,2025,list(range(2019,2025))),
        'primera-division-arg':_entry('primera-division-arg',M['primera-division-arg'],[2022,2023,2024,2025],2025,[2022,2023,2024]),
        'afcon':_entry('afcon',M['afcon'],[2021,2023,2025],2025,[2021,2023]),
        'j-league':_entry('j-league',M['j-league'],[2022,2023,2024,2025],2025,[2022,2023,2024]),
        'k-league-1':_entry('k-league-1',M['k-league-1'],[2022,2023,2024,2025],2025,[2022,2023,2024]),
        'saudi-pro-league':_entry('saudi-pro-league',M['saudi-pro-league'],[2022,2023,2024,2025],2025,[2022,2023,2024]),
        'fifa-world-cup':_entry('fifa-world-cup',M['fifa-world-cup'],[2018,2022,2026],2026,[2018,2022]),
        'fifa-club-world-cup':_entry('fifa-club-world-cup',M['fifa-club-world-cup'],[2023,2025],2025,[2023]),
    }
    regions = {}
    for slug,d in C.items():
        r=d['region']
        if r not in regions: regions[r]=[]
        regions[r].append(slug)
    return {'last_updated':'2026-02-19T00:00:00+00:00','total_comps':len(C),
            'regions':regions,'competitions':C,'source':'fallback_v2'}

def get_regions(catalog):
    result={}
    for region,slugs in catalog['regions'].items():
        labels=[]
        for slug in slugs:
            comp=catalog['competitions'].get(slug)
            if comp: labels.append(comp['label'])
        if labels: result[region]=labels
    return result

def get_comp_by_label(catalog, label):
    for k,v in catalog['competitions'].items():
        if v['label']==label: return {'slug':k,**v}
    return {}

def get_season_options(comp_info):
    return {lbl:yr for lbl,yr in zip(comp_info.get('season_labels',[]),comp_info.get('seasons',[]))}

def get_train_defaults(comp_info):
    return comp_info.get('train_labels',[])

def get_predict_default_label(comp_info):
    return comp_info.get('predict_label',str(comp_info.get('predict_season','')))

if __name__ == '__main__':
    import sys
    cat = load_or_refresh_catalog(force='--force' in sys.argv, fast_mode=False)
    print(f'OK {cat["total_comps"]} competiciones -> {CATALOG_FILE}')