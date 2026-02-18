
import requests, json, os, time, warnings
import pandas as pd
import numpy as np
from scipy.stats import poisson
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_score
import joblib

warnings.filterwarnings("ignore")

# ════════════════════════════════════════════════════════════════════
# MAPA COMPLETO DE COMPETICIONES
# ════════════════════════════════════════════════════════════════════
COMPETITION_MAP = {
    "🏴󠁧󠁢󠁥󠁮󠁧󠁿 EPL (Premier League)":       {"base":"epl",                    "years":[2016,2017,2018,2019,2020,2021,2022,2023,2024,2025]},
    "🏴󠁧󠁢󠁥󠁮󠁧󠁿 Championship (England)":      {"base":"championship",           "years":[2019,2020,2021,2022,2023,2024,2025]},
    "🏴󠁧󠁢󠁥󠁮󠁧󠁿 WSL (Women Super League)":    {"base":"wsl",                    "years":[2023,2024,2025]},
    "🇩🇪 Bundesliga":                    {"base":"bundesliga",             "years":[2018,2019,2020,2021,2022,2023,2024,2025]},
    "🇪🇸 La Liga":                        {"base":"la-liga",                "years":[2018,2019,2020,2021,2022,2023,2024,2025]},
    "🇮🇹 Serie A":                        {"base":"serie-a",                "years":[2017,2018,2019,2020,2021,2022,2023,2024,2025]},
    "🇫🇷 Ligue 1":                        {"base":"ligue-1",                "years":[2018,2019,2020,2021,2022,2023,2024,2025]},
    "🇳🇱 Eredivisie":                    {"base":"eredivisie",             "years":[2023,2024,2025]},
    "🇵🇹 Primeira Liga":                  {"base":"primeira-liga",          "years":[2020,2023,2024,2025]},
    "🇹🇷 Süper Lig":                      {"base":"super-lig",              "years":[2019,2020,2021,2022,2023,2024,2025]},
    "🌍 UEFA Champions League":           {"base":"champions-league",       "years":[2017,2018,2019,2020,2021,2022,2023,2024,2025]},
    "🌍 UEFA Europa League":              {"base":"europa-league",          "years":[2019,2020,2021,2022,2023,2024,2025]},
    "🌍 UEFA Conference League":          {"base":"conference-league",      "years":[2024,2025]},
    "🌍 UEFA Euro":                       {"base":"uefa-euro",              "years":[2016,2020,2024]},
    "🌍 UEFA Nations League":             {"base":"nations-league",         "years":[2024]},
    "🌍 UEFA Women's Euro":               {"base":"uefa-womens-euro",       "years":[2025]},
    "🇺🇸 MLS":                            {"base":"mls",                    "years":[2023,2024,2025,2026]},
    "🇺🇸 NWSL (Women)":                   {"base":"nwsl",                   "years":[2024,2025,2026]},
    "🌎 Copa América":                    {"base":"copa-america",           "years":[2016,2019,2021,2024]},
    "🌐 FIFA World Cup":                  {"base":"fifa-world-cup",         "years":[2018,2022,2026]},
    "🌐 FIFA Women's World Cup":          {"base":"fifa-womens-world-cup",  "years":[2019,2023]},
    "🌐 FIFA Club World Cup":             {"base":"fifa-club-world-cup",    "years":[2025]},
    "🌐 FIFA U-20 World Cup":             {"base":"fifa-u-20-world-cup",    "years":[2025]},
    "🌍 Africa Cup of Nations (AFCON)":   {"base":"afcon",                  "years":[2023,2025]},
    "🇦🇺 A-League Men":                   {"base":"aleague-men",            "years":[2023,2024,2025]},
    "🇦🇺 A-League Women":                 {"base":"aleague-women",          "years":[2023,2024,2025]},
}

REGIONS = {
    "🏴󠁧󠁢󠁥󠁮󠁧󠁿 Inglaterra":               ["🏴󠁧󠁢󠁥󠁮󠁧󠁿 EPL (Premier League)","🏴󠁧󠁢󠁥󠁮󠁧󠁿 Championship (England)","🏴󠁧󠁢󠁥󠁮󠁧󠁿 WSL (Women Super League)"],
    "🌍 Europa — Ligas Nacionales":       ["🇩🇪 Bundesliga","🇪🇸 La Liga","🇮🇹 Serie A","🇫🇷 Ligue 1","🇳🇱 Eredivisie","🇵🇹 Primeira Liga","🇹🇷 Süper Lig"],
    "🌍 UEFA — Competiciones":            ["🌍 UEFA Champions League","🌍 UEFA Europa League","🌍 UEFA Conference League","🌍 UEFA Euro","🌍 UEFA Nations League","🌍 UEFA Women's Euro"],
    "🌎 Américas":                        ["🇺🇸 MLS","🇺🇸 NWSL (Women)","🌎 Copa América"],
    "🌐 Mundiales FIFA":                  ["🌐 FIFA World Cup","🌐 FIFA Women's World Cup","🌐 FIFA Club World Cup","🌐 FIFA U-20 World Cup"],
    "🌍 África / Oceanía":                ["🌍 Africa Cup of Nations (AFCON)","🇦🇺 A-League Men","🇦🇺 A-League Women"],
}

HEADERS = {"User-Agent":"Mozilla/5.0","Accept":"application/json"}

# ════════════════════════════════════════════════════════════════════
# DESCARGA
# ════════════════════════════════════════════════════════════════════
class DataCollector:
    def __init__(self, cache_dir="./fd_cache", delay=1.5):
        self.cache_dir=cache_dir; self.delay=delay
        os.makedirs(cache_dir, exist_ok=True)

    def build_ids(self, comp_label, years):
        meta=COMPETITION_MAP.get(comp_label)
        if not meta: raise ValueError(f"Competición desconocida: {comp_label}")
        return [f"{meta['base']}-{y}" for y in years if y in meta["years"]]

    def available_years(self, comp_label):
        return COMPETITION_MAP.get(comp_label,{}).get("years",[])

    def download_json(self, cid, force=False):
        cache=os.path.join(self.cache_dir,f"{cid}.json")
        if os.path.exists(cache) and not force:
            with open(cache,"r",encoding="utf-8") as f: return json.load(f)
        url=f"https://fixturedownload.com/feed/json/{cid}"
        try:
            r=requests.get(url,headers=HEADERS,timeout=20); r.raise_for_status()
            data=r.json()
            with open(cache,"w",encoding="utf-8") as f: json.dump(data,f,ensure_ascii=False,indent=2)
            time.sleep(self.delay); return data
        except Exception as e:
            print(f"  ❌ {cid}: {e}"); return []

    def _to_df(self, data, cid):
        rows=[]
        for m in data:
            rows.append({"competition_id":cid,"round":m.get("RoundNumber",""),
                "date":m.get("DateUtc",""),"home_team":(m.get("HomeTeam") or "").strip(),
                "away_team":(m.get("AwayTeam") or "").strip(),
                "home_score":m.get("HomeTeamScore"),"away_score":m.get("AwayTeamScore"),
                "location":m.get("Location","")})
        return pd.DataFrame(rows)

    def download_many(self, comp_ids, force=False, progress_fn=None):
        frames=[]; total=len(comp_ids)
        for i,cid in enumerate(comp_ids):
            if progress_fn: progress_fn(f"Descargando {cid} ({i+1}/{total})...",(i+1)/total*0.4)
            data=self.download_json(cid,force=force)
            if data: frames.append(self._to_df(data,cid))
        return pd.concat(frames,ignore_index=True) if frames else pd.DataFrame()

# ════════════════════════════════════════════════════════════════════
# PROCESAMIENTO
# ════════════════════════════════════════════════════════════════════
class DataProcessor:
    def process(self, df):
        df=df.copy()
        df["date"]=pd.to_datetime(df["date"],errors="coerce",utc=True)
        df["home_score"]=pd.to_numeric(df["home_score"],errors="coerce")
        df["away_score"]=pd.to_numeric(df["away_score"],errors="coerce")
        played=df["home_score"].notna()&df["away_score"].notna()
        df.loc[played,"result"]=np.where(
            df.loc[played,"home_score"]>df.loc[played,"away_score"],"H",
            np.where(df.loc[played,"home_score"]<df.loc[played,"away_score"],"A","D"))
        df["played"]=played
        df["total_goals"]=df["home_score"].fillna(0)+df["away_score"].fillna(0)
        df=df[df["home_team"].str.len()>0].copy()
        return df.sort_values("date").reset_index(drop=True)

# ════════════════════════════════════════════════════════════════════
# ESTADÍSTICAS AVANZADAS
# ════════════════════════════════════════════════════════════════════
class StatsEngine:
    """Motor de estadísticas crudas y avanzadas"""

    def league_table(self, df_proc, comp_id):
        """Tabla de posiciones completa calculada desde datos reales"""
        pl=df_proc[df_proc["played"]&(df_proc["competition_id"]==comp_id)].copy()
        if pl.empty: return pd.DataFrame()
        teams=sorted(set(pl["home_team"])|set(pl["away_team"]))
        rows=[]
        for team in teams:
            home_m=pl[pl["home_team"]==team]
            away_m=pl[pl["away_team"]==team]
            pj=len(home_m)+len(away_m)
            if pj==0: continue
            gf=home_m["home_score"].sum()+away_m["away_score"].sum()
            gc=home_m["away_score"].sum()+away_m["home_score"].sum()
            pg=(home_m["result"]=="H").sum()+(away_m["result"]=="A").sum()
            pe=(home_m["result"]=="D").sum()+(away_m["result"]=="D").sum()
            pp=pj-pg-pe
            pts=pg*3+pe
            last5=[]
            all_m=pd.concat([
                home_m.assign(_res=home_m["result"].map({"H":"W","D":"D","A":"L"})),
                away_m.assign(_res=away_m["result"].map({"A":"W","D":"D","H":"L"})),
            ]).sort_values("date").tail(5)
            last5="".join(all_m["_res"].tolist())
            rows.append({"Equipo":team,"PJ":pj,"PG":pg,"PE":pe,"PP":pp,
                         "GF":int(gf),"GC":int(gc),"GD":int(gf-gc),
                         "PTS":int(pts),"PPP":round(pts/pj,2),"Forma":last5})
        df_t=pd.DataFrame(rows).sort_values(["PTS","GD","GF"],ascending=False).reset_index(drop=True)
        df_t.index+=1
        return df_t

    def team_full_stats(self, df_proc, team, comp_id=None):
        """Estadísticas completas de un equipo: global, local, visitante"""
        pl=df_proc[df_proc["played"]].copy()
        if comp_id: pl=pl[pl["competition_id"]==comp_id]
        home_m=pl[pl["home_team"]==team]
        away_m=pl[pl["away_team"]==team]
        all_m=pd.concat([home_m,away_m])
        if all_m.empty: return {}

        def calc(matches, team, side):
            if len(matches)==0:
                return {f"{side}_PJ":0,f"{side}_PG":0,f"{side}_PE":0,f"{side}_PP":0,
                        f"{side}_GF":0,f"{side}_GC":0,f"{side}_GD":0,f"{side}_PTS":0,f"{side}_PPP":0}
            if side=="home":
                gf=matches["home_score"].sum(); gc=matches["away_score"].sum()
                pg=(matches["result"]=="H").sum(); pe=(matches["result"]=="D").sum()
            else:
                gf=matches["away_score"].sum(); gc=matches["home_score"].sum()
                pg=(matches["result"]=="A").sum(); pe=(matches["result"]=="D").sum()
            pj=len(matches); pp=pj-pg-pe; pts=pg*3+pe
            return {f"{side}_PJ":pj,f"{side}_PG":pg,f"{side}_PE":pe,f"{side}_PP":pp,
                    f"{side}_GF":int(gf),f"{side}_GC":int(gc),f"{side}_GD":int(gf-gc),
                    f"{side}_PTS":int(pts),f"{side}_PPP":round(pts/pj,2)}

        stats={}
        stats.update(calc(home_m,team,"home"))
        stats.update(calc(away_m,team,"away"))
        stats["total_PJ"]=stats["home_PJ"]+stats["away_PJ"]
        stats["total_PG"]=stats["home_PG"]+stats["away_PG"]
        stats["total_PE"]=stats["home_PE"]+stats["away_PE"]
        stats["total_PP"]=stats["home_PP"]+stats["away_PP"]
        stats["total_GF"]=stats["home_GF"]+stats["away_GF"]
        stats["total_GC"]=stats["home_GC"]+stats["away_GC"]
        stats["total_GD"]=stats["total_GF"]-stats["total_GC"]
        stats["total_PTS"]=stats["home_PTS"]+stats["away_PTS"]
        stats["total_PPP"]=round(stats["total_PTS"]/max(1,stats["total_PJ"]),2)
        stats["avg_gf"]=round(stats["total_GF"]/max(1,stats["total_PJ"]),2)
        stats["avg_gc"]=round(stats["total_GC"]/max(1,stats["total_PJ"]),2)
        stats["avg_total_goals"]=round(stats["avg_gf"]+stats["avg_gc"],2)
        stats["clean_sheets"]=int((home_m["away_score"]==0).sum()+(away_m["home_score"]==0).sum())
        stats["failed_to_score"]=int((home_m["home_score"]==0).sum()+(away_m["away_score"]==0).sum())
        stats["btts_count"]=int(((home_m["home_score"]>0)&(home_m["away_score"]>0)).sum()+
                                ((away_m["away_score"]>0)&(away_m["home_score"]>0)).sum())
        stats["over_2_5_count"]=int((all_m["total_goals"]>2.5).sum()) if "total_goals" in all_m.columns else 0
        # Racha actual
        combined=pd.concat([
            home_m.assign(_res=home_m["result"].map({"H":"W","D":"D","A":"L"})),
            away_m.assign(_res=away_m["result"].map({"A":"W","D":"D","H":"L"})),
        ]).sort_values("date")
        streak_list=combined["_res"].tolist()
        if streak_list:
            cur=streak_list[-1]; n_streak=1
            for r in reversed(streak_list[:-1]):
                if r==cur: n_streak+=1
                else: break
            stats["streak"]=f"{cur}{n_streak}"
        else:
            stats["streak"]="—"
        stats["last5"]=streak_list[-5:] if len(streak_list)>=5 else streak_list
        stats["last5_matches"]=combined.tail(5)[
            ["date","home_team","away_team","home_score","away_score","result","_res"]
        ].to_dict("records")
        return stats

    def h2h_detail(self, df_proc, home, away, comp_id=None):
        """Historial H2H completo con estadísticas"""
        pl=df_proc[df_proc["played"]].copy()
        if comp_id: pl=pl[pl["competition_id"]==comp_id]
        h2h=pl[((pl["home_team"]==home)&(pl["away_team"]==away))|
               ((pl["home_team"]==away)&(pl["away_team"]==home))].sort_values("date",ascending=False)
        if h2h.empty: return {"matches":pd.DataFrame(),"summary":{}}
        hw=dd=aw=0; hgf=agf=0
        for _,m in h2h.iterrows():
            if m["home_team"]==home:
                hgf+=m["home_score"]; agf+=m["away_score"]
                if m["result"]=="H": hw+=1
                elif m["result"]=="D": dd+=1
                else: aw+=1
            else:
                hgf+=m["away_score"]; agf+=m["home_score"]
                if m["result"]=="A": hw+=1
                elif m["result"]=="D": dd+=1
                else: aw+=1
        n=len(h2h)
        summary={"total":n,f"{home}_wins":hw,"draws":dd,f"{away}_wins":aw,
                 f"{home}_goals":hgf,f"{away}_goals":agf,
                 "avg_goals":round((hgf+agf)/n,2),"btts":int(
                     ((h2h["home_score"]>0)&(h2h["away_score"]>0)).sum())}
        return {"matches":h2h,"summary":summary}

    def raw_matches(self, df_proc, comp_id=None, team=None, result=None, limit=500):
        """Datos crudos filtrables"""
        pl=df_proc[df_proc["played"]].copy()
        if comp_id: pl=pl[pl["competition_id"]==comp_id]
        if team: pl=pl[(pl["home_team"]==team)|(pl["away_team"]==team)]
        if result: pl=pl[pl["result"]==result]
        pl=pl.sort_values("date",ascending=False).head(limit)
        pl["date_fmt"]=pl["date"].dt.strftime("%Y-%m-%d")
        pl["score"]=pl["home_score"].astype(int).astype(str)+"-"+pl["away_score"].astype(int).astype(str)
        cols=["competition_id","round","date_fmt","home_team","score","away_team","result","location"]
        return pl[[c for c in cols if c in pl.columns]].rename(columns={
            "competition_id":"Liga","round":"Jornada","date_fmt":"Fecha",
            "home_team":"Local","score":"Marcador","away_team":"Visitante",
            "result":"Res","location":"Estadio"})

# ════════════════════════════════════════════════════════════════════
# FEATURES
# ════════════════════════════════════════════════════════════════════
class FeatureEngineer:
    def __init__(self, n=5):
        self.n=n

    def _stats(self, matches, team, pfx):
        z={f"{pfx}gf":0.,f"{pfx}gc":0.,f"{pfx}pts":0.,
           f"{pfx}wins":0.,f"{pfx}draws":0.,f"{pfx}losses":0.,f"{pfx}n":0}
        if len(matches)==0: return z
        gf=gc=pts=w=d=l=0
        for _,m in matches.iterrows():
            is_h=m["home_team"]==team
            gs=m["home_score"] if is_h else m["away_score"]
            gc_=m["away_score"] if is_h else m["home_score"]
            wr="H" if is_h else "A"
            if m["result"]==wr: w+=1;pts+=3
            elif m["result"]=="D": d+=1;pts+=1
            else: l+=1
            gf+=gs; gc+=gc_
        n=len(matches)
        return {f"{pfx}gf":round(gf/n,3),f"{pfx}gc":round(gc/n,3),f"{pfx}pts":round(pts/n,3),
                f"{pfx}wins":round(w/n,3),f"{pfx}draws":round(d/n,3),f"{pfx}losses":round(l/n,3),f"{pfx}n":n}

    def _h2h(self, h2h, home, away):
        z={"h2h_hw":0.,"h2h_d":0.,"h2h_aw":0.,"h2h_hgf":0.,"h2h_agf":0.,"h2h_n":0}
        if len(h2h)==0: return z
        hw=dd=aw=hgf=agf=0
        for _,m in h2h.iterrows():
            if m["home_team"]==home:
                hgf+=m["home_score"]; agf+=m["away_score"]
                if m["result"]=="H": hw+=1
                elif m["result"]=="D": dd+=1
                else: aw+=1
            else:
                agf+=m["home_score"]; hgf+=m["away_score"]
                if m["result"]=="A": hw+=1
                elif m["result"]=="D": dd+=1
                else: aw+=1
        n=len(h2h)
        return {"h2h_hw":round(hw/n,3),"h2h_d":round(dd/n,3),"h2h_aw":round(aw/n,3),
                "h2h_hgf":round(hgf/n,3),"h2h_agf":round(agf/n,3),"h2h_n":n}

    def build(self, df):
        dfp=df[df["played"]].copy().reset_index(drop=True)
        feats=[]
        for _,row in dfp.iterrows():
            date,comp,home,away=row["date"],row["competition_id"],row["home_team"],row["away_team"]
            hist=dfp[(dfp["date"]<date)&(dfp["competition_id"]==comp)]
            feat={"competition_id":comp,"date":date,"home_team":home,"away_team":away,
                  "home_score":row["home_score"],"away_score":row["away_score"],"result":row["result"]}
            feat.update(self._stats(hist[(hist["home_team"]==home)|(hist["away_team"]==home)].tail(self.n),home,"h_"))
            feat.update(self._stats(hist[hist["home_team"]==home].tail(self.n),home,"hh_"))
            feat.update(self._stats(hist[(hist["home_team"]==away)|(hist["away_team"]==away)].tail(self.n),away,"a_"))
            feat.update(self._stats(hist[hist["away_team"]==away].tail(self.n),away,"aa_"))
            feat.update(self._h2h(hist[((hist["home_team"]==home)&(hist["away_team"]==away))|
                                       ((hist["home_team"]==away)&(hist["away_team"]==home))].tail(6),home,away))
            feats.append(feat)
        return pd.DataFrame(feats)

    def predict_feat(self, df_proc, home, away, comp):
        hist=df_proc[df_proc["played"]&(df_proc["competition_id"]==comp)].sort_values("date")
        feat={"competition_id":comp,"home_team":home,"away_team":away}
        for pfx,team,filt in [
            ("h_",home,lambda h:(h["home_team"]==home)|(h["away_team"]==home)),
            ("hh_",home,lambda h:h["home_team"]==home),
            ("a_",away,lambda h:(h["home_team"]==away)|(h["away_team"]==away)),
            ("aa_",away,lambda h:h["away_team"]==away),
        ]:
            feat.update(self._stats(hist[filt(hist)].tail(self.n),team,pfx))
        feat.update(self._h2h(hist[((hist["home_team"]==home)&(hist["away_team"]==away))|
                                   ((hist["home_team"]==away)&(hist["away_team"]==home))].tail(6),home,away))
        return pd.DataFrame([feat])

# ════════════════════════════════════════════════════════════════════
# POISSON
# ════════════════════════════════════════════════════════════════════
class PoissonModel:
    def __init__(self):
        self.attack={}; self.defense={}; self.home_adv=1.0

    def fit(self, df_proc):
        pl=df_proc[df_proc["played"]]
        for t in sorted(set(pl["home_team"])|set(pl["away_team"])):
            sc=pd.concat([pl[pl["home_team"]==t]["home_score"],pl[pl["away_team"]==t]["away_score"]])
            co=pd.concat([pl[pl["home_team"]==t]["away_score"],pl[pl["away_team"]==t]["home_score"]])
            self.attack[t]=max(0.3,sc.mean()) if len(sc) else 1.0
            self.defense[t]=max(0.3,co.mean()) if len(co) else 1.0
        avg_h=pl["home_score"].mean(); avg_a=pl["away_score"].mean()
        self.home_adv=round(avg_h/avg_a,3) if avg_a else 1.0

    def predict(self, home, away, max_g=8):
        lh=self.attack.get(home,1.2)*self.home_adv/max(0.1,self.defense.get(away,1.0))
        la=self.attack.get(away,1.0)/max(0.1,self.defense.get(home,1.2))
        M=np.outer([poisson.pmf(i,lh) for i in range(max_g+1)],
                   [poisson.pmf(j,la) for j in range(max_g+1)])
        pH=float(np.sum(np.tril(M,-1))); pD=float(np.sum(np.diag(M))); pA=float(np.sum(np.triu(M,1)))
        idx=np.unravel_index(np.argmax(M),M.shape)
        over15=sum(M[i][j] for i in range(max_g+1) for j in range(max_g+1) if i+j>1)
        over25=sum(M[i][j] for i in range(max_g+1) for j in range(max_g+1) if i+j>2)
        over35=sum(M[i][j] for i in range(max_g+1) for j in range(max_g+1) if i+j>3)
        btts  =sum(M[i][j] for i in range(1,max_g+1) for j in range(1,max_g+1))
        handi_h=sum(M[i][j] for i in range(max_g+1) for j in range(max_g+1) if (i-0.5)>j)
        handi_a=sum(M[i][j] for i in range(max_g+1) for j in range(max_g+1) if (j-0.5)>i)
        return {"prob_H":round(pH,4),"prob_D":round(pD,4),"prob_A":round(pA,4),
                "lambda_home":round(lh,3),"lambda_away":round(la,3),
                "most_likely_score":f"{idx[0]}-{idx[1]}",
                "prob_over_1_5":round(float(over15),4),"prob_under_1_5":round(1-float(over15),4),
                "prob_over_2_5":round(float(over25),4),"prob_under_2_5":round(1-float(over25),4),
                "prob_over_3_5":round(float(over35),4),"prob_under_3_5":round(1-float(over35),4),
                "prob_btts":round(float(btts),4),"prob_no_btts":round(1-float(btts),4),
                "asian_handi_home":round(float(handi_h),4),
                "asian_handi_away":round(float(handi_a),4),
                "score_matrix":M.tolist()}

# ════════════════════════════════════════════════════════════════════
# ML
# ════════════════════════════════════════════════════════════════════
class MLModel:
    FCOLS=["h_gf","h_gc","h_pts","h_wins","h_draws","h_losses",
           "hh_gf","hh_gc","hh_pts","a_gf","a_gc","a_pts","a_wins","a_draws","a_losses",
           "aa_gf","aa_gc","aa_pts","h2h_hw","h2h_d","h2h_aw","h2h_hgf","h2h_agf"]

    def __init__(self):
        self.model=GradientBoostingClassifier(n_estimators=300,max_depth=4,
            learning_rate=0.05,subsample=0.8,random_state=42)
        self.le=LabelEncoder(); self.trained=False; self.cv_score=0.0

    def _X(self,df):
        for c in self.FCOLS:
            if c not in df.columns: df[c]=0.0
        return df[self.FCOLS].fillna(0.0)

    def fit(self, df_feat):
        df=df_feat.dropna(subset=["result"])
        if len(df)<50: return
        X=self._X(df.copy()); y=self.le.fit_transform(df["result"])
        cv=cross_val_score(self.model,X,y,cv=5,scoring="accuracy")
        self.cv_score=round(float(cv.mean()),4)
        self.model.fit(X,y); self.trained=True

    def predict(self,df_feat):
        if not self.trained: return None
        X=self._X(df_feat.copy()); proba=self.model.predict_proba(X)[0]
        out={f"ml_prob_{c}":round(float(p),4) for c,p in zip(self.le.classes_,proba)}
        out["ml_prediction"]=self.le.classes_[np.argmax(proba)]
        return out

    def backtest(self, df_feat, last_n=100):
        """Evalúa modelo en últimos N partidos conocidos"""
        df=df_feat.dropna(subset=["result"]).tail(last_n)
        if not self.trained or len(df)<10: return pd.DataFrame()
        X=self._X(df.copy()); proba=self.model.predict_proba(X)
        classes=self.le.classes_
        rows=[]
        for i,(_,row) in enumerate(df.iterrows()):
            pred_idx=np.argmax(proba[i]); pred=classes[pred_idx]
            rows.append({"date":str(row.get("date",""))[:10],
                "home":row.get("home_team",""),"away":row.get("away_team",""),
                "real":row["result"],"pred":pred,
                "correct":row["result"]==pred,
                "conf":round(float(proba[i][pred_idx]),3),
                "p_H":round(float(proba[i][list(classes).index("H")] if "H" in classes else 0),3),
                "p_D":round(float(proba[i][list(classes).index("D")] if "D" in classes else 0),3),
                "p_A":round(float(proba[i][list(classes).index("A")] if "A" in classes else 0),3)})
        return pd.DataFrame(rows)

    def importances(self):
        if not self.trained: return {}
        return dict(sorted(zip(self.FCOLS,self.model.feature_importances_),key=lambda x:-x[1]))

# ════════════════════════════════════════════════════════════════════
# SISTEMA CENTRAL
# ════════════════════════════════════════════════════════════════════
class FootballSystem:
    def __init__(self):
        self.collector=DataCollector(); self.processor=DataProcessor()
        self.engineer=FeatureEngineer(); self.poisson=PoissonModel()
        self.ml=MLModel(); self.stats_engine=StatsEngine()
        self.df_proc=None; self.df_feat=None
        self.comp_ids=[]; self.trained=False; self.stats={}

    def available_years(self,comp_label):
        return self.collector.available_years(comp_label)

    def train(self, comp_label, years, force=False, progress_fn=None):
        ids=self.collector.build_ids(comp_label,years)
        if not ids: return False,"Ningún año válido"
        self.comp_ids=ids
        if progress_fn: progress_fn("📥 Descargando datos...",0.05)
        raw=self.collector.download_many(ids,force=force,progress_fn=progress_fn)
        if raw.empty: return False,"Sin datos"
        if progress_fn: progress_fn("🔄 Procesando...",0.50)
        self.df_proc=self.processor.process(raw)
        if progress_fn: progress_fn("⚙️ Features...",0.68)
        self.df_feat=self.engineer.build(self.df_proc)
        if progress_fn: progress_fn("📊 Poisson...",0.82)
        self.poisson.fit(self.df_proc)
        if progress_fn: progress_fn("🤖 ML...",0.92)
        self.ml.fit(self.df_feat)
        self.trained=True
        pl=self.df_proc[self.df_proc["played"]]
        self.stats={"total":len(self.df_proc),"played":int(pl.shape[0]),
            "pending":int((~self.df_proc["played"]).sum()),
            "teams":sorted(set(pl["home_team"])|set(pl["away_team"])),
            "cv_score":self.ml.cv_score,"home_adv":self.poisson.home_adv,
            "seasons":years,"comp_label":comp_label}
        if progress_fn: progress_fn("✅ Listo",1.0)
        return True,"OK"

    def get_teams(self, comp_id=None):
        if self.df_proc is None: return []
        df=self.df_proc[self.df_proc["played"]]
        if comp_id: df=df[df["competition_id"]==comp_id]
        return sorted(set(df["home_team"])|set(df["away_team"]))

    def predict(self, home, away, comp_id=None):
        if not self.trained: return None
        cid=comp_id or self.comp_ids[-1]
        p=self.poisson.predict(home,away)
        f=self.engineer.predict_feat(self.df_proc,home,away,cid)
        m=self.ml.predict(f) or {}
        cH=(p["prob_H"]+m.get("ml_prob_H",p["prob_H"]))/2
        cD=(p["prob_D"]+m.get("ml_prob_D",p["prob_D"]))/2
        cA=(p["prob_A"]+m.get("ml_prob_A",p["prob_A"]))/2
        fp="H" if cH==max(cH,cD,cA) else("D" if cD==max(cH,cD,cA) else "A")
        conf_val=max(cH,cD,cA)
        conf_label="🟢 Alta" if conf_val>0.55 else("🟡 Media" if conf_val>0.42 else "🔴 Baja")
        return {
            "home_team":home,"away_team":away,"competition":cid,
            "poisson_H":p["prob_H"],"poisson_D":p["prob_D"],"poisson_A":p["prob_A"],
            "ml_H":m.get("ml_prob_H"),"ml_D":m.get("ml_prob_D"),"ml_A":m.get("ml_prob_A"),
            "combined_H":round(cH,4),"combined_D":round(cD,4),"combined_A":round(cA,4),
            "final_prediction":fp,"confidence":conf_label,"confidence_val":round(conf_val,4),
            "odds_H":round(1/cH,2) if cH>0 else 99,"odds_D":round(1/cD,2) if cD>0 else 99,
            "odds_A":round(1/cA,2) if cA>0 else 99,
            "expected_score":p["most_likely_score"],
            "over_1_5":p["prob_over_1_5"],"under_1_5":p["prob_under_1_5"],
            "over_2_5":p["prob_over_2_5"],"under_2_5":p["prob_under_2_5"],
            "over_3_5":p["prob_over_3_5"],"under_3_5":p["prob_under_3_5"],
            "btts":p["prob_btts"],"no_btts":p["prob_no_btts"],
            "asian_home":p["asian_handi_home"],"asian_away":p["asian_handi_away"],
            "lambda_home":p["lambda_home"],"lambda_away":p["lambda_away"],
            "score_matrix":p["score_matrix"],
        }

    def predict_pending(self, comp_id=None):
        if self.df_proc is None: return pd.DataFrame()
        cid=comp_id or self.comp_ids[-1]
        pending=self.df_proc[(~self.df_proc["played"])&(self.df_proc["competition_id"]==cid)]
        results=[]
        for _,row in pending.iterrows():
            p=self.predict(row["home_team"],row["away_team"],cid)
            if p:
                p["date"]=str(row["date"])[:10]; p["round"]=str(row["round"])
                results.append(p)
        return pd.DataFrame(results)

    # ── Accesos directos a StatsEngine ──
    def league_table(self, comp_id=None):
        cid=comp_id or self.comp_ids[-1]
        return self.stats_engine.league_table(self.df_proc,cid)

    def team_full_stats(self, team, comp_id=None):
        cid=comp_id or self.comp_ids[-1]
        return self.stats_engine.team_full_stats(self.df_proc,team,cid)

    def h2h(self, home, away, comp_id=None):
        cid=comp_id or self.comp_ids[-1]
        return self.stats_engine.h2h_detail(self.df_proc,home,away,cid)

    def raw_matches(self, comp_id=None, team=None, result=None):
        cid=comp_id or self.comp_ids[-1]
        return self.stats_engine.raw_matches(self.df_proc,cid,team,result)

    def backtest(self, last_n=100):
        return self.ml.backtest(self.df_feat,last_n)
