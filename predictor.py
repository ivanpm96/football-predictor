
import requests, json, os, time, warnings
import pandas as pd
import numpy as np
from scipy.stats import poisson
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_score
import joblib

warnings.filterwarnings("ignore")

COMPETITION_MAP = {
    "Premier League":   ("epl",           "Premier League"),
    "La Liga":          ("la-liga",        "La Liga"),
    "Bundesliga":       ("bundesliga",     "Bundesliga"),
    "Serie A":          ("serie-a",        "Serie A"),
    "Ligue 1":          ("ligue-1",        "Ligue 1"),
    "Champions League": ("uefa-champions-league", "Champions League"),
    "Europa League":    ("uefa-europa-league",    "Europa League"),
}

HEADERS = {"User-Agent": "Mozilla/5.0", "Accept": "application/json"}

# ─── DESCARGA ───────────────────────────────────────────────────────────
class DataCollector:
    def __init__(self, cache_dir="./fd_cache", delay=1.5):
        self.cache_dir = cache_dir
        self.delay = delay
        os.makedirs(cache_dir, exist_ok=True)

    def build_ids(self, comp_label, years):
        base = COMPETITION_MAP[comp_label][0]
        return [f"{base}-{y}" for y in years]

    def download_json(self, competition_id, force=False):
        cache = os.path.join(self.cache_dir, f"{competition_id}.json")
        if os.path.exists(cache) and not force:
            with open(cache, "r", encoding="utf-8") as f:
                return json.load(f)
        url = f"https://fixturedownload.com/feed/json/{competition_id}"
        try:
            r = requests.get(url, headers=HEADERS, timeout=20)
            r.raise_for_status()
            data = r.json()
            with open(cache, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            time.sleep(self.delay)
            return data
        except Exception as e:
            print(f"  ❌ Error {competition_id}: {e}")
            return []

    def _to_df(self, data, cid):
        rows = []
        for m in data:
            rows.append({
                "competition_id": cid,
                "round":      m.get("RoundNumber", ""),
                "date":       m.get("DateUtc", ""),
                "home_team":  m.get("HomeTeam", "").strip(),
                "away_team":  m.get("AwayTeam", "").strip(),
                "home_score": m.get("HomeTeamScore"),
                "away_score": m.get("AwayTeamScore"),
                "location":   m.get("Location", ""),
            })
        return pd.DataFrame(rows)

    def download_many(self, comp_ids, force=False, progress_fn=None):
        frames = []
        for i, cid in enumerate(comp_ids):
            if progress_fn:
                progress_fn(f"Descargando {cid}...", i / len(comp_ids))
            data = self.download_json(cid, force=force)
            if data:
                frames.append(self._to_df(data, cid))
        if progress_fn:
            progress_fn("Descarga completa", 1.0)
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


# ─── PROCESO ────────────────────────────────────────────────────────────
class DataProcessor:
    def process(self, df):
        df = df.copy()
        df["date"] = pd.to_datetime(df["date"], errors="coerce", utc=True)
        df["home_score"] = pd.to_numeric(df["home_score"], errors="coerce")
        df["away_score"] = pd.to_numeric(df["away_score"], errors="coerce")
        played = df["home_score"].notna() & df["away_score"].notna()
        df.loc[played, "result"] = np.where(
            df.loc[played, "home_score"] > df.loc[played, "away_score"], "H",
            np.where(df.loc[played, "home_score"] < df.loc[played, "away_score"], "A", "D"),
        )
        df["played"] = played
        return df.sort_values("date").reset_index(drop=True)


# ─── FEATURES ───────────────────────────────────────────────────────────
class FeatureEngineer:
    def __init__(self, n=5):
        self.n = n

    def _stats(self, matches, team, pfx):
        z = {f"{pfx}gf":0.0,f"{pfx}gc":0.0,f"{pfx}pts":0.0,
             f"{pfx}wins":0.0,f"{pfx}draws":0.0,f"{pfx}losses":0.0,f"{pfx}n":0}
        if len(matches) == 0:
            return z
        gf=gc=pts=w=d=l=0
        for _, m in matches.iterrows():
            if m["home_team"] == team:
                gs,gc_=m["home_score"],m["away_score"]
                if m["result"]=="H": w+=1;pts+=3
                elif m["result"]=="D": d+=1;pts+=1
                else: l+=1
            else:
                gs,gc_=m["away_score"],m["home_score"]
                if m["result"]=="A": w+=1;pts+=3
                elif m["result"]=="D": d+=1;pts+=1
                else: l+=1
            gf+=gs; gc+=gc_
        n=len(matches)
        return {f"{pfx}gf":round(gf/n,3),f"{pfx}gc":round(gc/n,3),
                f"{pfx}pts":round(pts/n,3),f"{pfx}wins":round(w/n,3),
                f"{pfx}draws":round(d/n,3),f"{pfx}losses":round(l/n,3),f"{pfx}n":n}

    def _h2h(self, h2h, home, away):
        if len(h2h)==0:
            return {"h2h_home_wins":0.0,"h2h_draws":0.0,"h2h_away_wins":0.0,
                    "h2h_home_gf":0.0,"h2h_away_gf":0.0,"h2h_n":0}
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
        return {"h2h_home_wins":round(hw/n,3),"h2h_draws":round(dd/n,3),
                "h2h_away_wins":round(aw/n,3),"h2h_home_gf":round(hgf/n,3),
                "h2h_away_gf":round(agf/n,3),"h2h_n":n}

    def build(self, df):
        dfp=df[df["played"]].copy().reset_index(drop=True)
        feats=[]
        for _,row in dfp.iterrows():
            date,comp,home,away = row["date"],row["competition_id"],row["home_team"],row["away_team"]
            hist=dfp[(dfp["date"]<date)&(dfp["competition_id"]==comp)]
            feat={"competition_id":comp,"date":date,"home_team":home,"away_team":away,
                  "home_score":row["home_score"],"away_score":row["away_score"],"result":row["result"]}
            feat.update(self._stats(hist[(hist["home_team"]==home)|(hist["away_team"]==home)].tail(self.n),home,"h_"))
            feat.update(self._stats(hist[hist["home_team"]==home].tail(self.n),home,"h_home_"))
            feat.update(self._stats(hist[(hist["home_team"]==away)|(hist["away_team"]==away)].tail(self.n),away,"a_"))
            feat.update(self._stats(hist[hist["away_team"]==away].tail(self.n),away,"a_away_"))
            feat.update(self._h2h(hist[((hist["home_team"]==home)&(hist["away_team"]==away))|
                                       ((hist["home_team"]==away)&(hist["away_team"]==home))].tail(6),home,away))
            feats.append(feat)
        return pd.DataFrame(feats)

    def predict_feat(self, df_proc, home, away, comp):
        hist=df_proc[df_proc["played"]&(df_proc["competition_id"]==comp)].sort_values("date")
        feat={"competition_id":comp,"home_team":home,"away_team":away}
        for pfx,team,filt in [
            ("h_",home,lambda h:(h["home_team"]==home)|(h["away_team"]==home)),
            ("h_home_",home,lambda h:h["home_team"]==home),
            ("a_",away,lambda h:(h["home_team"]==away)|(h["away_team"]==away)),
            ("a_away_",away,lambda h:h["away_team"]==away),
        ]:
            feat.update(self._stats(hist[filt(hist)].tail(self.n),team,pfx))
        feat.update(self._h2h(hist[((hist["home_team"]==home)&(hist["away_team"]==away))|
                                   ((hist["home_team"]==away)&(hist["away_team"]==home))].tail(6),home,away))
        return pd.DataFrame([feat])


# ─── POISSON ────────────────────────────────────────────────────────────
class PoissonModel:
    def __init__(self):
        self.attack={}; self.defense={}; self.home_adv=1.0

    def fit(self, df_proc):
        pl=df_proc[df_proc["played"]]
        teams=sorted(set(pl["home_team"])|set(pl["away_team"]))
        for t in teams:
            scored=pd.concat([pl[pl["home_team"]==t]["home_score"],pl[pl["away_team"]==t]["away_score"]])
            conceded=pd.concat([pl[pl["home_team"]==t]["away_score"],pl[pl["away_team"]==t]["home_score"]])
            self.attack[t]=max(0.3,scored.mean()) if len(scored) else 1.0
            self.defense[t]=max(0.3,conceded.mean()) if len(conceded) else 1.0
        avg_h=pl["home_score"].mean(); avg_a=pl["away_score"].mean()
        self.home_adv=avg_h/avg_a if avg_a else 1.0

    def predict(self, home, away, max_g=7):
        lh=self.attack.get(home,1.2)*self.home_adv/max(0.1,self.defense.get(away,1.0))
        la=self.attack.get(away,1.0)/max(0.1,self.defense.get(home,1.2))
        M=np.outer([poisson.pmf(i,lh) for i in range(max_g+1)],
                   [poisson.pmf(j,la) for j in range(max_g+1)])
        pH=float(np.sum(np.tril(M,-1))); pD=float(np.sum(np.diag(M))); pA=float(np.sum(np.triu(M,1)))
        idx=np.unravel_index(np.argmax(M),M.shape)
        over=sum(M[i][j] for i in range(max_g+1) for j in range(max_g+1) if i+j>2)
        return {"prob_H":round(pH,4),"prob_D":round(pD,4),"prob_A":round(pA,4),
                "lambda_home":round(lh,3),"lambda_away":round(la,3),
                "most_likely_score":f"{idx[0]}-{idx[1]}",
                "prob_over_2_5":round(float(over),4),"prob_under_2_5":round(1-float(over),4),
                "score_matrix":M.tolist()}


# ─── ML ─────────────────────────────────────────────────────────────────
class MLModel:
    FCOLS=["h_gf","h_gc","h_pts","h_wins","h_draws","h_losses",
           "h_home_gf","h_home_gc","h_home_pts",
           "a_gf","a_gc","a_pts","a_wins","a_draws","a_losses",
           "a_away_gf","a_away_gc","a_away_pts",
           "h2h_home_wins","h2h_draws","h2h_away_wins","h2h_home_gf","h2h_away_gf"]
    def __init__(self):
        self.model=GradientBoostingClassifier(n_estimators=200,max_depth=4,learning_rate=0.05,random_state=42)
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

    def feature_importance(self):
        if not self.trained: return {}
        return dict(sorted(zip(self.FCOLS,self.model.feature_importances_),key=lambda x:-x[1]))


# ─── SISTEMA CENTRAL ────────────────────────────────────────────────────
class FootballSystem:
    def __init__(self):
        self.collector=DataCollector()
        self.processor=DataProcessor()
        self.engineer=FeatureEngineer()
        self.poisson=PoissonModel()
        self.ml=MLModel()
        self.df_proc=None
        self.df_feat=None
        self.comp_ids=[]
        self.trained=False
        self.stats={}

    def train(self, comp_label, years, force=False, progress_fn=None):
        ids=self.collector.build_ids(comp_label,years)
        self.comp_ids=ids
        if progress_fn: progress_fn("Descargando datos...",0.1)
        raw=self.collector.download_many(ids,force=force,progress_fn=progress_fn)
        if raw.empty: return False,"Sin datos descargados"
        if progress_fn: progress_fn("Procesando datos...",0.5)
        self.df_proc=self.processor.process(raw)
        if progress_fn: progress_fn("Generando features...",0.7)
        self.df_feat=self.engineer.build(self.df_proc)
        if progress_fn: progress_fn("Entrenando Poisson...",0.85)
        self.poisson.fit(self.df_proc)
        if progress_fn: progress_fn("Entrenando ML...",0.9)
        self.ml.fit(self.df_feat)
        self.trained=True
        played=self.df_proc[self.df_proc["played"]]
        self.stats={
            "total_matches":len(self.df_proc),
            "played":int(played.shape[0]),
            "pending":int((~self.df_proc["played"]).sum()),
            "teams":sorted(set(played["home_team"])|set(played["away_team"])),
            "cv_score":self.ml.cv_score,
            "home_adv":round(self.poisson.home_adv,3),
            "seasons":years,
        }
        if progress_fn: progress_fn("¡Entrenamiento completo!",1.0)
        return True,"OK"

    def get_teams(self, comp_id=None):
        if self.df_proc is None: return []
        df=self.df_proc[self.df_proc["played"]]
        if comp_id:
            df=df[df["competition_id"]==comp_id]
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
        pred="H" if cH==max(cH,cD,cA) else ("D" if cD==max(cH,cD,cA) else "A")
        return {
            "home_team":home,"away_team":away,"competition":cid,
            "poisson_H":p["prob_H"],"poisson_D":p["prob_D"],"poisson_A":p["prob_A"],
            "ml_H":m.get("ml_prob_H"),"ml_D":m.get("ml_prob_D"),"ml_A":m.get("ml_prob_A"),
            "combined_H":round(cH,4),"combined_D":round(cD,4),"combined_A":round(cA,4),
            "final_prediction":pred,
            "expected_score":p["most_likely_score"],
            "over_2_5":p["prob_over_2_5"],"under_2_5":p["prob_under_2_5"],
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
                p["date"]=str(row["date"]); p["round"]=str(row["round"])
                results.append(p)
        return pd.DataFrame(results)
