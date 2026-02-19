# predictor.py v3-xG — Motor Poisson + GBM + Expected Goals
import requests, json, os, time, warnings
import pandas as pd
import numpy as np
from scipy.stats import poisson
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_score
from catalog_builder import load_or_refresh_catalog
warnings.filterwarnings('ignore')

HEADERS = {'User-Agent':'Mozilla/5.0','Accept':'application/json'}

class DataCollector:
    def __init__(self,cache_dir='./fd_cache',delay=1.2):
        self.cache_dir=cache_dir; self.delay=delay
        os.makedirs(cache_dir,exist_ok=True)
    def _ttl(self,cid):
        try:
            yr=int(cid.split('-')[-1]); cy=__import__('datetime').datetime.now().year
            return 6 if yr>=cy else(12 if yr>=cy-1 else 24)
        except: return 12
    def _stale(self,path,cid):
        if not os.path.exists(path): return True
        return (time.time()-os.path.getmtime(path))/3600>=self._ttl(cid)
    def download_json(self,cid,force=False):
        cache=os.path.join(self.cache_dir,f'{cid}.json')
        if os.path.exists(cache) and not force and not self._stale(cache,cid):
            with open(cache,'r',encoding='utf-8') as f: return json.load(f)
        url=f'https://fixturedownload.com/feed/json/{cid}'
        try:
            r=requests.get(url,headers=HEADERS,timeout=20); r.raise_for_status()
            data=r.json()
            with open(cache,'w',encoding='utf-8') as f: json.dump(data,f,ensure_ascii=False,indent=2)
            time.sleep(self.delay); return data
        except Exception as e:
            print(f'  ERROR {cid}: {e}')
            if os.path.exists(cache):
                with open(cache,'r',encoding='utf-8') as f: return json.load(f)
            return []
    def _to_df(self,data,cid):
        rows=[]
        for m in data:
            rows.append({'competition_id':cid,'round':m.get('RoundNumber',''),
                'date':m.get('DateUtc',''),'home_team':(m.get('HomeTeam') or '').strip(),
                'away_team':(m.get('AwayTeam') or '').strip(),
                'home_score':m.get('HomeTeamScore'),'away_score':m.get('AwayTeamScore'),
                'location':m.get('Location','')})
        return pd.DataFrame(rows)
    def download_many(self,comp_ids,force=False,progress_fn=None):
        frames=[]; total=len(comp_ids)
        for i,cid in enumerate(comp_ids):
            if progress_fn: progress_fn(f'Descargando {cid} ({i+1}/{total})',(i+1)/total*0.45)
            data=self.download_json(cid,force=force)
            if data: frames.append(self._to_df(data,cid))
        return pd.concat(frames,ignore_index=True) if frames else pd.DataFrame()

class DataProcessor:
    def process(self,df):
        df=df.copy()
        df['date']=pd.to_datetime(df['date'],errors='coerce',utc=True)
        df['home_score']=pd.to_numeric(df['home_score'],errors='coerce')
        df['away_score']=pd.to_numeric(df['away_score'],errors='coerce')
        played=df['home_score'].notna()&df['away_score'].notna()
        df.loc[played,'result']=np.where(
            df.loc[played,'home_score']>df.loc[played,'away_score'],'H',
            np.where(df.loc[played,'home_score']<df.loc[played,'away_score'],'A','D'))
        df['played']=played
        df['total_goals']=df['home_score'].fillna(0)+df['away_score'].fillna(0)
        df['home_xg']=None; df['away_xg']=None
        df['xg_diff']=None; df['has_xg']=False
        return df[df['home_team'].str.len()>0].sort_values('date').reset_index(drop=True)

class StatsEngine:
    def league_table(self,df,comp_id):
        pl=df[df['played']&(df['competition_id']==comp_id)].copy()
        if pl.empty: return pd.DataFrame()
        rows=[]
        for team in sorted(set(pl['home_team'])|set(pl['away_team'])):
            hm=pl[pl['home_team']==team]; am=pl[pl['away_team']==team]
            pj=len(hm)+len(am)
            if not pj: continue
            gf=hm['home_score'].sum()+am['away_score'].sum()
            gc=hm['away_score'].sum()+am['home_score'].sum()
            pg=(hm['result']=='H').sum()+(am['result']=='A').sum()
            pe=(hm['result']=='D').sum()+(am['result']=='D').sum()
            pp=pj-pg-pe; pts=pg*3+pe
            comb=pd.concat([
                hm.assign(_r=hm['result'].map({'H':'W','D':'D','A':'L'})),
                am.assign(_r=am['result'].map({'A':'W','D':'D','H':'L'})),
            ]).sort_values('date')
            forma=''.join(comb['_r'].tolist()[-5:])
            rows.append({'Equipo':team,'PJ':pj,'PG':pg,'PE':pe,'PP':pp,
                'GF':int(gf),'GC':int(gc),'GD':int(gf-gc),
                'PTS':int(pts),'PPP':round(pts/pj,2),'Forma':forma})
        return pd.DataFrame(rows).sort_values(['PTS','GD','GF'],ascending=False).reset_index(drop=True)
    def team_full_stats(self,df,team,comp_id=None):
        pl=df[df['played']].copy()
        if comp_id: pl=pl[pl['competition_id']==comp_id]
        hm=pl[pl['home_team']==team]; am=pl[pl['away_team']==team]
        if len(hm)+len(am)==0: return {}
        def calc(matches,side):
            if not len(matches): return {f'{side}_{k}':0 for k in ['PJ','PG','PE','PP','GF','GC','GD','PTS','PPP']}
            gf=matches['home_score' if side=='home' else 'away_score'].sum()
            gc=matches['away_score' if side=='home' else 'home_score'].sum()
            pg=(matches['result']==('H' if side=='home' else 'A')).sum()
            pe=(matches['result']=='D').sum(); pj=len(matches); pp=pj-pg-pe; pts=pg*3+pe
            return {f'{side}_PJ':pj,f'{side}_PG':pg,f'{side}_PE':pe,f'{side}_PP':pp,
                f'{side}_GF':int(gf),f'{side}_GC':int(gc),f'{side}_GD':int(gf-gc),
                f'{side}_PTS':int(pts),f'{side}_PPP':round(pts/pj,2)}
        s={}; s.update(calc(hm,'home')); s.update(calc(am,'away'))
        for k in ['PJ','GF','GC','PG','PE','PP','PTS']:
            s[f'total_{k}']=s[f'home_{k}']+s[f'away_{k}']
        s['total_GD']=s['total_GF']-s['total_GC']
        s['total_PPP']=round(s['total_PTS']/max(1,s['total_PJ']),2)
        s['avg_gf']=round(s['total_GF']/max(1,s['total_PJ']),2)
        s['avg_gc']=round(s['total_GC']/max(1,s['total_PJ']),2)
        s['clean_sheets']=int((hm['away_score']==0).sum()+(am['home_score']==0).sum())
        s['btts_count']=int(((hm['home_score']>0)&(hm['away_score']>0)).sum()+
                            ((am['away_score']>0)&(am['home_score']>0)).sum())
        comb=pd.concat([hm.assign(_r=hm['result'].map({'H':'W','D':'D','A':'L'})),
                        am.assign(_r=am['result'].map({'A':'W','D':'D','H':'L'}))]).sort_values('date')
        sl=comb['_r'].tolist()
        if sl:
            cur=sl[-1]; n=1
            for r in reversed(sl[:-1]):
                if r==cur: n+=1
                else: break
            s['streak']=f'{cur}{n}'
        else: s['streak']='—'
        s['last5']=sl[-5:]
        s['last5_matches']=comb.tail(5)[['date','home_team','away_team',
            'home_score','away_score','result','_r']].to_dict('records')
        # xG stats
        xg_pl = pl[pl['has_xg']==True] if 'has_xg' in pl.columns else pd.DataFrame()
        if not xg_pl.empty:
            xg_hm=xg_pl[xg_pl['home_team']==team]; xg_am=xg_pl[xg_pl['away_team']==team]
            xgf=xg_hm['home_xg'].sum()+xg_am['away_xg'].sum()
            xgc=xg_hm['away_xg'].sum()+xg_am['home_xg'].sum()
            n_xg=len(xg_hm)+len(xg_am)
            s['xGF']=round(float(xgf),2); s['xGA']=round(float(xgc),2)
            s['xGF_per_game']=round(float(xgf)/n_xg,3)
            s['xGA_per_game']=round(float(xgc)/n_xg,3)
            s['xg_overperf_att']=round(s['total_GF']-float(xgf),2)
            s['xg_overperf_def']=round(float(xgc)-s['total_GC'],2)
            s['luck_index']=round(s['xg_overperf_att']+s['xg_overperf_def'],2)
            s['has_xg']=True
        else:
            s['has_xg']=False
        return s
    def h2h_detail(self,df,home,away,comp_id=None):
        pl=df[df['played']].copy()
        if comp_id: pl=pl[pl['competition_id']==comp_id]
        h2h=pl[((pl['home_team']==home)&(pl['away_team']==away))|
               ((pl['home_team']==away)&(pl['away_team']==home))].sort_values('date',ascending=False)
        if h2h.empty: return {'matches':pd.DataFrame(),'summary':{}}
        hw=dd=aw=hgf=agf=0
        for _,m in h2h.iterrows():
            if m['home_team']==home:
                hgf+=m['home_score']; agf+=m['away_score']
                if m['result']=='H': hw+=1
                elif m['result']=='D': dd+=1
                else: aw+=1
            else:
                hgf+=m['away_score']; agf+=m['home_score']
                if m['result']=='A': hw+=1
                elif m['result']=='D': dd+=1
                else: aw+=1
        n=len(h2h)
        return {'matches':h2h,'summary':{'total':n,f'{home}_wins':hw,'draws':dd,
            f'{away}_wins':aw,f'{home}_goals':hgf,f'{away}_goals':agf,
            'avg_goals':round((hgf+agf)/n,2),
            'btts':int(((h2h['home_score']>0)&(h2h['away_score']>0)).sum())}}
    def raw_matches(self,df,comp_id=None,team=None,result=None,limit=500):
        pl=df[df['played']].copy()
        if comp_id: pl=pl[pl['competition_id']==comp_id]
        if team: pl=pl[(pl['home_team']==team)|(pl['away_team']==team)]
        if result: pl=pl[pl['result']==result]
        pl=pl.sort_values('date',ascending=False).head(limit)
        pl['date_fmt']=pl['date'].dt.strftime('%Y-%m-%d')
        pl['score']=pl['home_score'].astype(int).astype(str)+'-'+pl['away_score'].astype(int).astype(str)
        cols=['competition_id','round','date_fmt','home_team','score','away_team','result','location']
        xg_cols=['home_xg','away_xg','xg_diff']
        for c in xg_cols:
            if c in pl.columns: cols.append(c)
        return pl[cols].rename(columns={'competition_id':'Liga','round':'Jornada',
            'date_fmt':'Fecha','home_team':'Local','score':'Marcador',
            'away_team':'Visitante','result':'Res','location':'Estadio',
            'home_xg':'xG-L','away_xg':'xG-V','xg_diff':'xG-Diff'})

class FeatureEngineer:
    def __init__(self,n=5): self.n=n
    def _stats(self,matches,team,pfx):
        z={f'{pfx}gf':0.,f'{pfx}gc':0.,f'{pfx}pts':0.,f'{pfx}wins':0.,
           f'{pfx}draws':0.,f'{pfx}losses':0.,f'{pfx}n':0}
        if not len(matches): return z
        gf=gc=pts=w=d=l=0
        for _,m in matches.iterrows():
            is_h=m['home_team']==team
            gs=m['home_score'] if is_h else m['away_score']
            gc_=m['away_score'] if is_h else m['home_score']
            wr='H' if is_h else 'A'
            if m['result']==wr: w+=1;pts+=3
            elif m['result']=='D': d+=1;pts+=1
            else: l+=1
            gf+=gs; gc+=gc_
        n=len(matches)
        return {f'{pfx}gf':round(gf/n,3),f'{pfx}gc':round(gc/n,3),
            f'{pfx}pts':round(pts/n,3),f'{pfx}wins':round(w/n,3),
            f'{pfx}draws':round(d/n,3),f'{pfx}losses':round(l/n,3),f'{pfx}n':n}
    def _xg_stats(self,matches,team,pfx):
        z={f'{pfx}xgf':0.,f'{pfx}xgc':0.,f'{pfx}xg_diff':0.,f'{pfx}overperf':0.}
        if 'has_xg' not in matches.columns: return z
        m_xg=matches[matches['has_xg']==True]
        if m_xg.empty: return z
        xgf=xgc=gf=gc=0
        for _,m in m_xg.iterrows():
            is_h=m['home_team']==team
            xgf+=(m['home_xg'] if is_h else m['away_xg']) or 0
            xgc+=(m['away_xg'] if is_h else m['home_xg']) or 0
            gf+=m['home_score'] if is_h else m['away_score']
            gc+=m['away_score'] if is_h else m['home_score']
        n=len(m_xg)
        return {f'{pfx}xgf':round(xgf/n,3),f'{pfx}xgc':round(xgc/n,3),
            f'{pfx}xg_diff':round((xgf-xgc)/n,3),f'{pfx}overperf':round((gf-xgf)/n,3)}
    def _h2h(self,h2h,home,away):
        z={'h2h_hw':0.,'h2h_d':0.,'h2h_aw':0.,'h2h_hgf':0.,'h2h_agf':0.,'h2h_n':0}
        if not len(h2h): return z
        hw=dd=aw=hgf=agf=0
        for _,m in h2h.iterrows():
            if m['home_team']==home:
                hgf+=m['home_score']; agf+=m['away_score']
                if m['result']=='H': hw+=1
                elif m['result']=='D': dd+=1
                else: aw+=1
            else:
                agf+=m['home_score']; hgf+=m['away_score']
                if m['result']=='A': hw+=1
                elif m['result']=='D': dd+=1
                else: aw+=1
        n=len(h2h)
        return {'h2h_hw':round(hw/n,3),'h2h_d':round(dd/n,3),'h2h_aw':round(aw/n,3),
            'h2h_hgf':round(hgf/n,3),'h2h_agf':round(agf/n,3),'h2h_n':n}
    def build(self,df):
        dfp=df[df['played']].copy().reset_index(drop=True); feats=[]
        for _,row in dfp.iterrows():
            date,comp,home,away=row['date'],row['competition_id'],row['home_team'],row['away_team']
            hist=dfp[(dfp['date']<date)&(dfp['competition_id']==comp)]
            feat={'competition_id':comp,'date':date,'home_team':home,'away_team':away,
                'home_score':row['home_score'],'away_score':row['away_score'],'result':row['result']}
            hm_all=hist[(hist['home_team']==home)|(hist['away_team']==home)].tail(self.n)
            hm_hom=hist[hist['home_team']==home].tail(self.n)
            am_all=hist[(hist['home_team']==away)|(hist['away_team']==away)].tail(self.n)
            am_aw =hist[hist['away_team']==away].tail(self.n)
            h2h_m =hist[((hist['home_team']==home)&(hist['away_team']==away))|
                        ((hist['home_team']==away)&(hist['away_team']==home))].tail(6)
            feat.update(self._stats(hm_all,home,'h_'))
            feat.update(self._stats(hm_hom,home,'hh_'))
            feat.update(self._stats(am_all,away,'a_'))
            feat.update(self._stats(am_aw, away,'aa_'))
            feat.update(self._h2h(h2h_m,home,away))
            feat.update(self._xg_stats(hm_all,home,'h_'))
            feat.update(self._xg_stats(hm_hom,home,'hh_'))
            feat.update(self._xg_stats(am_all,away,'a_'))
            feat.update(self._xg_stats(am_aw, away,'aa_'))
            feats.append(feat)
        return pd.DataFrame(feats)
    def predict_feat(self,df_proc,home,away,comp):
        hist=df_proc[df_proc['played']&(df_proc['competition_id']==comp)].sort_values('date')
        feat={'competition_id':comp,'home_team':home,'away_team':away}
        for pfx,team,filt in [
            ('h_',home,lambda h:(h['home_team']==home)|(h['away_team']==home)),
            ('hh_',home,lambda h:h['home_team']==home),
            ('a_',away,lambda h:(h['home_team']==away)|(h['away_team']==away)),
            ('aa_',away,lambda h:h['away_team']==away),
        ]:
            matches=hist[filt(hist)].tail(self.n)
            feat.update(self._stats(matches,team,pfx))
            feat.update(self._xg_stats(matches,team,pfx))
        feat.update(self._h2h(hist[((hist['home_team']==home)&(hist['away_team']==away))|
            ((hist['home_team']==away)&(hist['away_team']==home))].tail(6),home,away))
        return pd.DataFrame([feat])

class PoissonModel:
    def __init__(self): self.attack={}; self.defense={}; self.home_adv=1.0
    def fit(self,df,xg_weight=0.5):
        pl=df[df['played']]
        has_xg='home_xg' in pl.columns and pl['has_xg'].any()
        for t in sorted(set(pl['home_team'])|set(pl['away_team'])):
            hm=pl[pl['home_team']==t]; am=pl[pl['away_team']==t]
            g_sc=pd.concat([hm['home_score'],am['away_score']])
            g_co=pd.concat([hm['away_score'],am['home_score']])
            r_att=g_sc.mean() if len(g_sc) else 1.0
            r_def=g_co.mean() if len(g_co) else 1.0
            if has_xg:
                hm_x=hm[hm['has_xg']==True]; am_x=am[am['has_xg']==True]
                xs=pd.concat([hm_x['home_xg'],am_x['away_xg']]).dropna()
                xc=pd.concat([hm_x['away_xg'],am_x['home_xg']]).dropna()
                x_att=xs.mean() if len(xs)>3 else r_att
                x_def=xc.mean() if len(xc)>3 else r_def
                self.attack[t]=max(0.3,r_att*(1-xg_weight)+x_att*xg_weight)
                self.defense[t]=max(0.3,r_def*(1-xg_weight)+x_def*xg_weight)
            else:
                self.attack[t]=max(0.3,r_att); self.defense[t]=max(0.3,r_def)
        ah=pl['home_score'].mean(); aa=pl['away_score'].mean()
        self.home_adv=round(ah/aa,3) if aa else 1.0
    def predict(self,home,away,max_g=8):
        lh=self.attack.get(home,1.2)*self.home_adv/max(0.1,self.defense.get(away,1.0))
        la=self.attack.get(away,1.0)/max(0.1,self.defense.get(home,1.2))
        M=np.outer([poisson.pmf(i,lh) for i in range(max_g+1)],
                   [poisson.pmf(j,la) for j in range(max_g+1)])
        pH=float(np.sum(np.tril(M,-1))); pD=float(np.sum(np.diag(M))); pA=float(np.sum(np.triu(M,1)))
        idx=np.unravel_index(np.argmax(M),M.shape)
        def s(fn): return sum(M[i][j] for i in range(max_g+1) for j in range(max_g+1) if fn(i,j))
        return {'prob_H':round(pH,4),'prob_D':round(pD,4),'prob_A':round(pA,4),
            'lambda_home':round(lh,3),'lambda_away':round(la,3),
            'most_likely_score':f'{idx[0]}-{idx[1]}',
            'prob_over_1_5':round(s(lambda i,j:i+j>1),4),
            'prob_over_2_5':round(s(lambda i,j:i+j>2),4),
            'prob_over_3_5':round(s(lambda i,j:i+j>3),4),
            'prob_btts':round(s(lambda i,j:i>0 and j>0),4),
            'asian_handi_home':round(s(lambda i,j:(i-0.5)>j),4),
            'asian_handi_away':round(s(lambda i,j:(j-0.5)>i),4),
            'score_matrix':M.tolist()}

class MLModel:
    FCOLS=['h_gf','h_gc','h_pts','h_wins','h_draws','h_losses',
           'hh_gf','hh_gc','hh_pts','a_gf','a_gc','a_pts','a_wins','a_draws','a_losses',
           'aa_gf','aa_gc','aa_pts','h2h_hw','h2h_d','h2h_aw','h2h_hgf','h2h_agf',
           'h_xgf','h_xgc','h_xg_diff','h_overperf',
           'hh_xgf','hh_xgc',
           'a_xgf','a_xgc','a_xg_diff','a_overperf',
           'aa_xgf','aa_xgc']
    def __init__(self):
        self.model=GradientBoostingClassifier(n_estimators=300,max_depth=4,
            learning_rate=0.05,subsample=0.8,random_state=42)
        self.le=LabelEncoder(); self.trained=False; self.cv_score=0.0
    def _X(self,df):
        for c in self.FCOLS:
            if c not in df.columns: df[c]=0.0
        return df[self.FCOLS].fillna(0.0)
    def fit(self,df_feat):
        df=df_feat.dropna(subset=['result'])
        if len(df)<30: return
        X=self._X(df.copy()); y=self.le.fit_transform(df['result'])
        cv=cross_val_score(self.model,X,y,cv=min(5,len(df)//20),scoring='accuracy')
        self.cv_score=round(float(cv.mean()),4)
        self.model.fit(X,y); self.trained=True
    def predict(self,df_feat):
        if not self.trained: return None
        X=self._X(df_feat.copy()); proba=self.model.predict_proba(X)[0]
        out={f'ml_prob_{c}':round(float(p),4) for c,p in zip(self.le.classes_,proba)}
        out['ml_prediction']=self.le.classes_[np.argmax(proba)]
        return out
    def backtest(self,df_feat,last_n=100):
        df=df_feat.dropna(subset=['result']).tail(last_n)
        if not self.trained or len(df)<10: return pd.DataFrame()
        X=self._X(df.copy()); proba=self.model.predict_proba(X)
        classes=self.le.classes_; rows=[]
        for i,(_,row) in enumerate(df.iterrows()):
            pred=classes[np.argmax(proba[i])]
            rows.append({'date':str(row.get('date',''))[:10],
                'home':row.get('home_team',''),'away':row.get('away_team',''),
                'real':row['result'],'pred':pred,'correct':row['result']==pred,
                'conf':round(float(proba[i][np.argmax(proba[i])]),3)})
        return pd.DataFrame(rows)
    def importances(self):
        if not self.trained: return {}
        return dict(sorted(zip(self.FCOLS,self.model.feature_importances_),key=lambda x:-x[1]))

class FootballSystem:
    def __init__(self):
        self.collector=DataCollector(); self.processor=DataProcessor()
        self.engineer=FeatureEngineer(); self.poisson=PoissonModel()
        self.ml=MLModel(); self.stats_engine=StatsEngine()
        self.df_proc=None; self.df_feat=None
        self.comp_ids=[]; self.trained=False; self.stats={}
        self.predict_comp_id=''; self.catalog=None; self.has_xg=False
    def load_catalog(self,force=False,progress_fn=None):
        self.catalog=load_or_refresh_catalog(force=force,progress_fn=progress_fn)
        return self.catalog
    def get_catalog(self):
        if self.catalog is None: self.catalog=load_or_refresh_catalog(fast_mode=True)
        return self.catalog
    def train(self,comp_slug,train_seasons,predict_season,force=False,progress_fn=None):
        train_ids=[f'{comp_slug}-{y}' for y in train_seasons]
        pred_id  =f'{comp_slug}-{predict_season}'
        all_ids  =train_ids+([pred_id] if pred_id not in train_ids else [])
        all_years=train_seasons+([predict_season] if predict_season not in train_seasons else [])
        self.comp_ids=all_ids; self.predict_comp_id=pred_id
        if progress_fn: progress_fn('Descargando datos FD...',0.05)
        raw=self.collector.download_many(all_ids,force=force,progress_fn=progress_fn)
        if raw.empty: return False,'Sin datos'
        if progress_fn: progress_fn('Procesando partidos...',0.48)
        self.df_proc=self.processor.process(raw)
        # ── xG integration ──────────────────────────────────────────────
        self.has_xg=False
        try:
            from xg_scraper import XGScraper, FD_TO_UNDERSTAT
            if comp_slug in FD_TO_UNDERSTAT:
                if progress_fn: progress_fn('Descargando xG de Understat...',0.55)
                xg_sc  =XGScraper()
                xg_df  =xg_sc.fetch_many_seasons(comp_slug,all_years,force=force)
                if xg_df is not None:
                    self.df_proc=xg_sc.merge_xg_into_results(self.df_proc,xg_df,comp_slug)
                    self.has_xg=True
        except Exception as e:
            print(f'  xG skip: {e}')
        # ── Features ────────────────────────────────────────────────────
        if progress_fn: progress_fn('Calculando features...',0.68)
        df_train=self.df_proc[self.df_proc['competition_id'].isin(train_ids)]
        self.df_feat=self.engineer.build(df_train)
        if progress_fn: progress_fn('Entrenando Poisson...',0.82)
        self.poisson.fit(df_train)
        if progress_fn: progress_fn('Entrenando GBM...',0.92)
        self.ml.fit(self.df_feat)
        self.trained=True
        pl_all =self.df_proc[self.df_proc['played']]
        pl_pred=self.df_proc[(self.df_proc['competition_id']==pred_id)&self.df_proc['played']]
        self.stats={
            'comp_slug':comp_slug,'predict_id':pred_id,
            'train_seasons':train_seasons,'predict_season':predict_season,
            'total_matches':len(self.df_proc),'played':int(pl_all.shape[0]),
            'pending':int((~self.df_proc[self.df_proc['competition_id']==pred_id]['played']).sum()),
            'played_current':int(pl_pred.shape[0]),
            'teams':sorted(set(pl_all['home_team'])|set(pl_all['away_team'])),
            'cv_score':self.ml.cv_score,'home_adv':self.poisson.home_adv,
            'has_xg':self.has_xg,
        }
        if progress_fn: progress_fn('Sistema listo',1.0)
        return True,'OK'
    def get_teams(self,comp_id=None):
        if self.df_proc is None: return []
        cid=comp_id or self.predict_comp_id
        df=self.df_proc[self.df_proc['played']&(self.df_proc['competition_id']==cid)]
        return sorted(set(df['home_team'])|set(df['away_team']))
    def predict(self,home,away):
        if not self.trained: return None
        cid=self.predict_comp_id
        p=self.poisson.predict(home,away)
        f=self.engineer.predict_feat(self.df_proc,home,away,cid)
        m=self.ml.predict(f) or {}
        cH=(p['prob_H']+m.get('ml_prob_H',p['prob_H']))/2
        cD=(p['prob_D']+m.get('ml_prob_D',p['prob_D']))/2
        cA=(p['prob_A']+m.get('ml_prob_A',p['prob_A']))/2
        fp='H' if cH==max(cH,cD,cA) else('D' if cD==max(cH,cD,cA) else 'A')
        cv=max(cH,cD,cA)
        cl='Alta' if cv>0.55 else('Media' if cv>0.42 else 'Baja')
        # xG del partido si disponible
        xg_info={}
        if self.has_xg:
            hs=self.stats_engine.team_full_stats(self.df_proc,home,cid)
            as_=self.stats_engine.team_full_stats(self.df_proc,away,cid)
            if hs.get('has_xg'):
                xg_info['home_xgF_pg']=hs.get('xGF_per_game',0)
                xg_info['home_xgA_pg']=hs.get('xGA_per_game',0)
                xg_info['home_luck'] =hs.get('luck_index',0)
                xg_info['home_overperf_att']=hs.get('xg_overperf_att',0)
            if as_ and as_.get('has_xg'):
                xg_info['away_xgF_pg']=as_.get('xGF_per_game',0)
                xg_info['away_xgA_pg']=as_.get('xGA_per_game',0)
                xg_info['away_luck'] =as_.get('luck_index',0)
                xg_info['away_overperf_att']=as_.get('xg_overperf_att',0)
        return {'home_team':home,'away_team':away,
            'poisson_H':p['prob_H'],'poisson_D':p['prob_D'],'poisson_A':p['prob_A'],
            'ml_H':m.get('ml_prob_H'),'ml_D':m.get('ml_prob_D'),'ml_A':m.get('ml_prob_A'),
            'combined_H':round(cH,4),'combined_D':round(cD,4),'combined_A':round(cA,4),
            'final_prediction':fp,'confidence':cl,'confidence_val':round(cv,4),
            'odds_H':round(1/cH,2) if cH>0 else 99,'odds_D':round(1/cD,2) if cD>0 else 99,
            'odds_A':round(1/cA,2) if cA>0 else 99,
            'expected_score':p['most_likely_score'],
            'over_1_5':p['prob_over_1_5'],'under_1_5':round(1-p['prob_over_1_5'],4),
            'over_2_5':p['prob_over_2_5'],'under_2_5':round(1-p['prob_over_2_5'],4),
            'over_3_5':p['prob_over_3_5'],'under_3_5':round(1-p['prob_over_3_5'],4),
            'btts':p['prob_btts'],'no_btts':round(1-p['prob_btts'],4),
            'asian_home':p['asian_handi_home'],'asian_away':p['asian_handi_away'],
            'lambda_home':p['lambda_home'],'lambda_away':p['lambda_away'],
            'score_matrix':p['score_matrix'],
            'xg_data':xg_info,'has_xg':self.has_xg}
    def predict_pending(self):
        if self.df_proc is None: return pd.DataFrame()
        cid=self.predict_comp_id
        pending=self.df_proc[(~self.df_proc['played'])&(self.df_proc['competition_id']==cid)]
        results=[]
        for _,row in pending.iterrows():
            p=self.predict(row['home_team'],row['away_team'])
            if p: p['date']=str(row['date'])[:10]; p['round']=str(row['round']); results.append(p)
        return pd.DataFrame(results)
    def league_table(self): return self.stats_engine.league_table(self.df_proc,self.predict_comp_id)
    def team_full_stats(self,team): return self.stats_engine.team_full_stats(self.df_proc,team,self.predict_comp_id)
    def h2h(self,home,away): return self.stats_engine.h2h_detail(self.df_proc,home,away)
    def raw_matches(self,comp_id=None,team=None,result=None):
        return self.stats_engine.raw_matches(self.df_proc,comp_id or self.predict_comp_id,team,result)
    def backtest(self,n=100): return self.ml.backtest(self.df_feat,n)
    def xg_league_table(self):
        from xg_scraper import xg_league_table
        if self.df_proc is None or not self.has_xg: return pd.DataFrame()
        return xg_league_table(self.df_proc,self.predict_comp_id)
    def xg_team_summary(self,team):
        from xg_scraper import xg_team_summary
        if self.df_proc is None or not self.has_xg: return {}
        return xg_team_summary(self.df_proc,team,self.predict_comp_id)