# auto_updater.py
import threading, time, json, os
from datetime import datetime, timezone, timedelta

LOG_FILE = './fd_cache/updater_log.json'

def _now(): return datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')

def load_log():
    if os.path.exists(LOG_FILE):
        try:
            with open(LOG_FILE,'r') as f: return json.load(f)
        except: pass
    return {'last_check':'Nunca','next_check':'—','total_new_matches':0,'checks':0}

def save_log(data):
    os.makedirs('./fd_cache',exist_ok=True)
    with open(LOG_FILE,'w') as f: json.dump(data,f,indent=2)

def get_ttl_hours(comp_ids):
    import datetime as dt
    cy = dt.datetime.now().year
    has_current = any(str(cy) in cid or str(cy+1) in cid for cid in comp_ids)
    return 6 if has_current else 12

class AutoUpdater:
    def __init__(self, system, comp_ids):
        self.system   = system
        self.comp_ids = comp_ids
        self._thread  = None
        self._running = False
        self._log     = load_log()

    def check_all(self):
        """Descarga datos frescos y detecta nuevos resultados."""
        try:
            old_played = 0
            if self.system.df_proc is not None:
                old_played = int(self.system.df_proc['played'].sum())
            new_data = self.system.collector.download_many(self.comp_ids, force=True)
            if new_data.empty:
                return {'new_matches':0,'error':'Sin datos'}
            new_proc = self.system.processor.process(new_data)
            new_played = int(new_proc['played'].sum())
            diff = max(0, new_played - old_played)
            if diff > 0:
                self.system.df_proc = new_proc
                df_train = new_proc[new_proc['competition_id'].isin(
                    [c for c in self.comp_ids if c != self.system.predict_comp_id]
                )]
                self.system.df_feat = self.system.engineer.build(df_train)
                self.system.poisson.fit(df_train)
                self.system.ml.fit(self.system.df_feat)
            self._log['last_check'] = _now()
            self._log['checks']     = self._log.get('checks',0) + 1
            self._log['total_new_matches'] = self._log.get('total_new_matches',0) + diff
            save_log(self._log)
            return {'new_matches': diff, 'total_played': new_played}
        except Exception as e:
            return {'new_matches':0,'error':str(e)}

    def force_update(self):
        return self.check_all()

    def _loop(self, interval_min):
        while self._running:
            next_dt = datetime.now(timezone.utc) + timedelta(minutes=interval_min)
            self._log['next_check'] = next_dt.strftime('%Y-%m-%d %H:%M UTC')
            save_log(self._log)
            time.sleep(interval_min * 60)
            if self._running:
                self.check_all()

    def start_background(self, interval_min=30):
        if self._thread and self._thread.is_alive(): return
        self._running = True
        self._thread  = threading.Thread(target=self._loop,args=(interval_min,),daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False