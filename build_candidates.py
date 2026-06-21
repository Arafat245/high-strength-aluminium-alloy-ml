"""Pool 4 datasets -> train trustworthy model -> in-distribution virtual screen ->
5 best cheap (Zn/Mg/Cu/Mn/Cr) 7xxx candidates for Thermo-Calc."""
import pandas as pd, numpy as np, warnings, pickle
warnings.filterwarnings("ignore")
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.metrics import r2_score, mean_absolute_error
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans
import xgboost as xgb
D="/mnt/sdb/arafat/Al_alloy_design/"
EL=['Si','Fe','Cu','Mn','Mg','Cr','Zn','V','Ti','Zr','Li','Ni','Be','Sc','Ag','B','Bi','Ga','Pb','Sn']
HT=['Tsol','Tage','tage']
MASS=dict(Al=26.98,Si=28.09,Fe=55.85,Cu=63.55,Mn=54.94,Mg=24.31,Cr=52.0,Zn=65.38,V=50.94,Ti=47.87,Zr=91.22,Li=6.94,Ni=58.69,Be=9.01,Sc=44.96,Ag=107.87,B=10.81,Bi=208.98,Ga=69.72,Pb=207.2,Sn=118.71)
VEC=dict(Al=3,Si=4,Fe=8,Cu=11,Mn=7,Mg=2,Cr=6,Zn=12,V=5,Ti=4,Zr=4,Li=1,Ni=10,Be=2,Sc=3,Ag=11,B=3,Bi=5,Ga=3,Pb=4,Sn=4)
CHI=dict(Al=1.61,Si=1.90,Fe=1.83,Cu=1.90,Mn=1.55,Mg=1.31,Cr=1.66,Zn=1.65,V=1.63,Ti=1.54,Zr=1.33,Li=0.98,Ni=1.91,Be=1.57,Sc=1.36,Ag=1.93,B=2.04,Bi=2.02,Ga=1.81,Pb=2.33,Sn=1.96)
RAD=dict(Al=143,Si=117,Fe=126,Cu=128,Mn=127,Mg=160,Cr=128,Zn=134,V=134,Ti=147,Zr=160,Li=152,Ni=124,Be=112,Sc=162,Ag=144,B=82,Bi=156,Ga=122,Pb=175,Sn=140)

def load(path,ren):
    df=pd.read_excel(path); df=df.rename(columns=ren)
    for e in EL+HT+['YS','UTS','EL']:
        if e not in df: df[e]=0 if e in EL else np.nan
    df=df[EL+HT+['YS','UTS','EL']].apply(pd.to_numeric,errors='coerce')
    df[EL]=df[EL].fillna(0); df[HT]=df[HT].fillna(0)
    return df

pool=pd.concat([
 load(D+"data/hu2024_dataset.xlsx",{'S/A Temp':'Tsol','Ageing Temp':'Tage','Ageing Time':'tage','YTS':'YS'}),
 load(D+"data/cstg_origin1293.xlsx",{'SS Temp':'Tsol','Ageing Temp':'Tage','Ageing Time':'tage','YTS':'YS'}),
 load(D+"data/haoyun_aged_forged.xlsx",{'El':'EL'}),
 load(D+"data/haoyun_df_cleaned.xlsx",{'El':'EL'}),
],ignore_index=True)
pool=pool.dropna(subset=['UTS'])
sig=pool[EL].round(2).astype(str).agg('|'.join,axis=1)+pool[HT].round(0).astype(str).agg('|'.join,axis=1)
pool=pool[~sig.duplicated()].reset_index(drop=True)
print("pooled, de-duplicated rows:",len(pool),"| with YS:",pool.YS.notna().sum())

def phys(d):
    w=d[EL].clip(lower=0).copy(); w['Al']=(100-w.sum(1)).clip(lower=0); els=EL+['Al']
    x=(w[els]/pd.Series(MASS)).div((w[els]/pd.Series(MASS)).sum(1),axis=0).fillna(0)
    o=pd.DataFrame(index=d.index); o['VEC']=x.mul(pd.Series(VEC)).sum(1); o['mchi']=x.mul(pd.Series(CHI)).sum(1)
    rb=x.mul(pd.Series(RAD)).sum(1); o['delta']=np.sqrt((x*pd.DataFrame({e:(1-RAD[e]/rb)**2 for e in els})).sum(1))*100
    return o
def feats(d): return pd.concat([d[EL],d[HT],phys(d)],axis=1)

X=feats(pool); grp=pool[EL].round(1).astype(str).agg('|'.join,axis=1)
def fit_ensemble(y_col):
    m=pool[y_col].notna(); y=pool.loc[m,y_col].values
    mdl=lambda s: xgb.XGBRegressor(n_estimators=600,max_depth=5,learning_rate=0.03,subsample=0.8,
        colsample_bytree=0.8,min_child_weight=3,random_state=s)
    pg=cross_val_predict(mdl(0),X[m],y,cv=GroupKFold(5),groups=grp[m])
    is7=((pool['Zn']>=3)&(pool['Zn']>pool['Cu'])&(pool['Zn']>pool['Si'])&(pool['Zn']>pool['Mg']))[m].values
    print(f"  {y_col}: grouped-CV R2={r2_score(y,pg):.3f} MAE={mean_absolute_error(y,pg):.1f} | "
          f"7xxx R2={r2_score(y[is7],pg[is7]):.3f} MAE={mean_absolute_error(y[is7],pg[is7]):.1f} (n7={is7.sum()})")
    return [mdl(s).fit(X[m].iloc[np.random.RandomState(s).choice(m.sum(),m.sum(),True)],
                       y[np.random.RandomState(s).choice(m.sum(),m.sum(),True)]) for s in range(12)]
print("Forward models (pooled, honest grouped CV):")
ENS={t:fit_ensemble(t) for t in ['UTS','YS','EL']}
COLS=list(X.columns)

def predict(d):
    Xf=feats(d).reindex(columns=COLS,fill_value=0); o={}
    for t in ['UTS','YS','EL']:
        P=np.column_stack([m.predict(Xf) for m in ENS[t]]); o[t]=P.mean(1); o[t+' s']=P.std(1)
    return o

# ---- inverse design: find NEW alloys (novelty up to CAP) with good (not max) strength ----
DV=['Zn','Mg','Cu','Mn','Cr']+HT
is7=(pool['Zn']>=3)&(pool['Zn']>pool['Cu'])&(pool['Zn']>pool['Si'])&(pool['Zn']>pool['Mg'])
ref=pool.loc[is7,DV].values; MU,SD=ref.mean(0),ref.std(0)+1e-9; RN=(ref-MU)/SD
CAP=1.5      # novelty ceiling (~P95 of real-data gaps): genuinely new but still reliable
FLOOR=520    # UTS floor (MPa): "somewhat good" strength; peak strength NOT required

# wider, still-realistic 7xxx search bounds
rng=np.random.RandomState(7); N=200000
B={'Zn':(4,9.5),'Mg':(1,3.2),'Cu':(0,2.8),'Mn':(0,0.5),'Cr':(0,0.35),'Tsol':(455,495),'Tage':(95,160),'tage':(6,60)}
C=pd.DataFrame({k:rng.uniform(*B[k],N) for k in B}); C['Fe']=0.15; C['Si']=0.10
for e in EL:
    if e not in C: C[e]=0.0
pr=predict(C)
for k,v in pr.items(): C[k]=v
C['novelty']=cdist((C[DV].values-MU)/SD,RN).min(1)
sel=C[(C['novelty']<=CAP)&(C['UTS']>=FLOOR)&(C['Zn']+C['Mg']+C['Cu']<=12.5)&(C['EL']>=8)].copy()
print(f"\n7xxx reference alloys: {is7.sum()} | novelty cap {CAP} | UTS floor {FLOOR} MPa | passing: {len(sel)}")
# 5 DISTINCT, as-NEW-as-possible alloys: cluster the design space, take max-novelty member per cluster
sel['cl']=KMeans(5,n_init=10,random_state=0).fit_predict((sel[DV].values-MU)/SD)
out=sel.sort_values('novelty',ascending=False).groupby('cl',as_index=False).first().sort_values('UTS',ascending=False).reset_index(drop=True)
out['score']=out['UTS']-out['UTS s']
out['Al']=(100-out[EL].sum(1)).round(2)
view=['Al','Zn','Mg','Cu','Mn','Cr','Fe','Si','Tsol','Tage','tage','UTS','UTS s','YS','YS s','EL','novelty']
for c in view: out[c]=out[c].round(2)
pd.set_option("display.width",200)
print("\n=== 5 NEW (novelty-prioritised) CHEAP 7xxx CANDIDATES (pre-Thermo-Calc) ===")
print(out[view].to_string(index=False))
print("\nBenchmarks: 7075-T6 ~572, 7050-T6 ~552, 7055 ~640 MPa.")
out[view+['score']].to_csv(D+"data/candidates_final.csv",index=False)
print("saved -> data/candidates_final.csv")
