"""Show what raising the novelty cap buys: best achievable strength + how far the
candidates drift from real alloys + how model uncertainty grows, at several caps."""
import pandas as pd, numpy as np, warnings
warnings.filterwarnings("ignore")
from scipy.spatial.distance import cdist
import xgboost as xgb
D="/mnt/sdb/arafat/Al_alloy_design/"
EL=['Si','Fe','Cu','Mn','Mg','Cr','Zn','V','Ti','Zr','Li','Ni','Be','Sc','Ag','B','Bi','Ga','Pb','Sn']
HT=['Tsol','Tage','tage']
MASS=dict(Al=26.98,Si=28.09,Fe=55.85,Cu=63.55,Mn=54.94,Mg=24.31,Cr=52.0,Zn=65.38,V=50.94,Ti=47.87,Zr=91.22,Li=6.94,Ni=58.69,Be=9.01,Sc=44.96,Ag=107.87,B=10.81,Bi=208.98,Ga=69.72,Pb=207.2,Sn=118.71)
VEC=dict(Al=3,Si=4,Fe=8,Cu=11,Mn=7,Mg=2,Cr=6,Zn=12,V=5,Ti=4,Zr=4,Li=1,Ni=10,Be=2,Sc=3,Ag=11,B=3,Bi=5,Ga=3,Pb=4,Sn=4)
CHI=dict(Al=1.61,Si=1.90,Fe=1.83,Cu=1.90,Mn=1.55,Mg=1.31,Cr=1.66,Zn=1.65,V=1.63,Ti=1.54,Zr=1.33,Li=0.98,Ni=1.91,Be=1.57,Sc=1.36,Ag=1.93,B=2.04,Bi=2.02,Ga=1.81,Pb=2.33,Sn=1.96)
RAD=dict(Al=143,Si=117,Fe=126,Cu=128,Mn=127,Mg=160,Cr=128,Zn=134,V=134,Ti=147,Zr=160,Li=152,Ni=124,Be=112,Sc=162,Ag=144,B=82,Bi=156,Ga=122,Pb=175,Sn=140)
def load(p,ren):
    df=pd.read_excel(p).rename(columns=ren)
    for e in EL+HT+['YS','UTS','EL']:
        if e not in df: df[e]=0 if e in EL else np.nan
    df=df[EL+HT+['YS','UTS','EL']].apply(pd.to_numeric,errors='coerce'); df[EL]=df[EL].fillna(0); df[HT]=df[HT].fillna(0); return df
pool=pd.concat([load(D+"data/hu2024_dataset.xlsx",{'S/A Temp':'Tsol','Ageing Temp':'Tage','Ageing Time':'tage','YTS':'YS'}),
  load(D+"data/cstg_origin1293.xlsx",{'SS Temp':'Tsol','Ageing Temp':'Tage','Ageing Time':'tage','YTS':'YS'}),
  load(D+"data/haoyun_aged_forged.xlsx",{'El':'EL'}),load(D+"data/haoyun_df_cleaned.xlsx",{'El':'EL'})],ignore_index=True).dropna(subset=['UTS'])
sig=pool[EL].round(2).astype(str).agg('|'.join,1)+pool[HT].round(0).astype(str).agg('|'.join,1); pool=pool[~sig.duplicated()].reset_index(drop=True)
def phys(d):
    w=d[EL].clip(lower=0).copy(); w['Al']=(100-w.sum(1)).clip(lower=0); els=EL+['Al']
    x=(w[els]/pd.Series(MASS)).div((w[els]/pd.Series(MASS)).sum(1),axis=0).fillna(0)
    o=pd.DataFrame(index=d.index); o['VEC']=x.mul(pd.Series(VEC)).sum(1); o['mchi']=x.mul(pd.Series(CHI)).sum(1)
    rb=x.mul(pd.Series(RAD)).sum(1); o['delta']=np.sqrt((x*pd.DataFrame({e:(1-RAD[e]/rb)**2 for e in els})).sum(1))*100; return o
def feats(d): return pd.concat([d[EL],d[HT],phys(d)],axis=1)
X=feats(pool); COLS=list(X.columns)
def ens(t):
    m=pool[t].notna(); y=pool.loc[m,t].values
    return [xgb.XGBRegressor(n_estimators=600,max_depth=5,learning_rate=0.03,subsample=0.8,colsample_bytree=0.8,min_child_weight=3,random_state=s).fit(
        X[m].iloc[np.random.RandomState(s).choice(m.sum(),m.sum(),True)],y[np.random.RandomState(s).choice(m.sum(),m.sum(),True)]) for s in range(12)]
E={t:ens(t) for t in ['UTS','YS','EL']}
def predict(d):
    Xf=feats(d).reindex(columns=COLS,fill_value=0); o={}
    for t in E: P=np.column_stack([m.predict(Xf) for m in E[t]]); o[t]=P.mean(1); o[t+'s']=P.std(1)
    return o
DV=['Zn','Mg','Cu','Mn','Cr']+HT
is7=(pool['Zn']>=3)&(pool['Zn']>pool['Cu'])&(pool['Zn']>pool['Si'])&(pool['Zn']>pool['Mg'])
ref=pool.loc[is7,DV].values; MU,SD=ref.mean(0),ref.std(0)+1e-9; RN=(ref-MU)/SD
dd=cdist(RN,RN); np.fill_diagonal(dd,np.inf); nn=dd.min(1)
print(f"7xxx alloys={is7.sum()}  | real-data nearest-neighbour gaps: P50={np.percentile(nn,50):.2f} P90={np.percentile(nn,90):.2f} P95={np.percentile(nn,95):.2f} max={nn.max():.2f}")
rng=np.random.RandomState(7); N=200000
B={'Zn':(4,9.5),'Mg':(1,3.2),'Cu':(0,2.8),'Mn':(0,0.5),'Cr':(0,0.35),'Tsol':(455,495),'Tage':(95,160),'tage':(6,60)}
C=pd.DataFrame({k:rng.uniform(*B[k],N) for k in B}); C['Fe']=0.15; C['Si']=0.10
for e in EL:
    if e not in C: C[e]=0.0
p=predict(C);
for k,v in p.items(): C[k]=v
C['novelty']=cdist((C[DV].values-MU)/SD,RN).min(1); C['score']=C['UTS']-C['UTSs']
C=C[(C['Zn']+C['Mg']+C['Cu']<=12.5)&(C['EL']>=8)]
print("\nNovelty cap | #pass | best UTS (±ens s) | that alloy Zn/Mg/Cu | mean UTS s of top-50 score")
for t in [1.32,1.5,2.0,2.5,3.0]:
    s=C[C['novelty']<=t]
    if not len(s): print(f"  {t:<4}  none"); continue
    b=s.loc[s['UTS'].idxmax()]; top=s.nlargest(50,'score')
    print(f"  {t:<4} | {len(s):>6} | {b['UTS']:5.0f} (±{b['UTSs']:.0f}) | Zn{b['Zn']:.1f} Mg{b['Mg']:.1f} Cu{b['Cu']:.1f} (nov {b['novelty']:.2f}) | {top['UTSs'].mean():.1f} MPa")
