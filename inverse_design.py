"""Inverse design: NSGA-II over 7xxx (Al-Zn-Mg-Cu) composition + T6 ageing schedule,
maximizing predicted UTS & YTS (ensemble mean) under a manufacturability constraint.
Ranks the Pareto front by strength, model confidence (ensemble std) and novelty,
and writes 5 candidates for Thermo-Calc screening."""
import pandas as pd, numpy as np, pickle, warnings
warnings.filterwarnings("ignore")
from pymoo.core.problem import Problem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
D="/mnt/sdb/arafat/Al_alloy_design/"
S=pickle.load(open(D+"models.pkl","rb")); MODELS,COLS,COMP,HT=S['models'],S['columns'],S['COMP'],S['HT']
MASS=dict(Al=26.98,Si=28.09,Fe=55.85,Cu=63.55,Mn=54.94,Mg=24.31,Cr=52.0,Zn=65.38,V=50.94,Ti=47.87,Zr=91.22,Li=6.94,Ni=58.69,Ga=69.72,Bi=208.98,Pb=207.2,B=10.81,Be=9.01,Sc=44.96)
VEC=dict(Al=3,Si=4,Fe=8,Cu=11,Mn=7,Mg=2,Cr=6,Zn=12,V=5,Ti=4,Zr=4,Li=1,Ni=10,Ga=3,Bi=5,Pb=4,B=3,Be=2,Sc=3)
CHI=dict(Al=1.61,Si=1.90,Fe=1.83,Cu=1.90,Mn=1.55,Mg=1.31,Cr=1.66,Zn=1.65,V=1.63,Ti=1.54,Zr=1.33,Li=0.98,Ni=1.91,Ga=1.81,Bi=2.02,Pb=2.33,B=2.04,Be=1.57,Sc=1.36)
RAD=dict(Al=143,Si=117,Fe=126,Cu=128,Mn=127,Mg=160,Cr=128,Zn=134,V=134,Ti=147,Zr=160,Li=152,Ni=124,Ga=122,Bi=156,Pb=175,B=82,Be=112,Sc=162)
CATFIX=dict(ManufacturingMethod='Extruded',HeatTreatmentMedium='Water',StrainHardeningMethod='No',
            AgeingType='Artificial',TreatmentAfterward='No',ProductionShape='Rod')
FREE=['Zn','Mg','Cu','Zr','Mn','Cr','Ti','Sc']
LO=np.array([4.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0, 460,100, 8])
HI=np.array([9.5,3.0,2.6,0.20,0.30,0.25,0.10,0.30, 490,150,48])
FIXED={'Fe':0.12,'Si':0.08,'V':0,'Li':0,'Ni':0,'Ga':0,'Bi':0,'Pb':0,'B':0,'Be':0}

def make_df(X):
    """rows of [8 free comp, S/A Temp, Ageing Temp, Ageing Time] -> raw feature DataFrame."""
    n=len(X); d=pd.DataFrame(index=range(n))
    for i,e in enumerate(FREE): d[e]=X[:,i]
    for e,v in FIXED.items(): d[e]=v
    d['S/A Temp']=X[:,8]; d['Ageing Temp']=X[:,9]; d['Ageing Time']=X[:,10]
    for k,v in CATFIX.items(): d[k]=v
    return d

def featurize(d):
    w=d[COMP].clip(lower=0).copy(); w['Al']=(100-w.sum(axis=1)).clip(lower=0)
    els=COMP+['Al']; moles=w[els]/pd.Series(MASS); x=moles.div(moles.sum(axis=1),axis=0).fillna(0)
    phys=pd.DataFrame(index=d.index)
    phys['VEC']=x.mul(pd.Series(VEC)).sum(axis=1); phys['mean_chi']=x.mul(pd.Series(CHI)).sum(axis=1)
    rbar=x.mul(pd.Series(RAD)).sum(axis=1)
    phys['delta']=np.sqrt((x*pd.DataFrame({e:(1-RAD[e]/rbar)**2 for e in els})).sum(axis=1))*100
    cats=pd.get_dummies(d[list(CATFIX)].astype(str),prefix=list(CATFIX))
    X=pd.concat([d[COMP],d[HT],phys,cats],axis=1).reindex(columns=COLS,fill_value=0)
    return X

def predict(d):
    X=featurize(d); out={}
    for t in ['UTS','YTS','EL']:
        P=np.column_stack([m.predict(X) for m in MODELS[t]])
        out[t+'_mean']=P.mean(1); out[t+'_std']=P.std(1)
    return out

# training 7xxx reference (for novelty), computed once
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans
tr=pd.read_excel(D+"data/hu2024_dataset.xlsx")
is7=(tr['Zn']>=3)&(tr['Zn']>tr['Cu'])&(tr['Zn']>tr['Si'])&(tr['Zn']>tr['Mg'])
ref=tr.loc[is7,FREE+HT].fillna(0).values
MU,SD=ref.mean(0),ref.std(0)+1e-9; REFN=(ref-MU)/SD
def novelty(d): return cdist((d[FREE+HT].values-MU)/SD,REFN).min(1)

class Design(Problem):
    # objectives: maximize UTS, maximize novelty, minimize prediction uncertainty
    def __init__(self): super().__init__(n_var=11,n_obj=3,n_constr=1,xl=LO,xu=HI)
    def _evaluate(self,X,out,*a,**k):
        d=make_df(X); p=predict(d)
        out["F"]=np.column_stack([-p['UTS_mean'], -novelty(d), p['UTS_std']])
        out["G"]=(X[:,0]+X[:,1]+X[:,2]-12.5).reshape(-1,1)   # Zn+Mg+Cu <= 12.5 (anti hot-tear/quench-sens)

res=minimize(Design(),NSGA2(pop_size=150),('n_gen',80),seed=1,verbose=False)
PF=make_df(res.X); pr=predict(PF)
for k,v in pr.items(): PF[k]=v
PF['novelty']=novelty(PF); PF['score']=PF['UTS_mean']-0.5*PF['UTS_std']
# diversity via KMeans(5) on standardized design vars; pick best-score member per cluster
Z=(PF[FREE+HT].values-MU)/SD
PF['cl']=KMeans(5,n_init=10,random_state=0).fit_predict(Z)
out=PF.sort_values('score',ascending=False).groupby('cl',as_index=False).first().sort_values('UTS_mean',ascending=False)
show=['Zn','Mg','Cu','Zr','Mn','Cr','Ti','Sc','S/A Temp','Ageing Temp','Ageing Time',
      'UTS_mean','UTS_std','YTS_mean','YTS_std','EL_mean','novelty']
pd.set_option("display.width",230)
print("Pareto size:",len(res.X),"| candidates selected:",len(out))
print("\n=== 5 CANDIDATE 7xxx ALLOYS (for Thermo-Calc screening) ===")
print(out[show].round(2).to_string(index=False))
print("\nBenchmarks: 7075-T6 UTS~572, 7050-T6~552, 7055-T77~640 MPa; dataset 7xxx max UTS=679.")
out['temper_recipe']=("SHT "+out['S/A Temp'].round(0).astype(int).astype(str)+
   "C + WQ + age "+out['Ageing Temp'].round(0).astype(int).astype(str)+"C/"+
   out['Ageing Time'].round(0).astype(int).astype(str)+"h (T6)")
out.to_csv(D+"data/candidates.csv",index=False)
print("\nsaved -> data/candidates.csv")
