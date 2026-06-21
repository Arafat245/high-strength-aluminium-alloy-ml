"""Forward model: (composition + heat-treatment) -> UTS / YTS, on Hu et al. 2024.
Trains an uncertainty-aware XGBoost ensemble, validates with grouped CV
(generalization to unseen alloys), saves models for inverse design."""
import pandas as pd, numpy as np, warnings, pickle
warnings.filterwarnings("ignore")
from sklearn.model_selection import KFold, GroupKFold, cross_val_predict
from sklearn.metrics import r2_score, mean_absolute_error
import xgboost as xgb
D="/mnt/sdb/arafat/Al_alloy_design/"

COMP=['Si','Fe','Cu','Mn','Mg','Cr','Zn','V','Ti','Zr','Li','Ni','Ga','Bi','Pb','B','Be','Sc']
HT=['S/A Temp','Ageing Temp','Ageing Time']
CAT=['ManufacturingMethod','HeatTreatmentMedium','StrainHardeningMethod','AgeingType','TreatmentAfterward','ProductionShape']
# compact elemental property tables (Al = balance)
MASS=dict(Al=26.98,Si=28.09,Fe=55.85,Cu=63.55,Mn=54.94,Mg=24.31,Cr=52.0,Zn=65.38,V=50.94,Ti=47.87,Zr=91.22,Li=6.94,Ni=58.69,Ga=69.72,Bi=208.98,Pb=207.2,B=10.81,Be=9.01,Sc=44.96)
VEC=dict(Al=3,Si=4,Fe=8,Cu=11,Mn=7,Mg=2,Cr=6,Zn=12,V=5,Ti=4,Zr=4,Li=1,Ni=10,Ga=3,Bi=5,Pb=4,B=3,Be=2,Sc=3)
CHI=dict(Al=1.61,Si=1.90,Fe=1.83,Cu=1.90,Mn=1.55,Mg=1.31,Cr=1.66,Zn=1.65,V=1.63,Ti=1.54,Zr=1.33,Li=0.98,Ni=1.91,Ga=1.81,Bi=2.02,Pb=2.33,B=2.04,Be=1.57,Sc=1.36)
RAD=dict(Al=143,Si=117,Fe=126,Cu=128,Mn=127,Mg=160,Cr=128,Zn=134,V=134,Ti=147,Zr=160,Li=152,Ni=124,Ga=122,Bi=156,Pb=175,B=82,Be=112,Sc=162)

def physics(df):
    """VEC, mean electronegativity, atomic-size mismatch delta from wt% (+Al balance)."""
    out=pd.DataFrame(index=df.index)
    w=df[COMP].clip(lower=0).copy(); w['Al']=(100-w.sum(axis=1)).clip(lower=0)
    els=COMP+['Al']
    moles=w[els]/pd.Series(MASS)
    x=moles.div(moles.sum(axis=1),axis=0).fillna(0)           # atomic fractions
    out['VEC']=x.mul(pd.Series(VEC)).sum(axis=1)
    out['mean_chi']=x.mul(pd.Series(CHI)).sum(axis=1)
    rbar=x.mul(pd.Series(RAD)).sum(axis=1)
    dev=pd.DataFrame({e:(1-RAD[e]/rbar)**2 for e in els})
    out['delta']=np.sqrt((x*dev).sum(axis=1))*100
    return out

def build_X(df):
    X=pd.concat([df[COMP], df[HT], physics(df),
                 pd.get_dummies(df[CAT].astype(str),prefix=CAT)],axis=1)
    return X

df=pd.read_excel(D+"data/hu2024_dataset.xlsx")
groups=df['Alloy'].astype(str).fillna('na').values
X=build_X(df)
print("feature matrix:",X.shape,"| n features:",X.shape[1])

def evaluate(target):
    m=df[target].notna()
    Xt,y,g=X[m],df.loc[m,target].values,groups[m]
    mdl=xgb.XGBRegressor(n_estimators=600,max_depth=5,learning_rate=0.03,
        subsample=0.8,colsample_bytree=0.8,min_child_weight=3,random_state=0)
    pr=cross_val_predict(mdl,Xt,y,cv=KFold(5,shuffle=True,random_state=0))
    pg=cross_val_predict(mdl,Xt,y,cv=GroupKFold(5),groups=g)
    print(f"\n{target}: n={m.sum()}  (distinct alloy groups={pd.Series(g).nunique()})")
    print("  random 5-fold:        R2=%.3f  MAE=%.1f MPa"%(r2_score(y,pr),mean_absolute_error(y,pr)))
    print("  grouped 5-fold (new): R2=%.3f  MAE=%.1f MPa"%(r2_score(y,pg),mean_absolute_error(y,pg)))
    # 7xxx-specific accuracy under grouped CV
    is7=((df['Zn']>=3)&(df['Zn']>df['Cu'])&(df['Zn']>df['Si'])&(df['Zn']>df['Mg']))[m].values
    if is7.sum()>10:
        print("  grouped CV on 7xxx rows: R2=%.3f  MAE=%.1f MPa (n=%d)"%(
            r2_score(y[is7],pg[is7]),mean_absolute_error(y[is7],pg[is7]),is7.sum()))
    return Xt,y

models={}
for tgt in ['UTS','YTS','EL']:
    Xt,y=evaluate(tgt)
    # final bootstrap ensemble (for mean + uncertainty in inverse design)
    ens=[]
    for s in range(12):
        idx=np.random.RandomState(s).choice(len(y),len(y),replace=True)
        mm=xgb.XGBRegressor(n_estimators=600,max_depth=5,learning_rate=0.03,subsample=0.8,
            colsample_bytree=0.8,min_child_weight=3,random_state=s)
        mm.fit(Xt.iloc[idx],y[idx]); ens.append(mm)
    models[tgt]=ens

# feature importance (UTS, first member)
imp=pd.Series(models['UTS'][0].feature_importances_,index=X.columns).sort_values(ascending=False)
print("\nTop-15 UTS feature importances:"); print(imp.head(15).round(3).to_string())

pickle.dump({'models':models,'columns':list(X.columns),'COMP':COMP,'HT':HT},
            open(D+"models.pkl","wb"))
print("\nsaved -> models.pkl")
