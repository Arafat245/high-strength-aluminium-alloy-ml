import pandas as pd, numpy as np, warnings
warnings.filterwarnings("ignore")
from sklearn.model_selection import KFold, GroupKFold, cross_val_predict
from sklearn.metrics import r2_score, mean_absolute_error
import xgboost as xgb
D = "/mnt/sdb/arafat/Al_alloy_design/Dataset/"

def hdr(t): print("\n"+"="*74+"\n"+t+"\n"+"="*74)

prop = pd.read_csv(D+"property.csv", encoding="utf-8-sig")
prop.columns=[c.strip().lstrip("﻿") for c in prop.columns]
comp = pd.read_csv(D+"composition.csv", encoding="utf-8-sig")
elem_all=[c for c in comp.columns if c not in["source","ft_doi_list","table_doi","name","AA_des"]]
noise=["Pr","Ho","Th","U","Tb","As","Re","Hg","Cf","Tm","Lu","Pd","Pt","F","Rh","Ru"]
elem=[e for e in elem_all if e not in noise]

# Build set [A]: property UTS joined to NAMED nominal composition via 4-digit designation
named=comp[comp["source"]=="named"].copy()
named["key"]=pd.to_numeric(named["name"],errors="coerce")
named=named.dropna(subset=["key"]); named["key"]=named["key"].astype(int)
named=named.drop_duplicates("key")
P=prop.copy(); P["key"]=pd.to_numeric(P["AA_des"],errors="coerce")
df=P.dropna(subset=["key"]).merge(named[["key"]+elem],on="key",how="inner").dropna(subset=["UTS"])
print("merged rows (composition+UTS):",len(df))
print("DISTINCT composition vectors (unique AA_des):",df["key"].nunique())
print("rows with temper:",df["temper"].notna().sum(),
      "| distinct tempers:",df["temper"].nunique())

hdr("Why composition alone is not enough: UTS spread for a FIXED alloy")
sp=df.groupby("key")["UTS"].agg(n="size",mean="mean",std="std",mn="min",mx="max")
sp["range"]=sp["mx"]-sp["mn"]
print("Mean within-alloy UTS std (MPa):",round(sp["std"].mean(),1))
print("Mean within-alloy UTS range (MPa):",round(sp["range"].mean(),1))
print("\nExamples (same composition, different temper/processing):")
print(sp[sp["n"]>=5].sort_values("range",ascending=False)
      .head(8).round(0).to_string())

hdr("Forward model: composition -> UTS  (XGBoost)")
X=df[elem].values; y=df["UTS"].values; groups=df["key"].values
m=xgb.XGBRegressor(n_estimators=400,max_depth=4,learning_rate=0.05,
                   subsample=0.8,colsample_bytree=0.8,random_state=0)
# (1) random 5-fold (rows of same alloy can leak across folds)
p1=cross_val_predict(m,X,y,cv=KFold(5,shuffle=True,random_state=0))
print("Random 5-fold:      R2=%.2f  MAE=%.0f MPa" %
      (r2_score(y,p1),mean_absolute_error(y,p1)))
# (2) GroupKFold by alloy => test on UNSEEN compositions (the real discovery setting)
gk=GroupKFold(n_splits=min(5,df["key"].nunique()))
p2=cross_val_predict(m,X,y,cv=gk,groups=groups)
print("Group 5-fold (new alloys): R2=%.2f  MAE=%.0f MPa" %
      (r2_score(y,p2),mean_absolute_error(y,p2)))
# baseline: predict global mean
print("Baseline (predict mean):   R2=%.2f  MAE=%.0f MPa" %
      (r2_score(y,np.full_like(y,y.mean())),mean_absolute_error(y,np.full_like(y,y.mean()))))

hdr("Does adding TEMPER help? (composition + one-hot temper)")
sub=df.copy(); sub["temper"]=sub["temper"].fillna("NA")
top_temp=sub["temper"].value_counts().head(12).index
for t in top_temp: sub["T_"+str(t)]=(sub["temper"]==t).astype(int)
tcols=[c for c in sub.columns if c.startswith("T_")]
X2=sub[elem+tcols].values; y2=sub["UTS"].values; g2=sub["key"].values
p3=cross_val_predict(m,X2,y2,cv=GroupKFold(min(5,sub['key'].nunique())),groups=g2)
print("comp+temper Group 5-fold:  R2=%.2f  MAE=%.0f MPa" %
      (r2_score(y2,p3),mean_absolute_error(y2,p3)))

hdr("Design-space mapping potential (all 13,358 compositions)")
from sklearn.decomposition import PCA
Xall=comp[elem].fillna(0).values
print("composition matrix:",Xall.shape,"| sparsity (zeros): %.0f%%"%(100*(Xall==0).mean()))
pca=PCA(10).fit(Xall)
print("variance explained by first 5 PCs:",
      np.round(pca.explained_variance_ratio_[:5],3),
      "(cumulative %.0f%%)"%(100*pca.explained_variance_ratio_[:5].sum()))
print("\nDONE")
