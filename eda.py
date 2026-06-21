import pandas as pd, numpy as np, re, warnings
warnings.filterwarnings("ignore")
pd.set_option("display.width", 160); pd.set_option("display.max_columns", 80)
D = "/mnt/sdb/arafat/Al_alloy_design/Dataset/"

def hdr(t): print("\n" + "="*78 + "\n" + t + "\n" + "="*78)

prop = pd.read_csv(D+"property.csv", encoding="utf-8-sig")
comp = pd.read_csv(D+"composition.csv", encoding="utf-8-sig")
prop.columns = [c.strip().lstrip("﻿") for c in prop.columns]

hdr("PROPERTY.CSV")
print("shape:", prop.shape)
print("columns:", list(prop.columns))
print("\n-- missingness (%) --")
print((prop.isna().mean()*100).round(1).to_string())
print("\n-- YS / UTS / elong describe --")
print(prop[["YS","UTS","elong"]].describe().round(1).to_string())
print("\nrows with BOTH YS & UTS:", prop[["YS","UTS"]].dropna().shape[0])
print("rows with UTS:", prop["UTS"].notna().sum(), "| rows with YS:", prop["YS"].notna().sum())
print("\n-- count & median strength by series --")
g = prop.groupby("series").agg(n=("UTS","size"),
      UTS_med=("UTS","median"), UTS_max=("UTS","max"),
      YS_med=("YS","median"), YS_max=("YS","max")).round(0)
print(g.to_string())
print("\n-- top 10 by UTS --")
print(prop.sort_values("UTS",ascending=False)[["series","AA_des","temper","YS","UTS","elong","flag"]].head(10).to_string(index=False))
print("\nflag (special processing) distribution:")
print(prop["flag"].value_counts(dropna=False).to_string())
print("\nunique AA_des in property:", prop["AA_des"].nunique())

hdr("COMPOSITION.CSV")
print("shape:", comp.shape)
elem_cols = [c for c in comp.columns if c not in ["source","ft_doi_list","table_doi","name","AA_des"]]
print("n element columns:", len(elem_cols))
print("\nsource breakdown:")
print(comp["source"].value_counts(dropna=False).to_string())
print("\nrows with AA_des:", comp["AA_des"].notna().sum(), "| unique AA_des:", comp["AA_des"].nunique())

# element presence
nz = (comp[elem_cols].fillna(0) != 0).sum().sort_values(ascending=False)
print("\n-- element presence (count of alloys with element != 0), top 25 --")
print(nz.head(25).to_string())
print("\n-- elements present in <5 alloys (likely text-extraction noise) --")
rare = nz[nz < 5]
print(f"{len(rare)} elements: " + ", ".join(f"{e}({n})" for e,n in rare.items()))

# composition sums
comp_sum = comp[elem_cols].fillna(0).sum(axis=1)
print("\n-- row composition sum (should be ~100 wt%) --")
print(comp_sum.describe().round(2).to_string())
print("rows summing 95-105:", ((comp_sum>=95)&(comp_sum<=105)).sum())

# sparsity
nzpr = (comp[elem_cols].fillna(0)!=0).sum(axis=1)
print("\nnonzero elements per alloy: mean %.1f  median %d  max %d" % (nzpr.mean(), nzpr.median(), nzpr.max()))

hdr("LINKAGE: can we build (composition -> strength) supervised data?")
pa = set(pd.to_numeric(prop["AA_des"],errors="coerce").dropna().astype(int))
ca = set(pd.to_numeric(comp["AA_des"],errors="coerce").dropna().astype(int))
# composition 'name' for named source often holds the designation
comp_name_num = set(pd.to_numeric(comp["name"],errors="coerce").dropna().astype(int))
print("unique AA_des  property:", len(pa), "| composition:", len(ca))
print("AA_des intersection (prop & comp.AA_des):", len(pa & ca))
print("prop AA_des that match comp.AA_des OR comp.name:", len(pa & (ca|comp_name_num)))

# DOI linkage
pdoi = set(prop["doi"].dropna().astype(str).str.strip())
cdoi = set(comp["table_doi"].dropna().astype(str).str.strip())
ft = comp["ft_doi_list"].dropna().astype(str)
ftset=set()
for s in ft:
    for tok in re.split(r"[\[\]',\s]+", s):
        if tok: ftset.add(tok.strip())
print("\nproperty DOIs:", len(pdoi), "| comp table_doi:", len(cdoi), "| comp ft_doi:", len(ftset))
print("prop DOI ∩ comp(table+ft) DOI:", len(pdoi & (cdoi|ftset)))

# Build supervised set via 'named' nominal compositions matched to property AA_des
named = comp[comp["source"]=="named"].copy()
named["key"] = pd.to_numeric(named["name"],errors="coerce")
named = named.dropna(subset=["key"]); named["key"]=named["key"].astype(int)
named = named.drop_duplicates("key")
P = prop.copy(); P["key"]=pd.to_numeric(P["AA_des"],errors="coerce")
merged = P.dropna(subset=["key"]).merge(named[["key"]+elem_cols], on="key", how="inner")
print("\n[A] property rows joinable to a NAMED nominal composition:", merged.shape[0],
      "(of", prop["UTS"].notna().sum(),"rows with UTS)")
print("    -> with UTS present:", merged["UTS"].notna().sum(), "| with YS present:", merged["YS"].notna().sum())

# Build supervised set via DOI: property row + composition from same paper (table)
ct = comp[comp["source"]=="table"].copy()
ct["d"]=ct["table_doi"].astype(str).str.strip()
P["d"]=P["doi"].astype(str).str.strip()
mdoi = P.merge(ct[["d"]+elem_cols], on="d", how="inner")
print("[B] property rows joinable by DOI to a TABLE composition:", mdoi.shape[0],
      "(may be many-to-many within a paper)")

print("\nDONE")
