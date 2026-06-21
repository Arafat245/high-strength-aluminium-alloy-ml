# Column dictionary — `candidates_final.csv`

The 5 best cheap, in-distribution 7xxx candidate alloys. One row = one candidate alloy
(composition + heat-treatment recipe + predicted properties). All compositions in **weight %**;
all strengths in **MPa**.

| Column | Full name | Units | Meaning |
|--------|-----------|-------|---------|
| `Al` | Aluminium | wt % | Base metal, computed as **balance** = 100 − (sum of all other elements). |
| `Zn` | Zinc | wt % | Main strengthening element (forms MgZn₂ / η′ precipitates). Design variable. |
| `Mg` | Magnesium | wt % | Main strengthening element (pairs with Zn). Design variable. |
| `Cu` | Copper | wt % | Strengthening + improves some properties; lowers weldability/SCC at high levels. Design variable. |
| `Mn` | Manganese | wt % | Cheap dispersoid former (grain-structure control). Design variable. |
| `Cr` | Chromium | wt % | Cheap dispersoid former (as in 7075). Design variable. |
| `Fe` | Iron | wt % | Impurity, **fixed at 0.15** (typical foundry level) — not optimised. |
| `Si` | Silicon | wt % | Impurity, **fixed at 0.10** — not optimised. |
| `Tsol` | **Solution(ising) temperature** | **°C** | Temperature the cast alloy is held at to **dissolve all solute into solid solution** before water-quenching (the "SHT" step). |
| `Tage` | **Ageing temperature** | **°C** | After quenching, the temperature the alloy is held at to **precipitate the nanoscale strengthening phases** (η′/MgZn₂) — the "age" step of a T6 temper. |
| `tage` | **Ageing time** | **hours** | How long the alloy is held at `Tage` to reach peak strength. |
| `UTS` | Ultimate tensile strength | MPa | **Predicted** peak stress before fracture — **mean** of the 12-model ensemble. |
| `UTS s` | UTS σ (standard deviation) | MPa | **Spread of the 12 ensemble models' UTS predictions** = the model's internal confidence (smaller = models agree). NOTE: understates true error; honest band is **±43 MPa** (grouped-CV). |
| `YS` | Yield strength | MPa | **Predicted** stress at which permanent deformation begins — ensemble **mean**. |
| `YS s` | YS σ (standard deviation) | MPa | Ensemble disagreement / uncertainty on yield strength. |
| `EL` | Elongation | % | Predicted ductility. **Low reliability** (model R² ≈ 0.3) — rough guide only. |
| `novelty` | Novelty distance | dimensionless | Standardised distance from the nearest real 7xxx alloy. ≈1 = trustworthy tweak of a known alloy; all candidates ≤ 1.32 (in-distribution → reliable). |
| `score` | Ranking score | MPa | The metric used to rank/select candidates: **`score = UTS − 1 × (UTS s)`** — rewards high strength, penalises uncertain predictions. The file is sorted by this. |

**The `s` suffix = sigma (σ)**, the standard deviation across the 12 bootstrap models — an
*uncertainty estimate*, not a property of the alloy.

## How to read the heat-treatment recipe

`Tsol`, `Tage`, `tage` together describe a **T6 temper**. Example (candidate #1):

> `Tsol 488, Tage 104, tage 24`  →  **"Solutionise at 488 °C, water-quench, then artificially age
> at 104 °C for 24 hours."**

## How to read a prediction

> `UTS 635, UTS s 13.6`  →  predicted UTS = **635 ± 14 MPa** (ensemble spread); realistic
> band **635 ± 43 MPa** (grouped-CV error). `YS 555, YS s 25` → yield = **555 ± 25 MPa**.

## Important caveats

- Predictions are **computational hypotheses**, reliable to about **±43 MPa** — they require
  Thermo-Calc screening and experimental casting/testing to confirm.
- Within ±43 MPa, candidates #1–#3 (UTS 635/640/639) are **statistically indistinguishable** in
  strength — use **YS** (#2 is best at 589) and **castability** (Thermo-Calc) to choose between them.
- The model does not see grain size or multi-stage ageing (RRA/T74); those are the levers for
  pushing beyond this strength range.
