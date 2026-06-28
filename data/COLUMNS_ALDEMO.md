# Column dictionary — `candidates_aldemo.csv`

5 new **Al-Cu-Si-Sc-Zr** alloys, designed to be validatable in Thermo-Calc's free
**ALDEMO** demo database (elements: Al, Cu, Si, Sc, Zr only). Parallel deliverable to
`candidates_final.csv` (the 7xxx set); see [`COLUMNS.md`](COLUMNS.md) for that one.

| Column | Meaning | Units |
|--------|---------|-------|
| `Al` | Aluminium — balance, = 100 − Σ(others) | wt % |
| `Cu` | Copper — **main strengthener**; precipitates θ′ (Al₂Cu) on ageing (the 2xxx mechanism) | wt % |
| `Si` | Silicon — in ALDEMO; kept low (minor, aids castability) | wt % |
| `Sc` | Scandium — forms Al₃Sc / Al₃(Sc,Zr) L1₂ dispersoids → grain refinement + thermal stability + strength | wt % |
| `Zr` | Zirconium — forms Al₃Zr / Al₃(Sc,Zr); **cheap (≈$30/kg)**, anti-recrystallisation | wt % |
| `Fe` | Iron — impurity, fixed 0.15 | wt % |
| `Tsol` | Solutionising temperature (dissolve Cu, then water-quench) | °C |
| `Tage` | Ageing temperature — low-T age for θ′ (Al₃(Sc,Zr) forms earlier, during casting/homogenisation) | °C |
| `tage` | Ageing time at `Tage` | hours |
| `UTS` | Ultimate tensile strength — **predicted** ensemble mean. **Conservative** (model under-credits Sc/Zr) | MPa |
| `UTS s` | σ across the 12 ensemble models. Honest band = **±35 MPa** (ALDEMO-subset grouped-CV MAE) | MPa |
| `YS` | Yield strength — predicted ensemble mean | MPa |
| `YS s` | σ across the 12 ensemble models | MPa |
| `EL` | Elongation — predicted; **low reliability** (R²≈0.4), rough guide only | % |
| `novelty` | Standardised distance to nearest real Al-Cu(-Sc-Zr) alloy. Space is sparse → 1–2 = genuinely new | — |
| `score` | Ranking metric = `UTS − 1×(UTS s)` | MPa |
| `near_real` | Closest real alloy in our data (composition + its measured UTS), for context | — |

## Reading the heat treatment
`Tsol 521, Tage 151, tage 35` → *"Solutionise ≈521 °C, water-quench, then age 151 °C for 35 h."*
**The Sc/Zr point:** Al₃(Sc,Zr) dispersoids form during **casting / homogenisation (≈350–450 °C)**,
not during this low-T age — their solvus is the first thing to check in Thermo-Calc.

## Reading a prediction
`UTS 452, UTS s 13` → predicted **452 ± 13 MPa** (ensemble); realistic band **452 ± 35 MPa**.
Predictions are **conservative**: real Al-Cu-Zr in our data already reaches 475–510 MPa.

## Caveats
- Computational hypotheses — confirm with Thermo-Calc (ALDEMO) + casting/testing.
- Within ±35 MPa all 5 are tied in UTS → choose by **castability + Sc cost**.
- Model captures **Cu/θ′** strength well; it **under-credits Al₃(Sc,Zr)** — that benefit
  (grain refinement, heat resistance) appears in Thermo-Calc, not in the UTS number.
- **Sc is expensive** (≈$3000/kg); **Zr is cheap**. Candidate #5 (lean Sc) is the lowest-cost pick.
