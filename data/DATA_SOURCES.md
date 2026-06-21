# Dataset provenance

All files were downloaded from public repositories (no scraping of license-restricted
databases). Local copies live in this `data/` folder. Originals were first fetched to
`/tmp/` by the research step and copied here for persistence.

| Local file | Source | DOI / repo | Rows×Cols | License |
|---|---|---|---|---|
| `hu2024_dataset.xlsx` | Hu et al. 2024, *Comput. Mater. Sci.* 244:113204 | GitHub: `UQ43815685/mitigating_data_bias_for_Al_alloys_design` (`dataset.xlsx`); paper DOI 10.1016/j.commatsci.2024.113204 | 1150×34 | **No license file** (default copyright — OK for private research, not redistribution) |
| `bhat2023_al_data.csv` | Bhat, Barnard & Birbilis 2023 | Mendeley Data, DOI **10.17632/b6br4yk6r3.1** (`al_data.csv`, *supervised* variant) | 1154×31 | **CC BY 4.0** (cleanest) |
| `cstg_origin1293.xlsx` | "TheFinal" age-hardenable wrought Al compilation | GitHub: `CSTG-bug/TheFinal` (`Data/Origin-1293.xlsx`) | 1292×38 | No license stated |
| `haoyun_aged_forged.xlsx` | Wrought-Aluminum-Alloys compilation | GitHub: `haoyunlianlian744/Wrought-Aluminum-Alloys` | 524×23 | No license stated |
| `haoyun_df_cleaned.xlsx` | same repo (cleaned subset) | GitHub: `haoyunlianlian744/Wrought-Aluminum-Alloys` (`data/df_cleaned.xlsx`) | 320×23 | No license stated |

## Re-download commands (verified reachable)
```bash
curl -L -o data/hu2024_dataset.xlsx \
  https://raw.githubusercontent.com/UQ43815685/mitigating_data_bias_for_Al_alloys_design/main/dataset.xlsx
# Bhat: https://data.mendeley.com/datasets/b6br4yk6r3/1  (download al_data.csv via browser/Mendeley API)
```

## Notes
- **Primary modeling base = `hu2024_dataset.xlsx`** — it has *numeric* heat-treatment
  (`S/A Temp`, `Ageing Temp`, `Ageing Time`), which feeds Thermo-Calc cross-checks.
- Bhat is the **clean-license corroborator** (processing is coarse: 10 categories, no temps).
- The CSTG / haoyun sets also carry numeric heat-treatment and can be **pooled after
  de-duplication** (all descend from overlapping wrought-Al literature compilations).
- "No license stated" ≠ permission to redistribute. For any publication, rely on the
  CC-BY Bhat set or contact the corresponding authors.
