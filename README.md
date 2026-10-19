# water-accounts-br-states

Code and result tables for the manuscript *Structural recoupling of water use and value added in Brazil: hybrid state water accounts, 2010–2021* (submitted to *Ecological Economics*, 2026).

Hybrid physical–monetary water accounts are built for the 27 Brazilian states, 2010–2021, from the national water agency's municipal estimates of withdrawal and consumption for seven purposes and sector-deflated gross value added from the regional accounts. The study computes water productivity and SDG 6.4.1 efficiency, classifies decoupling with Tapio elasticities (and their sensitivity to the averaging window), decomposes the change in withdrawal with an additive LMDI and estimates within- and between-state elasticities. The loss of water productivity is attributed mainly to the sector mix (rising share of agriculture) rather than to intensity.

## Layout

| Folder | Content |
|---|---|
| `scripts/` | Analysis (`analise_agua.py`, `robustez.py`, `revisao_17-10.py`), figures, the manuscript builder and the tested routines in `gwbench/contas.py` |
| `dados/` | Public inputs (ANA water uses by state, IBGE regional accounts GVA, national-accounts deflators, population, irrigated area 2017) |
| `resultados/` | State panel, Tapio elasticities, LMDI cells and summary, SDG 6.4.1, PCA index, numbers quoted in the text, revision files (`rev17_*`) |
| `figuras/` | Figs. 1–5 and A1 of the manuscript; `eq1–eq3.png` are the rendered equations used by the docx builder |

## Pipeline

1. `scripts/analise_agua.py` — state panel (`resultados/painel_uf_ano.csv`, `brasil_ano.csv`), Tapio elasticities, LMDI, SDG 6.4.1, panel elasticities and the composite index.
2. `scripts/robustez.py` — robustness checks of the index and elasticities.
3. `scripts/revisao_17-10.py` — revision analyses: LMDI with the household block, Tapio window sensitivity, wild-cluster bootstrap, SDG 6.4.1 sensitivity (`resultados/rev17_*`).
4. `scripts/figuras_agua.py` — Figs. 1–5 (300 dpi).
5. `scripts/manuscript_parts.py`, `manuscript_docx.py` — tables from `resultados/` and the manuscript in the journal format (manuscript source not included).

Run the scripts from any working directory: every path is resolved relative to the script's own location (`os.path.dirname(__file__)`).

## Data

Water withdrawal and consumption by municipality and purpose: ANA, *Manual de Usos Consuntivos da Água no Brasil* / SNIRH (https://www.gov.br/ana; open government data, CC BY-like attribution). Gross value added by state and sector: IBGE Regional Accounts, SIDRA table 5938 (https://sidra.ibge.gov.br; free reuse with attribution). Deflators: IBGE National Accounts. Population: IBGE estimates. Irrigated area: ANA Atlas Irrigação 2017. The small input files used are included in `dados/` for reproducibility.

The ANA municipal estimates and the full SIDRA extracts are **not** redistributed; `dados/` holds only the small state-level files actually read by the scripts, downloaded from the public sources above.

## Citation

See `CITATION.cff` (co-authors to be added). Licence: MIT.

## Archive

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23461524.svg)](https://doi.org/10.5281/zenodo.23461524)

Version v1.0.0: https://doi.org/10.5281/zenodo.23461525
