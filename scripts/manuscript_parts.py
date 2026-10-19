"""Tables (built from the result CSVs) and figure legends for the Ecological Economics manuscript."""
import json
from pathlib import Path

import pandas as pd

R = Path(__file__).resolve().parents[1]
O = R / "resultados"
N = json.load(open(O / "numeros_texto.json"))
q = pd.read_csv(O / "painel_uf_ano.csv")
P0, P1 = [2010, 2011, 2012], [2019, 2020, 2021]
br = q.groupby("ano").sum(numeric_only=True)
mm = lambda s, P: s.loc[P].mean()


def f(x, d=1):
    s = f"{x:,.{d}f}".replace(",", " ")
    return s.replace("-", "−")


def pct(a, b):
    return f((b / a - 1) * 100, 1) + "%"


TAB = {}
TAB["TAB1"] = """**Table 1.** Data sources and treatment.

| Variable | Source | Treatment |
| --- | --- | --- |
| Gross value added (GVA) by state and activity, current prices | IBGE Regional Accounts, SIDRA table 5938 (IBGE, 2023) | Agriculture, industry, services, and public administration; 27 states; 2010–2021 |
| Sectoral price deflators | IBGE Quarterly National Accounts, SIDRA tables 1846 and 6612 | Implicit deflator = annual sum of current-price GVA / annual sum of chained-volume GVA; rebased to 2015 |
| Water withdrawal and consumption by municipality, purpose, and year | ANA consumptive-use series 1931–2040 (ANA, 2019, 2022) | Seven purposes (urban and rural supply, manufacturing, mining, thermoelectric cooling, livestock, irrigation) summed to states; 2010–2021 |
| Irrigated area | IBGE Agricultural Census 2017, SIDRA table 6857 | Share of irrigated land Ai for the SDG 6.4.1 rainfed correction |
| Cropland area | IBGE Municipal Agricultural Survey 2017, SIDRA tables 1612 and 1613 | Planted area of temporary and permanent crops |
| Population | IBGE population estimates, SIDRA table 6579 | Per-capita supply; 2010 interpolated (census year) |
"""

rows = [
    ("Real GVA (R$ billion of 2015)", br.g_tot, 0),
    ("  Agriculture", br.g_agro, 0),
    ("  Industry", br.g_ind, 0),
    ("  Services and public administration", br.g_terc, 0),
    ("Withdrawal (km³ yr⁻¹)", br.ret_tot / 1e3, 1),
    ("  Irrigation", br.ret_ir / 1e3, 1),
    ("  Livestock", br.ret_an / 1e3, 1),
    ("  Industry (manufacturing, mining, thermoelectric)", br.ret_ind / 1e3, 1),
    ("  Human supply (urban + rural)", br.ret_hum / 1e3, 1),
    ("Consumption (km³ yr⁻¹)", br.con_tot / 1e3, 1),
    ("Water productivity, withdrawal (R$ m⁻³)", br.g_tot * 1e3 / br.ret_tot, 1),
    ("Water productivity, consumption (R$ m⁻³)", br.g_tot * 1e3 / br.con_tot, 1),
    ("SDG 6.4.1 water-use efficiency (R$ m⁻³)", None, 1),
    ("  Agriculture, GVA per m³ (no rainfed correction)", br.g_agro * 1e3 / br.ret_agro, 2),
    ("  Industry, GVA per m³", br.g_ind * 1e3 / br.ret_ind, 1),
    ("  Services and public administration, GVA per m³ of human supply (urban + rural)", br.g_terc * 1e3 / br.ret_hum, 1),
    ("Irrigation share of withdrawal (%)", 100 * br.ret_ir / br.ret_tot, 1),
    ("Human supply per capita (L person⁻¹ day⁻¹)", br.ret_hum * 1e9 / br["pop"] / 365, 0),
]
t2 = ["**Table 2.** National totals and indicators, three-year means at the ends of the period.", "",
      "| Indicator | 2010–2012 | 2019–2021 | Change |", "| --- | --- | --- | --- |"]
for nm, s, d in rows:
    if s is None:
        a, b = N["WUE641"]
    else:
        a, b = mm(s, P0), mm(s, P1)
    ch = f(b - a, 1) + " p.p." if "%" in nm and "share" in nm else pct(a, b)
    t2.append(f"| {nm} | {f(a, d)} | {f(b, d)} | {ch} |")
t2 += ["", "GVA deflated by sector with implicit national deflators (2015 prices). Water: ANA estimates. "
       "SDG 6.4.1 includes the rainfed correction Cr (Section 2.2); the agricultural line does not. The industrial line includes thermoelectric cooling (Table 4 excludes it). Ratios are means of the annual ratios "
       "over each three-year window, so they differ slightly from the ratio of the means (e.g., −8.6% instead of −8.8% for water productivity)."]
TAB["TAB2"] = "\n".join(t2) + "\n"

R7 = json.load(open(O / "rev17_numeros.json"))["lmdi7"]["especificacoes"]
t3 = ["**Table 3.** LMDI decomposition of the change in national water use (km³ yr⁻¹), 2010–2012 to 2019–2021.", "",
      "| Specification | Scale | Sector mix | Regional mix | Intensity | Population | Per-capita supply | Total change |",
      "| --- | --- | --- | --- | --- | --- | --- | --- |"]
for nm, v in R7.items():
    t3.append(f"| {nm} | {f(v['escala'] / 1e3, 2)} | {f(v['mix_setorial'] / 1e3, 2)} | {f(v['mix_regional'] / 1e3, 2)} | "
              f"{f(v['intensidade'] / 1e3, 2)} | {f((v['populacao'] + v['mix_regional_pop']) / 1e3, 2)} | {f(v['per_capita'] / 1e3, 2)} | {f(v['dW'] / 1e3, 2)} |")
t3 += ["", "Additive LMDI-I (Eq. 2). Scale, sector mix, regional mix and intensity refer to the productive block (27 states × 2 sectors: "
       "agriculture; industry). Population and per-capita supply refer to the household block (urban and rural human supply); "
       "the redistribution of population among states (at most 0.05 km³ yr⁻¹) is included in the population effect. "
       "Effects sum exactly to the total change. Negative values reduce water use."]
TAB["TAB3"] = "\n".join(t3) + "\n"

fe = pd.read_csv(O / "rev17_elasticidades_painel_complementos.csv")
lab = {"total": "Total withdrawal / total GVA", "agropecuária": "Agricultural withdrawal / agricultural GVA",
       "irrigação": "Irrigation withdrawal / agricultural GVA", "indústria": "Industrial withdrawal (excl. thermoelectric) / industrial GVA",
       "abastecimento humano (per capita)": "Human supply per capita / GVA per capita"}
t4 = ["**Table 4.** Elasticities of water withdrawal with respect to real GVA, 2010–2021: within-state (panel), long-difference and between-state estimates.", "",
      "| Pair | State + year effects | State effects + state trends | Long difference, 2010–12 to 2019–21 | Between (state means) |",
      "| --- | --- | --- | --- | --- |"]
for s_ in lab:
    a = fe[(fe.setor == s_) & (fe.efeitos == "UF + ano")].iloc[0]
    b = fe[(fe.setor == s_) & (fe.efeitos == "UF + tendência por UF")].iloc[0]
    c = fe[(fe.setor == s_) & (fe.efeitos.str.startswith("long"))].iloc[0]
    d = fe[(fe.setor == s_) & (fe.efeitos.str.startswith("between"))].iloc[0]
    t4.append(f"| {lab[s_]} | {f(a.elasticidade, 2)} ({f(a.ic95_inf, 2)} to {f(a.ic95_sup, 2)}); R²w {f(a.R2_within, 2)}; p {f(a.p_wild_cluster_boot, 2)} | "
              f"{f(b.elasticidade, 2)} ({f(b.ic95_inf, 2)} to {f(b.ic95_sup, 2)}); R²w {f(b.R2_within, 2)}; p {f(b.p_wild_cluster_boot, 2)} | "
              f"{f(c.elasticidade, 2)} ({f(c.ic95_inf, 2)} to {f(c.ic95_sup, 2)}) | {f(d.elasticidade, 2)} ({f(d.ic95_inf, 2)} to {f(d.ic95_sup, 2)}) |")
t4 += ["", "Panel columns: OLS in logs, 27 states × 12 years (n = 324); 95% confidence intervals (CI) from standard errors clustered by state; "
       "R²w, within-state R²; p, two-sided p-value from a wild-cluster bootstrap-t (Rademacher weights, 999 draws, null imposed). "
       "Long difference: change in log water on change in log GVA between three-year means, 27 observations, HC1 standard errors. "
       "Between: regression of state means over 2010–2021, 27 observations, HC1 standard errors."]
TAB["TAB4"] = "\n".join(t4) + "\n"

u = pd.read_csv(O / "uf_2019_21.csv")
tp = pd.read_csv(O / "tapio_retirada.csv")
tp = tp[tp.periodo == "2010–12 → 2019–21"]
cl = {"desacoplamento forte": "SD", "desacoplamento fraco": "WD", "acoplamento expansivo": "EC",
      "desacoplamento negativo expansivo": "END", "desacoplamento negativo forte": "SND", "desacoplamento recessivo": "RD",
      "acoplamento recessivo": "RC", "desacoplamento negativo fraco": "WND"}
ix = pd.read_csv(O / "indice_pca_uf.csv").set_index("sigla")
u = u.sort_values("wue_641", ascending=False)
t5 = ["**Table A.1.** State indicators, 2019–2021 means, and Tapio decoupling regimes between 2010–2012 and 2019–2021. States are ordered by SDG 6.4.1 efficiency (descending).", "",
      "| State | Region | GVA (R$ bn) | Withdrawal (hm³) | WP (R$ m⁻³) | SDG 6.4.1 (R$ m⁻³) | Agric. GVA m⁻³ | Irrigation (%) | Tapio, total | Tapio, agric. | Index rank (90% interval) |",
      "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
for _, r in u.iterrows():
    ra = tp[(tp.uf == r.sigla) & (tp.setor == "total")].iloc[0]
    rb = tp[(tp.uf == r.sigla) & (tp.setor == "agropecuária")].iloc[0]
    a = cl.get(ra.classe, ra.classe) + ("†" if abs(ra.dg) < 0.03 else "")
    b = cl.get(rb.classe, rb.classe) + ("†" if abs(rb.dg) < 0.03 else "")
    i = ix.loc[r.sigla]
    t5.append(f"| {r.sigla} | {r.regiao} | {f(r.g_tot, 1)} | {f(r.ret_tot, 0)} | {f(r.wp_ret, 1)} | {f(r.wue_641, 1)} | {f(r.wue_agro, 1)} | "
              f"{f(r['irr_%'], 0)} | {a} | {b} | {int(i.posicao)} ({int(i.pos_p05)}–{int(i.pos_p95)}) |")
t5 += ["", "WP, water productivity of withdrawal. Tapio regimes: SD strong decoupling; WD weak decoupling; EC expansive coupling; "
       "END expansive negative decoupling; SND strong negative decoupling; RD recessive decoupling; RC recessive coupling; "
       "WND weak negative decoupling. † |ΔGVA| < 3%: the elasticity is not meaningful and the regime is indeterminate (Section 3.2, Table B.1). Index: composite of five indicators with principal-component (varimax) weights (Appendix B); interval from 2000 bootstrap resamples of states."]
TAB["TABA1"] = "\n".join(t5) + "\n"

ts = pd.read_csv(O / "rev17_tapio_sensibilidade_total.csv").set_index("uf")
jan = [c for c in ts.columns if "→" in c]
short = ["3-yr (base)", "1-yr", "2-yr", "4-yr", "5-yr", "3-yr shifted"]
tb = ["**Table B.1.** Sensitivity of the Tapio regime (all sectors) to the averaging window at the ends of the period.", "",
      "| State | ΔGVA (%) | ΔW (%) | " + " | ".join(short) + " | Regimes |", "| --- | --- | --- | " + " | ".join(["---"] * 6) + " | --- |"]
for u_, r in ts.iterrows():
    flag = "†" if abs(r.dG_base) < 0.03 else ""
    tb.append(f"| {('Brazil' if u_ == 'BR' else u_)}{flag} | {f(100 * r.dG_base, 1)} | {f(100 * r.dW_base, 1)} | " + " | ".join(str(r[c]) for c in jan) + f" | {int(r.n_regimes)} |")
tb += ["", "Windows: 3-yr (base), 2010–12 → 2019–21; 1-yr, 2010 → 2021; 2-yr, 2010–11 → 2020–21; 4-yr, 2010–13 → 2018–21; 5-yr, 2010–14 → 2017–21; "
       "3-yr shifted, 2011–13 → 2018–20. ΔGVA and ΔW are the baseline changes. Regimes: number of distinct regimes across the six windows. "
       "† |ΔGVA| < 3% in the baseline. Regime codes as in Table A.1. The agricultural version is in the supplementary file rev17_tapio_sensibilidade_agro.csv."]
TAB["TABB1"] = "\n".join(tb) + "\n"

FIG = {
    "FIG1": ("fig1_national.png", "**Fig. 1.** Brazil, 2010–2021. (a) Real GVA (total and agriculture), water withdrawal (total and irrigation), and water productivity of withdrawal (index, 2010 = 100). "
             "(b) Composition of withdrawal by purpose."),
    "FIG2": ("fig2_tapio.png", "**Fig. 2.** Decoupling between real GVA and water withdrawal by state, 2010–2012 to 2019–2021 (three-year means). "
             "(a) All sectors. (b) Agriculture (irrigation and livestock against agricultural GVA). Shaded areas: strong (dark) and weak (light) "
             "decoupling; grey lines: elasticities of 0.8 and 1.2 (Tapio, 2005). Grey band and open markers: |ΔGVA| < 3%, where the elasticity is not meaningful (indeterminate regime). Points beyond the axis limits are drawn at the edge (asterisk)."),
    "FIG3": ("fig3_lmdi.png", "**Fig. 3.** LMDI decomposition of the change in national water withdrawal (Eq. 2). Scale, sector mix, regional mix and intensity apply to agriculture and industry; population and per-capita supply apply to human supply. (a) Whole period and subperiods "
             "before and after the 2015–2016 recession. (b) Whole period by block. Diamonds: net change."),
    "FIG4": ("fig4_wue_states.png", "**Fig. 4.** (a) SDG 6.4.1 water-use efficiency by state, 2019–2021, with the rainfed correction; dashed line: "
             "Brazil. (b) Water productivity of withdrawal against the irrigation share of withdrawal (Spearman ρ = −0.72). Logarithmic x-axis in (a) and y-axis in (b)."),
    "FIGB1": ("fig5_index_ranks.png", "**Fig. B.1.** Composite water–economy performance index, rank of each state (1 = best). Dots: principal-component weights; "
             "crosses: equal weights; bars: 5th–95th percentiles over 2000 bootstrap resamples of states."),
}
