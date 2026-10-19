"""Analyses added in the revision of 17 Oct 2026 (reply to the independent reviewer).

Run from agua_final/scripts:  python revisao_17-10.py
Reads dados/*.csv through analise_agua.painel()/wue_641(); writes resultados/rev17_*.{csv,json}.

1. LMDI with the household block decomposed into population x per-capita supply (7 effects), all specifications.
2. Tapio sensitivity to the averaging window and flag of states with |dGVA| < 3% (elasticity not meaningful).
3. Panel elasticities: within R2, between estimator, long-difference regression, wild-cluster bootstrap p-values.
4. SDG 6.4.1 sensitivity to the rainfed/irrigated yield ratio (0.375, 0.543, 0.563).
5. Checks quoted in the text: sum of purposes vs ANA totals, Spearman WP x irrigation share, MT projection.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import analise_agua as A  # noqa: E402

R = Path(__file__).resolve().parents[1]
O = R / "resultados"
P0, P1, PM = A.P0, A.P1, A.PM
p = A.painel()
q = A.wue_641(p)
out = {}


# ------------------------------------------------------------------ 1. LMDI, households = population x per capita
def lmdi7(pp, a0, a1, w="ret", ind_col=None):
    """Additive LMDI-I with two blocks.
    Productive block (agriculture, industry): W_rs = G * (G_s/G) * (G_rs/G_s) * (W_rs/G_rs)
        -> scale, sector mix, regional mix (GVA), intensity.
    Household block (urban + rural human supply): W_hr = P * (P_r/P) * (W_hr/P_r)
        -> population, regional mix (population), per-capita supply.
    G is total real GVA (all sectors), so the sector-mix effect measures the share of agriculture and industry in the
    whole economy, as in the four-factor version; the service sector no longer carries an 'intensity'."""
    m0, m1 = A.media(pp, a0), A.media(pp, a1)
    ind_col = ind_col or f"{w}_ind"
    wmap = {"agro": f"{w}_agro", "ind": ind_col}
    cel = []
    for s, g in (("agro", "g_agro"), ("ind", "g_ind")):
        for (_, r0), (_, r1) in zip(m0.iterrows(), m1.iterrows()):
            cel.append((r0.sigla, s, r0[g], r1[g], r0[wmap[s]], r1[wmap[s]], np.nan, np.nan))
    for (_, r0), (_, r1) in zip(m0.iterrows(), m1.iterrows()):
        cel.append((r0.sigla, "hh", np.nan, np.nan, r0[f"{w}_hum"], r1[f"{w}_hum"], r0["pop"], r1["pop"]))
    c = pd.DataFrame(cel, columns=["uf", "setor", "g0", "g1", "w0", "w1", "p0", "p1"])
    G0, G1 = m0.g_tot.sum(), m1.g_tot.sum()
    Pop0, Pop1 = m0["pop"].sum(), m1["pop"].sum()
    Lw = A._L(c.w1, c.w0)
    prod = c.setor != "hh"
    S0 = c.groupby("setor").g0.transform("sum")
    S1 = c.groupby("setor").g1.transform("sum")
    for k in ("escala", "mix_setorial", "mix_regional", "intensidade", "populacao", "mix_regional_pop", "per_capita"):
        c[k] = 0.0
    c.loc[prod, "escala"] = (Lw * np.log(G1 / G0))[prod]
    c.loc[prod, "mix_setorial"] = (Lw * np.log((S1 / G1) / (S0 / G0)))[prod]
    c.loc[prod, "mix_regional"] = (Lw * np.log((c.g1 / S1) / (c.g0 / S0)))[prod]
    c.loc[prod, "intensidade"] = (Lw * np.log((c.w1 / c.g1) / (c.w0 / c.g0)))[prod]
    hh = ~prod
    c.loc[hh, "populacao"] = (Lw * np.log(Pop1 / Pop0))[hh]
    c.loc[hh, "mix_regional_pop"] = (Lw * np.log((c.p1 / Pop1) / (c.p0 / Pop0)))[hh]
    c.loc[hh, "per_capita"] = (Lw * np.log((c.w1 / c.p1) / (c.w0 / c.p0)))[hh]
    c["dW"] = c.w1 - c.w0
    K = ["escala", "mix_setorial", "mix_regional", "intensidade", "populacao", "mix_regional_pop", "per_capita"]
    c["residuo"] = c.dW - c[K].sum(axis=1)
    return c


K7 = ["escala", "mix_setorial", "mix_regional", "intensidade", "populacao", "mix_regional_pop", "per_capita", "dW"]


def tot(c):
    return c[K7 + ["residuo"]].sum().round(1).to_dict()


# single aggregate deflator
d = pd.read_csv(R / "dados" / "deflatores_scn.csv").set_index("ano")
d = d / d.loc[2015]
q2 = q.copy()
for s in ("agro", "ind", "serv", "adm"):
    q2[f"g_{s}"] = q2[f"vab_{s}"] / q2.ano.map(d["va"]) / 1e3
q2["g_terc"] = q2.g_serv + q2.g_adm
q2["g_tot"] = q2[["g_agro", "g_ind", "g_serv", "g_adm"]].sum(axis=1)
P5a, P5b = (2010, 2011, 2012, 2013, 2014), (2017, 2018, 2019, 2020, 2021)
specs = {
    "Baseline (sectoral deflators, three-year ends, withdrawal)": lmdi7(q, P0, P1),
    "Single aggregate deflator (nominal structure)": lmdi7(q2, P0, P1),
    "Single-year ends, 2010 and 2021": lmdi7(q, (2010,), (2021,)),
    "Five-year ends, 2010–2014 and 2017–2021": lmdi7(q, P5a, P5b),
    "Thermoelectric withdrawal excluded": lmdi7(q, P0, P1, ind_col="ret_indx"),
    "Consumption instead of withdrawal": lmdi7(q, P0, P1, "con"),
    "Northeast states excluded": lmdi7(q[q.regiao != "NE"], P0, P1),
    "Subperiod 2010–2012 to 2014–2016": lmdi7(q, P0, PM),
    "Subperiod 2014–2016 to 2019–2021": lmdi7(q, PM, P1),
}
base = specs["Baseline (sectoral deflators, three-year ends, withdrawal)"]
base.to_csv(O / "rev17_lmdi7_celulas.csv", index=False)
tab = pd.DataFrame({k: tot(v) for k, v in specs.items()}).T
tab.index.name = "especificacao"
tab.to_csv(O / "rev17_lmdi7_especificacoes.csv")
out["lmdi7"] = {
    "nacional": tot(base),
    "por_setor": base.groupby("setor")[K7].sum().round(1).to_dict("index"),
    "por_uf": base.groupby("uf")[K7].sum().round(1).to_dict("index"),
    "especificacoes": {k: tot(v) for k, v in specs.items()},
}
# four-factor comparison (old table) for the record
old = A.lmdi(q, P0, P1, "ret")
out["lmdi4_nacional"] = old[["escala", "mix_setorial", "mix_regional", "intensidade", "dW"]].sum().round(1).to_dict()
# shares and intensity ratios for the text (productive sectors only)
br = q.groupby("ano").sum(numeric_only=True)
m = lambda s, P: s.loc[list(P)].mean()
out["share_withdrawal_agro"] = [m(br.ret_agro / br.ret_tot, P0), m(br.ret_agro / br.ret_tot, P1)]
out["share_gva_agro"] = [m(br.g_agro / br.g_tot, P0), m(br.g_agro / br.g_tot, P1)]
out["share_gva_ind"] = [m(br.g_ind / br.g_tot, P0), m(br.g_ind / br.g_tot, P1)]
out["intensity_ratio_agro_ind"] = [m((br.ret_agro / br.g_agro) / (br.ret_ind / br.g_ind), P0),
                                   m((br.ret_agro / br.g_agro) / (br.ret_ind / br.g_ind), P1)]
out["pop_growth"] = float(m(br["pop"], P1) / m(br["pop"], P0) - 1)
out["g_terc_growth"] = float(m(br.g_terc, P1) / m(br.g_terc, P0) - 1)
out["hum_pc_growth"] = float(m(br.ret_hum / br["pop"], P1) / m(br.ret_hum / br["pop"], P0) - 1)
# states where sector-mix / regional mix / intensity matter (productive block), and household effects by state
u = base.groupby("uf")[K7].sum()
out["lmdi7_uf_top"] = u.sort_values("dW", ascending=False).head(6).round(0).to_dict("index")
out["lmdi7_uf_bottom"] = u.sort_values("dW").head(4).round(0).to_dict("index")
sp = base[(base.uf == "SP")].set_index("setor")
out["SP_intensidade_por_setor"] = sp.intensidade.round(0).to_dict() | {"hh_per_capita": float(sp.loc["hh", "per_capita"].round(0)),
                                                                        "hh_pop": float(sp.loc["hh", "populacao"].round(0))}
mg = base[(base.uf == "MG")].set_index("setor")
out["MG_intensidade_por_setor"] = mg.intensidade.round(0).to_dict()

# ------------------------------------------------------------------ 2. Tapio sensitivity
LIM = 0.03
janelas = {
    "3-year means, 2010–12 → 2019–21 (baseline)": (P0, P1),
    "Single years, 2010 → 2021": ((2010,), (2021,)),
    "2-year means, 2010–11 → 2020–21": ((2010, 2011), (2020, 2021)),
    "4-year means, 2010–13 → 2018–21": ((2010, 2011, 2012, 2013), (2018, 2019, 2020, 2021)),
    "5-year means, 2010–14 → 2017–21": (P5a, P5b),
    "3-year means shifted, 2011–13 → 2018–20": ((2011, 2012, 2013), (2018, 2019, 2020)),
}
EN = {"desacoplamento forte": "SD", "desacoplamento fraco": "WD", "acoplamento expansivo": "EC",
      "desacoplamento negativo expansivo": "END", "desacoplamento negativo forte": "SND",
      "desacoplamento recessivo": "RD", "acoplamento recessivo": "RC", "desacoplamento negativo fraco": "WND"}
tp = A.tapio(q, janelas, "ret")
tp["classe_en"] = tp.classe.map(EN).fillna(tp.classe)
tp["dg_pequeno"] = tp.dg.abs() < LIM
tp.round(4).to_csv(O / "rev17_tapio_janelas_long.csv", index=False)
sens = {}
for s in ("total", "agropecuária"):
    w = tp[tp.setor == s].pivot(index="uf", columns="periodo", values="classe_en")
    w = w[list(janelas)]
    dgb = tp[(tp.setor == s) & (tp.periodo == list(janelas)[0])].set_index("uf")
    w.insert(0, "dG_base", dgb.dg.round(3))
    w.insert(1, "dW_base", dgb.dw.round(3))
    w["n_regimes"] = w[list(janelas)].nunique(axis=1)
    w["sinal_dW_estavel"] = tp[tp.setor == s].groupby("uf").dw.apply(lambda x: (np.sign(x) == np.sign(x.iloc[0])).all())
    w["indeterminado_|dG|<3%"] = w.dG_base.abs() < LIM
    w = w.sort_values("dG_base")
    w.to_csv(O / f"rev17_tapio_sensibilidade_{'total' if s == 'total' else 'agro'}.csv")
    ww = w.drop(index="BR")
    sens[s] = {
        "uf_|dG|<3%": ww[ww["indeterminado_|dG|<3%"]].index.tolist(),
        "n_uf_regime_estavel_em_todas_janelas": int((ww.n_regimes == 1).sum()),
        "n_uf_2_regimes": int((ww.n_regimes == 2).sum()),
        "n_uf_3+_regimes": int((ww.n_regimes >= 3).sum()),
        "uf_regime_muda": ww[ww.n_regimes > 1].n_regimes.to_dict(),
        "n_uf_sinal_dW_estavel": int(ww.sinal_dW_estavel.sum()),
        "uf_sinal_dW_muda": ww[~ww.sinal_dW_estavel].index.tolist(),
        "BR": w.loc["BR", list(janelas)].to_dict(),
        "BR_elasticidade": tp[(tp.setor == s) & (tp.uf == "BR")].set_index("periodo").elasticidade.round(2).to_dict(),
        "contagem_base_excluindo_indeterminados": ww[~ww["indeterminado_|dG|<3%"]][list(janelas)[0]].value_counts().to_dict(),
        "contagem_base": ww[list(janelas)[0]].value_counts().to_dict(),
    }
    # negative decoupling defined by signs only: water up and GVA down or growing less (END, SND) or both down (WND)
    b = ww[list(janelas)[0]]
    sens[s]["n_negative_decoupling_base(END+SND+WND)"] = int(b.isin(["END", "SND", "WND"]).sum())
    sens[s]["n_water_up_base"] = int((ww.dW_base > 0).sum())
    sens[s]["n_water_up_and_gva_flat_or_down(|dG|<3% or dG<0)"] = int(((ww.dW_base > 0) & ((ww.dG_base < 0) | (ww.dG_base.abs() < LIM))).sum())
out["tapio_sensibilidade"] = sens
# consumption-based classification for the same windows (national only)
tc = A.tapio(q, janelas, "con")
out["tapio_BR_consumo_janelas"] = tc[(tc.setor == "total") & (tc.uf == "BR")].set_index("periodo").elasticidade.round(2).to_dict()

# ------------------------------------------------------------------ 3. Panel elasticities, complements
import statsmodels.api as sm  # noqa: E402
import statsmodels.formula.api as smf  # noqa: E402

pairs = [("total", "g_tot", "ret_tot"), ("agropecuária", "g_agro", "ret_agro"), ("irrigação", "g_agro", "ret_ir"),
         ("indústria", "g_ind", "ret_indx"), ("abastecimento humano (per capita)", None, None)]
rng = np.random.default_rng(7)
B = 999
rows = []
for s, g, w in pairs:
    qq = q.copy()
    if g is None:
        qq = qq.assign(y=np.log(qq.ret_hum / qq["pop"]), x=np.log(qq.g_tot / qq["pop"]))
    else:
        qq = qq[qq[w] > 0].assign(y=lambda d: np.log(d[w]), x=lambda d: np.log(d[g]))
    for fe, lab in [("C(uf) + C(ano)", "UF + ano"), ("C(uf) + C(uf):ano", "UF + tendência por UF")]:
        mdl = smf.ols(f"y ~ x + {fe}", data=qq)
        mfit = mdl.fit(cov_type="cluster", cov_kwds={"groups": qq.uf})
        # within R2: R2 of the regression on state-demeaned y and x with the remaining controls demeaned too
        X = mdl.exog
        Xd = pd.DataFrame(X, columns=mdl.exog_names).assign(uf=qq.uf.values)
        yd = qq.y.values - qq.groupby("uf").y.transform("mean").values
        Xw = Xd.drop(columns=["uf"]).groupby(Xd.uf).transform(lambda c: c - c.mean()).values
        keep = Xw.std(axis=0) > 1e-12
        bw = np.linalg.lstsq(Xw[:, keep], yd, rcond=None)[0]
        r2w = 1 - ((yd - Xw[:, keep] @ bw) ** 2).sum() / ((yd - yd.mean()) ** 2).sum()
        # wild cluster bootstrap-t (Rademacher), H0: beta_x = 0, imposing the null
        t0 = mfit.params["x"] / mfit.bse["x"]
        mr = smf.ols(f"y ~ {fe}", data=qq).fit()
        e = mr.resid.values
        yhat = mr.fittedvalues.values
        ufs = qq.uf.values
        uniq = np.unique(ufs)
        tb = np.empty(B)
        for i in range(B):
            wgt = rng.choice([-1.0, 1.0], size=len(uniq))
            ys = yhat + e * wgt[np.searchsorted(uniq, ufs)]
            mb = sm.OLS(ys, X).fit(cov_type="cluster", cov_kwds={"groups": ufs})
            j = mdl.exog_names.index("x")
            tb[i] = mb.params[j] / mb.bse[j]
        pwild = float((np.abs(tb) >= abs(t0)).mean())
        rows.append({"setor": s, "efeitos": lab, "elasticidade": mfit.params["x"], "ep_cluster": mfit.bse["x"],
                     "ic95_inf": mfit.conf_int().loc["x", 0], "ic95_sup": mfit.conf_int().loc["x", 1],
                     "p_cluster": mfit.pvalues["x"], "p_wild_cluster_boot": pwild, "R2_within": r2w, "n": int(mfit.nobs)})
    # between estimator (state means) and long difference (3-year means at the ends)
    mu = qq.groupby("uf")[["x", "y"]].mean()
    mb = sm.OLS(mu.y, sm.add_constant(mu.x)).fit(cov_type="HC1")
    rows.append({"setor": s, "efeitos": "between (27 state means)", "elasticidade": mb.params["x"], "ep_cluster": mb.bse["x"],
                 "ic95_inf": mb.conf_int().loc["x", 0], "ic95_sup": mb.conf_int().loc["x", 1], "p_cluster": mb.pvalues["x"],
                 "p_wild_cluster_boot": np.nan, "R2_within": mb.rsquared, "n": int(mb.nobs)})
    a0 = qq[qq.ano.isin(P0)].groupby("uf")[["x", "y"]].mean()
    a1 = qq[qq.ano.isin(P1)].groupby("uf")[["x", "y"]].mean()
    dd = (a1 - a0).dropna()
    ml = sm.OLS(dd.y, sm.add_constant(dd.x)).fit(cov_type="HC1")
    rows.append({"setor": s, "efeitos": "long difference, 2010–12 → 2019–21 (27 states)", "elasticidade": ml.params["x"],
                 "ep_cluster": ml.bse["x"], "ic95_inf": ml.conf_int().loc["x", 0], "ic95_sup": ml.conf_int().loc["x", 1],
                 "p_cluster": ml.pvalues["x"], "p_wild_cluster_boot": np.nan, "R2_within": ml.rsquared, "n": int(ml.nobs)})
fe = pd.DataFrame(rows)
fe.round(4).to_csv(O / "rev17_elasticidades_painel_complementos.csv", index=False)
out["painel"] = fe.round(3).to_dict("records")
# within-state coefficient of variation of modelled water vs GVA (why precision is low)
cv = q.groupby("uf").agg(cv_w=("ret_tot", lambda s: s.std() / s.mean()), cv_g=("g_tot", lambda s: s.std() / s.mean()),
                         cv_ir=("ret_ir", lambda s: s.std() / s.mean()), cv_ga=("g_agro", lambda s: s.std() / s.mean()))
out["cv_within_uf_mediana"] = cv.median().round(3).to_dict()

# ------------------------------------------------------------------ 4. SDG 6.4.1: yield-ratio sensitivity
ir = pd.read_csv(R / "dados" / "irrigacao_uf_2017.csv")[["uf", "Ai"]]
sdg = {}
for ratio in (0.375, 0.543, 0.563):
    z = q.merge(ir, on="uf", suffixes=("", "_x"))
    Cr = 1 / (1 + z.Ai / ((1 - z.Ai) * ratio))
    awe = z.g_agro * (1 - Cr) * 1e3 / z.ret_agro
    W = z.ret_agro + z.ret_ind + z.ret_hum
    z["w641"] = (awe * z.ret_agro + z.wue_ind * z.ret_ind + z.wue_terc * z.ret_hum) / W
    nat = z.assign(n=z.w641 * W, dd=W).groupby("ano")[["n", "dd"]].sum()
    nat = nat.n / nat.dd
    st = A.media(z, P1).set_index("sigla").w641
    sdg[str(ratio)] = {"BR_2010_12": float(m(nat, P0)), "BR_2019_21": float(m(nat, P1)),
                       "BR_change_%": float((m(nat, P1) / m(nat, P0) - 1) * 100), "uf_2019_21": st.round(2).to_dict()}
ref = pd.Series(sdg["0.563"]["uf_2019_21"])
for k in ("0.375", "0.543"):
    diff = (pd.Series(sdg[k]["uf_2019_21"]) - ref)
    sdg[k]["max_abs_diff_uf_R$/m3"] = float(diff.abs().max())
    sdg[k]["uf_max_diff"] = str(diff.abs().idxmax())
out["sdg641_yield_ratio"] = sdg
pd.DataFrame({k: v["uf_2019_21"] for k, v in sdg.items()}).round(2).to_csv(O / "rev17_sdg641_razao_rendimento_uf.csv")

# ------------------------------------------------------------------ 5. Checks quoted in the text
a = pd.read_csv(R / "dados" / "ana_usos_uf.csv")
a = a[a.ano.isin(A.ANOS)]
soma = a[["ret_hu", "ret_hr", "ret_it", "ret_mn", "ret_te", "ret_an", "ret_ir"]].sum(axis=1)
dev = (soma / a.ret_tot - 1).abs()
chk = a[["uf", "ano"]].assign(soma_finalidades=soma, total_ANA=a.ret_tot, desvio_rel=soma / a.ret_tot - 1)
chk.round(6).to_csv(O / "rev17_checagem_soma_finalidades.csv", index=False)
out["soma_finalidades_desvio_max_%"] = float(dev.max() * 100)
from scipy.stats import spearmanr  # noqa: E402
u1 = pd.read_csv(O / "uf_2019_21.csv")
out["spearman_wp_vs_irr_share"] = float(spearmanr(u1.wp_ret, u1["irr_%"])[0])
out["spearman_wp_vs_irr_share_p"] = float(spearmanr(u1.wp_ret, u1["irr_%"])[1])
# MT: irrigation +70%/decade projection
mt = q[q.sigla == "MT"].set_index("ano")
ir0, ir1 = m(mt.ret_ir, P0), m(mt.ret_ir, P1)
g_ir = ir1 / ir0
w0, w1 = m(mt.ret_tot, P0), m(mt.ret_tot, P1)
ga = m(mt.g_agro, P1) / m(mt.g_agro, P0)
gt = m(mt.g_tot, P1) / m(mt.g_tot, P0)
out["MT_projecao"] = {"irrig_P0": float(ir0), "irrig_P1": float(ir1), "irrig_growth": float(g_ir - 1),
                      "ret_tot_P0": float(w0), "ret_tot_P1": float(w1), "ret_tot_growth": float(w1 / w0 - 1),
                      "g_agro_growth": float(ga - 1), "g_tot_growth": float(gt - 1),
                      "irrig_share_P1": float(ir1 / w1),
                      "share_irrig_after_1_decade_same_rate_other_uses_flat": float(ir1 * g_ir / (w1 - ir1 + ir1 * g_ir)),
                      "withdrawal_growth_next_decade_other_uses_flat": float((w1 - ir1 + ir1 * g_ir) / w1 - 1),
                      "withdrawal_growth_next_decade_other_uses_same_rate": float(((w1 - ir1) * ((w1 - ir1) / (w0 - ir0)) + ir1 * g_ir) / w1 - 1),
                      "ret_an_P1": float(m(mt.ret_an, P1)), "ret_an_growth": float(m(mt.ret_an, P1) / m(mt.ret_an, P0) - 1)}
# SDG vs WP difference by state (for the sentence in 3.4)
u1["diff_%"] = (1 - u1.wue_641 / u1.wp_ret) * 100
out["sdg_vs_wp_diff_%"] = u1.set_index("sigla")["diff_%"].round(1).sort_values(ascending=False).to_dict()

json.dump(out, open(O / "rev17_numeros.json", "w"), ensure_ascii=False, indent=1, default=float)
print(json.dumps(out, ensure_ascii=False, indent=1, default=float))
