"""Proposta 2 — produtividade hídrica, desacoplamento VAB–água, ODS 6.4.1 por UF (2010–2021).

Entradas (pasta ../):
  vab_uf.csv            SIDRA 5938: VAB corrente por UF (R$ milhões): total, agro, indústria, serviços (exc. adm), adm. pública
  ana_usos_uf.csv       ANA (2022), demanda hídrica municipal 1931–2040 agregada por UF (m³/s): retirada e consumo por finalidade
  deflatores_scn.csv    deflatores implícitos setoriais das Contas Nacionais Trimestrais (SIDRA 1846/6612)
  pop_raw.txt           estimativas de população (SIDRA 6579, mil hab.; 2010 interpolado)
Saídas: ../resultados/*.csv, ../figuras/*.png
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

R = Path(__file__).resolve().parents[1]
DD = R / "dados" if (R / "dados").exists() else R     # pasta com os CSVs de entrada
OUT = R / "resultados"
FIG = R / "figuras"
OUT.mkdir(exist_ok=True)
FIG.mkdir(exist_ok=True)

SIG = {11: "RO", 12: "AC", 13: "AM", 14: "RR", 15: "PA", 16: "AP", 17: "TO", 21: "MA", 22: "PI", 23: "CE", 24: "RN",
       25: "PB", 26: "PE", 27: "AL", 28: "SE", 29: "BA", 31: "MG", 32: "ES", 33: "RJ", 35: "SP", 41: "PR", 42: "SC",
       43: "RS", 50: "MS", 51: "MT", 52: "GO", 53: "DF"}
REG = {1: "N", 2: "NE", 3: "SE", 4: "S", 5: "CO"}
S2HM3 = 31.536            # m³/s → hm³/ano
ANOS = range(2010, 2022)
P0, P1 = (2010, 2011, 2012), (2019, 2020, 2021)    # médias trienais nas pontas (suaviza clima na irrigação)
PM = (2014, 2015, 2016)                            # meio: recessão 2015–16


# ------------------------------------------------------------------ painel
def painel():
    v = pd.read_csv(DD / "vab_uf.csv")
    a = pd.read_csv(DD / "ana_usos_uf.csv")
    d = pd.read_csv(DD / "deflatores_scn.csv").set_index("ano")
    d = d / d.loc[2015]                                # preços de 2015
    pop = []
    for l in open(DD / "pop_raw.txt"):
        u, x = l.strip().split("|")
        x = list(map(float, x.split()))
        anos = [2009] + list(range(2011, 2022))
        s = pd.Series(x, index=anos).reindex(range(2009, 2022)).interpolate()
        pop += [(int(u), y, s[y] * 1e3) for y in ANOS]
    pop = pd.DataFrame(pop, columns=["uf", "ano", "pop"])
    v = v.merge(pop, on=["uf", "ano"])
    v = v[v.ano.isin(ANOS)].copy()
    for s, c in [("agro", "agro"), ("ind", "ind"), ("serv", "serv"), ("adm", "adm")]:
        v[f"g_{s}"] = v[f"vab_{s}"] / v.ano.map(d[c]) / 1e3          # R$ bilhões de 2015
    v["g_tot"] = v[["g_agro", "g_ind", "g_serv", "g_adm"]].sum(axis=1)
    v["g_terc"] = v.g_serv + v.g_adm
    a = a[a.ano.isin(ANOS)].copy()
    for k in ("ret", "con"):
        for s in ("hu", "hr", "it", "mn", "te", "an", "ir", "tot"):
            a[f"{k}_{s}"] = a[f"{k}_{s}"] * S2HM3                       # hm³/ano
        a[f"{k}_agro"] = a[f"{k}_ir"] + a[f"{k}_an"]
        a[f"{k}_ind"] = a[f"{k}_it"] + a[f"{k}_mn"] + a[f"{k}_te"]
        a[f"{k}_indx"] = a[f"{k}_it"] + a[f"{k}_mn"]                   # sem termelétricas
        a[f"{k}_hum"] = a[f"{k}_hu"] + a[f"{k}_hr"]
        a[f"{k}_prod"] = a[f"{k}_agro"] + a[f"{k}_ind"]
    p = v.merge(a, on=["uf", "ano"])
    p["sigla"] = p.uf.map(SIG)
    p["regiao"] = (p.uf // 10).map(REG)
    return p


# ------------------------------------------------------------------ indicadores
def tapio_classe(dg, dw):
    """Tapio (2005): oito estados a partir das variações relativas de VAB (dg) e água (dw)."""
    if dg == 0:
        return "indefinido"
    e = dw / dg
    if dg > 0:
        if dw < 0:
            return "desacoplamento forte"
        if e < 0.8:
            return "desacoplamento fraco"
        if e <= 1.2:
            return "acoplamento expansivo"
        return "desacoplamento negativo expansivo"
    if dw > 0:
        return "desacoplamento negativo forte"
    if e > 1.2:
        return "desacoplamento recessivo"
    if e >= 0.8:
        return "acoplamento recessivo"
    return "desacoplamento negativo fraco"


def media(p, anos):
    return p[p.ano.isin(anos)].groupby(["uf", "sigla", "regiao"]).mean(numeric_only=True).reset_index()


def tapio(p, pares, w="ret", alvos=None):
    alvos = alvos or {"total": ("g_tot", f"{w}_tot"), "agropecuária": ("g_agro", f"{w}_agro"),
                      "indústria": ("g_ind", f"{w}_ind"), "indústria s/ termelétricas": ("g_ind", f"{w}_indx")}
    linhas = []
    for nome, (a0, a1) in pares.items():
        m0, m1 = media(p, a0), media(p, a1)
        br0, br1 = m0.sum(numeric_only=True), m1.sum(numeric_only=True)
        for s, (g, x) in alvos.items():
            for (_, r0), (_, r1) in list(zip(m0.iterrows(), m1.iterrows())) + [((None, br0), (None, br1))]:
                dg = r1[g] / r0[g] - 1
                dw = r1[x] / r0[x] - 1 if r0[x] > 0 else np.nan
                linhas.append({"periodo": nome, "setor": s, "uf": r0.get("sigla", "BR") if r0 is not br0 else "BR",
                               "dg": dg, "dw": dw, "elasticidade": dw / dg if dg else np.nan,
                               "classe": tapio_classe(dg, dw) if np.isfinite(dw) else "sem uso"})
    return pd.DataFrame(linhas)


def _L(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.where(np.isclose(a, b), a, (a - b) / (np.log(a) - np.log(b)))
    return np.where((a > 0) & (b > 0), out, 0.0)


def lmdi(p, a0, a1, w="ret", setores=(("agro", "g_agro"), ("ind", "g_ind"), ("terc", "g_terc"))):
    """LMDI-I aditivo (Ang 2004/2015). W = Σ_{r,s} G · (G_rs/G) · (W_rs/G_rs).
    Efeitos: escala (G nacional), estrutura (participação UF×setor no VAB nacional), intensidade.
    A demanda humana (urbana + rural) entra no setor terciário por convenção (abastecimento público)."""
    m0, m1 = media(p, a0), media(p, a1)
    wmap = {"agro": f"{w}_agro", "ind": f"{w}_ind", "terc": f"{w}_hum"}
    cel = []
    for s, g in setores:
        for (_, r0), (_, r1) in zip(m0.iterrows(), m1.iterrows()):
            cel.append((r0.sigla, s, r0[g], r1[g], r0[wmap[s]], r1[wmap[s]]))
    c = pd.DataFrame(cel, columns=["uf", "setor", "g0", "g1", "w0", "w1"])
    G0, G1 = c.g0.sum(), c.g1.sum()
    Lw = _L(c.w1, c.w0)
    S0, S1 = c.groupby("setor").g0.transform("sum"), c.groupby("setor").g1.transform("sum")
    c["escala"] = Lw * np.log(G1 / G0)
    c["estrutura"] = Lw * np.log((c.g1 / G1) / (c.g0 / G0))
    c["mix_setorial"] = Lw * np.log((S1 / G1) / (S0 / G0))          # estrutura = mix setorial + mix regional
    c["mix_regional"] = Lw * np.log((c.g1 / S1) / (c.g0 / S0))
    c["intensidade"] = Lw * np.log((c.w1 / c.g1) / (c.w0 / c.g0))
    c["dW"] = c.w1 - c.w0
    c["residuo"] = c.dW - c[["escala", "estrutura", "intensidade"]].sum(axis=1)
    return c


def wue_641(p):
    """ODS 6.4.1 (FAO 2019; UN-Water 2019): WUE = Awe·PA + Mwe·PM + Swe·PS, com Awe = VAB_agro·(1 − Cr)/V_agro e
    Cr = 1/(1 + Ai/((1 − Ai)·0,563)) (fração do VAB agrícola de sequeiro; 0,563 = razão padrão entre produtividades
    de sequeiro e irrigada). Ai por UF = área irrigada (Censo Agro 2017, SIDRA 6857) / [irrigada + área de lavouras
    (PAM 2017, SIDRA 1612+1613)], fixo no tempo. Serviços: VAB terciário / abastecimento humano (urbano + rural)."""
    q = p.copy()
    ir = pd.read_csv(DD / "irrigacao_uf_2017.csv")[["uf", "Ai", "Cr"]]
    q = q.merge(ir, on="uf")
    q["wue_agro"] = q.g_agro * 1e3 / q.ret_agro          # R$ de 2015 por m³ (g em bi R$, w em hm³ → R$/m³ ×1e3)
    q["awe"] = q.g_agro * (1 - q.Cr) * 1e3 / q.ret_agro  # só a parte irrigada do VAB agro (definição ODS)
    q["wue_ind"] = q.g_ind * 1e3 / q.ret_ind
    q["wue_terc"] = q.g_terc * 1e3 / q.ret_hum
    W = q.ret_agro + q.ret_ind + q.ret_hum
    q["wue_641"] = (q.awe * q.ret_agro + q.wue_ind * q.ret_ind + q.wue_terc * q.ret_hum) / W
    q["wp_ret"] = q.g_tot * 1e3 / q.ret_tot
    q["wp_con"] = q.g_tot * 1e3 / q.con_tot
    q["wp_ret_sem_te"] = q.g_tot * 1e3 / (q.ret_tot - q.ret_te)
    return q


def painel_fe(p):
    """Elasticidades dentro da UF: log W_s = β log G_s + α_UF + γ_ano (EP agrupado por UF)."""
    import statsmodels.formula.api as smf
    linhas = []
    for s, g, w in [("total", "g_tot", "ret_tot"), ("agropecuária", "g_agro", "ret_agro"),
                    ("irrigação", "g_agro", "ret_ir"), ("indústria", "g_ind", "ret_indx"),
                    ("abastecimento humano (per capita)", None, None)]:
        q = p.copy()
        if g is None:
            q = q.assign(y=np.log(q.ret_hum / q["pop"]), x=np.log(q.g_tot / q["pop"]))
        else:
            q = q[q[w] > 0].assign(y=lambda d: np.log(d[w]), x=lambda d: np.log(d[g]))
        for fe in ["C(uf) + C(ano)", "C(uf) + C(uf):ano"]:
            m = smf.ols(f"y ~ x + {fe}", data=q).fit(cov_type="cluster", cov_kwds={"groups": q.uf})
            linhas.append({"setor": s, "efeitos": "UF + ano" if "C(ano)" in fe else "UF + tendência por UF",
                           "elasticidade": m.params["x"], "ep": m.bse["x"],
                           "ic95_inf": m.conf_int().loc["x", 0], "ic95_sup": m.conf_int().loc["x", 1], "n": int(m.nobs)})
    return pd.DataFrame(linhas)


def indice_pca(q, b=2000, semente=42):
    """Índice composto de desempenho hídrico-econômico por UF (2019–21) com pesos objetivos (PCA)
    e incerteza das posições por bootstrap das UFs (Senna et al. 2019; OCDE/JRC 2008)."""
    m1, m0 = media(q, P1), media(q, P0)
    X = pd.DataFrame({
        "log_wp": np.log(m1.wp_ret),                                   # + produtividade da retirada
        "log_wue_agro": np.log(m1.wue_agro),                           # + produtividade agro
        "fracao_consumo": -(m1.con_tot / m1.ret_tot),                  # − fração consumida (menos retorno)
        "ret_pc": -np.log(m1.ret_hum / m1["pop"]),                     # − retirada humana per capita
        "ganho_wp": np.log(m1.wp_ret / m0.wp_ret.values) / 9,          # + taxa anual de ganho de produtividade
    })                                                                 # (o "desacoplamento" = ganho_wp×9: não entra)
    X.index = m1.sigla
    Z = (X - X.mean()) / X.std(ddof=1)

    try:
        from gwbench.contas import _pesos_pca as pesos
    except ImportError:                                               # gwbench fora do caminho: usa a cópia local
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from gwbench.contas import _pesos_pca as pesos                    # varimax + atribuição (Nicoletti et al. 2000)

    w, ev, k = pesos(Z)
    idx = Z.values @ w
    rng = np.random.default_rng(semente)
    ranks = np.zeros((b, len(Z)))
    W = np.zeros((b, Z.shape[1]))
    for i in range(b):
        s = rng.integers(0, len(Z), len(Z))
        Zb = Z.iloc[s]
        Zb = (Zb - Zb.mean()) / Zb.std(ddof=1)
        wb, _, _ = pesos(Zb.fillna(0))
        W[i] = wb
        ib = ((X - X.iloc[s].mean()) / X.iloc[s].std(ddof=1)).values @ wb
        ranks[i] = (-ib).argsort().argsort() + 1
    iguais = Z.values.mean(axis=1)
    from scipy.stats import spearmanr
    res = pd.DataFrame({"indice": idx, "posicao": (-idx).argsort().argsort() + 1,
                        "pos_p05": np.percentile(ranks, 5, axis=0), "pos_p50": np.median(ranks, axis=0),
                        "pos_p95": np.percentile(ranks, 95, axis=0), "indice_pesos_iguais": iguais}, index=Z.index)
    meta = {"autovalores": ev.round(3).tolist(), "componentes_retidos": k,
            "pesos": dict(zip(Z.columns, w.round(3))), "pesos_bootstrap_p05": dict(zip(Z.columns, np.percentile(W, 5, 0).round(3))),
            "pesos_bootstrap_p95": dict(zip(Z.columns, np.percentile(W, 95, 0).round(3))),
            "spearman_pca_vs_iguais": float(spearmanr(idx, iguais)[0])}
    return res.sort_values("posicao"), X, meta


# ------------------------------------------------------------------ execução
def main():
    p = painel()
    q = wue_641(p)
    q.to_csv(OUT / "painel_uf_ano.csv", index=False)
    br = q.groupby("ano").sum(numeric_only=True)
    br_ind = pd.DataFrame({
        "VAB_bi_R$2015": br.g_tot, "retirada_hm3": br.ret_tot, "consumo_hm3": br.con_tot,
        "WP_ret_R$/m3": br.g_tot * 1e3 / br.ret_tot, "WP_con_R$/m3": br.g_tot * 1e3 / br.con_tot,
        "WUE_6.4.1_R$/m3": q.assign(n=q.wue_641 * (q.ret_agro + q.ret_ind + q.ret_hum)).groupby("ano").n.sum()
        / (br.ret_agro + br.ret_ind + br.ret_hum),
        "irrigacao_%ret": 100 * br.ret_ir / br.ret_tot, "termeletrica_%ret": 100 * br.ret_te / br.ret_tot})
    br_ind.round(3).to_csv(OUT / "brasil_ano.csv")
    pares = {"2010–12 → 2019–21": (P0, P1), "2010–12 → 2014–16": (P0, PM), "2014–16 → 2019–21": (PM, P1)}
    t = tapio(q, pares, "ret")
    t.to_csv(OUT / "tapio_retirada.csv", index=False)
    tapio(q, pares, "con").to_csv(OUT / "tapio_consumo.csv", index=False)
    c = lmdi(q, P0, P1, "ret")
    c.to_csv(OUT / "lmdi_retirada_celulas.csv", index=False)
    K = ["escala", "mix_setorial", "mix_regional", "intensidade", "dW"]
    lm = {"nacional": c[["escala", "estrutura", "mix_setorial", "mix_regional", "intensidade", "dW", "residuo"]].sum().round(2).to_dict(),
          "por_setor": c.groupby("setor")[K].sum().round(2).to_dict("index"),
          "por_uf": c.groupby("uf")[K].sum().round(2).to_dict("index")}
    c2 = lmdi(q, P0, P1, "con")
    lm["consumo_nacional"] = c2[["escala", "mix_setorial", "mix_regional", "intensidade", "dW", "residuo"]].sum().round(2).to_dict()
    lm["consumo_por_setor"] = c2.groupby("setor")[K].sum().round(2).to_dict("index")
    c3 = lmdi(q[q.ano.isin(P0 + PM)], P0, PM, "ret")
    c4 = lmdi(q[q.ano.isin(PM + P1)], PM, P1, "ret")
    lm["subperiodo_2010_12_a_2014_16"] = c3[K].sum().round(2).to_dict()
    lm["subperiodo_2014_16_a_2019_21"] = c4[K].sum().round(2).to_dict()
    json.dump(lm, open(OUT / "lmdi_resumo.json", "w"), ensure_ascii=False, indent=1)
    fe = painel_fe(q)
    fe.round(3).to_csv(OUT / "elasticidades_painel.csv", index=False)
    ix, X, meta = indice_pca(q)
    ix.round(3).to_csv(OUT / "indice_pca_uf.csv")
    X.round(4).to_csv(OUT / "indice_pca_indicadores.csv")
    json.dump(meta, open(OUT / "indice_pca_meta.json", "w"), ensure_ascii=False, indent=1)
    uf = media(q, P1)[["sigla", "regiao", "g_tot", "ret_tot", "con_tot", "wp_ret", "wp_con", "wue_agro", "wue_ind",
                        "wue_terc", "wue_641", "ret_ir", "ret_te"]]
    uf["irr_%"] = 100 * uf.ret_ir / uf.ret_tot
    uf.round(3).sort_values("wue_641").to_csv(OUT / "uf_2019_21.csv", index=False)
    print(br_ind.round(2).to_string())
    print(t[(t.periodo == "2010–12 → 2019–21")].groupby(["setor", "classe"]).size().to_string())
    print(json.dumps(lm["nacional"]), json.dumps(lm["por_setor"]))
    print(fe.round(3).to_string())
    print(ix.round(2).to_string())
    print(json.dumps(meta, ensure_ascii=False))


if __name__ == "__main__":
    main()
