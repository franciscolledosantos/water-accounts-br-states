"""Robustness checks and numbers quoted in the text."""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).parent))
import analise_agua as A
R = Path(__file__).resolve().parents[1]; O = R / "resultados"
p = A.painel(); q = A.wue_641(p)
out = {}
P0, P1, PM = A.P0, A.P1, A.PM
br = q.groupby("ano").sum(numeric_only=True)
m = lambda s, P: s.loc[list(P)].mean()
out["WP_ret"] = [m(br.g_tot * 1e3 / br.ret_tot, P0), m(br.g_tot * 1e3 / br.ret_tot, P1)]
out["WP_con"] = [m(br.g_tot * 1e3 / br.con_tot, P0), m(br.g_tot * 1e3 / br.con_tot, P1)]
w = q.assign(n=q.wue_641 * (q.ret_agro + q.ret_ind + q.ret_hum), d=q.ret_agro + q.ret_ind + q.ret_hum).groupby("ano")[["n", "d"]].sum()
out["WUE641"] = [m(w.n / w.d, P0), m(w.n / w.d, P1)]
out["WP_agro"] = [m(br.g_agro * 1e3 / br.ret_agro, P0), m(br.g_agro * 1e3 / br.ret_agro, P1)]
out["WP_ind"] = [m(br.g_ind * 1e3 / br.ret_ind, P0), m(br.g_ind * 1e3 / br.ret_ind, P1)]
out["WP_terc"] = [m(br.g_terc * 1e3 / br.ret_hum, P0), m(br.g_terc * 1e3 / br.ret_hum, P1)]
out["ret_hum_pc_L_dia"] = [m(br.ret_hum * 1e9 / br["pop"] / 365, P0), m(br.ret_hum * 1e9 / br["pop"] / 365, P1)]
t = pd.read_csv(O / "tapio_retirada.csv")
tt = t[(t.uf != "BR")]
out["tapio_contagem"] = {f"{a}|{b}|{c}": int(n) for (a, b, c), n in tt.groupby(["periodo", "setor"]).classe.value_counts().items()}
out["tapio_BR"] = t[t.uf == "BR"][["periodo", "setor", "dg", "dw", "elasticidade", "classe"]].round(3).to_dict("records")
tc = pd.read_csv(O / "tapio_consumo.csv")
out["tapio_BR_consumo"] = tc[tc.uf == "BR"][["periodo", "setor", "dg", "dw", "elasticidade", "classe"]].round(3).to_dict("records")
# LMDI variants
def lm(qq, a0, a1, w="ret"):
    c = A.lmdi(qq, a0, a1, w)
    return c[["escala", "mix_setorial", "mix_regional", "intensidade", "dW"]].sum().round(0).to_dict()
out["lmdi_base"] = lm(q, P0, P1)
# (a) aggregate deflator for all sectors (structure measured in nominal shares)
d = pd.read_csv((R / "dados" if (R / "dados").exists() else R) / "deflatores_scn.csv").set_index("ano"); d = d / d.loc[2015]
q2 = q.copy()
for s, c in [("agro", "agro"), ("ind", "ind"), ("serv", "serv"), ("adm", "adm")]:
    q2[f"g_{s}"] = q2[f"vab_{s}"] / q2.ano.map(d["va"]) / 1e3
q2["g_terc"] = q2.g_serv + q2.g_adm; q2["g_tot"] = q2[["g_agro", "g_ind", "g_serv", "g_adm"]].sum(axis=1)
out["lmdi_deflator_unico"] = lm(q2, P0, P1)
# (b) single-year endpoints
out["lmdi_2010_2021"] = lm(q, (2010,), (2021,))
# (c) without thermoelectric
q3 = q.copy(); q3["ret_ind"] = q3.ret_indx
out["lmdi_sem_termeletricas"] = lm(q3, P0, P1)
# (d) consumption
out["lmdi_consumo"] = lm(q, P0, P1, "con")
# (e) excluding NE states hit by the 2012-17 drought
ne = q[q.regiao != "NE"]
out["lmdi_sem_NE"] = lm(ne, P0, P1)
# real GVA shares
out["share_agro_real"] = [m(br.g_agro / br.g_tot, P0), m(br.g_agro / br.g_tot, P1)]
out["share_agro_nominal"] = [m(br.vab_agro / br.vab_total, P0), m(br.vab_agro / br.vab_total, P1)]
out["share_ind_real"] = [m(br.g_ind / br.g_tot, P0), m(br.g_ind / br.g_tot, P1)]
# by-state LMDI top contributors
c = A.lmdi(q, P0, P1, "ret")
u = c.groupby("uf")[["escala", "mix_setorial", "mix_regional", "intensidade", "dW"]].sum()
out["lmdi_uf_top_dW"] = u.sort_values("dW", ascending=False).head(6).round(0).to_dict("index")
out["lmdi_uf_bottom_dW"] = u.sort_values("dW").head(5).round(0).to_dict("index")
# MT case
mt = q[q.sigla == "MT"].set_index("ano")
out["MT"] = {k: [round(m(mt[k], P0), 2), round(m(mt[k], P1), 2)] for k in ["g_tot", "g_agro", "ret_tot", "ret_ir", "ret_an", "ret_agro", "con_tot", "wp_ret", "wue_agro", "wue_641"]}
out["MT"]["share_agro_vab"] = [m(mt.g_agro / mt.g_tot, P0), m(mt.g_agro / mt.g_tot, P1)]
out["MT"]["share_BR_agro_vab"] = [m(mt.g_agro / br.g_agro, P0), m(mt.g_agro / br.g_agro, P1)]
out["MT"]["share_BR_irrig"] = [m(mt.ret_ir / br.ret_ir, P0), m(mt.ret_ir / br.ret_ir, P1)]
# irrigation contributions by state (change in hm3)
ir = q.groupby("sigla").apply(lambda g: m(g.set_index("ano").ret_ir, P1) - m(g.set_index("ano").ret_ir, P0), include_groups=False)
out["delta_irrig_top"] = ir.sort_values(ascending=False).head(6).round(0).to_dict()
out["delta_irrig_total"] = float(ir.sum())
ix = pd.read_csv(O / "indice_pca_uf.csv")
out["indice_largura_media_IC90"] = float((ix.pos_p95 - ix.pos_p05).mean())
json.dump(out, open(O / "numeros_texto.json", "w"), ensure_ascii=False, indent=1, default=float)
print(json.dumps(out, ensure_ascii=False, default=float)[:6000])
