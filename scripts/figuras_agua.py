"""Figures for the water-accounts manuscript (English labels)."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

R = Path(__file__).resolve().parents[1]
O, F = R / "resultados", R / "figuras"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
REGC = {"N": "#1b9e77", "NE": "#d95f02", "SE": "#7570b3", "S": "#e7298a", "CO": "#66a61e"}

q = pd.read_csv(O / "painel_uf_ano.csv")
br = q.groupby("ano").sum(numeric_only=True)

# ---------------------------------------------------------------- Fig 1: national trajectories
fig, ax = plt.subplots(1, 2, figsize=(7.2, 3.6))
idx = lambda s: 100 * s / s.iloc[0]
a = ax[0]
a.plot(br.index, idx(br.g_tot), "k-", lw=2, label="Real GVA")
a.plot(br.index, idx(br.g_agro), "-", c="#66a61e", label="Real GVA, agriculture")
a.plot(br.index, idx(br.ret_tot), "-", c="#2166ac", lw=2, label="Withdrawal, total")
a.plot(br.index, idx(br.ret_ir), "--", c="#2166ac", label="Withdrawal, irrigation")
a.plot(br.index, idx(br.g_tot / br.ret_tot), "-", c="#b2182b", lw=2, label="Water productivity")
a.axhline(100, c="0.7", lw=0.6)
a.set_ylabel("Index (2010 = 100)")
a.set_title("a", loc="left", fontweight="bold")
a.legend(frameon=False, fontsize=7, loc="upper left")
a = ax[1]
sh = pd.DataFrame({"Irrigation": br.ret_ir, "Livestock": br.ret_an, "Manufacturing + mining": br.ret_it + br.ret_mn,
                   "Thermoelectric": br.ret_te, "Urban supply": br.ret_hu, "Rural supply": br.ret_hr})
sh = sh.div(sh.sum(axis=1), axis=0) * 100
cols = ["#2166ac", "#92c5de", "#878787", "#4d4d4d", "#f4a582", "#fddbc7"]
a.stackplot(sh.index, sh.T.values, labels=sh.columns, colors=cols)
a.set_ylim(0, 100)
a.set_ylabel("Share of withdrawal (%)")
a.set_title("b", loc="left", fontweight="bold")
a.legend(frameon=False, fontsize=6.5, loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=3)  # rev. 17/10: outside the plot
fig.tight_layout()
fig.savefig(F / "fig1_national.png", dpi=300)
plt.close(fig)

# ---------------------------------------------------------------- Fig 2: Tapio scatter
t = pd.read_csv(O / "tapio_retirada.csv")
t = t[t.periodo == "2010–12 → 2019–21"]
reg = q.drop_duplicates("sigla").set_index("sigla").regiao
fig, ax = plt.subplots(1, 2, figsize=(7.2, 3.4))
for a, (s, lab) in zip(ax, [("total", "a  All sectors"), ("agropecuária", "b  Agriculture (irrigation + livestock)")]):
    d = t[(t.setor == s)]
    lim = max(0.05, np.nanmax(np.abs(d[["dg", "dw"]].values)) * 1.08)
    lim = min(lim, 1.2)
    x = np.linspace(0, lim, 10)
    a.fill_between(x, 0, 0.8 * x, color="#d9f0d3", lw=0, zorder=0)
    a.fill_between(x, 0, -lim, color="#a6dba0", lw=0, zorder=0)
    a.plot(x, 0.8 * x, c="0.6", lw=0.6)
    a.plot(x, 1.2 * x, c="0.6", lw=0.6)
    a.axhline(0, c="0.4", lw=0.6)
    a.axvline(0, c="0.4", lw=0.6)
    txt = []
    for _, r in d.iterrows():
        if r.uf == "BR":
            a.plot(r.dg, r.dw, "k*", ms=11, zorder=5)
            a.annotate("Brazil", (r.dg, r.dw), xytext=(4, 4), textcoords="offset points", fontsize=7, fontweight="bold")
            continue
        c = REGC[reg[r.uf]]
        if abs(r.dg) < 0.03:   # rev. 17/10: |dGVA| < 3% -> elasticity not meaningful, open marker
            a.plot(np.clip(r.dg, -lim, lim), np.clip(r.dw, -lim, lim), "o", mfc="w", mec=c, ms=4.5, mew=1.0, zorder=4)
        else:
            a.plot(np.clip(r.dg, -lim, lim), np.clip(r.dw, -lim, lim), "o", c=c, ms=4.5, mec="w", mew=0.4, zorder=4)
        cl = abs(r.dg) > lim or abs(r.dw) > lim
        txt.append(a.text(np.clip(r.dg, -lim, lim), np.clip(r.dw, -lim, lim), r.uf + ("*" if cl else ""), fontsize=6))
    from adjustText import adjust_text
    a.axvspan(-0.03, 0.03, color="0.92", lw=0, zorder=0)   # rev. 17/10: indeterminate band
    adjust_text(txt, ax=a, expand=(1.4, 1.6), force_text=(0.4, 0.6), arrowprops=dict(arrowstyle="-", color="0.6", lw=0.4))
    a.set_xlim(-lim * 0.45, lim)
    a.set_ylim(-lim * 0.45, lim)
    a.set_xlabel("Change in real GVA, 2010–12 to 2019–21")
    a.set_ylabel("Change in water withdrawal")
    a.set_title(lab, loc="left", fontweight="bold", fontsize=9)
    a.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    a.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
h = [plt.Line2D([], [], marker="o", ls="", c=c, label=k) for k, c in REGC.items()]
ax[1].legend(handles=h, frameon=False, fontsize=7, loc="upper left", title="Region", title_fontsize=7)
# rev. 17/10: in-panel note removed (overlapped PB); the caption explains shading, open dots and asterisks
fig.tight_layout()
fig.savefig(F / "fig2_tapio.png", dpi=300)
plt.close(fig)

# ---------------------------------------------------------------- Fig 3: LMDI
# rev. 17/10: seven-factor LMDI (households = population x per-capita supply); population redistribution among
# states (<= 0.05 km3/yr) is added to the population effect
lm7 = json.load(open(O / "rev17_numeros.json"))["lmdi7"]
def _v(d):
    d = dict(d)
    d["populacao"] = d.get("populacao", 0) + d.get("mix_regional_pop", 0)
    return d
lm = {"nacional": _v(lm7["especificacoes"]["Baseline (sectoral deflators, three-year ends, withdrawal)"]),
      "subperiodo_2010_12_a_2014_16": _v(lm7["especificacoes"]["Subperiod 2010–2012 to 2014–2016"]),
      "subperiodo_2014_16_a_2019_21": _v(lm7["especificacoes"]["Subperiod 2014–2016 to 2019–2021"]),
      "por_setor": {s: _v(v) for s, v in lm7["por_setor"].items()}}
fig, ax = plt.subplots(1, 2, figsize=(7.2, 3.4), gridspec_kw={"width_ratios": [1.15, 1]}, sharey=True)
a = ax[0]
K = ["escala", "mix_setorial", "mix_regional", "intensidade", "populacao", "per_capita"]
lab = ["Scale", "Sector mix", "Regional mix", "Intensity (agric., industry)", "Population", "Per-capita supply"]
cc = ["#878787", "#66a61e", "#e6ab02", "#2166ac", "#d95f02", "#fdb863"]
per = [("nacional", "2010–12 →\n2019–21"), ("subperiodo_2010_12_a_2014_16", "2010–12 →\n2014–16"),
       ("subperiodo_2014_16_a_2019_21", "2014–16 →\n2019–21")]
for i, (k, nm) in enumerate(per):
    v = lm[k]
    pos = neg = 0
    for j, f in enumerate(K):
        x = v[f] / 1e3
        if x >= 0:
            a.bar(i, x, bottom=pos, color=cc[j], width=0.6, label=lab[j] if i == 0 else None)
            pos += x
        else:
            a.bar(i, x, bottom=neg, color=cc[j], width=0.6, label=lab[j] if i == 0 else None)
            neg += x
    a.plot(i, v["dW"] / 1e3, "kD", ms=5, zorder=5, label="Net change" if i == 0 else None)
a.axhline(0, c="k", lw=0.6)
a.set_xticks(range(3), [p[1] for p in per], fontsize=7.5)
a.set_ylabel("Change in withdrawal (km³ yr⁻¹)")
a.set_title("a", loc="left", fontweight="bold")
hh, ll = a.get_legend_handles_labels()
a = ax[1]
ps = lm["por_setor"]
sec = [("agro", "Agriculture"), ("ind", "Industry"), ("hh", "Households\n(human supply)")]
for i, (s, nm) in enumerate(sec):
    pos = neg = 0
    for j, f in enumerate(K):
        x = ps[s][f] / 1e3
        if x >= 0:
            a.bar(i, x, bottom=pos, color=cc[j], width=0.6)
            pos += x
        else:
            a.bar(i, x, bottom=neg, color=cc[j], width=0.6)
            neg += x
    a.plot(i, ps[s]["dW"] / 1e3, "kD", ms=5, zorder=5)
a.axhline(0, c="k", lw=0.6)
a.set_xticks(range(3), [s[1] for s in sec], fontsize=7.5)
a.set_title("b", loc="left", fontweight="bold")
a.legend(hh, ll, frameon=False, fontsize=6.5, loc="upper right")
fig.tight_layout()
fig.savefig(F / "fig3_lmdi.png", dpi=300)
plt.close(fig)

# ---------------------------------------------------------------- Fig 4: WUE (SDG 6.4.1) by state
u = pd.read_csv(O / "uf_2019_21.csv").sort_values("wue_641")
fig, ax = plt.subplots(1, 2, figsize=(7.2, 4.2), gridspec_kw={"width_ratios": [1.3, 1]})
a = ax[0]
y = np.arange(len(u))
a.barh(y, u.wue_641, color=[REGC[r] for r in u.regiao], height=0.7)
a.set_xscale("log")
a.set_xticks([25, 50, 100, 200, 400], ["25", "50", "100", "200", "400"])
a.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
a.set_yticks(y, u.sigla, fontsize=7)
a.set_xlabel("Water-use efficiency, SDG 6.4.1 (R$ of 2015 m⁻³)")
a.set_title("a", loc="left", fontweight="bold")
nat = (q[q.ano.isin([2019, 2020, 2021])].assign(n=lambda d: d.wue_641 * (d.ret_agro + d.ret_ind + d.ret_hum))
       .pipe(lambda d: d.n.sum() / (d.ret_agro + d.ret_ind + d.ret_hum).sum()))
a.axvline(nat, c="k", ls="--", lw=0.8)
a.text(nat * 1.05, 0.5, f"Brazil {nat:.0f}", fontsize=7)
a = ax[1]
tx = []
for _, r in u.iterrows():
    a.plot(r["irr_%"], r.wp_ret, "o", c=REGC[r.regiao], ms=5, mec="w", mew=0.4)
    tx.append(a.text(r["irr_%"], r.wp_ret, r.sigla, fontsize=6))
a.set_yscale("log")
a.set_yticks([25, 50, 100, 200, 400], ["25", "50", "100", "200", "400"])
a.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
from adjustText import adjust_text
adjust_text(tx, ax=a, arrowprops=dict(arrowstyle="-", color="0.6", lw=0.4))
a.set_xlabel("Irrigation share of withdrawal (%)")
a.set_ylabel("Water productivity (R$ of 2015 m⁻³)")
a.set_title("b", loc="left", fontweight="bold")
h = [plt.Line2D([], [], marker="s", ls="", c=c, label=k) for k, c in REGC.items()]
ax[0].legend(handles=h, frameon=False, fontsize=7, loc="lower right")
fig.tight_layout()
fig.savefig(F / "fig4_wue_states.png", dpi=300)
plt.close(fig)

# ---------------------------------------------------------------- Fig 5: composite index
ix = pd.read_csv(O / "indice_pca_uf.csv").sort_values("posicao", ascending=False)
fig, a = plt.subplots(figsize=(4.2, 4.6))
y = np.arange(len(ix))
a.hlines(y, ix.pos_p05, ix.pos_p95, color="0.6", lw=2.5)
a.plot(ix.posicao, y, "o", c="k", ms=4, label="PCA weights")
iq = (-ix.indice_pesos_iguais).rank().astype(int)
a.plot(iq, y, "x", c="#d95f02", ms=4, label="Equal weights")
a.set_yticks(y, ix.sigla, fontsize=7)
a.set_xlabel("Rank (1 = best)")
a.invert_xaxis()
a.legend(frameon=False, fontsize=7, loc="lower right")
fig.tight_layout()
fig.savefig(F / "fig5_index_ranks.png", dpi=300)
plt.close(fig)
print("ok")
