"""Contas econômico-ambientais da água: produtividade, ODS 6.4.1, desacoplamento de Tapio, decomposição LMDI
e índice composto por PCA com incerteza por bootstrap.

Entrou na bancada pela regra do projeto (todo método usado ou sugerido num artigo vai para o gwbench e para o
material didático). Origem: artigo "Subnational water accounts" (proposta 2; UF × setor × ano, 2010–2021).
Não é aprendizado de máquina; são as ferramentas de contabilidade que acompanham os modelos de demanda.

Funções (todas recebem DataFrames/arrays simples, sem nomes de coluna fixos):
  tapio(dg, dw)                    → elasticidade e classe de Tapio (2005) a partir das variações relativas
  tapio_tabela(df, unidade, g, w, a0, a1) → Tapio por unidade entre médias de dois conjuntos de anos
  lmdi_aditivo(w0, w1, fatores0, fatores1) → LMDI-I aditivo (Ang 2004) para W = Π fatores, sem resíduo
  lmdi_regiao_setor(df0, df1, ...) → escala, mix setorial, mix regional e intensidade (W = G·S_s·R_rs·I_rs)
  cr_sequeiro(ai, razao=0.563)     → fração do VAB agrícola de sequeiro (metodologia ODS 6.4.1, FAO/UN-Water 2019)
  wue_641(...)                     → eficiência no uso da água (ODS 6.4.1)
  indice_pca(X, sinais, b)         → índice composto com pesos de PCA (Nicoletti et al. 2000) e bootstrap das posições
"""
from __future__ import annotations

import numpy as np
import pandas as pd

CLASSES_TAPIO = ("desacoplamento forte", "desacoplamento fraco", "acoplamento expansivo",
                 "desacoplamento negativo expansivo", "desacoplamento negativo forte", "desacoplamento recessivo",
                 "acoplamento recessivo", "desacoplamento negativo fraco")


# ------------------------------------------------------------------ Tapio
def tapio(dg: float, dw: float, lim=(0.8, 1.2)) -> tuple[float, str]:
    """Elasticidade e = (ΔW/W)/(ΔG/G) e um dos oito estados de Tapio (2005), com limiares 0,8 e 1,2."""
    if dg == 0 or not np.isfinite(dg) or not np.isfinite(dw):
        return np.nan, "indefinido"
    e = dw / dg
    lo, hi = lim
    if dg > 0:
        if dw < 0:
            return e, "desacoplamento forte"
        if e < lo:
            return e, "desacoplamento fraco"
        if e <= hi:
            return e, "acoplamento expansivo"
        return e, "desacoplamento negativo expansivo"
    if dw > 0:
        return e, "desacoplamento negativo forte"
    if e > hi:
        return e, "desacoplamento recessivo"
    if e >= lo:
        return e, "acoplamento recessivo"
    return e, "desacoplamento negativo fraco"


def tapio_tabela(df: pd.DataFrame, unidade: str, g: str, w: str, a0, a1, coluna_ano="ano") -> pd.DataFrame:
    """Tapio por unidade entre a média dos anos a0 e a dos anos a1 (médias de vários anos suavizam o clima)."""
    m0 = df[df[coluna_ano].isin(a0)].groupby(unidade)[[g, w]].mean()
    m1 = df[df[coluna_ano].isin(a1)].groupby(unidade)[[g, w]].mean()
    m0.loc["TOTAL"], m1.loc["TOTAL"] = m0.sum(), m1.sum()
    out = []
    for u in m0.index:
        dg, dw = m1.at[u, g] / m0.at[u, g] - 1, m1.at[u, w] / m0.at[u, w] - 1 if m0.at[u, w] > 0 else np.nan
        e, c = tapio(dg, dw)
        out.append({unidade: u, "dg": dg, "dw": dw, "elasticidade": e, "classe": c})
    return pd.DataFrame(out)


# ------------------------------------------------------------------ LMDI
def media_log(a, b):
    """Média logarítmica L(a, b) = (a − b)/(ln a − ln b), com L(a, a) = a e 0 quando algum é 0."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    with np.errstate(divide="ignore", invalid="ignore"):
        L = np.where(np.isclose(a, b), a, (a - b) / (np.log(a) - np.log(b)))
    return np.where((a > 0) & (b > 0), L, 0.0)


def lmdi_aditivo(w0, w1, fatores0: dict, fatores1: dict) -> pd.DataFrame:
    """LMDI-I aditivo (Ang 2004; 2015). w = Π_k x_k em cada célula; efeito_k = L(w1, w0)·ln(x_k1/x_k0).
    A soma dos efeitos reproduz Δw exatamente (sem resíduo) quando todos os fatores são positivos."""
    w0, w1 = np.asarray(w0, float), np.asarray(w1, float)
    L = media_log(w1, w0)
    out = {}
    for k in fatores0:
        x0, x1 = np.asarray(fatores0[k], float), np.asarray(fatores1[k], float)
        with np.errstate(divide="ignore", invalid="ignore"):
            out[k] = np.where((x0 > 0) & (x1 > 0), L * np.log(x1 / x0), 0.0)
    r = pd.DataFrame(out)
    r["dW"] = w1 - w0
    r["residuo"] = r["dW"] - r[list(fatores0)].sum(axis=1)
    return r


def lmdi_regiao_setor(df0: pd.DataFrame, df1: pd.DataFrame, regiao: str, setor: str, g: str, w: str) -> pd.DataFrame:
    """W = Σ_{r,s} G · (G_s/G) · (G_rs/G_s) · (W_rs/G_rs): escala, mix setorial, mix regional, intensidade.
    df0/df1: uma linha por (região, setor) com VAB (g) e água (w) nos dois períodos, mesma ordem de linhas."""
    a, b = df0.reset_index(drop=True), df1.reset_index(drop=True)
    G0, G1 = a[g].sum(), b[g].sum()
    S0, S1 = a.groupby(setor)[g].transform("sum"), b.groupby(setor)[g].transform("sum")
    f0 = {"escala": np.full(len(a), G0), "mix_setorial": S0 / G0, "mix_regional": a[g] / S0, "intensidade": a[w] / a[g]}
    f1 = {"escala": np.full(len(b), G1), "mix_setorial": S1 / G1, "mix_regional": b[g] / S1, "intensidade": b[w] / b[g]}
    r = lmdi_aditivo(a[w], b[w], f0, f1)
    return pd.concat([a[[regiao, setor]], r], axis=1)


# ------------------------------------------------------------------ ODS 6.4.1
def cr_sequeiro(ai, razao=0.563):
    """Cr = 1/(1 + Ai/((1 − Ai)·razão)); Ai = fração irrigada da área cultivada; razão = produtividade de
    sequeiro / irrigada (padrão 0,563 da metodologia passo a passo do 6.4.1, UN-Water/FAO 2019)."""
    ai = np.clip(np.asarray(ai, float), 0, 0.999)
    return 1.0 / (1.0 + ai / ((1 - ai) * razao))


def wue_641(g_agro, w_agro, g_ind, w_ind, g_serv, w_serv, ai=None, razao=0.563):
    """WUE = Awe·PA + Mwe·PM + Swe·PS (unidades de VAB por unidade de água). Sem ai, Cr = 0 (limite superior)."""
    cr = 0.0 if ai is None else cr_sequeiro(ai, razao)
    g_agro, w_agro, g_ind, w_ind, g_serv, w_serv = (np.asarray(x, float) for x in (g_agro, w_agro, g_ind, w_ind, g_serv, w_serv))
    W = w_agro + w_ind + w_serv
    awe = g_agro * (1 - cr) / w_agro
    return (awe * w_agro + g_ind + g_serv) / W


# ------------------------------------------------------------------ índice composto
def varimax(L: np.ndarray, iters: int = 100, tol: float = 1e-8) -> np.ndarray:
    """Rotação varimax (Kaiser 1958) de uma matriz de cargas p × k."""
    p, k = L.shape
    if k < 2:
        return L
    R = np.eye(k)
    d = 0.0
    for _ in range(iters):
        Lr = L @ R
        u, s, vt = np.linalg.svd(L.T @ (Lr ** 3 - Lr @ np.diag((Lr ** 2).sum(axis=0)) / p))
        R = u @ vt
        d_novo = s.sum()
        if d and d_novo < d * (1 + tol):
            break
        d = d_novo
    return L @ R


def _pesos_pca(Z: np.ndarray):
    """Pesos de Nicoletti et al. (2000) / OCDE-JRC (2008): componentes com autovalor > 1, cargas com rotação
    varimax; cada indicador entra pelo componente em que tem a maior carga ao quadrado, com peso
    = carga²/Σ cargas² do componente × (variância do componente rotacionado / variância retida)."""
    C = np.corrcoef(Z.T)
    ev, V = np.linalg.eigh(C)
    o = ev.argsort()[::-1]
    ev, V = ev[o], V[:, o]
    k = max(1, int((ev > 1).sum()))                                   # critério de Kaiser
    L2 = varimax(V[:, :k] * np.sqrt(ev[:k])) ** 2
    var = L2.sum(axis=0)
    dono = L2.argmax(axis=1)
    w = np.array([L2[j, dono[j]] / L2[dono == dono[j], dono[j]].sum() * var[dono[j]] / var[np.unique(dono)].sum()
                  for j in range(len(L2))])
    return w / w.sum(), ev, k


def indice_pca(X: pd.DataFrame, sinais: dict | None = None, b: int = 2000, semente: int = 0):
    """Índice composto: indicadores padronizados (sinal +1 = quanto maior melhor), pesos da PCA e,
    por bootstrap das unidades, intervalos das posições e dos pesos. Devolve (tabela, metadados)."""
    X = X.astype(float).copy()
    for c, s in (sinais or {}).items():
        X[c] = s * X[c]
    Z = ((X - X.mean()) / X.std(ddof=1)).values
    w, ev, k = _pesos_pca(Z)
    idx = Z @ w
    rng = np.random.default_rng(semente)
    R = np.empty((b, len(X)))
    Wb = np.empty((b, X.shape[1]))
    for i in range(b):
        s = rng.integers(0, len(X), len(X))
        Xb = X.iloc[s]
        sd = Xb.std(ddof=1).replace(0, np.nan)
        wb, _, _ = _pesos_pca(((Xb - Xb.mean()) / sd).fillna(0).values)
        Wb[i] = wb
        R[i] = (-(((X - Xb.mean()) / sd).fillna(0).values @ wb)).argsort().argsort() + 1
    tab = pd.DataFrame({"indice": idx, "posicao": (-idx).argsort().argsort() + 1,
                        "pos_p05": np.percentile(R, 5, 0), "pos_p50": np.median(R, 0), "pos_p95": np.percentile(R, 95, 0),
                        "indice_pesos_iguais": Z.mean(axis=1)}, index=X.index).sort_values("posicao")
    meta = {"autovalores": ev.tolist(), "componentes": k, "pesos": dict(zip(X.columns, w)),
            "pesos_p05": dict(zip(X.columns, np.percentile(Wb, 5, 0))), "pesos_p95": dict(zip(X.columns, np.percentile(Wb, 95, 0)))}
    return tab, meta
