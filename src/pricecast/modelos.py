"""PriceCast v2: dois modelos em um.

- Modelo QUENTE: produto já conhecido -> usa o histórico de preços dele.
- Modelo FRIO:   produto inédito -> usa só a descrição (texto) e a quantidade.
Os dois trabalham com log(preço), que acalma os preços extremos.
"""
import joblib
import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.feature_extraction.text import TfidfVectorizer

from .config import DIAS_CALIBRACAO, SEMENTE
from .features import VARS_QUENTE, aplicar_tabela, historico_por_linha, tabela_produtos


class ModeloPriceCast:
    def __init__(self, componentes_texto: int = 60):
        self.componentes_texto = componentes_texto
        self.intervalos = {"conhecido": (-0.3, 0.3), "inedito": (-1.0, 1.0)}

    # ---------- treino ----------
    def _treinar_nucleo(self, treino: pd.DataFrame):
        feats = historico_por_linha(treino)
        com_hist = feats["n_prev"] > 0
        alvo = np.log(feats.loc[com_hist, "Price"]) - feats.loc[com_hist, "lp_med15"]
        self.quente = HistGradientBoostingRegressor(
            max_iter=300, learning_rate=0.08, max_leaf_nodes=31,
            l2_regularization=1.0, random_state=SEMENTE)
        self.quente.fit(feats.loc[com_hist, VARS_QUENTE], alvo)

        self.tabela = tabela_produtos(treino)
        min_df = 2 if len(self.tabela) > 50 else 1      # base pequena (testes): não descartar palavras
        self.tfidf = TfidfVectorizer(ngram_range=(1, 2), min_df=min_df, sublinear_tf=True)
        matriz = self.tfidf.fit_transform(self.tabela["Description"])
        k = max(2, min(self.componentes_texto, matriz.shape[1] - 1))
        self.svd = TruncatedSVD(k, random_state=SEMENTE).fit(matriz)

        peso = 1 / treino.groupby("StockCode")["Price"].transform("size")   # cada produto vale igual
        self.frio = HistGradientBoostingRegressor(
            max_iter=250, learning_rate=0.06, max_leaf_nodes=15,
            l2_regularization=2.0, random_state=SEMENTE)
        self.frio.fit(self._entrada_fria(treino), np.log(treino["Price"]), sample_weight=peso)

    def _entrada_fria(self, d: pd.DataFrame) -> np.ndarray:
        texto = self.svd.transform(self.tfidf.transform(d["Description"].fillna("").str.upper()))
        return np.column_stack([texto, np.log(d["Quantity"])])

    def fit(self, treino: pd.DataFrame, calibrar: bool = True):
        if calibrar:   # treina sem os últimos dias para medir o erro "de verdade"
            corte = treino["InvoiceDate"].max() - pd.Timedelta(days=DIAS_CALIBRACAO)
            sub, val = treino[treino["InvoiceDate"] < corte], treino[treino["InvoiceDate"] >= corte]
            aux = ModeloPriceCast(self.componentes_texto)
            aux._treinar_nucleo(sub)
            res = aux._residuos(val)
            self.intervalos = {g: (float(np.quantile(r, .10)), float(np.quantile(r, .90)))
                               for g, r in res.items() if len(r) > 20} or self.intervalos
        self._treinar_nucleo(treino)
        return self

    def _residuos(self, val: pd.DataFrame) -> dict:
        prev = self.prever(val, com_faixa=False)
        r = np.log(val["Price"].values / prev["preco_previsto"].values)
        c = prev["tipo"].values == "conhecido"
        return {"conhecido": r[c], "inedito": r[~c]}

    # ---------- previsão ----------
    def prever(self, novos: pd.DataFrame, com_faixa: bool = True) -> pd.DataFrame:
        """Precisa das colunas: StockCode, Description, Quantity, InvoiceDate, Country."""
        novos = novos.copy()
        novos["StockCode"] = novos["StockCode"].astype(str).str.upper().str.strip()
        d = aplicar_tabela(novos, self.tabela)
        conhecido = (d["n_prev"] > 0).values
        preco = np.exp(self.frio.predict(self._entrada_fria(novos)))
        if conhecido.any():
            quente = np.exp(d.loc[conhecido, "lp_med15"] + self.quente.predict(d.loc[conhecido, VARS_QUENTE]))
            preco[conhecido] = quente.values
        out = pd.DataFrame({"preco_previsto": preco,
                            "tipo": np.where(conhecido, "conhecido", "inedito")}, index=novos.index)
        if com_faixa:
            for tipo, (lo, hi) in self.intervalos.items():
                m = out["tipo"] == tipo
                out.loc[m, "faixa_min"] = out.loc[m, "preco_previsto"] * np.exp(lo)
                out.loc[m, "faixa_max"] = out.loc[m, "preco_previsto"] * np.exp(hi)
        return out

    # ---------- salvar / carregar ----------
    def salvar(self, caminho):
        joblib.dump(self, caminho, compress=3)

    @staticmethod
    def carregar(caminho) -> "ModeloPriceCast":
        return joblib.load(caminho)
