"""Leitura e limpeza dos dados do Online Retail II."""
import pandas as pd

COLUNAS = ["Invoice", "StockCode", "Description", "Quantity",
           "InvoiceDate", "Price", "Customer ID", "Country"]


def carregar_bruto(caminho) -> pd.DataFrame:
    """Lê as DUAS abas do Excel (2009-2010 e 2010-2011) e junta."""
    abas = pd.read_excel(caminho, sheet_name=None, engine="openpyxl")
    return pd.concat(abas.values(), ignore_index=True)


def limpar(df: pd.DataFrame) -> pd.DataFrame:
    """Mantém só vendas reais de produtos, com preço e quantidade válidos."""
    d = df.copy()
    d["Invoice"] = d["Invoice"].astype(str)
    d = d[~d["Invoice"].str.startswith("C")]                  # cancelamentos
    d["StockCode"] = d["StockCode"].astype(str).str.upper().str.strip()
    d = d[d["StockCode"].str.match(r"^\d{5}[A-Z]{0,2}$")]     # só produtos (tira frete, taxas, ajustes)
    d = d[(d["Price"] > 0) & (d["Quantity"] > 0)]
    d = d.drop_duplicates()
    d["Description"] = (d["Description"].fillna("").astype(str).str.upper()
                        .str.replace(r"\s+", " ", regex=True).str.strip())
    d["InvoiceDate"] = pd.to_datetime(d["InvoiceDate"])
    return d.sort_values("InvoiceDate").reset_index(drop=True)
