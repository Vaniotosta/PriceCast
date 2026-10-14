"""Passo 1: lê o Excel (2 abas), limpa e salva em data/processed."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pricecast.config import DADOS_BRUTOS, DADOS_LIMPOS
from pricecast.dados import carregar_bruto, limpar

if __name__ == "__main__":
    print("Lendo o Excel (pode levar 1-2 minutos)...")
    bruto = carregar_bruto(DADOS_BRUTOS)
    limpo = limpar(bruto)
    limpo.to_csv(DADOS_LIMPOS, index=False, compression="gzip")
    print(f"Bruto: {len(bruto):,} linhas -> Limpo: {len(limpo):,} linhas")
    print(f"Período: {limpo.InvoiceDate.min()} até {limpo.InvoiceDate.max()}")
    print(f"Salvo em {DADOS_LIMPOS}")
