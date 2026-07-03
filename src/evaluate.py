"""
evaluate.py — Roda um modelo treinado e coleta métricas por domínio de teste.

STUB: será preenchido a partir da Fase 0/1.

Métricas (via mir_eval / rotinas do framework):
  - Multipitch: precision, recall, F1
  - Tablatura : precision, recall, F1, TDR
  - Sempre reportar média ± desvio entre dobras/seeds.

Domínios de teste (Fase 2+):
  in_domain (GuitarSet limpo) | in_domain_fx (GuitarSet c/ efeitos)
  out_of_domain (EGDB / EGSet12)

Saída: um CSV em results/tables/ (uma linha por dobra) + agregado média±desvio.
"""

# import pandas as pd
# from amt_tools.evaluate import (ComboEvaluator, MultipitchEvaluator,
#                                 TablatureEvaluator, validate)
# from .paths import P


def evaluate_model(checkpoint_path: str, test_domain: str, out_csv: str):
    """Avalia um checkpoint num domínio de teste e salva as métricas em CSV.

    TODO: implementar reusando os evaluators do amt-tools.
    """
    raise NotImplementedError("Preencher na Fase 0/1.")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--domain", required=True,
                    choices=["in_domain", "in_domain_fx", "out_of_domain"])
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    evaluate_model(args.checkpoint, args.domain, args.out)
