"""
train.py — Wrapper fino de treino por CONDIÇÃO, em cima do amt-tools.

STUB: será preenchido na Fase 0 (baseline) e estendido nas Fases 2-4.

Não reimplementa o loop de treino: reusa `amt_tools.train.train` e a validação
cruzada de 6 dobras (padrão da área — 6 músicos, treina em 5, testa em 1).
A referência de código é examples/papers/tabcnn.py do amt-tools e os
six_fold_cv_scripts/experiment.py dos repos de tablatura.

Ganchos de persistência (apontar para o Drive via src.paths.P):
  - GuitarSet(base_dir=P.guitarset_raw, save_loc=<cache local>, save_data=True)
  - root_dir da experiência = P.experiments / <nome_da_condição>

Condições previstas (configs/*.yaml):
  baseline_tabcnn | tabcnn_fx | tabcnn_pickup | tabcnn_fx_pickup | (idem FretNet)
"""

# import os
# import torch
# from amt_tools.datasets import GuitarSet
# from amt_tools.models import TabCNN
# from amt_tools.features import CQT
# from amt_tools.train import train as amt_train
# import amt_tools.tools as tools
# from .paths import P


def run_condition(config_path: str):
    """Roda a validação cruzada de 6 dobras para uma condição descrita no YAML.

    TODO(Fase 0): implementar para o baseline TabCNN (áudio limpo).
    """
    raise NotImplementedError("Preencher na Fase 0.")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="caminho do YAML da condição")
    args = ap.parse_args()
    run_condition(args.config)
