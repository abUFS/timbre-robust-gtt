"""
train.py — Validação cruzada de 6 dobras do TabCNN (Fase 0, sub-passo 0b).

Adaptação FIEL de amt-tools/examples/papers/tabcnn.py. O loop, o CQT, o
estimador, o avaliador e as chamadas train()/validate() são os mesmos do script
original. As mudanças são só as necessárias para o nosso setup:

  1) base_dir  -> GuitarSet extraído localmente (download_guitarset), em vez de
     None (que dispararia o download via mirdata, quebrado no 1.0.0).
  2) save_loc  -> cache CQT no DISCO LOCAL (rápido). Ver nota sobre reset abaixo.
  3) root_dir  -> P.experiments/<condição> no DRIVE, então:
       - o log_dir de cada dobra persiste => train(resume=True) RETOMA o treino
         do último checkpoint após um reset do Colab;
       - resultados por dobra viram CSV no Drive => dobras já concluídas são
         PULADAS ao reexecutar (retomada em granularidade de dobra).
  4) sacred removido (log/observers) — salvamos config + métricas manualmente.
  5) modo `quick` para um smoke run ponta-a-ponta antes do treino completo.

Uso (no Colab):
    from src.train import run_condition
    run_condition("configs/baseline_tabcnn.yaml", quick=True)   # valida pipeline
    run_condition("configs/baseline_tabcnn.yaml")               # rodada completa
"""

import csv
import json
import shutil
import statistics
from pathlib import Path

import yaml
import torch
from torch.utils.data import DataLoader


def _patch_torch_load():
    """PyTorch 2.6+ usa weights_only=True por padrão e recusa despicklar objetos
    (o amt-tools salva/carrega o MODELO inteiro via torch.save/torch.load). Como
    os checkpoints são gerados por nós (fonte confiável), restauramos o padrão
    antigo weights_only=False para os resumes/checkpoints funcionarem."""
    import functools
    if getattr(torch.load, "_tcc_patched", False):
        return
    _orig = torch.load

    @functools.wraps(_orig)
    def _load(*a, **k):
        k.setdefault("weights_only", False)
        return _orig(*a, **k)

    _load._tcc_patched = True
    torch.load = _load


_patch_torch_load()


# --- Torna os frameworks importáveis NO KERNEL ATUAL, sem depender do .pth da
# instalação editável (só lido na inicialização do interpretador). Portátil:
# usa TCC_FRAMEWORKS_ROOT (definido por env_setup) e cai para /content no Colab.
import sys as _sys
import os as _os
_roots = [_os.environ.get("TCC_FRAMEWORKS_ROOT", "/content")]
_names = ("amt-tools", "guitar-transcription-with-inhibition",
          "guitar-transcription-continuous")
for _r in _roots:
    for _n in _names:
        _p = _os.path.join(_r, _n)
        if _os.path.isdir(_p) and _p not in _sys.path:
            _sys.path.insert(0, _p)
import importlib as _importlib
_importlib.invalidate_caches()

from amt_tools.datasets import GuitarSet
from amt_tools.models import TabCNN
from amt_tools.features import CQT
from amt_tools.train import train as amt_train
from amt_tools.transcribe import ComboEstimator, \
                                 TablatureWrapper, \
                                 StackedMultiPitchCollapser
from amt_tools.evaluate import ComboEvaluator, \
                               LossWrapper, \
                               MultipitchEvaluator, \
                               TablatureEvaluator, \
                               SoftmaxAccuracy, \
                               validate
import amt_tools.tools as tools

from .paths import P
from . import download_guitarset


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
def load_config(path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def _resolve_device(cfg) -> object:
    """Devolve o gpu_id (int) se houver CUDA, senão 'cpu'."""
    if torch.cuda.is_available():
        return int(cfg.get("gpu_id", 0))
    print("[train] AVISO: sem CUDA — caindo para CPU (lento).")
    return "cpu"


def _cache_archive_name(cfg) -> str:
    return f"cache_{cfg.get('condition', 'cache')}"


def _pull_cache_if_available(cfg):
    """Restaura o cache CQT do Drive (se existir), evitando recomputar após reset."""
    from . import data_sync
    name = _cache_archive_name(cfg)
    if (P.archives / f"{name}.tar").exists():
        print("[cache] cache CQT achado no Drive — restaurando para o local...")
        try:
            data_sync.pull_archive(name, P.local("cache"))
        except Exception as e:  # noqa: BLE001
            print(f"[cache] falha ao restaurar (segue recomputando): {e}")


def _push_cache_once(cfg, save_loc):
    """Arquiva o cache CQT no Drive uma única vez (após a 1ª geração completa)."""
    from . import data_sync
    name = _cache_archive_name(cfg)
    if (P.archives / f"{name}.tar").exists():
        return  # já arquivado
    print("[cache] arquivando cache CQT no Drive (uma vez; ~1-2 min)...")
    try:
        data_sync.push_archive(name, save_loc)
    except Exception as e:  # noqa: BLE001
        print(f"[cache] falha ao arquivar (não crítico): {e}")


def _build_components(cfg):
    """Perfil + CQT + estimador + avaliador — idênticos ao tabcnn.py original."""
    profile = tools.GuitarProfile(num_frets=cfg.get("num_frets", 19))

    # CQT do experimento real: 8 oitavas, 2 bins por semitom (n_bins=192, bpo=24).
    data_proc = CQT(sample_rate=cfg["sample_rate"],
                    hop_length=cfg["hop_length"],
                    n_bins=192,
                    bins_per_octave=24)

    estimator = ComboEstimator([TablatureWrapper(profile=profile),
                                StackedMultiPitchCollapser(profile=profile)])

    evaluator = ComboEvaluator([LossWrapper(),
                                MultipitchEvaluator(),
                                TablatureEvaluator(profile=profile),
                                SoftmaxAccuracy()])
    return profile, data_proc, estimator, evaluator


# --------------------------------------------------------------------------- #
# Resultados
# --------------------------------------------------------------------------- #
def _flatten(d, prefix="") -> dict:
    """Achata um dict aninhado de métricas em {chave.pontilhada: float}."""
    out = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(_flatten(v, prefix=key + "."))
        else:
            try:
                out[key] = float(v)
            except (TypeError, ValueError):
                pass
    return out


def _fold_csv(root_dir: Path, k: int) -> Path:
    return root_dir / "results" / f"fold-{k}.csv"


def _save_fold(root_dir: Path, k: int, flat: dict):
    path = _fold_csv(root_dir, k)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["metric", "value"])
        for m, v in sorted(flat.items()):
            w.writerow([m, v])
    print(f"[fold {k}] métricas salvas em {path}")


def _load_fold(root_dir: Path, k: int):
    path = _fold_csv(root_dir, k)
    if not path.exists():
        return None
    flat = {}
    with open(path) as f:
        r = csv.DictReader(f)
        for row in r:
            flat[row["metric"]] = float(row["value"])
    return flat


def _write_summary(root_dir: Path, condition: str, per_fold: list):
    """Agrega média ± desvio entre dobras e salva CSV (root_dir e Drive results)."""
    keys = sorted({k for fold in per_fold for k in fold})
    rows = []
    for key in keys:
        vals = [fold[key] for fold in per_fold if key in fold]
        mean = statistics.mean(vals)
        std = statistics.pstdev(vals) if len(vals) > 1 else 0.0
        rows.append((key, mean, std, len(vals)))

    for dest in [root_dir / "results" / "summary.csv",
                 P.results_drive / f"{condition}_summary.csv"]:
        dest.parent.mkdir(parents=True, exist_ok=True)
        with open(dest, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["metric", "mean", "std", "n_folds"])
            w.writerows(rows)
    print(f"\n[summary] média ± desvio salvos ({len(rows)} métricas).")
    # Destaque das métricas-chave para a Fase 0.
    for key in ["tablature.f1", "tablature.tdr", "tablature.precision",
                "tablature.recall", "multipitch.f1"]:
        hit = [r for r in rows if r[0] == key]
        if hit:
            _, mean, std, n = hit[0]
            print(f"  {key:<22}: {mean:.4f} ± {std:.4f}  (n={n})")
    return rows


# --------------------------------------------------------------------------- #
# Loop principal
# --------------------------------------------------------------------------- #
def run_condition(config_path, quick: bool = False, folds=None):
    """Roda a validação cruzada de 6 dobras para uma condição do YAML.

    quick=True  -> smoke run: 1 dobra, poucas iterações, sem checkpoints, em
                   diretório separado (…__quick). Serve para validar o pipeline
                   ponta-a-ponta em minutos antes da rodada completa.
    folds       -> lista de índices [0..5] para rodar só algumas dobras.
    """
    cfg = load_config(config_path)
    condition = cfg.get("condition", Path(config_path).stem)

    # --- parâmetros (com override do modo quick) ---
    sample_rate = cfg["sample_rate"]
    hop_length = cfg["hop_length"]
    num_frames = cfg["num_frames"]
    iterations = cfg["iterations"]
    checkpoints = cfg["checkpoints"]
    batch_size = cfg["batch_size"]
    seed = cfg.get("seed", 0)
    reset_data = cfg.get("reset_data", False)

    if quick:
        iterations = min(iterations, 60)
        checkpoints = 0
        folds = [0]
        condition = condition + "__quick"
        print(f"[quick] rodada de smoke: {iterations} iterações, dobra 0 apenas.")

    device = _resolve_device(cfg)

    # --- caminhos ---
    base_dir = str(download_guitarset.local_base_dir())   # GuitarSet local
    save_loc = str(P.local("cache", cfg.get("condition", condition)))  # cache CQT local
    root_dir = P.experiments / condition                  # Drive (persistente)
    root_dir.mkdir(parents=True, exist_ok=True)
    # guarda uma cópia da config usada (reprodutibilidade)
    shutil.copy(config_path, root_dir / "config.yaml")

    print(f"[paths] base_dir : {base_dir}")
    print(f"[paths] save_loc : {save_loc}  (cache CQT, local)")
    print(f"[paths] root_dir : {root_dir}  (Drive)\n")

    profile, data_proc, estimator, evaluator = _build_components(cfg)

    # Restaura o cache CQT do Drive, se já existir (evita recomputar após reset).
    if not quick:
        _pull_cache_if_available(cfg)

    fold_indices = folds if folds is not None else list(range(6))
    per_fold = []

    for k in fold_indices:
        # Retomada em granularidade de dobra: se já concluiu, pula.
        done = _load_fold(root_dir, k)
        if done is not None and not quick:
            print(f"[fold {k}] já concluída — pulando (métricas no Drive).")
            per_fold.append(done)
            continue

        tools.seed_everything(seed)
        evaluator.set_patterns(['loss', 'pr', 're', 'f1', 'tdr', 'acc'])

        # Partição por músico: treina em 5, testa em 1.
        train_splits = GuitarSet.available_splits()
        test_splits = [train_splits.pop(k)]

        print(f"\n===== Dobra {k} — teste no músico {test_splits[0]} =====")
        print("Carregando partição de treino (gera/reusa o cache CQT)...")
        gset_train = GuitarSet(base_dir=base_dir,
                               splits=train_splits,
                               hop_length=hop_length,
                               sample_rate=sample_rate,
                               num_frames=num_frames,
                               data_proc=data_proc,
                               profile=profile,
                               reset_data=(reset_data and k == fold_indices[0]),
                               save_loc=save_loc)

        train_loader = DataLoader(dataset=gset_train,
                                  batch_size=batch_size,
                                  shuffle=True,
                                  num_workers=0,
                                  drop_last=True)

        print("Carregando partição de teste...")
        gset_test = GuitarSet(base_dir=base_dir,
                              splits=test_splits,
                              hop_length=hop_length,
                              sample_rate=sample_rate,
                              num_frames=None,
                              data_proc=data_proc,
                              profile=profile,
                              store_data=True,
                              save_loc=save_loc)

        # Após a 1ª dobra, todas as 360 faixas já estão em cache: arquiva no Drive
        # (antes do treino, para o cache estar salvo mesmo se houver reset).
        if not quick:
            _push_cache_once(cfg, save_loc)

        print("Inicializando modelo TabCNN...")
        model = TabCNN(dim_in=data_proc.get_feature_size(),
                       profile=profile,
                       in_channels=data_proc.get_num_channels(),
                       device=device)
        model.change_device()
        model.train()

        optimizer = torch.optim.Adadelta(model.parameters(), lr=1.0)

        # log_dir da dobra no Drive => train(resume=True) retoma após reset.
        model_dir = root_dir / "models" / f"fold-{k}"

        print("Treinando...")
        model = amt_train(model=model,
                          train_loader=train_loader,
                          optimizer=optimizer,
                          iterations=iterations,
                          checkpoints=checkpoints,
                          log_dir=str(model_dir),
                          resume=not quick,   # smoke run começa limpo
                          val_set=gset_test,
                          estimator=estimator,
                          evaluator=evaluator)

        print(f"Avaliando a partição de teste (músico {test_splits[0]})...")
        evaluator.set_save_dir(str(root_dir / "results"))
        evaluator.set_patterns(None)

        fold_results = validate(model, gset_test,
                                evaluator=evaluator, estimator=estimator)
        evaluator.reset_results()

        flat = _flatten(fold_results)
        per_fold.append(flat)
        if not quick:
            _save_fold(root_dir, k, flat)
        else:
            print(f"[quick] métricas da dobra 0 (parciais, poucas iterações):")
            for key in ["tablature.f1", "tablature.tdr", "multipitch.f1"]:
                if key in flat:
                    print(f"  {key}: {flat[key]:.4f}")

    if not quick and len(per_fold) > 1:
        _write_summary(root_dir, condition, per_fold)

    print("\n[done] condição:", condition)
    return per_fold


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    run_condition(args.config, quick=args.quick)