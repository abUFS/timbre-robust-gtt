"""
modal_app.py — Roda os experimentos no Modal (GPU sob demanda + Volume).

Separação idiomática:
  - CÓDIGO   : montado da sua máquina a cada `modal run` (add_local_dir) —
               sempre atual, sem git pull nem rebuild.
  - DEPS/FW  : assados na Image uma vez (instalação consistente).
  - DADOS    : Volume "tcc" (caches, checkpoints, dataset) — persistente.
  - BANCADA  : /tmp do container (rápida, efêmera).

Fluxo (da sua máquina, na raiz do repo, com o CLI do Modal):
  modal run modal_app.py --action prepare   # 1x: baixa o GuitarSet (2 zips) para o Volume
  modal run modal_app.py --action smoke      # valida o pipeline (1 dobra curta) na GPU
  modal run modal_app.py --action train      # 6 dobras; cada uma persiste no Volume (retomável)

Editou o código? Só salvar e rodar de novo — o mount pega a versão nova. Não
precisa de prepare nem rebuild. Só mudou dependência? Aí a Image reconstrói.
"""

import pathlib
import modal

APP_NAME = "tcc-guitar"
VOLUME_NAME = "tcc"          # <- EXATAMENTE o nome do seu Volume (case-sensitive!)
GPU = "T4"                   # "T4" (barato) | "A10G" | "A100" (rápido/caro)

LOCAL_PROJECT = pathlib.Path(__file__).parent   # raiz do repo (onde está este arquivo)
PROJECT_DIR = "/root/project"                    # onde o código é montado no container
VOL_PATH = "/vol"

vol = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)

# --- Image: deps + frameworks assados; código montado do local (sempre atual) -
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "libsndfile1", "ffmpeg")
    .pip_install(
        "torch",
        "pedalboard>=0.9.0", "PyYAML>=6.0", "resampy>=0.4.3",
    )
    .run_commands(
        "git clone --depth 1 https://github.com/cwitkowitz/amt-tools "
        "/opt/frameworks/amt-tools",
        "git clone --depth 1 "
        "https://github.com/cwitkowitz/guitar-transcription-with-inhibition "
        "/opt/frameworks/guitar-transcription-with-inhibition",
        "pip install -e /opt/frameworks/amt-tools",
        "pip install -e /opt/frameworks/guitar-transcription-with-inhibition",
    )
    # Monta o código local no container. Ignora venv/git/resultados/caches para
    # não subir lixo. (Se a sua versão do modal não aceitar `ignore=`, atualize o
    # modal ou garanta que .venv NÃO está dentro da pasta do repo.)
    .add_local_dir(
        str(LOCAL_PROJECT), remote_path=PROJECT_DIR,
        ignore=[".git", ".venv", "venv", "__pycache__", "*.pyc",
                "results", "notebooks/.ipynb_checkpoints"],
    )
)

app = modal.App(APP_NAME, image=image)


def _env():
    """Ambiente + caminhos dentro da função remota."""
    import os
    import sys
    os.environ["TCC_PLATFORM"] = "LOCAL"
    os.environ["TCC_DRIVE_ROOT"] = f"{VOL_PATH}/store"    # Volume = armazém (persiste)
    os.environ["TCC_LOCAL_ROOT"] = "/tmp/tcc-work"         # container = bancada (rápida)
    os.environ["TCC_FRAMEWORKS_ROOT"] = "/opt/frameworks"
    sys.path.insert(0, PROJECT_DIR)
    os.chdir(PROJECT_DIR)


def _ensure_data():
    """Garante o GuitarSet (só os 2 zips certos) no Volume + extraído na bancada.
    Idempotente; evita o download completo do mirdata (hexafônico de 3+ GB)."""
    from src import download_guitarset as dg
    dg.main()
    vol.commit()


@app.function(volumes={VOL_PATH: vol}, timeout=60 * 60)
def prepare():
    """1x: baixa os 2 zips do GuitarSet para o Volume."""
    _env()
    _ensure_data()
    print("prepare OK — dataset no Volume.")


@app.function(gpu=GPU, volumes={VOL_PATH: vol}, timeout=60 * 60)
def smoke(config: str = "configs/baseline_tabcnn.yaml"):
    _env()
    import torch
    print("CUDA:", torch.cuda.is_available(),
          torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")
    _ensure_data()
    from src.train import run_condition
    run_condition(config, quick=True)
    vol.commit()


@app.function(gpu=GPU, volumes={VOL_PATH: vol}, timeout=12 * 60 * 60)
def train_fold(fold: int, config: str = "configs/baseline_tabcnn.yaml",
               checkpoints: int = 10):
    _env()
    import torch
    print(f"[fold {fold}] CUDA:", torch.cuda.is_available(),
          torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")
    _ensure_data()
    from src.train import run_condition
    run_condition(config, folds=[fold], checkpoints=checkpoints)
    vol.commit()   # persiste checkpoints + cache archive + fold-CSV desta dobra


@app.function(volumes={VOL_PATH: vol}, timeout=60 * 60)
def finalize(config: str = "configs/baseline_tabcnn.yaml"):
    _env()
    from src.train import run_condition
    run_condition(config)   # todas as dobras já feitas -> só escreve o summary
    try:
        from src.compare_baseline import compare
        compare()
    except Exception as e:  # noqa: BLE001
        print(f"[compare] pulei ({e}) — rode após conferir o summary.")
    vol.commit()


@app.local_entrypoint()
def main(action: str = "smoke",
         config: str = "configs/baseline_tabcnn.yaml",
         checkpoints: int = 10):
    if action == "prepare":
        prepare.remote()
    elif action == "smoke":
        smoke.remote(config)
    elif action == "train":
        for k in range(6):
            print(f"=== disparando dobra {k} ===")
            train_fold.remote(fold=k, config=config, checkpoints=checkpoints)
        finalize.remote(config)
    elif action == "finalize":
        finalize.remote(config)
    else:
        print("ações: prepare | smoke | train | finalize")