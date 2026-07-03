"""
env_setup.py — Setup de ambiente PORTÁTIL (Colab ou local), numa chamada só.

    from src.env_setup import setup
    setup(platform="AUTO")     # "AUTO" | "COLAB" | "LOCAL"

O que faz, por plataforma:
  COLAB : monta o Drive, clona os frameworks em /content, instala as deps e os
          frameworks, injeta-os no sys.path do kernel (contorna o .pth da
          instalação editável) e verifica imports + GPU.
  LOCAL : não monta Drive nem reseta nada. Na 1ª vez, clona os frameworks numa
          pasta ao lado do repo e instala no seu venv; nas próximas, só verifica.
          Assim, "trocar um parâmetro" basta — sem ajustes manuais.

A diferença some depois do setup: run_condition()/compare() rodam idênticos.
"""

import importlib
import os
import subprocess
import sys
from pathlib import Path

from . import paths

FRAMEWORKS = {
    "amt-tools":
        "https://github.com/cwitkowitz/amt-tools.git",
    "guitar-transcription-with-inhibition":
        "https://github.com/cwitkowitz/guitar-transcription-with-inhibition.git",
}
FRETNET_REPO = {
    "guitar-transcription-continuous":
        "https://github.com/cwitkowitz/guitar-transcription-continuous.git",
}


def _run(cmd, **kw):
    print("  $", " ".join(str(c) for c in cmd))
    return subprocess.run(cmd, check=True, **kw)


def _pip(*args):
    _run([sys.executable, "-m", "pip", "install", "-q", *args])


def _clone(url, dst: Path):
    if dst.is_dir():
        return
    _run(["git", "clone", "--depth", "1", url, str(dst)])


def _frameworks_installed() -> bool:
    try:
        importlib.import_module("amt_tools")
        return True
    except Exception:  # noqa: BLE001
        return False


def _inject_path(frameworks_root: Path):
    for name in list(FRAMEWORKS) + list(FRETNET_REPO):
        p = frameworks_root / name
        if p.is_dir() and str(p) not in sys.path:
            sys.path.insert(0, str(p))
    importlib.invalidate_caches()


def setup(platform="AUTO", install=True, with_fretnet=False, frameworks_root=None):
    plat = paths._resolve_platform(platform)
    paths.configure(plat)
    P = paths.P
    print(f"== Setup de ambiente ({plat}) ==")

    # Onde os frameworks são clonados/instalados.
    if frameworks_root is None:
        frameworks_root = Path("/content") if plat == "COLAB" \
            else (P.repo_root.parent / "frameworks")
    frameworks_root = Path(frameworks_root)
    frameworks_root.mkdir(parents=True, exist_ok=True)
    os.environ["TCC_FRAMEWORKS_ROOT"] = str(frameworks_root)

    # 1) Drive (só Colab)
    if plat == "COLAB":
        try:
            from google.colab import drive
            if not os.path.isdir("/content/drive/MyDrive"):
                print("Montando o Google Drive...")
                drive.mount("/content/drive")
            else:
                print("Drive já montado.")
        except Exception as e:  # noqa: BLE001
            print(f"[aviso] não consegui montar o Drive automaticamente: {e}")

    # 2) Frameworks: instala se faltar (Colab: sempre pós-reset; Local: 1ª vez)
    need_install = install and not _frameworks_installed()
    repos = dict(FRAMEWORKS)
    if with_fretnet:
        repos.update(FRETNET_REPO)

    if need_install:
        print("Instalando frameworks (pode levar alguns minutos)...")
        for name, url in repos.items():
            _clone(url, frameworks_root / name)
        req = P.repo_root / "requirements-colab.txt"
        if req.exists():
            _pip("-r", str(req))
        _pip("-e", str(frameworks_root / "amt-tools"))
        _pip("-e", str(frameworks_root / "guitar-transcription-with-inhibition"))
        if with_fretnet:
            # continuous depende de muda (quebra no 3.12): instala sem deps.
            try:
                _pip("-e", str(frameworks_root / "guitar-transcription-continuous"),
                     "--no-deps")
            except Exception as e:  # noqa: BLE001
                print(f"[aviso] FretNet adiado (Fase 4): {e}")
    else:
        if _frameworks_installed():
            print("Frameworks já disponíveis — pulando instalação.")
        else:
            print("[aviso] frameworks ausentes e install=False — imports vão falhar.")

    # 3) Torna importável NO KERNEL ATUAL (sem restart)
    _inject_path(frameworks_root)

    # 4) Verificação
    return _verify()


def _verify():
    print("\n== Verificação ==")
    ok = True
    for mod, attr in [("amt_tools", None), ("amt_tools.datasets", "GuitarSet"),
                      ("amt_tools.models", "TabCNN"), ("amt_tools.features", "CQT")]:
        try:
            m = importlib.import_module(mod)
            if attr:
                getattr(m, attr)
            print(f"  OK    {mod}" + (f".{attr}" if attr else ""))
        except Exception as e:  # noqa: BLE001
            ok = False
            print(f"  ERRO  {mod}: {type(e).__name__}: {e}")

    try:
        import torch
        gpu = torch.cuda.is_available()
        print(f"  GPU   : {torch.cuda.get_device_name(0) if gpu else 'AUSENTE (CPU)'}")
        if not gpu:
            print("        !!! sem GPU — no Colab troque o runtime; local, "
                  "confira a instalação do torch com CUDA.")
    except Exception as e:  # noqa: BLE001
        ok = False
        print(f"  ERRO  torch: {e}")

    from .paths import P
    print()
    P.print_summary()
    print("\n" + ("Ambiente pronto." if ok else "Ambiente com pendências — veja acima."))
    return ok
