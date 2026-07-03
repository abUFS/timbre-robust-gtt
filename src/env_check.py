"""
env_check.py — Portão de sanidade do ambiente (rodar no fim do bootstrap).

Objetivo: dizer, em 5 segundos, se o ambiente está OK para começar a Fase 0,
ou EXATAMENTE o que quebrou. Como as dependências desses repos são de ~2022
(numpy 1.2x / librosa 0.9 / jams 0.3.4) e o Colab é moderno, este check é o
que você vai rodar repetidamente enquanto ajusta o requirements-colab.txt.

Rode:  python -m src.env_check      (ou !python -m src.env_check no Colab)
"""

import importlib
import sys


def _ver(mod):
    try:
        m = importlib.import_module(mod)
        return getattr(m, "__version__", "?")
    except Exception as e:  # noqa: BLE001
        return f"FALHOU ({type(e).__name__}: {e})"


def check_versions():
    print("== Versões das dependências frágeis ==")
    for mod in ["numpy", "scipy", "librosa", "numba", "jams", "mirdata",
                "mir_eval", "pandas", "matplotlib", "sacred", "torch",
                "pedalboard"]:
        print(f"  {mod:<12}: {_ver(mod)}")


def check_gpu() -> bool:
    print("\n== GPU / CUDA ==")
    try:
        import torch
        ok = torch.cuda.is_available()
        print(f"  cuda disponível : {ok}")
        if ok:
            print(f"  device          : {torch.cuda.get_device_name(0)}")
        else:
            print("  !!! SEM GPU — troque em Runtime > Change runtime type > "
                  "T4 GPU. Treinar em CPU é inviável.")
        return ok
    except Exception as e:  # noqa: BLE001
        print(f"  torch indisponível: {e}")
        return False


def _try_import(mod, attr):
    try:
        m = importlib.import_module(mod)
        if attr:
            getattr(m, attr)
        print(f"  OK    {mod}" + (f".{attr}" if attr else ""))
        return True
    except Exception as e:  # noqa: BLE001
        print(f"  ERRO  {mod}: {type(e).__name__}: {e}")
        return False


def check_frameworks() -> bool:
    """Retorna True se os essenciais da Fase 0 (TabCNN) importam.

    O FretNet (guitar_transcription_continuous) é ADIADO para a Fase 4 e não
    reprova o portão aqui — ele depende de `muda`, que não instala no 3.12.
    """
    print("\n== Imports ESSENCIAIS (Fase 0 — TabCNN) ==")
    essential = [
        ("amt_tools", None),
        ("amt_tools.datasets", "GuitarSet"),
        ("amt_tools.models", "TabCNN"),
        ("amt_tools.features", "CQT"),
        ("guitar_transcription_inhibition", None),
    ]
    essential_ok = all(_try_import(m, a) for m, a in essential)

    print("\n== Imports ADIADOS (Fase 4 — FretNet, não reprova) ==")
    _try_import("guitar_transcription_continuous", None)

    return essential_ok


def main():
    print(f"Python: {sys.version.split()[0]}\n")
    check_versions()
    gpu_ok = check_gpu()
    fw_ok = check_frameworks()
    print("\n" + ("=" * 50))
    if fw_ok and gpu_ok:
        print("AMBIENTE OK — pode seguir para a Fase 0 (rode o smoke test).")
    elif fw_ok and not gpu_ok:
        print("IMPORTS OK, mas SEM GPU — troque o runtime para GPU antes de treinar.")
    else:
        print("IMPORTS ESSENCIAIS FALHARAM — verifique a instalação dos frameworks.")
    return 0 if (fw_ok and gpu_ok) else 1


if __name__ == "__main__":
    raise SystemExit(main())