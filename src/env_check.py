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


def check_gpu():
    print("\n== GPU / CUDA ==")
    try:
        import torch
        ok = torch.cuda.is_available()
        print(f"  cuda disponível : {ok}")
        if ok:
            print(f"  device          : {torch.cuda.get_device_name(0)}")
    except Exception as e:  # noqa: BLE001
        print(f"  torch indisponível: {e}")


def check_frameworks():
    print("\n== Imports dos frameworks ==")
    targets = [
        ("amt_tools", None),
        ("amt_tools.datasets", "GuitarSet"),
        ("amt_tools.models", "TabCNN"),
        ("amt_tools.features", "CQT"),
        ("guitar_transcription_inhibition", None),
        ("guitar_transcription_continuous", None),
    ]
    all_ok = True
    for mod, attr in targets:
        try:
            m = importlib.import_module(mod)
            if attr:
                getattr(m, attr)
            print(f"  OK    {mod}" + (f".{attr}" if attr else ""))
        except Exception as e:  # noqa: BLE001
            all_ok = False
            print(f"  ERRO  {mod}: {type(e).__name__}: {e}")
    return all_ok


def main():
    print(f"Python: {sys.version.split()[0]}\n")
    check_versions()
    check_gpu()
    ok = check_frameworks()
    print("\n" + ("=" * 50))
    if ok:
        print("AMBIENTE OK — pode seguir para a Fase 0.")
    else:
        print("AMBIENTE COM PROBLEMAS — ajuste requirements-colab.txt e/ou "
              "considere o fallback condacolab (ver README).")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
