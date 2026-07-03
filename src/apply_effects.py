"""
apply_effects.py — Geração de augmentation por efeitos de áudio (Fases 1 e 2).

STUB: será preenchido na Fase 1. Deixado aqui com a estrutura pretendida e os
parâmetros "moderados" a documentar no TCC (cada efeito em um nível fixo).

Fase 1: aplicar UM efeito por vez ao conjunto de TESTE do GuitarSet.
Fase 2: aplicar efeitos aleatórios ao conjunto de TREINO (data augmentation).

As anotações .jams NÃO mudam — o efeito altera o timbre, não as notas.
"""

from pathlib import Path

# from pedalboard import (Pedalboard, Reverb, Delay, Chorus,
#                         Distortion, PeakFilter, Gain)
# from pedalboard.io import AudioFile


# Parâmetros "moderados" fixos — DOCUMENTAR no TCC (reprodutibilidade).
# TODO(Fase 1): fixar e justificar cada valor.
EFFECT_PRESETS = {
    "reverb":     {"room_size": 0.5, "wet_level": 0.3},
    "delay":      {"delay_seconds": 0.25, "feedback": 0.3, "mix": 0.3},
    "modulation": {"rate_hz": 1.0, "depth": 0.5, "mix": 0.5},   # chorus
    "eq":         {"cutoff_frequency_hz": 1000, "gain_db": 6.0},
    "distortion": {"drive_db": 25.0},  # caso à parte — o mais severo
}


def apply_single_effect(in_wav: Path, out_wav: Path, effect: str,
                        sample_rate: int = 22050) -> Path:
    """Aplica um único efeito a um .wav e salva o resultado.

    TODO(Fase 1): implementar com pedalboard usando EFFECT_PRESETS[effect].
    """
    raise NotImplementedError("Preencher na Fase 1.")


def build_test_fx_set(clean_dir: Path, out_root: Path, effects=None):
    """Gera cópias do conjunto de TESTE, um efeito por vez (Fase 1)."""
    raise NotImplementedError("Preencher na Fase 1.")


def build_train_fx_augmentation(clean_dir: Path, out_root: Path,
                                seed: int = 0):
    """Gera versões aumentadas do TREINO com efeitos aleatórios (Fase 2).

    Nota de design (decidido para Colab): gerar um conjunto FIXO com seed fixa
    em vez de on-the-fly — reprodutível e não roda pedalboard a cada época.
    """
    raise NotImplementedError("Preencher na Fase 2.")
