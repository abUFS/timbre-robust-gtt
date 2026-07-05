"""
apply_effects.py — Augmentation por efeitos de áudio (Fases 1 e 2).

Modelo de VARIANTES: cada efeito é uma condição nomeada em configs/effects.yaml,
com um `type` (classe do pedalboard) + parâmetros. Para incluir mais versões
(ex.: EQs diferentes), basta adicionar entradas no YAML — o código aqui itera
sobre os nomes; nada precisa mudar.

Fase 1: aplica UMA variante por vez e produz um `base_dir` completo por variante,
que se pluga direto no GuitarSet do amt-tools (annotation/ + audio_mono-mic/).

    <fx_root>/guitarset_fx_<variante>/
        annotation/                      -> link para os .jams limpos (mesmos)
        audio_mono-mic/<track>_mic.wav   -> áudio com o efeito

    from src import apply_effects as fx
    fx.list_variants()                 # nomes disponíveis
    fx.build_variant("distortion")     # gera o base_dir de uma variante
    fx.build_all()                     # gera todas
"""

import os
import shutil
from pathlib import Path

import yaml
from pedalboard import (Pedalboard, Reverb, Delay, Chorus, Distortion,
                        PeakFilter)
from pedalboard.io import AudioFile

from .paths import P
from . import download_guitarset

DEFAULT_CONFIG = "configs/effects.yaml"
WAV_SUBDIR = "audio_mono-mic"
WAV_SUFFIX = "_mic.wav"

# type do YAML -> classe do pedalboard
_TYPES = {
    "reverb": Reverb,
    "delay": Delay,
    "chorus": Chorus,
    "eq": PeakFilter,
    "distortion": Distortion,
}


def load_variants(config_path=DEFAULT_CONFIG) -> dict:
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    return cfg["variants"]


def list_variants(config_path=DEFAULT_CONFIG):
    return list(load_variants(config_path).keys())


def build_board(variant_def: dict) -> Pedalboard:
    """Monta o Pedalboard de uma variante (dispatch pelo campo `type`)."""
    params = dict(variant_def)
    kind = params.pop("type")
    if kind not in _TYPES:
        raise ValueError(f"type de efeito desconhecido: {kind}")
    return Pedalboard([_TYPES[kind](**params)])


def process_wav(in_path: Path, out_path: Path, board: Pedalboard):
    """Aplica o efeito a um .wav mantendo a taxa de amostragem nativa."""
    with AudioFile(str(in_path)) as f:
        audio = f.read(f.frames)      # (canais, amostras)
        sr = f.samplerate
    processed = board(audio, sr)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with AudioFile(str(out_path), "w", sr, processed.shape[0]) as f:
        f.write(processed)


def _link_annotations(clean_base: Path, fx_base: Path):
    src = clean_base / "annotation"
    dst = fx_base / "annotation"
    if dst.exists():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.symlink(src, dst, target_is_directory=True)
    except OSError:
        shutil.copytree(src, dst)


def fx_base_dir(variant: str) -> Path:
    """base_dir do GuitarSet para uma variante (na bancada local)."""
    return P.local("raw", f"guitarset_fx_{variant}")


def build_variant(variant: str, clean_base_dir=None, variants=None,
                  tracks=None, overwrite=False) -> Path:
    """Gera o base_dir completo de UMA variante. Idempotente por faixa."""
    variants = variants or load_variants()
    assert variant in variants, f"variante inválida: {variant}"
    clean_base = Path(clean_base_dir or download_guitarset.local_base_dir())
    fx_base = fx_base_dir(variant)

    _link_annotations(clean_base, fx_base)
    board = build_board(variants[variant])

    clean_wav_dir = clean_base / WAV_SUBDIR
    all_wavs = sorted(clean_wav_dir.glob(f"*{WAV_SUFFIX}"))
    if tracks is not None:
        keep = set(tracks)
        all_wavs = [w for w in all_wavs if w.name[:-len(WAV_SUFFIX)] in keep]

    out_dir = fx_base / WAV_SUBDIR
    out_dir.mkdir(parents=True, exist_ok=True)
    n_done = 0
    for i, wav in enumerate(all_wavs):
        out = out_dir / wav.name
        if out.exists() and not overwrite:
            continue
        process_wav(wav, out, board)
        n_done += 1
        if (i + 1) % 60 == 0:
            print(f"  [{variant}] {i + 1}/{len(all_wavs)} faixas...")
    print(f"[fx] {variant}: {n_done} geradas ({len(all_wavs)} no total) -> {fx_base}")
    return fx_base


def build_all(variants=None, **kw):
    names = list(variants) if variants else list_variants()
    all_defs = load_variants()
    return {n: build_variant(n, variants=all_defs, **kw) for n in names}


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="all",
                    help="nome da variante ou 'all'")
    args = ap.parse_args()
    if args.variant == "all":
        build_all()
    else:
        build_variant(args.variant)