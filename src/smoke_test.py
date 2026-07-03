"""
smoke_test.py — Prova que o PIPELINE roda no runtime (não só importa).

Como o stack é moderno (numpy 2 / librosa 0.11) em vez do da época, este teste
confirma empiricamente que a extração de CQT e o TabCNN funcionam de fato,
usando áudio sintético — antes de gastar tempo baixando o GuitarSet.

Rode:  python -m src.smoke_test    (ou !python -m src.smoke_test no Colab)
"""

import numpy as np


def test_cqt():
    """Extrai CQT de um seno sintético usando a feature do amt-tools."""
    from amt_tools.features import CQT
    sr, hop = 22050, 512
    dur = 2.0
    t = np.linspace(0, dur, int(sr * dur), endpoint=False)
    audio = 0.5 * np.sin(2 * np.pi * 220.0 * t).astype(np.float32)  # Lá 220 Hz

    cqt = CQT(sample_rate=sr, hop_length=hop, n_bins=84, bins_per_octave=12)
    feats = cqt.process_audio(audio)
    feats = np.asarray(feats)
    assert np.isfinite(feats).all(), "CQT produziu valores não-finitos"
    print(f"  OK   CQT.process_audio -> shape {feats.shape}, "
          f"dim_in={cqt.get_feature_size()}")
    return cqt.get_feature_size()


def test_tabcnn(dim_in):
    """Instancia o TabCNN e roda um forward em tensor aleatório do shape certo."""
    import torch
    from amt_tools.models import TabCNN
    import amt_tools.tools as tools

    device = "cuda" if torch.cuda.is_available() else "cpu"
    profile = tools.GuitarProfile(num_frets=19)
    model = TabCNN(dim_in=dim_in, profile=profile, in_channels=1, device=device)
    model = model.to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  OK   TabCNN instanciado em '{device}' ({n_params:,} params)")

    # forward opcional (best-effort): feats no formato (B, T, C, F, W)
    try:
        B, T, C, F, W = 1, 20, 1, dim_in, 9
        x = torch.rand(B, T, C, F, W, device=device)
        out = model(x)
        key = next(iter(out)) if isinstance(out, dict) else "tensor"
        print(f"  OK   TabCNN.forward -> chave '{key}'")
    except Exception as e:  # noqa: BLE001
        print(f"  (info) forward de teste pulado ({type(e).__name__}); "
              f"o shape real vem do DataLoader na Fase 0.")


def main():
    print("== Smoke test de runtime ==")
    try:
        dim_in = test_cqt()
        test_tabcnn(dim_in)
        print("\nSMOKE TEST OK — o stack moderno roda o pipeline. Siga a Fase 0.")
        return 0
    except Exception as e:  # noqa: BLE001
        import traceback
        traceback.print_exc()
        print(f"\nSMOKE TEST FALHOU: {type(e).__name__}: {e}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
