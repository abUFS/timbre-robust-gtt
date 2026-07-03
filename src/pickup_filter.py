"""
pickup_filter.py — Simulação (linear) do captador da guitarra (Fase 3).

STUB: será preenchido na Fase 3.

Ideia: aproximar a resposta em frequência do captador por um passa-baixa de
2ª ordem com frequência de corte (fc) e fator de ressonância (Q) variáveis.
O augmentation "+Captador" varre fc/Q para gerar variações tonais plausíveis.

Ponto de verificação da Fase 3: validar por espectrograma/oitiva que o filtro
produz variação tonal, não artefato, ANTES de gerar o dataset inteiro.
"""

# import numpy as np
# from scipy import signal


# Faixas típicas a varrer (ajustar/justificar na Fase 3).
FC_RANGE_HZ = (2000.0, 6000.0)   # corte do passa-baixa
Q_RANGE = (0.7, 2.5)             # ressonância (pico perto de fc)


def design_pickup_filter(fc_hz: float, q: float, sample_rate: int = 22050):
    """Projeta um passa-baixa ressonante de 2ª ordem (biquad).

    TODO(Fase 3): implementar com scipy.signal (ex.: iirpeak/iirfilter ou
    coeficientes de biquad low-pass com Q). Retornar (b, a).
    """
    raise NotImplementedError("Preencher na Fase 3.")


def apply_pickup(in_wav, out_wav, fc_hz: float, q: float,
                 sample_rate: int = 22050):
    """Aplica o filtro de captador a um .wav (scipy.signal.lfilter/sosfilt)."""
    raise NotImplementedError("Preencher na Fase 3.")
