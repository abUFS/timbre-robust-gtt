"""
paths.py — Configuração central de caminhos, com noção de PLATAFORMA.

Fonte única de verdade sobre onde as coisas moram, portátil entre Colab e local.
A diferença entre os ambientes é só isto:

  COLAB : "armazém" = Google Drive (persistente); "bancada" = /content (efêmero,
          apagado no reset). Precisa remontar tudo a cada sessão.
  LOCAL : "armazém" e "bancada" ficam ambos no disco local (persistente). Nada
          reseta; o setup é feito uma vez.

Selecione via TCC_PLATFORM = AUTO | COLAB | LOCAL (AUTO detecta o Colab).
Sobrescreva raízes com TCC_DRIVE_ROOT / TCC_LOCAL_ROOT se quiser.

    from src.paths import P
    P.print_summary()
"""

import os
import sys
from pathlib import Path


def _detect_colab() -> bool:
    return "google.colab" in sys.modules or os.path.isdir("/content/drive")


def _resolve_platform(explicit=None) -> str:
    plat = (explicit or os.environ.get("TCC_PLATFORM", "AUTO")).upper()
    if plat == "AUTO":
        return "COLAB" if _detect_colab() else "LOCAL"
    return plat


class _Paths:
    def __init__(self):
        self.configure()

    def configure(self, platform=None):
        """(Re)configura as raízes conforme a plataforma. Muta o singleton no
        lugar, então referências `from .paths import P` continuam válidas."""
        self.platform = _resolve_platform(platform)
        self.in_colab = self.platform == "COLAB"

        if self.platform == "COLAB":
            default_drive = "/content/drive/MyDrive/tcc-guitar"
            default_local = "/content/tcc-data"
        else:  # LOCAL
            base = os.path.expanduser("~/tcc-guitar-data")
            default_drive = os.path.join(base, "store")   # "armazém"
            default_local = os.path.join(base, "work")     # "bancada"

        self.drive_root = Path(os.environ.get("TCC_DRIVE_ROOT", default_drive))
        self.local_root = Path(os.environ.get("TCC_LOCAL_ROOT", default_local))
        self.repo_root = Path(__file__).resolve().parents[1]
        return self

    # ---- helpers -------------------------------------------------------
    def drive(self, *parts) -> Path:
        return self.drive_root.joinpath(*parts)

    def local(self, *parts) -> Path:
        return self.local_root.joinpath(*parts)

    # ---- caminhos persistentes ("armazém") -----------------------------
    @property
    def guitarset_raw(self) -> Path:
        return self.drive("raw", "guitarset")

    @property
    def egdb_raw(self) -> Path:
        return self.drive("raw", "egdb")

    @property
    def egset12_raw(self) -> Path:
        return self.drive("raw", "egset12")

    @property
    def archives(self) -> Path:
        return self.drive("archives")

    @property
    def experiments(self) -> Path:
        return self.drive("experiments")

    @property
    def results_drive(self) -> Path:
        return self.drive("results")

    # ---- caminhos de trabalho ("bancada") ------------------------------
    @property
    def cache_local(self) -> Path:
        return self.local("cache")

    # ---- setup ---------------------------------------------------------
    def ensure_dirs(self):
        for p in [self.guitarset_raw, self.egdb_raw, self.egset12_raw,
                  self.archives, self.experiments, self.results_drive,
                  self.cache_local]:
            p.mkdir(parents=True, exist_ok=True)

    def print_summary(self):
        print(f"{'PLATFORM':<16}: {self.platform}")
        print(f"{'DRIVE_ROOT':<16}: {self.drive_root}   (armazém)")
        print(f"{'LOCAL_ROOT':<16}: {self.local_root}   (bancada)")
        print(f"{'REPO_ROOT':<16}: {self.repo_root}")
        print("-" * 60)
        print(f"{'guitarset_raw':<16}: {self.guitarset_raw}")
        print(f"{'archives':<16}: {self.archives}")
        print(f"{'experiments':<16}: {self.experiments}")
        print(f"{'cache_local':<16}: {self.cache_local}")


P = _Paths()


def configure(platform=None):
    """Reconfigura o singleton global P para a plataforma dada."""
    return P.configure(platform)


if __name__ == "__main__":
    P.print_summary()
