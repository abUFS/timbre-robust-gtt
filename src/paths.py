"""
paths.py — Configuração central de caminhos do projeto.

Esta é a ÚNICA fonte de verdade sobre onde as coisas moram. Toda a lógica
que sobrevive (ou não) ao reset do Colab passa por aqui.

Regra de ouro:
  - CÓDIGO       -> GitHub (versionado, leve)          -> este repositório
  - DADOS PESADOS-> Google Drive (persistente)         -> DRIVE_ROOT
  - RUNTIME      -> disco local do Colab (rápido, efêmero) -> LOCAL_ROOT

O Drive é o "armazém": guarda áudio bruto, caches processados (em archives),
checkpoints e resultados. O disco local (/content) é onde a gente REALMENTE
trabalha, porque o Drive montado é lento para muitos arquivos pequenos.

Uso típico:
    from src.paths import P
    P.print_summary()
    raw = P.guitarset_raw          # onde o GuitarSet cru vive (Drive)
    cache = P.local("cache", "gset")  # onde o cache processado é usado (local)
"""

import os
import sys
from pathlib import Path


def _in_colab() -> bool:
    """Detecta se estamos rodando dentro do Google Colab."""
    return "google.colab" in sys.modules or os.path.isdir("/content")


class _Paths:
    def __init__(self):
        self.in_colab = _in_colab()

        # ---- RAÍZES ----------------------------------------------------
        # DRIVE_ROOT: persistente entre sessões. Sobrescreva com a env var
        # TCC_DRIVE_ROOT se quiser outro local no seu Drive.
        default_drive = "/content/drive/MyDrive/tcc-guitar" if self.in_colab \
            else os.path.expanduser("~/tcc-guitar-drive")
        self.drive_root = Path(os.environ.get("TCC_DRIVE_ROOT", default_drive))

        # LOCAL_ROOT: disco rápido e EFÊMERO. É aqui que se roda de fato.
        default_local = "/content/tcc-data" if self.in_colab \
            else os.path.expanduser("~/tcc-guitar-local")
        self.local_root = Path(os.environ.get("TCC_LOCAL_ROOT", default_local))

        # REPO_ROOT: raiz deste repositório (dois níveis acima deste arquivo).
        self.repo_root = Path(__file__).resolve().parents[1]

    # ---- HELPERS -------------------------------------------------------
    def drive(self, *parts) -> Path:
        p = self.drive_root.joinpath(*parts)
        return p

    def local(self, *parts) -> Path:
        p = self.local_root.joinpath(*parts)
        return p

    # ---- CAMINHOS NO DRIVE (persistentes) ------------------------------
    @property
    def guitarset_raw(self) -> Path:
        # Áudio (mono/mix) + anotações .jams originais do GuitarSet.
        return self.drive("raw", "guitarset")

    @property
    def egdb_raw(self) -> Path:
        return self.drive("raw", "egdb")

    @property
    def egset12_raw(self) -> Path:
        return self.drive("raw", "egset12")

    @property
    def archives(self) -> Path:
        # Caches processados empacotados como .tar (features CQT + alvos, FX etc.)
        return self.drive("archives")

    @property
    def experiments(self) -> Path:
        # root_dir das experiências: checkpoints + logs do sacred, por condição.
        return self.drive("experiments")

    @property
    def results_drive(self) -> Path:
        # Cópia persistente de tabelas/figuras (o repo também versiona as finais).
        return self.drive("results")

    # ---- CAMINHOS LOCAIS (working dir, rápidos) ------------------------
    @property
    def cache_local(self) -> Path:
        # save_loc do GuitarSet: onde features/alvos processados são lidos/escritos.
        return self.local("cache")

    # ---- SETUP ---------------------------------------------------------
    def ensure_dirs(self):
        """Cria as pastas persistentes no Drive se ainda não existirem."""
        for p in [self.guitarset_raw, self.egdb_raw, self.egset12_raw,
                  self.archives, self.experiments, self.results_drive,
                  self.cache_local]:
            p.mkdir(parents=True, exist_ok=True)

    def print_summary(self):
        print(f"{'Colab?':<16}: {self.in_colab}")
        print(f"{'DRIVE_ROOT':<16}: {self.drive_root}")
        print(f"{'LOCAL_ROOT':<16}: {self.local_root}")
        print(f"{'REPO_ROOT':<16}: {self.repo_root}")
        print("-" * 60)
        print(f"{'guitarset_raw':<16}: {self.guitarset_raw}")
        print(f"{'archives':<16}: {self.archives}")
        print(f"{'experiments':<16}: {self.experiments}")
        print(f"{'cache_local':<16}: {self.cache_local}")


# Instância única importada em todo o projeto.
P = _Paths()


if __name__ == "__main__":
    P.print_summary()
