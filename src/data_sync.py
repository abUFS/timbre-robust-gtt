"""
data_sync.py — Move dados pesados entre Drive (persistente) e disco local (rápido).

Por que isto existe:
  Ler/escrever milhares de arquivos pequenos (o cache .npz do amt-tools tem um
  por faixa) direto do Drive montado é LENTO e estoura quota. A solução é tratar
  o Drive como armazém de ARCHIVES (.tar): a gente empacota uma vez, e a cada
  sessão do Colab copia o .tar para o disco local e extrai lá. Todo o treino/
  avaliação roda do disco local.

Fluxo:
  1) Gerou o cache processado (Fase 0) em P.cache_local?
         push_archive("gset_cache", P.cache_local)     # sobe pro Drive, 1 vez
  2) Nova sessão do Colab (disco local vazio)?
         pull_archive("gset_cache", P.cache_local.parent)  # baixa e extrai
"""

import shutil
import subprocess
import tarfile
from pathlib import Path

from .paths import P


def push_archive(name: str, src_dir, compress: bool = False) -> Path:
    """Empacota `src_dir` num .tar e copia para P.archives/<name>.tar no Drive.

    compress=False (padrão) usa tar sem gzip: muito mais rápido e o áudio/npz
    já é pouco compressível. Use compress=True só se espaço no Drive apertar.
    """
    src_dir = Path(src_dir)
    assert src_dir.exists(), f"source não existe: {src_dir}"
    P.archives.mkdir(parents=True, exist_ok=True)

    ext = "tar.gz" if compress else "tar"
    local_tar = P.local(f"{name}.{ext}")
    mode = "w:gz" if compress else "w"

    print(f"[push] empacotando {src_dir} -> {local_tar} ...")
    with tarfile.open(local_tar, mode) as tf:
        # arcname = nome da pasta raiz dentro do tar
        tf.add(src_dir, arcname=src_dir.name)

    drive_tar = P.archives / local_tar.name
    print(f"[push] copiando para o Drive: {drive_tar} ...")
    shutil.copy2(local_tar, drive_tar)
    print(f"[push] OK ({_human(drive_tar.stat().st_size)})")
    return drive_tar


def pull_archive(name: str, dest_dir, compress: bool = False) -> Path:
    """Copia P.archives/<name>.tar do Drive para local e extrai em `dest_dir`."""
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)

    ext = "tar.gz" if compress else "tar"
    drive_tar = P.archives / f"{name}.{ext}"
    assert drive_tar.exists(), f"archive não encontrado no Drive: {drive_tar}"

    local_tar = P.local(drive_tar.name)
    print(f"[pull] copiando {drive_tar} -> {local_tar} ...")
    shutil.copy2(drive_tar, local_tar)

    print(f"[pull] extraindo em {dest_dir} ...")
    with tarfile.open(local_tar, "r:*") as tf:
        tf.extractall(dest_dir)
    print("[pull] OK")
    return dest_dir


def rsync_raw(name: str) -> Path:
    """Copia áudio bruto (GuitarSet/EGDB/...) do Drive para o disco local.

    Usa `cp -r`; para datasets brutos (poucos arquivos grandes) é suficiente.
    Retorna o caminho local. Ajuste `name` para 'guitarset'/'egdb'/'egset12'.
    """
    src = P.drive("raw", name)
    dst = P.local("raw", name)
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        print(f"[raw] {dst} já existe, pulando cópia.")
        return dst
    print(f"[raw] copiando {src} -> {dst} ...")
    subprocess.run(["cp", "-r", str(src), str(dst)], check=True)
    print("[raw] OK")
    return dst


def _human(n: int) -> str:
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if n < 1024:
            return f"{n:.1f}{unit}"
        n /= 1024
    return f"{n:.1f}PB"
