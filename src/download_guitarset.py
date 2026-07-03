"""
download_guitarset.py — Baixa o GuitarSet (só o necessário) e monta o layout
que o wrapper GuitarSet do amt-tools espera. Fase 0, sub-passo 0a.

Por que não usar o mirdata: o download do wrapper usa a API antiga do mirdata
(`guitarset.Dataset(...).download(...)`), que quebra no mirdata 1.0.0. Aqui
baixamos direto do Zenodo (record 3371780), com verificação de checksum MD5.

O wrapper lê apenas duas pastas sob `base_dir`:
    base_dir/annotation/<track>.jams
    base_dir/audio_mono-mic/<track>_mic.wav
Então baixamos só `annotation.zip` e `audio_mono-mic.zip` (ignoramos o áudio
hexafônico e o pickup_mix — economia grande de download/espaço).

Fluxo (idempotente):
    ensure_zips()     -> garante os 2 zips no Drive (baixa+verifica se faltar)
    extract_local()   -> extrai para o base_dir LOCAL (bancada), verifica 360/360
    main()            -> roda os dois e imprime um resumo

Uso:
    python -m src.download_guitarset        # (Drive já montado)
"""

import hashlib
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

from .paths import P

# Record 3371780 no Zenodo — filename: (url, md5, pasta_destino)
ZENODO = "https://zenodo.org/record/3371780/files"
PARTS = {
    "annotation.zip": (
        f"{ZENODO}/annotation.zip?download=1",
        "b39b78e63d3446f2e54ddb7a54df9b10",
        "annotation",
    ),
    "audio_mono-mic.zip": (
        f"{ZENODO}/audio_mono-mic.zip?download=1",
        "275966d6610ac34999b58426beb119c3",
        "audio_mono-mic",
    ),
}

# Onde os zips persistem (Drive) e onde a bancada é montada (local, rápido).
ZIP_DIR = P.drive("raw", "guitarset")        # Drive: 2 zips
BASE_DIR = P.local("raw", "guitarset")       # local: annotation/ + audio_mono-mic/


def local_base_dir() -> Path:
    """base_dir para passar ao GuitarSet(...). Chamável pelo train.py."""
    return BASE_DIR


def _md5(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def _download(url: str, dst: Path):
    """Baixa via requests (streaming). Cai para wget se requests falhar."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(dst.suffix + ".part")
    try:
        import requests
        from tqdm import tqdm
        with requests.get(url, stream=True, timeout=60) as r:
            r.raise_for_status()
            total = int(r.headers.get("content-length", 0))
            with open(tmp, "wb") as f, tqdm(
                total=total, unit="B", unit_scale=True, desc=dst.name
            ) as bar:
                for chunk in r.iter_content(chunk_size=1 << 20):
                    f.write(chunk)
                    bar.update(len(chunk))
        tmp.rename(dst)
    except Exception as e:  # noqa: BLE001
        print(f"[dl] requests falhou ({e}); tentando wget...")
        subprocess.run(["wget", "-q", "--show-progress", "-O", str(tmp), url],
                       check=True)
        tmp.rename(dst)


def ensure_zips():
    """Garante os 2 zips no Drive, com checksum ok (baixa só o que faltar)."""
    ZIP_DIR.mkdir(parents=True, exist_ok=True)
    for name, (url, md5, _) in PARTS.items():
        dst = ZIP_DIR / name
        if dst.exists():
            print(f"[zip] {name} já no Drive — verificando checksum...")
            if _md5(dst) == md5:
                print(f"[zip] {name} OK (checksum confere).")
                continue
            print(f"[zip] {name} checksum DIVERGE — rebaixando.")
            dst.unlink()
        print(f"[zip] baixando {name} do Zenodo...")
        _download(url, dst)
        got = _md5(dst)
        if got != md5:
            raise RuntimeError(
                f"Checksum de {name} não confere!\n  esperado: {md5}\n  obtido:   {got}")
        print(f"[zip] {name} baixado e verificado.")


def _extract_flat(zip_path: Path, dest_dir: Path):
    """Extrai um zip para dest_dir, achatando uma eventual pasta-raiz interna."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        names = [n for n in zf.namelist() if not n.endswith("/")]
        # Detecta prefixo comum (ex.: 'annotation/....jams' dentro do zip)
        tops = {n.split("/")[0] for n in names}
        strip = None
        if len(tops) == 1 and any("/" in n for n in names):
            strip = list(tops)[0] + "/"
        for member in zf.infolist():
            if member.is_dir():
                continue
            rel = member.filename
            if strip and rel.startswith(strip):
                rel = rel[len(strip):]
            if not rel:
                continue
            target = dest_dir / Path(rel).name  # achata para dest_dir/<arquivo>
            with zf.open(member) as src, open(target, "wb") as out:
                shutil.copyfileobj(src, out)


def extract_local(force: bool = False):
    """Extrai os zips do Drive para o base_dir LOCAL e verifica as contagens."""
    for name, (_, _, destsub) in PARTS.items():
        zip_path = ZIP_DIR / name
        assert zip_path.exists(), f"zip ausente no Drive: {zip_path} (rode ensure_zips)"
        dest = BASE_DIR / destsub
        if dest.exists() and any(dest.iterdir()) and not force:
            print(f"[ext] {destsub}/ já extraído localmente — pulando.")
            continue
        print(f"[ext] extraindo {name} -> {dest} ...")
        _extract_flat(zip_path, dest)

    # Verificação: 360 jams e 360 wavs, com nomes casando.
    ann = sorted((BASE_DIR / "annotation").glob("*.jams"))
    wav = sorted((BASE_DIR / "audio_mono-mic").glob("*_mic.wav"))
    print(f"\n[check] annotation/*.jams      : {len(ann)}")
    print(f"[check] audio_mono-mic/*_mic.wav: {len(wav)}")
    tracks = {p.stem for p in ann}
    wav_tracks = {p.name[:-len('_mic.wav')] for p in wav}
    missing = tracks - wav_tracks
    if len(ann) != 360 or len(wav) != 360:
        print("[check] AVISO: esperado 360/360. Confira o download.")
    if missing:
        print(f"[check] AVISO: {len(missing)} tracks sem áudio, ex.: "
              f"{sorted(missing)[:3]}")
    if len(ann) == 360 and len(wav) == 360 and not missing:
        print("[check] OK — layout completo e consistente.")
    return BASE_DIR


def main():
    print("== 0a: aquisição do GuitarSet ==")
    print(f"ZIP_DIR  (Drive): {ZIP_DIR}")
    print(f"BASE_DIR (local): {BASE_DIR}\n")
    ensure_zips()
    print()
    base = extract_local()
    print(f"\nbase_dir pronto para o GuitarSet(...): {base}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
