"""
compare_baseline.py — Passo 0d: tabela "publicado vs. reproduzido".

Lê o summary.csv gerado pelo train.py (média ± desvio entre as 6 dobras) e o
compara com os números publicados (configs/reference_tabcnn.yaml). Imprime a
tabela, salva um CSV em results/tables/ e dá um veredito de sucesso da Fase 0.

Critério: a reprodução "bate" se a média reproduzida cai dentro da margem
(padrão 0.03 absoluto) da publicada. Também reportamos se cai dentro de ±1
desvio publicado, que é um sinal mais generoso e honesto de equivalência.

Uso:
    from src.compare_baseline import compare
    compare()   # usa o summary da condição baseline por padrão
"""

import csv
from pathlib import Path

import yaml

from .paths import P

REF_YAML = "configs/reference_tabcnn.yaml"
DEFAULT_SUMMARY = "baseline_tabcnn_summary.csv"   # salvo em P.results_drive
KEY_METRICS = ["tablature.f1", "tablature.tdr", "multipitch.f1"]


def _load_summary(summary_path: Path) -> dict:
    out = {}
    with open(summary_path) as f:
        for row in csv.DictReader(f):
            out[row["metric"]] = {"mean": float(row["mean"]),
                                  "std": float(row["std"])}
    return out


def _match_key(repro: dict, ref_key: str):
    """Casa a chave publicada (ex. 'tablature.f1') com a chave reproduzida,
    tolerando prefixos (ex. 'tablature.tablature.f1')."""
    if ref_key in repro:
        return ref_key
    group, _, metric = ref_key.partition(".")
    for k in repro:
        if k.endswith(ref_key) or (group in k and k.endswith(metric)):
            return k
    return None


def compare(summary_path=None, ref_yaml=REF_YAML, tol=0.03):
    summary_path = Path(summary_path) if summary_path \
        else (P.results_drive / DEFAULT_SUMMARY)
    if not summary_path.exists():
        raise FileNotFoundError(
            f"summary não encontrado: {summary_path}\n"
            "Rode a validação cruzada completa antes (run_condition sem quick).")

    repro = _load_summary(summary_path)
    ref = yaml.safe_load(open(ref_yaml))
    ref_metrics = ref["metrics"]

    rows = []
    print(f"\n{'métrica':<20} {'publicado':>16} {'reproduzido':>16} "
          f"{'Δ':>8}  situação")
    print("-" * 76)
    for ref_key, pub in ref_metrics.items():
        rk = _match_key(repro, ref_key)
        if rk is None:
            print(f"{ref_key:<20} {'—':>16} {'AUSENTE':>16}")
            rows.append([ref_key, pub["mean"], pub["std"], "", "", "", ""])
            continue
        r = repro[rk]
        delta = r["mean"] - pub["mean"]
        within_margin = abs(delta) <= tol
        within_std = abs(delta) <= pub["std"]
        status = "ok" if within_margin else ("~ (±1σ)" if within_std else "VERIFICAR")
        print(f"{ref_key:<20} {pub['mean']:.3f}±{pub['std']:.3f}   "
              f"{r['mean']:.3f}±{r['std']:.3f}   {delta:+.3f}  {status}")
        rows.append([ref_key, pub["mean"], pub["std"],
                     r["mean"], r["std"], round(delta, 4),
                     "sim" if within_margin else ("~1sigma" if within_std else "nao")])

    # salva CSV (cópia local no repo — pode ser read-only sob mount do Modal)
    out_csv = P.repo_root / "results" / "tables" / "phase0_published_vs_reproduced.csv"
    try:
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        with open(out_csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["metric", "pub_mean", "pub_std",
                        "repro_mean", "repro_std", "delta", "within_margin"])
            w.writerows(rows)
    except OSError as e:
        print(f"[compare] cópia no repo pulada (read-only?): {e}")
        out_csv = None
    # cópia persistente no Drive/Volume (canônica)
    drive_csv = P.results_drive / "phase0_published_vs_reproduced.csv"
    drive_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(drive_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["metric", "pub_mean", "pub_std",
                    "repro_mean", "repro_std", "delta", "within_margin"])
        w.writerows(rows)

    # veredito
    print("-" * 76)
    verdicts = []
    for key in KEY_METRICS:
        rk = _match_key(repro, key)
        if rk is None:
            verdicts.append((key, None))
            continue
        delta = repro[rk]["mean"] - ref_metrics[key]["mean"]
        verdicts.append((key, abs(delta) <= tol or
                         abs(delta) <= ref_metrics[key]["std"]))
    ok_all = all(v is True for _, v in verdicts)
    for key, v in verdicts:
        tag = "OK" if v else ("AUSENTE" if v is None else "VERIFICAR")
        print(f"  {key:<18}: {tag}")
    print("\n" + ("FASE 0 REPRODUZIDA — pode seguir para a Fase 1."
                  if ok_all else
                  "Alguma métrica-chave fora da margem — investigar antes da Fase 1."))
    print(f"tabela salva em: {out_csv or drive_csv}")
    return rows


if __name__ == "__main__":
    compare()