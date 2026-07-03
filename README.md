# TCC — Robustez de transcrição de tablatura de guitarra a efeitos de áudio

Treino/avaliação de modelos de transcrição de tablatura (TabCNN e FretNet) e
medição de robustez a efeitos de áudio, com *data augmentation* por efeitos
genéricos e por simulação de captador. Projetado para rodar no **Google Colab**
sobrevivendo aos resets de sessão.

## Como o reset do Colab é resolvido

Três camadas, cada coisa no lugar certo:

| Camada | Onde | O que |
|---|---|---|
| **Código** | GitHub (este repo) | leve, versionado |
| **Dados pesados** | Google Drive (`DRIVE_ROOT`) | áudio bruto, caches (em `.tar`), checkpoints, resultados |
| **Runtime** | disco local do Colab (`/content`, efêmero) | onde o treino roda de fato (rápido) |

O Drive é o **armazém**; o disco local é a **bancada**. A cada sessão, o
notebook de bootstrap reconstrói a bancada a partir do armazém. Isso evita o
gargalo de ler milhares de arquivos pequenos direto do Drive montado.

Tudo isso é centralizado em [`src/paths.py`](src/paths.py) — é o único lugar
que sabe onde as coisas moram.

## Uso a cada sessão

1. Abra `notebooks/00_bootstrap.ipynb` no Colab
   (**File → Open notebook → GitHub**, cole a URL deste repo).
2. Rode as células em ordem: checa GPU → monta Drive → clona os 3 frameworks →
   instala dependências fixadas → **reinicia o runtime** → `env_check`.
3. Sincronize os dados do Drive para o disco local (célula de `data_sync`).
4. Siga a fase em que você está.

## Estrutura

```
tcc-guitar/
├── notebooks/
│   └── 00_bootstrap.ipynb   # setup de cada sessão do Colab
├── src/
│   ├── paths.py             # config central Drive/local (coração do projeto)
│   ├── data_sync.py         # empacota/extrai dados Drive <-> local
│   ├── env_check.py         # portão de sanidade (GPU + imports) da Fase 0
│   ├── apply_effects.py     # augmentation por efeitos (Fases 1-2) [stub]
│   ├── pickup_filter.py     # filtro de captador (Fase 3)        [stub]
│   ├── train.py             # wrapper de treino por condição      [stub]
│   └── evaluate.py          # avaliação + coleta de métricas      [stub]
├── configs/
│   └── baseline_tabcnn.yaml # params da condição baseline (Fase 0)
├── results/
│   ├── tables/              # CSVs por experimento
│   └── figures/             # gráficos
├── requirements-colab.txt   # dependências FIXADAS (era 2022)
└── .gitignore
```

## Frameworks (clonados em runtime, não versionados aqui)

- [`amt-tools`](https://github.com/cwitkowitz/amt-tools) — framework base; o
  **TabCNN** vive em `amt_tools/models/tabcnn.py`.
- [`guitar-transcription-with-inhibition`](https://github.com/cwitkowitz/guitar-transcription-with-inhibition)
  — necessário para o FretNet.
- [`guitar-transcription-continuous`](https://github.com/cwitkowitz/guitar-transcription-continuous)
  — o **FretNet** (Fase 4).

## Layout esperado no Drive (`DRIVE_ROOT`)

```
tcc-guitar/                  (= DRIVE_ROOT, ajustável via env TCC_DRIVE_ROOT)
├── raw/
│   ├── guitarset/           # áudio (mono/mix) + annotation/*.jams
│   ├── egdb/                # avaliação out-of-domain (Fase 2)
│   └── egset12/
├── archives/                # caches processados empacotados (*.tar)
├── experiments/             # checkpoints + logs sacred por condição
└── results/                 # cópia persistente de tabelas/figuras
```

## Nota sobre dependências (risco nº 1 da Fase 0)

`requirements-colab.txt` fixa uma constelação da época (numpy 1.23 / librosa 0.9
/ jams 0.3.4 / numba 0.56). O objetivo é fazer `python -m src.env_check` passar.
Se o Python atual do Colab brigar demais com `jams`/`numba`, o **plano B** é usar
[`condacolab`](https://github.com/conda-incubator/condacolab) para obter Python
3.10 num ambiente controlado (célula opcional comentada no bootstrap).
```
