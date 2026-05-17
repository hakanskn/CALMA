# Attention Replacement Framework (ARF) — Prototype

PhD tez prototip projesi. 9 aday yöntem (Yön A: 3 mimari, Yön B: 3 öğrenme, Yön C: 3 unified)
GPT-2 Small üzerinde standart attention baseline ile karşılaştırılır.

## Hızlı Başlangıç (Colab)

1. **Colab'da yeni notebook aç** → `notebooks/00_setup.ipynb` içeriğini kopyala-çalıştır
2. **Drive mount** edilir, repo klonlanır, requirements yüklenir
3. **Dashboard:** `notebooks/04_dashboard.ipynb` ile Gradio + ngrok aç → tarayıcıdan yönet

## Dizin Yapısı

```
arf_prototype/
├── configs/              YAML konfigürasyonlar (base, methods/, datasets/)
├── src/
│   ├── config.py         ExperimentConfig dataclass
│   ├── utils.py          seed_everything, drive helpers
│   ├── models/
│   │   ├── base_model.py GPT-2 wrapper
│   │   ├── attention/    Yön A: standard, A1 PCR, A2 LI-SCR, A3 RBR
│   │   ├── learning/     Yön B: B1 MI-S3T, B2 CMA, B3 FWML
│   │   └── unified/      Yön C: C1 PCUN, C2 EP-T, C3 CHA
│   ├── data/             HF dataset loading, preprocessing, domain shift
│   ├── training/         Trainer, evaluator, checkpointer
│   ├── logging_utils/    JSON + SQLite logger, result store
│   └── visualization/    Matplotlib plot generator
├── dashboard/            Gradio app + panels
├── notebooks/            00_setup → 04_dashboard
├── results/              Drive'a mount edilen sonuçlar (run-id bazlı)
├── train.py              Tek run entry point
└── batch_run.py          Toplu run entry point
```

## Run Modları

| Mod | Komut |
|---|---|
| Tekil run | `python train.py --method A1_PCR --seed 42` |
| Tek yöntem, 5 seed | `python train.py --method A1_PCR --all-seeds` |
| Yön grubu | `python batch_run.py --direction A` |
| Tüm yöntemler | `python batch_run.py --all --all-seeds` |
| Domain shift | `python train.py --method B1 --train-ds wikitext2 --eval-ds finance` |

## Dashboard

`notebooks/04_dashboard.ipynb` → Gradio + ngrok URL. Sekmeler:

- **Run** — tekil run başlat, parametre ayarla, canlı log
- **Batch Run** — A/B/C/All grubu, seed sayısı, paralel/sıralı
- **Results** — tüm tamamlanan run'lar, filtre, sıralama
- **Compare** — 2+ yöntem otomatik grafik
- **Plots** — kaydedilmiş grafikleri görüntüle/indir
- **Config** — YAML düzenle, kaydet, yükle

## Sonuç Saklama

Her run benzersiz `run_id` alır. Drive yolu:
`/content/drive/MyDrive/arf_results/runs/{run_id}/`

İçerik:
- `config.json` — tam config snapshot
- `metrics.json` — epoch + step metrikleri
- `log.txt` — text log
- `checkpoints/` — periyodik checkpoint
- `plots/` — bu run'a özel grafikler

Ana SQLite: `results/arf_results.db` — tüm run'ların özet kayıtları.

## 9 Yöntem Özeti

| Yön | Kod | İsim | Yenilik |
|---|---|---|---|
| A | A1 | Predictive Coding Router | Score = 1/prediction_error |
| A | A2 | Lateral Inhibition Sparse Router | Wilson-Cowan + Hebbian inhibition |
| A | A3 | Resonance-Based Router | ART-style memory bank + vigilance |
| B | B1 | Meta-Init Self-Supervised TTT | MAML + masked LM inner loop |
| B | B2 | Contrastive Meta-Adaptation | NT-Xent + meta-train domains |
| B | B3 | Fast Weight Meta-Learning | LoRA-rank Hebbian fast weights |
| C | C1 | Predictive Coding Unified Net | PC = routing + local update |
| C | C2 | Equilibrium Propagation TF | İki faz, backprop yok |
| C | C3 | Contrastive Hebbian Attention | h⁺/h⁻ farkı routing+learning |

## Versiyon

v0.1 — İskelet + 9 yöntem prototip. Mayıs 2026.
