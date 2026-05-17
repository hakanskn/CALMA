# Standalone Notebooks — Her Yöntem Tek Başına

11 bağımsız Jupyter notebook. Her biri tek bir yöntemi başından sona çalıştırır:
veri yükleme → model + yöntem inşası → eğitim → değerlendirme → grafik + JSON kayıt.

```
standalone_notebooks/
├── README.md
├── arf_lib.py                       ← ortak helper (Trainer/Evaluator/Data/Plot)
├── build_notebooks.py               ← notebook üretici
├── 00_baseline.ipynb                ← GPT-2 Small standart attention
├── 01_A1_PCR.ipynb                  ← Predictive Coding Router
├── 02_A2_LI_SCR.ipynb               ← Lateral Inhibition Sparse Router
├── 03_A3_RBR.ipynb                  ← Resonance-Based Router
├── 04_B1_MI_S3T.ipynb               ← Meta-Init Self-Supervised TTT (FOMAML)
├── 05_B2_CMA.ipynb                  ← Contrastive Meta-Adaptation
├── 06_B3_FWML.ipynb                 ← Fast Weight Meta-Learning
├── 07_C1_PCUN.ipynb                 ← Predictive Coding Unified Network
├── 08_C2_EP_T.ipynb                 ← Equilibrium Propagation Transformer
├── 09_C3_CHA.ipynb                  ← Contrastive Hebbian Attention
├── 10_compare_all.ipynb             ← 10 yöntemin _index.csv üzerinden kıyas
└── results/                         ← tüm run'lar buraya
    ├── _index.csv                   ← her run'dan bir satır otomatik
    ├── _compare/                    ← compare notebook çıktıları
    └── {METHOD}_{run_name}_{ts}/
        ├── config.json
        ├── metrics.json
        ├── log.txt
        ├── final_summary.txt
        └── plots/
            ├── loss_curve.png
            ├── ppl_curve.png
            └── stability.png
```

## Her notebook'un 8 hücresi

| # | Hücre | Açıklama |
|---|---|---|
| 1 | **Setup** | Colab tespit, pip install, Drive mount (varsa), `arf_lib` import |
| 2 | **PARAMETRELER** | Tüm hiperparametreler tek dict'te. Sadece bu hücreyi düzenle. |
| 3 | **Imports** | Standart kütüphaneler + `arf_lib`'ten yardımcılar |
| 4 | **Method** | Yöntem-spesifik sınıf (PCR, LI-SCR, FOMAML adapter, vs.) — notebook'ta inline |
| 5 | **Build** | `build_model_and_adapter(PARAMS)` — model + opsiyonel adapter |
| 6 | **Run** | `run_pipeline(model, PARAMS, adapter)` — tek satır, her şey otomatik |
| 7 | **Display** | Üretilen PNG'leri inline gösterir |

## Colab'da Kullanım

1. **Yeni Colab notebook aç** → GPU runtime seç (T4 yeterli, A100 daha hızlı)
2. **Tek hücrede repo klonla:**
   ```python
   !git clone https://github.com/hakanskn/CALMA.git /content/CALMA
   ```
3. **Çalıştırmak istediğin notebook'u aç** (Colab → File → Open notebook → GitHub →
   `hakanskn/CALMA` → `standalone_notebooks/01_A1_PCR.ipynb`)
4. **Hücre 2'deki PARAMS'ı düzenle** (smoke test için `limit_train_batches` küçük kalsın)
5. **Tüm hücreleri çalıştır** (`Runtime → Run all`)
6. Sonuçlar `./results/` altına yazılır. Drive'a yedeklemek istersen:
   ```python
   PARAMS["results_root"] = "/content/drive/MyDrive/arf_results"
   ```

## Tek-makinada Local Kullanım (CPU/küçük GPU ile sadece kod inceleme)

PRD gereği gerçek eğitim Colab'da yapılır. Lokal'de sadece notebook editing/syntax:

```powershell
cd standalone_notebooks
jupyter notebook        # veya VS Code üzerinden aç
```

## 10 Notebook Bittikten Sonra

`10_compare_all.ipynb`'ı çalıştır:
- `results/_index.csv` okunur (her notebook çalıştığında otomatik satır eklendi)
- 4 karşılaştırma grafiği üretilir:
  - PPL bar chart (mean ± std, baseline kırmızı çizgi)
  - Speed–accuracy scatter
  - Parameter count bar
  - Seed variance box plot (multi-seed varsa)
- Çıktılar `results/_compare/` altında.

## Notlar

- **Self-contained**: Her notebook hiçbir `arf_prototype/` modülüne bağlı değil.
  Sadece yanındaki `arf_lib.py`'yi import eder. Trainer/data loader/plot orada;
  her yöntemin kendi attention/adapter sınıfı notebook'un içinde.
- **PARAMS hücresi**: Bir notebook'u "farklı bir deney" yapmak için sadece Hücre 2'yi
  düzenle (örn. `dataset = "wikitext103"`, `num_epochs = 5`, method-özel param).
- **Sonuç indeksi**: `results/_index.csv` 10 notebook'tan gelen tüm satırlarla büyür.
  Aynı yöntemi farklı `run_name`'le çalıştırırsan birden çok satır olur.
- **Notebook'ları yeniden üret**: Şablonu değiştirip `python build_notebooks.py` ile
  11 notebook'u tek seferde yeniden basabilirsin.
