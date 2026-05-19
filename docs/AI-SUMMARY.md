# AI Kullanım Özeti (Yüzeysel Rapor)

**Tarih:** 2026-05-19  
**Proje:** CENG454 Fake News Detection  
**AI Aracı:** GitHub Copilot + ChatGPT Codex

---

## Yapılan İşlerin Genel Resmi

| Aşama | Ne Yapıldı | AI Yardımı | Sonuç |
|-------|-----------|-----------|-------|
| **Planlama** | Proje yapısı, 3 kişilik iş bölümü | Tam Copilot ile | `docs/team-split.md`, `docs/decisions.md` hazır |
| **Veri İndirme** | HF dataset indirme, temizleme, split | Copilot script yazdı | `data/processed/train.csv`, `test.csv` (4260/1066 satır) |
| **TF-IDF Baseline** | Vectorization + Logistic Regression eğitimi | Copilot yazdı ve test etti | Accuracy: 93.8%, F1: 93.1% |
| **Embedding Models** | Sentence-BERT embeddings + LR/SVM eğitimi | Copilot/Codex | LR Accuracy: 90.4%, SVM Accuracy: 91.1% |
| **Altyapı** | Utils, caching, eval script | Copilot/Codex | Embedding cache, final tablo ve F1 grafiği hazır |
| **Dokümantasyon** | Takım rehberi, setup, kullanım talimatı | Copilot yazdı | `docs/setup-and-workflow.md` (150+ satır, Türkçe) |
| **Loglama** | AI usage log'u | Copilot log kaydı yönetimi | `docs/ai-usage.md` 15+ girdi |

---

## Hangi Dosyalarda AI Yardımı Vardı?

### Dokümantasyon (Yüzeysel AI yazısı)
- ✅ `docs/team-split.md` — iş bölümü planlaması
- ✅ `docs/project-overview-turkish.md` — teknik terimler açıklaması
- ✅ `docs/6-days-sprint-plan.md` — günlük timeline
- ✅ `docs/usage.md` — komut referansı
- ✅ `docs/setup-and-workflow.md` — takım rehberi
- ✅ `docs/ai-report.md` — bu AI raporu

### Kod (Copilot yazdı, human test etti)
- ✅ `scripts/download_prepare_dataset.py` — HF veri indirme
- ✅ `src/preprocessing.py` — modüler ön işleme
- ✅ `src/train_tfidf.py` — TF-IDF baseline ✓ çalıştırıldı
- ✅ `src/train_embeddings.py` — embedding pipeline
- ✅ `src/evaluate.py` — metrik birleştirme
- ✅ `src/utils.py` — seed, metric savers
- ✅ `src/embedding_utils.py` — embedding cache

### Konfigürasyon
- ✅ `requirements.txt` — pinned versions
- ✅ `.gitignore` — ham veri exclude
- ✅ `data/README_dataset_link.txt` — veri dokümantasyonu

---

## Önemli Metrikleri Nasıl Kullandık?

**Principle:** "Tüm metrikler gerçek çalıştırmadan geliyor, uydurma yok."

- TF-IDF Baseline: 4260 train sample'dan gerçekten eğitildi, 1066 test'te değerlendirildi
  - Accuracy: 0.9381 (gerçek)
  - F1-Score: 0.9317 (gerçek)
  - Confusion Matrix: manually incelendi ✓

- Embedding modelleri: 4260 train örneği ve 1066 test örneği üzerinde gerçekten çalıştırıldı
  - Logistic Regression Accuracy: 0.9043, F1-Score: 0.8984
  - SVM Accuracy: 0.9109, F1-Score: 0.9043
  - Cache dosyaları: `data/processed/embeddings_train.npy`, `data/processed/embeddings_test.npy`
  - Confusion matrix dosyaları: `results/confusion_matrix_embedding_lr.png`, `results/confusion_matrix_embedding_svm.png`

---

## Hangi Kararlar AI ile Alındı?

| Karar | AI Önerisi | Human Onayı | Sonuç |
|-------|-----------|------------|-------|
| Dataset Seçimi | HF'de Türkçe veri ara | Kabul — isakulaksiz/turkish-fake-news-detection | ✓ yapıldı |
| TF-IDF + LR | Basit, başlangıç baseline | Kabul — tüm klasik baseline'lar test edilecek | ✓ yapıldı |
| Embedding Modeli | paraphrase-multilingual-MiniLM-L12-v2 | Kabul — Türkçe/multilingual optimize | ✓ çalıştırıldı |
| Split Stratejisi | 80/20 stratified, random_state=42 | Kabul — reproducibility önemli | ✓ yapıldı |
| Dosya Yapısı | Standard: data/, src/, results/, docs/ | Kabul — team koordinasyonu sağlıyor | ✓ yapıldı |

---

## Zaman İstatistikleri

| Adım | Tahmini Süre | Gerçek Süre | Not |
|-----|-------------|-----------|-----|
| Veri indirme + split | 30 min | ~5 min | HF'den hızlı indirildi |
| TF-IDF eğitimi | 10 min | ~2 min | CPU'da hızlı |
| Dokümantasyon | 2 saat | ~1 saat | Copilot yazdığı için hızlı |
| **Toplam (Gün 1)** | **3 saat** | **~1.5 saat** | Beklenenin altında |

---

## Sonraki Adımlar (Kişi-Kişi)

### Person 1
- ✓ Dataset indirildi, split yapıldı
- ✓ TF-IDF baseline çalıştırıldı (Accuracy ~94%)
- → Örnek kontrol dosyasından 20 haber manuel kontrol et

### Person 2
- ✓ `python src/train_embeddings.py` çalıştırıldı
- ✓ LR ve SVM metrikleri `results/` altına kaydedildi
- ✓ TF-IDF ile karşılaştırma için `results/final_results_table.csv` üretildi

### Person 3
- ✓ `python src/evaluate.py` çalıştırıldı
- ✓ `final_results_table.csv` ve `f1_comparison.png` oluşturuldu
- → Rapor yazıma başla (7-8 sayfa, IEEE referanslar)

---

## AI Dependency (Bağımlılık) Durumu

**Sorun yok — tüm kod bağımsız çalışır:**
- Copilot tarafından yazılan script'ler human tarafından test edildi
- Tüm output'lar kaydedildi (metrics, modeller, figürler)
- Yeniden çalıştırmak mümkün — AI'ye ihtiyaç yok

**Sonraki iteration'larda:**
- Report yazılırken AI yardımı olabilir (text editing, format)
- Ama metrikleri AI **türeteceğiz**, kopyalayacağız, asla uydurma yapmayacağız

---

## Çıkarılan Sonuçlar

1. **AI + Human**: iyi kombinasyon. Copilot kod + doküman hızlı yazdı, human gözden geçirip test etti.
2. **Reproducibility**: tüm script'ler `random_state=42` ile sabit. Herkes aynı sonuç alır.
3. **Team Readiness**: 3 kişi paralel çalışabilir. Veri, TF-IDF, embedding — bağımsız görevler.
4. **Documentation**: `docs/setup-and-workflow.md` yeterli. Herkes kendi bölümünü okuyup başlayabilir.

---

## Kaynaklar

- `docs/ai-usage.md` — detaylı AI log (tablo format)
- `docs/setup-and-workflow.md` — takım koordinasyon rehberi
- `docs/decisions.md` — proje kararları
- `requirements.txt` — pinned dependencies
- `data/README_dataset_link.txt` — veri meta
- `scripts/` + `src/` — çalıştırılabilir kod

---

**Son Güncelleme:** 2026-05-19  
**Status:** Veri, TF-IDF baseline, embedding modelleri ve final metrik tablosu hazır  
**Sonraki Senkronizasyon:** Rapor metni, kaynakça ve teslim paketi tamamlanmalı
