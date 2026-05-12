# Kurulum, Iş Akışı ve Takım Rehberi

Bu doküman projeyi beraber çalışmak için gereken tüm bilgisini içerir.

## Hızlı Başlangıç

### 1. Ortamı Ayarlama

```bash
# Repoya klonla ve dizine gir
cd CENG454

# Sanal ortam oluştur
python -m venv .venv

# Sanal ortamı aktifle (Windows PowerShell)
.\.venv\Scripts\Activate.ps1

# Paketleri yükle (pinlenmiş versiyonlar)
pip install -r requirements.txt
```

### 2. Veriyi İndir ve Ön İşle

```bash
# Hugging Face'ten dataseti indir, temizle, split et
python scripts/download_prepare_dataset.py

# Split bilgilerini kaydet
python scripts/save_split_info.py

# QC örneği oluştur (manuel kontrol için)
python scripts/generate_sample_for_review.py
```

Çıktılar:
- `data/raw/turkish_fake_news_raw.csv` (local, repoya konmaz)
- `data/processed/train.csv` ve `test.csv` (shared split)
- `data/processed/split_info.json` (split metadata)
- `data/processed/sample_label_check.csv` (20 örnek manuel QC için)

### 3. TF-IDF Baseline Eğit (Person 1)

```bash
python src/train_tfidf.py
```

Çıktılar:
- `results/tfidf_metrics.csv` (metrikler)
- `results/confusion_matrix_tfidf.png` (confusion matrix)
- `models/tfidf_model.pkl` (model)
- `models/tfidf_vectorizer.pkl` (vectorizer)

### 4. Embedding Modelleri Eğit (Person 2)

```bash
python src/train_embeddings.py
```

Çıktılar:
- `results/embedding_lr_metrics.csv` (LR metrikleri)
- `results/embedding_svm_metrics.csv` (SVM metrikleri)
- `models/embedding_lr.pkl` ve `embedding_svm.pkl` (modeller)
- `results/confusion_matrix_embedding_*.png` (figürler)

### 5. Sonuçları Birleştir ve Rapor Hazırla (Person 3)

```bash
# Tüm metrikleri topla
python src/evaluate.py
```

Çıktılar:
- `results/final_results_table.csv` (tüm metriklerin özeti)
- `results/f1_comparison.png` (F1-score karşılaştırması)

---

## Kişi-Kişi Sorumlulukları

### Person 1: Data & Baseline

**Adımlar:**
1. `scripts/download_prepare_dataset.py` çalıştır → `data/processed/train.csv`, `test.csv`
2. `scripts/generate_sample_for_review.py` çalıştır → `sample_label_check.csv` elde et
3. Örnek dosyasında 20 haber inceleyip etiket doğruluğu kontrol et
4. `src/train_tfidf.py` çalıştır → TF-IDF baseline sonuçlarını üret
5. `results/tfidf_metrics.csv` ve `confusion_matrix_tfidf.png`'yi kontrol et

**Çıktılar:**
- `data/processed/train.csv`
- `data/processed/test.csv`
- `data/processed/split_info.json`
- `results/tfidf_metrics.csv`
- `results/confusion_matrix_tfidf.png`
- `models/tfidf_model.pkl`
- `models/tfidf_vectorizer.pkl`

**QC Kontrol Listesi:**
- [ ] Train.csv ve test.csv satır sayıları doğru mu? (4260 + 1066 = 5326)
- [ ] Label dağılımı uygun mu? (stratified, fake/real dengeli)
- [ ] TF-IDF accuracy > %90 mı?
- [ ] Örnek kontrol dosyasından 20 haber incelenip doğru etiketlendi mi?

---

### Person 2: Embeddings & Models

**Adımlar:**
1. `src/train_embeddings.py` çalıştır → embedding modelleri eğit (LR ve SVM)
   - Not: İlk çalıştırmada model (~100MB) indirilecek, bunu bekle
   - CPU'da ~5-15 dakika alabilir
2. `results/embedding_lr_metrics.csv` ve `embedding_svm_metrics.csv`'yi kontrol et
3. Confusion matrix figürlerini gözden geçir

**Çıktılar:**
- `results/embedding_lr_metrics.csv`
- `results/embedding_svm_metrics.csv`
- `results/confusion_matrix_embedding_lr.png`
- `results/confusion_matrix_embedding_svm.png`
- `models/embedding_lr.pkl`
- `models/embedding_svm.pkl`
- `models/embedding_metadata.pkl`

**İyileştirmeler (zaman izin verirse):**
- Random Forest sınıflandırıcısı test et
- Embedding caching yapıldığını kontrol et (hız için)
- Hipo parametrleri (C, kernel) fine-tune et

**QC Kontrol Listesi:**
- [ ] LR ve SVM modelleri başarıyla eğitildi mi?
- [ ] Metrikleri TF-IDF ile karşılaştır
- [ ] Embedding modeli başarıyla yüklendi mi?
- [ ] Embedding extraction hızı uygun mu (caching var mı)?

---

### Person 3: Evaluation & Report

**Adımlar:**
1. `src/evaluate.py` çalıştır → tüm metrikleri birleştir
   ```bash
   python src/evaluate.py
   ```
2. `results/final_results_table.csv` ve `f1_comparison.png` gözden geçir
3. Raporu yazarken bu dosyaları kaynak olarak kullan
4. `docs/ai-usage.md` güncelle (tüm AI yardımlarını logla)
5. `docs/reproducibility-checklist.md`'yi yeniden kontrol et

**Raporun İçeriği (8 sayfa max):**
1. Introduction (1 sayfa) - fake news problemi, proje hedefi
2. Background & Related Work (1.5 sayfa) - TF-IDF, embeddings, Sentence-BERT (min 10 referans)
3. Dataset & Preprocessing (1 sayfa) - veri kaynağı, temizleme, split
4. Methods (1.5 sayfa) - TF-IDF, embedding modelleri, sınıflandırıcılar
5. Evaluation & Results (1.5 sayfa) - metrikler, confusion matrices, karşılaştırma
6. Discussion & Limitations (0.5 sayfa) - sonuçlar, sınırlamalar
7. Conclusion (0.5 sayfa) - özet

**Çıktılar:**
- Rapor PDF (max 8 sayfa)
- Bibliography (IEEE format, min 10 kaynağa)
- `results/final_results_table.csv`
- `results/f1_comparison.png`
- `docs/reproducibility-checklist.md` (güncellenmiş)
- `docs/ai-usage.md` (güncellenmiş)

**Rapor Hazırlama Kontrol Listesi:**
- [ ] Tüm metrikleri `final_results_table.csv`'den al (manuel kopya yapma!)
- [ ] Her figür numara ve başlık taşısın
- [ ] Confusion matrix'leri raporun evaluation bölümüne ekle
- [ ] İeeE referans formatında en az 10 kaynak ekle
- [ ] Rapor max 8 sayfa
- [ ] PDF, `docs/` ve `results/` klasörlerindeki tüm gerekli dosyaları içer

---

## Paylaşılan Kontractlar ve Kurallar

### Veri İçeriği Kontractı
- **Text Column:** `text` (başlık + açıklama birleştirilmiş)
- **Label Column:** `label` (değerler: `fake` veya `real`)
- **Split:** 80% train, 20% test, stratified, `random_state=42`

### Dosya Adlandırması Standardı
Herkes aynı dosya adlarını kullanmalı:

```
data/raw/turkish_fake_news_raw.csv       (local, repoya konmaz)
data/processed/train.csv                 (Person 1 üretir)
data/processed/test.csv                  (Person 1 üretir)
data/processed/split_info.json           (split metadata)

results/tfidf_metrics.csv                (Person 1 üretir)
results/embedding_lr_metrics.csv         (Person 2 üretir)
results/embedding_svm_metrics.csv        (Person 2 üretir)
results/final_results_table.csv          (Person 3 üretir)

results/confusion_matrix_tfidf.png
results/confusion_matrix_embedding_lr.png
results/confusion_matrix_embedding_svm.png
results/f1_comparison.png

models/tfidf_model.pkl
models/tfidf_vectorizer.pkl
models/embedding_lr.pkl
models/embedding_svm.pkl
```

### Reproducibility Kuralları
- **Random State:** `random_state=42` (tüm yerler, tüm kişiler)
- **Train/Test Split:** Person 1 alır, diğerleri aynı `train.csv` ve `test.csv` kullanır
- **Python Seed:** `src.utils.set_seed(42)` her scriptin başında çağrılmalı
- **Paket Versiyonları:** `requirements.txt` pinlenmiş (yapılmış)

---

## Yardımcı Fonksiyonlar

### `src/utils.py`

```python
from src.utils import set_seed, save_metrics_csv, load_split_files

# Seed ayarla
set_seed(42)

# Metrikleri kaydet
save_metrics_csv(y_test, predictions, 'tfidf', 'LogisticRegression', 'results/my_metrics.csv')

# Splitleri yükle
train, test = load_split_files('data/processed')
```

### `src/embedding_utils.py`

```python
from src.embedding_utils import load_or_compute_embeddings

# Embeddings'i yükle (cache'den) veya hesapla
embeddings = load_or_compute_embeddings(
    texts,
    model_name='paraphrase-multilingual-MiniLM-L12-v2',
    cache_path='data/processed/embeddings_train.npy'
)
```

---

## Hata Giderme

### Problem: "ModuleNotFoundError: No module named 'sentence_transformers'"
**Çözüm:** `pip install -r requirements.txt` çalıştır

### Problem: Ham veri dosyası çok büyük
**Çözüm:** Ham veriyi repoya koymayın. `.gitignore`'a `data/raw/` eklenmiş.

### Problem: Train/test split diferentes
**Çözüm:** Tüm kişiler aynı `data/processed/train.csv` ve `test.csv` kullansın; `random_state=42` ile aynalanmış.

### Problem: Embedding modeli indirilemiyor
**Çözüm:** İnternet bağlantısı kontrol et, veya `cache_dir` ayarlarını kontrol et.

---

## İletişim ve Koordinasyon

### Günlük Senkronizasyon
- **Gün 1 Sonunda:** Person 1 split ve baseline'ı tamamlarsa, Person 2 ve 3 başlayabilir
- **Gün 2-3:** Tüm kişiler paralel çalışır
- **Gün 4-5:** Sonuçlar birleştirilir
- **Gün 6:** Rapor ve teslim

### Dosya Paylaşımı
- Git checkout ve pull yap, en son kod/sonuçlardan eminne olun
- Conflict varsa (ör. metrikleri raporla güncelledikten sonra), git merge strategy'yi belirle
- `docs/ai-usage.md` ve `docs/decisions.md` en önemli, güncellenmeyi unutma

### Hızlı Kontrol
- `docs/ai-usage.md` → kim ne yardım aldı
- `docs/decisions.md` → dataset ve model seçimleri
- `docs/reproducibility-checklist.md` → son kontrol

---

## Kaynaklar ve Referanslar

- [Hugging Face Datasets Dokümantasyonu](https://huggingface.co/docs/datasets)
- [Sentence-BERT](https://sbert.net/)
- [Scikit-learn Text Classification](https://scikit-learn.org/stable/modules/feature_extraction.html#text-feature-extraction)
- `docs/project-overview-turkish.md` → teknik terimler sözlüğü
- `docs/usage.md` → kısa komut referansı

---

**Son Güncelleme:** 2026-05-12
**Sorumlu:** Person 3 (güncel tutmak)
