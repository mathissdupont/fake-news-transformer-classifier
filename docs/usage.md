Kullanım Talimatı
=================

Bu doküman proje içindeki temel scriptlerin nasıl çalıştırılacağını açıklar.

Önkoşullar
- Python 3.10+ (veya projedeki venv) yüklü olmalıdır.
- Sanal ortam aktifse komut örneklerindeki `python` çağrısını doğrudan kullanabilirsiniz.

Ortamı ayarlama (venv aktifse):

```bash
# Windows PowerShell (örnek):
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Veri indirme ve ön işleme
- Bu proje ham veriyi repoya koymaz; veriyi yeniden üretmek için aşağıdaki scripti çalıştırın.

```bash
python scripts/download_prepare_dataset.py
```

Bu script şu dosyaları oluşturur:
- `data/raw/turkish_fake_news_raw.csv` (local)
- `data/processed/turkish_fake_news_processed_full.csv`
- `data/processed/train.csv` ve `data/processed/test.csv` (80/20 stratified, `random_state=42`)

QC örneği (20 örnek):

```bash
python scripts/generate_sample_for_review.py
# Çıktı: data/processed/sample_label_check.csv
```

TF-IDF baseline eğitimi ve değerlendirme

```bash
python src/train_tfidf.py
# Çıktılar:
# - results/tfidf_metrics.csv
# - models/tfidf_model.pkl
# - models/tfidf_vectorizer.pkl
# - results/confusion_matrix_tfidf.png
```

Embedding tabanlı modeller (Person 2)

```bash
python src/train_embeddings.py
# Bu script Sentence-Transformers modelini indirir ve LR+SVM sonuçlarını kaydeder
# Çıktılar: results/embedding_lr_metrics.csv, results/embedding_svm_metrics.csv, models/*.pkl
```

Notlar
- Ham veriyi repoya koymayın; `data/README_dataset_link.txt` dosyasında indirmenin nasıl tekrar edileceği ve lisans bilgisi bulunur.
- Her AI destekli adımı `docs/ai-usage.md` dosyasına eklemeyi unutmayın.
