# 6 Günlük Sprint Planı

**Başlangıç:** Bugün (12 Mayıs 2026)
**Deadline:** 22 Mayıs 2026 Saat 23:59
**Toplam gün:** 6 tam gün + 5 gün tampon

## Kritik Varsayımlar

- Veri seti kolaylıkla bulunur (Turkish fake news veya WELFake)
- Takım üyeleri günde 4-6 saat çalışabilir
- Python ortamı hazır (✓ zaten kurulu)
- İnternet erişimi sorunsuz

## Zaman Bölümü

| Gün | Tema | Kimin Başı Döner |
|-----|------|------------------|
| 1 (Bugün) | Araştırma + Dataset | Hepsi |
| 2 | Veri Temizleme | Kişi 1 |
| 3 | TF-IDF + Embedding | Kişi 1 + 2 |
| 4 | Model Eğitimi | Kişi 2 + 3 |
| 5 | Sonuç Birleştirme | Kişi 3 |
| 6 | Rapor + Teslim | Kişi 3 + Hepsi |

**Tampon:** 5 gün düzeltme ve ek çalışma (deadline 22 Mayıs saat 23:59)

---

## GÜN 1: BUGÜN

### Ana Görev
Dataset bulma ve karar verme + Araştırma başlatma

### Kişi 1 (Veri)
**Sabah (2-3 saat):**
- [ ] Hocanın önerdiği 5 kaynağı aç (HF, Kaggle, Google Dataset Search, Papers With Code, TÜİK)
- [ ] "Turkish fake news" ve "WELFake" arat
- [ ] En az 3 aday dataset bul (metin + ikili etiket şartı)
- [ ] Hangi dataseti seçeceğini yazılı not et

**Öğleden sonra (2-3 saat):**
- [ ] Seçilen dataseti indir (raw veri)
- [ ] `data/raw/` klasörüne koy
- [ ] Satır sayısı, kolon adları, label dağılımını not et
- [ ] `data/README_dataset_link.txt` dosyasını doldur
- [ ] Seçimi `docs/decisions.md` içine yazılı tuttur

**Teslim:** Dataset dosyası + link bilgisi + veri özeti

### Kişi 2 (Model)
**Sabah-Öğleden sonra (3-4 saat):**
- [ ] "Sentence-BERT" ve "paraphrase-multilingual" ara
- [ ] Hangi embedding modelini kullanacağını kararlaştır
- [ ] Model adını `docs/decisions.md` içine yaz
- [ ] Modeli test et (import et, çalışıp çalışmadığını kontrol et)

**Teslim:** Model seçim kararı + test notu

### Kişi 3 (Rapor)
**Sabah-Öğleden sonra (3-4 saat):**
- [ ] "IEEE citation format" örnekleri ara
- [ ] Proje başlığına uygun 5 makale başlık ara
- [ ] Rapor template taslağı yaz (kaç bölüm, kaç sayfa?)
- [ ] `docs/research-notes.md` dosyasını başlat

**Teslim:** Rapor taslağı + 5 makale adayı

**GÜNÜN SONU:** Takım toplantısı 1 saat
- Veri seti onay
- Model onay
- Rapor planı onay
- AI kullanımlarını `docs/ai-usage.md` güncelle

---

## GÜN 2: YARINDA

### Ana Görev
Veri temizleme ve preprocessing

### Kişi 1 (Veri)
**Sabah (3-4 saat):**
- [ ] Dataseti Python'da aç (pandas)
- [ ] Eksik değerleri kontrol et
- [ ] Duplikatlı satırları bul
- [ ] Label dağılımını görselleştir (kaç sahte, kaç gerçek?)
- [ ] Veri kalitesi notunu yaz

**Öğleden sonra (3-4 saat):**
- [ ] Temizleme kodunu yaz (eksik satırlar, duplikatlar, type casting)
- [ ] Train/test split yap (80/20, stratified, random_state=42)
- [ ] `data/processed/train.csv` ve `data/processed/test.csv` kaydet
- [ ] `src/preprocessing.py` yazılı tut

**Teslim:** 
- Temizlenmiş train/test dosyaları
- Preprocessing script
- Veri kalitesi raporu

### Kişi 2 (Model)
**Sabah-Öğleden sonra (3-4 saat):**
- [ ] Embedding modeli (paraphrase-multilingual) test et
- [ ] Train seti için embedding çıkar (demo olarak 100 örnek)
- [ ] Runtime ve memory use'ı not et
- [ ] Caching stratejisini planla

**Teslim:** Test embedding + runtime notu

### Kişi 3 (Rapor)
**Sabah-Öğleden sonra (3-4 saat):**
- [ ] İlk 2 makaleyi oku ve özet yaz
- [ ] Rapor 1. bölüm (Giriş) taslağını yaz
- [ ] Rapor 2. bölüm (Background) başını yaz

**Teslim:** Rapor ilk 2 bölümün taslağı

**GÜNÜN SONU:** Takım kontrol
- Veri temizleme onay
- Embedding test sonuçları kontrol
- Rapor ilerleme kontrol
- AI günlüğü güncelle

---

## GÜN 3: +2 GÜN SONRA

### Ana Görev
TF-IDF baseline ve embedding eğitim başlatılması

### Kişi 1 (Veri)
**Sabah (2-3 saat):**
- [ ] TF-IDF + Logistic Regression script yaz
- [ ] Train seti üzerinde eğit
- [ ] Test seti üzerinde tahmin yap
- [ ] Metrics hesapla (accuracy, precision, recall, F1)
- [ ] Confusion matrix üret

**Öğleden sonra (2-3 saat):**
- [ ] Sonuçları `results/tfidf_metrics.csv` olarak kaydet
- [ ] Confusion matrix'i `results/confusion_matrix_tfidf.png` olarak kaydet
- [ ] TF-IDF model'i `models/tfidf_model.pkl` olarak kaydet
- [ ] Kısa hata analizi yaz

**Teslim:**
- TF-IDF script
- Metrikleri CSV
- Confusion matrix figürü
- Baseline başarı oranı

### Kişi 2 (Model)
**Sabah (2-3 saat):**
- [ ] Embedding extraction script yaz
- [ ] **Tüm** train seti için embedding çıkar
- [ ] **Tüm** test seti için embedding çıkar
- [ ] Embeddings'i cache et (numpy array olarak)

**Öğleden sonra (2-3 saat):**
- [ ] Logistic Regression classifier eğit
- [ ] SVM classifier eğit
- [ ] Her ikisinin tahminlerini al
- [ ] Metrikleri hesapla

**Teslim:**
- Embedding extraction script
- Cached embeddings (test + train)
- Logistic Regression + SVM metrikleri

### Kişi 3 (Rapor)
**Sabah-Öğleden sonra (3-4 saat):**
- [ ] Rapor 3. bölüm (Dataset & Preprocessing) yaz (asıl sonuçları bekliyorken)
- [ ] Rapor 4. bölüm (Proposed Approach) yapısını düzen
- [ ] 3 daha fazla makale oku ve özet yaz

**Teslim:** Rapor 3-4. bölümün taslağı

**GÜNÜN SONU:** Takım kontrol
- TF-IDF sonuçları incelensin
- Embedding metrikleri karşılaştırılsın
- Rapor ilerleme kontrol

---

## GÜN 4: +3 GÜN SONRA

### Ana Görev
Tüm model sonuçları tamamlanması

### Kişi 1 (Veri)
**Sabah-Öğleden sonra (2-3 saat):**
- [ ] Random Forest'ı test et (isteğe bağlı, zaman kalıyorsa)
- [ ] Tüm sonuçları kontrol et
- [ ] Veri ve baseline bölümü bitirildi mi kontrol et

**Teslim:** Tamamlanmış baseline görevler

### Kişi 2 (Model)
**Sabah (2-3 saat):**
- [ ] Random Forest classifier test et (eğer zaman varsa)
- [ ] Tüm metrikleri `results/embedding_*_metrics.csv` kaydet
- [ ] Tüm confusion matrix'leri `results/confusion_matrix_embedding_*.png` kaydet
- [ ] Hata analizi yazılı tut

**Öğleden sonra (2-3 saat):**
- [ ] Embedding görevlerini finalize et
- [ ] Runtime ve memory notlarını yaz
- [ ] Kişi 3'e "sonuç dosyaları hazır" bilgisi ver

**Teslim:**
- Tüm embedding metrikleri
- Tüm confusion matrix'ler
- Hata analizi

### Kişi 3 (Rapor)
**Sabah-Öğleden sonra (3-4 saat):**
- [ ] Evaluation bölümü yazısı (Kişi 1 ve 2'nin sonuçlarını bekliyorken)
- [ ] 2 daha fazla makale oku
- [ ] Bibliography draft tamamla (10+ kaynak)

**Teslim:** Rapor evaluation bölümü

**GÜNÜN SONU:** Birleştirme başlasın
- Tüm metrikleri Kişi 3 al
- Tüm figürleri Kişi 3 al
- AI günlüğü güncelle

---

## GÜN 5: +4 GÜN SONRA

### Ana Görev
Sonuçları birleştirme, karşılaştırma, rapor tamamlanması

### Kişi 1 & 2
**Sabah (1-2 saat):**
- [ ] Kişi 3'e son sorular varsa cevapla
- [ ] Metrikleri kontrol et
- [ ] Rapor için gerekli ek açıklamalar yaz

**Teslim:** Tamamlanmış görevler

### Kişi 3 (Rapor)
**Sabah-Öğleden sonra (5-6 saat):**
- [ ] Tüm metrikleri bir tabloda birleştir
  - TF-IDF Accuracy, Precision, Recall, F1
  - Embedding LR Accuracy, Precision, Recall, F1
  - Embedding SVM Accuracy, Precision, Recall, F1
  - (Embedding RF - eğer yapıldıysa)
- [ ] Sonuçları visualize et (karşılaştırma grafiği)
- [ ] TF-IDF vs Embedding'in hangisinin daha iyi olduğu yaz
- [ ] Discussion bölümü tamamla
- [ ] Conclusion bölümü tamamla
- [ ] Limitations bölümü tamamla

**Rapor Yazma Sırası:**
1. Introduction (Giriş)
2. Background & Related Work (Geçmiş çalışmalar) - 10 kaynak
3. Dataset & Preprocessing (Veri)
4. Proposed Approach (Yöntem)
5. Evaluation & Results (Sonuçlar ve grafikler)
6. Discussion & Limitations (Tartışma)
7. Conclusion & Future Work (Sonuç)
8. References (Kaynakça) - IEEE format

**Teslim:**
- Tamamlanmış rapor draft
- Bir grafik (TF-IDF vs Embedding karşılaştırması)
- Finalize edilmiş kaynakça

**GÜNÜN SONU:** Rapor okunması
- Grammatik kontrol
- Uzunluk kontrol (8 sayfa?)
- Tüm figürler sayılı mı?

---

## GÜN 6: +5 GÜN SONRA

### Ana Görev
Final kontroller, PDF yapma, teslim paketi hazırlama

### Kişi 1, 2, 3 (Hepsi)
**Sabah (2-3 saat):**
- [ ] Rapor son okuması yapılsın
- [ ] Yazım hataları düzeltilsin
- [ ] Figürlerin kalitesi kontrol edilsin
- [ ] Metriklerin doğru kopyalandığı kontrol edilsin

**Öğleden sonra (2-3 saat):**
- [ ] Rapor PDF'ye çevrilsin
- [ ] Teslim paketi hazırlansın:
  ```
  CENG454_Group_X/
  ├── report/
  │   └── final_report.pdf
  ├── notebooks/
  │   └── (exploration code varsa)
  ├── src/
  │   ├── preprocessing.py
  │   ├── train_tfidf.py
  │   ├── train_embeddings.py
  │   └── evaluate.py
  ├── results/
  │   ├── tfidf_metrics.csv
  │   ├── embedding_lr_metrics.csv
  │   ├── embedding_svm_metrics.csv
  │   ├── confusion_matrix_tfidf.png
  │   ├── confusion_matrix_embedding_lr.png
  │   ├── confusion_matrix_embedding_svm.png
  │   └── error_analysis.md
  ├── data/
  │   └── README_dataset_link.txt
  ├── README.md
  ├── docs/
  │   ├── ai-usage.md (TAMAMLANMIŞ)
  │   ├── decisions.md
  │   └── team-split.md
  └── requirements.txt
  ```
- [ ] `docs/ai-usage.md` kontrol edilsin (tüm AI kullanımları kayıtlı mı?)
- [ ] `docs/reproducibility-checklist.md` kontrol edilsin

**Akşam (1-2 saat):**
- [ ] ZIP dosyası oluşturulsun: `CENG454_Group_X.zip`
- [ ] ZIP dosyasının boyutu kontrol edilsin (çok fazla mı?)
- [ ] Teslim platformuna (UBYS) yüklenişi hazırlanılsın

**Teslim:**
- Final PDF rapor
- ZIP dosyası

---

## 7-11. GÜNLER: TAMPON (+6 ile +10 GÜN ARASINDA)

Eğer herhangi bir şey gecikmişse:
- [ ] Rapor düzeltmeleri
- [ ] Metriklerin yeniden kontrol edilmesi
- [ ] Figürlerin iyileştirilmesi
- [ ] Kaynakça çoğaltılması (10'dan fazla kaynak)
- [ ] Demo sunuşu hazırlığı

---

## Başarılı Olabilir mi?

**EVET, eğer:**

✓ Veri seti ilk gün bulunur
✓ Görevler paralel yapılır
✓ Her gün deadline'lar tutulur
✓ AI günlüğü sürekli güncellenirse
✓ Son günde panic yapılmazsa

**HAYIR, eğer:**

✗ Veri seti uzun süre aranırsa
✗ Embedding modeli yüklenmezse / çalışmazsa
✗ Biri hastalanırsa
✗ Metrikleri elle girmeye çalışılırsa (uydurulmuş olur)
✗ Son dakikalara bırakılırsa

---

## Kritik Başarı Faktörleri

### 1. Paralel Çalışma (HEM ÖNEMLİ!)
- Kişi 1 veri alıyor ⟶ Kişi 2 ve 3 kendi işlerine başlasın
- Beklemek yok. Her biri kendi görevini yapıyor.

### 2. Veri Seti Erken Bulunmalı
- Gün 1 sonunda dataset kullanıma hazır olmalı
- Aksi halde tüm zaman çizelgesi bozulur

### 3. Metriklerin Gerçek Olması
- **ÖNEMLİ:** Hocanın belgelerinde: "Results should not be invented"
- Her metrik kod tarafından çalıştırılarak hesaplanmalı
- Hiçbir şey aklıktan yazılmamalı

### 4. Reproducibility
- Her script ayrı çalışabilmeli
- Hiç kimse başka birinin veri dosyasını beklememelidir
- Train/test split'ler sabit (random_state=42)

### 5. AI Günlüğü Gerçek Zamanlı
- Her AI kullanımı TESCİL EDİLMELİ
- Raporunda da "we used ChatGPT for..." demeli
- Sahtekarlık olarak görülmemelidir

---

## Son Tavsiye

**Bunu hemen başlayın. Dur dur durma. 6 gün göz açıp kapayıncaya kadar geçer.**

Kişi 1, hemen dataset ara.
Kişi 2, hemen embedding modelini test et.
Kişi 3, hemen kaynakça arama başlat.

**YOL HARITA HAZIR, ÇALIŞA BAŞLA!**
