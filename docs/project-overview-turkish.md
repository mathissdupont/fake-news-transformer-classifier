# CENG454 Sahte Haber Tespiti Projesi - Tam Açıklama

Bu belgede projenin ne olduğu, neden yaptığımız ve hangi teknolojileri kullandığımız Türkçe olarak, terimin anlamlarıyla birlikte açıklanmıştır.

## 1. Proje Nedir? (En Basit Tanım)

Biz, **haberlerin sahte olup olmadığını otomatik olarak tespit etmeye çalışan bir yapay zeka programı** yapmaya çalışıyoruz.

- Girdi (Input): Bir haber metni
- Çıktı (Output): "Bu haber sahte" veya "Bu haber gerçek"
- Amaç: İnsan müdahalesi olmadan doğru tahminde bulunmak

---

## 2. Sahte Haberler Neden Sorun?

Sahte haberler (yanlış/yanıltıcı bilgi):
- **Halktaki güveni zedeler** - İnsanlar kime inanacaklarını bilmez
- **Siyaseti etkiler** - Yanlış bilgi üzerine kararlar alınır
- **Ekonomiye zarar verir** - Panika ve belirsizlik yaratır
- **Toplumsal kriz yaratır** - İnsanlar birbirine düşer

Örnek:
- "Kütüphanedeki kitaplar zehirli" (yalan)
- "Belirli bir grup tehlikeli" (genelleme)

Tüm bu haberler el ile kontrol edilemez, bu yüzden **otomatik bir sistem lazım**.

---

## 3. Metin Sınıflandırması (Text Classification) Nedir?

Metin sınıflandırması: bir metni **önceden tanımlı kategorilerden birine koymak**.

Örnekler:
- Email: "Spam" veya "Normal" kategorisine koyma
- Yorum: "Olumlu", "olumsuz" veya "nötr" kategorisine koyma
- **Bizim projede:** Haber "Sahte" veya "Gerçek" kategorisine koyma

Bu işi yapan sisteme **sınıflandırıcı (classifier)** denir.

---

## 4. Bilgisayar Metni Nasıl Anlar?

Bilgisayarlar metni direkt anlayamaz. Metni **sayıya çevirmek** gerekir.

### Analoji:
- İnsan: "Bu haberde 'hükümet' kelimesi tekrar tekrar geçiyor, muhtemelen siyasi bir haber" (anlamlı düşünce)
- Bilgisayar: "Bu metinde 532 farklı kelime var, 'hükümet' kelimesi 15 kez geçiyor" (sayısal veri)

Metni sayıya çevirmenin iki yöntemi var:

---

## 5. Yöntem 1: TF-IDF (Baseline/Temel Çizgi)

**TF-IDF** = "Term Frequency - Inverse Document Frequency" (Kelime Sıklığı - Ters Belge Sıklığı)

### Nasıl çalışır?

1. **Tüm kelimeleri say:**
   - Metinde "hükümet" 15 kez geçti
   - Metinde "ülke" 8 kez geçti
   - Metinde "yapı" 3 kez geçti

2. **Önemli kelimeleri belirle:**
   - Eğer "hükümet" tüm haberlerde geçiyorsa, çok önemli değil (TF-IDF'de önemi azalır)
   - Eğer "hükümet" sadece belirli haberler türünde geçiyorsa, çok önemli (TF-IDF'de önemi artar)

3. **Sonuç:**
   - Her haber bir **sayı vektörüne** (liste) çevrilir
   - Örnek: [0.5, 0.3, 0.1, 0.2, ...]
   - Bu vektör haberin "imzası" olur

### Avantajları:
- Hızlı
- Anlaşılması kolay
- Az bilgisayar gücü gerekli

### Dezavantajları:
- Kelimeleri **ayrı ayrı** görür, **anlam** görmez
- "Kedi" ile "köpek" arasındaki benzerliği anlamaz (ikisi de hayvan)
- Kelime sırası önemli değil (cümleler karışıtırılsa da aynı sonuç verir)

---

## 6. Yöntem 2: Transformer ve Sentence-BERT (Embedding)

**Transformer** = Yapay zeka mimarisi, Google tarafından icat edilmiş, **anlam** düzeyinde metni anlar

**BERT** = "Bidirectional Encoder Representations from Transformers"
- Çift yönlü okuma (soldan sağa ve sağdan sola)
- Bağlamdan anlam çıkarma

**Sentence-BERT** = BERT'ü özellikle **cümleler ve metinler için** optimize etmiş versiyon

### Nasıl çalışır?

1. **Metin veriliyor:**
   - "Hükümet yeni yasa çıkardı"

2. **Model okur ve anlam çıkarır:**
   - "hükümet" + "yasa" = kanun yapma eylemi
   - "yeni" + "yasa" = yenilik

3. **Sonuç:**
   - Haberin **anlamını** 384 boyutlu bir vektöre (sayı listesi) çevir
   - Örnek: [0.123, 0.456, 0.789, ..., 0.234]
   - Bu vektör haberin "anlam imzası" olur

### Avantajları:
- **Anlam** düzeyinde anlar
- "Kedi" ile "köpek" arasındaki benzerliği bulabilir
- Bağlama duyarlı (aynı kelime farklı anlamlarda farklı şekilde kodlanır)

### Dezavantajları:
- Daha yavaş
- Daha fazla bilgisayar gücü gerekli
- Eğitim için daha fazla veri gerekli

---

## 7. Vektör (Vector) Nedir?

**Vektör** = Bir harita gibi sayılar.

Analoji:
- Harita: Şehrin konumunu (X, Y) koordinatları ile gösterir
- Vektör: Haberin özelliklerini sayı listesiyle gösterir

Örnek:
- TF-IDF vektörü: [0.5, 0.3, 0.1, 0.2, 0.0] (5 boyutlu)
- Embedding vektörü: [0.123, 0.456, 0.789, ..., 0.234] (384 boyutlu)

**Boyut sayısı ne kadar fazlaysa, haberin daha ayrıntılı özellikleri kaydedilir.**

---

## 8. Sınıflandırıcı (Classifier) Nedir?

Vektörü elde ettikten sonra, bu vektöre bakarak **"Sahte" mi "Gerçek" mi** karar veren sistem = Sınıflandırıcı.

**Logistic Regression** (En basit):
- Matematiksel bir çizgi çizer
- Vektörü bu çizginin hangi tarafında olduğuna bakarak karar verir
- Basit ama etkili

**Support Vector Machine (SVM)** (Daha güçlü):
- Daha karmaşık sınırlar çizer
- Çok boyutlu uzayda çalışır
- Logistic Regression'dan daha güçlü

**Random Forest** (İsteğe bağlı):
- Birçok karar ağacını birleştirir
- Soru sorarak sonuca ulaşır (Evet/Hayır)

---

## 9. Projemizin Amacı Nedir?

Biz, **TF-IDF yöntemi** ile **Sentence-BERT embedding yöntemi**ni karşılaştırmak istiyoruz.

**Soru:** "Anlam tabanlı embedding'ler TF-IDF'ten daha mı iyi?"

Bunu test etmek için:
1. **Aynı veri seti** ile her iki yöntemi eğiteceğiz
2. **Aynı test seti** ile her iki yöntemi test edeceğiz
3. **Sonuçları karşılaştıracağız**

Beklentimiz: **Embedding'ler daha iyi sonuç versin** (çünkü anlam daha iyi yakalayıp, yanlışları daha az yapar)

---

## 10. Veri Seti (Dataset) Nedir?

**Veri seti** = Eğitim için kullanılan örnek haberler.

Yapısı:
```
Metin                                  | Etiket
"Hükümet yeni yasa çıkardı..."        | Gerçek
"Kütüphane kitapları zehirli..."      | Sahte
"Teknoloji şirketi yeni ürün..."      | Gerçek
...
```

**Biz en az 1000 haber örneği arıyoruz:**
- Bazı haberler **eğitim için** (train set) = modelı öğretir
- Bazı haberler **test için** (test set) = modelı sınar

Örnek bölünüş:
- 80% eğitim (800 haber)
- 20% test (200 haber)

---

## 11. Metrikleri Anlamak (Başarı Nasıl Ölçülür?)

Model başarısını ölçmek için 4 metrik kullanırız:

### A. Doğruluk (Accuracy)
"Kaç tanesini doğru tahmin ettik?"

Örnek:
- 100 haber test ettik
- 85'ini doğru tahmin ettik
- Accuracy = 85%

**Ama bu her zaman yeterli değil!** Çünkü eğer veri seti çarpık ise (80% gerçek, 20% sahte), model sadece "Hepsi gerçek" dese bile %80 accuracy alır. Bu aldatıcıdır.

### B. Kesinlik (Precision)
"Sahte dediğimiz haberlerin kaçı gerçekten sahtedir?"

Örnek:
- Model "sahte" dedi: 20 haber
- Bunların 18'i gerçekten sahte
- Kesinlik = 18/20 = 90%

**Kullanım:** Yanlış pozitif (gerçek haberi sahte demeyi) cezalandırmak istiyorsak önemlilir.

### C. Hatırlama (Recall)
"Gerçek sahte haberlerin kaçını bulabildi?"

Örnek:
- Veri setinde 50 sahte haber var
- Model bunların 45'ini buldu
- Hatırlama = 45/50 = 90%

**Kullanım:** Yanlış negatif (sahte haberi kaçırmayı) cezalandırmak istiyorsak önemlilir.

### D. F1-Skoru
Kesinlik ve Hatırlama'nın **dengeli ortalaması**.

Formül: F1 = 2 × (Kesinlik × Hatırlama) / (Kesinlik + Hatırlama)

**En önemlisi:** Veri seti çarpık (disbalans) ise, Accuracy yerine F1-Skoru bakılmalı.

---

## 12. Confusion Matrix (Karmaşıklık Matrisi) Nedir?

Modelin hatalarını görmek için 2×2 tablo:

```
                   Tahmin: Sahte    Tahmin: Gerçek
Gerçek: Sahte           45               5         (50 sahte haber)
Gerçek: Gerçek           3              47         (50 gerçek haber)
```

**Açıklama:**
- **45** = Doğru tahmin (Sahte → Sahte)
- **5** = Yanlış negatif (Sahte haberi Gerçek dedi) - KÖTÜ
- **3** = Yanlış pozitif (Gerçek haberi Sahte dedi) - KÖTÜ
- **47** = Doğru tahmin (Gerçek → Gerçek)

Bu tabloya bakarak modelun **nerede hata yaptığını** görürüz.

---

## 13. Sonuç: Projemiz Özet

| Adım | Ne Yapıyoruz? | Neden? |
|------|---------------|--------|
| 1. Veri Seti | Sahte/Gerçek haberleri toplıyoruz | Modelı eğitmek için |
| 2. TF-IDF Vektör | Kelimeleri sayı listelerine çeviriyoruz | Baseline oluşturmak için |
| 3. TF-IDF Model | Logistic Regression ile eğitiyoruz | Basit bir tahmin sistemi |
| 4. Embedding Vektör | Sentence-BERT ile anlam tabanlı vektör oluşturuyoruz | Daha iyi temsil için |
| 5. Embedding Model | Logistic Regression ve SVM ile eğitiyoruz | Embedding'in gücünü test etmek için |
| 6. Karşılaştırma | TF-IDF vs Embedding sonuçlarını karşılaştırıyoruz | Hangisinin daha iyi olduğunu görmek için |
| 7. Rapor | Sonuçları yazıyoruz ve bulgularımızı açıklıyoruz | İlim olsun diye |

---

## 14. Sözlük (Tüm Teknik Terimler)

| Terim | Türkçe | Açıklama |
|-------|--------|----------|
| **Fake News** | Sahte Haber | Yanlış, yanıltıcı veya uydurma haber |
| **Text Classification** | Metin Sınıflandırması | Metinleri kategorilere ayırma |
| **TF-IDF** | Terim Frekansı - Ters Belge Frekansı | Kelimelerin önemine göre vektör oluşturma |
| **Transformer** | Transformatör | Anlam tabanlı AI mimarisi |
| **BERT** | - | Google'ın iki yönlü dil modeli |
| **Sentence-BERT** | - | Cümleler için optimizlenmiş BERT |
| **Embedding** | Gömü | Metnin anlamını sayı listesi olarak gösterme |
| **Vector** | Vektör | Sayı listesi |
| **Classifier** | Sınıflandırıcı | Kategori tahmini yapan sistem |
| **Logistic Regression** | - | Basit bir sınıflandırma algoritması |
| **SVM** | Destek Vektör Makinesi | Güçlü bir sınıflandırma algoritması |
| **Train Set** | Eğitim Seti | Modeli öğretmek için kullanılan veriler |
| **Test Set** | Test Seti | Modeli sınamak için kullanılan veriler |
| **Accuracy** | Doğruluk | Toplam doğru tahminin yüzdesi |
| **Precision** | Kesinlik | Pozitif tahminlerin kaç tanesinin doğru olduğu |
| **Recall** | Hatırlama/Duyarlılık | Tüm pozitif örneklerin kaç tanesini bulduğu |
| **F1-Score** | F1-Skoru | Kesinlik ve Hatırlama'nın dengeli ortalaması |
| **Confusion Matrix** | Karmaşıklık Matrisi | Tahminlerin ve gerçek değerlerin 2×2 tablosu |
| **Dataset** | Veri Seti | Eğitim için kullanılan örnek verileri |
| **Metric** | Metrik | Başarıyı ölçme kriteri |
| **Model** | Model | Tahmin yapabilen eğitilmiş sistem |

---

## 15. Başlamadan Önce Bilmen Gereken Şey

Bu proje sadece **yazılım geliştirmesi** değil, **bilimsel deney**dir.

- Biz önceden sonucu bilmiyoruz
- TF-IDF daha iyi çıkabilir, embedding daha iyi çıkabilir
- Sonuç ne olursa olsun, **neden** olduğunu açıklamalıyız
- Metrikler **gerçek değerler** olmalı, uydurulmuş değil
- Tüm adımlar **tekrarlanabilir** olmalı

**Herkes kendi görevini yapıp AI kullanımını kayıt etsin, ama her sonuç insan tarafından doğrulanmalı.**

---

Bu belgede anlamadığın bir şey varsa, soru sor. Takımın herkesin aynı seviyede anlama sahip olması önemli.
