# Bugüne Kadar Yaptıklarımız

**Tarih:** 12 Mayıs 2026
**Proje:** CENG454 Fake News Detection

Bu not, şimdiye kadar yaptıklarımızı kısa ve anlaşılır biçimde özetler. Amaç sadece ne yaptığımızı değil, hangi teknik parçaların ne anlama geldiğini de netleştirmektir.

---

## 1. Şimdiye Kadar Ne Yaptık?

- Türkçe sahte haber veri setini bulduk ve seçtik.
- Veri setini indirip işledik.
- Ham veriden `text` ve `label` yapısına geçtik.
- Veriyi eğitim ve test olarak 80/20 oranında böldük.
- Bu bölmeyi yeniden üretilebilir yapmak için `random_state=42` kullandık.
- TF-IDF tabanlı ilk modeli çalıştırdık.
- Logistic Regression ile sonuç aldık ve metrikleri kaydettik.
- Embedding tabanlı yaklaşım için model seçtik ve eğitim altyapısını hazırladık.
- Sonuçların toplanması için değerlendirme scripti hazırladık.
- Takımın birlikte çalışabilmesi için kurulum ve iş akışı notları yazdık.
- Yapay zekayı nasıl kullandığımızı ayrıca `docs/ai-usage.md` içinde kayda geçirdik.

---

## 2. Bunlar Teknik Olarak Ne Demek?

### Veri Seti
Modelin öğreneceği örnek haberlerin tamamıdır. Bizim durumda veri seti Türkçe haber metinlerinden oluşuyor ve her örnek ya `fake` ya da `real` etiketi taşıyor.

### Ham Veri
İlk indirilen, henüz temizlenmemiş veri dosyasıdır. Genellikle doğrudan kullanılmaz; önce düzenlenir, sadeleştirilir ve analiz için uygun hale getirilir.

### Ön İşleme
Veriyi modele uygun hale getirme sürecidir. Biz burada başlık ve açıklamayı birleştirip tek bir metin alanı oluşturdık, etiketleri de `fake` ve `real` olarak standartlaştırdık.

### Train/Test Split
Veriyi iki parçaya ayırma işlemidir. `train` modeli eğitmek için, `test` ise modelin daha önce görmediği veride ne kadar iyi olduğunu ölçmek için kullanılır.

### Stratified Split
Veriyi ayırırken sınıf oranlarını koruma yöntemidir. Yani fake ve real haberlerin oranı train ve test tarafında da benzer kalır.

### `random_state=42`
Rastgele işlemleri sabitleyen sayıdır. Aynı veride tekrar çalıştırıldığında aynı split ve aynı sonuçların alınmasına yardım eder.

### TF-IDF
Metindeki önemli kelimeleri sayısal hale getiren klasik yöntemdir. Bir kelime bir haberde sık geçiyor ama tüm haberlerde çok sık geçmiyorsa daha önemli kabul edilir.

### Logistic Regression
İkili sınıflandırma için kullanılan basit ve güçlü bir algoritmadır. Biz bunu TF-IDF özellikleri üzerinde kullandık.

### Embedding
Bir metni anlamını kısmen koruyan yoğun sayısal vektöre dönüştürme işlemidir. TF-IDF gibi sadece kelime frekansına bakmak yerine cümlenin anlamsal yapısını da yakalamaya çalışır.

### Sentence-BERT
Metinleri embedding'e dönüştürmek için kullanılan bir model ailesidir. Biz Türkçe için çok dilli bir model seçtik: `paraphrase-multilingual-MiniLM-L12-v2`.

### Cache
Önceden hesaplanan sonuçları diskte saklama yöntemidir. Embedding hesaplamak zaman alabildiği için aynı veriyi tekrar hesaplamamak adına cache kullanıyoruz.

### Metrikler
Model başarısını ölçen sayılardır.
- **Accuracy:** Doğru tahminlerin toplam oranı.
- **Precision:** Sahte diye işaretlenenlerin ne kadarının gerçekten sahte olduğu.
- **Recall:** Gerçek sahte haberlerin ne kadarını yakaladığımız.
- **F1-Score:** Precision ve recall arasında denge kuran özet ölçü.

### Confusion Matrix
Modelin hangi sınıfları ne kadar doğru veya yanlış tahmin ettiğini gösteren tablo ya da görseldir.

---

## 3. Neden Bu Yolu Seçtik?

- Önce veri ve basit baseline kurduk, çünkü hızlıca gerçek bir referans sonuç görmek istedik.
- TF-IDF ile başlamak, embedding yaklaşımına göre daha hızlı ve yorumlanabilir olduğu için mantıklıydı.
- Embedding tarafını ayrıca hazırlamamızın sebebi, daha güçlü anlamsal temsil ile sonuçları karşılaştırmak istememiz.
- AI kullanımını ayrıca loglamamızın sebebi, projede neyin nasıl üretildiğini şeffaf biçimde göstermek.

---

## 4. Şu Anki Durum

- Veri hattı hazır.
- TF-IDF baseline tamamlandı.
- Embedding çalışmaları için altyapı hazır.
- Değerlendirme ve rapor yazımı için gereken dosyalar hazırlanıyor.

---

## 5. Bir Sonraki Adım

- Embedding eğitimini çalıştırmak.
- Sonuçları `results/` altında toplamak.
- Karşılaştırma tablosunu çıkarmak.
- Raporu yazarken bu notu teknik açıklama referansı olarak kullanmak.

---

## Kısa Özet

Bugüne kadar veri setini hazır hale getirdik, ilk modeli kurduk, sonuçları kaydettik ve takımın paralel çalışabileceği altyapıyı oluşturduk. En önemli öğrenme noktası şu: hızlı ilerlemek mümkün, ama her adımın neden yapıldığını da not etmek gerekiyor.