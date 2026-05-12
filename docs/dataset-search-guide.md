# Dataset Arama Rehberi

Hocanın belgede önerdiği 5 kaynaktan **sırasıyla** dataseti nasıl arayacağını adım adım açıklama.

## Hedef: Fake News Dataset Bulma

**Ne arıyoruz:**
- Minimum 1000 haber örneği
- İkili etiket: "fake" ve "real" (veya 0 ve 1)
- Tam metin: başlık değil, haber metni gerekli
- Public access: ücretsiz indirilebilir

**Tercihen:** Türkçe dataset. Yoksa İngilizce WELFake backup.

---

## KAYNAK 1: HUGGING FACE DATASETS

**URL:** https://huggingface.co/datasets

### Adım 1: Siteye Git
- Browser'de linki aç
- Arama kutusuna tıkla (sağ üstteki arama simgesi)

### Adım 2: Arama Kelimeleri

Sırasıyla bu arama terimlerini dene:

**1. Türkçe fake news:**
```
Turkish fake news
```
- Sonuç: Turkish Fake News Detection (TASSA vb)
- İndirme: Dataset sayfasında "Download" düğmesi var
- Kontrol et: readme'de kolon adları yazılı mı?

**2. Multilingual fake news:**
```
multilingual fake news
```
- Sonuç: Multilingual Fake News Detection
- Kontrol et: Turkish kolonu var mı?

**3. Binary news classification:**
```
binary news classification
```
- Sonuç: Çeşitli haber sınıflandırma datasetleri
- Kontrol et: True/False veya Real/Fake etiketi var mı?

**4. Misinformation detection Turkish:**
```
misinformation Turkish
```
- Sonuç: Varsa Türkçe yanlış bilgi datasetleri

### Adım 3: Uygun Dataseti Seç

Her dataseti açıp kontrol et:
- [ ] Kaç satır var? (minimum 1000)
- [ ] Hangi kolonlar var? (text ve label lazım)
- [ ] Label değerleri neler? (fake/real mi, 0/1 mi?)
- [ ] License nedir? (açık mı, kapalı mı?)
- [ ] Nasıl indirilir? (Python kodu mı, CSV mi?)

---

## KAYNAK 2: KAGGLE DATASETS

**URL:** https://www.kaggle.com/datasets

### Adım 1: Siteye Git
- Kaggle hesabı gerekli (Gmail ile giriş yapabilir)
- Datasets linkine tıkla

### Adım 2: Arama Kelimeleri

**1. Fake News:**
```
fake news
```
- En ünlüsü: "WELFake: Dataset of News Articles for Fake News Detection"
- Kontrol et:
  - İngilizce ama çok popüler (100k+ views)
  - 72,650 haber
  - label: "0 (fake)" ve "1 (real)"
  - Kolonlar: title, text, label, date vb

**2. Turkish News:**
```
Turkish fake news
```
- Sonuç: Varsa Türkçe fake news datasetleri

**3. News Classification:**
```
news classification dataset
```
- Sonuç: Çeşitli haber sınıflandırma datasetleri

### Adım 3: İndirme

İyi görünen dataseti açınca:
- [ ] "Download" düğmesine tıkla
- [ ] CSV veya ZIP indir
- [ ] Bilgisayarına kaydet (`data/raw/` klasörüne)

**WELFake örneği:** Kaggle'da çok kolay bulunur, her zaman indirilebilir, kolayca açılır.

---

## KAYNAK 3: GOOGLE DATASET SEARCH

**URL:** https://datasetsearch.research.google.com

### Adım 1: Siteye Git
- Direkt linke tıkla
- Arama kutusu açılacak

### Adım 2: Arama

**1. Fake News:**
```
fake news dataset
```
- Sonuç: Google tüm internette fake news datasetini ara
- Akademik siteler, GitHub, vb ortaya çıkacak

**2. Turkish Fake News:**
```
Turkish fake news dataset
```
- Sonuç: Türkiye'den yayınlanmış datasetler bulunabilir

**3. Misinformation:**
```
misinformation dataset
```
- Sonuç: Çeşitli kaynaklardan veri setleri

### Adım 3: Sonuçlar Nasıl Değerlendirilir?

Google Dataset Search'de sonuç açınca:
- Üstte: "Available at" bölümü var
- Oradan dataset'e giden link tıkla
- Dataset kimin tarafından paylaşıldığını kontrol et (üniversite, araştırma vb)
- Indirme talimatları oku

**Avantajı:** Çok sayıda alternatif bulursun.

---

## KAYNAK 4: PAPERS WITH CODE

**URL:** https://paperswithcode.com/datasets

### Adım 1: Siteye Git
- Datasets sekmesine tıkla

### Adım 2: Arama

**1. Fake News:**
```
fake news
```
- Sonuç: Akademik makalelerle bağlantılı datasetler
- Her dataseti tıkla, "Paper" linkini oku (metodoloji)
- "Links" kısmından dataseti indir

**2. News Classification:**
```
news classification
```
- Sonuç: Haber sınıflandırma datasetleri

### Adım 3: Bilimsel Seçim

Papers With Code'un avantajı:
- [ ] Hangi makalede kullanıldığını bilirsin (referans için)
- [ ] Veri setiyle ilgili bilimsel detaylı sunum
- [ ] Genelde kaliteli, kontrol edilmiş veri

---

## KAYNAK 5: TÜİK (Türk İstatistik Kurumu)

**URL:** https://data.tuik.gov.tr

### Adım 1: Siteye Git
- Direkt linke tıkla
- Türkçe interface açılacak

### Adım 2: Arama

Burası ekonomik/demografik veri için, fake news için **pek yaramaz ama:**
- Haber yazarken Türkiye'ye dair gerçek istatistiklere ihtiyacın olabilir
- Doğruluk kontrolü için kullanabilir

**Fake news için TÜİK'ten bekleme, ama Twitter/bilimsel dergileri kontrol et.**

---

## ARAMA DOSYASI: BÜTÜN ANAHTAR KELİMELER

Tüm kaynaklarda dene, sırasıyla:

### 1. Türkçe Veri (Tercih Edilen)
```
Turkish fake news
Turkish misinformation
Turkish news classification
Türkçe sahte haber
```

### 2. Multilingual Veri (Plan B)
```
multilingual fake news
multilingual news
```

### 3. İngilizce Veri (Plan C - WELFake)
```
WELFake
fake news classification
binary news classification
news veracity detection
```

### 4. Akademik Arama
```
fake news detection dataset paper
misinformation dataset benchmark
```

---

## DATASET KARŞILAŞTIRMA ÖRNEĞİ

Birkaç dataseti bulunca, bu tabloyu doldur:

| Dataset Adı | Kaynak | Satır Sayısı | Dil | Label Tipi | İndirme Kolayı | Tercih |
|-------------|--------|--------------|-----|-----------|----------------|--------|
| Turkish Fake News TASSA | HF | 5000+ | Türkçe | fake/real | Evet | ⭐⭐⭐ |
| WELFake | Kaggle | 72,650 | İngilizce | 0/1 | Evet | ⭐⭐ |
| [Diğer] | [Kaynak] | [?] | [?] | [?] | [?] | [?] |

---

## SEÇIM KRİTERLERİ (Özet)

**YAŞAR mı? (GO/NO-GO)**

- [ ] En az 1000 satır var mı? → Hayırsa: Geç
- [ ] Text kolonu tam haber metni mi? → Hayırsa: Geç
- [ ] Label ikili mi (fake/real veya 0/1)? → Hayırsa: Geç
- [ ] Ücretsiz indirilebilir mi? → Hayırsa: Geç
- [ ] Belge/readme var mı? → Varsa: Git

**EĞER TÜRKÇESİ YOKSA:** WELFake'i seç. Çok bilinir, çok insan kullanır, indirmesi kolay.

---

## GÜN 1'DE YAPILACAK SUNUT

**Kişi 1 akşama kadar raporla:**
1. Hangi dataseti seçtin?
2. Hangi kaynaktan aldın?
3. URL ve indirme linki nedir?
4. İstatistikler (satır sayısı, label dağılımı, kolon adları)
5. Neden bu dataseti seçtin?

**Örnek:**
```
Dataset: WELFake
Kaynak: Kaggle
Satır: 72,650
Label: 0=fake, 1=real
Kolonlar: title, text, label, date
Tercih Nedeni: Türkçe yok, ama en popüler ve en güvenilir
```

---

## ACIL DURUM

Eğer 4 saat araştırma sonrası hiç iyi dataset bulamazsan:
- **KARAR:** WELFake'i indir ve kullan
- Fakat mutlaka `docs/decisions.md` içine "Neden WELFake seçtim?" yaz
- Demo sunuşunda sor sorulacak bu husus
