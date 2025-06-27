# Bitirme Projesi

## Proje Hakkında

**Sentetik Veri Üretim Sistemi**, gerçek veri setlerinden model tabanlı veya rastgele yöntemlerle sentetik veri üretimi sağlayan, Flask tabanlı bir web uygulamasıdır. Uygulama, SDV (Synthetic Data Vault) ve SDMetrics kütüphanelerini kullanarak yüksek kaliteli sentetik veriler üretir ve bu verilerin kalitesini çeşitli metriklerle değerlendirir. Kullanıcı dostu arayüzü sayesinde, veri bilimi projelerinde veri gizliliğini koruyarak analiz ve modelleme süreçlerini kolaylaştırır.

## Özellikler

- **Model Tabanlı Sentetik Veri Üretimi:** Gerçek verilerden öğrenen modellerle sentetik veri oluşturma.
- **Rastgele Sentetik Veri Üretimi:** Kolon tiplerine göre rastgele veri üretimi.
- **Veri Kalite Analizi:** ML Efficacy, İstatistiksel Benzerlik, Korelasyon Benzerliği, Kategori Kapsamı gibi metriklerle sentetik verinin kalitesini ölçme.
- **Kullanıcı Yönetimi:** Kayıt, giriş, profil ve hesap ayarları.
- **Dosya Yükleme ve İndirme:** Gerçek ve sentetik veri dosyalarını yükleme ve indirme.
- **Modern ve Kullanıcı Dostu Arayüz:** Responsive ve sezgisel web arayüzü.

## Kurulum

### Gereksinimler

- Python 3.12+
- pip

### Bağımlılıkların Kurulumu

```bash
pip install -r requirements.txt
```

### Veritabanı Kurulumu

```bash
flask db upgrade
```

### Uygulamayı Başlatma

```bash
python app.py
```

Uygulama varsayılan olarak `http://127.0.0.1:5000` adresinde çalışacaktır.

## Kullanım

1. **Kayıt Olun / Giriş Yapın:** Hesabınızı oluşturun veya mevcut hesabınızla giriş yapın.
2. **Veri Yükleyin:** Gerçek veri setinizi yükleyin.
3. **Sentetik Veri Üretin:** Model tabanlı veya rastgele sentetik veri üretimini başlatın.
4. **Kalite Analizini Görüntüleyin:** Üretilen sentetik verinin kalitesini çeşitli metriklerle analiz edin.
5. **Verileri İndirin:** Gerçek veya sentetik veri setlerini indirin.

## Örnek Çıktılar

- **Sentetik Veri Dosyası:**  
  `uploads/synthetic_data_20250613_100714.csv`  
- **Kalite Metrikleri:**  
  - ML Efficacy: 0.87
  - İstatistiksel Benzerlik: 0.92
  - Korelasyon Benzerliği: 0.89
  - Kategori Kapsamı: 1.00

## Klasör Yapısı

```
SynthNewPro/
│
├── app.py
├── requirements.txt
├── data/
├── generated_files/
├── instance/
├── migrations/
├── static/
├── templates/
├── uploads/
└── readme.md
```

## Kullanılan Teknolojiler

- **Backend:** Flask, SDV, SDMetrics, pandas, scikit-learn
- **Frontend:** HTML, CSS, JavaScript (Jinja2 templating)
- **Veritabanı:** SQLite

## Katkı Sağlama

Katkıda bulunmak isterseniz lütfen bir fork oluşturun, değişikliklerinizi yapın ve bir pull request gönderin. Her türlü öneri ve hata bildirimi için issue açabilirsiniz.

## İletişim

Her türlü soru ve öneriniz için [ederojind@gmail.com] adresinden veya GitHub Issues üzerinden iletişime geçebilirsiniz.
