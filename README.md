# Meyve ve Sebze Tanıma Projesi - CMPE 462 Assignment 1

Bu proje, meyve ve sebze tanıma için çoklu modalite veri seti kullanarak lojistik regresyon sınıflandırıcı geliştirmeyi amaçlamaktadır.

## Proje Kategorileri

- Muz
- Domates
- Salatalık
- Mandalina
- Patates

## Kurulum

1. Gerekli paketleri yükleyin:
```bash
pip install -r requirements.txt
```

2. SpaCy Türkçe modelini yükleyin (metin özellik çıkarımı için):
```bash
python -m spacy download tr_core_news_sm
```

## Proje Yapısı

```
462/
├── data/                    # Veri setleri
│   ├── raw/                # Ham veriler
│   ├── processed/          # İşlenmiş veriler
│   ├── train/              # Eğitim seti
│   ├── test/               # Test seti
│   └── val/                # Validasyon seti
├── src/                    # Kaynak kod
│   ├── data_collection/    # Veri toplama modülleri
│   ├── feature_extraction/ # Özellik çıkarım modülleri
│   ├── models/             # Model implementasyonları
│   ├── evaluation/         # Değerlendirme modülleri
│   └── utils/              # Yardımcı fonksiyonlar
├── notebooks/              # Jupyter notebook'lar
└── reports/                # Rapor dosyaları
```

## Kullanım

### Veri Toplama

```python
from src.data_collection.image_collector import ImageCollector
from src.data_collection.metadata_collector import MetadataCollector
from src.data_collection.text_collector import TextCollector

# Görüntü toplama
image_collector = ImageCollector()
image_collector.collect_images(categories=['muz', 'domates', 'salatalik', 'mandalina', 'patates'])

# Metadata toplama
metadata_collector = MetadataCollector()
metadata_collector.collect_metadata()

# Metin verileri toplama
text_collector = TextCollector()
text_collector.collect_descriptions()
```

### Özellik Çıkarımı

```python
from src.feature_extraction.image_features import extract_image_features
from src.feature_extraction.text_features import extract_text_features
from src.feature_extraction.feature_fusion import fuse_features

# Görüntü özellikleri
image_features = extract_image_features(image_paths)

# Metin özellikleri
text_features = extract_text_features(text_descriptions)

# Özellik birleştirme
fused_features = fuse_features(image_features, text_features, metadata_features)
```

### Model Eğitimi

```python
from src.models.one_vs_all import OneVsAllClassifier
from src.models.logistic_regression import LogisticRegression

# Model oluşturma
model = OneVsAllClassifier(LogisticRegression, n_classes=5)
model.fit(X_train, y_train)

# Tahmin
predictions = model.predict(X_test)
```

## Sonuçları Yeniden Üretme

1. Veri setini hazırlayın (data/ klasörüne yerleştirin)
2. `src/main.py` scriptini çalıştırın:
```bash
python src/main.py
```

3. Veya notebook'ları sırayla çalıştırın:
   - `notebooks/01_data_exploration.ipynb`
   - `notebooks/02_feature_extraction.ipynb`
   - `notebooks/03_model_training.ipynb`

## Notlar

- Her kategoriden en az 50 örnek manuel olarak toplanmalıdır
- Toplam veri seti: 3000 örnek (kategori başına 600)
- Eğitim seti: 2500 örnek
- Test seti: 500 örnek
- Validasyon seti: Eğitim setinden 500 örnek

## Lisans

Bu proje CMPE 462 dersi kapsamında geliştirilmiştir.

