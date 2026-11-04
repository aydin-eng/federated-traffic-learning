# 🚗 Traffic Vehicle Detection & YOLO11 Training Pipeline
## Bitirme Projesi (Graduation Thesis Project)

Bu depo, Akıllı Ulaşım Sistemleri (ITS) ve Trafik İzleme senaryoları için optimize edilmiş **YOLO formatındaki araç tespit veri setini (`vehicle_dataset_v3`)**, sınıf dengesizliğini gideren **veri artırma (Data Augmentation) pipeline'ını (`vehicle_dataset_v3_augmented`)** ve **Google Colab üzerinde GPU ile YOLO11 Nano (`yolo11n`) model eğitimi notebook'unu** barındırmaktadır.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aydin-eng/federated-traffic-learning/blob/main/traffic_vehicle_detection_yolo11.ipynb)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![YOLO11](https://img.shields.io/badge/Model-YOLO11n-orange.svg)](https://docs.ultralytics.com/)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)

---

## ⚡ Google Colab ile 1 Tıkla Eğitim (Quick Start on Colab)

Model eğitimini doğrudan Google Colab GPU'su (T4/V100) üzerinde çalıştırmak için aşağıdaki butona tıklayabilirsiniz:

👉 [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aydin-eng/federated-traffic-learning/blob/main/traffic_vehicle_detection_yolo11.ipynb)

Notebook dosyası (`traffic_vehicle_detection_yolo11.ipynb`):
1. Colab GPU donanımını otomatik kontrol eder.
2. Bu depoyu klonlayıp `vehicle_dataset_v3_augmented` veri setini hazırlar.
3. `yolo11n.pt` modelini yükler.
4. Eğitimi başlatır (`epochs=50`, `imgsz=640`, `batch=16`).
5. Test kümesi üzerinde değerlendirme yaparak mAP50, Precision, Recall metriklerini ve Confusion Matrix grafiklerini çizer.
6. Test görselleri üzerinde tahmin (inference) yaparak görselleştirir.

---

## 🏷️ Tespit Edilen Araç Sınıfları (Classes)

| Sınıf ID | Sınıf Adı | Açıklama |
|---|---|---|
| **0** | `car` | Otomobil / Binek Araç |
| **1** | `motorcycle` | Motosiklet / Scooter |
| **2** | `bus` | Toplu Taşıma / Otobüs |
| **3** | `truck` | Kamyon / Ağır Vasıta |

---

## 🔬 Uygulanan Veri Artırma Yöntemleri (Augmentation Techniques)

Trafik senaryoları ve YOLO nesne tespiti formatı dikkate alınarak geliştirilen `augment_dataset.py` betiğinde aşağıdaki dönüşümler kombine edilmiştir:

1. **Sınıf Odaklı Akıllı Artırma (Class-Aware Augmentation):**
   * Nadir sınıfları (`bus`, `truck`, `motorcycle`) içeren görsellere **10 farklı varyasyon** uygulanarak sınıf dengesizliği giderilmiştir.
   * `car` içeren görsellere **3 varyasyon** uygulanmıştır.
   * Arka plan negatif örneklerine **1 çevresel varyasyon** eklenmiştir.

2. **Geometrik Dönüşümler (Kutu Koordinatı Korumalı):**
   * **Yatay Çevirme (Horizontal Flip):** İki yönlü trafik akışı simülasyonu ($x_{center} = 1.0 - x_{center}$).
   * **Hafif Yakınlaştırma (Center Zoom - 1.1x):** Araçların kameraya olan mesafesi ve ölçek değişimi.

3. **Çevresel ve Işık Şartları (Trafik Kameraları Simülasyonu):**
   * **Gece / Tünel / Alacakaranlık:** Düşük ışık ve kontrast simülasyonu.
   * **Güneş Yansıması (Sunlight Glare):** Yüksek parlaklık ve güneş ışığı etkisi.
   * **Sis ve Yağmurlu Hava:** Gauss bulanıklığı ve kontrast yumuşatma (ıslak/sisli lens etkisi).
   * **Hareket Bulanıklığı (Motion Blur):** Yüksek hızda seyreden araçların hareket bulanıklığı.
   * **Renk / HSV Değişimi:** Farklı kamera sensörlerinin beyaz dengesi (white balance) profilleri.

4. **Kısmi Kapanma (Occlusion Simulation):**
   * Trafikte araçların levha, direk veya başka bir araç tarafından kısmen kapanması simülasyonu (Cutout).

> ⚠️ **Bilimsel Standart:** Veri artırma işlemleri **yalnızca `train` kümesine** uygulanmıştır. `val` ve `test` kümeleri orijinal haliyle korunmuştur.

---

## 📊 Veri Artırma Sonuçları ve Dağılım Tablosu

| Metrik | Orijinal (`vehicle_dataset_v3`) | Artırılmış (`vehicle_dataset_v3_augmented`) | Değişim |
|---|---|---|---|
| **Train Görsel Sayısı** | 88 Görsel | **260 Görsel** | **~%295 Artış (+172 Görsel)** |
| **Validation Görsel Sayısı** | 19 Görsel | 19 Görsel | Orijinal Korundu |
| **Test Görsel Sayısı** | 19 Görsel | 19 Görsel | Orijinal Korundu |
| **Toplam Görsel Sayısı** | 126 Görsel | **298 Görsel** | **2.36x Kat Artış** |

### 🎯 Eğitim Kümesindeki Sınıf Dağılımı (Nesne / Kutu Sayıları)

| Sınıf Adı | Orijinal Train Sayısı | Artırılmış Train Sayısı | Çarpan (Artış) |
|---|---|---|---|
| **`car`** | 30 nesne | **216 nesne** | **7.2x** |
| **`motorcycle`** | 3 nesne | **33 nesne** | **11.0x** |
| **`bus`** | 4 nesne | **44 nesne** | **11.0x** |
| **`truck`** | 8 nesne | **88 nesne** | **11.0x** |

---

## 📁 Dizin Yapısı (Directory Structure)

```text
federated-traffic-learning/
├── traffic_vehicle_detection_yolo11.ipynb  # Google Colab YOLO11 eğitim notebook'u
├── vehicle_dataset_v3/                     # Orijinal veri seti (126 görsel)
├── vehicle_dataset_v3_augmented/           # Artırılmış ve dengelenmiş veri seti (298 görsel)
│   ├── data.yaml                           # YOLO veri yolu ve sınıf tanımlamaları
│   ├── images/
│   │   ├── train/                          # 260 eğitim görseli
│   │   ├── val/                            # 19 doğrulama görseli
│   │   └── test/                           # 19 test görseli
│   └── labels/
│       ├── train/                          # 260 YOLO etiket dosyası
│       ├── val/                            # 19 YOLO etiket dosyası
│       └── test/                           # 19 YOLO etiket dosyası
├── augment_dataset.py                      # Kutu korumalı veri artırma betiği
├── requirements.txt                        # Gerekli bağımlılıklar
├── .gitignore                              # Git takip kuralları
└── README.md                               # Detaylı dokümantasyon
```

---

## 👤 Geliştirici & Bilgiler
* **Öğrenci:** Aydın Gürler ([@aydin-eng](https://github.com/aydin-eng))
* **Bitirme Projesi Konusu:** Trafik İzleme ve Araç Tespiti için Veri Artırma ve Federe Öğrenme
