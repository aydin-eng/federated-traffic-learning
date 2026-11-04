# 🚗 Traffic Vehicle Detection & YOLO11 Training Pipeline
## Bitirme Projesi (Graduation Thesis Project)

Bu çalışma; Akıllı Ulaşım Sistemleri (ITS) ve Trafik İzleme senaryoları için optimize edilmiş **YOLO formatındaki araç tespit veri setini (`vehicle_dataset_v3_augmented`)**, **Google Colab üzerinde GPU ile eğitilmiş YOLO11 Nano modelini (`weights/best.pt`)** ve **eğitim öncesi vs. eğitim sonrası model karşılaştırma sonuçlarını** barındırmaktadır.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aydin-eng/federated-traffic-learning/blob/main/traffic_vehicle_detection_yolo11.ipynb)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![YOLO11](https://img.shields.io/badge/Model-YOLO11n-orange.svg)](https://docs.ultralytics.com/)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)

---

## 📸 Model Karşılaştırma Sonuçları (Before vs. After)

Aşağıdaki karşılaştırmalarda; sol tarafta genel nesnelerle önceden eğitilmiş standart **Base YOLO11n (Pretrained)** modeli, sağ tarafta ise bizim **`vehicle_dataset_v3_augmented`** veri kümemiz ile trafik tespiti için özel olarak eğittiğimiz **Fine-Tuned YOLO11n** modelinin test görüntüleri üzerindeki sonuçları yer almaktadır:

### 1. Motosiklet ve Trafik Tespiti (Motorcycle Detection)
> *Standart model yoldaki yayaları ve alakasız nesneleri de işlerken, fine-tuned modelimiz doğrudan trafik nesnesi olan motosiklete odaklanmaktadır.*

![Motorcycle Comparison](comparison_results/comparison_3_motorcycle.jpg)

### 2. Otobüs ve Otomobil Tespiti (Bus & Car Detection)
![Bus and Car Comparison](comparison_results/comparison_2_bus_car.jpg)

### 3. Otomobil ve Kamyon Tespiti (Car & Truck Detection)
![Car and Truck Comparison](comparison_results/comparison_1_car_truck.jpg)

---

## 📈 Model Eğitim Başarısı ve Metrikler (Training Results)

Model, Google Colab GPU ortamında 33 epoch boyunca `vehicle_dataset_v3_augmented` veri kümesiyle eğitilmiştir:

| Metrik | Değer | Açıklama |
|---|---|---|
| **Kullanılan Model** | `YOLO11n (Nano)` | 2.6M Parametre (~5.5 MB) |
| **Eğitim Süresi** | 33 Epoch | Early Stopping ile en iyi ağırlık seçildi |
| **Precision (B)** | **%71.98 (0.7198)** | Modelin tespit ettiği araçların doğruluk oranı |
| **mAP@50 (B)** | **%24.88 (0.2488)** | Ortalama Kesinlik (IoU 0.50) |
| **Görsel Boyutu** | 640x640 | Standart YOLO çözünürlüğü |

### 📊 Eğitim Kayıp Grafikleri (Loss & Metric Curves)
Eğitim süresince `box_loss`, `cls_loss`, `dfl_loss` kayıplarının düşüşü ve kesinlik eğrileri:

![Eğitim Kayıp Grafikleri](training_results/results.png)

### 🎯 Karışıklık Matrisi (Confusion Matrix)
Modelin araç sınıflarını (`car`, `motorcycle`, `bus`, `truck`) ayırt etme başarısı:

![Confusion Matrix](training_results/confusion_matrix_normalized.png)

---

## 🏷️ Tespit Edilen Araç Sınıfları (Classes)

| Sınıf ID | Sınıf Adı | Açıklama |
|---|---|---|
| **0** | `car` | Otomobil / Binek Araç |
| **1** | `motorcycle` | Motosiklet / Scooter |
| **2** | `bus` | Toplu Taşıma / Otobüs |
| **3** | `truck` | Kamyon / Ağır Vasıta |

---

## 📁 Dizin Yapısı (Directory Structure)

```text
federated-traffic-learning/
├── weights/
│   └── best.pt                             # Eğitilen en başarılı YOLO11 model ağırlığı (~5.5 MB)
├── comparison_results/                     # Eğitim öncesi vs sonrası karşılaştırma görselleri
│   ├── comparison_1_car_truck.jpg
│   ├── comparison_2_bus_car.jpg
│   └── comparison_3_motorcycle.jpg
├── training_results/                       # Eğitim çıktıları, grafikler ve metrikler
│   ├── results.png                         # Epoch kayıp ve mAP grafikleri
│   ├── confusion_matrix_normalized.png     # Normalize edilmiş karışıklık matrisi
│   ├── BoxPR_curve.png                     # Precision-Recall eğrisi
│   ├── BoxF1_curve.png                     # F1 Skoru eğrisi
│   └── results.csv                         # Sayısal metrik tablosu
├── traffic_vehicle_detection_yolo11.ipynb  # Google Colab GPU eğitim notebook'u
├── vehicle_dataset_v3_augmented/           # 298 görsellik artırılmış ana veri seti
├── vehicle_dataset_v3/                     # 126 görsellik orijinal veri seti
├── augment_dataset.py                      # Kutu korumalı veri artırma betiği
├── requirements.txt                        # Gerekli bağımlılıklar
├── .gitignore                              # Git takip kuralları
└── README.md                               # Kapsamlı proje dokümantasyonu
```

---

## 🚀 Eğitilen Modeli Kullanma (Inference)

Eğitilen `weights/best.pt` ağırlıklarını kendi Python ortamınızda test etmek için:

```python
from ultralytics import YOLO

# Eğitilmiş modeli yükle
model = YOLO('weights/best.pt')

# Herhangi bir trafik görselinde veya videoda tahmin yap
results = model.predict(source='ornek_trafik_goruntusu.jpg', conf=0.25, save=True)
```

---

## 👤 Geliştirici & Bilgiler
* **Öğrenci:** Aydın Gürler ([@aydin-eng](https://github.com/aydin-eng))
* **Bitirme Projesi Konusu:** Trafik İzleme ve Araç Tespiti için Veri Artırma ve Federe Öğrenme
