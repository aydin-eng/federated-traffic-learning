# 🚗 Federated Learning - Traffic Vehicle Detection Dataset
## Bitirme Projesi (Graduation Thesis Project)

Bu depo, Akıllı Ulaşım Sistemleri (ITS) ve Trafik Yönetimi kapsamında **Federe Öğrenme (Federated Learning)** tabanlı araç tespiti modellerinin eğitimi ve değerlendirilmesi için hazırlanmış **YOLO formatındaki araç veri setini (`vehicle_dataset_v3`)** barındırmaktadır.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![YOLO](https://img.shields.io/badge/Format-YOLO-orange.svg)](https://docs.ultralytics.com/)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)

---

## 📌 Veri Seti Hakkında (Dataset Overview)
Federe öğrenme mimarisinde her bir trafik kamerası, radar veya kavşak sensörü yerel bir istemci (client) olarak kabul edilir. Bu veri seti, şehir trafiğindeki ana araç kategorilerini tespit etmek, kavşak yoğunluğunu analiz etmek ve gizlilik odaklı dağıtık yapay zeka modelleri eğitmek için optimize edilmiştir.

### 🏷️ Sınıf Bilgileri (Classes & Labels)
Veri seti 4 temel araç sınıfını içermektedir:

| Sınıf ID | Sınıf Adı | Açıklama |
|---|---|---|
| **0** | `car` | Otomobil / Binek Araç |
| **1** | `motorcycle` | Motosiklet / Scooter |
| **2** | `bus` | Otobüs / Toplu Taşıma Araçları |
| **3** | `truck` | Kamyon / Ağır Vasıta / Tır |

---

## 📊 Veri Dağılımı ve Bölütleme (Dataset Split)

Veri seti, nadir sınıfların (özellikle otobüs, kamyon ve motosiklet) homojen dağılması ve modelin genelleme başarısını artırmak amacıyla dengelenerek 3 alt kümeye ayrılmıştır:

| Alt Küme | Görsel Sayısı | Oran | Amaç |
|---|---|---|---|
| **Train** | 88 Görsel | ~%70 | Yerel modellerin eğitimi (Local Client Training) |
| **Validation** | 19 Görsel | ~%15 | Hiperparametre optimizasyonu ve doğrulama |
| **Test** | 19 Görsel | ~%15 | Global modelin nihai performans ölçümü (Test Evaluation) |
| **TOPLAM** | **126 Görsel** | **%100** | Tam Veri Seti |

> **Not:** Veri seti içerisinde YOLO mimarisinin yanlış pozitif (false positive) tespitlerini önlemek amacıyla standart arka plan (background) görselleri de dengeli şekilde yer almaktadır.

---

## 📁 Dizin Yapısı (Directory Structure)

```text
federated-traffic-learning/
├── vehicle_dataset_v3/
│   ├── data.yaml              # YOLO veri yolu ve sınıf tanımlamaları
│   ├── images/                # Görseller (.jpg / .png)
│   │   ├── train/             # 88 eğitim görseli
│   │   ├── val/               # 19 doğrulama görseli
│   │   └── test/              # 19 test görseli
│   └── labels/                # YOLO etiket dosyaları (.txt)
│       ├── train/             # 88 etiket dosyası
│       ├── val/               # 19 etiket dosyası
│       └── test/              # 19 etiket dosyası
├── requirements.txt           # Gerekli temel Python kütüphaneleri
├── .gitignore                 # Takip dışı dosya kuralları
└── README.md                  # Veri seti detay ve dokümantasyon dosyası
```

---

## 📝 Etiket Formatı (YOLO Annotation Format)
Her görüntüye karşılık gelen `.txt` etiket dosyasında her bir araç nesnesi için normalize edilmiş format kullanılır:
```text
<class_id> <x_center> <y_center> <width> <height>
```
* Değerler `0.0` ile `1.0` aralığında normalize edilmiştir.

---

## 🚀 Hızlı Başlangıç (Quick Start)

### 1. Bağımlılıkları Yükleme
```bash
pip install -r requirements.txt
```

### 2. YOLO ile Eğitimi Başlatma Örneği
```python
from ultralytics import YOLO

# Modeli yükle (Örn: YOLOv11)
model = YOLO('yolo11n.pt')

# vehicle_dataset_v3 ile eğitimi başlat
results = model.train(
    data='vehicle_dataset_v3/data.yaml',
    epochs=50,
    imgsz=640,
    batch=8
)
```

---

## 👤 Geliştirici & Bilgiler
* **Öğrenci:** Aydın Gürler ([@aydin-eng](https://github.com/aydin-eng))
* **Bitirme Projesi Konusu:** Trafik Yönetimi ve Araç Tespiti için Federe Öğrenme (Federated Learning for Traffic & Vehicle Detection)
