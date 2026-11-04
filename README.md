# 🚗 Traffic Vehicle Detection Dataset & Augmentation Pipeline
## Bitirme Projesi (Graduation Thesis Project)

Bu çalışma, Akıllı Ulaşım Sistemleri (ITS) ve Trafik İzleme senaryoları için optimize edilmiş **YOLO formatındaki araç tespit veri setini (`vehicle_dataset_v3`)** ve veri artırma (Data Augmentation) süreçlerini barındırmaktadır.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![YOLO Format](https://img.shields.io/badge/Format-YOLO-orange.svg)](https://docs.ultralytics.com/)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)

---

## 📌 Proje ve Veri Seti Odağı
Bu aşamadaki temel amaç; trafik kameralarından elde edilen kısıtlı araç verisini farklı hava koşulları, kamera açıları ve trafik yoğunluğu senaryolarına karşı **veri artırma (Data Augmentation)** teknikleriyle zenginleştirmek ve özellikle az temsil edilen araç sınıflarını (otobüs, kamyon, motosiklet) güçlendirmektir.

### 🏷️ Tespit Edilen Araç Sınıfları (Classes)

| Sınıf ID | Sınıf Adı | Açıklama |
|---|---|---|
| **0** | `car` | Otomobil / Binek Araç |
| **1** | `motorcycle` | Motosiklet / Scooter |
| **2** | `bus` | Toplu Taşıma / Otobüs |
| **3** | `truck` | Kamyon / Ağır Vasıta |

---

## 📊 Mevcut Veri Dağılımı (Mevcut Durum)

| Alt Küme | Görsel Sayısı | Oran | Açıklama |
|---|---|---|---|
| **Train** | 88 Görsel | ~%70 | Veri artırma (Augmentation) uygulanacak ana eğitim kümesi |
| **Validation** | 19 Görsel | ~%15 | Doğrulama kümesi (Orijinal hali korunur) |
| **Test** | 19 Görsel | ~%15 | Test ve nihai değerlendirme kümesi (Orijinal hali korunur) |
| **TOPLAM** | **126 Görsel** | **%100** | Temel Veri Seti |

---

## 📁 Dizin Yapısı (Directory Structure)

```text
federated-traffic-learning/
├── vehicle_dataset_v3/
│   ├── data.yaml              # Veri yolu ve sınıf tanımlamaları
│   ├── images/                # Görseller (.jpg / .png)
│   │   ├── train/             # Eğitim görselleri
│   │   ├── val/               # Doğrulama görselleri
│   │   └── test/              # Test görselleri
│   └── labels/                # Normalize edilmiş YOLO etiketleri (.txt)
│       ├── train/             # Eğitim etiketleri
│       ├── val/               # Doğrulama etiketleri
│       └── test/              # Test etiketleri
├── requirements.txt           # Gerekli kütüphaneler
├── .gitignore                 # Takip dışı dosya kuralları
└── README.md                  # Veri seti ve artırma dokümantasyonu
```

---

## 📝 Etiket Formatı (YOLO Bounding Box Format)
Görsellerin `.txt` etiket dosyalarında her satırda bir araç tespit kutusu yer alır:
```text
<class_id> <x_center> <y_center> <width> <height>
```
* Bounding Box koordinatları `[0.0, 1.0]` aralığına normalize edilmiştir.

---

## 🎯 Planlanan Veri Artırma (Data Augmentation) Stratejisi
1. **Çevresel / Hava Koşulları:** Parlaklık, kontrast değişimi, sis/bulanıklık ve gece-gündüz simülasyonu.
2. **Geometrik Dönüşümler:** Yatay çevirme (horizontal flip), hafif açı değişimleri ve ölçeklendirme (farklı kamera açıları ve mesafeleri).
3. **Kutu Koruma (Bounding Box Synchronization):** Tüm dönüşümlerde YOLO etiket koordinatlarının otomatik olarak yeniden hesaplanması.
4. **Sınıf Dengeleme:** Az sayıda bulunan otobüs, kamyon ve motosiklet sınıflarına öncelikli ve hedefli artırma yapılması.

---

## 👤 Geliştirici & Bilgiler
* **Öğrenci:** Aydın Gürler ([@aydin-eng](https://github.com/aydin-eng))
* **Bitirme Projesi Konusu:** Trafik İzleme ve Araç Tespiti için Veri Artırma ve Federe Öğrenme
