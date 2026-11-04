# 🚗 Federated Learning for Traffic & Vehicle Detection (Trafik ve Araç Tespiti için Federe Öğrenme)

Bitirme Projesi (Graduation Thesis Project) kapsamında geliştirilen; gizlilik korumalı (privacy-preserving), dağıtık **Federe Öğrenme (Federated Learning)** ve **YOLO Tabanlı Araç Tespiti (Vehicle Detection)** sistemi.

[![GitHub License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red.svg)](https://pytorch.org/)
[![YOLO11](https://img.shields.io/badge/YOLO-v11-orange.svg)](https://github.com/ultralytics/ultralytics)
[![Flower](https://img.shields.io/badge/Flower-FL-yellow.svg)](https://flower.ai/)

---

## 📌 Proje Özeti (Project Overview)
Geleneksel akıllı ulaşım sistemlerinde (ITS), şehir kameralarından ve sensörlerden toplanan yüksek çözünürlüklü video ve araç verileri merkezi sunuculara aktarılmaktadır. Ancak bu durum:
* **Gizlilik ve Güvenlik:** Araç plakalarının ve sürüş rotalarının tek bir merkezde toplanarak ifşa olma riski,
* **Ağ Trafiği ve Bant Genişliği:** Terabaytlarca görsel verinin ağ üzerinden sürekli taşınmasının yarattığı tıkanıklık,
* **Gerçek Zamanlı Gecikme:** Merkezi modellerin tek noktadan arızalanma (single point of failure) riski doğurur.

Bu bitirme projesinde, **YOLO (You Only Look Once)** mimarisi ve **Federe Öğrenme (Flower - flwr)** entegre edilerek:
1. Her kavşak veya trafik kamerası yerel bir **Federe İstemci (Edge Client)** olarak konumlandırılır.
2. Kamera görüntüleri ve araç verileri **asla cihaz dışına çıkmaz**.
3. Yalnızca model ağırlık gradyanları merkezi sunucuya gönderilir ve **FedAvg** algoritması ile global araç tespit modeli eğitilir.

---

## 📊 Veri Seti Ayrıntıları (Dataset Details)

Projede araç ve trafik tespitine yönelik özenle filtrelenmiş ve dengelenmiş çok sınıflı YOLO veri seti yer almaktadır:

### 🏷️ Tespit Edilen Sınıflar (Classes)
| ID | Sınıf Adı | Açıklama |
|---|---|---|
| **0** | car | Binek Otomobiller |
| **1** | motorcycle | Motosiklet ve Scooterlar |
| **2** | us | Toplu Taşıma / Otobüsler |
| **3** | 	ruck | Kamyon ve Ağır Vasıtalar |

### 📁 Veri Seti Dizin Mimarisi
* **ehicle_dataset_v3/**: En güncel, filtrelenmiş ve etiket kontrolleri tamamlanmış ana veri seti kümesi.
  * images/train, images/val, images/test: Görüntü dosyaları (.jpg / .png).
  * labels/train, labels/val, labels/test: Normalize edilmiş YOLO sınırlayıcı kutu (bounding box) etiketleri (<class_id> <x_center> <y_center> <width> <height>).
  * data.yaml: Veri yolu ve sınıf isimlerini tanımlayan konfigürasyon dosyası.
* **ehicle_dataset_clean/**: Temizlenmiş referans araç veri kümesi.
* **ehicle_dataset_balanced/**: Sınıf dengesizliği (class imbalance) giderilmiş eğitim veri kümesi.
* **coco128/**: Ön test ve transfer learning karşılaştırmaları için kullanılan alt küme.

---

## 🛠️ Proje Dosya Yapısı (File Structure)

`	ext
federated-traffic-learning/
├── vehicle_dataset_v3/             # Ana araç tespit veri seti (train/val/test)
├── vehicle_dataset_clean/          # Filtrelenmiş temiz veri seti
├── vehicle_dataset_balanced/       # Dengelenmiş veri kümesi
├── coco128/                        # Benchmark alt kümesi
├── runs/                           # Önceden yapılan eğitim sonuçları ve metrikler
├── yolo11n.pt                      # YOLOv11 Nano önceden eğitilmiş model ağırlığı
├── prepare_dataset_v3.py           # Veri seti hazırlama ve bölütleme betiği
├── check_distribution.py           # Sınıf dağılımını analiz eden script
├── check_label.py                  # Etiket doğrulama ve görselleştirme betiği
├── predict.py                      # Model çıkarım (inference) ve test betiği
├── requirements.txt                # Gerekli bağımlılıklar
├── .gitignore                      # Git takip dışı dosyalar
└── README.md                       # Detaylı proje kılavuzu
`

---

## 🚀 Kurulum ve Kullanım (Getting Started)

### 1. Bağımlılıkları Yükleyin
`ash
pip install -r requirements.txt
`

### 2. Veri Seti Dağılımını Kontrol Edin
`ash
python check_distribution.py
`

### 3. Modeli Test Edin (Inference)
`ash
python predict.py
`

---

## 👤 Geliştirici & Yazar
* **Öğrenci:** Aydın Gürler ([@aydin-eng](https://github.com/aydin-eng))
* **Bitirme Tezi Konusu:** Federe Öğrenme ile Trafik Analizi ve Araç Tespiti
