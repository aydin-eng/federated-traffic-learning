# 🚗 Federated Traffic Learning: UA-DETRAC ile YOLO11 Araç Tespiti
## 🎓 Lisans Bitirme Projesi (Graduation Thesis Project)

Bu proje; Akıllı Ulaşım Sistemleri (ITS) ve Trafik İzleme senaryoları için **gerçek trafik gözetleme kameralarından elde edilen UA-DETRAC Benchmark Veri Seti** üzerinde optimize edilmiş **YOLO11 Nano** modelini ve ilerleyen aşamalarda **Federal Öğrenme (Federated Learning)** mimarisinde kullanılacak temel yerel model altyapısını içermektedir.

[![Open In Colab (Baseline YOLO11n)](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aydin-eng/federated-traffic-learning/blob/main/traffic_vehicle_detection_yolo11.ipynb)
[![Open In Colab (Balanced v2 YOLO11s)](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aydin-eng/federated-traffic-learning/blob/main/traffic_vehicle_detection_balanced_yolo11.ipynb)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![YOLO11](https://img.shields.io/badge/Model-YOLO11n%20%7C%20YOLO11s-orange.svg)](https://docs.ultralytics.com/)
[![Dataset: UA--DETRAC](https://img.shields.io/badge/Dataset-UA--DETRAC-success.svg)](https://sites.google.com/view/daweidu/projects/ua-detrac)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)

---

## 📸 Model Karşılaştırma ve Gerçek Saha Testi (Before vs. After)

Standart COCO ağırlıklı temel model (**Base YOLO11n**) ile gerçek trafik gözetleme kamerası açılarından oluşan **UA-DETRAC veri seti** üzerinde eğittiğimiz modelimizin bağımsız gerçek trafik sahnesindeki karşılaştırması:

![UA-DETRAC Model Karşılaştırması](comparison_results/detrac_vs_base_comparison.jpg)

### 🔍 Karşılaştırma Analizi:
* **Base YOLO11n (Pretrained COCO):** 29 araç tespit etti. Uzaktaki araçları ve yoğun şeritleri kaçırdı; bazı minibüs ve ticari araçları hatalı otobüs sınıflandırması yaptı. En yüksek güven skoru 0.82 oldu.
* **UA-DETRAC Fine-Tuned Model:** **36 araç tespit etti!** (+7 araç farkı). Kamera açısından uzaklaşan ve kısmen örtüşen (occluded) araçları başarıyla yakaladı. Güven skoru **0.90** seviyesine ulaştı ve kamyon/ağır vasıta sınıflandırması çok daha kararlı hale geldi.

---

## 📈 Model Başarım Metrikleri ve Karşılaştırmalı Tablo

Modelimiz, Google Colab Tesla T4 GPU ortamında 21 epoch boyunca eğitilmiş ve en yüksek mAP skoru 11. epoch'ta elde edilerek `best.pt` olarak kaydedilmiştir:

| Metrik | İlk Aşama (COCO Alt Kümesi) | **Yeni UA-DETRAC Modeli (GÜNCEL)** | Gelişim / İlerleme |
|---|---|---|---|
| **Eğitim Verisi** | 12 Görsel (Veri artırımlı 298) | **~4.000+ Gerçek Trafik Görseli** | **~14x Veri Hacmi** |
| **Toplam Etiket Sayısı**| ~1.200 Araç Kutusu | **~40.000+ Gerçek Araç Kutusu** | **~33x Daha Fazla Örnek** |
| **Precision (Kesinlik)** | %99.1 (Aşırı ezber/Overfit) | **%81.5 (0.8149)** | Dengeli ve gerçekçi genelleme |
| **Recall (Duyarlılık)** | %25.0 (Çoğu aracı kaçırıyordu) | **%58.3 (0.5828)** | **+33.3 Puan Artış!** |
| **mAP@50 (Ortalama Başarı)**| %27.6 (0.2764) | **%68.4 (0.6836)** | **+%40.8 Puan Artış (2.5 Kat!)** |
| **mAP@50-95 (Kapsamlı Başarı)**| %13.5 (0.1349) | **%49.1 (0.4914)** | **+%35.6 Puan Artış (3.6 Kat!)** |
| **Model Boyutu** | 5.5 MB (YOLO11n) | **5.5 MB (YOLO11n)** | Edge/IoT cihazlara tam uyumlu |

---

## 📊 Eğitim Grafikleri ve Metrik Eğrileri

### 1. Kayıp ve Metrik Gelişimi (Training & Validation Curves)
Eğitim süresince `box_loss`, `cls_loss`, `dfl_loss` kayıplarının düzenli düşüşü ve validation mAP değerlerinin yükselişi:

![Eğitim Kayıp Grafikleri](training_results/results.png)

### 2. Karışıklık Matrisi (Confusion Matrix)
Modelin araç sınıflarını (`car`, `bus`, `truck`) ayrıştırma ve arka plan (background) ayrımı:

| Normalize Karışıklık Matrisi | Ham Karışıklık Matrisi |
|---|---|
| ![Normalized Confusion Matrix](training_results/confusion_matrix_normalized.png) | ![Confusion Matrix](training_results/confusion_matrix.png) |

### 3. Precision-Recall (PR) & F1 Skoru Eğrileri
Tüm sınıflar genelinde kesinlik-duyarlılık dengesi:

| Precision-Recall Eğrisi (mAP@50 = 0.684) | F1-Score Eğrisi |
|---|---|
| ![PR Curve](training_results/BoxPR_curve.png) | ![F1 Curve](training_results/BoxF1_curve.png) |

---

## 🏷️ Tespit Edilen Araç Sınıfları (Classes)

| Sınıf ID | Sınıf Adı | Açıklama |
|---|---|---|
| **0** | `car` | Otomobil, Sedan, Hatchback, SUV, Minibüs |
| **1** | `motorcycle` | Motosiklet, Scooter |
| **2** | `bus` | Toplu Taşıma, Belediye Otobüsü, Servis |
| **3** | `truck` | Kamyon, TIR, Ağır Vasıta, Kamyonet |

---

## ⚖️ Sınıf Dengesizliği Çözümü ve Çoklu-Kaynak Pipeline (v2 — YOLO11s)

İlk UA-DETRAC eğitiminde `car` sınıfında %86.3 mAP@50 elde edilmesine rağmen nadir sınıflarda veri dağılımı kaynaklı darboğazlar tespit edilmiştir (`car:truck` oranı **135:1**, `truck` mAP@50 **0.374**, UA-DETRAC içinde `motorcycle` etiketinin bulunmaması ve `<ignored_region>` alanlarının maskelenmemesi kaynaklı sahte arka plan cezaları).

Bu sorunları gidermek için modüler çoklu-kaynak eğitim pipeline'ı (`v2`) eklenmiştir:

1. **Akıllı Sınıf-Duyarlı Örnekleme ve `<ignored_region>` Maskeleme (`src/data/prepare_balanced_dataset.py`):**
   * Uniform `step=20` yerine sınıf yoğunluğuna duyarlı örnekleme (`train_truck_step=2`, `train_bus_step=15`, `train_car_step=120`, `max_car_only=3`) ile UA-DETRAC `car:truck` oranı **135:1 → 12.1:1** seviyesine indirilmiş, doğrulama kümesindeki `truck` kutu sayısı **17 → 337** ve `bus` kutu sayısı **66 → 484** düzeyine çıkarılmıştır.
   * Video sekansı bazlı katmanlı bölme (*Stratified Video Split*) ile `train` (49 sekans) ve `val` (10 sekans) arasında **0 sekans örtüşmesi** sağlanmıştır.
   * XML `<ignored_region>` alanları `(114, 114, 114)` nötr gri ile maskelenirken geçerli etiketli araç pikselleri korunur; sonraki Federated Learning aşaması için sekans ve hava durumu (`sunny`, `cloudy`, `night`, `rainy`) bilgileri `dataset_metadata.csv` olarak dışa aktarılır.
2. **VisDrone2019-DET Çoklu-Kaynak Füzyonu (`src/data/fuse_external_vehicles.py`):**
   * Yukarıdan trafik kamerası açısındaki **VisDrone2019-DET** veri setinden yalnızca azınlık sınıflar (`motorcycle`, `truck`, `bus`) içeren görseller (`max_train=1800`, `max_val=350`) 4-sınıf şemasına birleştirilir.
3. **Nadir Sınıf Odaklı Augmentation (`src/data/augment_minority.py`):**
   * Yalnızca `train` kümesindeki `motorcycle` ve `truck` karelerine trafik kamerasına özel dönüşümler (`hflip`, `night`, `rainfog`, `sunlight`, `hsv`) uygulanır; `car`-only karelere kopya üretilmez ve `val`/`test` kümelerine dokunulmaz.
4. **Optimize YOLO11s Eğitimi (`traffic_vehicle_detection_balanced_yolo11.ipynb`):**
   * Küçük motosiklet ve uzak kamyonları yakalamak için `yolo11s.pt`, `imgsz=800`, `epochs=50`, `cos_lr=True`, `lr0=0.005`, `cls=0.8`, `mixup=0.15`, `copy_paste=0.15` ile eğitilir ve sınıf bazlı metrikler `per_class_metrics.json` olarak raporlanır.

---

## 🌐 Federal Öğrenme (Federated Learning) Entegrasyonuna Doğru

Bu model, bitirme projesinin bir sonraki aşaması olan **Federal Öğrenme** mimarisinde:
1. **Yerel Model (Local Model):** Farklı kavşak ve trafik kameralarında (Edge node / İstemci) yerel verilerle eğitilecek olan omurga modelini temsil eder.
2. **Gizlilik Dostu Eğitim:** Kamera ham görüntüleri sunucuya gönderilmeksizin sadece ağırlık gradyanları paylaşılacaktır.
3. **Küresel Birleştirme (FedAvg / Aggregation):** Eğitilen yerel modeller merkezi sunucuda birleştirilerek her türlü hava koşuluna ve trafik yoğunluğuna dayanıklı genel bir trafik zekası oluşturulacaktır.

---

## 📁 Proje Dizin Yapısı (Directory Structure)

```text
federated-traffic-learning/
├── src/
│   └── data/
│       ├── prepare_balanced_dataset.py           # Adaptive UA-DETRAC çıkarımı, ignored_region maskeleme ve FL metadata
│       ├── fuse_external_vehicles.py             # VisDrone2019-DET motorcycle/truck/bus füzyonu
│       └── augment_minority.py                   # Yalnızca nadir sınıflar (motorcycle, truck) için train augmentation
├── weights/
│   ├── best.pt                                   # UA-DETRAC baseline modeli (~5.5 MB)
│   └── last.pt                                   # Son epoch ağırlığı
├── comparison_results/                           # Çıkarım karşılaştırma görselleri
│   ├── detrac_vs_base_comparison.jpg             # Yeni model vs Pretrained Base kıyaslaması
│   ├── pred_detrac_yolo11n.jpg                   # Yeni model tahmin çıktısı
│   └── pred_base_yolo11n.jpg                     # Base model tahmin çıktısı
├── test_samples/                                 # Bağımsız test fotoğrafları
│   └── user_test_photo.jpg                       # Gerçek trafik test fotoğrafı
├── training_results/                             # 21 Epochluk baseline eğitim çıktıları ve analiz grafikleri
│   ├── results.png                               # Epoch metrik ve kayıp grafikleri
│   ├── results.csv                               # Sayısal metrik kayıtları
│   ├── confusion_matrix_normalized.png           # Normalize edilmiş karışıklık matrisi
│   ├── BoxPR_curve.png                           # Precision-Recall eğrisi
│   ├── BoxF1_curve.png                           # F1 eğrisi
│   └── args.yaml                                 # Ultralytics eğitim hiperparametreleri
├── traffic_vehicle_detection_yolo11.ipynb        # Baseline UA-DETRAC YOLO11n Colab notebook'u
├── traffic_vehicle_detection_balanced_yolo11.ipynb # Dengeli v2 (UA-DETRAC + VisDrone2019) YOLO11s Colab notebook'u
├── detrac_xml_annotations.zip                    # 93 video dizisine ait resmi XML etiketleri
├── yolo11_results.zip                            # Colab'dan indirilen baseline eğitim çıktıları
└── README.md                                     # Kapsamlı proje dokümantasyonu
```

---

## 🚀 Modeli Yerelde Çalıştırma (Inference)

Eğitilen en güncel `best.pt` ağırlıklarını kendi görsel veya videonuzda test etmek için:

```python
from ultralytics import YOLO

# Eğitilmiş modeli yükle
model = YOLO('weights/best.pt')

# Herhangi bir trafik görselinde çıkarım yap
results = model.predict(source='test_samples/user_test_photo.jpg', conf=0.25, save=True)

# Tespit edilen araçları yazdır
for box in results[0].boxes:
    cls_id = int(box.cls[0])
    conf = float(box.conf[0])
    print(f"Tespit: {model.names[cls_id]} - Güven: {conf:.2f}")
```
