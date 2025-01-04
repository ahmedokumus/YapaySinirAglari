# İşaret Dili Rakamları Sınıflandırma - Çok Katmanlı Algılayıcı YSA

Bu proje, işaret dili rakamlarını tanımak için Çok Katmanlı Algılayıcı (Multilayer Perceptron) yapay sinir ağı kullanarak geliştirilmiş bir sınıflandırma uygulamasıdır.

## Proje Hakkında

Bu projede, işaret dili rakamlarının görüntülerini sınıflandırmak için bir yapay sinir ağı modeli geliştirilmiştir. Model, MNIST işaret dili veri seti üzerinde eğitilmiş ve test edilmiştir.

### Özellikler

- Çok Katmanlı Algılayıcı (MLP) mimarisi
- Sigmoid aktivasyon fonksiyonu
- Geri yayılım algoritması ile eğitim
- Farklı öğrenme oranları ve gizli katman düğüm sayıları ile deneyler
- Karmaşıklık matrisi ile performans analizi
- Ağırlık ve bias değerlerinin kaydedilmesi

## Gereksinimler

```
numpy==1.26.1
pandas==2.1.4
matplotlib==3.8.2
scikit-learn==1.3.2
tqdm==4.66.1
```

## Proje Yapısı

- `main.py`: Ana program dosyası
- `model-classes.py`: Model sınıfları ve veri ön işleme
- `multi-test.py`: Çoklu test senaryoları
- `single-test.py`: Tekli test senaryoları
- `projectDatas/`: Veri setleri dizini
  - `sign_mnist_train/`: Eğitim verileri
  - `sign_mnist_test/`: Test verileri

## Kullanım

1. Gerekli kütüphaneleri yükleyin:
```bash
pip install -r requirements.txt
```

2. Programı çalıştırın:
```bash
python main.py
```

## Model Mimarisi

- Giriş Katmanı: 784 nöron (28x28 piksel görüntüler)
- Gizli Katman: Değişken sayıda nöron (100-200 arası test edildi)
- Çıkış Katmanı: 25 nöron (sınıf sayısı)

## Performans

Model performansı, farklı hiperparametreler ile test edilmiş ve sonuçlar `success_rates.txt` dosyasında kaydedilmiştir. Ayrıca, karmaşıklık matrisi ile modelin sınıflandırma performansı görselleştirilmiştir.

### Farklı Hiperparametreler ile Elde Edilen Başarı Oranları

1. **173 Gizli Nöron, Öğrenme Oranı: 0.05**
   - Ortalama Başarı Oranı: %76.07
   - En Yüksek Başarı Oranı: %77.49 (74. epoch)

2. **200 Gizli Nöron, Öğrenme Oranı: 0.03**
   - Ortalama Başarı Oranı: %73.56
   - En Yüksek Başarı Oranı: %74.49 (305. epoch)

3. **184 Gizli Nöron, Öğrenme Oranı: 0.01**
   - Ortalama Başarı Oranı: %73.76
   - En Yüksek Başarı Oranı: %76.82 (825. epoch)

4. **100 Gizli Nöron, Öğrenme Oranı: 0.07**
   - Ortalama Başarı Oranı: %74.65
   - En Yüksek Başarı Oranı: %76.39 (463. epoch)
   - Model daha az gizli nöron kullanmasına rağmen iyi bir performans göstermiştir
   - Yüksek öğrenme oranı sayesinde hızlı öğrenme gerçekleşmiştir

Genel olarak, 173 gizli nöronlu ve 0.05 öğrenme oranlı model en iyi performansı göstermiştir. Bu model daha hızlı öğrenme ve daha yüksek doğruluk oranı sağlamıştır. Ancak, 100 gizli nöronlu model de daha az karmaşık yapısına rağmen rekabetçi sonuçlar elde etmiştir.

### Model Ağırlıkları

Eğitim sırasında her 25 epoch'ta bir kaydedilen ağırlık ve bias değerleri CSV dosyaları halinde mevcuttur. Bu dosyalar boyut nedeniyle git deposuna eklenmemiştir, ancak istenildiği takdirde paylaşılabilir. Her model için aşağıdaki dosyalar mevcuttur:

- `weight_bias_value_X.csv`: X. epoch'taki ağırlık ve bias değerleri
- Her dosya yaklaşık 3.4MB boyutundadır
- Dosyalar modelin tam olarak replike edilebilmesi için gerekli tüm parametreleri içermektedir

## Katkıda Bulunanlar

Bu proje, yapay sinir ağları dersi kapsamında geliştirilmiştir. 