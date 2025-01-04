def main():
    # Dosya adını ve yolunu belirtin
    dosya_adı = 'hidden_node_100__learning_rate_###/success_rates.txt'

    try:
        # Dosyayı açın ve satırları okuyun
        with open(dosya_adı, 'r') as dosya:
            satırlar = dosya.readlines()

        # Başarı oranlarını içeren satırları çıkartın
        basari_oranlari = []
        for satır in satırlar:
            # Satırdan başarı oranını çıkar
            basari_oran_str = satır.split(': ')[1].strip()

            # Başarı oranını listeye ekle
            basari_oranlari.append(float(basari_oran_str))

        # Ortalamayı hesaplayın
        ortalama = sum(basari_oranlari) / len(basari_oranlari)

        # Sonucu ekrana yazdırın
        print(f"Başarı Oranları Ortalaması: {ortalama}")

    except FileNotFoundError:
        print(f"{dosya_adı} adlı dosya bulunamadı.")
    except IndexError:
        print("Hata: Dosya içeriği uygun formatta değil.")
    except Exception as e:
        print(f"Bir hata oluştu: {e}")

if __name__ == "__main__":
    main()
