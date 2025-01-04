import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
import csv

# Sinir ağı sınıfının tanımı.
class neuralNetwork:

    # Sinir ağını başlat.
    def __init__(self, inputNodes, hiddenNodes, outputNodes, akj=None, ajm=None, bj=None, bm=None):

        # Her giriş, gizli ve çıkış katmanındaki düğüm sayısını ayarla.
        self.inodes = inputNodes
        self.hnodes = hiddenNodes
        self.onodes = outputNodes

        self.akj = akj
        self.ajm = ajm
        self.bj = bj
        self.bm = bm

        # # # if (akj is None and ajm is None and bj is None and bm is None):
        # # #     # Eğer ağırlıklar ve sapmalar sağlanmamışsa, onları rastgele başlat.
        # # #     self.akj = np.random.uniform(size=self.inodes * self.hnodes, low=-1, high=1).reshape(self.hnodes, self.inodes)
        # # #     self.ajm = np.random.uniform(size=self.hnodes * self.onodes, low=-1, high=1).reshape(self.hnodes, self.onodes)

        # # #     self.bj = np.random.uniform(size=self.hnodes, low=-1, high=1).reshape(self.hnodes, 1)
        # # #     self.bm = np.random.uniform(size=self.onodes, low=-1, high=1).reshape(self.onodes, 1)

        # Aktivasyon fonksiyonu
        self.activation_mode = None

        # Sinir ağının başarısını test etmek için kullanılır.
        self.actual_list = np.array([])
        self.predict_list = np.array([])

        # MSE (Ortalama Kare Hatası) hesaplamak için kullanılır.
        self.error_list = np.array([])

    def activation_func(self, x, mod=None):

        # Net türev hesaplaması için aktivasyon fonksiyonlarına dayalı olarak "mod" değişkenini saklar.
        self.activation_mode = mod

        if mod == 'Si' or mod == None: # Si sigmoid, 
            return 1 / (1 + np.exp(-x))
        elif mod == 'St': # St basamak,
            return np.array(x > 0, dtype=np.float32)
        elif mod == 'Re': # Re relu,
            return np.maximum(x, 0)
        elif mod == 'Ht': # Ht hiperbolik tanjant
            return np.tanh(x)

    # Sinir ağının ağırlıklarına göre çıkış tahmin et.
    # Temelde ileri yayılım yap ve çıkış katmanının çıkışlarını döndür.
    def predict(self, input_list):

        # Giriş listesini 2D dizisine dönüştürür.
        inputs = np.array(input_list, ndmin=2).T

        # Gizli katmana Net değerini hesaplar.
        Net_j = np.dot(self.akj, inputs) + self.bj

        Output_j = self.activation_func(Net_j)

        # Çıkış katmanına Net Değerini hesaplar.
        Net_m = np.transpose(np.dot(np.transpose(Output_j), self.ajm)) + self.bm
        Output_m = self.activation_func(Net_m)

        return Output_m

    # Lineer interpolasyon kullanarak değerleri normalize eder
    def normalize(self, original_range, desired_range, input):

        original_range = np.array(original_range)
        desired_range = np.array(desired_range)

        # Lineer interpolasyon hesaplaması
        normalized_input = np.interp(input, original_range, desired_range)

        return normalized_input

# Tüm gösterme ve kaydetme yöntemlerini içeren sınıf tanımı.
class ResultPrinter:
    def read_weights(file_path):
        # Farklı bölümler için ağırlıkları depolamak için boş listeleri başlat
        akj, ajm, bj, bm = [], [], [], []

        # CSV dosyasını okumak için aç
        with open(file_path, 'r') as file:

            # Bir CSV okuyucu nesnesi oluştur
            reader = csv.reader(file)
            current_section = None 

            # CSV dosyasındaki her satırı döngü ile geç
            for row in reader:
                if row:
                    # Satırın iki nokta üst üste ile bittiğini kontrol et (yeni bir bölümü gösterir)
                    if row[0].endswith(":"):
                        current_section = row[0].strip(":").lower()
                    else:
                        # Satırdaki değerleri float'a çevir ve bunları ilgili bölümde depola
                        weights = np.array([float(value) for value in row])
                        if current_section == "akj":
                            akj.append(weights)
                        elif current_section == "ajm":
                            ajm.append(weights)
                        elif current_section == "bj":
                            bj.append(weights)
                        elif current_section == "bm":
                            bm.append(weights)

        # Listeleri NumPy dizilerine dönüştür ve bunları döndür
        return np.array(akj), np.array(ajm), np.array(bj), np.array(bm)

    # Bu fonksiyon karışıklık matrisini yazdırır ve çizer.
    # Normalleştirme, `normalize=True` olarak ayarlanarak uygulanabilir.
    def show_confusion_matrix(x_expected, y_predicted, normalize=False, title=None, cmap=plt.cm.Blues):
        if not title:
            title = 'Confusion matrix'

        # Compute confusion matrix.
        # Karışıklık matrisini hesapla.
        cm = confusion_matrix(x_expected, y_predicted)
        classes = np.unique(np.concatenate([x_expected, y_predicted]))

        if normalize:
            # Normalize the confusion matrix if specified.
            # Belirtilmişse karışıklık matrisini normalize et.
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            print("Normalized confusion matrix")
        else:
            print('Confusion matrix, without normalization')

        """print(cm)"""

        # Create a new figure and axis for plotting.
        # Çizim için yeni bir rakam ve eksen oluştur.
        fig, ax = plt.subplots()

        # Display the confusion matrix as an image.
        # Karışıklık matrisini bir resim olarak görüntüle.
        im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
        ax.figure.colorbar(im, ax=ax)

        # Set axis labels and tick marks.
        # Eksen etiketlerini ve işaretlerini ayarla.
        ax.set(xticks=np.arange(len(classes)),
               yticks=np.arange(len(classes)),
               xticklabels=classes, yticklabels=classes,
               title=title,
               ylabel='Expected Value',
               xlabel='Predicted Value')

        # Rotate the tick labels for better visibility.
        # Daha iyi görünürlük için işaret etiketlerini döndür.
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

        # Loop over data dimensions and create text annotations for each cell.
        # Veri boyutları üzerinde döngü yapın ve her hücre için metin açıklamaları oluşturun.
        fmt = '.2f' if normalize else 'd'
        thresh = cm.max() / 2.
        for i in range(len(classes)):
            for j in range(len(classes)):
                ax.text(j, i, format(cm[i, j], fmt),
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black")

                # Check if the prediction is correct, and if so, add a light green border.
                # Tahminin doğru olup olmadığını kontrol edin ve doğruysa açık yeşil bir kenar ekleyin.
                if i == j and cm[i, j] > 0:
                    # Add a rectangle with a light green border around correctly predicted cells.
                    # Doğru tahmin edilen hücrelerin etrafında açık yeşil bir kenar ile bir dikdörtgen ekleyin.
                    rect = plt.Rectangle((j - 0.5 + 0.05, i - 0.5 + 0.05), 0.9, 0.9, fill=False, edgecolor='lime',
                                         linewidth=5, alpha=1)
                    ax.add_patch(rect)

        # Adjust layout for better appearance.
        # Daha iyi görünüm için düzeni ayarla.
        fig.tight_layout()

        # Return the figure and axis.
        # Şekil ve ekseni döndür.
        return fig, ax

# Initialise neural network parameters.
# Sinir ağı parametrelerini başlat.
input_nodes = 784
hidden_nodes = 173
output_nodes = 25

# Create instance of result printer.
# Sonuç yazıcısının örneğini oluştur.
result_printer = ResultPrinter()

### Test the neural network. ##########################################################

# Open the test data file and read the lines.
# Test veri dosyasını açın ve satırları okuyun.
test_data_file = open("projectDatas/sign_mnist_test/sign_mnist_test.csv", 'r')
test_data_list = test_data_file.readlines()
test_data_file.close()
prediction_count = 0

### Perform shift process between files here with the current epoch
# Form the file path
# Dosya yolu oluştur
file_path = "hidden_node_173__learning_rate_0.05/weight_bias_value_50.csv"

# Read the weights from the file.
# Ağırlıkları dosyadan oku.
akj, ajm, bj, bm = ResultPrinter.read_weights(file_path)
# Create an instance of the neural network.
# Sinir ağı örneği oluştur.
n = neuralNetwork(input_nodes, hidden_nodes, output_nodes, akj, ajm, bj, bm)

# Reset prediction count for each file
# Her dosya için tahmin sayısını sıfırla
prediction_count = 0
prediction_count2 = 0
prediction_count3 = 0
for record in test_data_list:

    # Split the record by the ',' commas.
    # Kaydı ',' virgülle bölelim.
    all_values = record.split(',')

    # Min and max values for original input and normalized input.
    # Orijinal giriş ve normalize giriş için min ve max değerler.
    original_range = [0, 255]
    desired_range = [0.01, 0.99]

    # Normalized inputs
    # Normalize edilmiş girişler
    inputs = (np.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01

    # Calculate predicted list.
    # Tahmin edilen listeyi hesapla.
    predicted_result = n.predict(inputs)

    # Save predicted and expected results in lists.
    # Tahmin edilen ve beklenen sonuçları listelerde sakla.
    n.predict_list = np.append(n.predict_list, np.argmax(predicted_result))
    n.actual_list = np.append(n.actual_list, int(all_values[0]))

    # Show predicted an expected results in test set.
    # Test setinde tahmin edilen ve beklenen sonuçları göster.
    # print("Predicted: %d, Expected: %d " % (np.argmax(predicted_result), int(all_values[0])))

    # Calculate total correct prediction count.
    # Toplam doğru tahmin sayısını hesapla.
    x = np.argmax(predicted_result)
    if x == int(all_values[0]):
        prediction_count += 1

    if x - 1 == int(all_values[0]):
        prediction_count3 += 1

    # 9 25
    if x == 9:
        prediction_count2 += 1
    if x == 25:
        prediction_count2 += 1

### Show confusion_matrix ####################################################################
# Karışıklık matrisini göster
x_expected = n.actual_list  # Beklenen değer
y_predicted = n.predict_list # Tahmin edilen değer

ResultPrinter.show_confusion_matrix(x_expected=x_expected, y_predicted=y_predicted, normalize=False)
plt.show()
###########################################################################

# Calculate success rate of neural network.
print(f"{file_path}\nSuccess: %f" % (prediction_count / len(test_data_list)))
###########################################################################