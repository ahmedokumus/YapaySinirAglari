import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
import csv

# Neural network class definition.
class neuralNetwork:   
    
    # Initialise the neural network.
    def __init__(self, inputNodes, hiddenNodes, outputNodes,akj=None,ajm=None,bj=None,bm=None):
        
        # Set number of nodes in each input, hidden, output layer.
        self.inodes = inputNodes
        self.hnodes = hiddenNodes
        self.onodes = outputNodes

        self.akj = akj
        self.ajm = ajm
        self.bj = bj
        self.bm = bm

        if (akj is None and ajm is None and bj is None and bm is None):   
            self.akj = np.random.uniform(size=self.inodes * self.hnodes, low=-1, high=1).reshape(self.hnodes, self.inodes)
            self.ajm = np.random.uniform(size=self.hnodes * self.onodes, low=-1, high=1).reshape(self.hnodes,self.onodes)        

            self.bj = np.random.uniform(size=self.hnodes, low=-1, high=1).reshape(self.hnodes, 1)
            self.bm = np.random.uniform(size=self.onodes, low=-1, high=1).reshape(self.onodes, 1)
        
        # Activation mode
        self.activation_mode = None

        # Used for testing neural network's success.
        self.actual_list = np.array([])
        self.predict_list = np.array([])

        # Used for calculating MSE (Mean Squared Error).
        self.error_list = np.array([])

    # Activation function is;     
    #   Si for sigmoid, St for step,
    #   Re for relu, Ht for hyperbolic tangent.
    def activation_func(self, x, mod = None):

        # Stores the "mod" variable so that the net derivative calculation 
        # based on activation functions can be calculated in the "train" function.
        self.activation_mode = mod

        if mod == 'Si' or mod == None:
            return 1 / (1 + np.exp(-x))
        elif mod == 'St':
            return np.array(x > 0, dtype=np.float32)
        elif mod == 'Re':
            return np.maximum(x,0)
        elif mod == 'Ht':
            return np.tanh(x)
        
    # Predict output according to neural network's weights.
    # Basically make forward propagation and return output layer's outputs.
    def predict(self, input_list):

        # Convert input list to 2d array.
        inputs = np.array(input_list, ndmin=2).T

        # Calculate signals into hidden layer.
        Net_j = np.dot(self.akj, inputs) + self.bj

        Output_j = self.activation_func(Net_j)

        # Calculate signals into output layer
        Net_m = np.transpose(np.dot(np.transpose(Output_j),self.ajm)) + self.bm
        Output_m = self.activation_func(Net_m)

        return Output_m
    
    # Normalize values using lineer interpolation
    def normalize(self, original_range, desired_range, input):
        
        original_range = np.array(original_range)
        desired_range = np.array(desired_range)

        # Calculation of lineer interpolation 
        normalized_input = np.interp(input, original_range, desired_range)

        return normalized_input

# Class definition which include all showing and saving methods inside.
class ResultPrinter:
    def read_weights(file_path):
        # Initialize empty lists to store weights for different sections
        akj, ajm, bj, bm = [], [], [], []

        # Open the CSV file for reading
        with open(file_path, 'r') as file:
            # Create a CSV reader object
            reader = csv.reader(file)
            current_section = None  # Variable to keep track of the current section

            # Loop through each row in the CSV file
            for row in reader:
                if row:
                    # Check if the row ends with a colon (indicating a new section)
                    if row[0].endswith(":"):
                        current_section = row[0].strip(":").lower()
                    else:
                        # Convert the values in the row to floats and store them in the corresponding section
                        weights = np.array([float(value) for value in row])
                        if current_section == "akj":
                            akj.append(weights)
                        elif current_section == "ajm":
                            ajm.append(weights)
                        elif current_section == "bj":
                            bj.append(weights)
                        elif current_section == "bm":
                            bm.append(weights)

        # Convert the lists to NumPy arrays and return them
        return np.array(akj), np.array(ajm), np.array(bj), np.array(bm)


    # This function prints and plots the confusion matrix.
    # Normalization can be applied by setting `normalize=True`.
    def show_confusion_matrix(x_expected, y_predicted, normalize=False, title=None, cmap=plt.cm.Blues):
        if not title:
            title = 'Confusion matrix'

        # Compute confusion matrix.
        cm = confusion_matrix(x_expected, y_predicted)
        classes = np.unique(np.concatenate([x_expected, y_predicted]))

        if normalize:
            # Normalize the confusion matrix if specified.
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            print("Normalized confusion matrix")
        else:
            print('Confusion matrix, without normalization')

        """print(cm)"""

        # Create a new figure and axis for plotting.
        fig, ax = plt.subplots()
        
        # Display the confusion matrix as an image.
        im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
        ax.figure.colorbar(im, ax=ax)
        
        # Set axis labels and tick marks.
        ax.set(xticks=np.arange(len(classes)),
            yticks=np.arange(len(classes)),
            xticklabels=classes, yticklabels=classes,
            title=title,
            ylabel='Expected Value',
            xlabel='Predicted Value')

        # Rotate the tick labels for better visibility.
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

        # Loop over data dimensions and create text annotations for each cell.
        fmt = '.2f' if normalize else 'd'
        thresh = cm.max() / 2.
        for i in range(len(classes)):
            for j in range(len(classes)):
                ax.text(j, i, format(cm[i, j], fmt),
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black")
                
                # Check if the prediction is correct, and if so, add a light green border.
                if i == j and cm[i, j] > 0:
                    # Add a rectangle with a light green border around correctly predicted cells.
                    rect = plt.Rectangle((j - 0.5 + 0.05, i - 0.5 + 0.05), 0.9, 0.9, fill=False, edgecolor='lime', linewidth=5, alpha=1)
                    ax.add_patch(rect)

        # Adjust layout for better appearance.
        fig.tight_layout()
        
        # Return the figure and axis.
        return fig, ax
    
# Initialise neural network parameters.
input_nodes = 784
hidden_nodes = 173
output_nodes = 25

# Create instance of result printer.
result_printer = ResultPrinter()

### Test the neural network.
###########################################################################

# Specify the base file path and name
base_file_path = "hidden_node_173__learning_rate_0.05/weight_bias_value_"

#test_data_file = open("C:/Users/Emre/Desktop/ysa_yeni_donem/Ysa_Proje/dataset.csv", 'r')
test_data_file = open("projectDatas/sign_mnist_test/sign_mnist_test.csv", 'r')
#test_data_file = open("C:/Users/Emre/Desktop/ysa_yeni_donem/Ysa_Proje/Data/fashion-mnist_test.csv", 'r')
test_data_list = test_data_file.readlines()
test_data_file.close()
prediction_count = 0

# Loop through each file for the compare their success.
for epoch in range(25, 551, 25):
    ### Perform shift process between files here with the current epoch
    # Form the file path
    file_path = base_file_path + str(epoch) + ".csv"

    akj, ajm, bj, bm = ResultPrinter.read_weights(file_path)
    # Create an instance of the neural network.
    n = neuralNetwork(input_nodes, hidden_nodes, output_nodes, akj, ajm, bj, bm)

    # Reset prediction count for each file
    prediction_count = 0
    prediction_count2 = 0
    prediction_count3 = 0
    for record in test_data_list:

        # Split the record by the ',' commas.
        all_values = record.split(',')

        # Min and max values for original input and normalized input.
        original_range = [0, 255]
        desired_range = [0.01, 0.99]

        # Normalized inputs
        inputs = (np.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01

        # Calculate predicted list. 
        predicted_result = n.predict(inputs)

        # Save predicted and expected results in lists.
        n.predict_list = np.append(n.predict_list, np.argmax(predicted_result))
        n.actual_list = np.append(n.actual_list, int(all_values[0]))

        # Show predicted an expected results in test set.
        #print("Predicted: %d, Expected: %d " % (np.argmax(predicted_result), int(all_values[0])))

        # Calculate total correct prediction count.
    
    
        x=np.argmax(predicted_result)
        if x == int(all_values[0]):
            prediction_count += 1

        if x-1 == int(all_values[0]):
            prediction_count3 += 1

        # 9 25
        if x == 9:
            prediction_count2 += 1 
        if x == 25:
            prediction_count2 += 1
    # print(prediction_count)
    # print(prediction_count2)
    # print(prediction_count3)
        
    
    ### Show confusion_matrix
    ###########################################################################
    x_expected = n.actual_list
    y_predicted = n.predict_list

    ResultPrinter.show_confusion_matrix(x_expected=x_expected, y_predicted=y_predicted, normalize=False)
    plt.show()
    ###########################################################################

    # Calculate success rate of neural network.
    print(f"weight_bias_value_{str(epoch)} Success: %f" % (prediction_count / len(test_data_list)))
###########################################################################