import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from IPython.core.display import display, HTML

# For visualize with jupyter directly use '%run -i visulation.py'

# Data csv has a label which is: 
# label,pixel1,pixel2,.....,pixel784
# after that all data contuine as that format

# be carefull if your data have same spesific proparty 
# code must be modified if your dataset properties different 

#filename = 'C:/Users/Emrecan/Desktop/ysa_yeni_donem/Ysa_Proje/ornekleme_sonucu.csv'  # Update the file path
filename = 'C:/Users/ahmed/source/pythonRepos/YapaySinirAglari/projectDatas/sign_mnist_train.csv'  # Update the file path

df = pd.read_csv(filename)

def show_dataframe(dataframe):

    # To normal display
    display(dataframe)

    # To force show all data in column
    # display(HTML(dataframe.to_html()))

def show_histogram(data):

    # plotting a bar plot
    value_counts = data.value_counts().sort_index()
    plt.bar(value_counts.index, value_counts.values)
    plt.show()


def show_image(df, cmap='gray'):

    # Create a new figure with size (20, 20)
    plt.figure(figsize=(20, 20))
    for i in range(25):

        # Select a random index from the dataframe
        rnd_img_idx = np.random.randint(0, len(df))

        # Get the values of the random selected row by using the random index
        img = df.iloc[rnd_img_idx]

        # Create subplot, 5x5 grid, position i+1
        plt.subplot(5, 5, i + 1)

        # Show the selected image, reshape and normalize the image and using color map
        plt.imshow(np.array(img.iloc[1:]).astype('uint8').reshape(28, 28), cmap=cmap)  # Assuming images are 28x28

        # Hide the axis on image
        plt.axis('off')

        # Add title with selected image's attributes
        plt.title(f"Label: {img['label']}")
        
    # show the final plot
    plt.show()

# To show first 5 row
head = df.head()
show_dataframe(head)

# To plotting a histogram
show_histogram(df["label"])

# To plotting first 25 images
show_image(df)