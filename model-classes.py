import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# read train
train=pd.read_csv("projectDatas/test/sign_mnist_train.csv")
print(train.shape)
train.head()

# read test
test=pd.read_csv("projectDatas/test/sign_mnist_test.csv")
print(test.shape)
test.head()

Y_train=train["label"]
Y_test=test["label"]
X_train=train.drop(labels=["label"],axis=1)
X_test=test.drop(labels=["label"],axis=1)

plt.figure(figsize=(15, 7))
sns.countplot(x=Y_train, order=Y_train.value_counts().index, palette="viridis")
plt.title("Number of Sign Language Classes")
plt.xlabel("Class")
plt.ylabel("Count")

# Show the plot
plt.show()