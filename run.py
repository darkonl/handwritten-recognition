import pandas as pd
import os
from PIL import Image
import numpy as np

# Moving to 'data' as working directory
os.chdir('data')
print("New working directory", os.getcwd())

# Listing available files in the working directory
print("Listing working directory files", os.listdir(os.getcwd()))

# Reading training examples
training = pd.read_csv("train.csv")

# Printing first 10 rows and data's shape
print(training.head(10))

# getting first row's information
# index 0 contains the label
# other indexes contain pixel's value
digit0 = np.array(training.iloc[0][1:])
digit0 = digit0.reshape(28, 28)
print(digit0.shape)

img = Image.fromarray(digit0)
img.save('testrgb.png')
