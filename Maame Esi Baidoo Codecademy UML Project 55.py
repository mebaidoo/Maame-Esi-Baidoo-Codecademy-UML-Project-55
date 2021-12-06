#Handwriting Recognition using K-Means

#Importing necessary modules
import codecademylib3_seaborn
import numpy as np
from matplotlib import pyplot as plt
from sklearn import datasets

#Loading the digits data from datasets
digits = datasets.load_digits()

#Inspecting the data
#print(digits)
print(digits.DESCR)
#print(digits.data)
print(digits.target)

#Visualizing the image at index 100
plt.gray()
plt.matshow(digits.images[100])
plt.show()

#Looking at the label (number) associated with this image
print(digits.target[100])

#Clustering all the images into groups
from sklearn.cluster import KMeans
#Using 10 clusters since there are 10 different digits in the images
model = KMeans(n_clusters = 10, random_state = 42)
model.fit(digits.data)

#Visualizing the centroids
fig = plt.figure(figsize = (8, 3))
plt.suptitle("Cluser Center Images", fontsize=14, fontweight="bold")
#Plotting each of the ten centroids in the figure
for i in range(10):
  # Initialize subplots in a grid of 2X5, at i+1th position
  ax = fig.add_subplot(2, 5, 1 + i)
  # Making each of the cluster centers into an 8x8 2D array and displaying images
  ax.imshow(model.cluster_centers_[i].reshape((8, 8)), cmap=plt.cm.binary)
plt.show()

#Testing the model
new_samples = np.array([
[0.00,0.00,0.61,4.27,4.57,1.53,0.00,0.00,0.00,0.53,6.18,7.47,7.32,5.34,0.00,0.00,0.00,0.53,5.80,2.59,5.49,6.10,0.00,0.00,0.00,0.00,0.00,0.00,4.88,6.86,0.00,0.00,0.00,0.00,3.20,6.79,7.62,7.40,4.80,1.53,0.00,0.00,6.10,6.86,5.19,5.11,6.86,3.36,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00],
[0.00,0.00,0.00,1.22,0.08,0.00,0.00,0.00,0.00,0.00,1.15,7.63,2.67,0.00,0.00,0.00,0.00,0.00,1.53,7.62,2.29,0.00,0.00,0.00,0.00,0.00,1.53,7.62,2.29,0.00,0.00,0.00,0.00,0.00,1.53,7.62,2.29,0.00,0.00,0.00,0.00,0.00,0.99,7.47,4.96,0.00,0.00,0.00,0.00,0.00,0.00,2.90,2.90,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00],
[0.00,0.00,0.08,1.22,1.52,0.15,0.00,0.00,0.00,0.46,5.95,7.62,7.62,3.66,0.00,0.00,0.00,2.44,7.62,3.81,7.02,4.58,0.00,0.00,0.00,0.00,0.61,0.00,6.64,5.26,0.00,0.00,0.00,0.69,4.50,4.57,7.09,6.71,3.66,1.07,0.00,1.14,6.02,6.10,6.10,6.41,7.63,3.82,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00],
[0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.15,4.12,1.30,0.00,0.00,0.00,0.00,0.00,0.30,7.24,4.80,0.00,0.00,0.00,0.00,0.00,0.00,5.87,6.33,0.00,0.00,0.00,0.00,0.00,0.00,4.65,6.86,0.00,0.00,0.00,0.00,0.00,0.00,4.58,6.87,0.00,0.00,0.00,0.00,0.00,0.00,4.50,7.32,0.00,0.00,0.00,0.00,0.00,0.00,0.69,1.60,0.00,0.00,0.00]
])

#Predicting the labels for new_samples
new_labels = model.predict(new_samples)
print(new_labels)
#Mapping out each of the labels with their corresponding digits
for i in range(len(new_labels)):
  if new_labels[i] == 0:
    print(0, end='')
  elif new_labels[i] == 1:
    print(9, end='')
  elif new_labels[i] == 2:
    print(2, end='')
  elif new_labels[i] == 3:
    print(1, end='')
  elif new_labels[i] == 4:
    print(6, end='')
  elif new_labels[i] == 5:
    print(8, end='')
  elif new_labels[i] == 6:
    print(4, end='')
  elif new_labels[i] == 7:
    print(5, end='')
  elif new_labels[i] == 8:
    print(7, end='')
  elif new_labels[i] == 9:
    print(3, end='')