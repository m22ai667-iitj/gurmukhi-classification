import numpy as np
import tensorflow as tensor
from tensorflow import keras
import pandas as pd
from matplotlib import pyplot as plt
%matplotlib inline
import os
import cv2
from keras.layers import Dense, Flatten

# Paths to image folders
train_pth = 'C:\\Users\\kille\\ML_Assignment_1'
val_pth = 'C:\\Users\\kille\\ML_Assignment_1'

# Path to 'train data set' folder
files_path = train_pth
# Set the image size
img_size = (32, 32)

# Creating empty lists for the images and labels
images = []
labels = []

# Loop over each folder from '0' to '9'
for label in range(10):
     folder_path = os.path.join(files_path, 'train', str(label))
     # Loop over each image in the folder
     for file in os.listdir(folder_path):
         file_path = os.path.join(folder_path, file)
     if file_path.endswith(('.tiff','.bmp')):
         # Load the image and resize it to the desired size
         img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
         img = cv2.resize(img, img_size)
         # Append the image and label to the lists
         images.append(img)
         labels.append(label)

# Converting the lists to NumPy arrays
images = np.array(images)
labels = np.array(labels)

# Saveing the arrays in NumPy format
np.save('x_train.npy', images)
np.save('y_train.npy', labels)

# Set the path to the folder containing the 'val' folder
data_dir_val = val_pth
# Set the image size
img_size_val = (32, 32)
# Create empty lists for the images and labels
images_val = []
labels_val = []


# Loop over each folder from '0' to '9'
for label in range(10):
 folder_path = os.path.join(data_dir_val, 'val\\', str(label))

 # Loop over each image in the folder
 for file in os.listdir(folder_path):
     file_path = os.path.join(folder_path, file)
     if file_path.endswith(('.tiff','.bmp')):
         # Load the image and resize it to the desired size
         img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
         img = cv2.resize(img, img_size_val)
         # Append the image and label to the lists
         images_val.append(img)
         labels_val.append(label)
        
# Convert the lists to NumPy arrays
images_val = np.array(images_val)
labels_val = np.array(labels_val)

# Save the arrays in NumPy format
np.save('x_test.npy', images_val)
np.save('y_test.npy', labels_val)

# Loading the dataset
x_train = np.load('x_train.npy')
y_train = np.load('y_train.npy')
x_test = np.load('x_test.npy')
y_test = np.load('y_test.npy')

# test the images are loaded correctly
print(len(x_train))
print(len(x_test))
x_train[0].shape
x_train[0]
plt.matshow(x_train[0])
plt.matshow(x_train[9])
print(x_train.shape)
print(x_test.shape)
y_train
y_test
plt.matshow(x_test[150])

model = keras.Sequential([
 keras.layers.Flatten(),keras.layers.Dense(10, input_shape=(1024,),activation = 'sigmoid')
])
# compile the nn
model.compile(optimizer='adam',
 loss='sparse_categorical_crossentropy',
 metrics=['accuracy'])
# train the model
# some 10 iterations done here
model.fit(x_train, y_train,epochs= 10, validation_data=(x_test, y_test))

# now scale and try to check the accuracy, divide dataset by 255
x_train_scaled = x_train/255
x_test_scaled = x_test/255
model.fit(x_train_scaled, y_train,epochs= 10, validation_data=(x_test_scaled, y_test))

# evaluate test dataset
model.evaluate(x_test_scaled,y_test)

# predict 1st image
plt.matshow(x_test[0])
y_predicted = model.predict(x_test_scaled)
y_predicted[0]
# this showing the 10 results for the input '0', we need to look for the value which is max
print('Predicted Value is ',np.argmax(y_predicted[0]))
# test some more values
plt.matshow(x_test[88])
print('Predicted Value is ',np.argmax(y_predicted[88]))
plt.matshow(x_test[177])
print('Predicted Value is ',np.argmax(y_predicted[177]))


# some predictions may not be not right
# build confusion matrix to see how our prediction looks like
# convert to concrete values
y_predicted_labels=[np.argmax(i) for i in y_predicted]
print(y_predicted_labels, len(y_predicted_labels))
conf_mat = tf.math.confusion_matrix(labels=y_test, predictions=y_predicted_labels)
conf_mat


import seaborn as sn
plt.figure(figsize = (10,10))
sn.heatmap(conf_mat,annot=True,fmt='d')
plt.xlabel('Predicted')
plt.ylabel('Actual')


# in 1st Dense layer,the input is 32 x 32 = 1024 neurons, which will give 10 output(numbers from 0 to 9)
# 2nd Dense layer,the input is 10 neurons from above layers output
# we can add more layers for accuracy
model2 = keras.Sequential([
 keras.layers.Flatten(),
 keras.layers.Dense(1024,input_shape=(1024,), activation='relu'),
 keras.layers.Dense(10, activation='softmax')
])
# compile the nn
model2.compile(optimizer='adam',
 loss='sparse_categorical_crossentropy',
 metrics=['accuracy']
 )
# train the model
# some 10 iterations done here
history = model2.fit(x_train_scaled, y_train,epochs= 10, validation_data=(x_test_scaled, y_test))


# Observation : due to multiple layers the compiling will take more time to execute
# we also got amazing accuracy than earlier
# evaluate test dataset on modified model
model2.evaluate(x_test_scaled,y_test)


# Earlier we got 0.9213483333587646 now we got 0.9606741666793823 accuracy

# redo the confusion matrix
# build confusion matrix to see how our prediction looks like
# convert to concrete values
y_predicted = model2.predict(x_test_scaled)
y_predicted[0]
y_predicted_labels=[np.argmax(i) for i in y_predicted]
print(y_predicted_labels, len(y_predicted_labels))
conf_mat = tf.math.confusion_matrix(labels=y_test, predictions=y_predicted_labels)
conf_mat


plt.figure(figsize = (10,10))
sn.heatmap(conf_mat,annot=True,fmt='d')
plt.xlabel('Predicted')
plt.ylabel('Actual')


# Evaluate the model
test_loss, test_acc = model.evaluate(x_test, y_test)
print('Test accuracy:', test_acc)
# Plot the training and validation accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

