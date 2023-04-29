import numpy as np
import tensorflow as tensor
from tensorflow import keras
import pandas as pd
from matplotlib import pyplot as plt
% matplotlib
inline
import os
import cv2
from keras.layers import Dense, Flatten

# paths to image folders
train_loc = 'C:\\Users\\kille\\ML_Assignment_1'
val_loc = 'C:\\Users\\kille\\ML_Assignment_1'

# path to 'train' folder
files_loc = train_loc
# Set the image size
img_size = (32, 32)

# creating empty lists for the images and labels
imgs = []
labls = []

# Loop over each folder from '0' to '9'
for label in range(10):
    folder_path = os.path.join(files_loc, 'train', str(label))
    # Loop over each image in the folder
    for file in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file)
    if file_path.endswith(('.tiff', '.bmp')):
        # Load the image and resize it to the desired size
        img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, img_size)
        # Append the image and label to the lists
        imgs.append(img)
        labls.append(label)

# Convert the lists to NumPy arrays
imgs = np.array(imgs)
labls = np.array(labls)

# Save the arrays in NumPy format
np.save('x_train.npy', imgs)
np.save('y_train.npy', labls)

# Set the path to the folder containing the 'val' folder
data_dir_val = val_loc
# Set the image size
img_size_val = (32, 32)
# Create empty lists for the images and labels
img_val = []
lab_val = []

# Loop over each folder from '0' to '9'
for label in range(10):
    folder_path = os.path.join(data_dir_val, 'val\\', str(label))

    # Loop over each image in the folder
    for file in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file)
        if file_path.endswith(('.tiff', '.bmp')):
            # Load the image and resize it to the desired size
            img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, img_size_val)
            # Append the image and label to the lists
            img_val.append(img)
            lab_val.append(label)

# Convert the lists to NumPy arrays
img_val = np.array(img_val)
lab_val = np.array(lab_val)

# Save the arrays in NumPy format
np.save('x_test.npy', img_val)
np.save('y_test.npy', lab_val)

# Load the dataset
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

mod = keras.Sequential([
    keras.layers.Flatten(), keras.layers.Dense(10, input_shape=(1024,), activation='sigmoid')
])
# compile the nn
mod.compile(optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy'])
# train the model
# 10 iterations
mod.fit(x_train, y_train, epochs=10, validation_data=(x_test, y_test))

# now scale and try to check the accuracy, divide dataset by 255
x_trn_scale = x_train / 255
x_tst_scale = x_test / 255
mod.fit(x_trn_scale, y_train, epochs=10, validation_data=(x_tst_scale, y_test))

# evaluate test dataset
mod.evaluate(x_tst_scale, y_test)

# predict 1st image
plt.matshow(x_test[0])
y_pred = mod.predict(x_tst_scale)
y_pred[0]
# this showing the 10 results for the input '0', we need to look for the value which is max
print('Predicted Value :', np.argmax(y_pred[0]))
# test some more values
plt.matshow(x_test[88])
print('Predicted Value :', np.argmax(y_pred[88]))
plt.matshow(x_test[177])
print('Predicted Value :', np.argmax(y_pred[177]))

# some predictions may not be proper
# building confusion matrix

y_pred_labls = [np.argmax(i) for i in y_pred]
print(y_pred_labls, len(y_pred_labls))
conf_mat = tensor.math.confusion_matrix(labels=y_test, predictions=y_pred_labls)
conf_mat

import seaborn as sn

plt.figure(figsize=(10, 10))
sn.heatmap(conf_mat, annot=True, fmt='d')
plt.xlabel('Predicted')
plt.ylabel('Actual')

# 1st Dense layer,the input is 32 x 32 = 1024 neurons, which will give 10 output(numbers from 0 to 9)
# In the 2nd Dense layer, the input is 10 neurons from above layers output

model2 = keras.Sequential([
    keras.layers.Flatten(),
    keras.layers.Dense(1024, input_shape=(1024,), activation='relu'),
    keras.layers.Dense(10, activation='softmax')
])
# compile the nn
model2.compile(optimizer='adam',
               loss='sparse_categorical_crossentropy',
               metrics=['accuracy']
               )
# train the model
# some 10 iterations done here
bn  = model2.fit(x_trn_scale, y_train, epochs=10, validation_data=(x_tst_scale, y_test))

# Observation : due to multiple layers the compiling will take more time to execute
model2.evaluate(x_tst_scale, y_test)

# Earlier we got 0.9213483333587646 now we got 0.9606741666793823 accuracy

# redo the confusion matrix
# build confusion matrix for accuracy

y_pred = model2.predict(x_tst_scale)
y_pred[0]
y_pred_labls = [np.argmax(i) for i in y_pred]
print(y_pred_labls, len(y_pred_labls))
conf_mat = tensor.math.confusion_matrix(labels=y_test, predictions=y_pred_labls)
conf_mat

plt.figure(figsize=(10, 10))
sn.heatmap(conf_mat, annot=True, fmt='d')
plt.xlabel('Predicted')
plt.ylabel('Actual')

# Evaluate the model
test_loss, test_acc = mod.evaluate(x_test, y_test)
print('Test accuracy:', test_acc)
# Plot the training and validation accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()
