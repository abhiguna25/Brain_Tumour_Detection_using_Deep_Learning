import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import streamlit as st
import cv2
import itertools
import keras
from keras.models import Sequential
from keras.layers import Conv2D,Flatten,Dense,MaxPooling2D,Dropout
from sklearn.metrics import accuracy_score

import ipywidgets as widgets
import io
from PIL import Image
import tqdm
from sklearn.model_selection import train_test_split
import cv2
from sklearn.utils import shuffle
import tensorflow as tf

st.title("Brain Tumour and Alzeihmer Detection using Python")

st.subheader("\n\n\n\nMade By:\n\tAbhishek Sharma 0108CS191004\n\tAyush Gupta 0108CS191003\n\tAbhishek Suryawanshi 0108CS191005\n\tPremansh Chauhan 0108CS191005")

#Input Image Code
uploaded_file = st.file_uploader("Input MRI Image", type="jpg")

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    opencv_image = cv2.imdecode(file_bytes, 1
)
    st.image(opencv_image, channels="BGR")
    opencv_image = cv2.resize(opencv_image, (150,150))
    plt.imshow(opencv_image)
    opencv_image = opencv_image.reshape(1,150,150,3)
    # st.subheader(np.array(opencv_image))
    

X_train = []
Y_train = []
image_size = 150
labels = ['glioma_tumor','meningioma_tumor','no_tumor','pituitary_tumor']
for i in labels:
    folderPath = os.path.join('C:/Users/Dell Guna/Desktop/data/Training',i)
    for j in os.listdir(folderPath):
        img = cv2.imread(os.path.join(folderPath,j))
        img = cv2.resize(img,(image_size,image_size))
        X_train.append(img)
        Y_train.append(i)
        
for i in labels:
    folderPath = os.path.join('C:/Users/Dell Guna/Desktop/data/Testing',i)
    for j in os.listdir(folderPath):
        img = cv2.imread(os.path.join(folderPath,j))
        img = cv2.resize(img,(image_size,image_size))
        X_train.append(img)
        Y_train.append(i)
 
X_train = np.array(X_train)
Y_train = np.array(Y_train)

X_train,Y_train = shuffle(X_train,Y_train,random_state=101)
# X_train.shape

X_train,X_test,y_train,y_test = train_test_split(X_train,Y_train,test_size=0.1,random_state=101)

y_train_new = []
for i in y_train:
    y_train_new.append(labels.index(i))
y_train=y_train_new
y_train = tf.keras.utils.to_categorical(y_train)

y_test_new = []
for i in y_test:
    y_test_new.append(labels.index(i))
y_test=y_test_new
y_test = tf.keras.utils.to_categorical(y_test)

model = Sequential()
model.add(Conv2D(32,(3,3),activation = 'relu',input_shape=(150,150,3)))
model.add(Conv2D(64,(3,3),activation='relu'))
model.add(MaxPooling2D(2,2))
model.add(Dropout(0.3))
model.add(Conv2D(64,(3,3),activation='relu'))
model.add(Conv2D(64,(3,3),activation='relu'))
model.add(Dropout(0.3))
model.add(MaxPooling2D(2,2))
model.add(Dropout(0.3))
model.add(Conv2D(128,(3,3),activation='relu'))
model.add(Conv2D(128,(3,3),activation='relu'))
model.add(Conv2D(128,(3,3),activation='relu'))
model.add(MaxPooling2D(2,2))
model.add(Dropout(0.3))
model.add(Conv2D(128,(3,3),activation='relu'))
model.add(Conv2D(256,(3,3),activation='relu'))
model.add(MaxPooling2D(2,2))
model.add(Dropout(0.3))
model.add(Flatten())
model.add(Dense(512,activation = 'relu'))
model.add(Dense(512,activation = 'relu'))
model.add(Dropout(0.3))
model.add(Dense(4,activation='softmax'))

# model.summary()
model.compile(loss='categorical_crossentropy',optimizer='Adam',metrics=['accuracy'])

# history = model.fit(X_train,y_train,epochs=20,validation_split=0.1)
model.fit(X_train,y_train,epochs=1,validation_split=0.1)

# acc = history.history['accuracy']
# val_acc = history.history['val_accuracy']
# epochs = range(len(acc))

# loss = history.history['loss']
# val_loss = history.history['val_loss']
# epochs = range(len(loss))

a=model.predict(opencv_image)
indices = a.argmax()
# print(indices)
st.write("Test Result: ",labels[indices])
