import tensorflow as tf
# from tensorflow.keras.preprocessing.image import ImageDataGenerator
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
# from keras.models import Sequential
# from keras.layers import Conv2D,Flatten,Dense,MaxPooling2D,Dropout
from sklearn.metrics import accuracy_score

# import ipywidgets as widgets
import io
from PIL import Image
import tqdm
from sklearn.model_selection import train_test_split
import cv2
from sklearn.utils import shuffle
import tensorflow as tf

# from model import predict1

st.title("Brain Tumour and Alzeihmer Detection using Python")

st.subheader("\n\n\n\nMade By:\n\tAbhishek Sharma 0108CS191004\n\tAyush Gupta 0108CS191003\n\tAbhishek Suryawanshi 0108CS191005\n\tPremansh Chauhan 0108CS191083")

labels = ['Glioma_Tumor','Meningioma_Tumor','No_Tumor','Pituitary_Tumor']

uploaded_file = st.file_uploader("Input MRI Image", type=['jpg','png'])

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    opencv_image = cv2.imdecode(file_bytes, 1)
    st.image(opencv_image, channels="BGR")
    opencv_image = cv2.resize(opencv_image, (150,150))
    plt.figure(figsize=(3,3))
    plt.imshow(opencv_image)
    opencv_image = opencv_image.reshape(1,150,150,3)

# hdf=pd.HDFStore('C:/Users/Dell Guna/Desktop/braintumor_model.h5',mode='r')
model_new=tf.keras.models.load_model('braintumor_model.h5')
if st.button('SHOW RESULT'):
    a=model_new.predict(opencv_image)
    indices = a.argmax()
    # print(labels[indices])
    # st.write("Test Result: ",labels[indices])
    X="Test Result: "+labels[indices]
    st.metric("",X)
    # st.score("Score: ",model_new.score())