#!/usr/bin/env python
# coding: utf-8

# https://www.datacamp.com/community/tutorials/autoencoder-classifier-python

import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0" #model will be trained on GPU 0

import keras
import tensorflow as tf
from sklearn.metrics import cohen_kappa_score
# from metrics import cohen_kappa
# from keras import backend as K
# from ml_metrics import kappa
from matplotlib import pyplot as plt
import numpy as np
import gzip
# %matplotlib inline
from keras.models import Model, load_model
from keras.optimizers import RMSprop
from keras.layers import Input,Dense,Flatten,Dropout,merge,Reshape,Conv2D,MaxPooling2D,UpSampling2D,Conv2DTranspose
from keras.layers.normalization import BatchNormalization
from keras.models import Model,Sequential
from keras.callbacks import ModelCheckpoint, TensorBoard, Callback
from keras.optimizers import Adadelta, RMSprop,SGD,Adam
from keras import regularizers
from keras import backend as K
from keras.utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator

batch_size = 1
epochs = 25
inChannel = 3
x, y = 128, 128
input_img = Input(shape = (x, y, inChannel))
num_classes = 2

valid_datagen = ImageDataGenerator()
validation_generator = valid_datagen.flow_from_directory("./../data/MURA-v1.1/data2/valid/",
                                                  target_size=(x,y),
                                                  interpolation='bicubic',
                                                  class_mode='categorical',
                                                  shuffle=False,
                                                  batch_size=batch_size)

model = load_model('/pool001/bibek/hst/semisupervised/mid_classification_complete.h5')

filenames = validation_generator.filenames
nb_samples = len(filenames)

predict = model.predict_generator(validation_generator, steps=nb_samples)
y_pred = np.argmax(predict, axis=1)
labels = validation_generator.classes

print('kappa:', cohen_kappa_score(labels, y_pred))

