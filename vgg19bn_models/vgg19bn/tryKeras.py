import keras
from keras.preprocessing import image
from keras.applications import vgg19, resnet50, InceptionResNetV2
import os, math
import numpy as np
from keras import backend as K
from keras.models import Sequential
from keras.layers import Activation
#following added by JMS 
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers.core import Dense, Flatten, Dropout
from keras.optimizers import SGD, Adam
from keras.metrics import categorical_crossentropy
from keras.preprocessing.image import ImageDataGenerator
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras.models import load_model
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import MinMaxScaler
import itertools
import matplotlib.pyplot as plt
from keras.models import load_model
import h5py
import functools

IMAGE_SIZE    = (256, 256)
NUM_CLASSES   = 2
BATCH_SIZE    = 8  # try reducing batch size or freeze more layers if your GPU runs out of memory
FREEZE_LAYERS = 2  # freeze the first this many layers for training
NUM_EPOCHS    = 20





train_datagen = ImageDataGenerator(
                                   rotation_range=40,
                                   width_shift_range=0.2,
                                   height_shift_range=0.2,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   channel_shift_range=10,
                                   horizontal_flip=True,
                                   fill_mode='nearest')

train_batches = train_datagen.flow_from_directory("./../data/MURA-v1.1/data2/train/",
                                                  target_size=IMAGE_SIZE,
                                                  interpolation='bicubic',
                                                  class_mode='categorical',
                                                  shuffle=True,
                                                  batch_size=BATCH_SIZE)

valid_datagen = ImageDataGenerator()
valid_batches = valid_datagen.flow_from_directory("./../data/MURA-v1.1/data2/valid/",
                                                  target_size=IMAGE_SIZE,
                                                  interpolation='bicubic',
                                                  class_mode='categorical',
                                                  shuffle=False,
                                                  batch_size=BATCH_SIZE)



train_generator = train_datagen.flow_from_directory(
    directory="./../data/MURA-v1.1/data2/train/",
    target_size=IMAGE_SIZE,
    color_mode="rgb",
    batch_size=32,
    class_mode="categorical",
    shuffle=True,
    seed=42
)

valid_generator = valid_datagen.flow_from_directory(
    directory="./../data/MURA-v1.1/data2/valid/",
    target_size=IMAGE_SIZE,
    color_mode="rgb",
    batch_size=32,
    class_mode="categorical",
    shuffle=True,
    seed=42
)

net = vgg19.VGG19(include_top=False,
                        weights=None,
                        input_tensor=None,
                        input_shape=(IMAGE_SIZE[0],IMAGE_SIZE[1],3))
x = net.output
x = Flatten()(x)
x = Dropout(0.5)(x)
output_layer = Dense(2, activation='softmax', name='softmax')(x)

net_final = Model(inputs=net.input, outputs=output_layer)

print(len(net_final.layers))
for layer in net_final.layers[:FREEZE_LAYERS]:
    layer.trainable = True
for layer in net_final.layers[FREEZE_LAYERS:]:
    layer.trainable = True
    
    
learning_rate = 0.001
decay_rate = learning_rate / NUM_EPOCHS
momentum = 0.8
adam = Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999, decay=decay_rate, epsilon=0.0000001)

net_final.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])


# train the model
net_final.fit_generator(train_batches,
                        steps_per_epoch = train_batches.samples // BATCH_SIZE,
                        validation_data = valid_batches,
                        validation_steps = valid_batches.samples // BATCH_SIZE,
                        epochs = NUM_EPOCHS)

# save trained weights
net_final.save('model-inception_resnet_v2-final.h5')