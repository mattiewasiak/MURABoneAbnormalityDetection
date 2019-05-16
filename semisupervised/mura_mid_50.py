#!/usr/bin/env python
# coding: utf-8

# https://www.datacamp.com/community/tutorials/autoencoder-classifier-python

import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0" #model will be trained on GPU 0

import keras
import tensorflow as tf
from matplotlib import pyplot as plt
import numpy as np
import gzip
# %matplotlib inline
from keras.models import Model
from keras.optimizers import RMSprop
from keras.layers import Input,Dense,Flatten,Dropout,merge,Reshape,Conv2D,MaxPooling2D,UpSampling2D,Conv2DTranspose
from keras.layers.normalization import BatchNormalization
from keras.models import Model,Sequential
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.optimizers import Adadelta, RMSprop,SGD,Adam
from keras import regularizers
from keras import backend as K
from keras.utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator

batch_size = 64
epochs = 25
inChannel = 3
x, y = 128, 128
input_img = Input(shape = (x, y, inChannel))
num_classes = 2

train_datagen = ImageDataGenerator()
#                                    rotation_range=40,
#                                    width_shift_range=0.2,
#                                    height_shift_range=0.2,
#                                    shear_range=0.2,
#                                    zoom_range=0.2,
#                                    channel_shift_range=10,
#                                    horizontal_flip=True,
#                                    fill_mode='nearest')

train_batches = train_datagen.flow_from_directory("./../data/MURA-v1.1/data2/train/",
                                                  target_size=(x,y),
                                                  interpolation='bicubic',
                                                  class_mode='categorical',
                                                  shuffle=True,
                                                  batch_size=batch_size)

train_auto = train_datagen.flow_from_directory("./../data/MURA-v1.1/data2/train/",
                                                  target_size=(x,y),
                                                  interpolation='bicubic',
                                                  class_mode='input',
                                                  shuffle=True,
                                                  batch_size=batch_size)

valid_datagen = ImageDataGenerator()
valid_batches = valid_datagen.flow_from_directory("./../data/MURA-v1.1/data2/valid/",
                                                  target_size=(x,y),
                                                  interpolation='bicubic',
                                                  class_mode='categorical',
                                                  shuffle=False,
                                                  batch_size=batch_size)

valid_auto = valid_datagen.flow_from_directory("./../data/MURA-v1.1/data2/valid/",
                                                  target_size=(x,y),
                                                  interpolation='bicubic',
                                                  class_mode='input',
                                                  shuffle=False,
                                                  batch_size=batch_size)

def encoder(input_img):
    #encoder
    #input = 28 x 28 x 1 (wide and thin)
    conv1 = Conv2D(32, (5,5), activation='relu', padding='same')(input_img) #28 x 28 x 32
    conv1 = BatchNormalization()(conv1)
    conv1 = Conv2D(32, (5,5), activation='relu', padding='same')(conv1)
    conv1 = BatchNormalization()(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1) #14 x 14 x 32
    conv2 = Conv2D(64, (5,5), activation='relu', padding='same')(pool1) #14 x 14 x 64
    conv2 = BatchNormalization()(conv2)
    conv2 = Conv2D(64, (5,5), activation='relu', padding='same')(conv2)
    conv2 = BatchNormalization()(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2) #7 x 7 x 64
    conv3 = Conv2D(128, (5,5), activation='relu', padding='same')(pool2) #7 x 7 x 128 (small and thick)
    conv3 = BatchNormalization()(conv3)
    conv3 = Conv2D(128, (5,5), activation='relu', padding='same')(conv3)
    conv3 = BatchNormalization()(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3) #7 x 7 x 64
    conv4 = Conv2D(256, (5,5), activation='relu', padding='same')(pool3) #7 x 7 x 256 (small and thick)
    conv4 = BatchNormalization()(conv4)
    conv4 = Conv2D(256, (5,5), activation='relu', padding='same')(conv4)
    conv4 = BatchNormalization()(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4) #7 x 7 x 64
    conv5 = Conv2D(512, (5,5), activation='relu', padding='same')(pool4) #7 x 7 x 256 (small and thick)
    conv5 = BatchNormalization()(conv5)
    conv5 = Conv2D(512, (5,5), activation='relu', padding='same')(conv5)
    conv5 = BatchNormalization()(conv5)
    return conv5

def decoder(conv5):    
    #decoder
    conv6 = Conv2D(256, (5,5), activation='relu', padding='same')(conv5) #7 x 7 x 128
    conv6 = BatchNormalization()(conv6)
    conv6 = Conv2D(256, (5,5), activation='relu', padding='same')(conv6)
    conv6 = BatchNormalization()(conv6)
    up1 = UpSampling2D((2,2))(conv6) #14 x 14 x 64
    conv7 = Conv2D(128, (5,5), activation='relu', padding='same')(up1) #7 x 7 x 128
    conv7 = BatchNormalization()(conv7)
    conv7 = Conv2D(128, (5,5), activation='relu', padding='same')(conv7)
    conv7 = BatchNormalization()(conv7)
    up2 = UpSampling2D((2,2))(conv7) #14 x 14 x 64
    conv8 = Conv2D(64, (5,5), activation='relu', padding='same')(up2) #7 x 7 x 64
    conv8 = BatchNormalization()(conv8)
    conv8 = Conv2D(64, (5,5), activation='relu', padding='same')(conv8)
    conv8 = BatchNormalization()(conv8)
    up3 = UpSampling2D((2,2))(conv8) #14 x 14 x 64
    conv9 = Conv2D(32, (5,5), activation='relu', padding='same')(up3) # 14 x 14 x 32
    conv9 = BatchNormalization()(conv9)
    conv9 = Conv2D(32, (5,5), activation='relu', padding='same')(conv9)
    conv9 = BatchNormalization()(conv9)
    up4 = UpSampling2D((2,2))(conv9) # 28 x 28 x 32
    decoded = Conv2D(3, (5,5), activation='sigmoid', padding='same')(up4) # 28 x 28 x 1
    return decoded

def fc(enco):
    flat = Flatten()(enco)
    den = Dense(128, activation='relu')(flat)
    out = Dense(num_classes, activation='softmax')(den)
    return out

# # custom metric with TF
# def cohens_kappa(y_true, y_pred):
#     y_true_classes = tf.argmax(y_true, 1)
#     y_pred_classes = tf.argmax(y_pred, 1)
#     return tf.contrib.metrics.cohen_kappa(y_true_classes, y_pred_classes, 10)[1]

autoencoder = Model(input_img, decoder(encoder(input_img)))
autoencoder.compile(loss='mean_squared_error', 
                    optimizer = RMSprop())
print(autoencoder.summary())

# # Prepare callbacks for model saving.
# save_dir = '/pool001/bibek/hst/semisupervised/'\
#            +'deep_autoencoder1'
# log_dir = os.path.join(save_dir, 'log')
# model_name = '%s.{epoch:03d}.h5' % 'ae_c5_d10'

# if not os.path.isdir(save_dir):
#     os.makedirs(save_dir)
# if not os.path.isdir(log_dir):
#     os.makedirs(log_dir)
# filepath = os.path.join(save_dir, model_name)

# checkpoint = ModelCheckpoint(filepath=filepath,
#                              monitor='loss',
#                              verbose=1,
#                              save_best_only=True,
#                              mode='min')

# tensorboard = TensorBoard(log_dir=log_dir, 
#                           batch_size=batch_size)

# callbacks = [checkpoint, tensorboard]

# autoencoder_train = autoencoder.fit_generator(train_auto,
#                         steps_per_epoch = train_auto.samples // batch_size,
#                         validation_data = valid_auto,
#                         validation_steps = valid_auto.samples // batch_size,
#                         epochs = epochs,
#                         callbacks = callbacks)

# loss = autoencoder_train.history['loss']
# val_loss = autoencoder_train.history['val_loss']
# epochs_plot = range(epochs)
# plt.figure()
# plt.plot(epochs_plot, loss, 'bo', label='Training loss')
# plt.plot(epochs_plot, val_loss, 'b', label='Validation loss')
# plt.title('Training and validation loss')
# plt.legend()
# plt.show()

# autoencoder.save('deep_autoencoder.h5')

autoencoder.load_weights('/pool001/bibek/hst/semisupervised/deep_autoencoder1/ae_c5_d10.017.h5')

encode = encoder(input_img)
full_model = Model(input_img,fc(encode))

print(len(full_model.layers))
print(len(autoencoder.layers))
      
print(full_model.layers)
print(autoencoder.layers)

# full layers: 35

# for l1,l2 in zip(full_model.layers[:19],autoencoder.layers[:19]):
#     l1.set_weights(l2.get_weights())

# print(autoencoder.get_weights()[0][1])
# print(full_model.get_weights()[0][1])

# for layer in full_model.layers[:19]:
#     layer.trainable = False
    
# full_model.compile(loss=keras.losses.categorical_crossentropy, 
#                    optimizer=keras.optimizers.Adam(),
#                    metrics=['accuracy']) #, cohens_kappa])
# print(full_model.summary())

# # Prepare callbacks for model saving.
# save_dir = '/pool001/bibek/hst/semisupervised/'\
#            +'deep_fixedclf50'
# log_dir = os.path.join(save_dir, 'log')
# model_name = '%s.{epoch:03d}.h5' % 'clf_128'

# if not os.path.isdir(save_dir):
#     os.makedirs(save_dir)
# if not os.path.isdir(log_dir):
#     os.makedirs(log_dir)
# filepath = os.path.join(save_dir, model_name)

# checkpoint = ModelCheckpoint(filepath=filepath,
#                              monitor='acc',
#                              verbose=1,
#                              save_best_only=True,
#                              mode='max')

# tensorboard = TensorBoard(log_dir=log_dir, 
#                           batch_size=batch_size)

# callbacks = [checkpoint, tensorboard]

# classify_train = full_model.fit_generator(train_batches,
#                         steps_per_epoch = train_batches.samples * 0.5 // batch_size,
#                         validation_data = valid_batches,
#                         validation_steps = valid_batches.samples // batch_size,
#                         epochs = epochs,
#                         callbacks = callbacks,
#                         shuffle = False)

# full_model.save('deep_autoencoder_classification_50.h5')

for layer in full_model.layers[:19]:
    layer.trainable = True
    
full_model.compile(loss=keras.losses.categorical_crossentropy, 
                   optimizer=keras.optimizers.Adam(),
                   metrics=['accuracy']) #, cohens_kappa])

# Prepare callbacks for model saving.
save_dir = '/pool001/bibek/hst/semisupervised/'\
           +'mid_trainclf50_1'
log_dir = os.path.join(save_dir, 'log')
model_name = '%s.{epoch:03d}.h5' % 'clf_128'

if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
if not os.path.isdir(log_dir):
    os.makedirs(log_dir)
filepath = os.path.join(save_dir, model_name)

checkpoint = ModelCheckpoint(filepath=filepath,
                             monitor='acc',
                             verbose=1,
                             save_best_only=True,
                             mode='max')

tensorboard = TensorBoard(log_dir=log_dir, 
                          batch_size=batch_size)

callbacks = [checkpoint, tensorboard]

classify_train = full_model.fit_generator(train_batches,
                        steps_per_epoch = train_batches.samples * 0.5 // batch_size,
                        validation_data = valid_batches,
                        validation_steps = valid_batches.samples // batch_size,
                        epochs = epochs,
                        callbacks = callbacks,
                        shuffle = True)

full_model.save('mid_classification_complete_50_1.h5')

accuracy = classify_train.history['acc']
val_accuracy = classify_train.history['val_acc']
loss = classify_train.history['loss']
val_loss = classify_train.history['val_loss']
epochs = range(len(accuracy))
plt.plot(epochs, accuracy, 'bo', label='Training accuracy')
plt.plot(epochs, val_accuracy, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.legend()
plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()

# test_eval = full_model.evaluate(test_data, test_Y_one_hot, verbose=0)
# print('Test loss:', test_eval[0])
# print('Test accuracy:', test_eval[1])
