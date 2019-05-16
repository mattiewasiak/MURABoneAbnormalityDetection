from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

#!/usr/bin/env python
# coding: utf-8

import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0" #model will be trained on GPU 0

import keras
import tensorflow as tf
from sklearn.metrics import cohen_kappa_score
from matplotlib import pyplot as plt
import numpy as np
import gzip
from keras.models import Model
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

batch_size = 64
epochs = 50
inChannel = 3
x, y = 128, 128
input_img = Input(shape = (x, y, inChannel))
num_classes = 2

train_datagen = ImageDataGenerator(rescale=1./255)

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

valid_datagen = ImageDataGenerator(rescale=1./255)
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

'''Example of VAE on MNIST dataset using MLP
The VAE has a modular design. The encoder, decoder and VAE
are 3 models that share weights. After training the VAE model,
the encoder can be used to generate latent vectors.
The decoder can be used to generate MNIST digits by sampling the
latent vector from a Gaussian distribution with mean = 0 and std = 1.
# Reference
[1] Kingma, Diederik P., and Max Welling.
"Auto-Encoding Variational Bayes."
https://arxiv.org/abs/1312.6114
'''
from keras.layers import Lambda, Input, Dense
from keras.models import Model
from keras.datasets import mnist
from keras.losses import mse, binary_crossentropy
from keras.utils import plot_model
from keras import backend as K
from keras import objectives

import numpy as np
import matplotlib.pyplot as plt
import argparse
import os

# reparameterization trick
# instead of sampling from Q(z|X), sample epsilon = N(0,I)
# z = z_mean + sqrt(var) * epsilon
def sampling(args):
    """Reparameterization trick by sampling from an isotropic unit Gaussian.
    # Arguments
        args (tensor): mean and log of variance of Q(z|X)
    # Returns
        z (tensor): sampled latent vector
    """

    z_mean, z_log_var = args
    batch = K.shape(z_mean)[0]
    dim = K.int_shape(z_mean)[1]
    # by default, random_normal has mean = 0 and std = 1.0
    epsilon = K.random_normal(shape=(batch, dim))
    return z_mean + K.exp(0.5 * z_log_var) * epsilon

# network parameters
original_dim = x*y*inChannel
input_shape = (x, y, inChannel, )
intermediate_dim = 512
latent_dim = 10

# VAE model = encoder + decoder
# build encoder model
inputs = Input(shape=input_shape, name='encoder_input')
re_inputs = Reshape((-1,))(inputs)
x_hid = Dense(intermediate_dim, activation='relu')(re_inputs)
z_mean = Dense(latent_dim, name='z_mean')(x_hid)
z_log_var = Dense(latent_dim, name='z_log_var', kernel_initializer='zeros', bias_initializer='zeros')(x_hid)

# use reparameterization trick to push the sampling out as input
# note that "output_shape" isn't necessary with the TensorFlow backend
z = Lambda(sampling, output_shape=(latent_dim,), name='z')([z_mean, z_log_var])

# instantiate encoder model
encoder = Model(inputs, [z_mean, z_log_var, z], name='encoder')
encoder.summary()

# build decoder model
latent_inputs = Input(shape=(latent_dim,), name='z_sampling')
x_hid = Dense(intermediate_dim, activation='relu')(latent_inputs)
outputs = Dense(original_dim, activation='sigmoid')(x_hid)
outputs = Reshape((x, y, inChannel))(outputs)

# instantiate decoder model
decoder = Model(latent_inputs, outputs, name='decoder')
decoder.summary()

# instantiate VAE model
outputs = decoder(encoder(inputs)[2])
vae = Model(inputs, outputs, name='vae_mlp')

def vae_loss(x, x_decoded_mean):
    x = tf.reshape(x, (-1,))
    x_decoded_mean = tf.reshape(x_decoded_mean, (-1,))
    xent_loss = original_dim * objectives.mse(x, x_decoded_mean)
    kl_loss = -0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
    return xent_loss #xent_loss #+ kl_loss

vae.compile(optimizer='adam', loss=vae_loss)
vae.summary()

# # Prepare callbacks for model saving.
# save_dir = '/pool001/bibek/hst/semisupervised/'\
#            +'vae100'
# log_dir = os.path.join(save_dir, 'log')
# model_name = '%s.{epoch:03d}.h5' % 'vae'

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

# # train the autoencoder
# vae_train = vae.fit_generator(train_auto,
#                         steps_per_epoch = train_auto.samples // batch_size,
#                         validation_data = valid_auto,
#                         validation_steps = valid_auto.samples // batch_size,
#                         epochs = epochs,
#                         callbacks = callbacks)

# vae.save_weights('vae_mlp100.h5')

# vae.load_weights('/pool001/bibek/hst/semisupervised/vae1/vae.015.h5')

def fc(enco):
    den = Dense(128, activation='relu')(enco)
    out = Dense(num_classes, activation='softmax')(den)
    return out

# build classifier model
latent_inputs = Input(shape=(latent_dim,), name='z_sampling')
outputs = fc(latent_inputs)

# instantiate classifier model
clf = Model(latent_inputs, outputs, name='classifier')
clf.summary()

# instantiate VAE model
outputs = clf(encoder(inputs)[2])
full_model = Model(inputs, outputs, name='vae_clf')

# for l1,l2 in zip(full_model.layers[:2],vae.layers[:2]):
#     l1.set_weights(l2.get_weights())

# print(vae.get_weights()[0][1])
# print(full_model.get_weights()[0][1])

# for layer in full_model.layers[0:2]:
#     layer.trainable = False
    
# full_model.compile(loss=keras.losses.categorical_crossentropy, 
#                    optimizer=keras.optimizers.Adam(),
#                    metrics=['accuracy'])
# print(full_model.summary())

# # Prepare callbacks for model saving.
# save_dir = '/pool001/bibek/hst/semisupervised/'\
#            +'fixedvaeclf100'
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

# K.get_session().run(tf.local_variables_initializer())

# classify_train = full_model.fit_generator(train_batches,
#                         steps_per_epoch = train_batches.samples // batch_size,
#                         validation_data = valid_batches,
#                         validation_steps = valid_batches.samples // batch_size,
#                         epochs = epochs,
#                         callbacks = callbacks)

# full_model.save('vae_classification100.h5')

# for layer in full_model.layers[0:2]:
#     layer.trainable = True
    
# full_model.compile(loss=keras.losses.categorical_crossentropy, 
#                    optimizer=keras.optimizers.Adam(),
#                    metrics=['accuracy']) #, kappa])

# # Prepare callbacks for model saving.
# save_dir = '/pool001/bibek/hst/semisupervised/'\
#            +'trainvaeclf100'
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
#                         steps_per_epoch = train_batches.samples // batch_size,
#                         validation_data = valid_batches,
#                         validation_steps = valid_batches.samples // batch_size,
#                         epochs = epochs,
#                         callbacks = callbacks)

# full_model.save('vae_classification_complete100.h5')

batch_size = 1
# epochs = 25
# inChannel = 3
# x, y = 128, 128
# input_img = Input(shape = (x, y, inChannel))
# num_classes = 2

test_datagen = ImageDataGenerator()
test_generator = test_datagen.flow_from_directory("./../data/MURA-v1.1/data2/valid/",
                                                  target_size=(x,y),
                                                  interpolation='bicubic',
                                                  class_mode='categorical',
                                                  shuffle=False,
                                                  batch_size=batch_size)

full_model.load_weights('/pool001/bibek/hst/semisupervised/fixedvaeclf100/clf_128.047.h5')

filenames = test_generator.filenames
nb_samples = len(filenames)

predict = full_model.predict_generator(test_generator, steps=nb_samples)
y_pred = np.argmax(predict, axis=1)
labels = test_generator.classes

print('kappa:', cohen_kappa_score(labels, y_pred))