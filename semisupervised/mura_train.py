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
    conv4 = Conv2D(256, (5,5), activation='relu', padding='same')(conv3) #7 x 7 x 256 (small and thick)
    conv4 = BatchNormalization()(conv4)
    conv4 = Conv2D(256, (5,5), activation='relu', padding='same')(conv4)
    conv4 = BatchNormalization()(conv4)
    return conv4

def decoder(conv4):    
    #decoder
    conv5 = Conv2D(128, (5,5), activation='relu', padding='same')(conv4) #7 x 7 x 128
    conv5 = BatchNormalization()(conv5)
    conv5 = Conv2D(128, (5,5), activation='relu', padding='same')(conv5)
    conv5 = BatchNormalization()(conv5)
    conv6 = Conv2D(64, (5,5), activation='relu', padding='same')(conv5) #7 x 7 x 64
    conv6 = BatchNormalization()(conv6)
    conv6 = Conv2D(64, (5,5), activation='relu', padding='same')(conv6)
    conv6 = BatchNormalization()(conv6)
    up1 = UpSampling2D((2,2))(conv6) #14 x 14 x 64
    conv7 = Conv2D(32, (5,5), activation='relu', padding='same')(up1) # 14 x 14 x 32
    conv7 = BatchNormalization()(conv7)
    conv7 = Conv2D(32, (5,5), activation='relu', padding='same')(conv7)
    conv7 = BatchNormalization()(conv7)
    up2 = UpSampling2D((2,2))(conv7) # 28 x 28 x 32
    decoded = Conv2D(3, (5,5), activation='sigmoid', padding='same')(up2) # 28 x 28 x 1
    return decoded

def fc(enco):
    flat = Flatten()(enco)
    den = Dense(128, activation='relu')(flat)
    out = Dense(num_classes, activation='softmax')(den)
    return out

class SkMetrics(Callback):
    def on_epoch_end(self, epoch, logs={}):
        score = np.asarray(self.model.predict(self.validation_data[0]))
        predict = np.round(np.asarray(self.model.predict(self.validation_data[0])))
        targ = self.validation_data[1]
        print('validation kappa:', cohen_kappa_score(targ, predict))
        
skmetrics = SkMetrics()

def kappa(y_pred, y_true, y_pow=2, eps=1e-10, N=5, bsize=256, name='kappa'):
    """A continuous differentiable approximation of discrete kappa loss.
        Args:
            y_pred: 2D tensor or array, [batch_size, num_classes]
            y_true: 2D tensor or array,[batch_size, num_classes]
            y_pow: int,  e.g. y_pow=2
            N: typically num_classes of the model
            bsize: batch_size of the training or validation ops
            eps: a float, prevents divide by zero
            name: Optional scope/name for op_scope.
        Returns:
            A tensor with the kappa loss."""

    with tf.name_scope(name):
        y_true = tf.to_float(y_true)
        repeat_op = tf.to_float(tf.tile(tf.reshape(tf.range(0, N), [N, 1]), [1, N]))
        repeat_op_sq = tf.square((repeat_op - tf.transpose(repeat_op)))
        weights = repeat_op_sq / tf.to_float((N - 1) ** 2)
    
        pred_ = y_pred ** y_pow
        try:
            pred_norm = pred_ / (eps + tf.reshape(tf.reduce_sum(pred_, 1), [-1, 1]))
        except Exception:
            pred_norm = pred_ / (eps + tf.reshape(tf.reduce_sum(pred_, 1), [bsize, 1]))
    
        hist_rater_a = tf.reduce_sum(pred_norm, 0)
        hist_rater_b = tf.reduce_sum(y_true, 0)
    
        conf_mat = tf.matmul(tf.transpose(pred_norm), y_true)
    
        nom = tf.reduce_sum(weights * conf_mat)
        denom = tf.reduce_sum(weights * tf.matmul(
            tf.reshape(hist_rater_a, [N, 1]), tf.reshape(hist_rater_b, [1, N])) /
                              tf.to_float(bsize))
    
        return nom / (denom + eps)

# # custom metric with TF
# def cohens_kappa(y_true, y_pred):
#     tf.enable_eager_execution()
#     y_true_classes = np.argmax(y_true.numpy(), 1)
#     y_pred_classes = np.argmax(y_pred.numpy(), 1)
#     return cohen_kappa(y_true_classes, y_pred_classes, 2)[1]
# #     return tf.contrib.metrics.cohen_kappa(y_true_classes, y_pred_classes, 10)[1]

autoencoder = Model(input_img, decoder(encoder(input_img)))
autoencoder.compile(loss='mean_squared_error', 
                    optimizer = RMSprop())
print(autoencoder.summary())

# # Prepare callbacks for model saving.
# save_dir = '/pool001/bibek/hst/semisupervised/'\
#            +'autoencoder1'
# log_dir = os.path.join(save_dir, 'log')
# model_name = '%s.{epoch:03d}.h5' % 'ae_c4_d8'

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

# autoencoder.save('autoencoder.h5')

autoencoder.load_weights('/pool001/bibek/hst/semisupervised/autoencoder1/ae_c4_d8.022.h5')

encode = encoder(input_img)
full_model = Model(input_img,fc(encode))

for l1,l2 in zip(full_model.layers[:19],autoencoder.layers[0:19]):
    l1.set_weights(l2.get_weights())

print(autoencoder.get_weights()[0][1])
print(full_model.get_weights()[0][1])

for layer in full_model.layers[0:19]:
    layer.trainable = False
    
full_model.compile(loss=keras.losses.categorical_crossentropy, 
                   optimizer=keras.optimizers.Adam(),
                   metrics=['accuracy'])
print(full_model.summary())

# Prepare callbacks for model saving.
save_dir = '/pool001/bibek/hst/semisupervised/'\
           +'fixedclf1'
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

K.get_session().run(tf.local_variables_initializer())

# classify_train = full_model.fit_generator(train_batches,
#                         steps_per_epoch = train_batches.samples // batch_size,
#                         validation_data = valid_batches,
#                         validation_steps = valid_batches.samples // batch_size,
#                         epochs = epochs,
#                         callbacks = callbacks)

# full_model.save('autoencoder_classification.h5')

full_model.load_weights('/pool001/bibek/hst/semisupervised/fixedclf1/clf_128.017.h5')

for layer in full_model.layers[0:19]:
    layer.trainable = True
    
full_model.compile(loss=keras.losses.categorical_crossentropy, 
                   optimizer=keras.optimizers.Adam(),
                   metrics=['accuracy']) #, kappa])

# Prepare callbacks for model saving.
save_dir = '/pool001/bibek/hst/semisupervised/'\
           +'trainclf1'
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
                        steps_per_epoch = train_batches.samples // batch_size,
                        validation_data = valid_batches,
                        validation_steps = valid_batches.samples // batch_size,
                        epochs = epochs,
                        callbacks = callbacks)

full_model.save('classification_complete.h5')

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
