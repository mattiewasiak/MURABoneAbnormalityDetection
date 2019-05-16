# Bone Abnormality Detection in the MURA Dataset


data_augmentation: folder contains code for applying various data augmentations and determining which have the best effects on accuracy and Kappa score on a DenseNet169 architecture


model_selection: folder contains code for determining the best baseline architecture for bone abnormality detection (DenseNet, ResNet, VGG)

semisupervised: folder contains code for creating an autoencoder classifier (both a convolutional and a variational)

vgg19bn_models: folder contains code for creating a single VGG19 model for bone abnormality detection as well as seven individual VGG19 models for individual bone abnormality detection and a VGG19 model for classifying bones
