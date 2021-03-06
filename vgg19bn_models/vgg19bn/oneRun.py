import shutil
import gc
import torch

import fastai
from fastai.vision import *
from fastai.widgets import *
from fastai.callbacks import *

print("vgg19 bn 30 epochs borders")

Config.DEFAULT_CONFIG = {
        'data_path': './../data/MURA-v1.1/',
        'model_path': './models/'
    }

Config.create('/tmp/myconfig.yml')
Config.DEFAULT_CONFIG_PATH = '/tmp/myconfig.yml'

path = Config.data_path()

fnames_train = get_image_files('../data/MURA-v1.1/train/', recurse=True)
print(len(fnames_train))

fnames_valid = get_image_files('../data/MURA-v1.1/valid/', recurse=True)
print(len(fnames_valid))

pat_label = re.compile(r'/XR_([^/]+)/[^/]+/[^/]+/[^/]+.png$')
pat_patient = re.compile(r'/[^/]+/patient([^/]+)/[^/]+/[^/]+.png$')
pat_study = re.compile(r'/([^/]+)_[^/]+/[^/]+.png$')

mura = ['elbow', 'finger', 'forearm', 'hand', 'humerus', 'shoulder', 'wrist']

study_train_dict = dict()
study_valid_dict = dict()

for m in mura:
    study_train_dict[m] = list()
    study_valid_dict[m] = list()
    
for src in fnames_train:
    # get image label
    label = pat_label.search(str(src))
    label = label.group(1)
    # get patient number
    patient = pat_patient.search(str(src))
    patient = patient.group(1)
    # get study name
    study = pat_study.search(str(src))
    study = study.group(1)
    # add to label list
    s = 'patient' + patient + '_' + study
    study_train_dict[label.lower()].append(s)

for src in fnames_valid:
    # get image label
    label = pat_label.search(str(src))
    label = label.group(1)
    # get patient number
    patient = pat_patient.search(str(src))
    patient = patient.group(1)
    # get study name
    study = pat_study.search(str(src))
    study = study.group(1)
    # add to label list
    s = 'patient' + patient + '_' + study
    study_valid_dict[label.lower()].append(s)
    
num_train_studies = 0
num_valid_studies = 0

for m in mura:
    # train
    myset = set(study_train_dict[m])
    num_train_studies += len(myset)
    # valid
    myset = set(study_valid_dict[m])
    num_valid_studies += len(myset)
    
def _corners(x):
    h,w = x.shape[1:]
    x[:, 0:40, :] = 0
    x[:, :, 0:40] = 0
    x[:, h-40:, :] = 0
    x[:, :, w-40:] = 0
    return x
    
corners = TfmPixel(_corners)
    
size = (320,320)
bs = 40

np.random.seed(42)
# data = ImageDataBunch.from_folder('../data/MURA-v1.1/data2/', ds_tfms=get_transforms(do_flip=True, max_rotate=30, max_warp=0.0, p_lighting=0, xtra_tfms=[cutout(n_holes=(0,3), length=(5,35),p=1)]), size=size, bs=bs).normalize(imagenet_stats)

data = ImageDataBunch.from_folder('../data/MURA-v1.1/data2/', ds_tfms=get_transforms(do_flip=True,max_warp=0.0, p_lighting=0, max_rotate=30, xtra_tfms=[corners(p=1)]), size=size, bs=bs).normalize(imagenet_stats)

kappa = KappaScore()
kappa.weights = "quadratic"

learner = cnn_learner(data, models.vgg19_bn, metrics=[error_rate, accuracy, kappa], wd=0.1, model_dir="./models/").to_fp32()
learner.unfreeze()
learner.fit_one_cycle(30, callbacks=[ShowGraph(learner) ,SaveModelCallback(learner)])

# learner.fit_one_cycle(10, callbacks=[ShowGraph(learner) ,SaveModelCallback(learner)])
# learner.fit_one_cycle(5, callbacks=[ShowGraph(learner) ,SaveModelCallback(learner)])
# learner.fit_one_cycle(5, callbacks=[ShowGraph(learner) ,SaveModelCallback(learner)])
# learner.fit_one_cycle(5, callbacks=[ShowGraph(learner) ,SaveModelCallback(learner)])
# learner.fit_one_cycle(5, callbacks=[ShowGraph(learner) ,SaveModelCallback(learner)])
# learner.fit_one_cycle(5, callbacks=[ShowGraph(learner) ,SaveModelCallback(learner)])

learner.save('vgg19bn-onerun-30whole-border')
