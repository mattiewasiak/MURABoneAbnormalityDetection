import shutil
import gc
import torch

import fastai
from fastai.vision import *
from fastai.widgets import *
from fastai.callbacks import *

print("vgg19 bn 2")

Config.DEFAULT_CONFIG = {
        'data_path': './../data/MURA-v1.1/',
        'model_path': './models/'
    }

Config.create('/tmp/myconfig.yml')
Config.DEFAULT_CONFIG_PATH = '/tmp/myconfig.yml'

path = Config.data_path()

fnames_train = get_image_files('../data/MURA-v1.1/train/XR_WRIST', recurse=True)
print(len(fnames_train))

fnames_valid = get_image_files('../data/MURA-v1.1/valid/XR_WRIST', recurse=True)
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
"""
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
"""


"""

import os

directory_file = os.listdir("../data/MURA-v1.1/data2/train/0/")
print(len(directory_file))
elbows = []
count = 0
for i in directory_file:
    
    if count % 1000 == 0:
        print(count)
    count += 1
    if "ELBOW" in i:
        elbows.append("train/0/" + i)
        

directory_file = os.listdir("../data/MURA-v1.1/data2/train/1/")
print(len(directory_file))
elbows2 = []
count = 0
for i in directory_file:
    
    if count % 1000 == 0:
        print(count)
    count += 1
    if "ELBOW" in i:
        elbows2.append("train/1/" + i)
        
directory_file = os.listdir("../data/MURA-v1.1/data2/valid/0/")
print(len(directory_file))
val_elbows = []
count = 0
for i in directory_file:
    
    if count % 1000 == 0:
        print(count)
    count += 1
    if "ELBOW" in i:
        val_elbows.append("valid/0/" + i)
        

directory_file = os.listdir("../data/MURA-v1.1/data2/valid/1/")
print(len(directory_file))
val_elbows2 = []
count = 0
for i in directory_file:
    
    if count % 1000 == 0:
        print(count)
    count += 1
    if "ELBOW" in i:
        val_elbows2.append("valid/1/" + i)
        
        
values = [0]*len(elbows) + [1]*len(elbows2) + [0]*len(val_elbows) + [1]*len(val_elbows2)
valid = [False]*(len(elbows)+len(elbows2)) + [True]*(len(val_elbows) + len(val_elbows2))

import pandas as pd

elbows_df = pd.DataFrame({"name": elbows+elbows2+val_elbows+val_elbows2, "value": values, "is_valid": valid})

elbows_df.to_csv("all_elbows.csv", index = False)

"""

size = 320
bs = 32

np.random.seed(24)


def create_from_ll(lls:LabelLists, bs:int=64, val_bs:int=None, ds_tfms:Optional[TfmList]=None,
                num_workers:int=defaults.cpus, dl_tfms:Optional[Collection[Callable]]=None, device:torch.device=None,
                test:Optional[PathOrStr]=None, collate_fn:Callable=data_collate, size:int=None, no_check:bool=False,
                resize_method:ResizeMethod=None, mult:int=None, padding_mode:str='reflection',
                mode:str='bilinear', tfm_y:bool=False)->'ImageDataBunch':
    "Create an `ImageDataBunch` from `LabelLists` `lls` with potential `ds_tfms`."
    lls = lls.transform(tfms=get_transforms(do_flip=True, max_rotate=30, max_warp=0.0, p_lighting=0, xtra_tfms=[]), size=size, resize_method=resize_method, mult=mult, padding_mode=padding_mode,
                        mode=mode, tfm_y=tfm_y)
    
    return lls.databunch(bs=bs, val_bs=bs, dl_tfms=dl_tfms, num_workers=num_workers, collate_fn=collate_fn,
                         device=device, no_check=True)

src = (ImageList.from_df(df=pd.read_csv('all_bones.csv'), path='../data/MURA-v1.1/data2/', folder=None, suffix='', cols=0)
                .split_from_df("is_valid")
                .label_from_df(label_delim=None, cols=1))
data = create_from_ll(src, size = size, bs = bs).normalize(imagenet_stats)
"""
df=pd.read_csv('all_elbows.csv')
print(list(df))
df=df[['name', 'value']]
data = ImageDataBunch.from_df(path='../data/MURA-v1.1/data2/', df=df, ds_tfms=get_transforms(do_flip=True, max_rotate=30, max_warp=0.0, p_lighting=0, xtra_tfms=[]), size=size, bs=bs).normalize(imagenet_stats)

"""

    

print('CLASSIFYING BONES')
kappa = KappaScore()
kappa.weights = "quadratic"

learner = cnn_learner(data, models.vgg19_bn, metrics=[error_rate, accuracy, kappa], wd=0.1, model_dir="./models/").to_fp32()
"""
learner.unfreeze()
learner.fit_one_cycle(2, max_lr=1e-4, callbacks=[ShowGraph(learner) ,SaveModelCallback(learner)])

learner.save('vgg19bn-allfullval')
"""

print('FINISHED CLASSIFYING')


learner.load('vgg19bn-allfullval')

bone_groups = {0:[], 1:[], 2:[], 3:[], 4:[], 5:[], 6:[]}
data_list = ImageList.from_df(df=pd.read_csv('all_bones.csv'), path='../data/MURA-v1.1/data2/', folder=None, suffix='', cols=0)
df = pd.read_csv('all_bones.csv')
print('read in data frame')
count = 0
for img in data_list:
    curr = df.loc[count]
    prediction = learner.predict(img)
    label = int(prediction[1])
    bone_groups[label].append((curr['name'], curr['value'], curr['is_valid']))
    
    count += 1

    
bone_map = {"ELBOW":0, "FINGER": 1, "FOREARM":2, "HAND":3, "HUMERUS":4, "SHOULDER":5, "WRIST":6}


def run_model_on_predictions(bone_name, bones):
    print(bone_name)
    name = [i[0] for i in bones]
    value = ['positive' in i[0] for i in bones]
    valid = [i[2] for i in bones]
    
    df = pd.DataFrame({"name": name, "value": value, "is_valid": valid})
    df.to_csv("classified_" + bone_name + ".csv", index = False)
    
    src = (ImageList.from_df(df=pd.read_csv("classified_" + bone_name + ".csv"), path='../data/MURA-v1.1/data2/', folder=None, suffix='', cols=0)
                .split_from_df("is_valid")
                .label_from_df(label_delim=None, cols=1))
    data = create_from_ll(src, size = size, bs = bs).normalize(imagenet_stats)
    
    learner = cnn_learner(data, models.vgg19_bn, metrics=[error_rate, accuracy, kappa], wd=0.1, model_dir="./models/").to_fp32()
    learner.unfreeze()
    learner.fit_one_cycle(25, max_lr=1e-4, callbacks=[ShowGraph(learner) ,SaveModelCallback(learner)])

    learner.save('vgg19bn-predicted-' + bone_name)
    
    print()
    print()
    print()
    
    
    

for i in list(bone_map):
    run_model_on_predictions(i, bone_groups[bone_map[i]])















