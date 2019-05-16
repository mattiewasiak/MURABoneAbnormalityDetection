import gc
import torch
from fastai.vision import *
from fastai.widgets import *
from fastai.callbacks import * 
import shutil
import fastai
print(f'fastai: {fastai.__version__}')
print(f'cuda: {torch.cuda.is_available()}')

Config.DEFAULT_CONFIG = {
        'data_path': './../data/MURA-v1.1/',
        'model_path': './models/'
    }

Config.create('/tmp/myconfig.yml')
Config.DEFAULT_CONFIG_PATH = '/tmp/myconfig.yml'

path = Config.data_path()

size = 320
bs = 32

np.random.seed(42)
data = ImageDataBunch.from_folder('../data/MURA-v1.1/data2/', ds_tfms=get_transforms(do_flip=True, max_rotate=30, max_warp=0.0, p_lighting=0, xtra_tfms=[cutout(n_holes=(0,3), length=(5,35),p=1)]), size=size, bs=bs).normalize(imagenet_stats)


kappa = KappaScore()
kappa.weights = "quadratic"

learner = cnn_learner(data, models.vgg19_bn, metrics=[error_rate, accuracy, kappa], wd=0.1, model_dir="./models/").to_fp32()
# #learner.unfreeze()
# learner.fit_one_cycle(5,max_lr=1e-3, callbacks=[ShowGraph(learner) ,SaveModelCallback(learner)])

learner.load('vgg19bn-onerun-35whole')

val_preds,val_targets = learner.get_preds() 
from os import listdir
from os.path import isfile, join
files = [f for f in listdir('../data/MURA-v1.1/data2/valid/0/') if isfile(join('../data/MURA-v1.1/data2/valid/0/', f))]
files += [f for f in listdir('../data/MURA-v1.1/data2/valid/1/') if isfile(join('../data/MURA-v1.1/data2/valid/1/', f))]
files[0]

print(len(val_preds))
print(len(files))

studies = []
studies_labels = []
for file in files:
    study = file[:-11]
    if study not in studies:
        studies.append(study)
        if 'negative' in study:
            studies_labels.append(0)
        else:
            studies_labels.append(1)
            
study_actual = dict()
for study in studies:
    if 'negative' in study:
        study_actual[study] = 0
    else:
        study_actual[study] = 1

study_dict = dict()
for study in studies:
    if study not in study_dict:
        study_dict[study] = []
        
file_count = 0
for file in files:
    study = file[:-11]
    study_dict[study].append(float(val_preds[file_count][1]))
    file_count += 1
    
study_preds = dict()
for study in study_dict:
    combined_pr = sum(study_dict[study])/len(study_dict[study])
    if combined_pr >= 0.5:
        study_preds[study] = 1
    else:
        study_preds[study] = 0
        
print(len(study_actual))
print(len(study_preds))

please = 0
for i in study_actual:
    if study_actual[i] == study_preds[i]:
        please += 1
print("By study accuracy: ", please/1200)


#kappa
actual_list = []
pred_list = []
for i in study_actual:
    actual_list.append(study_actual[i])
    pred_list.append(study_preds[i])

p_o = 0
p_e = 0

agreed = 0
for i in range(len(actual_list)):
    if actual_list[i] == pred_list[i]:
        agreed += 1
p_o = agreed/len(actual_list)
print("p_o: ", p_o)

actual_yes = sum(actual_list)/len(actual_list)
pred_yes = sum(pred_list)/len(pred_list)

actual_no = 1-sum(actual_list)/len(actual_list)
pred_no = 1-sum(pred_list)/len(pred_list)

p_e = actual_yes*pred_yes + actual_no*pred_no
print("p_e: ", p_e)

kappa = (p_o-p_e)/(1-p_e)
print("KAPPA BY STUDY: ", kappa)
