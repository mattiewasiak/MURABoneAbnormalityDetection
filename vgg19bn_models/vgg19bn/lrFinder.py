import shutil
import gc
import torch

import fastai
from fastai.vision import *
from fastai.widgets import *
from fastai.callbacks import *

print("vgg19 bn epoch:0-4")

Config.DEFAULT_CONFIG = {
        'data_path': './../data/MURA-v1.1/',
        'model_path': './models/'
    }

Config.create('/tmp/myconfig.yml')
Config.DEFAULT_CONFIG_PATH = '/tmp/myconfig.yml'

path = Config.data_path()

 
size = 320
bs = 32

np.random.seed(24)
data = ImageDataBunch.from_df(path='../data/MURA-v1.1/data2/', df=pd.read_csv('all_elbows.csv'), ds_tfms=get_transforms(do_flip=True, max_rotate=30, max_warp=0.0, p_lighting=0, xtra_tfms=[cutout(n_holes=(0,3), length=(5,35),p=1)]), size=size, bs=bs).normalize(imagenet_stats)


kappa = KappaScore()
kappa.weights = "quadratic"

learner = cnn_learner(data, models.vgg19_bn, metrics=[error_rate, accuracy, kappa], wd=0.1, model_dir="./models/").to_fp32()
"""
learner.load('vgg19bn-1')
learner.freeze()
"""
learner.unfreeze()

learner.lr_find()      
plt = learner.recorder.plot(return_fig = True)
plt.savefig('lrGraph-elbows.png')

"""
learner.load('vgg19bn-1')
learner.freeze();
learner.unfreeze()
# learner.freeze_to(5);
learner.lr_find()      
plt = learner.recorder.plot(return_fig = True)
plt.savefig('lrGraph2.png')

learner.load('vgg19bn-1')
learner.freeze();
learner.unfreeze()
# learner.freeze_to(10);
learner.lr_find()      
plt = learner.recorder.plot(return_fig = True)
plt.savefig('lrGraph3.png')

learner.load('vgg19bn-1')
learner.freeze();
learner.unfreeze()
# learner.freeze_to(20);
learner.lr_find()      
plt = learner.recorder.plot(return_fig = True)
plt.savefig('lrGraph4.png')
"""
