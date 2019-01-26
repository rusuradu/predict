from fastai.vision.transform import *
from fastai.vision import *
from fastai.metrics import *
import fastai.data_block
from fastai.datasets import *
from fastai.layers import *
import torch
from timeit import default_timer as timer

# mnist = untar_data(URLs.MNIST_TINY)
# tfms = get_transforms(do_flip=False)

import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split

PATH = './'
TRAIN = '../ShipDetection/TrainFull/'
TEST = '../ShipDetection/TestFull/'
SEGMENTATION = '../ShiptDetection/train_ship_segmentations.csv'
exclude_list = ['6384c3e78.jpg', '13703f040.jpg', '14715c06d.jpg',  '33e0ff2d5.jpg',
                '4d4e09f2a.jpg', '877691df8.jpg', '8b909bb20.jpg', 'a8d99130e.jpg',
                'ad55c3143.jpg', 'c8260c541.jpg', 'd6c7f17c7.jpg', 'dc3e7c901.jpg',
                'e44dffe88.jpg', 'ef87bad36.jpg', 'f083256d8.jpg'] #corrupted image

nw = 4   #number of workers for data loader
arch = models.resnet34 #specify target architecture

train_names = [f for f in os.listdir(TRAIN)]
test_names = [f for f in os.listdir(TEST)]
for el in exclude_list:
    if(el in train_names): train_names.remove(el)
    if(el in test_names): test_names.remove(el)
tr_n, val_n = train_test_split(train_names, test_size=0.05, random_state=42)


# , bs=4, num_workers=0
tfm = [rotate(degrees=(-90, 90)), dihedral()]
md = (ImageDataBunch.from_csv(
    '..\\_10kRun\\Propose\\',
    folder='train',
    valid_pct=0.2,
    csv_name='labels.csv',
    bs=10,
    num_workers=1,
    tmfs=tfm))

learn = create_cnn(md, models.resnet34, metrics=[accuracy], ps=0.2)


if __name__ == '__main__':
    start = timer()
    learn.fit(1)
    print("time to fit % d" % (timer() - start))
    learn.save('Resnet34_Raducu_10k_small_first')

