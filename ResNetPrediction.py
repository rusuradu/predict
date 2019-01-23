from fastai.vision.transform import *
from fastai.vision import *
from fastai.metrics import *
import fastai.data_block
from fastai.datasets import *
from fastai.layers import *
import torch
from timeit import default_timer as timer
from fastai.basic_train import *

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
#5% of data in the validation set is sufficient for model evaluation
tr_n, val_n = train_test_split(train_names, test_size=0.05, random_state=42)


# class pdFilesDataset(FilesDataset):
#     def __init__(self, fnames, path, transform):
#         self.segmentation_df = pd.read_csv(SEGMENTATION).set_index('ImageId')
#         super().__init__(fnames, transform, path)
#
#     def get_x(self, i):
#         img = open_image(os.path.join(self.path, self.fnames[i]))
#         if self.sz == 768:
#             return img
#         else:
#             # see here
#             return img.resize(self.sz)
#
#     def get_y(self, i):
#         if (self.path == TEST): return 0
#         masks = self.segmentation_df.loc[self.fnames[i]]['EncodedPixels']
#         if (type(masks) == float):
#             return 0  # NAN - no ship
#         else:
#             return 1
#
#     def get_c(self):
#         return 2  # number of classes

def get_data(sz,bs):
    #data augmentation

    # aug_tfms = [RandomRotate(20, tfm_y=TfmType.NO),
    #             RandomDihedral(tfm_y=TfmType.NO),
    #             RandomLighting(0.05, 0.05, tfm_y=TfmType.NO)]


    # tfms = tfms_from_model(arch, sz, crop_type=CropType.NO, tfm_y=TfmType.NO,
    #             aug_tfms=aug_tfms)

    return md


def export(self, fname:str='export.pkl'):
    "Export the state of the `Learner` in `self.path/fname`."
    args = ['opt_func', 'loss_func', 'metrics', 'true_wd', 'bn_wd', 'wd', 'train_bn', 'model_dir', 'callback_fns']
    state = {a:getattr(self,a) for a in args}
    state['cb_state'] = {cb.__class__:cb.get_state() for cb in self.callbacks}
    #layer_groups -> need to find a way
    #TO SEE: do we save model structure and weights separately?
    state['model'] = self.model
    xtra = dict(normalize=self.data.norm.keywords) if getattr(self.data, 'norm', False) else {}
    state['data'] = self.data.valid_ds.get_state(**xtra)
    state['cls'] = self.__class__
    torch.save(state, open(self.path/fname, 'wb'))

sz = 256 #image size
bs = 64  #batch size


# , bs=4, num_workers=0
tfm = [rotate(degrees=(-90, 90)), dihedral()]
md = (ImageDataBunch.from_csv(
    '..\\ShipDetection\\1percent\\',
    folder='train',
    valid_pct=0.5,
    csv_name='labels.csv',
    bs=5,
    num_workers=0,
    tmfs=tfm,
    test='test'))

img = md.test_ds[0][0]

empty_data = ImageDataBunch.load_empty('..\\ShipDetection\\1percent\\')
#learn = create_cnn(empty_data, models.resnet34)
#learn = learn.load('Resnet34_lable_256_1_Raducu')

learn = create_cnn(md, models.resnet34, metrics=[accuracy], ps=0.2).load('Resnet34_lable_256_1_Raducu')
for i in range(len(md.test_ds)):
    img = md.test_ds[i][0]
    print(learn.predict(img))

i = 20


#learn = ConvLearner.pretrained(arch, md, ps=0.5) #dropout 50%
#learn.opt_fn = optim.Adam



#learn.lr_find()
#learn.sched.plot()
# if __name__ == '__main__':
#     start = timer()
#     learn.fit(1)
#     print("time to fit % d" % (timer() - start))
#     learn.save('Resnet34_lable_256_1_Raducu')

#learn.unfreeze()
#lr = np.array([1e-4, 5e-4, 2e-3])

#learn.fit(lr, 1) #, cycle_len=2, use_clr=(20,8))

#log_preds,y = learn.predict_with_targs(is_test=True)

# log_preds,y = learn.predict(is_test=True)
#
# probs = np.exp(log_preds)[:, 1]
# pred = (probs > 0.5).astype(int)
#
# df = pd.DataFrame({'id': test_names, 'p_ship': probs})
# df.to_csv('ship_detection123.csv', header=True, index=False)