import argparse
from utils import helper_functions
import subprocess
from pathlib import Path
from skimage import io
import numpy as np
import os
from distutils.dir_util import copy_tree
from shutil import copy
from layers.BayesDropout import  BayesDropout
from predict import predict_bayes
from loss_functions import *
from keras.losses import binary_crossentropy
from keras.models import load_model
import json


parser = argparse.ArgumentParser(description='Glacier Front Segmentation')

parser.add_argument('--batch_size', default=-1, type=int, help='batch size (integer value), if -1 set batch size according to available gpu memery')

parser.add_argument('--out_path', type=str, help='Output path for results')
parser.add_argument('--data_path', type=str, help='Path containing training and val data')
parser.add_argument('--model', default='unet_Enze19_2', type=str, help='Training Model to use')
parser.add_argument('--threshold', type=float, default=1e-4, help='Uncertainty threshold for training selection')
args = parser.parse_args()

data_path = Path(args.data_path)
options = json.load(open(Path(data_path, 'options.json'), 'r'))

if 'loss_parms' in options:
    loss_function = get_loss_function(options['loss'], options['loss_parms'])
else:
    loss_function = get_loss_function(options['loss'])

model_file = next(data_path.glob('model_*.h5'))
model = load_model(str(model_file.absolute()), custom_objects={'loss': loss_function, 'BayesDropout':BayesDropout})

predict_bayes(model, Path(data_path, 'unlabeled'), data_path, batch_size=args.batch_size, patch_size = options['patch_size'], cutoff = options['cutoff'])






out_path = Path(args.out_path)

if not Path(out_path, 'train', 'images').exists():
    Path(out_path, 'train', 'images').mkdir(parents=True)
if not Path(out_path, 'train', 'masks').exists():
    Path(out_path, 'train', 'masks').mkdir(parents=True)

if not Path(out_path, 'unlabeled').exists():
    Path(out_path, 'unlabeled').mkdir(parents=True)

copy(Path(data_path, 'options.json'), Path(out_path, 'options.json'))
min_uncert = np.ones(4)
min_imgs = 4 * [""]
for f in Path(args.data_path, 'unlabeled').glob('*.png'):
    uncertainty_img = io.imread(Path(data_path, f.stem + '_uncertainty.png'), as_gray=True)
    uncertainty = uncertainty_img / 65535
    uncertainty_mean = np.mean(uncertainty)
    if uncertainty_mean < min_uncert.max():
        min_uncert[min_uncert.argmax()] = uncertainty_mean
        min_imgs[min_uncert.argmax()] = f.stem

for f in Path(args.data_path, 'unlabeled').glob('*.png'):
    if f.stem in min_imgs:
        copy(f, Path(out_path, 'train', 'images', f.name))
        copy(Path(data_path, f.stem + '_pred.png'), Path(out_path, 'train', 'masks', f.stem + '_zones.png'))
    else:
        copy(f, Path(out_path, 'unlabeled', f.name))



    #if np.mean(uncertainty) < args.threshold:
    #    copy(f, Path(out_path, 'train', 'images', f.name))
    #    copy(Path(data_path, f.stem + '_pred.png'), Path(out_path, 'train', 'masks', f.stem + '_zones.png'))
    #else:
    #    copy(f, Path(out_path, 'unlabeled', f.name))






