import argparse
from utils import helper_functions
import subprocess
from pathlib import Path
from skimage import io
import numpy as np
import os
import main as train_main
from distutils.dir_util import copy_tree
from shutil import copy


parser = argparse.ArgumentParser(description='Glacier Front Segmentation')

parser.add_argument('--epochs', default=250, type=int, help='number of training epochs (integer value > 0)')
parser.add_argument('--patience', default=30, type=int, help='how long to wait for improvements before Early_stopping')
parser.add_argument('--batch_size', default=-1, type=int, help='batch size (integer value), if -1 set batch size according to available gpu memery')
parser.add_argument('--patch_size', default=256, type=int, help='size of the image patches (patch_size x patch_size')

parser.add_argument('--early_stopping', default=1, type=int,
                    help='If 1, classifier is using early stopping based on val loss with patience 20 (0/1)')
parser.add_argument('--image_patches', default=0, type=int, help='Training data is already split into image patches')

parser.add_argument('--out_path', type=str, help='Output path for results')
parser.add_argument('--data_path', type=str, help='Path containing training and val data')
parser.add_argument('--model', default='unet_Enze19_2', type=str, help='Training Model to use')
parser.add_argument('--drop_rate', default=0.5, type=float, help='Dropout for Bayesian Unet')
#parser.add_argument('--cyclic', default='None', type=str, help='Which cyclic learning policy to use', choices=['None', 'triangular', 'triangular2', 'exp_range' ])
#parser.add_argument('--cyclic-parms', action=helper_functions.StoreDictKeyPair, metavar="KEY1=VAL1,KEY2=VAL2...",
#                    help='dictionary with parameters for cyclic learning')
parser.add_argument('--learning_rate', type=float, default=1e-4, help='Initial learning rate')
parser.add_argument('--threshold', type=float, default=1e-4, help='Uncertainty threshold for training selection')
args = parser.parse_args()

parms = " "

prediction_path = Path(args.out_path, 'prediction')
if not prediction_path.exists():
    prediction_path.mkdir(parents=True)
for arg, value in args.__dict__.items():
    if arg == 'threshold':
        continue
    if arg == 'out_path':
        continue
    parms += "--" + arg + " " + str(value) + " "


subprocess.call("python3 main.py " + parms + " --evaluate 0 --out_path " + str(prediction_path), shell=True)



out_path = Path(args.out_path)
model_file = next(prediction_path.glob('model_*.h5'))
copy(model_file, out_path)

if not Path(out_path, 'train', 'images').exists():
    Path(out_path, 'train', 'images').mkdir(parents=True)
if not Path(out_path, 'train', 'masks').exists():
    Path(out_path, 'train', 'masks').mkdir(parents=True)

if not Path(out_path, 'test', 'images').exists():
    Path(out_path, 'test', 'images').mkdir(parents=True)

copy_tree(str(Path(args.data_path, 'val')), str(Path(out_path, 'val')))

for f in Path(args.data_path, 'test', 'images').glob('*.png'):
    uncertainty_img = io.imread(Path(prediction_path, f.stem + '_uncertainty.png'), as_gray=True)
    uncertainty = uncertainty_img / 65535
    if np.mean(uncertainty) < args.threshold:
        copy(f, Path(out_path, 'train', 'images', f.name))
        copy(Path(prediction_path, f.stem + '_pred.png'), Path(out_path, 'train', 'images', f.stem + '_zones.png'))
    else:
        copy(f, Path(out_path, 'test', 'images', f.name))






