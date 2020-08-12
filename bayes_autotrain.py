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
from predict import predict_patches_only, predict_bayes
from loss_functions import *
from keras.losses import binary_crossentropy
from keras.models import load_model
import json
import pickle
from utils.evaluate import evaluate
from train import train
from keras.callbacks import ModelCheckpoint, EarlyStopping, CSVLogger


parser = argparse.ArgumentParser(description='Glacier Front Segmentation')

parser.add_argument('--batch_size', default=-1, type=int, help='batch size (integer value), if -1 set batch size according to available gpu memery')

parser.add_argument('--out_path', type=str, help='Output path for results')
parser.add_argument('--val_path', type=str, help='Path val data')
parser.add_argument('--img_path', type=str, help='Path containing unlabeled imgs')
parser.add_argument('--max_iterations', type=int, help='Maximum number of automatic training iterations')
parser.add_argument('--model_path', type=str, help='Path containing Pretrained Model')
parser.add_argument('--threshold', type=float, default=1e-4, help='Uncertainty threshold for training selection')
parser.add_argument('--var_threshold', type=float, default=1e-3, help='Minimum variance for training selection')
parser.add_argument('--test_path', type=str, help='Path containing test data')
args = parser.parse_args()

model_path = Path(args.model_path)
options = json.load(open(Path(model_path, 'options.json'), 'r'))

if 'loss_parms' in options:
    loss_function = get_loss_function(options['loss'], options['loss_parms'])
else:
    loss_function = get_loss_function(options['loss'])

model_file = next(model_path.glob('model_*.h5'))
model = load_model(str(model_file.absolute()), custom_objects={'loss': loss_function, 'BayesDropout':BayesDropout})
history = pickle.load(open(Path(model_path, 'history_' + model.name + '.pkl'), 'rb'))

batch_size=args.batch_size
patch_size = patch_size = options['patch_size']
cutoff = options['cutoff']

img_path = Path(args.img_path)
iter_path = Path(args.out_path, str(0))
if not iter_path.exists():
    iter_path.mkdir(parents=True)
for iter in range(args.max_iterations):
    train_path = Path(iter_path, 'train')
    if not train_path.exists():
        train_path.mkdir()



    predict_patches_only(model, img_path, train_path, batch_size=batch_size, patch_size=patch_size, cutoff=cutoff)
    unlabeled_path = Path(iter_path, 'unlabeled')
    if not unlabeled_path.exists():
        unlabeled_path.mkdir()
    for f in Path(train_path, 'images').glob('*.png'):
        img = io.imread(f, as_gray=True)
        if img.var() < args.var_threshold:  # Remove images where not much is going on
            os.remove(f)
            os.remove(Path(train_path, 'uncertainty', f.name))
            os.remove(Path(train_path, 'masks', f.name))
            continue


        uncertainty_img = io.imread(Path(train_path, 'uncertainty', f.name))
        uncertainty = uncertainty_img / 65535

        if uncertainty.mean > args.threshold:
            os.rename(f, Path(unlabeled_path, f.name))
            os.remove(Path(train_path, 'uncertainty', f.name))
            os.remove(Path(train_path, 'masks', f.name))

    # no new appropiate images for training => stop autotrainging
    if len(list(Path(train_path, 'images').glob('*.png'))) == 0:
        print("No images with low enough uncertainty left")
        break

    Path(iter_path, 'val').symlink_to(args.val_path, target_is_directory=True)
    callbacks = []
    callbacks.append(EarlyStopping('val_loss', patience=args.patience, verbose=0, mode='auto', restore_best_weights=True))

    new_iter_path = Path(args.out_path, str(iter + 1))
    if not new_iter_path.exists():
        new_iter_path.mkdir()
    model, history = train(model, iter_path, new_iter_path, batch_size=batch_size, patch_size=patch_size, callbacks=callbacks)

    if args.test_path and Path(args.test_path).exists():
        test_path = Path(args.test_path)
        predict_bayes(model, Path(test_path, 'images'), Path(new_iter_path, 'eval'), batch_size=batch_size, patch_size=patch_size, cutoff=cutoff)
        evaluate(Path(test_path, 'images'), Path(test_path, 'masks'), Path(new_iter_path, 'eval'))




