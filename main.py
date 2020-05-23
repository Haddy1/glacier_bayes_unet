import argparse
import time
from pathlib import Path
import pickle
import json

import cv2
import matplotlib.pyplot as plt
import numpy as np
from keras.callbacks import ModelCheckpoint, EarlyStopping, CSVLogger
from keras.models import load_model
import os

from utils.data import trainGenerator
from model import unet_Enze19_2
from utils import helper_functions
from loss_functions import *
from keras.losses import *
from shutil import copy, rmtree
import tensorflow as tf
from preprocessing.preprocessor import Preprocessor
from preprocessing import data_generator, filter

# Hyper-parameter tuning
parser = argparse.ArgumentParser(description='Glacier Front Segmentation')

parser.add_argument('--epochs', default=250, type=int, help='number of training epochs (integer value > 0)')
parser.add_argument('--patience', default=30, type=int, help='how long to wait for improvements before Early_stopping')
parser.add_argument('--batch_size', default=100, type=int, help='batch size (integer value)')
parser.add_argument('--patch_size', default=256, type=int, help='batch size (integer value)')

parser.add_argument('--EARLY_STOPPING', default=1, type=int,
                    help='If 1, classifier is using early stopping based on val loss with patience 20 (0/1)')
parser.add_argument("--loss", help="loss function for the deep classifiers training ",
                    choices=["binary_crossentropy", "focal_loss", "combined_loss"], default="binary_crossentropy")
parser.add_argument('--loss_parms', action=helper_functions.StoreDictKeyPair, metavar="KEY1=VAL1,KEY2=VAL2...",
                    help='dictionary with parameters for loss function')
parser.add_argument('--image_aug', action=helper_functions.StoreDictKeyPair, metavar="KEY1=VAL1,KEY2=VAL2...",
                    help='dictionary with the augmentation for keras Image Processing',
                    default={'horizontal_flip': False, 'rotation_range': 0, 'fill_mode': 'nearest'})
parser.add_argument("--denoise", help="Denoise filter",
                    choices=["none", "bilateral", "median", 'nlmeans', "enhanced_lee", "kuan"], default="None")
parser.add_argument('--denoise_parms', action=helper_functions.StoreDictKeyPair, metavar="KEY1=VAL1,KEY2=VAL2...",
                    help='dictionary with parameters for denoise filter')
parser.add_argument('--contrast', default=0, type=int, help='Contrast Enhancement')
parser.add_argument('--image_patches', default=0, type=int, help='Training data is already split into image patches')

parser.add_argument('--out_path', type=str, help='Output path for results')
parser.add_argument('--data_path', type=str, help='Path containing training and val data')
parser.add_argument('--resume_training', type=str, help='Resume training from checkpoint')

# parser.add_argument('--Random_Seed', default=1, type=int, help='random seed number value (any integer value)')

args = parser.parse_args()

START = time.time()

if args.resume_training:
    if 'hdf5' in args.resume_training:
        checkpoint_file = Path(args.resume_training)
    else:
        checkpoint_file = next(Path(args.resume_traininig).glob('*.hdf5'))
    if not checkpoint_file.exists():
        print(checkpoint_file + ' does not exist')
        exit(-1)

if args.resume_training and Path(checkpoint_file.parent, 'arguments.json').exists():
    resume_arg = args.resume_training
    debug = args.debug
    args.__dict__ = json.load(open(Path(checkpoint_file.parent, 'arguments.json'), 'r'))
    args.__dict__['resume_training'] = resume_arg
    args.__dict__['debug'] = debug

patch_size = args.patch_size
batch_size = args.batch_size

if args.data_path:
    data_path = Path(args.data_path)
else:
    data_path = Path('data')

if args.out_path:
    out_path = Path(args.out_path)
else:
    out_path = Path('data_' + str(patch_size) + '/test/masks_predicted_' + time.strftime("%y%m%d-%H%M%S"))

if args.image_patches:
    patches_path = data_path
else:
    patches_path = Path(out_path, 'patches')

if not out_path.exists():
    out_path.mkdir(parents=True)

# log all arguments including default ones
with open(Path(out_path, 'arguments.json'), 'w') as f:
    f.write(json.dumps(vars(args)))

# Preprocessing
preprocessor = Preprocessor()
if args.denoise:
    preprocessor.add_filter(filter.get_denoise_filter(args.denoise, args.denoise_parms))

if args.contrast:
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(25, 25))  # CLAHE adaptive contrast enhancement
    preprocessor.add_filter(clahe.apply)

if not args.image_patches and not args.resume_training:
    data_generator.process_data(Path(data_path, 'train'), Path(patches_path, 'train'), patch_size=patch_size,
                                preprocessor=preprocessor)
    data_generator.process_data(Path(data_path, 'val'), Path(patches_path, 'val'), patch_size=patch_size,
                                preprocessor=preprocessor)

# copy image file list to output
for d in patches_path.iterdir():
    if Path(d, 'image_list.json').exists():
        copy(Path(d, 'image_list.json'), Path(out_path, d.name + '_image_list.json'))

train_Generator = trainGenerator(batch_size=batch_size,
                                 train_path=str(Path(patches_path, 'train')),
                                 image_folder='images',
                                 mask_folder='masks',
                                 aug_dict=args.image_aug,
                                 save_to_dir=None)

val_Generator = trainGenerator(batch_size=batch_size,
                               train_path=str(Path(patches_path, 'val')),
                               image_folder='images',
                               mask_folder='masks',
                               aug_dict=None,
                               save_to_dir=None)

loss_function = get_loss_function(args.loss, args.loss_parms)

if args.resume_training:

    model = load_model(str(checkpoint_file.absolute()), custom_objects={'loss': loss_function})
else:
    model = unet_Enze19_2(loss_function=loss_function, input_size=(patch_size, patch_size, 1))

model_checkpoint = ModelCheckpoint(str(Path(out_path, 'unet_zone.hdf5')), monitor='val_loss', verbose=0,
                                   save_best_only=True)
early_stopping = EarlyStopping('val_loss', patience=args.patience, verbose=0, mode='auto', restore_best_weights=True)
csv_logger = CSVLogger(str(Path(out_path, model.name + '_history,.csv')), append=True)

num_samples = len([file for file in Path(patches_path, 'train/images').rglob('*.png')])  # number of training samples
num_val_samples = len([file for file in Path(patches_path, 'val/images').rglob('*.png')])  # number of val samples

steps_per_epoch = np.ceil(num_samples / batch_size)
validation_steps = np.ceil(num_val_samples / batch_size)
history = model.fit_generator(train_Generator,
                              steps_per_epoch=steps_per_epoch,
                              epochs=args.epochs,
                              validation_data=val_Generator,
                              validation_steps=validation_steps,
                              callbacks=[model_checkpoint, early_stopping, csv_logger])

# # save model
model.save(str(Path(out_path, 'model_' + model.name + '.h5').absolute()))
try:
    pickle.dump(history.history, open(Path(out_path, 'history_' + model.name + '.pkl'), 'wb'))
except:
    print("History could not be saved")
##########
##########
# save loss plot
plt.figure()
plt.rcParams.update({'font.size': 18})
plt.plot(model.history.epoch, model.history.history['loss'], 'X-', label='training loss', linewidth=4.0)
plt.plot(model.history.epoch, model.history.history['val_loss'], 'o-', label='val loss', linewidth=4.0)
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend(loc='upper right')
plt.minorticks_on()
plt.grid(which='minor', linestyle='--')
plt.savefig(str(Path(str(out_path), 'loss_plot.png')), bbox_inches='tight', format='png', dpi=200)
plt.show()

# Cleanup
os.remove(Path(str(out_path), 'unet_zone.hdf5'))
if Path(out_path, 'patches').exists():
    rmtree(Path(out_path, 'patches'))

#####################
#####################
import skimage.io as io
from utils.evaluate import evaluate
from pathlib import Path
from preprocessing.image_patches import extract_grayscale_patches, reconstruct_from_grayscale_patches

test_path = str(Path(data_path, 'test'))

if not out_path:
    out_path = Path('output')

DICE_all = []
EUCL_all = []
Specificity_all = []
Sensitivity_all = []
F1_all = []
test_file_names = []
Perf = {}

img_list = None

for filename in Path(test_path, 'images').rglob('*.png'):
    img = io.imread(filename, as_gray=True)
    img = preprocessor.process(img)
    img = img / 255
    img_pad = cv2.copyMakeBorder(img, 0, (patch_size - img.shape[0]) % patch_size, 0,
                                 (patch_size - img.shape[1]) % patch_size, cv2.BORDER_CONSTANT)
    p_img, i_img = extract_grayscale_patches(img_pad, (patch_size, patch_size), stride=(patch_size, patch_size))
    p_img = np.reshape(p_img, p_img.shape + (1,))

    p_img_predicted = model.predict(p_img)

    p_img_predicted = np.reshape(p_img_predicted, p_img_predicted.shape[:-1])
    mask_predicted = reconstruct_from_grayscale_patches(p_img_predicted, i_img)[0]
    mask_predicted = mask_predicted[:img.shape[0], :img.shape[1]]

    # to 8 bit image
    mask_predicted = cv2.normalize(src=mask_predicted, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX,
                                   dtype=cv2.CV_8U)

    # quantization to make the binary masks
    mask_predicted[mask_predicted < 127] = 0
    mask_predicted[mask_predicted >= 127] = 255

    io.imsave(Path(str(out_path), Path(filename).name), mask_predicted)

evaluate(test_path, out_path)

END = time.time()
print('Execution Time: ', END - START)
