import argparse
import time
from pathlib import Path
import pickle
import json
from shutil import copy

import cv2
import matplotlib.pyplot as plt
import numpy as np
from keras.callbacks import ModelCheckpoint, EarlyStopping
from scipy.spatial import distance
import os

from data import trainGenerator
from model import unet_Enze19_2
from utils import helper_functions
from loss_functions import *
from keras.losses import *  # Don't remove
from shutil import copy

#%% Hyper-parameter tuning
parser = argparse.ArgumentParser(description='Glacier Front Segmentation')

parser.add_argument('--epochs', default=100, type=int, help='number of training epochs (integer value > 0)')
parser.add_argument('--batch_size', default=100, type=int, help='batch size (integer value)')
parser.add_argument('--patch_size', default=256, type=int, help='batch size (integer value)')

parser.add_argument('--EARLY_STOPPING', default=1, type=int, help='If 1, classifier is using early stopping based on validation loss with patience 20 (0/1)')
parser.add_argument("--loss", help="loss function for the deep classifiers training ", choices=["binary_crossentropy", "focal_loss", "combined_loss"], default="binary_crossentropy")
parser.add_argument('--loss_parms', action=helper_functions.StoreDictKeyPair, metavar="KEY1=VAL1,KEY2=VAL2...", help='dictionary with parameters for loss function')
parser.add_argument('--image_aug', action=helper_functions.StoreDictKeyPair, metavar="KEY1=VAL1,KEY2=VAL2...",
                    help='dictionary with the augmentation for keras Image Processing', default={'horizontal_flip':True,'rotation_range':90, 'fill_mode':nearest})

parser.add_argument('--out', type=str, help='Output path for results')
parser.add_argument('--data_path', type=str, help='Path containing training and validation data')

# parser.add_argument('--Random_Seed', default=1, type=int, help='random seed number value (any integer value)')

args = parser.parse_args()

#%%
START=time.time()

PATCH_SIZE = args.patch_size
batch_size = args.batch_size


if args.data_path:
    data_path = Path(args.data_path)
else:
    data_path = Path('data')

num_samples = len([file for file in Path(data_path, 'train/images').rglob('*.png')]) # number of training samples
num_val_samples = len([file for file in Path(data_path, 'val/images').rglob('*.png')]) # number of validation samples

if args.out:
    out_path = Path(args.out)
else:
    out_path = Path('data_' + str(PATCH_SIZE) + '/test/masks_predicted_' + time.strftime("%y%m%d-%H%M%S"))



if not out_path.exists():
    out_path.mkdir(parents=True)

# log all arguments including default ones
with open(Path(out_path, 'arguments.json'), 'w') as f:
    f.write(json.dumps(vars(args)))

# copy image file list to output
for d in data_path.iterdir():
    if Path(d, 'image_list.json').exists():
        copy(Path(d, 'image_list.json'), Path(out_path, d.name + '_image_list.json'))


bin = binary_crossentropy
if args.loss == "combined_loss":
    if not args.loss_parms:
        print("combined_loss needs loss functions as parameter")
    else:
        functions = []
        split = []
        for func_name, weight in args.loss_parms.items():

            # for functions with additional parameters
            # generate loss function with default parameters
            # and standard y_true,y_pred signature
            if func_name == "focal_loss":
                function = locals()[func_name]()
            else:
                function = locals()[func_name]
            functions.append(function)
            split.append(float(weight))

        loss_function = combined_loss(functions, split)

# for loss functions with additional parameters call to get function with y_true, y_pred arguments
elif args.loss == 'focal_loss':
    if args.loss_parms:
        loss_function = locals()[args.loss](**args.loss_parms)
        #loss_function = locals()[args.LOSS](alpha=args.alpha, gamma=args.gamma)
    else:
        loss_function = locals()[args.loss]()
else:
    loss_function = locals()[args.loss]


data_gen_args = dict(horizontal_flip=True,
                     rotation_range=90,
                    fill_mode='nearest')

train_Generator = trainGenerator(batch_size = batch_size,
                        train_path = str(Path(data_path, 'train')),
                        image_folder = 'images',
                        mask_folder = 'masks_zones',
                        aug_dict = args.image_aug,
                        save_to_dir = None)

val_Generator = trainGenerator(batch_size = batch_size,
                        train_path = str(Path(data_path, 'val')),
                        image_folder = 'images',
                        mask_folder = 'masks_zones',
                        aug_dict = None,
                        save_to_dir = None)



model = unet_Enze19_2(loss_function=loss_function)
model_checkpoint = ModelCheckpoint(str(Path(str(out_path), 'unet_zone.hdf5')), monitor='val_loss', verbose=0, save_best_only=True)
early_stopping = EarlyStopping('val_loss', patience=20, verbose=0, mode='auto', restore_best_weights=True)


steps_per_epoch = np.ceil(num_samples / batch_size)
validation_steps = np.ceil(num_val_samples / batch_size)
History = model.fit_generator(train_Generator,
                    steps_per_epoch=steps_per_epoch,
                    epochs=args.epochs,
                    validation_data=val_Generator,
                    validation_steps=validation_steps,
                    callbacks=[model_checkpoint, early_stopping])
#model.fit_generator(train_Generator, steps_per_epoch=300, epochs=1, callbacks=[model_checkpoint], class_weight=[0.0000001,0.9999999])

##########
##########
# save loss plot
plt.figure()
plt.rcParams.update({'font.size': 18})
plt.plot(model.history.epoch, model.history.history['loss'], 'X-', label='training loss', linewidth=4.0)
plt.plot(model.history.epoch, model.history.history['val_loss'], 'o-', label='validation loss', linewidth=4.0)
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend(loc='upper right')
plt.minorticks_on()
plt.grid(which='minor', linestyle='--')
plt.savefig(str(Path(str(out_path), 'loss_plot.png')), bbox_inches='tight', format='png', dpi=200)
plt.show()

# # save model
model.save(str(Path(out_path,'model_' + model.name + '.h5').absolute()))
pickle.dumps(open(Path(out_path, 'loss_function' + model.name + '.pkl'), 'w'), loss_function)

# Don"t need checkpoint model anymore
os.remove(Path(str(out_path), 'unet_zone.hdf5'))

##########
##########

#%%
# testGene = testGenerator('data_256/test/images', num_image=5) #########
# results = model.predict_generator(testGene, 5, verbose=0) #########
# saveResult_Amir('data_256/test/masks_predicted', results) #########

#####################
#####################
import skimage.io as io
import Amir_utils
from pathlib import Path
from sklearn.metrics import f1_score, recall_score
test_path = str(Path(data_path,'test'))

if not out_path:
    out_path = Path('output')

DICE_all = []
EUCL_all = []
Specificity_all =[]
Sensitivity_all = []
F1_all = []
test_file_names = []
Perf = {}

img_list = None


for filename in Path(test_path,'images').rglob('*.png'):
    img = io.imread(filename, as_gray=True)
    img = img / 255
    img_pad = cv2.copyMakeBorder(img, 0, (PATCH_SIZE-img.shape[0]) % PATCH_SIZE, 0, (PATCH_SIZE-img.shape[1]) % PATCH_SIZE, cv2.BORDER_CONSTANT)
    p_img, i_img = Amir_utils.extract_grayscale_patches(img_pad, (PATCH_SIZE,PATCH_SIZE), stride = (PATCH_SIZE,PATCH_SIZE))
    p_img = np.reshape(p_img,p_img.shape+(1,))

    p_img_predicted = model.predict(p_img)

    p_img_predicted = np.reshape(p_img_predicted,p_img_predicted.shape[:-1])
    img_mask_predicted_recons = Amir_utils.reconstruct_from_grayscale_patches(p_img_predicted,i_img)[0]

    # unpad and normalize
    img_mask_predicted_recons_unpad = img_mask_predicted_recons[0:img.shape[0],0:img.shape[1]]
    img_mask_predicted_recons_unpad_norm = cv2.normalize(src=img_mask_predicted_recons_unpad, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    # quantization to make the binary masks
    img_mask_predicted_recons_unpad_norm[img_mask_predicted_recons_unpad_norm < 127] = 0
    img_mask_predicted_recons_unpad_norm[img_mask_predicted_recons_unpad_norm >= 127] = 255

    io.imsave(Path(str(out_path), Path(filename).name), img_mask_predicted_recons_unpad_norm)

#    mask_max = img_mask_predicted_recons_unpad_norm.max()
#    if mask_max > 0:
#        mask_predicted_norm = img_mask_predicted_recons_unpad_norm / img_mask_predicted_recons_unpad_norm.max()
#    else:
#        mask_predicted_norm = img_mask_predicted_recons_unpad_norm
#
#    mask_predicted_flat = mask_predicted_norm.flatten().astype(int)
#
#    gt_path = str(Path(test_path,'masks_zones'))
#    gt = io.imread(str(Path(gt_path,filename.name)), as_gray=True)
#
#    max = gt.max()
#    if max >0:
#        gt_norm = gt / gt.max()
#    else:
#        gt_norm = gt
#    gt_flat = gt_norm.flatten().astype(int)
#
#    Specificity_all.append(helper_functions.specificity(gt_flat, mask_predicted_flat))
#    Sensitivity_all.append(recall_score(gt_flat, mask_predicted_flat))
#    F1_all.append(f1_score(gt_flat, mask_predicted_flat))
#
#    # DICE
#    DICE_all.append(helper_functions.dice_coefficient(gt_flat, mask_predicted_flat))
#    EUCL_all.append(distance.euclidean(gt_flat, mask_predicted_flat))
#    test_file_names.append(filename.name)
#
#Perf['Specificity_all'] = Specificity_all
#Perf['Specificity_avg'] = np.mean(Specificity_all)
#Perf['Sensitivity_all'] = Sensitivity_all
#Perf['Sensitivity_avg'] = np.mean(Sensitivity_all)
#Perf['F1_score_all'] = F1_all
#Perf['F1_score_avg'] = np.mean(F1_all)
#Perf['DICE_all'] = DICE_all
#Perf['DICE_avg'] = np.mean(DICE_all)
#Perf['EUCL_all'] = EUCL_all
#Perf['EUCL_avg'] = np.mean(EUCL_all)
#Perf['test_file_names'] = test_file_names
#
#pickle.dump(Perf, open(Path(out_path, 'performance.pkl'), 'wb'))
#
#with open(str(Path(str(out_path) , 'ReportOnModel.txt')), 'w') as f:
#    f.write('Dice\tEuclidian\n')
#    f.write(str(Perf['DICE_avg']) + '\t'
#          + str(Perf['EUCL_avg']) + '\n')
#    f.write('Sensitivity\tSpecificitiy\tf1_score\n')
#    f.write(str(Perf['Sensitivity_avg']) + '\t'
#      + str(Perf['Specificity_avg']) + '\t'
#      + str(Perf['F1_score_avg']) + '\n')

END=time.time()
print('Execution Time: ', END-START)
