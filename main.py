import argparse
import os
import time
from pathlib import Path
import json

# matplotlib.use('ps')
import cv2
import matplotlib.pyplot as plt
import numpy as np
from keras.callbacks import ModelCheckpoint
from scipy.spatial import distance

from data import trainGenerator
from model import unet_Enze19_2
from loss_functions import focal_loss
from keras.losses import *

#%% Hyper-parameter tuning
parser = argparse.ArgumentParser(description='Glacier Front Segmentation')

# parser.add_argument('--Test_Size', default=0.2, type=float, help='Test set ratio for splitting from the whole dataset (values between 0 and 1)')
# parser.add_argument('--Validation_Size', default=0.1, type=float, help='Validation set ratio for splitting from the training set (values between 0 and 1)')

# parser.add_argument('--Classifier', default='unet', type=str, help='Classifier to use (unet/unet_Enze19)')
parser.add_argument('--Epochs', default=100, type=int, help='number of training epochs (integer value > 0)')
parser.add_argument('--Batch_Size', default=100, type=int, help='batch size (integer value)')
parser.add_argument('--Patch_Size', default=256, type=int, help='batch size (integer value)')

# parser.add_argument('--EARLY_STOPPING', default=1, type=int, help='If 1, classifier is using early stopping based on validation loss with patience 20 (0/1)')
parser.add_argument("--LOSS", help="loss function for the deep classifiers training ", choices=["binary_crossentropy", "focal_loss"], default="binary_crossentropy")
#parser.add_argument('--Loss_Parms', type=str, help='dictionary with parameters for loss function')
parser.add_argument('--alpha', type=float, help='parameter for loss function')
parser.add_argument('--gamma', type=float, help='parameter for loss function')

parser.add_argument('--OUTPATH', default='output/results/21.09/', type=str, help='Output path for results')

# parser.add_argument('--Random_Seed', default=1, type=int, help='random seed number value (any integer value)')

args = parser.parse_args()

#%%
START=time.time()

PATCH_SIZE = args.Patch_Size
batch_size = args.Batch_Size 


num_samples = len([file for file in Path('data_'+str(PATCH_SIZE)+'/train/images/').rglob('*.png')]) # number of training samples
num_val_samples = len([file for file in Path('data_'+str(PATCH_SIZE)+'/val/images/').rglob('*.png')]) # number of validation samples

Out_Path = Path('data_'+str(PATCH_SIZE)+'/test/masks_predicted_'+time.strftime("%y%m%d-%H%M%S")) # adding the time in the folder name helps to keep the results for multiple back to back exequations of the code
if not os.path.exists(Path('data/train/aug')): os.makedirs(Path('data/train/aug'))
if not os.path.exists(Out_Path): os.makedirs(Out_Path)

#data_gen_args = dict(rotation_range=0.2,
#                    width_shift_range=0.05,
#                    height_shift_range=0.05,
#                    shear_range=0.05,
#                    zoom_range=0.05,
#                    horizontal_flip=True,
#                    fill_mode='nearest')
data_gen_args = dict(horizontal_flip=True,
                    fill_mode='nearest')

train_Generator = trainGenerator(batch_size = batch_size,
                        train_path = str(Path('data_'+str(PATCH_SIZE)+'/train')),
                        image_folder = 'images',
                        mask_folder = 'masks_zones', 
                        aug_dict = data_gen_args, 
                        save_to_dir = None)

val_Generator = trainGenerator(batch_size = batch_size,
                        train_path = str(Path('data_'+str(PATCH_SIZE)+'/val')),
                        image_folder = 'images',
                        mask_folder = 'masks_zones', 
                        aug_dict = None, 
                        save_to_dir = None)


if args.alpha and args.gamma:
    #loss_parms = json.loads(args.Loss_Parms)
    #loss_function = locals()[args.LOSS](**loss_parms)
    loss_function = locals()[args.LOSS](alpha=args.alpha, gamma=args.gamma)
else:
    loss_function = locals()[args.LOSS]()


model = unet_Enze19_2(loss_function=loss_function)
model_checkpoint = ModelCheckpoint(str(Path(str(Out_Path),'unet_zone.hdf5')), monitor='val_loss', verbose=0, save_best_only=True)

steps_per_epoch = np.ceil(num_samples / batch_size)
validation_steps = np.ceil(num_val_samples / batch_size)
History = model.fit_generator(train_Generator, 
                    steps_per_epoch=steps_per_epoch, 
                    epochs=args.Epochs, 
                    validation_data=val_Generator,
                    validation_steps=validation_steps,
                    callbacks=[model_checkpoint])
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
plt.savefig(str(Path(str(Out_Path),'loss_plot.png')), bbox_inches='tight', format='png', dpi=200)
plt.show()

# # save model
# model.save(str(Path(OUTPUT_PATH + 'MyModel' + SaveName + '.h5').absolute()))
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
test_path = str(Path('data_'+str(PATCH_SIZE)+'/test/'))

DICE_all = []
EUCL_all = []
test_file_names = []
Perf = {}
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
    
    io.imsave(Path(str(Out_Path), Path(filename).name), img_mask_predicted_recons_unpad_norm) 
    
    # DICE
    gt_path = str(Path(test_path,'masks_zones'))
    gt_name = filename.name.partition('.')[0] + '_zones.png'
    gt = io.imread(str(Path(gt_path,gt_name)), as_gray=True)
    
    DICE_all.append(distance.dice(gt.flatten(), img_mask_predicted_recons_unpad_norm.flatten()))
    DICE_avg = np.mean(DICE_all)
    EUCL_all.append(distance.euclidean(gt.flatten(), img_mask_predicted_recons_unpad_norm.flatten()))
    EUCL_avg = np.mean(EUCL_all)
    test_file_names.append(filename.name)

Perf['DICE_all'] = DICE_all
Perf['DICE_avg'] = DICE_avg
Perf['EUCL_all'] = EUCL_all
Perf['EUCL_avg'] = EUCL_avg
Perf['test_file_names'] = test_file_names
np.savez(str(Path(str(Out_Path),'Performance.npz')), Perf)

with open(str(Path(str(Out_Path) , 'ReportOnModel.txt')), 'a') as f:
    f.write(str(Perf['DICE_avg']) + ', ' 
            + str(Perf['EUCL_avg']) + '\n')

END=time.time()
print('Execution Time: ', END-START)
