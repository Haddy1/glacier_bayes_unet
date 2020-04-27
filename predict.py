import argparse
from loss_functions import *
import keras
from pathlib import Path
import imageio as io
import Amir_utils
import cv2
import json
import numpy as np


parser = argparse.ArgumentParser(description='Glacier Front Segmentation')
parser.add_argument('--model_path', type=str, help='trained model path')
args = parser.parse_args()


model_path = args.model_path
args = json.load(open(Path(model_path, 'arguments.txt'), 'r'))
PATCH_SIZE =  args['patch_size']

out_path = Path(args['out'])
if not out_path.exists():
    out_path.mkdir(parents=True)


if args['loss'] == "combined_loss":
    functions = []
    split = []
    for func_name, weight in args['loss_parms'].items():

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


model = keras.models.load_model(str(Path(model_path, 'model_unet_Enze19_2.h5')), custom_objects=loss_function)
test_path = Path(args['data_path'], 'test')
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


