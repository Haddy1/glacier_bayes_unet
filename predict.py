import sys
import json
from pathlib import Path
from keras.models import load_model
import pickle
import argparse
from loss_functions import *
from keras.losses import binary_crossentropy
import numpy as np
import skimage.io as io
import cv2
from utils.evaluate import evaluate
from preprocessing.image_patches import extract_grayscale_patches, reconstruct_from_grayscale_patches

parser = argparse.ArgumentParser(description='Glacier Front Segmentation Prediction')
parser.add_argument('--model_path', type=str, help='Path containing trained model')
parser.add_argument('--data_path', type=str, help='Path containing trained model')
parser.add_argument('--out_path', type=str, help='output path for predictions')
parser.add_argument('--evaluate', action='store_true', help='evaluate model - requires data_path with labeled data')
parser.add_argument('--batch_size', default=1, type=int, help='batch size (integer value)')
parser.add_argument('--debug', action='store_true')
args = parser.parse_args()

if args.debug:
    gpus = tf.config.experimental.list_physical_devices('GPU')

    # Currently, memory growth needs to be the same across GPUs
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

   # tf.config.experimental.set_virtual_device_configuration(gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=4096)])

model_path = Path(args.model_path)

options = json.load(open(Path(model_path, 'arguments.json'), 'r'))
model_file = next(model_path.glob('model_*.h5'))
model_name = model_file.name[6:-3]
if options['loss'] == "combined_loss":
    if not 'loss_parms' in options:
        print("combined_loss needs loss functions as parameter")
    else:
        functions = []
        split = []
        for func_name, weight in options['loss_parms'].items():

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
elif options['loss'] == 'focal_loss':
    if 'loss_parms' in options:
        loss_function = locals()[options['loss']](**options['loss_parms'])
    else:
        loss_function = locals()[options['loss']]()
elif options['loss'] == 'binary_crossentropy':
    loss_function = binary_crossentropy
else:
    loss_function = locals()[options['loss']]


model = load_model(str(model_file.absolute()), custom_objects={ 'loss': loss_function})


out_path = Path(args.out_path)

if not out_path.exists():
    out_path.mkdir(parents=True)


if args.evaluate:
    img_path = Path(args.data_path, 'images')
else:
    img_path = Path(args.data_path)

patch_size = options['patch_size']

DICE_all = []
EUCL_all = []
Specificity_all =[]
Sensitivity_all = []
F1_all = []
test_file_names = []
Perf = {}
img_list = None


for filename in img_path.rglob('*.png'):
    print(filename)
    img = io.imread(filename, as_gray=True)
    img = img / 255
    img_pad = cv2.copyMakeBorder(img, 0, (patch_size - img.shape[0]) % patch_size, 0, (patch_size - img.shape[1]) % patch_size, cv2.BORDER_CONSTANT)
    p_img, i_img = extract_grayscale_patches(img_pad, (patch_size, patch_size), stride = (patch_size, patch_size))
    p_img = np.reshape(p_img,p_img.shape+(1,))

    p_img_predicted = model.predict(p_img, batch_size=args.batch_size)

    p_img_predicted = np.reshape(p_img_predicted,p_img_predicted.shape[:-1])
    mask_predicted = reconstruct_from_grayscale_patches(p_img_predicted,i_img)[0]
    mask_predicted = mask_predicted[:img.shape[0], :img.shape[1]]

    # to 8 bit image
    mask_predicted = cv2.normalize(src=mask_predicted, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    # quantization to make the binary masks
    mask_predicted[mask_predicted < 127] = 0
    mask_predicted[mask_predicted >= 127] = 255

    io.imsave(Path(str(out_path), Path(filename).name), mask_predicted)

if args.evaluate:
    evaluate(args.data_path, out_path)

