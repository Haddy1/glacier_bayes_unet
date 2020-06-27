import sys
import json
import seaborn as sns
from pathlib import Path
from keras.models import load_model
import pickle
import argparse
from loss_functions import *
from layers.BayesDropout import  BayesDropout
from preprocessing.preprocessor import Preprocessor
from preprocessing import filter
from keras.losses import binary_crossentropy
import numpy as np
import skimage.io as io
import cv2
from utils.evaluate import evaluate, evaluate_dice_only
from preprocessing.image_patches import extract_grayscale_patches, reconstruct_from_grayscale_patches
import matplotlib.pyplot as plt
import shutil

def predict(model, img_path, out_path, batch_size=16, patch_size=256, cutoff=0.5, preprocessor=None):
    if not Path(out_path).exists():
        Path(out_path).mkdir(parents=True)

    for filename in Path(img_path).rglob('*.png'):
        print(filename)
        img = io.imread(filename, as_gray=True)
        if preprocessor is not None:
            img = preprocessor.process(img)
        img = img / 255
        img_pad = cv2.copyMakeBorder(img, 0, (patch_size - img.shape[0]) % patch_size, 0, (patch_size - img.shape[1]) % patch_size, cv2.BORDER_CONSTANT)
        p_img, i_img = extract_grayscale_patches(img_pad, (patch_size, patch_size), stride = (patch_size, patch_size))
        p_img = np.reshape(p_img,p_img.shape+(1,))

        p_img_predicted = model.predict(p_img, batch_size=batch_size)

        p_img_predicted = np.reshape(p_img_predicted,p_img_predicted.shape[:-1])
        mask_predicted = reconstruct_from_grayscale_patches(p_img_predicted,i_img)[0]
        mask_predicted = mask_predicted[:img.shape[0], :img.shape[1]]


        # thresholding to make binary mask
        mask_predicted[mask_predicted < cutoff] = 0
        mask_predicted[mask_predicted >= cutoff] = 255

        io.imsave(Path(out_path, filename.stem + '_pred.png'), mask_predicted.astype(np.uint8))

def predict_bayes(model, img_path, out_path, batch_size=16, patch_size=256, cutoff=0.5, preprocessor=None, mc_iterations = 50):
    if not Path(out_path).exists():
        Path(out_path).mkdir(parents=True)

    for filename in Path(img_path).rglob('*.png'):
        #print(filename)
        img = io.imread(filename, as_gray=True)
        if preprocessor is not None:
            img = preprocessor.process(img)
        img = img / 255
        img_pad = cv2.copyMakeBorder(img, 0, (patch_size - img.shape[0]) % patch_size, 0, (patch_size - img.shape[1]) % patch_size, cv2.BORDER_CONSTANT)
        p_img, i_img = extract_grayscale_patches(img_pad, (patch_size, patch_size), stride = (patch_size, patch_size))
        p_img = np.reshape(p_img,p_img.shape+(1,))

        predictions = []
        for i in range(mc_iterations):
            prediction = model.predict(p_img, batch_size=batch_size)
            predictions.append(prediction)
        predictions = np.array(predictions)


        p_img_predicted = predictions.mean(axis=0)
        p_img_predicted = np.reshape(p_img_predicted,p_img_predicted.shape[:-1])
        mask_predicted = reconstruct_from_grayscale_patches(p_img_predicted,i_img)[0]
        mask_predicted = mask_predicted[:img.shape[0], :img.shape[1]]

        mask_predicted_img = (65535 * mask_predicted).astype(np.uint16)
        cv2.imwrite(str(Path(out_path, filename.stem + '_pred_perc.png')), mask_predicted_img)

        # thresholding to make binary mask
        mask_predicted[mask_predicted < cutoff] = 0
        mask_predicted[mask_predicted >= cutoff] = 255

        io.imsave(Path(out_path, filename.stem + '_pred.png'), mask_predicted.astype(np.uint8))

        p_uncertainty = predictions.var(axis=0)
        p_uncertainty = np.reshape(p_uncertainty,p_uncertainty.shape[:-1])
        uncertainty = reconstruct_from_grayscale_patches(p_uncertainty,i_img)[0]
        uncertainty = uncertainty[:img.shape[0], :img.shape[1]]
        uncertainty_img = (65535 * uncertainty).astype(np.uint16)
        cv2.imwrite(str(Path(out_path, filename.stem + '_uncertainty.png')), uncertainty_img)
        #np.save(Path(out_path, filename.stem + '_uncertainty.npy'), uncertainty)


def get_cutoff_point(model, val_path, out_path, batch_size=16, patch_size=256, preprocessor=None):
    tmp_dir = Path(out_path, 'cutoff_tmp')
    dice = []
    for i in range(1,10):
        i = i/10
        predict(model, Path(val_path, 'images'), tmp_dir, batch_size=batch_size,
                       patch_size=patch_size, preprocessor=preprocessor, cutoff=i)
        dice_mean = np.mean(evaluate_dice_only(Path(val_path, 'images'), Path(val_path, 'masks'), tmp_dir))
        dice.append(dice_mean)

    dice = np.array(dice)
    argmax = np.argmax(dice)

    np.save(Path(out_path, 'dice_cutoff.npy'), dice)
    max_dice = dice[argmax]
    max_cutoff = (np.arange(1,10)/10)[argmax]

    plt.rcParams.update({'font.size': 18})
    plt.figure()
    plt.plot((max_cutoff, max_cutoff),(0, max_dice), linestyle=':', linewidth=2, color='grey')
    plt.plot(np.arange(1,10)/10, dice)
    plt.annotate(f'{max_dice:.2f}', (max_cutoff, max_dice), fontsize='x-small')
    plt.ylabel('Dice')
    plt.xlabel('Cutoff Point')
    plt.savefig(str(Path(out_path, 'cutoff.png')), bbox_inches='tight', format='png', dpi=200)

    shutil.rmtree(tmp_dir, ignore_errors=True)

    return max_cutoff

if __name__ is '__main__':
    parser = argparse.ArgumentParser(description='Glacier Front Segmentation Prediction')
    parser.add_argument('--model_path', type=str, help='Path containing trained model')
    parser.add_argument('--img_path', type=str, help='Path containing images to be segmented')
    parser.add_argument('--out_path', type=str, help='output path for predictions')
    parser.add_argument('--gt_path', type=str, help='Path containing the ground truth, necessary for evaluation')
    parser.add_argument('--batch_size', default=1, type=int, help='batch size (integer value)')
    parser.add_argument('--cutoff', type=float, help='cutoff point of binarisation')
    args = parser.parse_args()

    model_path = Path(args.model_path)
    options = json.load(open(Path(model_path, 'options.json'), 'r'))
    # Preprocessing
    preprocessor = Preprocessor()
    if 'denoise' in options:
        if 'denoise_parms' in options:
            preprocessor.add_filter(filter.get_denoise_filter(options['denoise']))
        else:
            preprocessor.add_filter(filter.get_denoise_filter(options['denoise'], options['denoise_parms']))

    if 'contrast' in options and options['contrast']:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(25, 25))  # CLAHE adaptive contrast enhancement
        preprocessor.add_filter(clahe.apply)

    if 'loss_parms' in options:
        loss_function = get_loss_function(options['loss'], options['loss_parms'])
    else:
        loss_function = get_loss_function(options['loss'])

    model_file = next(model_path.glob('model_*.h5'))
    model_name = model_file.name[6:-3]
    model = load_model(str(model_file.absolute()), custom_objects={ 'loss': loss_function, 'BayesDropout':BayesDropout})
    print(model_name)

    if args.cutoff:
        cutoff = args.cutoff
    elif 'cutoff' in options:
        cutoff = options['cutoff']
    else:
        cutoff = 0.5


    out_path = Path(args.out_path)

    if not out_path.exists():
        out_path.mkdir(parents=True)

    if 'bayes' in model_name:
        print("Bayes")
        predict_bayes(model, args.img_path, out_path, batch_size=args.batch_size, patch_size=options['patch_size'], cutoff=cutoff, preprocessor=preprocessor)
    else:
        predict(model, args.img_path, out_path, batch_size=args.batch_size, patch_size=options['patch_size'], cutoff=cutoff, preprocessor=preprocessor)

    if args.gt_path:
        evaluate(args.img_path, args.gt_path, out_path)

