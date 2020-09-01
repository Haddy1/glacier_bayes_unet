import sys
import json
import seaborn as sns
from pathlib import Path
from keras.models import load_model
import pickle
import argparse
from loss_functions import *
from keras.losses import binary_crossentropy
from layers.BayesDropout import  BayesDropout
from preprocessing.preprocessor import Preprocessor
from preprocessing import filter
import numpy as np
import skimage.io as io
import cv2
from utils.metrics import dice_coefficient
from utils.evaluate import evaluate#, evaluate_dice_only
from preprocessing.image_patches import extract_grayscale_patches, reconstruct_from_grayscale_patches
import matplotlib.pyplot as plt
import pandas as pd
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

        if cutoff is not None:
            # thresholding to make binary mask
            mask_predicted[mask_predicted < cutoff] = 0
            mask_predicted[mask_predicted >= cutoff] = 255
        else:
            mask_predicted = 255 * mask_predicted


        io.imsave(Path(out_path, filename.stem + '_pred.png'), mask_predicted.astype(np.uint8))

def predict_bayes(model, img_path, out_path, batch_size=16, patch_size=256, cutoff=0.5, preprocessor=None, mc_iterations = 20, uncertainty_threshold=1e-3):
    if not Path(out_path).exists():
        Path(out_path).mkdir(parents=True)

    predict_options = {'mc_iterations': mc_iterations}

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

        mask_predicted_img = 65535 * mask_predicted
        io.imsave(Path(out_path, filename.stem + '_pred_perc.png'), mask_predicted_img.astype(np.uint16))

        if cutoff is not None:
            # thresholding to make binary mask
            mask_predicted[mask_predicted < cutoff] = 0
            mask_predicted[mask_predicted >= cutoff] = 255
        else:
            mask_predicted = 255 * mask_predicted

        io.imsave(Path(out_path, filename.stem + '_pred.png'), mask_predicted.astype(np.uint8))

        p_uncertainty = predictions.var(axis=0)
        p_uncertainty = np.reshape(p_uncertainty,p_uncertainty.shape[:-1])
        uncertainty = reconstruct_from_grayscale_patches(p_uncertainty,i_img)[0]
        uncertainty = uncertainty[:img.shape[0], :img.shape[1]]
        uncertainty_img = (65535 * uncertainty).astype(np.uint16)
        io.imsave(Path(out_path, filename.stem + '_uncertainty.png'), uncertainty_img, check_contrast=False )

        confidence_img = mask_predicted[:,:,None] * np.ones((img.shape[0], img.shape[1], 3)) # broadcast to rgb img
        confidence_img = confidence_img.astype(np.uint8)
        confidence_img[uncertainty > uncertainty_threshold, :] = np.array([255, 0,0])   # make  uncertain pixels red
        io.imsave(Path(out_path, filename.stem + '_confidence.png'), confidence_img)
        #np.save(Path(out_path, filename.stem + '_uncertainty.npy'), uncertainty)


def get_cutoff_point(model, val_set, out_path=None, batch_size = 16, cutoff_pts=np.arange(0.2, 0.8, 0.025), mc_iterations=None):
    dice_all = []
    img_set, mask_set = val_set
    results = model.predict(img_set, batch_size = batch_size)
    if mc_iterations:
        for iter in range(1,mc_iterations):
            iter_results = model.predict(img_set, batch_size = batch_size)
            for i in range(len(results)):
                results[i] += iter_results[i]
        for i in range(len(results)):
            results[i] /= mc_iterations

    for cutoff in cutoff_pts:
        dice = []
        for gt, pred in zip(mask_set, results):
            pred_img = (pred > cutoff).astype(int)
            dice.append(dice_coefficient(gt, pred_img))
        dice_all.append(np.mean(dice))

    cutoff_pts_list = np.array(cutoff_pts)
    dice_all = np.array(dice_all)
    argmax = np.argmax(dice_all)
    cutoff_pt = cutoff_pts_list[argmax]
    max_dice = dice_all[argmax]
    if out_path:
        df = pd.DataFrame({'cutoff_pts':cutoff_pts_list, 'dice': dice_all})
        df.to_pickle(Path(out_path, 'dice_cutoff.pkl')) # Save all values for later plot changes

        plt.rcParams.update({'font.size': 18})
        plt.figure()
        plt.plot((cutoff_pt, cutoff_pt),(0, max_dice), linestyle=':', linewidth=2, color='grey')
        plt.plot(cutoff_pts_list, dice)
        plt.annotate(f'{max_dice:.2f}', (cutoff_pts_list, max_dice), fontsize='x-small')
        plt.ylabel('Dice')
        plt.xlabel('Cutoff Point')
        plt.savefig(str(Path(out_path, 'cutoff.png')), bbox_inches='tight', format='png', dpi=200)

    return cutoff_pt








#def get_cutoff_point_(model, val_path, out_path, batch_size=16, patch_size=256, preprocessor=None):
#    tmp_dir = Path(out_path, 'cutoff_tmp', patches_only=False)
#
#    if 'bayes' in model.name:
#        if patches_only:
#            predict_patches_only(model, args.img_path, out_path, batch_size=args.batch_size, patch_size=options['patch_size'], cutoff=cutoff, preprocessor=preprocessor)
#        else:
#            predict_bayes(model, Path(val_path, 'images'), tmp_dir, batch_size=batch_size,
#                    patch_size=patch_size, preprocessor=preprocessor, cutoff=None)
#    else:
#        predict(model, Path(val_path, 'images'), tmp_dir, batch_size=batch_size,
#                patch_size=patch_size, preprocessor=preprocessor, cutoff=None)
#
#    # Read images into memory
#    imgs = []
#    gt_imgs = []
#    pred_imgs = []
#    for filename in Path(val_path, 'images').glob('*.png'):
#        imgs.append(io.imread(filename, as_gray=True))
#
#        if Path(val_path, 'masks', filename.stem + '_zones.png').exists():
#            gt_imgs.append(io.imread(Path(val_path, 'masks', filename.stem + '_zones.png'), as_gray=True))
#        else:
#            gt_imgs.append(io.imread(Path(val_path, 'masks', filename.stem + '.png'), as_gray=True))
#        if (Path(tmp_dir, filename.stem + '_pred.png')).exists():
#            pred_img = io.imread(Path(tmp_dir, filename.stem + '_pred.png'), as_gray=True)
#        elif Path(tmp_dir,filename.name).exists():
#            pred_img = io.imread(Path(tmp_dir,filename.name), as_gray=True) # Legacy before predictions got pred indentifier
#        pred_img = pred_img / 255
#        pred_imgs.append(pred_img)
#
#    # Try different cutoff points
#    dice = []
#    cutoff_range = np.arange(0.3, 0.725, 0.025)
#
#    for cutoff in cutoff_range:
#        pred_bin = [pred >= cutoff for pred in pred_imgs]
#        dice_mean = np.mean(evaluate_dice_only(imgs, gt_imgs, pred_bin))
#        dice.append(dice_mean)
#
#    dice = np.array(dice)
#    argmax = np.argmax(dice)
#
#    np.save(Path(out_path, 'dice_cutoff.npy'), dice)  # Save all values for later plot changes
#    max_dice = dice[argmax]
#    max_cutoff = cutoff_range[argmax]
#
#    plt.rcParams.update({'font.size': 18})
#    plt.figure()
#    plt.plot((max_cutoff, max_cutoff),(0, max_dice), linestyle=':', linewidth=2, color='grey')
#    plt.plot(cutoff_range, dice)
#    plt.annotate(f'{max_dice:.2f}', (max_cutoff, max_dice), fontsize='x-small')
#    plt.ylabel('Dice')
#    plt.xlabel('Cutoff Point')
#    plt.savefig(str(Path(out_path, 'cutoff.png')), bbox_inches='tight', format='png', dpi=200)
#
#    shutil.rmtree(tmp_dir, ignore_errors=True)
#
#    return max_cutoff

def predict_patches_only(model, img_path, out_path, batch_size=16, patch_size=256, cutoff=0.5, preprocessor=None, mc_iterations = 20, uncertainty_threshold=1e-3):
    if not Path(out_path).exists():
        Path(out_path).mkdir(parents=True)

    if not Path(out_path, 'images').exists():
        Path(out_path, 'images').mkdir()
    #if not Path(out_path, 'masks').exists():
    #    Path(out_path, 'masks').mkdir()
    #if not Path(out_path, 'uncertainty').exists():
    #    Path(out_path, 'uncertainty').mkdir()

    patches = []
    index = []
    for filename in Path(img_path).rglob('*.png'):
        #print(filename)
        img = io.imread(filename, as_gray=True)
        if preprocessor is not None:
            img = preprocessor.process(img)
        img = img / 255
        patches.append(img)
        index.append(filename.stem)
    for b_index in range((len(patches) // batch_size) +1):
        if (b_index+1 * batch_size < len(patches)):
            batch = np.array(patches[b_index * batch_size:(b_index+1)*batch_size])
        else:
            batch = np.array(patches[b_index * batch_size:])

        batch = np.reshape(batch,batch.shape+(1,))

        predictions = []
        for i in range(mc_iterations):
            prediction = model.predict(batch, batch_size=batch_size)
            predictions.append(prediction)
        predictions = np.array(predictions)
        patches_predicted = predictions.mean(axis=0)
        patches_uncertainty = predictions.var(axis=0)
        batch = np.reshape(batch, batch.shape[:-1])
        patches_predicted = patches_predicted.reshape(batch.shape)
        patches_uncertainty = patches_uncertainty.reshape(batch.shape)

        i = b_index * batch_size
        for patch, mask_predicted, uncertainty in zip(batch, patches_predicted, patches_uncertainty):
            patch = 255 * patch
            io.imsave(Path(out_path, index[i] + '.png'), patch.astype(np.uint8))


            if cutoff is not None:
                # thresholding to make binary mask
                mask_predicted[mask_predicted < cutoff] = 0
                mask_predicted[mask_predicted >= cutoff] = 255
            else:
                mask_predicted = 255 * mask_predicted

            io.imsave(Path(out_path, index[i] + '_pred.png'), mask_predicted.astype(np.uint8))

            uncertainty_img = (65535 * uncertainty).astype(np.uint16)
            io.imsave(Path(out_path, index[i] + '_uncertainty.png'), uncertainty_img, check_contrast=False )

            i += 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Glacier Front Segmentation Prediction')
    parser.add_argument('--model_path', type=str, help='Path containing trained model')
    parser.add_argument('--img_path', type=str, help='Path containing images to be segmented')
    parser.add_argument('--out_path', type=str, help='output path for predictions')
    parser.add_argument('--gt_path', type=str, help='Path containing the ground truth, necessary for evaluation_scripts')
    parser.add_argument('--batch_size', default=1, type=int, help='batch size (integer value)')
    parser.add_argument('--cutoff', type=float, help='cutoff point of binarisation')
    parser.add_argument('--patches_only', type=int, default=0)
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
        if args.patches_only:
            predict_patches_only(model, args.img_path, out_path, batch_size=args.batch_size, patch_size=options['patch_size'], cutoff=cutoff, preprocessor=preprocessor)
        else:
            predict_bayes(model, args.img_path, out_path, batch_size=args.batch_size, patch_size=options['patch_size'], cutoff=cutoff, preprocessor=preprocessor)
    else:
        predict(model, args.img_path, out_path, batch_size=args.batch_size, patch_size=options['patch_size'], cutoff=cutoff, preprocessor=preprocessor)

    if args.gt_path:
        evaluate(args.gt_path, out_path)

