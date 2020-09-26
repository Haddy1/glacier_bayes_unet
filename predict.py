import sys
import json
import seaborn as sns
from pathlib import Path
from tensorflow.keras.models import load_model
from utils.data import *
import tensorflow as tf
import pickle
import argparse
from loss_functions import *
from tensorflow.keras.losses import binary_crossentropy
from layers.BayesDropout import  BayesDropout
from preprocessing.preprocessor import Preprocessor
from preprocessing import filter
import numpy as np
import skimage.io as io
import cv2
from utils.metrics import dice_coefficient, dice_coefficient_tf, dice_coefficient_cutoff
from utils.evaluate import evaluate#, evaluate_dice_only
from preprocessing.image_patches import extract_grayscale_patches, reconstruct_from_grayscale_patches
from preprocessing.data_generator import process_imgs
from utils.helper_functions import nat_sort
import matplotlib.pyplot as plt
import pandas as pd
from multiprocessing import Pool
from functools import partial
import shutil

def predict(model, img_path, out_path, uncert_path=None, uncert_threshold=None, batch_size=16, patch_size=256, cutoff=0.5, preprocessor=None):
    if not Path(out_path).exists():
        Path(out_path).mkdir(parents=True)

    for filename in Path(img_path).rglob('*.png'):
        print(filename)
        img = io.imread(filename, as_gray=True)
        img = img / 255

        if preprocessor is not None:
            img = preprocessor.process(img)
        img_pad = cv2.copyMakeBorder(img, 0, (patch_size - img.shape[0]) % patch_size, 0, (patch_size - img.shape[1]) % patch_size, cv2.BORDER_CONSTANT)
        p_img, i_img = extract_grayscale_patches(img_pad, (patch_size, patch_size), stride = (patch_size, patch_size))
        p_img = np.reshape(p_img,p_img.shape+(1,))

        if uncert_path is not None:
            if Path(uncert_path, filename.stem + '_uncertainty.png').exists():
                uncert = io.imread(Path(uncert_path, filename.stem + '_uncertainty.png'), as_gray=True)
            else:
                uncert = io.imread(Path(uncert_path, filename.stem + '.png'), as_gray=True)

            if preprocessor is not None:
                uncert = preprocessor.process(uncert)

            if uncert_threshold is not None:
                uncert = (uncert >= uncert_threshold).astype(float)
            else:
                uncert = uncert / 65535
            uncert_pad = cv2.copyMakeBorder(uncert, 0, (patch_size - uncert.shape[0]) % patch_size, 0, (patch_size - uncert.shape[1]) % patch_size, cv2.BORDER_CONSTANT)
            p_uncert, i_uncert = extract_grayscale_patches(uncert_pad, (patch_size, patch_size), stride = (patch_size, patch_size))
            p_uncert = np.reshape(p_uncert,p_uncert.shape+(1,))
            p_img = np.array([np.concatenate((img, uncert), axis=2) for img, uncert in zip(p_img, p_uncert)])

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

def predict_bayes(model, img_path, out_path, uncert_path=None, uncert_threshold=None, batch_size=16, patch_size=256, cutoff=0.5, preprocessor=None, mc_iterations = 20):
    if not Path(out_path).exists():
        Path(out_path).mkdir(parents=True)

    for filename in Path(img_path).rglob('*.png'):
        #print(filename)
        img = io.imread(filename, as_gray=True)
        img = img / 255
        if preprocessor is not None:
            img = preprocessor.process(img)
        img_pad = cv2.copyMakeBorder(img, 0, (patch_size - img.shape[0]) % patch_size, 0, (patch_size - img.shape[1]) % patch_size, cv2.BORDER_CONSTANT)
        p_img, i_img = extract_grayscale_patches(img_pad, (patch_size, patch_size), stride = (patch_size, patch_size))
        p_img = np.reshape(p_img,p_img.shape+(1,))

        if uncert_path is not None:
            if Path(uncert_path, filename.stem + '_uncertainty.png').exists():
                uncert = io.imread(Path(uncert_path, filename.stem + '_uncertainty.png'), as_gray=True)
            else:
                uncert = io.imread(Path(uncert_path, filename.stem + '.png'), as_gray=True)
            if preprocessor is not None:
                uncert = preprocessor.process(uncert)

            uncert = uncert / 65535
            if uncert_threshold is not None:
                uncert = (uncert >= uncert_threshold).astype(float)
            uncert_pad = cv2.copyMakeBorder(uncert, 0, (patch_size - uncert.shape[0]) % patch_size, 0, (patch_size - uncert.shape[1]) % patch_size, cv2.BORDER_CONSTANT)
            p_uncert, i_uncert = extract_grayscale_patches(uncert_pad, (patch_size, patch_size), stride = (patch_size, patch_size))
            p_uncert = np.reshape(p_uncert,p_uncert.shape+(1,))
            p_img = np.array([np.concatenate((img, uncert), axis=2) for img, uncert in zip(p_img, p_uncert)])


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
        if uncert_threshold is not None:
            confidence_img[uncertainty >= uncert_threshold, :] = np.array([255, 0,0])   # make  uncertain pixels red
        io.imsave(Path(out_path, filename.stem + '_confidence.png'), confidence_img)
        #np.save(Path(out_path, filename.stem + '_uncertainty.npy'), uncertainty)

def get_cutoff_point(model, val_path, out_path, batch_size=16, patch_size=256, cutoff_pts=np.arange(0.2, 0.8, 0.025), preprocessor=None, mc_iterations=20, uncert_threshold=None):
    if not Path(out_path).exists():
        Path(out_path).mkdir(parents=True)
    img_path = Path(val_path, 'images')
    uncert_path = Path(val_path, 'uncertainty')
    gt_path = Path(val_path, 'masks')
    dice_all = np.zeros(len(cutoff_pts))
    n_img = len(list(Path(img_path).rglob('*.png')))
    p = Pool()
    for filename in Path(img_path).rglob('*.png'):
        img = io.imread(filename, as_gray=True)
        img = img / 255
        if preprocessor is not None:
            img = preprocessor.process(img)
        img_pad = cv2.copyMakeBorder(img, 0, (patch_size - img.shape[0]) % patch_size, 0, (patch_size - img.shape[1]) % patch_size, cv2.BORDER_CONSTANT)
        p_img, i_img = extract_grayscale_patches(img_pad, (patch_size, patch_size), stride = (patch_size, patch_size))
        p_img = np.reshape(p_img,p_img.shape+(1,))

        if model.input_shape[3] == 2 and uncert_path is not None:
            if Path(uncert_path, filename.stem + '_uncertainty.png').exists():
                uncert = io.imread(Path(uncert_path, filename.stem + '_uncertainty.png'), as_gray=True)
            else:
                uncert = io.imread(Path(uncert_path, filename.stem + '.png'), as_gray=True)
            if preprocessor is not None:
                uncert = preprocessor.process(uncert)

            uncert = uncert / 65535
            if uncert_threshold is not None:
                uncert = (uncert >= uncert_threshold).astype(float)
            uncert_pad = cv2.copyMakeBorder(uncert, 0, (patch_size - uncert.shape[0]) % patch_size, 0, (patch_size - uncert.shape[1]) % patch_size, cv2.BORDER_CONSTANT)
            p_uncert, i_uncert = extract_grayscale_patches(uncert_pad, (patch_size, patch_size), stride = (patch_size, patch_size))
            p_uncert = np.reshape(p_uncert,p_uncert.shape+(1,))
            p_img = np.array([np.concatenate((img, uncert), axis=2) for img, uncert in zip(p_img, p_uncert)])


        predictions = []
        for i in range(mc_iterations):
            prediction = model.predict(p_img, batch_size=batch_size)
            predictions.append(prediction)
        predictions = np.array(predictions)


        p_img_predicted = predictions.mean(axis=0)
        p_img_predicted = np.reshape(p_img_predicted,p_img_predicted.shape[:-1])
        mask_predicted = reconstruct_from_grayscale_patches(p_img_predicted,i_img)[0]
        mask_predicted = mask_predicted[:img.shape[0], :img.shape[1]]

        pred_flat = mask_predicted.flatten()

        gt_img = io.imread(Path(gt_path, filename.stem + '_zones.png'))
        gt = (gt_img > 200).astype(int)
        gt_flat = gt.flatten()
        dice_eval = partial(dice_coefficient_cutoff, gt_flat, pred_flat)
        dice_all += np.array(p.map(dice_eval, cutoff_pts))




    cutoff_pts_list = np.array(cutoff_pts)
    dice_all = np.array(dice_all) / n_img
    argmax = np.argmax(dice_all)
    cutoff_pt = cutoff_pts_list[argmax]
    max_dice = dice_all[argmax]
    df = pd.DataFrame({'cutoff_pts':cutoff_pts_list, 'dice': dice_all})
    df.to_pickle(Path(out_path, 'dice_cutoff.pkl')) # Save all values for later plot changes

    plt.rcParams.update({'font.size': 18})
    plt.figure()
    plt.plot((cutoff_pt, cutoff_pt),(0, max_dice), linestyle=':', linewidth=2, color='grey')
    plt.plot(cutoff_pts_list, dice_all)
    plt.annotate(f'{max_dice:.2f}', (cutoff_pt, max_dice), fontsize='x-small')
    plt.ylabel('Dice')
    plt.xlabel('Cutoff Point')
    plt.savefig(str(Path(out_path, 'cutoff.png')), bbox_inches='tight', format='png', dpi=200)

    return cutoff_pt, dice_all


def eval_dice(gt, pred, cutoff):
    assert gt.shape == pred.shape
    pred_bin = pred > cutoff
    dice.append(dice_coefficient_tf(gt, pred_bin))


def get_cutoff_point_(model, val_path, out_path, batch_size=16, patch_size=256, cutoff_pts=np.arange(0.2, 0.8, 0.025), preprocessor=None, mc_iterations=20, uncert_threshold=None):
    tmp_dir = Path(out_path, 'cutoff_tmp', patches_only=False)

    index_data = process_imgs(Path(val_path, 'images'), Path(tmp_dir, 'images'), patch_size=patch_size, preprocessor=preprocessor)
    if model.input_shape[3] == 2:
        process_imgs(Path(val_path, 'uncertainty'), Path(tmp_dir, 'uncertainty'), patch_size=patch_size, preprocessor=preprocessor)
        img_generator = imgGeneratorUncertainty(batch_size, tmp_dir, 'images', 'uncertainty', target_size=(patch_size, patch_size), shuffle=False, uncert_threshold=uncert_threshold)
    else:
        img_generator = imgGenerator(batch_size, tmp_dir, 'images', target_size=(patch_size, patch_size), shuffle=False)
    max_steps = np.ceil(len(list(Path(tmp_dir, 'images').glob("*.png"))) / batch_size)
    results = model.predict(img_generator, steps=max_steps)
    if 'bayes' in model.name and mc_iterations:
        for iter in range(1,mc_iterations):
            results += model.predict(img_generator, steps=max_steps)
        results /= mc_iterations
    results = results.squeeze()
    np.save('cache.npy', results)
    results = np.load('cache.npy')

    p = Pool()
    dice_all = np.zeros(len(cutoff_pts))
    # Restore full images from patches
    for img_name, metadata in index_data.items():
        patches = results[metadata['indices']]
        origin = np.array(metadata['origin'])
        if  patches.shape[0] > 1:
            pred, _ = reconstruct_from_grayscale_patches(patches, origin)
        else:
            pred = patches
        pred = pred.squeeze()
        # Restore original shape
        shape = metadata['img_shape']
        pred= pred[:shape[0], :shape[1]]
        #pred_imgs.append(pred_img[:shape[0], :shape[1]])

        if Path(val_path, 'masks', img_name + '_zones.png').exists():
            gt_img = io.imread(Path(val_path, 'masks', img_name + '_zones.png'), as_gray=True)
        else:
            gt_img = io.imread(Path(val_path, 'masks', img_name + '.png'), as_gray=True)

        gt = (gt_img > 200).astype(int)

        pred_flat = pred.flatten()
        gt_flat = gt.flatten()

        dice_eval = partial(dice_coefficient_cutoff, gt_flat, pred_flat)
        dice_all += np.array(p.map(dice_eval, cutoff_pts))


    cutoff_pts_list = np.array(cutoff_pts)
    dice_all = np.array(dice_all) / len(index_data)
    print(dice_all)
    argmax = np.argmax(dice_all)
    cutoff_pt = cutoff_pts_list[argmax]
    max_dice = dice_all[argmax]
    df = pd.DataFrame({'cutoff_pts':cutoff_pts_list, 'dice': dice_all})
    df.to_pickle(Path(out_path, 'dice_cutoff.pkl')) # Save all values for later plot changes

    plt.rcParams.update({'font.size': 18})
    plt.figure()
    plt.plot((cutoff_pt, cutoff_pt),(0, max_dice), linestyle=':', linewidth=2, color='grey')
    plt.plot(cutoff_pts_list, dice_all)
    plt.annotate(f'{max_dice:.2f}', (cutoff_pt, max_dice), fontsize='x-small')
    plt.ylabel('Dice')
    plt.xlabel('Cutoff Point')
    plt.savefig(str(Path(out_path, 'cutoff.png')), bbox_inches='tight', format='png', dpi=200)



    #shutil.rmtree(tmp_dir, ignore_errors=True)

    return cutoff_pt, dice_all


def predict_patches_only(model, img_path, out_path, uncert_path, uncert_threshold=None, batch_size=16, patch_size=256, cutoff=0.5, preprocessor=None, mc_iterations = 20):
    if not Path(out_path).exists():
        Path(out_path).mkdir(parents=True)

    patches = []
    index = []
    for filename in Path(img_path).rglob('*.png'):
        #print(filename)
        img = io.imread(filename, as_gray=True)
        img = img / 255
        if uncert_path is not None:
            if Path(uncert_path, filename.stem + '_uncertainty.png').exists():
                uncert = io.imread(Path(uncert_path, filename.stem + '_uncertainty.png'), as_gray=True)
            else:
                uncert = io.imread(Path(uncert_path, filename.stem + '.png'), as_gray=True)

            uncert = uncert / 65535
            if uncert_threshold is not None:
                uncert = (uncert >= uncert_threshold).astype(float)
            img = np.stack((img, uncert), axis=-1)
        if preprocessor is not None:
            img = preprocessor.process(img)
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
    parser.add_argument('--uncert_path', type=str, help='Path containing uncertainty images')
    parser.add_argument('--uncert_threshold', type=float, help='Threshold for uncertainty binarisation')
    parser.add_argument('--gt_path', type=str, help='Path containing the ground truth, necessary for evaluation_scripts')
    parser.add_argument('--batch_size', default=1, type=int, help='batch size (integer value)')
    parser.add_argument('--cutoff', type=float, help='cutoff point of binarisation')
    parser.add_argument('--patches_only', type=int, default=0)
    args = parser.parse_args()

    if not Path(args.model_path).exists():
        print(args.model_path + " does not exist")
    if not Path(args.img_path).exists():
        print(args.img_path + " does not exist")
    if args.gt_path:
        if not Path(args.gt_path).exists():
            print(args.gt_path + " does not exist")
    if args.uncert_path:
        if not Path(args.uncert_path).exists():
            print(args.uncert_path + " does not exist")
        uncert_path = Path(args.uncert_path)
    else:
        uncert_path = None

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
            predict_patches_only(model,
                                 args.img_path,
                                 out_path, uncert_path,
                                 batch_size=args.batch_size,
                                 patch_size=options['patch_size'],
                                 cutoff=cutoff,
                                 preprocessor=preprocessor,
                                 uncert_threshold=args.uncert_threshold)
        else:
            predict_bayes(model,
                          args.img_path,
                          out_path,
                          uncert_path,
                          batch_size=args.batch_size,
                          patch_size=options['patch_size'],
                          cutoff=cutoff,
                          preprocessor=preprocessor,
                          uncert_threshold=args.uncert_threshold)
    else:
        predict(model,
                args.img_path,
                out_path,uncert_path,
                batch_size=args.batch_size,
                patch_size=options['patch_size'],
                cutoff=cutoff,
                preprocessor=preprocessor)

    if args.gt_path:
        evaluate(args.gt_path, out_path)

