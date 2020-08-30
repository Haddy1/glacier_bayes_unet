import skimage.io as io
from pathlib import Path
import seaborn as sns
from scipy.spatial import distance
import numpy as np
from sklearn.metrics import f1_score, recall_score
from utils import helper_functions
import pickle
import pandas as pd
from matplotlib import pyplot as plt
import tensorflow as tf
import json
from multiprocessing import Pool

#def evaluate_tf(img_path, gt_path, prediction_path):
#    img_set = tf.data.Dataset.list_files(str(img_path) + "/*.png")
#    img_set = img_set.take(-1)
#
#    if len(list(Path(gt_path).glob("*_zones.png"))) > 0:
#        gt_set = tf.data.Dataset.list_files(str(gt_path) + "/*_zones.png")
#    else:
#        gt_set = tf.data.Dataset.list_files(str(gt_path) + "/*.png")
#    gt_set = gt_set.take(-1)
#
#    if len(list(Path(prediction_path).glob("*_pred.png"))) > 0:
#        pred_set = tf.data.Dataset.list_files(str(prediction_path) + "/*_pred.png")
#    else:
#        pred_set = tf.data.Dataset.list_files(str(prediction_path) + "/*.png")
#    pred_set = pred_set.take(-1)





def evaluate(img_path, gt_path, prediction_path):

    #scores = pd.DataFrame(columns=['image', 'dice'])
    scores = {}
    scores['image'] = []
    scores['dice'] = []
    scores['euclidian'] = []
    scores['IOU'] = []
    scores['specificity'] = []
    scores['sensitivity'] = []

    for filename in Path(img_path).rglob('*.png'):

        gt_name = filename.name.partition('.')[0] + '_zones.png'
        if not Path(gt_name, gt_name).exists():
            gt_name = filename.name.partition('.')[0] + '.png'
        gt_img = io.imread(str(Path(gt_path,gt_name)), as_gray=True)
        if (Path(prediction_path, filename.stem + '_pred.png')).exists():
            pred_img = io.imread(Path(prediction_path, filename.stem + '_pred.png'), as_gray=True)
        elif Path(prediction_path,filename.name).exists():
            pred_img = io.imread(Path(prediction_path,filename.name), as_gray=True) # Legacy before predictions got pred indentifier
        else:
            print(str(Path(prediction_path,filename.name)) + " not found")
            continue


        gt = (gt_img > 200).astype(int)
        pred = (pred_img > 200).astype(int)

        gt_flat = gt.flatten()
        pred_flat = pred.flatten()


        scores['image'].append(filename.name)
        scores['dice'].append(helper_functions.dice_coefficient(gt_flat, pred_flat))
        scores['euclidian'].append(distance.euclidean(gt_flat, pred_flat))
        scores['IOU'].append(helper_functions.IOU(gt_flat, pred_flat))
        scores['specificity'].append(helper_functions.specificity(gt_flat, pred_flat))
        scores['sensitivity'].append(recall_score(gt_flat, pred_flat))

        if Path(prediction_path, filename.stem + '_uncertainty.png').exists():
            if not 'uncertainty' in scores:
                scores['uncertainty'] = []

            uncertainty_img = io.imread(Path(prediction_path, filename.stem + '_uncertainty.png'), as_gray=True)
            uncertainty = uncertainty_img / 65535
            scores['uncertainty'].append(uncertainty.mean())
            plt.figure()
            sns.heatmap(uncertainty, vmin=0, vmax=0.15, xticklabels=False, yticklabels=False)
            plt.savefig(Path(prediction_path, filename.stem + '_heatmap.png'), bbox_inches='tight', format='png', dpi=300)
            plt.close()



        diff = 255 * (gt != pred).astype(np.uint8)
        io.imsave(Path(prediction_path, filename.stem + '_diff.png'), diff, check_contrast=False)


    scores = pd.DataFrame.from_dict(scores)
    print(prediction_path)
    print('Dice\tIOU\tEucl\tSensitivity\tSpecificitiy')
    print(
                  str(np.mean(scores['dice'])) + '\t'
                  + str(np.mean(scores['IOU'])) + '\t'
                  + str(np.mean(scores['euclidian'])) + '\t'
                  + str(np.mean(scores['sensitivity'])) + '\t'
                  + str(np.mean(scores['specificity'])) + '\n')

    with open(str(Path(prediction_path, 'ReportOnModel.txt')), 'w') as f:
        f.write('Dice\tIOU\tEucl\tSensitivity\tSpecificitiy\n')
        f.write(
            str(np.mean(scores['dice'])) + '\t'
            + str(np.mean(scores['IOU'])) + '\t'
            + str(np.mean(scores['euclidian'])) + '\t'
            + str(np.mean(scores['sensitivity'])) + '\t'
            + str(np.mean(scores['specificity'])) + '\n')

    #pickle.dump(Perf, open(Path(prediction_path, 'scores.pkl'), 'wb'))
    scores.to_pickle(Path(prediction_path, 'scores.pkl'))
    return scores

#def evaluate_dice_only(imgs, gt_imgs, predictions):
#    dice = []
#    for img, gt_img, pred in zip(imgs, gt_imgs, predictions):
#        gt = (gt_img > 200).astype(int)
#        gt_flat = gt.flatten()
#        pred_flat = pred.flatten()
#        dice.append(helper_functions.dice_coefficient(gt_flat, pred_flat))
#    return dice


#def evaluate_dice_only_(img_path, gt_path, prediction_path):
#    dice = []
#    for filename in Path(img_path).rglob('*.png'):
#        gt_img = io.imread(Path(gt_path, filename.stem + '_zones.png'), as_gray=True)
#        if (Path(prediction_path, filename.stem + '_pred.png')).exists():
#            pred_img = io.imread(Path(prediction_path, filename.stem + '_pred.png'), as_gray=True)
#        elif Path(prediction_path,filename.name).exists():
#            pred_img = io.imread(Path(prediction_path,filename.name), as_gray=True) # Legacy before predictions got pred indentifier
#        else:
#            print(str(Path(prediction_path,filename.name)) + " not found")
#            continue
#
#
#        gt = (gt_img > 200).astype(int)
#        pred = (pred_img > 200).astype(int)
#
#        gt_flat = gt.flatten()
#        pred_flat = pred.flatten()
#        print(filename.name)
#        dice.append(helper_functions.dice_coefficient(gt_flat, pred_flat))
#    return dice

def plot_history(history, out_file, xlim=None, ylim=None, title=None):
    plt.figure()
    plt.rcParams.update({'font.size': 18})
    plt.plot( history['loss'], 'X-', label='training loss', linewidth=4.0)
    plt.plot(history['val_loss'], 'o-', label='val loss', linewidth=4.0)
    if xlim is not None:
        plt.xlim(xlim)
    if ylim is not None:
        plt.ylim(ylim)
    plt.xlabel('epoch')
    plt.ylabel('loss')
    if title:
        plt.title(title)
    plt.legend(loc='upper right')
    plt.minorticks_on()
    plt.grid(which='minor', linestyle='--')
    plt.savefig(out_file, bbox_inches='tight', format='png', dpi=200)
    plt.show()

def eval_uncertainty(file, out_file, vmin=0, vmax=0.2):
    uncertainty = np.load(file)
    plt.figure()
    sns.heatmap(uncertainty, vmin=vmin, vmax=vmax, xticklabels=False, yticklabels=False)
    plt.savefig(out_file, bbox_inches='tight', format='png', dpi=200)

if __name__ is '__main__':
    path = Path('/home/andreas/glacier-front-detection/output_pix2pix_front_only/Jakobshavn_front_only')

    test_path = Path('/home/andreas/glacier-front-detection/datasets/Jakobshavn_front_only/test')
    #history = pickle.load(open(next(path.glob('history*.pkl')), 'rb'))
    #history = pd.read_csv(Path(path,'model_1_history.csv'))
    #plot_history(history, Path(path, 'loss_plot.png') , xlim=(-10,30), ylim=(0,0.75), title='Set1')

    evaluate_tf(Path(test_path,'images'), Path(test_path,'masks'), path)

    #evaluate(Path(test_path, 'images'), Path(test_path, 'masks'), path)
    #test_path = Path('/home/andreas/glacier-front-detection/datasets/Jakobshavn_front_only/test')
    #evaluate(Path(test_path, 'images'), Path(test_path, 'masks'), '/home/andreas/glacier-front-detection/output_pix2pix_/output_Jakobshavn_pix2pix')
#    for d in Path('/home/andreas/glacier-front-detection/output_pix2pix_retrain').iterdir():
#        if d.is_dir():
#            test_path = Path('/home/andreas/glacier-front-detection/datasets/Jakobshavn_front_only/test')
#            evaluate(Path(test_path, 'images'), Path(test_path, 'masks'), d)
#            history = pickle.load(open(next(d.glob('history*.pkl')), 'rb'))
#            plot_history(history, Path(d, 'loss_plot.png')) # , xlim=(-10,130), ylim=(0,0.8))
