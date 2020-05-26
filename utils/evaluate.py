import skimage.io as io
from pathlib import Path
from scipy.spatial import distance
import numpy as np
from sklearn.metrics import f1_score, recall_score
from utils import helper_functions
import pickle
import pandas as pd
from matplotlib import pyplot as plt
import json

def evaluate(test_path, prediction_path):

    #scores = pd.DataFrame(columns=['image', 'dice'])
    scores = {}
    scores['image'] = []
    scores['dice'] = []
    scores['euclidian'] = []
    scores['IOU'] = []
    scores['specificity'] = []
    scores['sensitivity'] = []

    for filename in Path(test_path,'images').rglob('*.png'):

        gt_path = str(Path(test_path,'masks'))
        gt_name = filename.name.partition('.')[0] + '_zones.png'
        gt = io.imread(str(Path(gt_path,gt_name)), as_gray=True)
        pred = io.imread(Path(prediction_path,filename.name), as_gray=True)


        gt = (gt > 200).astype(int)
        pred = (pred > 200).astype(int)

        gt_flat = gt.flatten()
        pred_flat = pred.flatten()


        scores['image'].append(filename)
        scores['dice'].append(helper_functions.dice_coefficient(gt_flat, pred_flat))
        scores['euclidian'].append(distance.euclidean(gt_flat, pred_flat))
        scores['IOU'].append(helper_functions.IOU(gt_flat, pred_flat))
        scores['specificity'].append(helper_functions.specificity(gt_flat, pred_flat))
        scores['sensitivity'].append(recall_score(gt_flat, pred_flat))


    scores = pd.DataFrame(scores)
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

def evaluate_dice_only(test_path, prediction_path):
    dice = []
    for filename in Path(test_path,'images').rglob('*.png'):
        gt_path = str(Path(test_path,'masks'))
        gt_name = filename.name.partition('.')[0] + '_zones.png'
        gt = io.imread(str(Path(gt_path,gt_name)), as_gray=True)
        pred = io.imread(Path(prediction_path,filename.name), as_gray=True)


        gt = (gt > 200).astype(int)
        pred = (pred > 200).astype(int)

        gt_flat = gt.flatten()
        pred_flat = pred.flatten()
        dice.append(helper_functions.dice_coefficient(gt_flat, pred_flat))
    return np.mean(dice)

def plot_history(history, out_file, xlim=None, ylim=None):
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
    plt.legend(loc='upper right')
    plt.minorticks_on()
    plt.grid(which='minor', linestyle='--')
    plt.savefig(out_file, bbox_inches='tight', format='png', dpi=200)
    plt.show()

if __name__ is '__main__':
    for d in Path('/home/andreas/glacier-front-detection/output_filter').iterdir():
        if d.is_dir():
            evaluate(Path('/home/andreas/glacier-front-detection/front_detection_dataset/test'), d)
            history = pickle.load(open(next(d.glob('history*.pkl')), 'rb'))
            plot_history(history, Path(d, 'loss_plot.png') , xlim=(-10,250), ylim=(0,1.0))
