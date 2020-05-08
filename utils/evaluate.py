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

        gt_path = str(Path(test_path,'masks_zones'))
        gt_name = filename.name.partition('.')[0] + '_zones.png'
        gt = io.imread(str(Path(gt_path,gt_name)), as_gray=True)
        pred = io.imread(Path(prediction_path,filename.name), as_gray=True)


        gt = (gt > 200).astype(int)
        pred = (pred > 200).astype(int)

        gt_flat = gt.flatten()
        pred_flat = pred.flatten()


        scores['image'].append(filename)
        scores['dice'].append(f1_score(gt_flat, pred_flat))
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

if __name__ is '__main__':
    for d in Path('/home/andreas/glacier-front-detection/output_combined').iterdir():
        if d.is_dir():
            evaluate(Path('/home/andreas/glacier-front-detection/front_detection_dataset/test'), d)
