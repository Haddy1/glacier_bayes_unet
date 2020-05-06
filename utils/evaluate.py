import skimage.io as io
from pathlib import Path
from scipy.spatial import distance
import numpy as np
from sklearn.metrics import f1_score, recall_score
from utils import helper_functions
import pickle
from matplotlib import pyplot as plt
import json

def evaluate(test_path, prediction_path, model_history=None):
    DICE_all = []
    EUCL_all = []
    IOU_all = []
    Specificity_all =[]
    Sensitivity_all = []
    test_file_names = []
    Perf = {}
    for filename in Path(test_path,'images').rglob('*.png'):
        gt_path = str(Path(test_path,'masks_zones'))
        gt_name = filename.name.partition('.')[0] + '_zones.png'
        gt = io.imread(str(Path(gt_path,gt_name)), as_gray=True)
        pred = io.imread(Path(prediction_path,filename.name), as_gray=True)


        gt = (gt > 200).astype(int)
        pred = (pred > 200).astype(int)

        gt_flat = gt.flatten()

        pred_flat = pred.flatten()

        Specificity_all.append(helper_functions.specificity(gt_flat, pred_flat))
        Sensitivity_all.append(recall_score(gt_flat, pred_flat))

        DICE_all.append(helper_functions.dice_coefficient(gt_flat, pred_flat))
        DICE_avg = np.mean(DICE_all)
        EUCL_all.append(distance.euclidean(gt_flat, pred_flat))
        EUCL_avg = np.mean(EUCL_all)
        IOU_all.append(helper_functions.IOU(gt_flat, pred_flat))
        test_file_names.append(filename.name)


    Perf['Specificity_all'] = Specificity_all
    Perf['Specificity_avg'] = np.mean(Specificity_all)
    Perf['Specificity_var'] = np.var(Specificity_all)
    Perf['Sensitivity_all'] = Sensitivity_all
    Perf['Sensitivity_avg'] = np.mean(Sensitivity_all)
    Perf['Sensitivity_var'] = np.var(Sensitivity_all)
    Perf['DICE_all'] = DICE_all
    Perf['DICE_avg'] = DICE_avg
    Perf['DICE_avg'] = np.var(DICE_all)
    Perf['IOU_all'] = IOU_all
    Perf['IOU_avg'] = np.mean(IOU_all)
    Perf['IOU_avg'] = np.var(IOU_all)
    Perf['EUCL_all'] = EUCL_all
    Perf['EUCL_avg'] = EUCL_avg
    Perf['EUCL_avg'] = np.var(EUCL_all)
    Perf['test_file_names'] = test_file_names
    print(prediction_path)
    print('Dice\tIOU\tEucl\tSensitivity\tSpecificitiy')
    print(
                  str(Perf['DICE_avg']) + '\t'
                  + str(Perf['IOU_avg']) + '\t'
                  + str(Perf['EUCL_avg']) + '\t'
                  + str(Perf['Sensitivity_avg']) + '\t'
            + str(Perf['Specificity_avg']) + '\n')

    with open(str(Path(prediction_path, 'ReportOnModel.txt')), 'w') as f:
        f.write('Dice\tIOU\tEucl\tSensitivity\tSpecificitiy\n')
        f.write(
                str(Perf['DICE_avg']) + '\t'
                + str(Perf['IOU_avg']) + '\t'
                + str(Perf['EUCL_avg']) + '\t'
                + str(Perf['Sensitivity_avg']) + '\t'
                + str(Perf['Specificity_avg']) + '\n')

    pickle.dump(Perf, open(Path(prediction_path, 'scores.pkl'), 'wb'))

if __name__ is '__main__':
    for d in Path('/home/andreas/glacier-front-detection/output_256').iterdir():
        if d.is_dir():
            evaluate(Path('/home/andreas/glacier-front-detection/front_detection_dataset/test'), d)
