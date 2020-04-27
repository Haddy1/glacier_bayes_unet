import numpy as np
from pathlib import Path
import imageio
from matplotlib import pyplot as plt

#%%
import skimage.io as io
from pathlib import Path
import Amir_utils
from scipy.spatial import distance
import numpy as np
from sklearn.metrics import classification_report, f1_score, precision_score, recall_score
import helper_functions
# DICE
def own_dice(u,v):
    c_uv = np.equal(u,v).sum()
    c_u = (u > 0).sum()
    c_v = (v > 0).sum()
    return 2 * c_uv / (c_u + c_v)
DICE_all = []
EUCL_all = []
Specificity_all =[]
Sensitivity_all = []
F1_all = []
test_file_names = []
Perf = {}
test_path="data_256/test"
out_path = Path("output")

#for predict_path in out_path.glob('result*'):
predict_path = Path('output/combined_loss_0_0421-004910')
for filename in Path(test_path,'images').rglob('*.png'):
    gt_path = str(Path(test_path,'masks_zones'))
    gt_name = filename.name.partition('.')[0] + '_zones.png'
    gt = io.imread(str(Path(gt_path,gt_name)), as_gray=True)
    pred = io.imread(Path(predict_path,filename.name), as_gray=True)


    gt = (gt > 200).astype(int)
    pred = (pred > 200).astype(int)

    gt_flat = gt.flatten()

    pred_flat = pred.flatten()

    Specificity_all.append(helper_functions.specificity(gt_flat, pred_flat))
    Sensitivity_all.append(recall_score(gt_flat, pred_flat))
    F1_all.append(f1_score(gt_flat, pred_flat))

   # DICE_all.append(distance.dice(gt_bool.flatten(), pred_bool.flatten()))
    DICE_all.append(helper_functions.dice_coefficient(gt_flat, pred_flat))
    DICE_avg = np.mean(DICE_all)
    EUCL_all.append(distance.euclidean(gt_flat, pred_flat))
    EUCL_avg = np.mean(EUCL_all)
    test_file_names.append(filename.name)


Perf['Specificity_all'] = Specificity_all
Perf['Specificity_avg'] = np.mean(Specificity_all)
Perf['Sensitivity_all'] = Sensitivity_all
Perf['Sensitivity_avg'] = np.mean(Sensitivity_all)
Perf['F1_score_all'] = F1_all
Perf['F1_score_avg'] = np.mean(F1_all)
Perf['DICE_all'] = DICE_all
Perf['DICE_avg'] = DICE_avg
Perf['EUCL_all'] = EUCL_all
Perf['EUCL_avg'] = EUCL_avg
Perf['test_file_names'] = test_file_names
print(predict_path)
print('Dice\tEuclidian')
print(str(Perf['DICE_avg']) + '\t'
        + str(Perf['EUCL_avg']) + '\n')
print('Sensitivity\tSpecificitiy\tf1_score')
print(str(Perf['Sensitivity_avg']) + '\t'
        + str(Perf['Specificity_avg']) + '\t'
        + str(Perf['F1_score_avg']) + '\n')

with open(str(Path(predict_path , 'ReportOnModel.txt')), 'w') as f:
    f.write('Dice\tEuclidian\n')
    f.write(str(Perf['DICE_avg']) + '\t'
            + str(Perf['EUCL_avg']) + '\n')
    f.write('Sensitivity\tSpecificitiy\tf1_score\n')
    f.write(str(Perf['Sensitivity_avg']) + '\t'
            + str(Perf['Specificity_avg']) + '\t'
            + str(Perf['F1_score_avg']) + '\n')

