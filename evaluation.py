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
predict_path = Path("output/results_0412-235548_binary_crossentropy")

for filename in Path(test_path,'images').rglob('*.png'):
    gt_path = str(Path(test_path,'masks_zones'))
    gt_name = filename.name.partition('.')[0] + '_zones.png'
    gt = io.imread(str(Path(gt_path,gt_name)), as_gray=True)
    img_mask_predicted_recons_unpad_norm = io.imread(Path(predict_path,filename.name), as_gray=True)

    gt_bool = gt/gt.max()
    gt_flat = gt_bool.flatten()

    pred_bool = img_mask_predicted_recons_unpad_norm / img_mask_predicted_recons_unpad_norm.max()
    mask_predicted_flat = pred_bool.flatten()

    Specificity_all.append(helper_functions.specificity(gt_flat, mask_predicted_flat))
    Sensitivity_all.append(recall_score(gt_flat, mask_predicted_flat))
    F1_all.append(f1_score(gt_flat, mask_predicted_flat))

   # DICE_all.append(distance.dice(gt_bool.flatten(), pred_bool.flatten()))
    DICE_all.append(distance.dice(gt_flat, mask_predicted_flat))
    DICE_avg = np.mean(DICE_all)
    EUCL_all.append(distance.euclidean(gt_flat, mask_predicted_flat))
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
print('Dice\tEuclidian')
print(str(Perf['DICE_avg']) + '\t'
        + str(Perf['EUCL_avg']) + '\n')
print('Sensitivity\tSpecificitiy\tf1_score')
print(str(Perf['Sensitivity_avg']) + '\t'
        + str(Perf['Specificity_avg']) + '\t'
        + str(Perf['F1_score_avg']) + '\n')

#%%
#import imageio
#path = Path("data_256/test/masks_predicted_focal-loss_alpha-1_gamma-0_200406-222725")
#data = np.load(path.joinpath("Performance.npz"), allow_pickle=True)['arr_0']
##test_file_names = data['test_file_names']
#with open(path.joinpath("ReportOnModel.txt")) as f:
#    print(f.read())
#loss_plot = imageio.imread(path.joinpath("loss_plot.png"))
#plt.imshow(loss_plot)
#plt.show()

#%%
#import imageio
#path = Path("data_256/test/masks_predicted_focal-loss_alpha-0.5_gamma-0.5_200407-005722")
#data = np.load(path.joinpath("Performance.npz"), allow_pickle=True)['arr_0']
##test_file_names = data['test_file_names']
#with open(path.joinpath("ReportOnModel.txt")) as f:
#    print(f.read())
#loss_plot = imageio.imread(path.joinpath("loss_plot.png"))
#plt.imshow(loss_plot)
#plt.show()

