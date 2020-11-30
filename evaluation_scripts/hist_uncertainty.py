from pathlib import Path
import numpy as np
from matplotlib.ticker import FormatStrFormatter
dir = Path('test_all_drop')
import json
test_masks = Path('front_detection_dataset/test')
import matplotlib.pyplot as plt
import seaborn as sns
from layers.BayesDropout import  BayesDropout
import cv2
import skimage.io as io
from utils import helper_functions
from utils.evaluate import evaluate
from tensorflow.keras.models import load_model
from predict import predict_bayes, get_cutoff_point
import utils.evaluate
import math

helper_functions.set_font_size()
uncert_path = Path("/home/andreas/glacier-front-detection/datasets/front_detection_dataset/train/uncertainty")
uncert = None
for f in uncert_path.glob("*.png"):
    if uncert is None:
        uncert = io.imread(f).flatten() / 65535
    else:
        uncert = np.concatenate((uncert, io.imread(f).flatten() / 65535), axis=0)


hist, edges = np.histogram(uncert)
np.save("hist.npy", (hist, edges))
#hist, edges = np.load("evaluation_scripts/hist.npy", allow_pickle=True)
fig, ax = plt.subplots()
#uncert.sort()
#print(int(math.ceil(0.9*len(uncert))))
#threshold = uncert[int(math.ceil(0.9*len(uncert)))]
#print(threshold)
diff = edges[1] - edges[0]
ax.bar(edges[:-1] + 0.5*diff , hist, width= 0.8*diff)
ax.set_xticks(edges)
ax.set_xlabel("Uncertainty")
ax.set_ylabel("Pixel Count")
ax.set_yscale('log')
ax.xaxis.set_major_formatter(FormatStrFormatter('%.3f'))
ax.grid(True, axis='x')
plt.show()

fig.savefig(str(Path("/home/andreas/thesis/reports/uncertainty_training/imgs", 'cutoff.png')), bbox_inches='tight', format='png', dpi=200)
