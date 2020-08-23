from skimage import io
from pathlib import Path
import numpy as np
import cv2
import pandas as pd
import json
import matplotlib.pyplot as plt
from shutil import copy

out = Path('/home/andreas/thesis/reports/pix2pix')
plt.rcParams.update({'font.size': 18})
path = Path('/home/andreas/glacier-front-detection/output_autotrain')

#scores = pd.read_pickle('../output_bayes/bayes/scores.pkl')
#train_imgs = json.load(open('../output_bayes/bayes/train_image_list.json', 'r')).keys()
#n_train_imgs = [len(train_imgs)]
#iterations = [0]
#dice = [scores['dice'].mean()]
#scores = pd.read_pickle(Path(path, 'data_split/output/scores.pkl'))
#train_imgs = json.load(open(Path(path, 'data_split/train_image_list.json'), 'r')).keys()
#n_train_imgs = [len(train_imgs)]
#dice.append(scores['dice'].mean())
#iterations.append(0.5)
iterations = []
dice = []
n_train_imgs = []
for score_file in path.rglob('scores.pkl'):
    dirname = score_file.parent.parent.stem
    if dirname == 'data_split':
        continue
    i = int(dirname)
    if i > 0:

        n_train_imgs.append(len(list(Path(path,str(i-1), 'train/patches/images').glob('*.png'))))
    if i == 0:
        n_train_imgs.append(0)
    #train_imgs = json.load(open(Path(score_file.parent.parent, 'train_image_list.json'), 'r')).keys()
    #n_train_imgs.append(len(train_imgs))

    iterations.append(i)
    #scores = pd.read_pickle(score_file)
    with open(Path(score_file.parent, 'ReportOnModel.txt'), 'r') as f:
        scores = (f.readlines()[1]).split()
        dice_value = float(scores[0])


    #dice.append(scores['dice'].mean())
    dice.append(dice_value)

    copy(Path(score_file.parent.parent, 'loss_plot.png'), Path(out, 'imgs', 'autotrain_loss_plot' + str(i) + '.png'))


iterations = np.array(iterations)
dice = np.array(dice)
n_train_imgs = np.array(n_train_imgs)

argsort = iterations.argsort()
dice = dice[argsort]
n_train_imgs = n_train_imgs[argsort]

fig, ax1 = plt.subplots()

color = 'tab:red'
ax1.set_xlabel('Train Iteration')
ax1.set_ylabel('Dice Coefficient', color =color)
ax1.plot(dice, color=color)
ax1.tick_params(axis='y', labelcolor = color)

ax2 = ax1.twinx()
color = 'tab:blue'
ax2.set_ylabel('Nr Training Imgs', color=color)
ax2.plot(n_train_imgs, color=color)
ax2.tick_params(axis='y', labelcolor=color)
#plt.savefig()
plt.savefig(str(Path(out, 'imgs', 'autotrain.png')), bbox_inches='tight', format='png', dpi=200)
plt.show()

print(n_train_imgs.sum())