import re
import sys
from pathlib import Path
import json
from shutil import copy
from matplotlib.ticker import StrMethodFormatter, FormatStrFormatter, FuncFormatter
import pandas as pd
import numpy as np
from matplotlib import  pyplot as plt
import matplotlib
import seaborn as sns
from skimage import io
import os
os.chdir('../')
plt.rcParams.update({'font.size': 18})

identifier = 'Jakobshavn_front_only'
path = Path('/home/andreas/glacier-front-detection/output_pix2pix_front_only/' + identifier)
out = Path('/home/andreas/thesis/reports/pix2pix')
if not out.exists():
    out.mkdir(parents=True)
if not Path(out, 'imgs').exists():
    Path(out, 'imgs').mkdir()

#imgs = ['2009-08-04_TSX_6_1', '2014-07-25_TSX_6_1']
imgs = ['0', '66', '90', '95', '111', '743']
imgs_new = ['667', '354', '109', '92', '297', '8']
#for i in ['2006-02-23_RSAT_20_3', '1993-08-29_ERS_20_5', '2011-09-21_TDX_5_1']:
for i in range(len(imgs)):
    copy(Path('/home/andreas/glacier-front-detection/datasets/Jakobshavn_front_only/test/patches/images/', imgs_new[i] + '.png'), Path(out, 'imgs', imgs[i] + '.png'))
    copy(Path(path, imgs_new[i] + '_pred.png'), Path(out, 'imgs', identifier + "_" + imgs[i] + '_pred.png'))
    #copy(Path(path, i + '_diff.png'), Path(out, 'imgs', i + '_diff.png'))

    uncert = io.imread(Path(path, imgs_new[i] + '_uncertainty.png'), as_gray=True) / 65535
    uncert_norm = (uncert / uncert.max()) * 255
    io.imsave(Path(out, 'imgs', identifier + "_" + imgs[i] + '_uncert.png'), uncert_norm)



frames = []
labels = []
scores = pd.read_pickle(Path(path, 'scores.pkl'))

label = path.name.capitalize()


if Path(path, 'loss_plot.png').exists():
    copy(Path(path, 'loss_plot.png'), Path(out, 'imgs', 'loss' + label + '.png'))

labels = sorted(labels, reverse=False)
#%%

def fperc(x):
    return str(round(x * 100,2))


with open(Path(out,'results_' +identifier + '.tex'), 'w') as f:
    for column in scores.columns:
        if column == 'image':
            continue
        if column == 'euclidian':
            f.write(' & ' + column + ' (\pm SD)')
        else:
            f.write(' & ' + column + '\% (\pm SD)')
    f.write(' \\\\\n')
    line = label
    for column in scores.columns:
        if column == 'image':
            continue
        if column == 'euclidian':
            line += ' & ' + str(round(scores[column].mean(),2)) + ' (\\pm ' + str(round(scores[column].std(), 2)) + ')'
        elif column == 'uncertainty':
            line += ' & ' + str(round(scores[column].mean().mean(),2)) #+ ' (\\pm ' + str(round(scores[column].std(), 2)) + ')'
        else:
            line += ' & ' + fperc(scores[column].mean()) + ' (\\pm ' + fperc(scores[column].std()) + ')'
    line += " \\\\\n"
    f.write(line)
    for img in imgs:
        img = img + '.png'
        line = img
        row = scores.loc[scores['image'] == img]
        for column in row:
            if column == 'image':
                continue
            if column == 'euclidian':
                line += ' & ' + str(round(float(row[column]),2))
            elif column == 'uncertainty':
                line += ' & ' + fperc(float(row[column].mean().mean()))
            else:
                entry = row[column]
                line += ' & ' + fperc(float(row[column]))
        line += " \\\\\n"
        f.write(line)


#%%
bin_scores = []
legend = []
for column in scores.keys():
    if column == 'image' or column == 'euclidian' or 'uncertainty' in column:
        continue
    bin_scores.append(np.array(scores[column]))

    if column == 'IOU':
        legend.append("IOU")
    else:
        legend.append(column.capitalize())
plt.figure()
ax = plt.gca()
#sns.distplot(bin_scores)
plt.hist(bin_scores, bins=np.arange(0.6, 1.05, 0.05))
ax.xaxis.grid(True)
plt.legend(legend)
#plt.xticks(np.(0.6,1, 0.1), )
plt.xticks(np.arange(0.6, 1, 0.05))
ax.set_ylabel('Image Count')
ax.set_xlabel("Score")
plt.xlim((0.6,1))
ax.yaxis.set_major_locator(plt.MaxNLocator(integer=True))
plt.savefig(str(Path(out, 'imgs', 'hist_' + identifier + '.png')), bbox_inches='tight', format='png', dpi=200)
plt.show()


#%%
# Boxplot
#scores.index.names = ['Split', 'index']
#scores = scores.reset_index()
#ax = sns.boxplot(x='Split', y='dice', data=scores, orient='v')
#plt.ylabel('Dice Coefficient')
#plt.xlabel('')
#plt.show()

#%%
#plt.rcParams['axes.formatter.limits'] = (-3, 3)
#fig = plt.figure()
#ax = sns.regplot(x='uncertainty', y='dice', data=scores)
#ax.set_xlim((0,scores['uncertainty'].max()))
#plt.xlabel("Uncertainty")
#plt.ylabel("Dice Coefficient")
#plt.savefig(str(Path(out, 'imgs', 'uncert_corr_' + identifier + '.png')), bbox_inches='tight', format='png', dpi=200)
#plt.show()
#
#plt.figure()
#bins =  np.linspace(0,5e-3, 11)
##scores_bayes['bins'] = pd.qcut(scores_bayes['uncertainty'], q=10)
#dice_bins = []
#for i in range(len(bins)-1):
#    dice_bins.append(scores['dice'][(scores['uncertainty'] >= bins[i]) & (scores['uncertainty'] < bins[i+1])])
#
#data = {}
#data['bins'] = bins[:-1]
#data['dice'] = dice_bins
#df = pd.DataFrame.from_dict(data)
#
#plt.rcParams.update({'font.size': 14})
##ax = sns.boxplot(data=scores_bayes, y='dice', x='bins')
#ax = sns.boxplot(data=dice_bins)
#sns.swarmplot(data=dice_bins, color=".25")
#
#bins_str = ["{:.1f}".format(1e3 * bin) for bin in list(bins)]
#plt.xticks(np.arange(0,11)-0.5, bins_str, horizontalalignment='right')
#ax.xaxis.grid(True)
#plt.xlabel("Uncertainty")
#plt.ylabel("Dice Coefficient")
#
#plt.gca().get_xaxis().get_major_formatter().set_offset_string('$10^{-3}$')
##ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
#plt.savefig(str(Path(out, 'imgs', 'uncert_dist_' + identifier + '.png')), bbox_inches='tight', format='png', dpi=200)
#
#plt.show()


