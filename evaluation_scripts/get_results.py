import re
import sys
from pathlib import Path
import json
from shutil import copy
import pandas as pd
import numpy as np
from matplotlib import  pyplot as plt
import seaborn as sns
import os
os.chdir('../')
plt.rcParams.update({'font.size': 18})
identifier = 'bayes'
path = Path('output_' + identifier)
out = Path('/home/andreas/thesis/reports/bayes')
if not out.exists():
    out.mkdir(parents=True)

all_results = {}
frames = []
labels = []
for dir in path.iterdir():
    if not Path(dir, 'scores.pkl').exists():
        continue
    frames.append(pd.read_pickle(Path(dir, 'scores.pkl')))


    if Path(dir, 'options.json').exists():
        arguments = json.load(open(Path(dir, 'options.json'), 'r'))
    if identifier == 'combined' or identifier == 'flip':
        loss_split = arguments['loss_parms']
        label= str(loss_split['binary_crossentropy']) + '_' + str(loss_split['focal_loss'])
    elif identifier == 'filter':
        label = arguments['denoise']
        if label == 'nlmeans':
            if arguments['denoise_parms'] is not None:
                label = 'nlmeans h=70'
            else:
                label = 'nlmeans h=3'
    elif identifier == 'rotate':
        label = str(arguments['image_aug']['rotation_range'])
    else:
        label = dir.name

    labels.append(label)
    #all_results[denoise_filter] = results

    if not Path(out, 'imgs').exists():
        Path(out, 'imgs').mkdir()
    if Path(dir, 'loss_plot.png').exists():
        copy(Path(dir, 'loss_plot.png'), Path(out, 'imgs', 'loss' + label + '.png'))
    #copy(Path(dir, '2011-01-04_PALSAR_20_4_pred.png'), Path(out, 'imgs', '2011-01-04_PALSAR_20_4_pred_' + identifier + '_' + label + '.png'))
    #copy(Path(dir, '2006-10-18_ERS_20_4_pred.png'), Path(out, 'imgs', '2006-10-18_ERS_20_4_pred_' + identifier + '_' + label + '.png'))
    #copy(Path(dir, 'loss_plot.png'), Path(out, 'loss' + denoise_filter + '.png'))

scores = pd.concat(frames, keys = labels)
scores = scores.sort_index(ascending=True)

scores.to_pickle(Path(out, identifier + '_all_scores.pkl'))
labels = sorted(labels, reverse=False)
#%%

def fperc(x):
    return str(round(x * 100,2))


with open(Path(out,'results_' +identifier + '_horizontal.tex'), 'w') as f:
    for column in scores.columns:
        if column == 'image':
            continue
        if column == 'euclidian':
            f.write(' & ' + column.capitalize() + ' (\pm SD)')
        elif column == 'IOU':
            f.write(' & ' + 'IOU' + ' (\pm SD)')
        else:
            f.write(' & ' + column.capitalize() + '\% (\pm SD)')
    f.write(' \\\\\n')
    for label , results in scores.groupby(level=0, sort=False):
        line = label.capitalize()
        for column in results.columns:
            if column == 'image':
                continue
            if column == 'euclidian':
                line += ' & ' + str(round(results[column].mean(),2)) + ' (\\pm ' + str(round(results[column].std(), 2)) + ')'
            else:
                line += ' & ' + fperc(results[column].mean()) + ' (\\pm ' + fperc(results[column].std()) + ')'
        line += " \\\\\n"
        f.write(line)

with open(Path(out,'results_' +identifier + '_vertical.tex'), 'w') as f:
    for label in scores.index.levels[0]:
        f.write(' & ' + label.capitalize())
    f.write(' \\\\\n')
    for column in scores:
        if column == 'image':
            continue
        if column == 'IOU':
            line = 'IOU';
        else:
            line = column.capitalize()
        for label , results in scores[column].groupby(level=0, sort=False):
            if column == 'euclidian':
                line += ' & ' + str(round(results.mean(),2)) + ' (\\pm ' + str(round(results.std(), 2)) + ')'
            else:
                line += ' & ' + fperc(results.mean()) + ' (\\pm ' + fperc(results.std()) + ')'
        line += " \\\\\n"
        f.write(line)

#%%
#binplot
for column in scores.keys():
    if column == 'image':
        continue
    score = []
    max = scores[column].max()
    for label in labels:
        score.append(scores.loc[label, column])

    ax = plt.figure().gca()
    ax.yaxis.set_major_locator(plt.MaxNLocator(integer=True))

    if (column == 'euclidian'):
        plt.hist(score, bins=np.linspace(0, max, 6))
    elif 'uncertainty' in column:
        plt.hist(score, bins=np.linspace(0, 0.004, 6))
    else:
        plt.hist(score, bins=np.linspace(0.4,1,7))
    plt.legend(labels)

    if (column == 'euclidian'):
        plt.xticks(np.linspace(0, max, 6))
    elif 'uncertainty' in column:
        plt.ticklabel_format(axis='x', style='sci', scilimits=(0,0))
        plt.xticks(np.linspace(0, 0.004, 6))
    else:
        plt.xticks(np.arange(0.4,1, 0.1))
    ax.set_ylabel('Score Count')
    ax.set_xlabel('Score')
    plt.title(column)
    plt.savefig(str(Path(out, 'imgs', 'hist_' + identifier + '_' + column + '.png')), bbox_inches='tight', format='png', dpi=200)
    plt.show()

#%%
# Boxplot
scores.index.names = ['Split', 'index']
scores = scores.reset_index()
ax = sns.boxplot(x='Split', y='dice', data=scores, orient='v')
plt.xticks(
    rotation=45,
    horizontalalignment='right')
plt.ylabel('Dice Coefficient')
plt.show()
for column in scores.keys():
    if column == 'image':
        continue
    if column == 'euclidian':
        continue

    score = []
    max = scores[column].max()
    for label in labels:
        score.append(scores.loc[label, column])

    ax = plt.figure().gca()
    ax.yaxis.set_major_locator(plt.MaxNLocator(integer=True))

    if 'uncertainty' in column:
        continue
    else:
        sns.boxplot(data=score, orient='v')
    #plt.legend(labels)

    if (column == 'euclidian'):
        plt.xticks(np.linspace(0, max, 6))
    elif 'uncertainty' in column:
        plt.ticklabel_format(axis='x', style='sci', scilimits=(0,0))
        plt.xticks(np.linspace(0, 0.004, 6))
    else:
        plt.xticks(np.arange(0.4,1, 0.1))
    ax.set_ylabel('Score Count')
    ax.set_xlabel('Score')
    plt.title(column)
    plt.savefig(str(Path(out, 'imgs', 'hist_' + identifier + '_' + column + '.png')), bbox_inches='tight', format='png', dpi=200)
    plt.show()



