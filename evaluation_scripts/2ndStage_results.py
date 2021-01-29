import re
import sys
from pathlib import Path
import json
import string
import pickle
from shutil import copy
from matplotlib.ticker import StrMethodFormatter, FormatStrFormatter, FuncFormatter
import pandas as pd
import numpy as np
from matplotlib import  pyplot as plt
import matplotlib
import seaborn as sns
from skimage import io
import os
plt.rcParams.update({'font.size': 18})
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
    #plt.show()

path = Path('/home/andreas/glacier-front-detection/output/2ndStage_threshold')
out = Path('/home/andreas/thesis/presentation')
test_path = Path('/home/andreas/glacier-front-detection/datasets/front_detection_dataset/test')
if not out.exists():
    out.mkdir(parents=True)
if not Path(out, 'imgs').exists():
    Path(out, 'imgs').mkdir()

imgs = ['1998-02-18_ERS_20_1', '2009-12-03_PALSAR_20_5', '2012-04-28_TDX_5_1']

for basename in imgs:
    copy(Path(test_path, 'images', basename + '.png'), Path(out, 'imgs', basename + '.png'))
    mask = io.imread(Path(test_path, 'masks', basename + '_zones.png'))
    mask[mask== 127] = 0
    io.imsave(Path(out, 'imgs', basename + '_zones.png'), mask)

score_frames = []
identifiers = []
for d in path.iterdir():
    if not d.is_dir():
        continue
    identifier = d.name
    for basename in imgs:
        copy(Path(d, basename + '_pred.png'), Path(out, 'imgs', identifier + "_" + basename + '_pred.png'))
        #copy(Path(path, i + '_diff.png'), Path(out, 'imgs', i + '_diff.png'))

        if Path(d, basename + '_uncertainty.png').exists():
            uncert = io.imread(Path(d, basename + '_uncertainty.png'), as_gray=True)
            uncert = (uncert - uncert.min()) / (uncert.max() - uncert.min())
            io.imsave(Path(out, 'imgs', identifier + "_" + basename + '_uncertainty.png'), uncert)
    try:
        history = pd.read_csv(next(d.glob('*history.csv')))
        plot_history(history, Path(out, 'imgs', identifier + "_loss.png"), ylim=(0,0.8))
        copy(Path(d, 'cutoff.png'), Path(out, 'imgs', identifier + "_cutoff.png"))
    except:
        print("Not Found")

    scores = pd.read_pickle(Path(d, 'scores.pkl'))
    if 'uncertainty_var' in scores.columns:
        scores = scores.drop('uncertainty_var', axis=1)
    if 'uncertainty_mean' in scores.columns:
        scores = scores.rename(columns={"uncertainty_mean": "uncertainty"})
    if 'line_accuracy' in scores.columns:
        scores = scores.drop('line_accuracy', axis=1)
    score_frames.append(scores)
    identifier = d.name.replace("_", " ")
    identifiers.append(identifier)

all_scores= pd.concat(score_frames, keys=identifiers)
columns = ['dice', 'IOU', 'uncertainty']
with open(Path(out,'results_' + path.name + '.tex'), 'w') as f:
    line = "\\begin{tabular}{c c "
    for _ in columns:
        line += "c "
    line += "}\n"
    f.write(line)
    f.write("Method & Image")
    for column in columns:
        if column == 'image':
            continue
        if column == 'euclidian':
            f.write(' & ' + column.title() + ' ($\\pm$ SD)')
        else:
            f.write(' & ' + column.title() + '\% ($\\pm$ SD)')
    f.write(' \\\\\n')



    labels = []
    for label , scores in all_scores.groupby(level=0, sort=True):

        def fperc(x):
            return str(round(x * 100,2))

        f.write("\\midrule\n")
        f.write("\\multirow{" + str(len(imgs) +1) + "}{*}{\\shortstack[l]{" +label.replace(" ", "\\\\ ") + "}} \n")
        c = 0
        for img in imgs:
            img = img + '.png'
            line = "& " + string.ascii_uppercase[c]
            c += 1
            row = scores.loc[scores['image'] == img]
            for column in columns:
                if column == 'image':
                    continue
                if column == 'euclidian':
                    line += ' & ' + str(round(float(row[column]),2))
                elif column == 'uncertainty' or column == 'uncertainty_mean':
                    line += ' & ' + fperc(float(row[column]))
                else:
                    entry = row[column]
                    line += ' & ' + fperc(float(row[column]))
            line += " \\\\\n"
            f.write(line)
        line = "& Set"
        for column in columns:
            if column == 'image':
                continue
            if column == 'euclidian':
                line += ' & ' + str(round(scores[column].mean(),2)) + ' ($\\pm$ ' + str(round(scores[column].std(), 2)) + ')'
            else:
                line += ' & ' + fperc(scores[column].mean()) + ' ($\\pm$ ' + fperc(scores[column].std()) + ')'
        line += " \\\\\n"
        f.write(line)
    f.write("\\end{tabular}\n")


scores_mean = all_scores.mean(skipna=True, numeric_only=True, level=0)
scores_mean = scores_mean.loc[['Zhang', 'Bayesian', 'Optimized']]


for column in scores_mean.columns:
    if column == 'image':
        continue

    #    if column == 'method':
    #        continue
    #    #score = scores_mean[column]
    plt.figure()
    if column == 'euclidian':
        ax = sns.barplot(scores_mean.index, scores_mean[column], ci=None)
        ax.grid(False)
        ax.set(xlabel='Method', ylabel=column.title())
    elif column == 'uncertainty':
        uncertainty_scores = scores_mean.loc[['Bayesian', 'Optimized']]
        ax = sns.barplot(uncertainty_scores.index, uncertainty_scores[column], ci=None)
        ax.grid(False)
        ax.set(xlabel='Method', ylabel=column.title())
    else:
        ax = sns.barplot(scores_mean.index, 100 * scores_mean[column], ci=None)
        ax.grid(False)
        ax.set(ylim=(85,100), xlabel='Method', ylabel=column.title() + " %")
    if column != 'uncertainty':
        for p in ax.patches:
            ax.annotate(format(p.get_height(), '.2f'),
                        (p.get_x() + p.get_width() / 2., p.get_height()),
                        ha = 'center', va = 'center',
                        size=15,
                        xytext = (0, -12),
                        textcoords = 'offset points')
    plt.savefig(str(Path(out, 'imgs', 'score_Unet_' + column + '.png')), bbox_inches='tight', format='png', dpi=200)
    plt.show()



def graphicsline(img, width="width=0.12\\textwidth"):
    return  "\\includegraphics[" + width + "]{imgs/" + img + "}"
with open(Path(out,'image_table.tex'), 'w') as f:
    f.write("\\begingroup\n")
    f.write("\\setlength{\\tabcolsep}{1pt}\n")
    line = "\\begin{tabular}{c c "
    for _ in imgs:
        line += "c "
    line += "}\n"
    f.write(line)

    line = "& "
    for c in range(len(imgs)):
        line += "& " + string.ascii_uppercase[c]
    line += "\\\\\n"
    f.write(line)

    f.write("& Input Image \n")
    for basename in imgs:
        f.write("& " + graphicsline(basename + ".png") + "\n")
    f.write("\\\\\n")
    f.write("& Ground Truth\n")
    for basename in imgs:
        f.write("& " + graphicsline(basename + "_zones.png") + "\n")
    f.write("\\\\\n")

    dirs = [d for d in path.iterdir() if d.is_dir()]
    dirs.sort()
    for d in dirs:

        identifier = d.name

        f.write("\\midrule\n")
        if Path(d, basename + '_uncertainty.png').exists():
            f.write("\\multirow{3}{*}{\\shortstack[l]{ " +identifier.replace("_", " ").title() + "}} \n")
        else:
            f.write("\\shortstack[l]{ " + identifier.replace("_", " ").title() + "} \n")
        f.write("& Prediction\n")
        for basename in imgs:
            f.write("& " + graphicsline(identifier + "_" + basename + "_pred.png") + "\n")
        f.write("\\\\\n")
        if Path(d, basename + '_uncertainty.png').exists():
            f.write("& Uncertainty")
            for basename in imgs:
                f.write("& " + graphicsline(identifier + "_" + basename + "_uncertainty.png") + "\n")
            f.write("\\\\\n")
    f.write("\\end{tabular}\n")
    f.write("\\endgroup\n")


