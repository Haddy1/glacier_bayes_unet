import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from utils.evaluate import evaluate_dice_only
import pickle
import numpy as np
import re

plt.rcParams.update({'font.size': 18})
dice = []
with open('scores_dice.txt', 'r') as f:
    entries = []
    for line in f:
        if 'Name' in line:
            dice.append(np.mean(entries))
            entries = []
        split = line.rstrip().split()
        try:
            entries.append(float(split[-1]))
        except:
            continue

dice = np.array(dice)
#np.savetxt('output_mc_counts/dice.txt', dice)

uncert = []
with open('scores_uncert.txt', 'r') as f:
    entries = []
    for line in f:
        if 'Name' in line:
            uncert.append(np.mean(entries))
            entries = []
        split = line.rstrip().split()
        try:
            entries.append(float(split[-1]))
        except:
            continue

uncertainty = np.array(uncert)
#np.savetxt('output_mc_counts/uncert.txt', uncert)



#plt.rcParams.update({'font.size': 18})

#scores_1 = pickle.load(open('scores_1.pkl', 'rb'))
#scores_rest = pickle.load(open('scores_val.pkl', 'rb'))
#
##scores = scores_1 + scores_rest
#scores = pickle.load(open('scores_1-20.pkl', 'rb'))
#scores = scores + scores_rest[4:]
#
#x =list(range(1,21,1)) + list(range(25, 55, 5))
#
#dice = [np.mean(x['dice']) for x in scores]
#uncertainty = [np.mean(x['uncertainty_mean']) for x in scores]
#
#
plt.plot(range(1,41), dice)
plt.ylabel("Dice Coefficient")
plt.xlabel("Monte Carlo Iterations")
plt.plot((20,20), (0.94,dice[19]), linestyle='--', color='grey')
plt.xlim(1, 40)
plt.ylim(0.94, 0.95)

plt.savefig(str(Path('/home/andreas/thesis/reports/bayes/imgs', 'monte_carlo_dice.png')), bbox_inches='tight', format='png', dpi=200)
plt.show()

plt.figure()
plt.plot(range(1,41), uncertainty)
plt.plot((20,20), (np.min(uncertainty),uncertainty[19]), linestyle='--', color='grey')
plt.xlim(1, 40)
plt.xlabel("Monte Carlo Iterations")
plt.ylabel("Mean Uncertainty")
plt.savefig(str(Path('/home/andreas/thesis/reports/bayes/imgs', 'monte_carlo_uncert.png')), bbox_inches='tight', format='png', dpi=200)
plt.show()
