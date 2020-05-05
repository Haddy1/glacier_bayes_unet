import re
import sys
from pathlib import Path
import json
from shutil import copy

path = Path(sys.argv[1])
out = Path(sys.argv[2])
if not out.exists():
    out.mkdir()
regex = re.compile(path.name)
dirs = [d for d in path.parent.iterdir() if regex.search(str(d))]
all_results = {}
for dir in dirs:
    if not Path(dir, 'ReportOnModel.txt').exists():
        continue
    results = {}
    with open(Path(dir, 'ReportOnModel.txt'), 'r') as f:
          content = f.readlines()
    content = [x.strip() for x in content]

    scores = content[1].split()
    results['Dice'] = str(round(100 * float(scores[0]), 2))
    results['IOU'] = str(round(100 * float(scores[1]), 2))
    results['Eucl'] = scores[2]
    results['Sensitivity'] = str(round(100 * float(scores[3]), 2))
    results['Specificity'] = str(round(100 * float(scores[4]), 2))


    arguments = json.load(open(Path(dir, 'arguments.json'), 'r'))
    #loss_split = arguments['loss_parms']
    #split = str(loss_split['binary_crossentropy']) + '_' + str(loss_split['focal_loss'])
    #all_results[split] = results
    denoise_filter = arguments['denoise']
    all_results[denoise_filter] = results

    #copy(Path(dir, 'loss_plot.png'), Path(out, 'loss' + split + '.png'))
    copy(Path(dir, 'loss_plot.png'), Path(out, 'loss' + denoise_filter + '.png'))

with open(Path(out,'results.tex'), 'w') as f:
    f.write('& Dice & IOU & Eucl & Sensitivity & Specificity\\\\\n')
    for split, results in sorted(all_results.items(), reverse=True):
        line = split
        line += ' & ' + results['Dice']
        line += ' & ' + results['IOU']
        line += ' & ' + results['Eucl']
        line += ' & ' + results['Sensitivity']
        line += ' & ' + results['Specificity'] + " \\\\\n"
        f.write(line)







