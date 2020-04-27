
import sys
from pathlib import Path
import json

dirs = Path().glob(sys.argv[1])

all_results = {}
for dir in dirs:
    if not Path(dir, 'ReportOnModel.txt').exists():
        continue
    results = {}
    with open(Path(dir, 'ReportOnModel.txt'), 'r') as f:
          content = f.readlines()
    content = [x.strip() for x in content]

    standard = content[3].split()
    results['Sensitivity'] = standard[0]
    results['Specificity'] = standard[1]
    results['f1_score'] = standard[2]

    dice_eucl = content[1].split()
    results['Dice'] = dice_eucl[0]
    results['Eucl'] = dice_eucl[1]

    arguments = json.load(open(Path(dir, 'arguments.json'), 'r'))
    loss_split = arguments['loss_parms']
    split = str(loss_split['binary_crossentropy']) + '/' + str(loss_split['focal_loss'])
    all_results[split] = results

print('\tSensitivity\tSpecificity\tf1_score\tDice\tEucl')
for split, results in all_results.items():
    line = split
    line += '\t' + results['Sensitivity']
    line += '\t' + results['Specificity']
    line += '\t' + results['f1_score']
    line += '\t' + results['Dice']
    line += '\t' + results['Eucl']
    print (line)







