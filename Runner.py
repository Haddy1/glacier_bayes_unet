import os
from datetime import datetime

now = datetime.now()
timestamp = now.strftime("%m%d-%H%M%S")

alpha = 1
gamma = 0
outpath =  "output/results_" + timestamp + "/"
os('python3 main.py --OUTPATH ' + outpath + ' --Patch_Size 256 --Batch_Size 16 --LOSS focal_loss --alpha ' + str(alpha) + ' --gamma ' + str(gamma))
