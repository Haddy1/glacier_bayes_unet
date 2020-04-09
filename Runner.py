import os
from datetime import datetime

now = datetime.now()
timestamp = now.strftime("%m%d-%H%M%S")

batch_size = 16
loss_function = 'focal_loss'
outpath =  "output/results_" + timestamp + "_" + loss_function
os.system('python3 main.py --OUTPATH ' + outpath + ' --Patch_Size 256 --Batch_Size ' + str(batch_size) + '--LOSS ' + loss_function)
