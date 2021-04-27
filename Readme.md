**Dataset Structure:**  
datasets  
    |---> train  
    |---> val  
    |---> test  

train  
    |--->  images: contains SAR-Images  
    |--->  masks: contains labeled glacier segmentation  

**Training + Inference**  
python main.py --data_path *path to datasets* --out_path *output path* --model *model name* --batch_size 16  

data_path: folder containing train,val and test datasets  
out_path:  folder for segmentation predictions and evaluation results  

available models: unet: model used by Zhang et. al  
                  unet_bayes: Bayesian U-Net  
                  two-stage: 2-Stage Optimization, with two unet_bayes models  

**Output**

*model_name*_history.h5 :   Trained model  
*model_name*_history.csv:   training history  
options.json:               Options and parameters used  
loss_plot.png:              Plot of loss history  
cutoff.png:                 Plot of binarization threshold  
dice_cutoff:                Tried cutoff points + resulting dice   
val_image_list.json:        validation image names + patch numbers  
train_image_list.json:      train image names + patch numbers  
scores.pkl:                 Pandas Dataframe with evaluation results for each image  