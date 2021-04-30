U-Net Segmentation

Data Generation
usage: data_generator.py [-h] [--out_path OUT_PATH] [--csv_file CSV_FILE] [--patch_size PATCH_SIZE]

Dataset Generator

optional arguments:
  -h, --help            show this help message and exit
  --out_path OUT_PATH   output path for dataset
  --csv_file CSV_FILE   Csv file containing img paths for dataset
  --patch_size PATCH_SIZE
                        Size of generated image patches

Example:
python3 preprocessing/data_generator.py --csv_file validation_images.csv --out_path front_detection_dataset/val

-----------------------------------
Training + Inference

usage: main.py [-h] [--epochs EPOCHS] [--patience PATIENCE] [--batch_size BATCH_SIZE] [--patch_size PATCH_SIZE] [--no_early_stopping] [--loss {binary_crossentropy,focal_loss,combined_loss}]
               [--loss_parms KEY1=VAL1,KEY2=VAL2...] [--image_aug KEY1=VAL1,KEY2=VAL2...] [--denoise {none,bilateral,median,nlmeans,enhanced_lee,kuan}] [--denoise_parms KEY1=VAL1,KEY2=VAL2...] [--patches_only]
               [--out_path OUT_PATH] [--data_path DATA_PATH] [--model {unet,unet_bayes,two_stage}] [--drop_rate DROP_RATE] [--learning_rate LEARNING_RATE] [--no_predict] [--no_evaluate] [--mc_iterations MC_ITERATIONS]
               [--second_stage] [--uncert_threshold UNCERT_THRESHOLD] [--multi_class]

Glacier Front Segmentation

optional arguments:
  -h, --help            show this help message and exit
  --epochs EPOCHS       number of training epochs (integer value > 0)
  --patience PATIENCE   how long to wait for improvements before Early_stopping
  --batch_size BATCH_SIZE
                        batch size (integer value), if -1 set batch size according to available gpu memery
  --patch_size PATCH_SIZE
                        size of the image patches (patch_size x patch_size
  --no_early_stopping   Dont Use Early Stopping
  --loss {binary_crossentropy,focal_loss,combined_loss}
                        loss function for the deep classifiers training
  --loss_parms KEY1=VAL1,KEY2=VAL2...
                        dictionary with parameters for loss function
  --image_aug KEY1=VAL1,KEY2=VAL2...
                        dictionary with the augmentation for keras Image Processing
  --denoise {none,bilateral,median,nlmeans,enhanced_lee,kuan}
                        Denoise filter
  --denoise_parms KEY1=VAL1,KEY2=VAL2...
                        dictionary with parameters for denoise filter
  --patches_only        Training data is already split into image patches
  --out_path OUT_PATH   Output path for results
  --data_path DATA_PATH
                        Path containing training and val data
  --model {unet,unet_bayes,two_stage}
                        Training Model to use - can be pretrained model
  --drop_rate DROP_RATE
                        Dropout for Bayesian Unet
  --learning_rate LEARNING_RATE
                        Initial learning rate
  --no_predict          Dont predict testset
  --no_evaluate         Dont evaluate
  --mc_iterations MC_ITERATIONS
                        Nr Monte Carlo Iterations for Bayes model
  --second_stage        Second Stage training
  --uncert_threshold UNCERT_THRESHOLD
                        Threshold for uncertainty binarisation
  --multi_class         Use MultiClass Segmentation

python main.py --data_path front_detection_dataset --out_path out_optimized --model two_stage --batch_size 16

-----------------------------------
Prediction Only

usage: predict.py [-h] [--model_path MODEL_PATH] [--img_path IMG_PATH] [--out_path OUT_PATH] [--uncert_path UNCERT_PATH] [--uncert_threshold UNCERT_THRESHOLD] [--gt_path GT_PATH] [--batch_size BATCH_SIZE] [--cutoff CUTOFF]
                  [--patches_only]

Glacier Front Segmentation Prediction

optional arguments:
  -h, --help            show this help message and exit
  --model_path MODEL_PATH
                        Path containing trained model
  --img_path IMG_PATH   Path containing images to be segmented
  --out_path OUT_PATH   output path for predictions
  --uncert_path UNCERT_PATH
                        Path containing uncertainty images
  --uncert_threshold UNCERT_THRESHOLD
                        Threshold for uncertainty binarisation
  --gt_path GT_PATH     Path containing the ground truth, necessary for evaluation_scripts
  --batch_size BATCH_SIZE
                        batch size (integer value)
  --cutoff CUTOFF       cutoff point of binarisation
  --patches_only        optimized prediction algorithm for small image patches

Example for unet evaluation of pix2pix generated images:
python predict.py --model_path out_unet --out_path out_unet_pix2pix_eval --img_path pix2pix_generated/images --gt_path Jakobshavn/test/patches/masks --batch_size 16 --patches_only


Complete Example:
python3 preprocessing/data_generator.py --csv_file validation_images.csv --out_path front_detection_dataset/val
python3 preprocessing/data_generator.py --csv_file train_images.csv --out_path front_detection_dataset/train
python3 preprocessing/data_generator.py --csv_file test_images.csv --out_path front_detection_dataset/test
python3 main.py --data_path front_detection_dataset --out_path out_optimized --model two_stage --batch_size 16
