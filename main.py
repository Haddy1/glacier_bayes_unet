import argparse
import time
from pathlib import Path
import pickle
import json
import datetime

import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow.keras as keras
from tensorflow.keras.models import load_model
import os
from CLR.clr_callback import CyclicLR
from layers.BayesDropout import  BayesDropout
import pandas as pd

from utils.data import trainGenerator, imgGenerator
import models
from utils import helper_functions
from loss_functions import *
from tensorflow.keras.losses import *
from shutil import copy, rmtree
from preprocessing.preprocessor import Preprocessor
from preprocessing import data_generator, filter
from predict import predict, get_cutoff_point
from utils import  evaluate
from train import train

if __name__ == '__main__':
    # Hyper-parameter tuning
    parser = argparse.ArgumentParser(description='Glacier Front Segmentation')

    parser.add_argument('--epochs', default=250, type=int, help='number of training epochs (integer value > 0)')
    parser.add_argument('--patience', default=30, type=int, help='how long to wait for improvements before Early_stopping')
    parser.add_argument('--batch_size', default=-1, type=int, help='batch size (integer value), if -1 set batch size according to available gpu memery')
    parser.add_argument('--patch_size', default=256, type=int, help='size of the image patches (patch_size x patch_size')

    parser.add_argument('--no_early_stopping', action='store_true',
                        help='Dont Use Early Stopping')
    parser.add_argument("--loss", help="loss function for the deep classifiers training ",
                        choices=["binary_crossentropy", "focal_loss", "combined_loss"], default="binary_crossentropy")
    parser.add_argument('--loss_parms', action=helper_functions.StoreDictKeyPair, metavar="KEY1=VAL1,KEY2=VAL2...",
                        help='dictionary with parameters for loss function')
    parser.add_argument('--image_aug', action=helper_functions.StoreDictKeyPair, metavar="KEY1=VAL1,KEY2=VAL2...",
                        help='dictionary with the augmentation for keras Image Processing',
                        default={'horizontal_flip': False, 'rotation_range': 0, 'fill_mode': 'nearest'})
    parser.add_argument("--denoise", help="Denoise filter",
                        choices=["none", "bilateral", "median", 'nlmeans', "enhanced_lee", "kuan"], default="None")
    parser.add_argument('--denoise_parms', action=helper_functions.StoreDictKeyPair, metavar="KEY1=VAL1,KEY2=VAL2...",
                        help='dictionary with parameters for denoise filter')
    parser.add_argument('--contrast', action='store_true', help='Contrast Enhancement')
    parser.add_argument('--patches_only', action='store_true', help='Training data is already split into image patches')

    parser.add_argument('--out_path', type=str, help='Output path for results')
    parser.add_argument('--data_path', type=str, help='Path containing training and val data')
    parser.add_argument('--model', default='uncert_net', type=str, help='Training Model to use - can be pretrained model')
    parser.add_argument('--drop_rate', default=0.5, type=float, help='Dropout for Bayesian Unet')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Initial learning rate')
    parser.add_argument('--no_predict', action='store_true', help='Dont predict testset')
    parser.add_argument('--no_evaluate', action='store_true', help='Dont evaluate')
    parser.add_argument('--mc_iterations', type=int, default=20, help='Nr Monte Carlo Iterations for Bayes model')
    parser.add_argument('--second_stage', action='store_true', help='Second Stage training')
    parser.add_argument('--uncert_threshold', type=float, default=None, help='Threshold for uncertainty binarisation')
    parser.add_argument('--multi_class', action='store_true', help='Use MultiClass Segmentation')

    # parser.add_argument('--Random_Seed', default=1, type=int, help='random seed number value (any integer value)')

    args = parser.parse_args()

    START = time.time()

    patch_size = args.patch_size

    if args.batch_size == -1:
        gpu_mem = helper_functions.get_gpu_memory()[0]
        if gpu_mem < 6000: batch_size = 4
        elif gpu_mem < 10000: batch_size = 16
        else: batch_size = 25
        args.batch_size = batch_size
    else:
        batch_size = args.batch_size

    train_path = Path(args.data_path, 'train')
    if not Path(train_path, 'images').exists():
        if Path(train_path,'patches/images').exists() and args.model != 'uncert_net':
            train_path = Path(train_path, 'patches')
        else:
            raise FileNotFoundError("training images Path not found")
    if len(list(Path(train_path, 'images').glob("*.png"))) == 0:
        raise FileNotFoundError("No training images were found")

    val_path = Path(args.data_path, 'val')
    if not Path(val_path, 'images').exists():
        if Path(val_path,'patches/images').exists():
            val_path = Path(val_path, 'patches')
        else:
            raise FileNotFoundError("Validation images Path not found")
    if len(list(Path(val_path, 'images').glob("*.png"))) == 0:
        raise FileNotFoundError("No validation images were found")

    if not args.no_predict:
        test_path = Path(args.data_path, 'test')
        if not Path(test_path, 'images').exists():
            if Path(test_path,'patches/images').exists():
                test_path= Path(test_path, 'patches')
            else:
                raise FileNotFoundError("test images Path not found")
        if len(list(Path(test_path, 'images').glob("*.png"))) == 0:
            raise FileNotFoundError("No test images were found")


    if args.out_path:
        out_path = Path(args.out_path)
    else:
        out_path = Path('data_' + str(patch_size) + '/test/masks_predicted_' + time.strftime("%y%m%d-%H%M%S"))


    if not out_path.exists():
        out_path.mkdir(parents=True)

    # log all arguments including default ones
    with open(Path(out_path, 'options.json'), 'w') as f:
        f.write(json.dumps(vars(args)))

    if args.second_stage:
        uncert_threshold = args.uncert_threshold
    else:
        uncert_threshold = None

    # Preprocessing
    preprocessor = Preprocessor()
    if args.denoise:
        preprocessor.add_filter(filter.get_denoise_filter(args.denoise, args.denoise_parms))

    if args.contrast:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(25, 25))  # CLAHE adaptive contrast enhancement
        preprocessor.add_filter(clahe.apply)

    loss_function = get_loss_function(args.loss, args.loss_parms)

    if 'bayes' in args.model or 'uncert' in args.model:
        mc_iterations = args.mc_iterations
    else:
        mc_iterations = 1

    if args.model == 'uncert_net':
        #1st Stage
        model_1st, history_1st, cutoff_1st = train('unet_bayes', train_path, val_path, Path(out_path, '1stStage'), args, loss_function=loss_function, preprocessor=preprocessor)
        predict(model_1st,
                Path(train_path, 'images'),
                Path(out_path, '1stStage, train'),
                batch_size=batch_size,
                patch_size=patch_size,
                preprocessor=preprocessor,
                cutoff=cutoff_1st,
                mc_iterations=args.mc_iterations)
        predict(model_1st,
                Path(train_path, 'images'),
                Path(out_path, '1stStage, val'),
                batch_size=batch_size,
                patch_size=patch_size,
                preprocessor=preprocessor,
                cutoff=cutoff_1st,
                mc_iterations=args.mc_iterations)

        ##2ndStage
        data_generator.process_imgs(Path(out_path,'1stStage/train/uncertainty'), Path(out_path, '1stStage/train/uncertainty/patches'))
        data_generator.process_imgs(Path(out_path,'1stStage/val/uncertainty'), Path(out_path, '1stStage/val/uncertainty/patches'))

        model, history, cutoff = train('unet_bayes', train_path, val_path, out_path, args,
                                                   train_uncert_path=Path(out_path,'1stStage/train/uncertainty/patches'),
                                                   val_uncert_path=Path(out_path, '1stStage/val/uncertainty/patches'),
                                                   loss_function=loss_function, preprocessor=preprocessor)

        # predict 1st Stage
        if not args.no_predict:
            predict(model_1st,
                    Path(test_path, 'images'),
                    Path(out_path, '1stStage'),
                    batch_size=batch_size,
                    patch_size=patch_size,
                    preprocessor=preprocessor,
                    cutoff=cutoff_1st,
                    mc_iterations=args.mc_iterations)
        if not args.no_evaluate:
            evaluate.evaluate(Path(test_path, 'masks'), out_path)

        # predict 2ndStage
            predict(model,
                    Path(test_path, 'images'),
                    out_path,
                    uncert_path=Path(out_path,'1stStage/uncertainty'),
                    batch_size=batch_size,
                    patch_size=patch_size,
                    preprocessor=preprocessor,
                    cutoff=cutoff,
                    mc_iterations=args.mc_iterations)
            if not args.no_evaluate:
                evaluate.evaluate(Path(test_path, 'masks'), out_path)

    # single stage mode
    else:
        model, history, cutoff = train(args.model, train_path, val_path, out_path, args, loss_function=loss_function, preprocessor=preprocessor)
        if not args.no_predict:
            if args.second_stage:
                uncert_test_path = Path(test_path, 'uncertainty')
            else:
                uncert_test_path = None
            predict(model,
                    Path(test_path, 'images'),
                    out_path,
                    uncert_path=uncert_test_path,
                    batch_size=batch_size,
                    patch_size=patch_size,
                    preprocessor=preprocessor,
                    cutoff=cutoff,
                    mc_iterations=args.mc_iterations)
            evaluate.evaluate(Path(test_path, 'masks'), out_path)
            if not args.no_evaluate:
                evaluate.evaluate(Path(test_path, 'masks'), out_path)




    END = time.time()
    print('Execution Time: ', END - START)
