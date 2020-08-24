import argparse
import time
from pathlib import Path
import pickle
import json

import cv2
import matplotlib.pyplot as plt
import numpy as np
from keras.callbacks import ModelCheckpoint, EarlyStopping, CSVLogger
from keras.models import load_model
import os
from CLR.clr_callback import CyclicLR
from layers.BayesDropout import  BayesDropout

from utils.data import trainGenerator
import models
from utils import helper_functions
from loss_functions import *
from keras.losses import *
from shutil import copy, rmtree
from preprocessing.preprocessor import Preprocessor
from preprocessing import data_generator, filter
from predict import predict, get_cutoff_point, predict_bayes
from utils import  evaluate

if __name__ == '__main__':
    # Hyper-parameter tuning
    parser = argparse.ArgumentParser(description='Glacier Front Segmentation')

    parser.add_argument('--epochs', default=250, type=int, help='number of training epochs (integer value > 0)')
    parser.add_argument('--patience', default=30, type=int, help='how long to wait for improvements before Early_stopping')
    parser.add_argument('--batch_size', default=-1, type=int, help='batch size (integer value), if -1 set batch size according to available gpu memery')
    parser.add_argument('--patch_size', default=256, type=int, help='size of the image patches (patch_size x patch_size')

    parser.add_argument('--early_stopping', default=1, type=int,
                        help='If 1, classifier is using early stopping based on val loss with patience 20 (0/1)')
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
    parser.add_argument('--contrast', default=0, type=int, help='Contrast Enhancement')
    parser.add_argument('--image_patches', default=0, type=int, help='Training data is already split into image patches')

    parser.add_argument('--out_path', type=str, help='Output path for results')
    parser.add_argument('--data_path', type=str, help='Path containing training and val data')
    parser.add_argument('--model', default='unet_Enze19_2', type=str, help='Training Model to use - can be pretrained model')
    parser.add_argument('--drop_rate', default=0.5, type=float, help='Dropout for Bayesian Unet')
    parser.add_argument('--cyclic', default='None', type=str, help='Which cyclic learning policy to use', choices=['None', 'triangular', 'triangular2', 'exp_range' ])
    parser.add_argument('--cyclic-parms', action=helper_functions.StoreDictKeyPair, metavar="KEY1=VAL1,KEY2=VAL2...",
                        help='dictionary with parameters for cyclic learning')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Initial learning rate')
    parser.add_argument('--predict', type=int, default=1, help='Evaluate prediction')
    parser.add_argument('--evaluate', type=int, default=1, help='Evaluate prediction')
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


    if args.data_path:
        data_path = Path(args.data_path)
    else:
        data_path = Path('data')

    if args.out_path:
        out_path = Path(args.out_path)
    else:
        out_path = Path('data_' + str(patch_size) + '/test/masks_predicted_' + time.strftime("%y%m%d-%H%M%S"))


    if not out_path.exists():
        out_path.mkdir(parents=True)

    # log all arguments including default ones
    with open(Path(out_path, 'options.json'), 'w') as f:
        f.write(json.dumps(vars(args)))

    # Preprocessing
    preprocessor = Preprocessor()
    if args.denoise:
        preprocessor.add_filter(filter.get_denoise_filter(args.denoise, args.denoise_parms))

    if args.contrast:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(25, 25))  # CLAHE adaptive contrast enhancement
        preprocessor.add_filter(clahe.apply)

    patches_path_train = Path(data_path, 'train/patches')
    if not patches_path_train.exists() or not Path(patches_path_train, 'image_list.json').exists() or len(json.load(open(Path(patches_path_train, 'image_list.json'), 'r')).keys()) == 0:
        if args.image_patches:
            patches_path_train = Path(data_path, 'train')
        data_generator.process_data(Path(data_path, 'train'), Path(patches_path_train), patch_size=patch_size,
                                        preprocessor=preprocessor)

    patches_path_val = Path(data_path, 'val/patches')
    if not patches_path_val.exists() or not Path(patches_path_val, 'image_list.json').exists() or len(json.load(open(Path(patches_path_val, 'image_list.json'), 'r')).keys()) == 0:
        if args.image_patches:
            patches_path_val = Path(data_path, 'val')
        data_generator.process_data(Path(data_path, 'val'), Path(patches_path_val), patch_size=patch_size,
                                        preprocessor=preprocessor)

    # copy image file list to output
    if Path(patches_path_train, 'image_list.json').exists():
        copy(Path(patches_path_train, 'image_list.json'), Path(out_path, 'train_image_list.json'))
    if Path(patches_path_val, 'image_list.json').exists():
        copy(Path(patches_path_val, 'image_list.json'), Path(out_path, 'val_image_list.json'))

    print(patches_path_train)

    train_Generator = trainGenerator(batch_size=batch_size,
                                     train_path=str(patches_path_train),
                                     image_folder='images',
                                     mask_folder='masks',
                                     aug_dict=None,
                                     save_to_dir=None)

    val_Generator = trainGenerator(batch_size=batch_size,
                                   train_path=str(patches_path_val),
                                   image_folder='images',
                                   mask_folder='masks',
                                   aug_dict=None,
                                   save_to_dir=None)

    loss_function = get_loss_function(args.loss, args.loss_parms)

    # Load pretrained model from file if it exists
    if '.hdf5' in args.model or '.h5' in args.model:
        model = load_model(args.model, custom_objects={'loss': loss_function, 'BayesDropout':BayesDropout})
    elif Path(args.model).exists():
        checkpoint_file = next(Path(args.model).glob('*.hdf5'))
        model = load_model(str(checkpoint_file.absolute()), custom_objects={'loss': loss_function, 'BayesDropout':BayesDropout})
    else:
        model_func = getattr(models, args.model)
        if 'bayes' in args.model:
            model = model_func(loss_function=loss_function, input_size=(patch_size, patch_size, 1), drop_rate=args.drop_rate)

        model = model_func(loss_function=loss_function, input_size=(patch_size, patch_size, 1))

    callbacks = []
    callbacks.append(CSVLogger(str(Path(out_path, model.name + '_history.csv')), append=True))

    model_checkpoint = ModelCheckpoint(str(Path(out_path, model.name + '_checkpoint.hdf5')), monitor='val_loss', verbose=0,
                                       save_best_only=True)

    callbacks.append(model_checkpoint)

    print(patches_path_train)
    num_samples = len([file for file in Path(patches_path_train, 'images').rglob('*.png')])  # number of training samples
    num_val_samples = len([file for file in Path(patches_path_val, 'images').rglob('*.png')])  # number of val samples

    print(num_samples)
    print(num_val_samples)


    if args.early_stopping:
        callbacks.append(EarlyStopping('val_loss', patience=args.patience, verbose=0, mode='auto', restore_best_weights=True))

    if args.cyclic is not 'None':
        if args.cyclic_parms is not None:
            cyclic_parms = args.cyclic_parms
        else:
            cyclic_parms = {}
        if not 'base_lr' in cyclic_parms:
            cyclic_parms['base_lr'] = 1e-4
        if not 'max_lr' in cyclic_parms:
            cyclic_parms['max_lr'] = 6e-4
        if not 'step_size' in cyclic_parms:
            cyclic_parms['step_size'] = int(4 * num_samples / batch_size)
        clr = CyclicLR(mode=args.cyclic, base_lr=cyclic_parms['base_lr'], max_lr=cyclic_parms['max_lr'], step_size=cyclic_parms['step_size'])
        callbacks.append(clr)
        args.__dict__['cyclic_parms'] = cyclic_parms # save changes in options
        # Update options file
        with open(Path(out_path, 'options.json'), 'w') as f:
            f.write(json.dumps(vars(args))) # Update options file



    steps_per_epoch = np.ceil(num_samples / batch_size)
    validation_steps = np.ceil(num_val_samples / batch_size)
    history = model.fit_generator(train_Generator,
                                  steps_per_epoch=steps_per_epoch,
                                  epochs=args.epochs,
                                  validation_data=val_Generator,
                                  validation_steps=validation_steps,
                                  callbacks=callbacks)

    # # save model
    model.save(str(Path(out_path, 'model_' + model.name + '.h5').absolute()))
    try:
        pickle.dump(history.history, open(Path(out_path, 'history_' + model.name + '.pkl'), 'wb'))
    except:
        print("History could not be saved")
    ##########
    ##########
    # save loss plot
    plt.figure()
    plt.rcParams.update({'font.size': 18})
    plt.plot(model.history.epoch, model.history.history['loss'], 'X-', label='training loss', linewidth=4.0)
    plt.plot(model.history.epoch, model.history.history['val_loss'], 'o-', label='val loss', linewidth=4.0)
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend(loc='upper right')
    plt.minorticks_on()
    plt.grid(which='minor', linestyle='--')
    plt.savefig(str(Path(str(out_path), 'loss_plot.png')), bbox_inches='tight', format='png', dpi=200)
    plt.show()

    # Cleanup
    if Path(out_path, 'model_' + model.name + '.h5').exists(): # Only cleanup if finished training model exists
        if Path(out_path, model.name + '_checkpoint.hdf5').exists():
            os.remove(Path(out_path, model.name + '_checkpoint.hdf5'))
        if Path(out_path, 'patches').exists():
            rmtree(Path(out_path, 'patches'))

    if args.image_patches and not Path(data_path, 'val/patches').exists():
        print('Cannot optimize cutoff point since only patches of the validation images exist')
        cutoff = 0.5
    else:
        print("Finding optimal cutoff point")
        cutoff = get_cutoff_point(model, Path(data_path, 'val'), out_path, batch_size=batch_size, patch_size=patch_size, preprocessor=preprocessor)
        # resave arguments including cutoff point
        with open(Path(out_path, 'options.json'), 'w') as f:
            args.__dict__['cutoff'] = cutoff
            f.write(json.dumps(vars(args)))

    if args.predict:
        test_path = str(Path(data_path, 'test'))
        if 'bayes' in model.name:
            predict_bayes(model, Path(test_path, 'images'), out_path, batch_size=batch_size, patch_size=patch_size, preprocessor=preprocessor, cutoff=cutoff)
        else:
            predict(model, Path(test_path, 'images'), out_path, batch_size=batch_size, patch_size=patch_size, preprocessor=preprocessor, cutoff=cutoff)
        if args.evaluate:
            evaluate.evaluate(Path(test_path, 'images'), Path(test_path, 'masks'), out_path)

    END = time.time()
    print('Execution Time: ', END - START)
