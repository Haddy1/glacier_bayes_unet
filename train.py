import numpy as np
from pathlib import Path
from preprocessing import data_generator
from utils.data import trainGenerator
from shutil import copy, rmtree
from os import remove
import matplotlib.pyplot as plt
from tensorflow.keras.losses import binary_crossentropy
from tensorflow.keras.models import load_model
import tensorflow.keras as keras
from layers.BayesDropout import  BayesDropout
import json
import pickle
import models
import pandas as pd
from predict import get_cutoff_point
from utils.data import imgGenerator
def train(model, train_path, val_path, out_path, args, loss_function=binary_crossentropy, preprocessor=None, second_stage=False):

    # Create image patches from images if they not already exist
    patches_path_train = Path(train_path, 'patches')
    if not Path(patches_path_train, 'image_list.json').exists() or len(json.load(open(Path(patches_path_train, 'image_list.json'), 'r')).keys()) == 0:
        if args.patches_only:
            patches_path_train = train_path
        else:
            data_generator.process_data(train_path, patches_path_train, patch_size=args.patch_size,
                                    preprocessor=preprocessor)

    patches_path_val = Path(val_path, 'patches')
    if not Path(patches_path_val, 'image_list.json').exists() or len(json.load(open(Path(patches_path_val, 'image_list.json'), 'r')).keys()) == 0:
        if args.patches_only:
            patches_path_val = val_path
        else:
            data_generator.process_data(val_path, patches_path_val, patch_size=args.patch_size,
                                    preprocessor=preprocessor)

    # copy image file list to output
    if Path(patches_path_train, 'image_list.json').exists():
        copy(Path(patches_path_train, 'image_list.json'), Path(out_path, 'train_image_list.json'))
    if Path(patches_path_val, 'image_list.json').exists():
        copy(Path(patches_path_val, 'image_list.json'), Path(out_path, 'val_image_list.json'))

    if second_stage:
        uncertainty_folder='uncertainty'
    else:
        uncertainty_folder=None

    train_Generator = trainGenerator(batch_size=args.batch_size,
                                     train_path=str(patches_path_train),
                                     image_folder='images',
                                     mask_folder='masks',
                                     uncertainty_folder=uncertainty_folder,
                                     flag_multi_class=args.multi_class,
                                     uncert_threshold=args.uncert_threshold,
                                     aug_dict=None,
                                     save_to_dir=None)
    val_Generator = trainGenerator(batch_size=args.batch_size,
                                   train_path=str(patches_path_val),
                                   image_folder='images',
                                   mask_folder='masks',
                                   uncertainty_folder=uncertainty_folder,
                                   flag_multi_class=args.multi_class,
                                   uncert_threshold=args.uncert_threshold,
                                   aug_dict=None,
                                   save_to_dir=None)

    # Load pretrained model from file if it exists
    if '.hdf5' in model or '.h5' in model:
        model = load_model(model, custom_objects={'loss': loss_function, 'BayesDropout':BayesDropout})
    elif Path(model).exists():
        checkpoint_file = next(Path(model).glob('*.hdf5'))
        model = load_model(str(checkpoint_file.absolute()), custom_objects={'loss': args.loss_function, 'BayesDropout':BayesDropout})
    else:
        model_func = getattr(models, model)

        if second_stage:
            input_size = (args.patch_size, args.patch_size, 2)
        else:
            input_size = (args.patch_size, args.patch_size, 1)
        if args.multi_class:
            output_channels = 3
        else:
            output_channels = 1
        if 'bayes' in model:
            model = model_func(loss_function=loss_function,
                               input_size=input_size,
                               output_channels=output_channels,
                               drop_rate=args.drop_rate)
        else:
            model = model_func(loss_function=loss_function,
                               input_size=input_size,
                               output_channels=output_channels)

    callbacks = []
    callbacks.append(keras.callbacks.CSVLogger(str(Path(out_path, model.name + '_history.csv')), append=True))

    model_checkpoint = keras.callbacks.ModelCheckpoint(str(Path(out_path, model.name + '_checkpoint.hdf5')),
                                                       monitor='val_loss',
                                                       verbose=0,
                                                       save_best_only=True)
    callbacks.append(model_checkpoint)
    callbacks.append(keras.callbacks.TensorBoard(log_dir=str(Path(out_path, "logs/fit"))))

    print(patches_path_train)
    num_samples = len([file for file in Path(patches_path_train, 'images').rglob('*.png')])  # number of training samples
    num_val_samples = len([file for file in Path(patches_path_val, 'images').rglob('*.png')])  # number of val samples

    print(num_samples)
    print(num_val_samples)


    callbacks.append(keras.callbacks.EarlyStopping('val_loss', patience=args.patience, verbose=0, mode='auto', restore_best_weights=True))

    steps_per_epoch = np.ceil(num_samples / args.batch_size)
    validation_steps = np.ceil(num_val_samples / args.batch_size)
    model.fit_generator(train_Generator,
                        steps_per_epoch=steps_per_epoch,
                        epochs=args.epochs,
                        validation_data=val_Generator,
                        validation_steps=validation_steps,
                        callbacks=callbacks)

    model.save(str(Path(out_path, 'model_' + model.name + '.h5').absolute()))

    history = pd.read_csv(Path(out_path, model.name + '_history.csv'))
    try:
        pickle.dump(history, open(Path(out_path, 'history_' + model.name + '.pkl'), 'wb'))
    except IOError as err:
        print("History could not be saved: " + str(err))

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
            remove(Path(out_path, model.name + '_checkpoint.hdf5'))
        if Path(out_path, 'patches').exists():
            rmtree(Path(out_path, 'patches'))

    if not args.multi_class:
        print("Finding optimal cutoff point")
        img_generator = imgGenerator(args.batch_size, patches_path_val, 'images')
        mask_generator = imgGenerator(args.batch_size, patches_path_val, 'masks')
        cutoff, _ = get_cutoff_point(model,
                                     val_path,
                                     out_path=out_path,
                                     batch_size=args.batch_size,
                                     mc_iterations=args.mc_iterations,
                                     uncert_threshold=args.uncert_threshold)

        # resave arguments including cutoff point
        with open(Path(out_path, 'options.json'), 'w') as f:
            args.__dict__['cutoff'] = cutoff
            f.write(json.dumps(vars(args)))
    else:
        cutoff = None
    return model, history, cutoff
