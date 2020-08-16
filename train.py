import numpy as np
from pathlib import Path
from preprocessing import data_generator
from utils.data import trainGenerator
from shutil import copy, rmtree
from os import remove
import json
import pickle
import matplotlib.pyplot as plt

def train(model, train_path, val_path, out_path, patch_size=256, batch_size=16, callbacks=None, epochs=250, preprocessor = None):
    patches_path_train = Path(train_path, 'patches')
    if not patches_path_train.exists() or not Path(patches_path_train, 'image_list.json').exists() or len(json.load(open(Path(patches_path_train, 'image_list.json'), 'r')).keys()) == 0:
        data_generator.process_data(train_path, patches_path_train, patch_size=patch_size,
                                    preprocessor=preprocessor)

    patches_path_val = Path(val_path, 'patches')
    if not patches_path_val.exists() or not Path(patches_path_val, 'image_list.json').exists() or len(json.load(open(Path(patches_path_val, 'image_list.json'), 'r')).keys()) == 0:
        data_generator.process_data(val_path, patches_path_val, patch_size=patch_size,
                                    preprocessor=preprocessor)

    num_samples = len([file for file in Path(patches_path_train, 'images').rglob('*.png')])  # number of training samples
    num_val_samples = len([file for file in Path(patches_path_val, 'images').rglob('*.png')])  # number of val samples
    print(num_samples)
    print(num_val_samples)

    # copy image file list to output
    if Path(patches_path_train, 'image_list.json').exists():
        copy(Path(patches_path_train, 'image_list.json'), Path(out_path, 'train_image_list.json'))
    if Path(patches_path_val, 'image_list.json').exists():
        copy(Path(patches_path_val, 'image_list.json'), Path(out_path, 'val_image_list.json'))
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

    steps_per_epoch = np.ceil(num_samples / batch_size)
    validation_steps = np.ceil(num_val_samples / batch_size)
    history = model.fit_generator(train_Generator,
                                  steps_per_epoch=steps_per_epoch,
                                  epochs=epochs,
                                  validation_data=val_Generator,
                                  validation_steps=validation_steps,
                                  callbacks=callbacks)

    model.save(str(Path(out_path, 'model_' + model.name + '.h5').absolute()))
    try:
        pickle.dump(history.history, open(Path(out_path, 'history_' + model.name + '.pkl'), 'wb'))
    except:
        print("History could not be saved")

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

    return model, history
