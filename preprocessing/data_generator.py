from pathlib import Path
import numpy as np
import cv2
from skimage import io
from preprocessing import image_patches, preprocessor,augmentation
#import image_patches, preprocessor,augmentation
import json
import random
from shutil import copy, rmtree



def process_data(in_dir, out_dir, patch_size=256, preprocessor = None, img_list=None, augment = None):

    if not Path(out_dir).exists():
        Path(out_dir).mkdir(parents=True)

    if not Path(out_dir, 'images').exists():
        Path(out_dir, 'images').mkdir()
    if not Path(out_dir, 'masks').exists():
        Path(out_dir, 'masks').mkdir()

    if img_list:
        files_img = img_list
    else:
        files_img = Path(in_dir, 'images').glob('*.png')

    patch_counter = 0
    img_patch_index = {}
    for f in files_img:
        basename = f.stem
        img = cv2.imread(str(f), cv2.IMREAD_GRAYSCALE)
        shape = img.shape
        if preprocessor is not None:
            img = preprocessor.process(img)

        img = cv2.copyMakeBorder(img, 0, (patch_size - img.shape[0]) % patch_size, 0, (patch_size - img.shape[1]) % patch_size, cv2.BORDER_CONSTANT)

        mask_zones = cv2.imread(str(Path(in_dir, 'masks', basename + '_zones.png')), cv2.IMREAD_GRAYSCALE)
        mask_zones = cv2.copyMakeBorder(mask_zones, 0, (patch_size - mask_zones.shape[0]) % patch_size, 0, (patch_size - mask_zones.shape[1]) % patch_size, cv2.BORDER_CONSTANT)




        mask_zones[mask_zones == 127] = 0
        mask_zones[mask_zones == 254] = 255

        if augment is not None:
            imgs, augs = augment(img)
            masks_zones, _ = augment(mask_zones)
        else:
            imgs = [img]
            masks_zones = [mask_zones]
            augs = ['']


        for img, mask_zones,augmentation  in zip(imgs, masks_zones, augs):

            p_mask_zones, i_mask_zones = image_patches.extract_grayscale_patches(mask_zones, (patch_size, patch_size), stride = (patch_size, patch_size))
            p_img, i_img = image_patches.extract_grayscale_patches(img, (patch_size, patch_size), stride = (patch_size, patch_size))

            patch_indices = []
            for j in range(p_mask_zones.shape[0]):
                if np.count_nonzero(p_mask_zones[j])/(patch_size*patch_size) >= 0 and np.count_nonzero(p_mask_zones[j])/(patch_size*patch_size) <= 1:
                    cv2.imwrite(str(Path(out_dir, 'images/'+str(patch_counter)+'.png')), p_img[j])
                    cv2.imwrite(str(Path(out_dir, 'masks/'+str(patch_counter)+'.png')), p_mask_zones[j])

                    patch_indices.append(patch_counter) # store patch nrs used for image
                    patch_counter += 1


            patch_meta_data = {}
            patch_meta_data['origin'] = [i_mask_zones[0].tolist(), i_mask_zones[1].tolist()]
            patch_meta_data['indices'] = patch_indices
            # Todo: Fix img shape for augmentations
            patch_meta_data['img_shape'] = list(shape)

            img_patch_index[basename+augmentation] = patch_meta_data


    with open(Path(out_dir, 'image_list.json'), 'w') as f:
        json.dump(img_patch_index, f)


def generate_subset(data_dir, out_dir, set_size=None, patch_size=256, preprocessor=None, augment=None, patches_only=False, split=None, img_list=None):
    if not Path(data_dir).exists():
        print(str(data_dir) + " does not exist")


    if img_list is not None:
        files_img = img_list
    else:
        files_img = list(Path(data_dir, 'images').glob('*.png'))

    if set_size is not None:
        img_subset = random.sample(files_img, set_size)
    else:
        img_subset = files_img



    if not patches_only:
        if not Path(out_dir, 'images').exists():
            Path(out_dir, 'images').mkdir(parents=True)
        if not Path(out_dir, 'masks').exists():
            Path(out_dir, 'masks').mkdir()

        for f in img_subset:
            print(f)
            basename = f.stem
            img = cv2.imread(str(f), cv2.IMREAD_GRAYSCALE)
            if preprocessor is not None:
                img = preprocessor.process(img)
            mask_zones = cv2.imread(str(Path(data_dir, 'masks', basename + '_zones.png')), cv2.IMREAD_GRAYSCALE)
            if augment is not None:
                imgs, augs = augment(img)
                masks_zones, _ = augment(mask_zones)
            else:
                imgs = [img]
                masks_zones= [mask_zones]
                augs = ['']

            for img, mask_zones ,augmentation  in zip(imgs, masks_zones, augs):
                cv2.imwrite(str(Path(out_dir, 'images', basename + augmentation + '.png')), img)
                cv2.imwrite(str(Path(out_dir, 'masks', basename + augmentation + '_zones.png')), mask_zones)

    if patch_size is not None:
        process_data(data_dir, Path(out_dir, 'patches'), patch_size=patch_size, preprocessor=preprocessor, img_list=img_subset, augment=augment)


def split_set(data_dir, out_dir1, out_dir2, split):
    if not Path(out_dir1).exists():
        Path(out_dir1, 'images').mkdir(parents=True)
        Path(out_dir1, 'masks').mkdir(parents=True)
    if not Path(out_dir2).exists():
        Path(out_dir2, 'images').mkdir(parents=True)
        Path(out_dir2, 'masks').mkdir(parents=True)

    files_img = list(Path(data_dir, 'images').glob('*.png'))
    random.shuffle(files_img)
    if split < 1:
        split_point = int(split * len(files_img))
    else:
        split_point = split
    set1 = files_img[:split_point]
    set2 = files_img[split_point:]

    for f in set1:
        basename = f.stem
        copy(f, Path(out_dir1, 'images'))
        copy(Path(data_dir, 'masks', basename + '_zones.png'), Path(out_dir1, 'masks'))

    for f in set2:
        basename = f.stem
        copy(f, Path(out_dir2, 'images'))
        copy(Path(data_dir, 'masks', basename + '_zones.png'), Path(out_dir2, 'masks'))


def bayes_train_gen(img_path, pred_path, out_path, uncertainty_threshold = 1e-3):

    if not Path(out_path).exists():
        Path(out_path).mkdir(parents=True)
    if not Path(out_path, 'images').exists():
        Path(out_path, 'images').mkdir(parents=True)
    if not Path(out_path, 'masks').exists():
        Path(out_path, 'masks').mkdir(parents=True)

    for f in Path(img_path).rglob('*.png'):
        basename = f.stem
        uncertainty = io.imloadPath(pred_path, basename + '_uncertainty.png') / 65535
        if np.mean(uncertainty) < uncertainty_threshold:
            copy(f, Path(out_path, 'images'))
            copy(Path(pred_path, basename + '_pred.png'), Path(out_path, 'images'))


if __name__ == "__main__":
    random.seed(42)
    patch_size = 256

    preprocessor = preprocessor.Preprocessor()

    out_dir = Path('/home/andreas/glacier-front-detection/datasets/Jakobshavn_split_128')
    data_dir = Path('/home/andreas/glacier-front-detection/datasets/Jakobshavn_split')

    #split_set(Path(data_dir, 'train'), Path(out_dir, 'train'), Path(out_dir, 'unlabeled'), split=0.5)
    generate_subset(Path(data_dir, 'train'), Path(out_dir, 'train'), patch_size=128)
    #generate_subset(Path(out_dir, 'val'), Path(out_dir, 'val'), patch_size=256)
    #generate_subset(Path(out_dir, 'test'), Path(out_dir, 'test'), patch_size=256)
    #split_set(Path(out_dir, 'rest'), Path(out_dir, 'val'), Path(out_dir, 'test'), split=0.5)
    #generate_subset(Path(out_dir, 'test'), Path(out_dir, 'test'), patch_size=None)

    #split_set(Path(data_dir), Path(out_dir, 'unlabeled'), Path(out_dir, 'tmp'), split=0.6)
    #generate_subset(Path(out_dir, 'tmp/set1'), Path(out_dir, 'train1'), patch_size=patch_size,patches_only=True)
    #generate_subset(Path(out_dir, 'tmp/set2'), Path(out_dir, 'train2'), patch_size=patch_size,patches_only=True)

    #rmtree(Path(out_dir, 'tmp'))

    #split_set(Path(out_dir, 'tmp2'), Path(out_dir, 'tmp_val'), Path(out_dir, 'test'), split=0.5)
    #generate_subset(Path(out_dir, 'tmp_val'), Path(out_dir, 'val'), patch_size=patch_size)
    #generate_subset(Path(out_dir, 'tmp/set2'), Path(out_dir, 'val2'), patch_size=patch_size)
