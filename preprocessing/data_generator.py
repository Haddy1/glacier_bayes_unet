from pathlib import Path
import numpy as np
import cv2
from preprocessing import image_patches, preprocessor,augmentation
import json
import random
import shutil



def process_data(in_dir, out_dir, patch_size=256, preprocessor = None, img_list=None, augment = None):
    if not Path(out_dir).exists():
        Path(out_dir).mkdir(parents=True)

    if not Path(out_dir, 'images').exists():
        Path(out_dir, 'images').mkdir()
    if not Path(out_dir, 'masks').exists():
        Path(out_dir, 'masks').mkdir()
    if not Path(out_dir, 'lines').exists():
        Path(out_dir, 'lines').mkdir()

    if img_list:
        files_img = img_list
    else:
        files_img = Path(in_dir, 'images').glob('*.png')

    patch_counter = 0
    img_patch_index = {}
    for f in files_img:
        basename = f.stem
        img = cv2.imread(str(f), cv2.IMREAD_GRAYSCALE)
        if preprocessor is not None:
            img = preprocessor.process(img)
        mask_front = cv2.imread(str(Path(in_dir, 'lines', basename + '_front.png')), cv2.IMREAD_GRAYSCALE)
        mask_zones = cv2.imread(str(Path(in_dir, 'masks', basename + '_zones.png')), cv2.IMREAD_GRAYSCALE)



        mask_zones[mask_zones == 127] = 0
        mask_zones[mask_zones == 254] = 255

        if augment is not None:
            imgs, augs = augment(img)
            masks_zones, _ = augment(mask_zones)
            masks_front, _ = augment(mask_front)
        else:
            imgs = [img]
            masks_zones = [mask_zones]
            masks_front = [mask_front]
            augs = ['']

        for img, mask_zones, mask_front,augmentation  in zip(imgs, masks_zones, masks_front, augs):

            p_mask_zones, i_mask_zones = image_patches.extract_grayscale_patches(mask_zones, (patch_size, patch_size), stride = (patch_size, patch_size))
            p_img, i_img = image_patches.extract_grayscale_patches(img, (patch_size, patch_size), stride = (patch_size, patch_size))
            p_mask_front, i_mask_front = image_patches.extract_grayscale_patches(mask_front, (patch_size, patch_size), stride = (patch_size, patch_size))


            patch_indices = []
            for j in range(p_mask_zones.shape[0]):
                # if np.count_nonzero(p_masks_zone[j])/(patch_size*patch_size) > 0.05 and np.count_nonzero(p_masks_zone[j])/(patch_size*patch_size) < 0.95: # only those patches that has both background and foreground
                if np.count_nonzero(p_mask_zones[j])/(patch_size*patch_size) >= 0 and np.count_nonzero(p_mask_zones[j])/(patch_size*patch_size) <= 1:
                    cv2.imwrite(str(Path(out_dir, 'images/'+str(patch_counter)+'.png')), p_img[j])
                    cv2.imwrite(str(Path(out_dir, 'masks/'+str(patch_counter)+'.png')), p_mask_zones[j])
                    cv2.imwrite(str(Path(out_dir, 'lines/'+str(patch_counter)+'.png')), p_mask_front[j])
                    patch_indices.append(patch_counter) # store patch nrs used for image
                    patch_counter += 1


            patch_meta_data = {}
            patch_meta_data['origin'] = [i_mask_zones[0].tolist(), i_mask_zones[1].tolist()]
            patch_meta_data['indices'] = patch_indices
            patch_meta_data['img_shape'] = list(img.shape)

            img_patch_index[basename+augmentation] = patch_meta_data


    with open(Path(out_dir, 'image_list.json'), 'w') as f:
        json.dump(img_patch_index, f)


def generate_subset(data_dir, out_dir, set_size=None, patch_size=256, preprocessor=None, augment=None, patches_only=False):
    files_img = list(Path(data_dir, 'images').glob('*.png'))
    if set_size is not None:
        img_subset = random.sample(files_img, set_size)
    else:
        img_subset = files_img


    if not patches_only:
        if not Path(out_dir, 'images').exists():
            Path(out_dir, 'images').mkdir(parents=True)
        if not Path(out_dir, 'lines').exists():
            Path(out_dir, 'lines').mkdir()
        if not Path(out_dir, 'masks').exists():
            Path(out_dir, 'masks').mkdir()

        for f in img_subset:
            print(f)
            basename = f.stem
            img = cv2.imread(str(f), cv2.IMREAD_GRAYSCALE)
            if preprocessor is not None:
                img = preprocessor.process(img)
            mask_front = cv2.imread(str(Path(data_dir, 'lines', basename + '_front.png')), cv2.IMREAD_GRAYSCALE)
            mask_zones = cv2.imread(str(Path(data_dir, 'masks', basename + '_zones.png')), cv2.IMREAD_GRAYSCALE)
            if augment is not None:
                imgs, augs = augment(img)
                masks_zones, _ = augment(mask_zones)
                masks_front, _ = augment(mask_front)
            else:
                imgs = [img]
                masks_zones= [mask_zones]
                masks_front = [mask_front]
                augs = ['']

            for img, mask_zones, mask_front,augmentation  in zip(imgs, masks_zones, masks_front, augs):
                cv2.imwrite(str(Path(out_dir, 'images', basename + augmentation + '.png')), img)
                cv2.imwrite(str(Path(out_dir, 'masks', basename + augmentation + '_zones.png')), mask_zones)
                cv2.imwrite(str(Path(out_dir, 'lines', basename + augmentation + '_front.png')), mask_front)

    if patch_size is not None:
        process_data(data_dir, Path(out_dir, 'patches'), patch_size=patch_size, preprocessor=preprocessor, img_list=img_subset, augment=augment)



if __name__ == "__main__":
    random.seed(42)
    patch_size = 256

    preprocessor = preprocessor.Preprocessor()

    out_dir = Path('/home/andreas/glacier-front-detection/data_256_small')
    data_dir = Path('/home/andreas/glacier-front-detection/front_detection_dataset')


    generate_subset(Path(data_dir, 'test'), Path(out_dir, 'test'), patch_size=None, set_size=20)
    generate_subset(Path(data_dir, 'train'), Path(out_dir, 'train'), patch_size=patch_size,patches_only=True, set_size=50)
    generate_subset(Path(data_dir, 'val'), Path(out_dir, 'val'), set_size=20)
