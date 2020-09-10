from __future__ import print_function
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import os
import glob
import skimage.io as io
import skimage.transform as trans
from pathlib import Path
from utils.helper_functions import nat_sort
import cv2


def adjustData(img,mask,flag_multi_class,num_class):
    if(flag_multi_class):
        img = img / 255
        mask = mask[:,:,:,0] if(len(mask.shape) == 4) else mask[:,:,0]
        new_mask = np.zeros(mask.shape + (num_class,))
        for i in range(num_class):
            #for one pixel in the image, find the class in mask and convert it into one-hot vector
            #index = np.where(mask == i)
            #index_mask = (index[0],index[1],index[2],np.zeros(len(index[0]),dtype = np.int64) + i) if (len(mask.shape) == 4) else (index[0],index[1],np.zeros(len(index[0]),dtype = np.int64) + i)
            #new_mask[index_mask] = 1
            new_mask[mask == i,i] = 1
        new_mask = np.reshape(new_mask,(new_mask.shape[0],new_mask.shape[1]*new_mask.shape[2],new_mask.shape[3])) if flag_multi_class else np.reshape(new_mask,(new_mask.shape[0]*new_mask.shape[1],new_mask.shape[2]))
        mask = new_mask
    elif(np.max(img) > 1):
        img = img / 255
        mask = mask / 255
        mask[mask > 0.5] = 1
        mask[mask <= 0.5] = 0
    return (img,mask)



def trainGenerator(batch_size,train_path,image_folder,mask_folder,aug_dict=None,image_color_mode = "grayscale",
                    mask_color_mode = "grayscale",image_save_prefix  = "image",mask_save_prefix  = "mask",
                    flag_multi_class = False,num_class = 2,save_to_dir = None,target_size = (256,256),seed = 1, shuffle=True):
    '''
    can generate image and mask at the same time
    use the same seed for image_datagen and mask_datagen to ensure the transformation for image and mask is the same
    if you want to visualize the results of generator, set save_to_dir = "your path"
    '''
    if aug_dict == None:
        image_datagen = ImageDataGenerator()
        mask_datagen = ImageDataGenerator()
    else:
        image_datagen = ImageDataGenerator(**aug_dict)
        mask_datagen = ImageDataGenerator(**aug_dict)
        
    image_generator = image_datagen.flow_from_directory(
        train_path,
        classes = [image_folder],
        class_mode = None,
        color_mode = image_color_mode,
        target_size = target_size,
        batch_size = batch_size,
        save_to_dir = save_to_dir,
        save_prefix  = image_save_prefix,
        shuffle=shuffle,
        seed = seed)
    mask_generator = mask_datagen.flow_from_directory(
        train_path,
        classes = [mask_folder],
        class_mode = None,
        color_mode = mask_color_mode,
        target_size = target_size,
        batch_size = batch_size,
        save_to_dir = save_to_dir,
        save_prefix  = mask_save_prefix,
        shuffle=shuffle,
        seed = seed)

    if len(image_generator) != len(mask_generator.filepaths):
        raise AssertionError("Different nr of input images and mask images")

    train_generator = zip(image_generator, mask_generator)
    for (img,mask) in train_generator:
        img,mask = adjustData(img,mask,flag_multi_class,num_class)
        yield (img,mask)

def trainGeneratorUncertainty(batch_size,train_path,image_folder,mask_folder, uncertainty_folder,aug_dict=None,image_color_mode = "grayscale",
                   mask_color_mode = "grayscale",image_save_prefix  = "image",mask_save_prefix  = "mask",
                   flag_multi_class = False,num_class = 2,save_to_dir = None,target_size = (256,256),seed = 1, shuffle=True):
    '''
    can generate image and mask at the same time
    use the same seed for image_datagen and mask_datagen to ensure the transformation for image and mask is the same
    if you want to visualize the results of generator, set save_to_dir = "your path"
    '''
    if aug_dict == None:
        image_datagen = ImageDataGenerator()
        mask_datagen = ImageDataGenerator()
    else:
        image_datagen = ImageDataGenerator(**aug_dict)
        mask_datagen = ImageDataGenerator(**aug_dict)

    image_generator = image_datagen.flow_from_directory(
        train_path,
        classes = [image_folder],
        class_mode = None,
        color_mode = image_color_mode,
        target_size = target_size,
        batch_size = batch_size,
        save_to_dir = save_to_dir,
        save_prefix  = image_save_prefix,
        shuffle=shuffle,
        seed = seed)
    uncertainty_generator= image_datagen.flow_from_directory(
        train_path,
        classes = [uncertainty_folder],
        class_mode = None,
        color_mode = image_color_mode,
        target_size = target_size,
        batch_size = batch_size,
        save_to_dir = save_to_dir,
        save_prefix  = image_save_prefix,
        shuffle=shuffle,
        seed = seed)
    mask_generator = mask_datagen.flow_from_directory(
        train_path,
        classes = [mask_folder],
        class_mode = None,
        color_mode = mask_color_mode,
        target_size = target_size,
        batch_size = batch_size,
        save_to_dir = save_to_dir,
        save_prefix  = mask_save_prefix,
        shuffle=shuffle,
        seed = seed)

    if len(image_generator) != len(mask_generator.filepaths):
        raise AssertionError("Different nr of input images and mask images")

    train_generator = zip(image_generator, uncertainty_generator, mask_generator)
    for (img,uncertainty,mask) in train_generator:
        uncertainty = uncertainty / 65535
        img,mask = adjustData(img,mask,flag_multi_class,num_class)
        combined = np.concatenate((img, uncertainty), axis=2)
        yield (img,mask)

def imgGenerator(batch_size,train_path,image_folder,aug_dict=None,image_color_mode = "grayscale",
                 target_size = (256,256),seed = 1, shuffle=True):

    if aug_dict == None:
        image_datagen = ImageDataGenerator(preprocessing_function=lambda img: img /255)
    else:
        image_datagen = ImageDataGenerator(**aug_dict)

    image_generator = image_datagen.flow_from_directory(
        train_path,
        classes = [image_folder],
        class_mode = None,
        color_mode = image_color_mode,
        target_size = target_size,
        batch_size = batch_size,
        shuffle=shuffle,
        seed = seed)
    if not shuffle:
        # sort files with natural string sorting i.e. 0,1,2,10, not 0,1,10,2
        nat_sort(image_generator._filepaths)
        image_generator.filenames = [f[(len(str(image_generator.directory)))+1:] for f in image_generator._filepaths]

    return image_generator

class ImgGenerator():
    def __init__(self, x, batch_size=16, as_gray=True, target_size=(256,256)):
        self.img_files = x
        if len(target_size) == 3:
            self.image_shape = target_size
        else:
            self.image_shape = (target_size[0], target_size[1], 1)
        self.n= len(self.img_files)
        self.batch_index = 0
        self.batch_size = batch_size
        self.img_as_gray = as_gray
        self.target_size = target_size

    def __len__(self):
        return self.n
    def __iter__(self):
        self.batch_index = 0
        return self
    def __next__(self):
        if self.batch_index * self.batch_size >= self.n:
            self.batch_index = 0
            raise StopIteration
        else:
            batch_img = []
            for i in range(self.batch_index * self.batch_size, self.batch_index * self.batch_size + self.batch_size):
                if i < self.n:
                    img_file = self.img_files[i]
                    img = io.imread(img_file, as_gray=self.img_as_gray)
                    if(np.max(img) > 1):
                        img = img / 255
                    img = cv2.resize(img, self.target_size)
                    if len(img.shape) == 2: # Add channel axis for grayscale img
                        img = img[..., np.newaxis]
                    batch_img.append(img)

            batch_img = np.array(batch_img)
            self.batch_index += 1
            return batch_img



class ImgDataGenerator():
    def __init__(self, batch_size, data_path, image_folder, mask_folder, img_as_gray=True, mask_as_gray=True,
                 flag_multi_class = False, num_class=2,target_size=(256,256), seed=1):
        self.img_as_gray = img_as_gray
        self.mask_as_gray = mask_as_gray
        self.flag_multi_class = flag_multi_class
        self.num_class = num_class
        self.target_size = target_size
        self.seed = seed
        if len(target_size) == 3:
            self.image_shape = target_size
        else:
            self.image_shape = (target_size[0], target_size[1], 1)
        img_files = []
        mask_files = []
        self.batch_size = batch_size
        for f in Path(data_path, image_folder).glob("*.png"):
            img_files.append(f)
            if Path(data_path, 'masks', f.stem + '_zones.png').exists():
                mask_files.append(Path(data_path, 'masks', f.stem + '_zones.png'))
            else:
                mask_files.append(Path(mask_folder, 'masks', f.stem + '.png'))

        self.set = np.column_stack((np.array(img_files), np.array(mask_files)))
        rs = np.random.RandomState(self.seed)
        rs.shuffle(self.set)
        self.img_files = self.set[:,0]
        self.mask_files = self.set[:,1]
        self.batch_index = 0
        self.n = len(img_files)
        self.samples = self.n
    def __len__(self):
        return self.n

    def __iter__(self):
        self.batch_index = 0
        return self
    def __next__(self):
        actual_index = self.batch_index * self.batch_size
        if actual_index >= self.n:
            raise StopIteration
        batch_img = []
        batch_mask = []
        actual_index = self.batch_index * self.batch_size
        for i in range(actual_index, actual_index + self.batch_size):
            if i >= self.n:
                break

            img_file, mask_file = self.set[i]
            img = io.imread(img_file, as_gray=self.img_as_gray)
            img = cv2.resize(img, self.target_size)
            if len(img.shape) == 2: # Add channel axis for grayscale img
                img = img[..., np.newaxis]

            mask = io.imread(mask_file, as_gray=self.mask_as_gray)
            mask = cv2.resize(mask, self.target_size)
            if len(mask.shape) == 2: #Add channel axis for grayscale img
                mask = mask[..., np.newaxis]


            img, mask = adjustData(img, mask, self.flag_multi_class, self.num_class)
            batch_img.append(img)
            batch_mask.append(mask)

        batch_img = np.array(batch_img)
        batch_mask = np.array(batch_mask)
        self.batch_index += 1
        return (batch_img, batch_mask)

    def iter_img(self):
        self.batch_index = 0
        print("Reset")

        while self.batch_index * self.batch_size < self.n:
            actual_index = self.batch_index * self.batch_size
            batch_img = []
            for i in range(actual_index, actual_index + self.batch_size):
                if i >= self.n:
                    break
                img_file = self.img_files[i]
                img = io.imread(img_file, as_gray=self.img_as_gray)
                img = cv2.resize(img, self.target_size)
                if len(img.shape) == 2: # Add channel axis for grayscale img
                    img = img[..., np.newaxis]
                if(np.max(img) > 1):
                    img = img / 255
                batch_img.append(img)

            batch_img = np.array(batch_img)
            self.batch_index += 1
            yield batch_img


    def iter_mask(self):
        self.batch_index = 0
        while self.batch_index * self.batch_size < self.n:
            actual_index = self.batch_index * self.batch_size
            batch_mask = []
            for i in range(actual_index, actual_index + self.batch_size):
                if i >= self.n:
                    break
                mask_file = self.mask_files[i]
                mask = io.imread(mask_file, as_gray=self.img_as_gray)
                mask = cv2.resize(mask, self.target_size)
                if len(mask.shape) == 2: #Add channel axis for grayscale img
                    mask = mask[..., np.newaxis]

                if(self.flag_multi_class):
                    mask = mask[:,:,:,0] if(len(mask.shape) == 4) else mask[:,:,0]
                    new_mask = np.zeros(mask.shape + (self.num_class,))
                    for i in range(self.num_class):
                        #for one pixel in the image, find the class in mask and convert it into one-hot vector
                        #index = np.where(mask == i)
                        #index_mask = (index[0],index[1],index[2],np.zeros(len(index[0]),dtype = np.int64) + i) if (len(mask.shape) == 4) else (index[0],index[1],np.zeros(len(index[0]),dtype = np.int64) + i)
                        #new_mask[index_mask] = 1
                        new_mask[mask == i,i] = 1
                    new_mask = np.reshape(new_mask,(new_mask.shape[0],new_mask.shape[1]*new_mask.shape[2],new_mask.shape[3])) if self.flag_multi_class else np.reshape(new_mask,(new_mask.shape[0]*new_mask.shape[1],new_mask.shape[2]))
                    mask = new_mask
                elif(np.max(mask) > 1):
                    mask= mask/ 255
                batch_mask.append(mask)
            batch_mask= np.array(batch_mask)
            self.batch_index += 1
            yield batch_mask





def testGenerator(test_path,num_image = 10,target_size = (256,256),flag_multi_class = False,as_gray = True):
    for i in range(num_image):
        img = io.imread(os.path.join(test_path,"%d.png"%i),as_gray = as_gray)
        img = img / 255
        img = trans.resize(img,target_size)
        img = np.reshape(img,img.shape+(1,)) if (not flag_multi_class) else img
        img = np.reshape(img,(1,)+img.shape)
        yield img
        
def testGenerator_Amir(test_path,num_image = 10,target_size = (256,256),flag_multi_class = False,as_gray = True):
    for i in range(num_image):
        img = io.imread(os.path.join(test_path,"%d.png"%i),as_gray = as_gray)
        img = img / 255
        # img = trans.resize(img,target_size)
        # img = np.reshape(img,img.shape+(1,)) if (not flag_multi_class) else img
        # img = np.reshape(img,(1,)+img.shape)
        yield img


def geneTrainNpy(image_path,mask_path,flag_multi_class = False,num_class = 2,image_prefix = "image",mask_prefix = "mask",image_as_gray = True,mask_as_gray = True):
    image_name_arr = glob.glob(os.path.join(image_path,"%s*.png"%image_prefix))
    image_arr = []
    mask_arr = []
    for index,item in enumerate(image_name_arr):
        img = io.imread(item,as_gray = image_as_gray)
        img = np.reshape(img,img.shape + (1,)) if image_as_gray else img
        mask = io.imread(item.replace(image_path,mask_path).replace(image_prefix,mask_prefix),as_gray = mask_as_gray)
        mask = np.reshape(mask,mask.shape + (1,)) if mask_as_gray else mask
        img,mask = adjustData(img,mask,flag_multi_class,num_class)
        image_arr.append(img)
        mask_arr.append(mask)
    image_arr = np.array(image_arr)
    mask_arr = np.array(mask_arr)
    return image_arr,mask_arr


def labelVisualize(num_class,color_dict,img):
    img = img[:,:,0] if len(img.shape) == 3 else img
    img_out = np.zeros(img.shape + (3,))
    for i in range(num_class):
        img_out[img == i,:] = color_dict[i]
    return img_out / 255


