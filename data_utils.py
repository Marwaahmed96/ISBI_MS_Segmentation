import os
from collections import defaultdict
import numpy as np
import pandas as pd
import nibabel as nib
from operator import add
from pathlib import Path
import h5py
import glob
from config import *
from images import *


def load_data_patches(options,phase='train', fold=0):
    #patches generated in hdf5 files load it
    if os.path.isdir(options['h5_path']) and glob.glob(options['h5_path']+'*.hdf5'):
        pass
    else:
        # generate patches
        generate_data_patches(options)
    # load patches
    #files=glob.glob(options['h5_path']+'*.hdf5')
    files=[]
    df = pd.read_csv(options['train_csv_path'])
    if phase =='train':
        files = df.loc[df['fold'] != fold,'f5_path'].values
    else:
        files = df.loc[df['fold'] == fold,'f5_path'].values
                      
    files_data={}
    files_ref={}
    patches=0
    for file in files:
        print(file)
        #with h5py.File(raw_path, 'r') as f:
        raw_file = h5py.File(file, 'r') # should not close it immediately
        # raw_data = raw_file["raw_data"]
        raw_data = defaultdict(list)

        for i in raw_file.keys():
            # to get the matrix: self.data[i][:]
            # d.data[i][j][0], d.data[i][j][1]
            raw_data[i] = raw_file[i]
        patches+= raw_data['patches'][()]
        patient_id= raw_data['id'][()]
        files_data[patient_id]= raw_data
        files_ref[patient_id]=raw_file
    
    return files_data,files_ref,patches
    
def generate_data_patches(options):
    x_dict, y_dict = get_data_path(options['train_csv_path'], options['modalities'], options['masks'])
    train_data=pd.read_csv(options['train_csv_path'])
    for idx in x_dict:
        train_x_data={idx: x_dict[idx]}
        train_y_data={idx: y_dict[idx]}
        X,Y=load_training_data(train_x_data, train_y_data, options)
        print(X.shape, Y.shape)
        Path(options['h5_path']).mkdir(parents=True, exist_ok=True)
        f5_path=options['h5_path']+'file_'+idx+'.hdf5'
        index=train_data.loc[train_data.patient_id+train_data.study==idx].index[0]
        train_data.loc[index, "f5_path"] = f5_path
    
        #for i in raw_data:
        with h5py.File(f5_path, 'w') as f:
            print(X.shape,'patches',X.shape[0],'modalities',X.shape[-1])
            f.create_dataset("id", data=idx)
            f.create_dataset("patches", data=X.shape[0])
            f.create_dataset("modalities", data=X.shape[-1])
            f.create_dataset(str('X'), data=X)    
            f.create_dataset(str('Y'), data=Y)
        train_data.to_csv(options['train_csv_path'], index=False)
    
def save_data_patches():
    pass
    
def get_data_path(train_csv_path,modalities, masks, phase="train"):
    print("get_data_path")
    train_df = pd.read_csv(train_csv_path)

    train_x_data ={}
    train_y_data ={}

    #load data pathes from train csv file
    for index, row in train_df.iterrows():

        train_x_item=defaultdict(list)
        for modality in modalities:
            train_x_item[modality]=row['root_path']+row[modality]
        train_x_data[row['patient_id']+row['study']]=train_x_item

        train_y_item=defaultdict(list)
        for mask in masks:
            train_y_item[mask]=row['root_path']+row[mask]
        train_y_data[row['patient_id']+row['study']]=train_y_item

    return train_x_data,train_y_data


def select_training_voxels(input_masks, threshold=2, datatype=np.float32):
    """
    Select voxels for training based on a intensity threshold

    Inputs:
        - input_masks: list containing all subject image paths for a single modality
        - threshold: minimum threshold to apply (after normalizing images with 0 mean and 1 std)
    
    Output:
        - rois: list where each element contains the subject binary mask for selected voxels [len(x), len(y), len(z)]
    """

    # load images and normalize their intensities
    images = [np.moveaxis(load_img(image_name),(0,1,2),(2,1,0)) for image_name in input_masks]
    images_norm = [normalize_image(im,datatype) for im in images]

    # select voxels with intensity higher than threshold
    rois = [image > threshold for image in images_norm]
    return rois

def load_training_data(train_x_data, train_y_data, options, model = None):
    
    X, Y =load_X_Y_data(train_x_data, train_y_data, options, model)
    # apply randomization if selected
    if options['randomize_train']:
        
        seed = np.random.randint(np.iinfo(np.int32).max)
        np.random.seed(seed)
        X = np.random.permutation(X.astype(dtype=np.float32))
        np.random.seed(seed)
        Y = np.random.permutation(Y.astype(dtype=np.int32))

    # fully convolutional / voxel labels
    if options['fully_convolutional']:
        # Y = [ num_samples, 1, p1, p2, p3]
        Y = np.expand_dims(Y, axis = 1)
    #else:
        # Y = [num_samples,]
    #    if Y.shape[3] == 1:
    #        Y = Y[:, Y.shape[1] / 2, Y.shape[2] / 2, :]
    #    else:
    #        Y = Y[:, Y.shape[1] / 2, Y.shape[2] / 2, Y.shape[3] / 2]
    #    Y = np.squeeze(Y)
    
    # (patches, x, y, z, channels)
    X=np.moveaxis(X,1,-1)
    Y=np.moveaxis(Y,1,-1)
    return X, Y

def load_X_Y_data(train_x_data, train_y_data, options, model = None):
    '''
    Load training and label samples for all given scans and modalities.

    Inputs: 

    train_x_data: a nested dictionary containing training image paths: 
        train_x_data['scan_name']['modality'] = path_to_image_modality

    train_y_data: a dictionary containing labels 
        train_y_data['scan_name'] = path_to_label

    options: dictionary containing general hyper-parameters:
        - options['min_th'] = min threshold to remove voxels for training
        - options['size'] = tuple containing patch size, either 2D (p1, p2, 1) or 3D (p1, p2, p3)
        - options['randomize_train'] = randomizes data 
        - options['fully_conv'] = fully_convolutional labels. If false, 

    model: CNN model used to select training candidates

    Outputs:
        - X: np.array [num_samples, num_channels, p1, p2, p2]
        - Y: np.array [num_samples, 1, p1, p2, p2] if fully conv, [num_samples, 1] otherwise

    '''
    
    # get_scan names and number of modalities used 
    scans = [s for s in train_x_data.keys()]
    modalities = train_x_data[scans[0]].keys()

    # select voxels for training:
    #   if model is no passed, training samples are extract by discarding CSF and darker WM in FLAIR, and use all remaining voxels.
    #   if model is passes, use the trained model to extract all voxels with probability > 0.5 
    if model is None:
        flair_scans = [train_x_data[s]['flair'] for s in scans]
        selected_voxels = select_training_voxels(flair_scans, options['min_th'])
    else:
        selected_voxels = select_voxels_from_previous_model(model, train_x_data, options)
        
    # extract patches and labels for each of the modalities
    data = []

    for m in modalities:
        x_data = [train_x_data[s][m] for s in scans]
        y_data = [train_y_data[s]['mask1'] for s in scans]
        x_patches, y_patches = load_train_patches(x_data, y_data, selected_voxels, options['patch_size'])
        data.append(x_patches)
    # stack patches in channels [samples, channels, p1, p2, p3]
    X = np.stack(data, axis = 1)
    Y = y_patches
    return X, Y

def load_train_patches(x_data, y_data, selected_voxels, patch_size, random_state = 42, datatype=np.float32):
    """
    Load train patches with size equal to patch_size, given a list of selected voxels

    Inputs: 
       - x_data: list containing all subject image paths for a single modality
       - y_data: list containing all subject image paths for the labels
       - selected_voxels: list where each element contains the subject binary mask for selected voxels [len(x), len(y), len(z)]
       - tuple containing patch size, either 2D (p1, p2, 1) or 3D (p1, p2, p3)
    
    Outputs:
       - X: Train X data matrix for the particular channel [num_samples, p1, p2, p3]
       - Y: Train Y labels [num_samples, p1, p2, p3]
    """
    
    # load images and normalize their intensties
    images = [load_img(name) for name in x_data]
    images_norm = [normalize_image(im, datatype) for im in images]

    # load labels 
    lesion_masks = [load_img(name) for name in y_data]
    nolesion_masks = [np.logical_and(np.logical_not(lesion), brain) for lesion, brain in zip(lesion_masks, selected_voxels)]

    # Get all the x,y,z coordinates for each image
    lesion_centers = [get_mask_voxels(mask) for mask in lesion_masks]
    nolesion_centers = [get_mask_voxels(mask) for mask in nolesion_masks]
   
    # load all positive samples (lesion voxels) and the same number of random negatives samples
    np.random.seed(random_state) 

    x_pos_patches = [np.array(get_patches(image, centers, patch_size)) for image, centers in zip(images_norm, lesion_centers)]
    y_pos_patches = [np.array(get_patches(image, centers, patch_size)) for image, centers in zip(lesion_masks, lesion_centers)]
    
    indices = [np.random.permutation(range(0, len(centers1))).tolist()[:len(centers2)] for centers1, centers2 in zip(nolesion_centers, lesion_centers)]
    nolesion_small = [itemgetter(*idx)(centers) for centers, idx in zip(nolesion_centers, indices)]
    x_neg_patches = [np.array(get_patches(image, centers, patch_size)) for image, centers in zip(images_norm, nolesion_small)]
    y_neg_patches = [np.array(get_patches(image, centers, patch_size)) for image, centers in zip(lesion_masks, nolesion_small)]

    # concatenate positive and negative patches for each subject
    X = np.concatenate([np.concatenate([x1, x2]) for x1, x2 in zip(x_pos_patches, x_neg_patches)], axis = 0)
    Y = np.concatenate([np.concatenate([y1, y2]) for y1, y2 in zip(y_pos_patches, y_neg_patches)], axis= 0)
    
    return X, Y

def get_mask_voxels(mask):
    """
    Compute x,y,z coordinates of a binary mask 

    Input: 
       - mask: binary mask
    
    Output: 
       - list of tuples containing the (x,y,z) coordinate for each of the input voxels
    """
    
    indices = np.stack(np.nonzero(mask), axis=1)
    indices = [tuple(idx) for idx in indices]
    return indices

def get_patches(image, centers, patch_size=(15, 15, 15)):
    """
    Get image patches of arbitrary size based on a set of centers
    """
    # If the size has even numbers, the patch will be centered. If not, it will try to create an square almost centered.
    # By doing this we allow pooling when using encoders/unets.
    patches = []
    list_of_tuples = all([isinstance(center, tuple) for center in centers])
    sizes_match = [len(center) == len(patch_size) for center in centers]

    if list_of_tuples and sizes_match:
        patch_half = tuple([ [idx//2 , (idx//2) -1 if idx%2==0 else idx//2] for idx in patch_size])
        new_centers = [map(add, center, (p[0] for p in patch_half)) for center in centers]
        padding = tuple((idx[0], size-idx[1]) for idx, size in zip(patch_half, patch_size))
        new_image = np.pad(image, padding, mode='constant', constant_values=0)

        slices = [[slice(c_idx-p_idx[0], c_idx+p_idx[1]+1) for (c_idx, p_idx, s_idx) in zip(center, patch_half, patch_size)] for center in new_centers]
        patches = [new_image[idx] for idx in slices]
        
    return patches