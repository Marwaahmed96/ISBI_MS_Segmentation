import numpy as np
import nibabel as nib
import ipywidgets as widgets
from ipywidgets import interact, interactive
import matplotlib.pyplot as plt
from operator import itemgetter

def load_img(file_path,datatype=None):
    data = nib.load(file_path)
    if datatype == None:
        data = np.asarray(data.dataobj)
    else:
        data = np.asarray(data.dataobj).astype(dtype=datatype)
    return data

def show3D_images(images):
    # show image with [None, None, : ,: ,:] dimension
    def show_frame(id):
        length = len(images)
        for i in range(length):
            ax = plt.subplot(1, length, i+1)
            if (i == 0):
                ax.set_title("Input")
            if (i == 1):
                ax.set_title("Target")
            if (i == 2):
                ax.set_title("Output")
            plt.imshow(images[i][id, :, :], cmap='gray')
    interact(show_frame, 
             id=widgets.IntSlider(min=0, max=images[0].shape[0]-1, step=1, value=images[0].shape[0]/2))
    
def load_img_header(file_path):
    file=nib.load(file_path)
    return file.header

def normalize_image(image,datatype=np.float32):
    return (image.astype(dtype=datatype) - image[np.nonzero(image)].mean()) / image[np.nonzero(image)].std()

def mean_std_normalize(data: np.ndarray):
    data_min = np.min(data)
    return (data - data_min) / (np.max(data) - data_min)