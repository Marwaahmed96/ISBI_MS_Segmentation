import os, sys, random
from math import floor
from operator import itemgetter

import shutil
import glob
from pathlib import Path

import pandas as pd
import numpy as np
import nibabel as nib

from sklearn.model_selection import StratifiedKFold
#from skimage.transform import resize
import h5py

import matplotlib.pyplot as plt
import ipywidgets as widgets
from ipywidgets import interact, interactive
from collections import defaultdict
from config import *
from images import *
from data_utils import *
from generator import *
from u_net_model import *
from reconstruct import *
from metric import*
from collections import defaultdict

import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv3D, Dropout, MaxPooling3D, UpSampling3D, Activation, BatchNormalization, PReLU, Conv3DTranspose, concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger, LearningRateScheduler, ReduceLROnPlateau, EarlyStopping
#tf.enable_eager_execution()

# force channels-last ordering
tf.keras.backend.set_image_data_format('channels_last')
print(tf.keras.backend.image_data_format())


def main():
    train_df=pd.read_csv(options["train_csv_path"])
    data,ref,patches=load_data_patches(options)

if __name__ == "__main__":
    main()
    