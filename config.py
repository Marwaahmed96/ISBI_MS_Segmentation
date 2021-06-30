# --------------------------------------------------------------------------------
# configuration file
#
# Sergi Valverde 2017
# --------------------------------------------------------------------------------

options = {}


# --------------------------------------------------
# Database options
# --------------------------------------------------

# path to training image folder. In all our experiments, training images were inside
# a folder and image modalities were consistent across training images. In the case of
# leave-one-out experiments, the same folder is used
options['root_dir']='/media/marwa/F2F25460F2542ADD/MedicalAnalysis/'
options['data_path']=options['root_dir']+'DataSets/ISBIOrig/training/'
options['code_path']=options['root_dir']+'Code/ISBI_MS_Segmentation/'
options['train_folder'] = options['root_dir']+'DataSets/ISBI/train/'
options['test_folder']  = options['root_dir']+'DataSets/ISBI/test/'

# image modalities used (T1, FLAIR, PD, T2, ...) 
options['modalities'] = ['flair','t2','pd','mprage']

options['masks']=['mask1','mask2']
options["train_csv_path"]=options['train_folder'] +"train_data.csv"
options['h5_path'] = options['root_dir']+'DataSets/ISBI/h5df_files/'
# image modalities nifti file names in the same order of options['modalities']
options['x_names'] = ['T1.nii.gz', 'FLAIR.nii.gz']

# lesion annotation nifti file names  
options['y_names'] = ['lesion_bin.nii.gz']


# --------------------------------------------------
# Experiment options
# --------------------------------------------------

# Select an experiment name to store net weights and segmentation masks
options['experiment'] = 'test_CNN_NI'

# minimum threshold used to select candidate voxels for training. Note that images are
# normalized to 0 mean 1 standard deviation before thresholding. So a value of t > 0.5 on FLAIR is
# reasonable in most cases to extract all WM lesion candidates 
options['min_th'] = 0.5

# randomize training features before fitting the model.  
options['randomize_train'] = True

# Select between pixel-wise or fully-convolutional training models. Although implemented, fully-convolutional
# models have been not tested with this cascaded model 
options['fully_convolutional'] = True



# --------------------------------------------------
# model parameters
# --------------------------------------------------

# 3D patch size. So, far only implemented for 3D CNN models. 
options['patch_size'] = (11,11,11)

# file paths to store the network parameter weights. These can be reused for posterior use. 
options['weight_paths'] = options['code_path']+'weights'

# percentage of the training vector that is going to be used to validate the model during training
options['train_split'] = 0.25

# maximum number of epochs used to train the model
options['max_epochs'] = 200

# maximum number of epochs without improving validation before stopping training 
options['patience'] = 25

# Number of samples used to test at once. This parameter should be around 50000 for machines
# with less than 32GB of RAM
options['batch_size'] = 200 #50000

# verbositiy of CNN messaging: 00 (none), 01 (low), 10 (medium), 11 (high)
options['net_verbose'] = 11

# post-processing binary threshold.
options['t_bin'] = 0.8

# post-processing minimum lesion size of soutput candidates
options['l_min'] = 20

