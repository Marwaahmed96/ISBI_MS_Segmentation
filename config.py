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

# pathes
options['root_dir']='/media/marwa/F2F25460F2542ADD/MedicalAnalysis/'
options['data_path']=options['root_dir']+'DataSets/ISBIOrig/training/'
options['code_path']=options['root_dir']+'Code/ISBI_MS_Segmentation/'
options['train_folder'] = options['root_dir']+'DataSets/ISBI/train/'
options['test_folder']  = options['root_dir']+'DataSets/ISBI/test/'
options["train_csv_path"]=options['train_folder'] +"train_data.csv"
options['h5_path'] = options['root_dir']+'DataSets/ISBI/h5df_files/'

# image modalities used (T1, FLAIR, PD, T2, ...) 
options['modalities'] = ['flair','t2','pd','mprage']
options['masks']=['mask1','mask2']


# image modalities nifti file names in the same order of options['modalities']
options['x_names'] = ['flair.nii', 'pd.nii','mprage.nii','t2.nii']

# lesion annotation nifti file names  
options['y_names'] = ['mask1.nii', 'mask2.nii']


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
options['patch_size'] = (24,24,16)

# file paths to store the network parameter weights. These can be reused for posterior use. 
options['weight_paths'] = options['code_path']+'weights/'

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

options['seed'] = 55  
options['k_fold'] = 4
    
#model config
options['channels'] = len(options['modalities'])
options['out_channels'] = 1
options['input_shape'] = (*options['patch_size'], options['channels'])
options['depth'] = 4 # depth of layers for V/Unet
options['n_base_filters'] = 32
options['pooling_kernel'] = (2, 2, 2)  # pool size for the max pooling operations
options['deconvolution'] = True  # if False, will use upsampling instead of deconvolution
    
#model train config
#where the model weights initialization so each time begin with the same weight to compare between different models
options['initial_weights_path'] = options['weight_paths']+'initial_weights.hdf5'
#Where to save the model weights during train
#,TPR,FPR,FNR,Tversky,dice_coefficient
options['metrics'] = ['mse']

options['patience'] = 5  # learning rate will be reduced after this many epochs if the validation loss is not improving
options['early_stop'] = 20  # training will be stopped after this many epochs without the validation loss improving
options['initial_learning_rate'] = 1e-3
options['learning_rate_drop'] = 0.1  # factor by which the learning rate will be reduced
options['n_epochs'] = 50

