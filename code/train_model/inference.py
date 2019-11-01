import os
# # Assign GPU no
os.environ["CUDA_VISIBLE_DEVICES"]=os.environ['SGE_GPU']

import tensorflow as tf
config=tf.ConfigProto()
config.gpu_options.allow_growth=True
config.allow_soft_placement=True

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np

#to make directories
import pathlib
import nibabel as nib

import sys
sys.path.append('../')

from utils import *

import argparse
parser = argparse.ArgumentParser()
#data set type
parser.add_argument('--dataset', type=str, default='prostate_ge', choices=['prostate_md','prostate_ge'])

parser.add_argument('--ip_path', type=str, default=None)
#parser.add_argument('--ip_path', type=str, default='/usr/krishnch/datasets/prostate/001/img.nii.gz')

#version of run
parser.add_argument('--out_path', type=str, default=None)

parse_config = parser.parse_args()
#parse_config = parser.parse_args(args=[])

if parse_config.dataset == 'prostate_md':
    #print('load prostate_md configs')
    import experiment_init.init_prostate_md as cfg
    import experiment_init.data_cfg_prostate_md as data_list
elif parse_config.dataset == 'prostate_ge':
    #print('load prostate_md configs')
    import experiment_init.init_prostate_ge as cfg
    import experiment_init.data_cfg_prostate_ge as data_list
else:
    raise ValueError(parse_config.dataset)

######################################
# class loaders
# ####################################
#  load dataloader object
from dataloaders import dataloaderObj
dt = dataloaderObj(cfg)

#if parse_config.dataset == 'prostate_md':
#    #print('set prostate_md orig img dataloader handle')
#    orig_img_dt=dt.load_prostate_imgs

#  load model object
from models import modelObj
model = modelObj(cfg)
#  load f1_utils object
from f1_utils import f1_utilsObj
f1_util = f1_utilsObj(cfg,dt)

######################################
#define save_dir for the model
if parse_config.dataset == 'prostate_md':
    save_dir='../../models/tr_baseline_unet_prostate_md_deform/'
elif parse_config.dataset == 'prostate_ge':
    save_dir=str(cfg.base_dir)+'/models/'+str(parse_config.dataset)+'/tr_baseline_unet_rand_deform/'
print('save dir ',save_dir)

#find the model with best dice score on validation images
mp_best=get_max_chkpt_file(save_dir)
print('load mp',mp_best)

######################################
# define U-Net model graph
tf.reset_default_graph()
ae = model.unet(learn_rate_seg=0.001,dsc_loss=1,en_1hot=1)

# restore best model and predict segmentations on test subjects
saver = tf.train.Saver()
sess = tf.Session(config=config)
saver.restore(sess, mp_best)
print("Model restored")
######################################

#define image path
img_path=str(parse_config.ip_path)

#checking if image path exists or not
if(os.path.isfile(img_path)==False):
    print('img path does not exist')
    sys.exit()
    
#segmentation output path directory 
# by default same as input directory unless defined
out_path = str(parse_config.out_path)
if(out_path=='None'):
    #input directory path
    print('out dir same as input dir')
    out_path = os.path.dirname(parse_config.ip_path)
    
# Load the input image
image_data_test_load = nib.load(img_path)
image_data_test_sys=image_data_test_load.get_data()
pixel_size=image_data_test_load.header['pixdim'][1:4]
affine_tst=image_data_test_load.affine
#image_data_test_sys=image_data_test_sys[:,:,:,0]

# Normalize input data using min-max normalization
image_data_test_sys=dt.normalize_minmax_data(image_data_test_sys)

#calculate axis on which the axial plane lies and change it to last axis
shape_val = image_data_test_sys.shape
if(shape_val[0] < shape_val[2]):
    print('swap axis')
    image_data_test_sys = np.swapaxes(image_data_test_sys,0,2)

#dummy labels with zeros in native resolution
label_sys=np.zeros_like(image_data_test_sys)

#crop to defined resolution and dimensions
cropped_img_sys,cropped_mask_sys = dt.preprocess_data(image_data_test_sys, label_sys, pixel_size)

# Make directory for the test image mask output if it does not exist
pathlib.Path(out_path).mkdir(parents=True, exist_ok=True)

#inference on the loaded image
# Calc predicted segmentation & save it
pred_sf_mask = f1_util.calc_pred_sf_mask(sess, ae, cropped_img_sys)
#pred_sf_mask = f1_util.calc_pred_sf_mask_full(sess, ae, cropped_img_sys)
re_pred_mask_sys,_ = f1_util.reshape_img_and_f1_score(pred_sf_mask, label_sys, pixel_size)


#save the nifti segmentation file
array_img = nib.Nifti1Image(re_pred_mask_sys.astype(np.int16), affine_tst)
pred_filename = str(out_path)+'/pred_mask.nii.gz'
nib.save(array_img, pred_filename)


