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

import sys
sys.path.append('../')

from utils import *

import argparse
parser = argparse.ArgumentParser()
#data set type
parser.add_argument('--dataset', type=str, default='prostate_ge', choices=['prostate_md','prostate_ge'])
#no of training images
parser.add_argument('--no_of_tr_imgs', type=str, default='tr6', choices=['tr1', 'tr6', 'tr10', 'tr18', 'tr20', 'tr40'])
#combination of training images
parser.add_argument('--comb_tr_imgs', type=str, default='c1', choices=['c1', 'c2', 'c3', 'c4', 'c5'])
#learning rate of seg unet
parser.add_argument('--lr_seg', type=float, default=0.001)

#data aug - 0 - disabled, 1 - enabled
parser.add_argument('--data_aug', type=int, default=1, choices=[0,1])

#sigma for deformation fields gen
parser.add_argument('--sigma', type=float, default=5)
#random contrast enable
parser.add_argument('--contrast_en', type=float, default=0)

#version of run
parser.add_argument('--ver', type=int, default=0)

# segmentation loss to optimize
# 0 for weighted cross entropy, 1 for dice score loss
parser.add_argument('--dsc_loss', type=int, default=0)

parse_config = parser.parse_args()

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

if parse_config.dataset == 'prostate_md':
    #print('set prostate_md orig img dataloader handle')
    orig_img_dt=dt.load_prostate_imgs
elif parse_config.dataset == 'prostate_ge':
    #print('set prostate_ge orig img dataloader handle')
    orig_img_dt=dt.load_prostate_imgs

#  load model object
from models import modelObj
model = modelObj(cfg)
#  load f1_utils object
from f1_utils import f1_utilsObj
f1_util = f1_utilsObj(cfg,dt)

######################################
#define save_dir for the model
if(parse_config.contrast_en==1):
    save_dir=str(cfg.base_dir)+'/models/'+str(parse_config.dataset)+'/tr_baseline_unet_rand_deform_contrast/'
else:
    save_dir=str(cfg.base_dir)+'/models/'+str(parse_config.dataset)+'/tr_baseline_unet_rand_deform/'

if(parse_config.data_aug==0):
    save_dir=str(save_dir)+'/no_data_aug/'
    cfg.aug_en=parse_config.data_aug
else:
    save_dir=str(save_dir)+'/with_data_aug/'

save_dir=str(save_dir)+str(parse_config.no_of_tr_imgs)+'/'+str(parse_config.comb_tr_imgs)+\
         '_v'+str(parse_config.ver)+'/unet_dsc_'+str(parse_config.dsc_loss)+'_sig_'+str(parse_config.sigma)+'_lr_seg_'+str(parse_config.lr_seg)+'/'
print('save dir ',save_dir)
######################################

######################################
# load train and val images
train_list = data_list.train_data(parse_config.no_of_tr_imgs,parse_config.comb_tr_imgs)
#load train data cropped images directly
print('loading train imgs')#,train_list)
train_imgs,train_labels = load_imgs(dt=dt,orig_img_dt=orig_img_dt,test_list=train_list)

if(parse_config.no_of_tr_imgs=='tr1'):
    train_imgs_copy=np.copy(train_imgs)
    train_labels_copy=np.copy(train_labels)
    while(train_imgs.shape[2]<cfg.batch_size):
        train_imgs=np.concatenate((train_imgs,train_imgs_copy),axis=2)
        train_labels=np.concatenate((train_labels,train_labels_copy),axis=2)
    del train_imgs_copy,train_labels_copy

val_list = data_list.val_data()
#load both val data and its cropped images
print('loading val imgs')
val_label_orig,val_img_crop,val_label_crop,pixel_val_list=load_val_imgs(val_list,dt,orig_img_dt)

# get test list
print('get test imgs list')
test_list = data_list.test_data()
struct_name=cfg.struct_name
val_step_update=cfg.val_step_update
######################################

######################################
# Define checkpoint file to save CNN architecture and learnt hyperparameters
checkpoint_filename='unet_'+str(parse_config.dataset)
logs_path = str(save_dir)+'tensorflow_logs/'
best_model_dir=str(save_dir)+'best_model/'
######################################

######################################
#  training parameters
start_epoch=0
n_epochs = 10000
disp_step= 500
mean_f1_val_prev=0.1
threshold_f1=0.00001
pathlib.Path(best_model_dir).mkdir(parents=True, exist_ok=True)
######################################

######################################
# define deformation graph and session
tf.reset_default_graph()
df_ae_rd= model.deform_net()

if(parse_config.contrast_en==1):
    # contrast and brightness adding network
    df_ae_ri = model.contrast_net()

######################################
# define model graph
#tf.reset_default_graph()
ae = model.unet(learn_rate_seg=parse_config.lr_seg,dsc_loss=parse_config.dsc_loss,en_1hot=1)
######################################

######################################
#writer for train summary
train_writer = tf.summary.FileWriter(logs_path)
#writer for dice score and val summary
dsc_writer = tf.summary.FileWriter(logs_path)
val_sum_writer = tf.summary.FileWriter(logs_path)
######################################

######################################
# create a session and initialize variable to use the graph
sess = tf.Session(config=config)
sess.run(tf.global_variables_initializer())
# Save training data
saver = tf.train.Saver(max_to_keep=2)
######################################

# Run for n_epochs
for epoch_i in range(start_epoch,n_epochs):

    #sample Labelled shuffled batch
    ld_img_batch,ld_label_batch=shuffle_minibatch([train_imgs,train_labels],batch_size=cfg.batch_size,num_channels=cfg.num_channels,axis=2)

    if(cfg.aug_en==1):
        # Apply affine transformations
        ld_img_batch,ld_label_batch=augmentation_function([ld_img_batch,ld_label_batch],dt)

    #calc random deformation fields
    rand_deform_v = calc_deform(cfg,0,parse_config.sigma)
    ld_img_batch_tmp=np.copy(ld_img_batch)
    ld_label_batch_1hot = sess.run(df_ae_rd['y_tmp_1hot'],feed_dict={df_ae_rd['y_tmp']:ld_label_batch})
    #print('tmp label shape',ld_label_batch.shape)
    ld_label_batch_tmp=np.copy(ld_label_batch)
    ###########################
    # #use RAGAN model data to get data aug
    ##########################
    ld_img_batch = sess.run(df_ae_rd['deform_x'],feed_dict={df_ae_rd['x_tmp']:ld_img_batch_tmp,df_ae_rd['flow_v']:rand_deform_v})
    ld_label_batch=sess.run([df_ae_rd['deform_y_1hot']],feed_dict={df_ae_rd['y_tmp']:ld_label_batch_tmp,df_ae_rd['flow_v']:rand_deform_v})
    ld_label_batch=ld_label_batch[0]

    if(parse_config.contrast_en==1):
        #add random contrast and brightness over random deformations
        ld_img_batch,_=sess.run([df_ae_ri['rd_fin'],df_ae_ri['rd_cont']], feed_dict={df_ae_ri['x_tmp']: ld_img_batch})

    #pick number of deformed images
    no_orig=np.random.randint(5, 15)
    ld_img_batch[0:no_orig] = ld_img_batch_tmp[0:no_orig]
    ld_label_batch[0:no_orig] = ld_label_batch_1hot[0:no_orig]

    if(epoch_i==0):
        print('e',ld_img_batch.shape,ld_label_batch.shape)
    #Optimer the net on this batch
    train_summary,_ =sess.run([ae['train_summary'],ae['optimizer_unet_seg']], feed_dict={ae['x']: ld_img_batch,\
                                                            ae['y_l']: ld_label_batch,ae['train_phase']: True})

    if(epoch_i%val_step_update==0):
        train_writer.add_summary(train_summary, epoch_i)
        train_writer.flush()

    if(epoch_i%val_step_update==0):
        ##Save the model with best DSC for Validation Image
        mean_f1_arr=[]
        f1_arr=[]
        for val_id_no in range(0,len(val_list)):
            val_img_crop_tmp=val_img_crop[val_id_no]
            val_label_crop_tmp=val_label_crop[val_id_no]
            val_label_orig_tmp=val_label_orig[val_id_no]
            pixel_size_val=pixel_val_list[val_id_no]

            # Compute segmentation mask and dice_score for each validation subject
            pred_sf_mask = f1_util.calc_pred_sf_mask_full(sess, ae, val_img_crop_tmp)
            re_pred_mask_sys,f1_val = f1_util.reshape_img_and_f1_score(pred_sf_mask, val_label_orig_tmp, pixel_size_val)

            #concatenate dice scores of each val image
            mean_f1_arr.append(np.mean(f1_val[1:cfg.num_classes]))
            f1_arr.append(f1_val[1:cfg.num_classes])

        #avg mean over 2 val subjects
        mean_f1_arr=np.asarray(mean_f1_arr)
        mean_f1=np.mean(mean_f1_arr)
        f1_arr=np.asarray(f1_arr)

        if (mean_f1-mean_f1_val_prev>threshold_f1):
            print("prev f1_val; present_f1_val", mean_f1_val_prev, mean_f1, mean_f1_arr)
            mean_f1_val_prev = mean_f1
            # to save the best model with maximum dice score over the entire n_epochs
            print("best model saved at epoch no. ", epoch_i)
            mp_best = str(best_model_dir) + str(checkpoint_filename) + '_best_model_epoch_' + str(epoch_i) + ".ckpt"
            saver.save(sess, mp_best)

        #print('f1',f1_arr)
        #calc. and save validation image dice summary
        dsc_summary_msg = sess.run(ae['val_dsc_summary'], feed_dict={ae['rv_dice']:np.mean(f1_arr[:,0]),\
                                ae['myo_dice']:np.mean(f1_arr),ae['lv_dice']:np.mean(f1_arr),ae['mean_dice']: mean_f1})
        val_sum_writer.add_summary(dsc_summary_msg, epoch_i)
        val_sum_writer.flush()

    if ((epoch_i==n_epochs-1) and (epoch_i != start_epoch)):
        # model saved at last epoch
        mp = str(save_dir) + str(checkpoint_filename) + '_epochs_' + str(epoch_i) + ".ckpt"
        saver.save(sess, mp)
        try:
            mp_best
        except NameError:
            mp_best=mp

sess.close()
######################################
#if(parse_config.dataset=='prostate_md'):
# restore best model and predict segmentations on test subjects
saver_new = tf.train.Saver()
sess_new = tf.Session(config=config)
saver_new.restore(sess_new, mp_best)
print("best model chkpt",mp_best)
print("Model restored")

########################
# To compute inference on test images on the model that yields best dice score on validation images
f1_util.pred_segs_prostate_test_subjs(sess_new,ae,save_dir,orig_img_dt,test_list,struct_name)

if(parse_config.dataset=='prostate_md'):
    ######################################
    # To compute inference on validation images on the best model
    save_dir_tmp=str(save_dir)+'/val_imgs_dsc/'
    f1_util.pred_segs_prostate_test_subjs(sess_new,ae,save_dir_tmp,orig_img_dt,val_list,struct_name)
    ######################################
