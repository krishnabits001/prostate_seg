#contains models for segmentation project - SSL

import tensorflow as tf
import numpy as np

# Load layers and losses
from layers_bn import layersObj
layers = layersObj()

from losses import lossObj
loss = lossObj()

class modelObj:
    def __init__(self,cfg):
        self.img_size_x=cfg.img_size_x
        self.img_size_y=cfg.img_size_y
        self.num_classes=cfg.num_classes
        self.interp_val = cfg.interp_val
        self.img_size_flat=cfg.img_size_flat
        self.batch_size=cfg.batch_size

    def deform_net(self):
        # placeholders for the network
        x_tmp = tf.placeholder(tf.float32, shape=[self.batch_size, self.img_size_x, self.img_size_y, 1], name='x_tmp')
        v_tmp = tf.placeholder(tf.float32, shape=[self.batch_size, self.img_size_x, self.img_size_y, 2], name='v_tmp')
        y_tmp = tf.placeholder(tf.int32, shape=[self.batch_size, self.img_size_x, self.img_size_y], name='y_tmp')

        y_tmp_1hot = tf.one_hot(y_tmp,depth=self.num_classes)
        w_tmp = tf.contrib.image.dense_image_warp(image=x_tmp,flow=v_tmp,name='dense_image_warp_tmp')
        w_tmp_1hot = tf.contrib.image.dense_image_warp(image=y_tmp_1hot,flow=v_tmp,name='dense_image_warp_tmp_1hot')

        return {'x_tmp':x_tmp,'flow_v':v_tmp,'deform_x':w_tmp,'y_tmp':y_tmp,'y_tmp_1hot':y_tmp_1hot,'deform_y_1hot':w_tmp_1hot}

    def contrast_net(self):
        # placeholders for the network
        x_tmp = tf.placeholder(tf.float32, shape=[self.batch_size, self.img_size_x, self.img_size_y, 1], name='x_tmp')

        rd_cont = tf.image.random_contrast(x_tmp,lower=0.8,upper=1.2,seed=1)
        rd_brit = tf.image.random_brightness(x_tmp,max_delta=0.1,seed=1)
        c_ind=np.arange(0,int(self.batch_size/2),dtype=np.int32)
        b_ind=np.arange(int(self.batch_size/2),int(self.batch_size),dtype=np.int32)

        rd_fin = tf.concat((tf.gather(rd_cont,c_ind),tf.gather(rd_brit,b_ind)),axis=0)

        return {'x_tmp':x_tmp,'rd_fin':rd_fin,'rd_cont':rd_cont,'rd_brit':rd_brit}

    def unet(self,learn_rate_seg=0.001,fs_de=2,dsc_loss=1,en_1hot=0):

        no_filters=[1, 16, 32, 64, 128, 256]
        #default U-Net filters
        #no_filters = [1, 64, 128, 256, 512, 1024]

        if(self.num_classes==2):
            class_weights = tf.constant([[0.05, 0.95]],name='class_weights')
        elif(self.num_classes==3):
            class_weights = tf.constant([[0.05, 0.5, 0.45]],name='class_weights')
        elif(self.num_classes==4):
            class_weights = tf.constant([[0.1, 0.3, 0.3, 0.3]],name='class_weights')

        num_channels=no_filters[0]
        # placeholders for the network
        # Inputs
        x = tf.placeholder(tf.float32, shape=[None, self.img_size_x, self.img_size_y, num_channels], name='x')
        #x = tf.placeholder(tf.float32, shape=[None, None, None, num_channels], name='x')
        if(en_1hot==1):
            y_l = tf.placeholder(tf.float32, shape=[None, self.img_size_x, self.img_size_y,self.num_classes], name='y_l')
            #y_l = tf.placeholder(tf.float32, shape=[None, None, None, self.num_classes], name='y_l')
        else:
            y_l = tf.placeholder(tf.int32, shape=[None, self.img_size_x, self.img_size_y], name='y_l')
            #y_l = tf.placeholder(tf.int32, shape=[None, None, None], name='y_l')
        select_mask = tf.placeholder(tf.bool, name='select_mask')
        train_phase = tf.placeholder(tf.bool, name='train_phase')

        if(en_1hot==0):
            y_l_onehot=tf.one_hot(y_l,depth=self.num_classes)
        else:
            y_l_onehot=y_l

    ############################################
        #U-Net like Network
    ############################################
        #Encoder - Downsampling Path
    ############################################
        # 2x 3x3 conv and 1 maxpool
        # Level 1
        enc_c1_a = layers.conv2d_layer(ip_layer=x,name='enc_c1_a', num_filters=no_filters[1], use_relu=True, use_batch_norm=True, training_phase=train_phase)
        enc_c1_b = layers.conv2d_layer(ip_layer=enc_c1_a, name='enc_c1_b', num_filters=no_filters[1], use_relu=True, use_batch_norm=True, training_phase=train_phase)
        enc_c1_pool = layers.max_pool_layer2d(enc_c1_b, kernel_size=(2, 2), strides=(2, 2), padding="SAME", name='enc_c1_pool')

        # Level 2
        enc_c2_a = layers.conv2d_layer(ip_layer=enc_c1_pool,name='enc_c2_a', num_filters=no_filters[2], use_relu=True, use_batch_norm=True, training_phase=train_phase)
        enc_c2_b = layers.conv2d_layer(ip_layer=enc_c2_a, name='enc_c2_b', num_filters=no_filters[2], use_relu=True, use_batch_norm=True, training_phase=train_phase)
        enc_c2_pool = layers.max_pool_layer2d(enc_c2_b, kernel_size=(2, 2), strides=(2, 2), padding="SAME", name='enc_c2_pool')

        # Level 3
        enc_c3_a = layers.conv2d_layer(ip_layer=enc_c2_pool,name='enc_c3_a', num_filters=no_filters[3], use_relu=True, use_batch_norm=True, training_phase=train_phase)
        enc_c3_b = layers.conv2d_layer(ip_layer=enc_c3_a, name='enc_c3_b', num_filters=no_filters[3], use_relu=True, use_batch_norm=True, training_phase=train_phase)
        enc_c3_pool = layers.max_pool_layer2d(enc_c3_b, kernel_size=(2, 2), strides=(2, 2), padding="SAME", name='enc_c3_pool')

        # Level 4
        enc_c4_a = layers.conv2d_layer(ip_layer=enc_c3_pool,name='enc_c4_a', num_filters=no_filters[4], use_relu=True, use_batch_norm=True, training_phase=train_phase)
        enc_c4_b = layers.conv2d_layer(ip_layer=enc_c4_a, name='enc_c4_b', num_filters=no_filters[4], use_relu=True, use_batch_norm=True, training_phase=train_phase)
        enc_c4_pool = layers.max_pool_layer2d(enc_c4_b, kernel_size=(2, 2), strides=(2, 2), padding="SAME", name='enc_c4_pool')

        # Level 5 - 2x Conv
        enc_c5_a = layers.conv2d_layer(ip_layer=enc_c4_pool,name='enc_c5_a', num_filters=no_filters[5], use_relu=True, use_batch_norm=True, training_phase=train_phase)
        enc_c5_b = layers.conv2d_layer(ip_layer=enc_c5_a, name='enc_c5_b', num_filters=no_filters[5], use_relu=True, use_batch_norm=True, training_phase=train_phase)

        ########################
        # Decoder - Upsampling Path
        ########################
        #Upsample + 2x2 conv to half the no. of feature channels + SKIP connection (concate the conv. layers)
        # Level 5 - 1 upsampling layer + 1 conv op. + skip connection + 2x conv op.
        scale_val=2
        dec_up5 = layers.upsample_layer(ip_layer=enc_c5_b, method=self.interp_val, scale_factor=scale_val)
        dec_dc5 = layers.conv2d_layer(ip_layer=dec_up5,name='dec_dc5', kernel_size=(fs_de,fs_de),num_filters=no_filters[4], use_relu=True, use_batch_norm=True, training_phase=train_phase)
        dec_cat_c5 = tf.concat((dec_dc5,enc_c4_b),axis=3,name='dec_cat_c5')
        dec_c4_a = layers.conv2d_layer(ip_layer=dec_cat_c5,name='dec_c4_a', num_filters=no_filters[4], use_relu=True, use_batch_norm=True, training_phase=train_phase)
        dec_c4_b = layers.conv2d_layer(ip_layer=dec_c4_a,name='dec_c4_b', num_filters=no_filters[4], use_relu=True, use_batch_norm=True, training_phase=train_phase)

        # Level 4
        dec_up4 = layers.upsample_layer(ip_layer=dec_c4_b, method=self.interp_val, scale_factor=scale_val)
        dec_dc4 = layers.conv2d_layer(ip_layer=dec_up4,name='dec_dc4', kernel_size=(fs_de,fs_de),num_filters=no_filters[3], use_relu=True, use_batch_norm=True, training_phase=train_phase)
        dec_cat_c4 = tf.concat((dec_dc4,enc_c3_b),axis=3,name='dec_cat_c4')
        dec_c3_a = layers.conv2d_layer(ip_layer=dec_cat_c4,name='dec_c3_a', num_filters=no_filters[3], use_relu=True, use_batch_norm=True, training_phase=train_phase)
        dec_c3_b = layers.conv2d_layer(ip_layer=dec_c3_a,name='dec_c3_b', num_filters=no_filters[3], use_relu=True, use_batch_norm=True, training_phase=train_phase)

        # Level 3
        dec_up3 = layers.upsample_layer(ip_layer=dec_c3_b, method=self.interp_val, scale_factor=scale_val)
        dec_dc3 = layers.conv2d_layer(ip_layer=dec_up3,name='dec_dc3', kernel_size=(fs_de,fs_de),num_filters=no_filters[2],use_relu=True, use_batch_norm=True, training_phase=train_phase)
        dec_cat_c3 = tf.concat((dec_dc3,enc_c2_b),axis=3,name='dec_cat_c3')
        dec_c2_a = layers.conv2d_layer(ip_layer=dec_cat_c3,name='dec_c2_a', num_filters=no_filters[2], use_relu=True, use_batch_norm=True, training_phase=train_phase)
        dec_c2_b = layers.conv2d_layer(ip_layer=dec_c2_a,name='dec_c2_b', num_filters=no_filters[2], use_relu=True, use_batch_norm=True, training_phase=train_phase)

        # Level 2
        dec_up2 = layers.upsample_layer(ip_layer=dec_c2_b, method=self.interp_val, scale_factor=scale_val)
        dec_dc2 = layers.conv2d_layer(ip_layer=dec_up2,name='dec_dc2', kernel_size=(fs_de,fs_de),num_filters=no_filters[1], use_relu=True, use_batch_norm=True, training_phase=train_phase)
        dec_cat_c2 = tf.concat((dec_dc2,enc_c1_b),axis=3,name='dec_cat_c2')
        dec_c1_a = layers.conv2d_layer(ip_layer=dec_cat_c2,name='dec_c1_a', num_filters=no_filters[1], use_relu=True, use_batch_norm=True, training_phase=train_phase)

        # Level 1
        seg_c1_a = layers.conv2d_layer(ip_layer=dec_c1_a,name='seg_c1_a',num_filters=no_filters[1], use_relu=True, use_batch_norm=True, training_phase=train_phase)
        seg_c1_b = layers.conv2d_layer(ip_layer=seg_c1_a,name='seg_c1_b', num_filters=no_filters[1], use_relu=True, use_batch_norm=True, training_phase=train_phase)
        seg_c1_c = layers.conv2d_layer(ip_layer=seg_c1_b,name='seg_c1_c', num_filters=no_filters[1], use_relu=True, use_batch_norm=True, training_phase=train_phase)

        #Final output layer - Logits before softmax
        seg_fin_layer = layers.conv2d_layer(ip_layer=seg_c1_c,name='seg_fin_layer', num_filters=self.num_classes,use_relu=False, use_batch_norm=False, training_phase=train_phase)

        # Predict Class
        y_pred = tf.nn.softmax(seg_fin_layer)
        y_pred_cls = tf.argmax(y_pred,axis=3)

        ########################
        # Simple Cross Entropy (CE) between predicted labels and true labels
        if(dsc_loss==1):
            # For dice score loss function
            #without background
            seg_cost = loss.dice_loss_without_backgrnd(logits=seg_fin_layer, labels=y_l_onehot)
            #with background
            #seg_cost = dice_loss_with_backgrnd(logits=seg_fin_layer, labels=y_l_onehot)
        else:
            # For Weighted Cross Entropy loss function with background
            seg_cost = loss.pixel_wise_cross_entropy_loss_weighted(logits=seg_fin_layer, labels=y_l_onehot, class_weights=class_weights)

        # var list of u-net (segmentation net)
        seg_net_vars = []
        for v in tf.trainable_variables():
            var_name = v.name
            if 'enc_' in var_name: seg_net_vars.append(v)
            elif 'dec_' in var_name: seg_net_vars.append(v)
            elif 'seg_' in var_name: seg_net_vars.append(v)

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            cost_seg=tf.reduce_mean(seg_cost)
            optimizer_unet_seg = tf.train.AdamOptimizer(learn_rate_seg).minimize(cost_seg,var_list=seg_net_vars)

        seg_summary = tf.summary.scalar('seg_cost', tf.reduce_mean(seg_cost))
        # Merge all the summaries and write them out to /tmp/mnist_logs (by default)
        train_summary = tf.summary.merge([seg_summary])
        # For dice score summary
        rv_dice = tf.placeholder(tf.float32, shape=[], name='rv_dice')
        rv_dice_summary = tf.summary.scalar('rv_val_dice', rv_dice)
        myo_dice = tf.placeholder(tf.float32, shape=[], name='myo_dice')
        myo_dice_summary = tf.summary.scalar('myo_val_dice', myo_dice)
        lv_dice = tf.placeholder(tf.float32, shape=[], name='lv_dice')
        lv_dice_summary = tf.summary.scalar('lv_val_dice', lv_dice)

        mean_dice = tf.placeholder(tf.float32, shape=[], name='mean_dice')
        mean_dice_summary = tf.summary.scalar('mean_val_dice', mean_dice)

        val_dsc_summary = tf.summary.merge([mean_dice_summary,rv_dice_summary,myo_dice_summary,lv_dice_summary])

        val_totalc = tf.placeholder(tf.float32, shape=[], name='val_totalc')
        val_totalc_sum= tf.summary.scalar('val_totalc_', val_totalc)
        val_summary = tf.summary.merge([val_totalc_sum])

        return {'x': x, 'y_l':y_l, 'train_phase':train_phase,'select_mask': select_mask,'seg_cost': cost_seg, \
                'y_pred' : y_pred, 'y_pred_cls': y_pred_cls, 'optimizer_unet_seg':optimizer_unet_seg,\
                'train_summary':train_summary,'seg_fin_layer':seg_fin_layer, \
                'rv_dice':rv_dice,'myo_dice':myo_dice,'lv_dice':lv_dice,'mean_dice':mean_dice,'val_dsc_summary':val_dsc_summary,\
                'val_totalc':val_totalc,'val_summary':val_summary}


