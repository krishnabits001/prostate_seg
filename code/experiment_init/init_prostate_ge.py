################################################################
# Definitions required for CNN graph
################################################################
#Interpolation type and up scaling factor
interp_val=0 # 0 - bilinear interpolation; 1- nearest neighbour interpolation
################################################################

################################################################
# data dimensions, num of classes and resolution
################################################################
#Data Dimensions
img_size_x = 448#416
img_size_y = 448#416
# Images are stored in one-dimensional arrays of this length.
img_size_flat = img_size_x * img_size_y
# Number of colour channels for the images: 1 channel for gray-scale.
num_channels = 1
# Number of classes : # 0-background, 1-pz, 2-cz (merge both the foreground labels into 1)
#num_classes=3
num_classes=2
size=(img_size_x,img_size_y)
#target_resolution=(0.625,0.625)
target_resolution=(0.3906,0.3906)
################################################################
#data paths
################################################################
#validation_update_step to save values
val_step_update=10
#base dir of network
base_dir='/usr/bmicnas01/data-biwi-01/krishnch/projects/prostate_seg/'
#data path tr
data_path_tr='/usr/bmicnas01/data-biwi-01/krishnch/datasets/prostate_md_tmp/orig/'
#cropped imgs data_path
#data_path_tr_cropped='/usr/bmicnas01/data-biwi-01/krishnch/datasets/prostate_md/orig_cropped/'
#slic on orig imgs data_path
#slic_path_tr_cropped='/usr/bmicnas01/data-biwi-01/krishnch/datasets/prostate_md/orig_slic/'
################################################################

################################################################
#network optimization parameters
################################################################
#enable data augmentation
aug_en=1
#batch_size
batch_size=20
struct_name=['pz']
#struct_name=['pz','tz']
