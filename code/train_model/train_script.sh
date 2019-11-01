#For medical decatholon challenge data - public data
#python tr_unet_baseline.py --dataset=prostate_md --no_of_tr_imgs=tr18 --comb_tr_imgs=c1 --lr_seg=0.001 --ver=0

#For GE scanner private data
python tr_unet_baseline.py --dataset=prostate_ge --no_of_tr_imgs=tr6 --comb_tr_imgs=c1 --lr_seg=0.001 --ver=0

