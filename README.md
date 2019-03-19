# prostate_seg
for segmentation of prostate

Requirements:</br>
Python 3.6.0,</br>
Tensorflow 1.8.0,</br>
rest of the requirements are mentioned in the "requirements.txt" file.

I) To clone the git repository.</br>
git clone https://github.com/krishnabits001/prostate_seg.git </br>

II) Install python, required packages and tensorflow.</br>
Then, install python packages required using below command or the packages mentioned in the file.</br>
pip install -r requirements.txt </br>

To install tensorflow </br>
pip install tensorflow-gpu=1.8.0 </br>


III) Config files contents : One can modify the contents of the below config file to run the required inference.</br>
code/experiment_init/init_prostate_md.py </br>
We can set the target resolution and image dimensions here. </br>
Target resolution of (0.625,0.625) and image dimensions of (256,256) used for training of the model are mentioned in this file. </br>
        
IV) For inference to get segmentation mask on an input test image use the below command (also mentioned in the code/train_model/inf_script.sh) : </br>
Specify the path of the input image in the variable "ip_path" as specified below.</br>
The output segmentation mask would be stored in the same path as "ip_path" with the name "pred_mask.nii.gz". 
Alternatively if you want the segmentation mask to be stored in a different path you can specify it in the variable "out_path" at run time. </br>

cd code/train_model/ </br>
python inference.py --ip_path='/usr/krishnch/datasets/prostate/001/img.nii.gz' </br>


