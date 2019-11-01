import sys

def train_data(no_of_tr_imgs,comb_of_tr_imgs):
    #print('train data')
    if(no_of_tr_imgs=='tr6' and comb_of_tr_imgs=='c1'):
        labeled_id_list=["001","002","003","004","005","006"]
    else:
        print('Error! Select valid combination of training images')
        sys.exit()
    return labeled_id_list

def val_data():
    #print('val data')
    val_list=["007","008","009","010"]
    return val_list

def test_data():
    #print('test data')
    test_list=["001","002","003","004","005","006","007","008","009","010"]
    return test_list
