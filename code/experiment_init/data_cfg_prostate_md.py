import sys

def train_data(no_of_tr_imgs,comb_of_tr_imgs):
    #print('train data')
    if(no_of_tr_imgs=='tr5' and comb_of_tr_imgs=='c1'):
        labeled_id_list=["001","002","004","006","007"]
    #elif(no_of_tr_imgs=='tr5' and comb_of_tr_imgs=='c2'):
    #    labeled_id_list=["006","007","008","009","010"]
    #elif(no_of_tr_imgs=='tr5' and comb_of_tr_imgs=='c3'):
    #    labeled_id_list=["011","012","013","014","017"]
    #elif(no_of_tr_imgs=='tr5' and comb_of_tr_imgs=='c4'):
    #    labeled_id_list=["017","018","003","002","006"]
    #elif(no_of_tr_imgs=='tr5' and comb_of_tr_imgs=='c5'):
    #    labeled_id_list=["001","009","010","012","018"]
    elif(no_of_tr_imgs=='tr3' and comb_of_tr_imgs=='c1'):
        labeled_id_list=["000","001","002"]
    elif(no_of_tr_imgs=='tr3' and comb_of_tr_imgs=='c2'):
        labeled_id_list=["006","007","010"]
    elif(no_of_tr_imgs=='tr3' and comb_of_tr_imgs=='c3'):
        labeled_id_list=["016","017","004"]
    elif(no_of_tr_imgs=='tr3' and comb_of_tr_imgs=='c4'):
        labeled_id_list=["000","006","004"]
    elif(no_of_tr_imgs=='tr3' and comb_of_tr_imgs=='c5'):
        labeled_id_list=["010","016","004"]
    elif(no_of_tr_imgs=='tr1' and comb_of_tr_imgs=='c1'):
        labeled_id_list=["004"]
    elif(no_of_tr_imgs=='tr1' and comb_of_tr_imgs=='c2'):
        labeled_id_list=["006"]
    elif(no_of_tr_imgs=='tr1' and comb_of_tr_imgs=='c3'):
        labeled_id_list=["002"]
    elif(no_of_tr_imgs=='tr1' and comb_of_tr_imgs=='c4'):
        labeled_id_list=["000"]
    elif(no_of_tr_imgs=='tr1' and comb_of_tr_imgs=='c5'):
        labeled_id_list=["007"]
    elif(no_of_tr_imgs=='tr18' and comb_of_tr_imgs=='c1'):
        labeled_id_list=["000","001","002","004","006","007",\
                         "010","013","014","016","017","020",\
                         "021","024","025","028","018","037"]
    elif(no_of_tr_imgs=='tr20' and comb_of_tr_imgs=='c1'):
        labeled_id_list=["000","001","002","004","006","007",\
                         "010","013","014","016","017","020",\
                         "021","024","025","028","018","037","031","042"]
    else:
        print('Error! Select valid combination of training images')
        sys.exit()
    return labeled_id_list

def val_data():
    #print('val data')
    #val_list=["013","014"]
    val_list=["031","032","035","038","040","029","034","041"]
    #val_list=["032","035","038","040","029","034","041"]
    return val_list

def tf_train_data():
    #print('tf_train_list data')
    tf_train_list=["000","001","002","004","006","007","010",\
                   "016","017"]
    return tf_train_list

def unlabeled_data():
    #print('unlabeled data')
    unlabeled_list=["003","005","008","009","011",\
                    "012","015","019","023","026",\
                    "027","030","033","036","045",\
                    "022","020","021","024","025"]
    return unlabeled_list

#def unlabeled_data_tr5():
#    #print('unlabeled data tr5')
#    unlabeled_list=["031","032","033","034","035"]
#    return unlabeled_list
#
#def unlabeled_data_tr10():
#    #print('unlabeled data tr10')
#    unlabeled_list=["031","032","033","034","035","036","037","038","039","040"]
#    return unlabeled_list
#
#def unlabeled_data_tr20():
#    #print('unlabeled data tr20')
#    unlabeled_list=["031","032","033","034","035","036","037","038","039","040",\
#                    "041","042","043","044","045","046","047","048","049","050"]
#    return unlabeled_list

def all_label_unl_data():
    #print('unlabeled data')
    unlabeled_list=["000","001","002","004","006","007","010",\
                    "016","017","020","021","024",\
                    "025","028","029","031","047"]
    return unlabeled_list

def test_data():
    #print('test data')
    #test_list=["028","029","031","034","035","038","039","040",\
    #           "042","043","044","046","047"]
    test_list=["039","042","043","044","046","047"]
    #test_list=["039","043","044","046","047"]
    return test_list
