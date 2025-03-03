
import os
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import multilabel_confusion_matrix, confusion_matrix
import math
from itertools import product
import argparse
import sys
from utils import *
import calendar
import time
from tensorflow.python.client import device_lib
import pandas as pd
import yaml


def load_config(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def make_input_files(trainset, lv1_dir, lv2_dir, mappings, num_folds):

    def write_labels(filename, labels):
        with open(filename, "w") as f:
            for label in labels:
                f.write(f"{label}\n")
                
    def write_fasta(fasta_name, keys, sequences):
        with open(fasta_name, "w") as f:
            for i, key in enumerate(keys):
                seq = sequences[i]
                f.write(f">{key}\n")
                f.write(f"{seq}\n")

    def mulocdeep_map(locations, mapping, level):
        return ";".join([str(mapping[level][loc]) for loc in locations.split(";")])
        
    trainset = pd.read_csv(trainset)
    mappings = load_config(mappings)
    assert lv1_dir == lv2_dir
    os.makedirs(lv1_dir, exist_ok=True)

    for i in range(num_folds):
        train = trainset[trainset.fold != i]
        val = trainset[trainset.fold == i]

        #Will train 2 models one for level 1+3 and other for level 1+2
        #level1 is "lv2" in MULocDeep 
        # and level2/3 is "lv1" in MULocDeep
        for j in [2,3]:
            for k in [1,2]:

                #Sequences
                filename = f"{lv1_dir}/lv{k}_train_fold{i}_seq"
                if not os.path.exists(filename):
                    write_fasta(
                        filename,
                        #f"data/{dir_name}/level1_{j}/lv{k}_train_fold{i}_seq", 
                        train.uniprot_id.to_list(),
                        train.sequence.to_list())
                    
                filename = f"{lv1_dir}/lv{k}_val_fold{i}_seq"
                if not os.path.exists(filename):
                    write_fasta(
                        filename,
                        #f"data/{dir_name}/level1_{j}/lv{k}_val_fold{i}_seq", 
                        val.uniprot_id.to_list(),
                        val.sequence.to_list())
                    
                #Labels
                filename = f"{lv1_dir}/lv{k}_train_fold{i}_lab"
                if not os.path.exists(filename):
                    write_labels(
                        filename,
                        #f"data/MULocDeep/level1_{j}/lv{k}_train_fold{i}_lab",
                        train.level1.apply(lambda x: mulocdeep_map(x, mappings, f"level1_{j}"))
                        )
                
                filename = f"{lv1_dir}/lv{k}_val_fold{i}_lab"
                if not os.path.exists(filename):
                    write_labels(
                        filename,
                        #f"data/MULocDeep/level1_{j}/lv{k}_val_fold{i}_lab",
                        val.level1.apply(lambda x: mulocdeep_map(x, mappings, f"level1_{j}"))
                        )



def process_eachseq(seq,pssmfile,mask_seq,new_pssms):
    seql = len(seq)
    if os.path.exists(pssmfile):
        print("found " + pssmfile + "\n")
        pssm = readPSSM(pssmfile)
        if pssm.shape[0] != seql:
            print(pssmfile + " is wrong shape\n")
            print("using Blosum62\n")
            pssm = convertSampleToBlosum62(seq)
    else:
        print("using Blosum62\n")
        pssm = convertSampleToBlosum62(seq)
    pssm = pssm.astype(float)
    PhyChem = convertSampleToPhysicsVector_pca(seq)
    pssm = np.concatenate((PhyChem, pssm), axis=1)
    print(id)
    if seql <= 1000:
        padnum = 1000 - seql
        padmatrix = np.zeros([padnum, 25])
        pssm = np.concatenate((pssm, padmatrix), axis=0)
        new_pssms.append(pssm)
        mask_seq.append(gen_mask_mat(seql, padnum))
    else:
        pssm = np.concatenate((pssm[0:500, :], pssm[seql - 500:seql, :]), axis=0)
        new_pssms.append(pssm)
        mask_seq.append(gen_mask_mat(1000, 0))
    

def endpad(seqfile, labelfile, pssmdir="", npzfile = "", coarse=10, fine=8):
    if not os.path.exists(npzfile):
        new_pssms = []
        labels = []
        mask_seq = []
        ids=[]
        f = open(seqfile, "r")
        f2 = open(labelfile, "r")
        index=0
        for line in f:
            
            if ">"in line:
                if index!=0:
                    process_eachseq(seq,pssmfile,mask_seq,new_pssms)
                pssmfile = pssmdir + line[1:].strip() + "_pssm.txt"
                label = f2.readline().strip()
                labels.append(label)
                seq=''
                id = line.strip()[1:]
                ids.append(id)
                
            else:
               seq+=line.strip()
            
            index+=1
        process_eachseq(seq,pssmfile,mask_seq,new_pssms)
        x = np.array(new_pssms)
        y = [convertlabels_to_categorical(i, coarse, fine) for i in labels] #replace with convertlabels_to_categorical_seq2loc - zoe
        y = np.array(y)
        mask = np.array(mask_seq)
        np.savez(npzfile, x=x, y=y, mask=mask, ids=ids)
        return [x, y, mask,ids]
    else:
        print(npzfile)
        mask = np.load(npzfile)['mask']
        x = np.load(npzfile)['x']
        y = np.load(npzfile)['y']
        ids=np.load(npzfile)['ids']
        return [x, y, mask,ids]



def train_MULocDeep(lv1_dir,lv2_dir,pssm_dir,output_dir,foldnum,coarse=10,fine=10):
    os.makedirs(lv2_dir+"/npzfiles", exist_ok=True)
    os.makedirs(lv1_dir+"/npzfiles", exist_ok=True)
    # get small data = #lv2 annotation = more finegrained = matrix output = (10,8)
    [train_x, train_y, train_mask, train_ids] = endpad(
        lv2_dir+"lv2_train_fold" + str(foldnum) + "_seq",
        lv2_dir+"lv2_train_fold" + str(foldnum) + "_lab",
        pssm_dir,
        lv2_dir+"npzfiles/lv2_train_fold"+str(foldnum)+"_seq.npz",
        coarse, 
        fine)
    [val_x, val_y, val_mask,val_ids] = endpad(
        lv2_dir+"lv2_val_fold" + str(foldnum) + "_seq",
        lv2_dir+"lv2_val_fold" + str(foldnum) + "_lab",
        pssm_dir,
        lv2_dir+"npzfiles/lv2_val_fold"+str(foldnum)+"_seq.npz",
        coarse, 
        fine)

    # get big data = #lv1 annotation = more coarsegrained = vector output (10,1)
    [train_x_big, train_y_big, train_mask_big, train_ids_big] = endpad(
        lv1_dir + "lv1_train_fold" + str(foldnum) + "_seq",
        lv1_dir + "lv1_train_fold" + str(foldnum) + "_lab",
        pssm_dir,
        lv1_dir+"npzfiles/lv1_train_fold" + str(foldnum) + "_seq.npz",
        coarse, 
        fine)

    [val_x_big, val_y_big, val_mask_big, val_ids_big] = endpad(
        lv1_dir + "lv1_val_fold" + str(foldnum) + "_seq",
        lv1_dir + "lv1_val_fold" + str(foldnum) + "_lab",
        pssm_dir,
        lv1_dir+"npzfiles/lv1_val_fold" + str(foldnum) + "_seq.npz",
        coarse, 
        fine)

    batch_size = 128
    print("doing " + str(foldnum) + "th fold")
    model_big, model_small = singlemodel(train_x, coarse=coarse, fine=fine)

    filepath_acc_big_lv1 = output_dir+"fold" + str(
        foldnum) + "_big_lv1_acc-weights.hdf5"  # -improvement-{epoch:02d}-{val_loss:.2f}.hdf5"

    filepath_acc_small_lv2 = output_dir+"fold" + str(
        foldnum) + "_small_lv2_acc-weights.hdf5"  # -improvement-{epoch:02d}-{val_loss:.2f}.hdf5"

    filepath_loss_big_lv1 = output_dir+"fold" + str(
        foldnum) + "_big_lv1_loss-weights.hdf5"  # -improvement-{epoch:02d}-{val_loss:.2f}.hdf5"

    filepath_loss_small_lv2 = output_dir+"fold" + str(
        foldnum) + "_small_lv2_loss-weights.hdf5"  # -improvement-{epoch:02d}-{val_loss:.2f}.hdf5"


    checkpoint_acc_big_lev1 = ModelCheckpoint(filepath_acc_big_lv1, monitor='val_accuracy', save_best_only=True,
                                          mode='max',
                                          save_weights_only=True, verbose=1)

    checkpoint_acc_small_lev2 = ModelCheckpoint(filepath_acc_small_lv2, monitor='val_lev2_accuracy', save_best_only=True,
                                          mode='max',
                                          save_weights_only=True, verbose=1)
    
    checkpoint_loss_big_lev1 = ModelCheckpoint(filepath_loss_big_lv1, monitor='val_loss', save_best_only=True,
                                          mode='min',
                                          save_weights_only=True, verbose=1)
    
    checkpoint_loss_small_lev2 = ModelCheckpoint(filepath_loss_small_lv2, monitor='val_lev2_loss', save_best_only=True,
                                          mode='min',
                                          save_weights_only=True, verbose=1)
    

    for i in range(25): #was 80
        # train small model
        print("epoch "+str(i)+"\n")
        fitHistory_batch_small = model_small.fit([train_x, train_mask.reshape(-1, 1000, 1)],
                                                 [train_y,getTrue4out1(train_y)],
                                                 batch_size=batch_size, epochs=1,
                                                 validation_data=(
                                                 [val_x, val_mask.reshape(-1, 1000, 1)], [val_y,getTrue4out1(val_y)]),
                                                 callbacks=[checkpoint_acc_small_lev2,checkpoint_loss_small_lev2],verbose=1)
        
        # train big model
        fitHistory_batch_big = model_big.fit([train_x_big, train_mask_big.reshape(-1, 1000, 1)],
                                             getTrue4out1(train_y_big),
                                             batch_size=batch_size, epochs=1,
                                             validation_data=(
                                             [val_x_big, val_mask_big.reshape(-1, 1000, 1)], [getTrue4out1(val_y_big)]),
                                             callbacks=[checkpoint_acc_big_lev1,checkpoint_loss_big_lev1], verbose=1)



def train_var(input_var,pssm_dir,output_dir,foldnum):
    # get small data
    [train_x,train_y,train_mask,train_ids]=endpad(input_var+"deeploc_40nr_train_fold"+str(foldnum)+"_seq",
                                        input_var+"deeploc_40nr_train_fold"+str(foldnum)+"_label",
                                        pssm_dir,
                                        "./data/npzfiles/train_fold"+str(foldnum)+"_seq.npz")
    [val_x,val_y,val_mask,val_ids]=endpad(input_var+"deeploc_40nr_val_fold"+str(foldnum)+"_seq",
                                  input_var+"deeploc_40nr_val_fold"+str(foldnum)+"_label",
                                  pssm_dir,
                                  "./data/npzfiles/val_fold"+str(foldnum)+"_seq.npz")
    batch_size = 128
    print("doing " + str(foldnum) + "th fold")
    model = var_model(train_x)

    filepath_acc = output_dir+"fold" + str(
        foldnum) + "acc-weights.hdf5"  # -improvement-{epoch:02d}-{val_loss:.2f}.hdf5"
    checkpoint_acc = ModelCheckpoint(filepath_acc, monitor='val_accuracy', save_best_only=True, mode='max',
                                 save_weights_only=True, verbose=1)
    fitHistory_batch = model.fit([train_x,train_mask.reshape(-1,1000,1)],getTrue4out1(train_y),
                                 batch_size=batch_size, epochs=60,
                                 validation_data=([val_x,val_mask.reshape(-1,1000,1)], getTrue4out1(val_y)),
                                 callbacks=[checkpoint_acc],verbose=1)




def main():
    parser=argparse.ArgumentParser(description='MULocDeep: interpretable protein localization classifier at sub-cellular and sub-organellar levels')
    parser.add_argument('--trainset', type=str, help='csv with protein subcellular locations and sequences', required=False, default=None)
    parser.add_argument('--mapping', type=str, help='yaml that maps locations to decimal representation', required=False, default=None)
    parser.add_argument('--lv1_input_dir', dest='lv1_dir', type=str, 
                        help='sub-cellular training data, contains folds of protein sequences and labels', required=False)
    parser.add_argument('--lv2_input_dir', dest='lv2_dir', type=str,
                        help='sub-cellular training data, contains folds of protein sequences and labels', required=False)
    parser.add_argument('--input_dir', dest='var_dir', type=str,
                        help='data for traing the variant model, contains folds of protein sequences and labels', required=False)
    parser.add_argument('--MULocDeep_model', dest='modeltype', action='store_true',
                        help='Add this to train the MULocDeep model, otherwise train a variant model', required=False)
    parser.add_argument('--model_output', dest='outputdir', type=str, help='the name of the directory where the trained model stores', required=True)
    parser.add_argument('--existPSSM', dest='existPSSM', type=str,
                        help='the name of the existing PSSM directory if there is one.', required=False, default="")
    parser.add_argument('--numfolds', type=int,
                        help='number of cross validation folds', required=False, default=8)
    parser.add_argument('--coarse', type=int,
                        help='number of cross validation folds', required=False, default=10)
    parser.add_argument('--fine', type=int,
                        help='number of cross validation folds', required=False, default=8)
    parser.add_argument('--db', type=str,
                        help='database to construct pssms from', required=False)
    parser.set_defaults(feature=True)
    args = parser.parse_args()
    model_type=args.modeltype
    trainset = args.trainset
    mapping = args.mapping
    input_lv1=args.lv1_dir
    input_lv2 = args.lv2_dir
    input_var=args.var_dir
    outputdir=args.outputdir
    existPSSM = args.existPSSM
    numfolds = args.numfolds
    coarse_numclasses = args.coarse
    fine_numclasses = args.fine
    db = args.db

    print(device_lib.list_local_devices())

    #Make input files if they don't already exist
    if trainset is not None:
        make_input_files(trainset, input_lv1, input_lv2, mapping, numfolds)

    if model_type==True:
        if not input_lv1[len(input_lv1) - 1] == "/":
            input_lv1 = input_lv1 + "/"
        if not input_lv2[len(input_lv2) - 1] == "/":
            input_lv2 = input_lv2 + "/"
        if not outputdir[len(outputdir) - 1] == "/":
            outputdir = outputdir + "/"
        if not os.path.exists(outputdir):
            os.makedirs(outputdir)
        if existPSSM != "":
            if not existPSSM[len(existPSSM) - 1] == "/":
                existPSSM = existPSSM + "/"
        if ((existPSSM == "") or (not os.path.exists(existPSSM))):
            ts = calendar.timegm(time.gmtime())
            pssmdir = outputdir + str(ts) + "_pssm/"
            if not os.path.exists(pssmdir):
                os.makedirs(pssmdir)
            for foldnum in range(numfolds): #change to 5 - zoe
                process_input_train(input_lv1 + "lv1_train_fold" + str(foldnum) + "_seq", pssmdir, db)
                process_input_train(input_lv1 + "lv1_val_fold" + str(foldnum) + "_seq", pssmdir, db)
                process_input_train(input_lv2 + "lv2_train_fold" + str(foldnum) + "_seq", pssmdir, db)
                process_input_train(input_lv2 + "lv2_val_fold" + str(foldnum) + "_seq", pssmdir, db)
                train_MULocDeep(input_lv1, input_lv2, pssmdir, outputdir, foldnum, coarse=coarse_numclasses, fine=fine_numclasses)
        else:
            for foldnum in range(numfolds): #change to 5 - zoe
                train_MULocDeep(input_lv1, input_lv2, existPSSM, outputdir, foldnum, coarse=coarse_numclasses, fine=fine_numclasses)
    elif model_type==False:
        if not input_var[len(input_var) - 1] == "/":
            input_var = input_var + "/"
        if not outputdir[len(outputdir) - 1] == "/":
            outputdir = outputdir + "/"
        if not os.path.exists(outputdir):
            os.makedirs(outputdir)
        if existPSSM != "":
            if not existPSSM[len(existPSSM) - 1] == "/":
                existPSSM = existPSSM + "/"
        if ((existPSSM == "") or (not os.path.exists(existPSSM))):
            ts = calendar.timegm(time.gmtime())
            pssmdir = outputdir + str(ts) + "_pssm/"
            if not os.path.exists(pssmdir):
                os.makedirs(pssmdir)
            for foldnum in range(numfolds): #change to 5 - zoe
                process_input_train(input_var+"deeploc_40nr_train_fold" + str(foldnum) + "_seq", pssmdir, db)
                process_input_train(input_var + "deeploc_40nr_var_fold" + str(foldnum) + "_seq", pssmdir, db)
                train_var(input_var, pssmdir, outputdir, foldnum)
        else:
            for foldnum in range(numfolds): #change to 5 - zoe
                train_var(input_var, existPSSM, outputdir, foldnum)



if __name__ == "__main__":
    main()

