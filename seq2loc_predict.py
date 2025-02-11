import numpy as np
import pandas as pd
from tensorflow.keras import layers
from tensorflow.keras import optimizers
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
from hier_attention_mask import Attention
from tensorflow.keras import backend as K
from tensorflow.keras.metrics import categorical_accuracy, binary_crossentropy
import os
import argparse
import sys
from utils import *


def process_eachseq(seq,pssmfile,mask_seq,new_pssms):
    seql = len(seq)
    if os.path.exists(pssmfile):
        print("found " + pssmfile + "\n")
        pssm = readPSSM(pssmfile)
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
    

def endpad(df, pssmdir="", coarse=10, fine=8):
    ids = df.uniprot_id.to_numpy()
    if "Sequence" in df.columns:
        seqs = df.Sequence.to_list()
    else:
        seqs = df.sequence.to_list()
    new_pssms = []
    mask_seq = []
    for i, seq in enumerate(seqs):
        id = ids[i]
        pssmfile = f"{pssmdir}{id}_pssm.txt"
        process_eachseq(seq,pssmfile,mask_seq,new_pssms)
    x = np.array(new_pssms)
    mask = np.array(mask_seq)
    return [x,mask]


def main():
    parser=argparse.ArgumentParser(description='MULocDeep: interpretable protein localization classifier at sub-cellular and sub-organellar levels')
    parser.add_argument('-input',  dest='inputfile', type=str, help='csv files with uniprot_ids and seqs', required=True)
    parser.add_argument('-model_dir', type=str,
                        help='path to directory with model weights', required=False, default="")
    parser.add_argument('-existPSSM', dest='existPSSM', type=str,
                        help='the name of the existing PSSM directory if there is one.', required=True, default="")
    parser.add_argument('-savename', type=str,
                        help='path to save with predictions', required=False, default="")
    parser.add_argument('--gpu', dest='core', action='store_true',
                        help='Use gpu for prediction.', required=False)
    parser.add_argument('--cpu', dest='core', action='store_false',
                        help='Use cpu for prediction.', required=False)
    parser.add_argument('--crossval', action='store_true',
                        help='Eval on validation fols', required=False)
    parser.add_argument('--numfolds', type=int, default=5, required=False)
    parser.add_argument('--coarse', type=int, default=7, required=False)
    parser.add_argument('--fine', type=int, default=6, required=False)
    parser.add_argument('--numclasses', type=int, default=21, required=False)
    parser.set_defaults(feature=True)
    args = parser.parse_args()
    inputfile=args.inputfile
    model_dir = args.model_dir
    existPSSM=args.existPSSM
    savename=args.savename
    core=args.core
    crossval = args.crossval
    numfolds = args.numfolds
    coarse = args.coarse
    fine= args.fine
    numclasses = args.numclasses
    

    print(crossval)
    assert os.path.exists(existPSSM)
    os.makedirs(os.path.dirname(savename), exist_ok=True)
    if existPSSM!="":
       if not existPSSM[len(existPSSM) - 1] == "/":
         existPSSM = existPSSM + "/"


    df = pd.read_csv(inputfile)
    assert "uniprot_id" in df.columns
    if crossval: assert "fold" in df.columns
    else: #if using hous only want to execute endpad once
        test_x, test_mask = endpad(df, pssmdir=existPSSM, coarse=coarse, fine=fine)

    
    
    for foldnum in range(numfolds):
        #Get Data
        if crossval: 
            data = df[df.fold == foldnum] #get validation fold
            test_x, test_mask = endpad(data, pssmdir=existPSSM, coarse=coarse, fine=fine)


        if foldnum==0:
            att_matrix_N = np.zeros((numfolds, test_x.shape[0], 1000))
            cross_pred_small = np.zeros((test_x.shape[0], coarse, fine, numfolds))
            #Get model
            if core:
                print("using gpu")
                model_big, model_small = singlemodel(test_x, coarse=coarse, fine=fine)
            else:
                model_big, model_small = singlemodel_cpu(test_x, coarse=coarse, fine=fine)
        model_small.load_weights(f"{model_dir}/fold{foldnum}_small_lv2_acc-weights.hdf5") #says "big" but call it small model

        cross_pred_small[:,:,:,foldnum] = model_small.predict([test_x, test_mask.reshape(-1, 1000, 1)])[0]
        model_att = Model(inputs=model_big.inputs, outputs=model_big.layers[-11].output[1])
        att_pred = model_att.predict([test_x, test_mask.reshape(-1, 1000, 1)])
        att_matrix_N[foldnum, :] = att_pred.sum(axis=1) / numclasses
    
    print(att_matrix_N.shape)
    print(cross_pred_small.shape)
    ids = df.uniprot_id.to_numpy()
    print(ids.shape)

    #SAVE cross_pred_small and att_matrix_N
    np.savez(savename, preds=cross_pred_small, attn=att_matrix_N, ids=ids)

if __name__ == "__main__":
    main()

