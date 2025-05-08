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
import yaml
from metrics import *
import warnings
from Bio import BiopythonDeprecationWarning
warnings.simplefilter('ignore', BiopythonDeprecationWarning)

def load_config(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def get_fine_idxs(coarse, fine, mapping, level):
    m = np.zeros((coarse, fine))
    labels = np.full((coarse, fine), "", dtype=object)
    for label, idx in mapping[level].items():
        i,j = str(idx).split(".")
        i = int(i)
        j = int(j)
        m[i,j] = 1
        labels[i,j] = label
    # flatten will line up row after row
    # m is shape (coarse, fine) so we want cols to be lined up one after the other
    # Thus need to transpose before flattening
    m = m.flatten()
    labels = labels.flatten()
    idx = np.where(m == 1)[0]
    labels = [labels[i] for i in idx]
    return idx, labels

def get_y(df, level, categories):
    targets = []
    for locs in df[f"level{level}"].str.split(";").to_list():
        targets.append([1 if loc in locs else 0 for loc in categories])
    return np.array(targets)

def process_eachseq(seq,pssmfile,mask_seq,new_pssms):
    seql = len(seq)
    if os.path.exists(pssmfile):
        print("found " + pssmfile + "\n")
        pssm = readPSSM(pssmfile)
    else:
        print("using Blosum62\n")
        pssm = convertSampleToBlosum62(seq)
    if pssm.shape[0] != len(seq):
        print("Saved PSSM matrix wrong size\n")
        print("using Blosum62\n")
        pssm = convertSampleToBlosum62(seq)
    pssm = pssm.astype(float)
    PhyChem = convertSampleToPhysicsVector_pca(seq)
    pssm = np.concatenate((PhyChem, pssm), axis=1)
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
        pssmfile = f"{pssmdir}/{id}_pssm.txt"
        process_eachseq(seq,pssmfile,mask_seq,new_pssms)
    x = np.array(new_pssms)
    mask = np.array(mask_seq)
    return [x,mask]



def main():
    parser=argparse.ArgumentParser(description='MULocDeep: interpretable protein localization classifier at sub-cellular and sub-organellar levels')
    parser.add_argument('--crossval_csv', type=str, help='csv files with uniprot_ids and seqs', required=True)
    parser.add_argument('--test_csv', type=str, help='csv files with uniprot_ids and seqs', required=True)
    parser.add_argument('--model_dir', type=str,
                        help='path to directory with model weights', required=False, default="")
    parser.add_argument('--existPSSM', dest='existPSSM', type=str,
                        help='the name of the existing PSSM directory if there is one.', required=True, default="")
    parser.add_argument('--savedir', type=str,
                        help='path to save predictions and metrics', required=False, default="")
    parser.add_argument('--gpu', dest='core', action='store_true',
                        help='Use gpu for prediction.', required=False)
    parser.add_argument('--cpu', dest='core', action='store_false',
                        help='Use cpu for prediction.', required=False)
    parser.add_argument('--numfolds', type=int, default=5, required=False)
    parser.add_argument('--numclasses', type=int, default=21, required=False)
    parser.add_argument('--id_col', type=str, default="uniprot_id", required=False)
    parser.add_argument('--level', type=str, default="level1_3", required=True)
    parser.set_defaults(feature=True)
    
    args = parser.parse_args()
    crossval_csv=args.crossval_csv
    test_csv=args.test_csv
    model_dir = args.model_dir
    existPSSM=args.existPSSM
    savedir=args.savedir
    core=args.core
    numfolds = args.numfolds
    id_col = args.id_col
    level = args.level

    os.makedirs(savedir, exist_ok=True)

    if level == "level1_3":
        coarse = 8
        fine = 7
        fine_level = 1
        coarse_level = 3
    elif level == "level1_2":
        coarse = 10
        fine = 5
        fine_level = 1
        coarse_level = 2
    else:
        print("Level not recognized. Please use level1_2 or level1_3")
        sys.exit("Exiting program")

    
    trainset = pd.read_csv(crossval_csv)
    assert id_col in trainset.columns
    testset = pd.read_csv(test_csv)
    assert id_col in testset.columns
    test_x, test_mask = endpad(testset, pssmdir=existPSSM, coarse=coarse, fine=fine)

    #Get orde of categories
    mapping = load_config("./data/mulocdeep_mapping.yaml")
    fine_idxs, fine_labels = get_fine_idxs(coarse, fine, mapping, level)
    coarse_labels = mapping[f"level{coarse_level}"].keys()
    coarse_labels = np.array(list(coarse_labels))

    #Get Target Values for Test set
    test_y_coarse = get_y(testset, coarse_level, coarse_labels)
    test_y_fine = get_y(testset, fine_level, fine_labels)

    all_test_probs_fine = []
    all_test_probs_coarse= []
    all_test_preds_fine = []
    all_test_preds_coarse = []
    val_perclass_fine_df = []
    val_perclass_coarse_df = []
    val_avg_df = []
    
    for foldnum in range(numfolds):
        #Get Validation Set
        val_fold = trainset[trainset["fold"] == foldnum]
        val_x, val_mask = endpad(val_fold, pssmdir=existPSSM, coarse=coarse, fine=fine)
        val_y_coarse = get_y(val_fold, coarse_level, coarse_labels)
        val_y_fine = get_y(val_fold, fine_level, fine_labels)

        #Load Model Checkpoint
        if core:
            print("using gpu")
            model_big, model_small = singlemodel(val_x, coarse=coarse, fine=fine)
        else:
            model_big, model_small = singlemodel_cpu(val_x, coarse=coarse, fine=fine)
        model_small.load_weights(f"{model_dir}/fold{foldnum}_small_lv2_acc-weights.hdf5") #says "big" but call it small model

        val_probs = model_small.predict([val_x, val_mask.reshape(-1, 1000, 1)])[0] #(num samples, coarse, fine)
        val_probs_fine = val_probs.reshape(val_probs.shape[0], -1)[:, fine_idxs]  #(num samples, num_classes=21)
        thresholds_fine = [get_best_threshold_mcc(val_y_fine[:, i], val_probs_fine[:, i]) for i in range(val_y_fine.shape[1])]
        thresholds_fine = np.array(thresholds_fine)
        _, fine_metrics_perclass, fine_metrics_avg = all_metrics(val_y_fine, val_probs_fine, thresholds=thresholds_fine)
        fine_metrics_perclass["fold"] = foldnum
        fine_metrics_perclass["label"] = fine_labels
        fine_metrics_avg["fold"] = foldnum
        fine_metrics_avg["level"] = fine_level
        fine_metrics_perclass["label"] = fine_labels
        
        val_probs_coarse = val_probs.max(axis=2)  #(num samples, coarse)
        thresholds_coarse = [get_best_threshold_mcc(val_y_coarse[:, i], val_probs_coarse[:, i]) for i in range(val_y_coarse.shape[1])]
        thresholds_coarse = np.array(thresholds_coarse)
        _, coarse_metrics_perclass, coarse_metrics_avg = all_metrics(val_y_coarse, val_probs_coarse, thresholds=thresholds_coarse)
        coarse_metrics_perclass["fold"] = foldnum
        coarse_metrics_perclass["label"] = coarse_labels
        coarse_metrics_avg["fold"] = foldnum
        coarse_metrics_avg["level"] = coarse_level
        

        val_perclass_fine_df.append(fine_metrics_perclass)
        val_perclass_coarse_df.append(coarse_metrics_perclass)
        val_avg_df.append(pd.concat([fine_metrics_avg, coarse_metrics_avg]))
        #TODO: Save thresholds

        #Other way is to do thresholding on matrix first and then take or along fine dim to get coarse bin preds
        #But then need to deal with case where no values pass threshold
        '''
        thresholds_matrix = np.zeros(coarse*fine)
        for idx, threshold in zip(fine_idxs, thresholds):
                thresholds_matrix[idx] = threshold
        thresholds_matrix = thresholds_matrix.reshape((coarse, fine))
        val_preds = val_probs > thresholds_matrix
        '''

        test_probs = model_small.predict([test_x, test_mask.reshape(-1, 1000, 1)])[0] #(num samples, coarse, fine)
        test_probs_fine = test_probs.reshape(test_probs.shape[0], -1)[:, fine_idxs]  #(num samples, num_classes=21)
        test_preds_fine = test_probs_fine > thresholds_fine[np.newaxis, :]
        print((test_preds_fine.sum(axis=1)>=1).mean())
        #test_max_fine = (test_probs_fine==test_probs_fine.max(axis=1))
        #test_preds_fine = np.logical_or(test_preds_fine, test_max_fine)
        
        test_probs_coarse = test_probs.max(axis=2)  #(num samples, coarse)
        test_preds_coarse = test_probs_coarse > thresholds_coarse[np.newaxis, :]
        print((test_preds_coarse.sum(axis=1)>=1).mean())
        #test_max_coarse = (test_probs_fine==test_probs_coarse.max(axis=1))
        #test_preds_coarse = np.logical_or(test_preds_coarse, test_max_coarse)

        all_test_probs_fine.append(test_probs_fine)
        all_test_probs_coarse.append(test_probs_coarse)
        all_test_preds_fine.append(test_preds_fine)
        all_test_preds_coarse.append(test_preds_coarse)
    
    #Save Validation Metrics
    val_perclass_fine_df = pd.concat(val_perclass_fine_df)
    val_perclass_coarse_df = pd.concat(val_perclass_coarse_df)
    val_avg_df = pd.concat(val_avg_df)
    val_perclass_fine_df.to_csv(f"{savedir}/val_perclass_metrics_level{fine_level}.csv", index=False)
    val_perclass_coarse_df.to_csv(f"{savedir}/val_perclass_metrics_level{coarse_level}.csv", index=False)
    val_avg_df.to_csv(f"{savedir}/val_avg_metrics.csv", index=False)


    test_probs_fine = np.array(all_test_probs_fine) #(num folds, num samples, num classes)
    test_probs_coarse = np.array(all_test_probs_coarse)
    test_preds_fine = np.array(all_test_preds_fine)
    test_preds_coarse = np.array(all_test_preds_coarse)
    test_probs_fine = test_probs_fine.mean(axis=0) #(num samples, num classes)
    test_probs_coarse = test_probs_coarse.mean(axis=0)
    test_preds_fine = np.array(test_preds_fine.mean(axis=0) > 0.5, dtype=np.int32) #corrected
    test_preds_coarse = (test_preds_coarse.mean(axis=0) > 0.5).astype(np.int32) #corrected

    np.savez(
            f"{savedir}/test_outputs.npz", 
            ids=testset[id_col].to_numpy(),
            probs_fine=test_probs_fine, 
            probs_coarse=test_probs_coarse,
            preds_fine=test_preds_fine,
            preds_coarse=test_preds_coarse,
            targets_fine=test_y_fine,
            targets_coarse=test_y_coarse
            )

    idxs = np.where(test_y_fine.sum(axis=0) != 0)[0]
    test_y_fine = test_y_fine[:, idxs]
    test_probs_fine = test_probs_fine[:, idxs]
    test_preds_fine = test_preds_fine[:, idxs]


    _, fine_metrics_perclass, fine_metrics_avg = all_metrics(
                                                    test_y_fine,
                                                    test_probs_fine,
                                                    y_pred_bin=test_preds_fine)
    fine_metrics_perclass["label"] = np.array(fine_labels)[idxs]
    fine_metrics_avg["level"] = fine_level
    

    idxs = np.where(test_y_coarse.sum(axis=0) != 0)[0]
    test_y_coarse = test_y_coarse[:, idxs]
    test_probs_coarse = test_probs_coarse[:, idxs]
    test_preds_coarse = test_preds_coarse[:, idxs]

    _, coarse_metrics_perclass, coarse_metrics_avg = all_metrics(
                                                    test_y_coarse,
                                                    test_probs_coarse, 
                                                    y_pred_bin=test_preds_coarse)
    coarse_metrics_perclass["label"] = coarse_labels[idxs]
    coarse_metrics_avg["level"] = coarse_level
        


    #Save Test Metrics
    test_avg_df = pd.concat([fine_metrics_avg, coarse_metrics_avg])
    fine_metrics_perclass.to_csv(f"{savedir}/test_perclass_metrics_level{fine_level}.csv", index=False)
    coarse_metrics_perclass.to_csv(f"{savedir}/test_perclass_metrics_level{coarse_level}.csv", index=False)
    test_avg_df.to_csv(f"{savedir}/test_avg_metrics.csv", index=False)

if __name__ == "__main__":
    main()