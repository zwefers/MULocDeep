import numpy as np
import pandas as pd
from sklearn.metrics import matthews_corrcoef, average_precision_score
from sklearn.metrics import recall_score, precision_score, f1_score
from sklearn.metrics import jaccard_score, roc_auc_score
from sklearn.metrics import label_ranking_average_precision_score, coverage_error



# Taken from DeepLoc2.0
# taken from https://www.kaggle.com/cpmpml/optimizing-probabilities-for-best-mcc
def mcc(tp, tn, fp, fn):
    sup = tp * tn - fp * fn
    inf = (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
    if inf==0:
        return 0
    else:
        return sup / np.sqrt(inf)


def get_best_threshold_mcc(y_true, y_prob):
    idx = np.argsort(y_prob)
    y_true_sort = y_true[idx]
    n = y_true.shape[0]
    nump = 1.0 * np.sum(y_true) # number of positive
    numn = n - nump # number of negative
    tp = nump
    tn = 0.0
    fp = numn
    fn = 0.0
    best_mcc = 0.0
    best_id = -1
    prev_proba = -1
    best_proba = -1
    mccs = np.zeros(n)
    for i in range(n):
        # all items with idx < i are predicted negative while others are predicted positive
        # only evaluate mcc when probability changes
        proba = y_prob[idx[i]]
        if proba != prev_proba:
            prev_proba = proba
            new_mcc = mcc(tp, tn, fp, fn)
            if new_mcc >= best_mcc:
                best_mcc = new_mcc
                best_id = i
                best_proba = proba
        mccs[i] = new_mcc
        if y_true_sort[i] == 1:
            tp -= 1.0
            fn += 1.0
        else:
            fp -= 1.0
            tn += 1.0

    y_pred = (y_prob >= best_proba).astype(int)
    score = matthews_corrcoef(y_true, y_pred)
    # print(score, best_mcc)
    # plt.plot(mccs)
    return best_proba

def roc_auc_wrap(y_true, y_pred):
    #Some classes in HOU have no true positives, so sklearn throws error
    idx = np.where((y_true == y_true[0, :]).all(axis=0))[0]
    max_idx = y_true.shape[1]-1
    y_true = np.delete(y_true, idx, axis=1)
    y_pred = np.delete(y_pred, idx, axis=1)
    rocauc = roc_auc_score(y_true, y_pred, average=None)

    #can't insert at index beyond length of roauc,
    #so if last category has no TP, need to add this way
    if max_idx in idx:
        rocauc = np.insert(rocauc, idx[:-1], 0)
        rocauc = np.append(rocauc, 0)
    else:
        rocauc = np.insert(rocauc, idx, 0)

    rocauc_macro = roc_auc_score(y_true, y_pred, average="macro")
    rocauc_micro = roc_auc_score(y_true, y_pred, average="micro")
    return rocauc, rocauc_macro, rocauc_micro

def thresh_wrap(y_pred, thresholds):
    #Handles case where no localization category passes threshold
    max_idx = y_pred.argmax(axis=1)
    y_pred_bin = (y_pred > thresholds)
    for i in range(len(max_idx)):
        if y_pred_bin[i,:].sum() == 0:
            y_pred_bin[i, max_idx[i]] = 1 #set localization with max probability to 1
    return y_pred_bin.astype(np.int16)


def all_metrics(y_true, y_pred, y_pred_bin=None, thresholds=None, continuous=True):
    if continuous:
        if y_pred_bin is None:
            if thresholds is None:
                thresholds = [get_best_threshold_mcc(y_true[:, i], y_pred[:, i]) for i in range(y_true.shape[1])]
                thresholds = np.array(thresholds)
                print(thresholds)
            y_pred_bin = thresh_wrap(y_pred, thresholds)
        macro_ap = average_precision_score(y_true, y_pred, average="macro") #continuous
        micro_ap = average_precision_score(y_true, y_pred, average="micro") #continuous
        rocauc_perclass, rocauc_macro, rocauc_micro = roc_auc_wrap(y_true, y_pred) #continuous
        mlrap = label_ranking_average_precision_score(y_true, y_pred) #continuous
        cov_error = coverage_error(y_true, y_pred) #continuous
    else: #Need this for random predictor which only has binary outputs
        y_pred_bin = y_pred
        macro_ap = np.nan
        micro_ap = np.nan
        rocauc_macro = np.nan
        rocauc_micro = np.nan
        rocauc_perclass = np.full((y_true.shape[1],), np.nan)
        mlrap = np.nan
        cov_error = np.nan

    mcc_perclass = np.array([matthews_corrcoef(y_true[:, i], y_pred_bin[:, i]) for i in range(y_true.shape[1])])
    acc = (y_true == y_pred_bin).all(axis=1).mean()
    acc_perclass = (y_true == y_pred_bin).mean(axis=0)
    recall_perclass = np.array([recall_score(y_true[:, i], y_pred_bin[:, i]) for i in range(y_true.shape[1])])
    precision_perclass = np.array([precision_score(y_true[:, i], y_pred_bin[:, i]) for i in range(y_true.shape[1])])
    f1_perclass = f1_score(y_true, y_pred_bin, average=None)
    f1_macro = f1_score(y_true, y_pred_bin, average="macro")
    f1_micro = f1_score(y_true, y_pred_bin, average="micro")
    jaccard_perclass = jaccard_score(y_true, y_pred_bin, average=None)
    jaccard_macro = jaccard_score(y_true, y_pred_bin, average="macro")
    jaccard_micro = jaccard_score(y_true, y_pred_bin, average="micro")

    num_labels = y_pred_bin.sum(axis=1).mean() #average number of predicted labels
    

    metrics_dict = {
                    "macro_ap": macro_ap,
                    "micro_ap": micro_ap,
                    "mcc_perclass": mcc_perclass,
                    "acc": acc,
                    "acc_perclass": acc_perclass,
                    "recall_perclass": recall_perclass,
                    "precision_perclass": precision_perclass,
                    "f1_perclass": f1_perclass,
                    "f1_macro": f1_macro,
                    "f1_micro": f1_micro,
                    "jaccard_perclass": jaccard_perclass,
                    "jaccard_macro": jaccard_macro,
                    "jaccard_micro": jaccard_micro,
                    "rocauc_perclass": rocauc_perclass,
                    "rocauc_macro": rocauc_macro,
                    "rocauc_micro": rocauc_micro,
                    "mlrap": mlrap,
                    "coverage_error": cov_error,
                    "num_labels": num_labels
                    }

    metrics_perclass = np.array([
                        mcc_perclass, 
                        acc_perclass, 
                        recall_perclass, 
                        precision_perclass, 
                        f1_perclass,
                        jaccard_perclass,
                        rocauc_perclass
                        ])
    
    cols = [
            "mcc_perclass",
            "acc_perclass",
            "recall_perclass",
            "precision_perclass",
            "f1_perclass",
            "jaccard_perclass",
            "rocauc_perclass"
            ]
    
    metrics_perclass = pd.DataFrame(metrics_perclass.T, columns=cols) 

    metrics_avg = [[
                    macro_ap,
                    micro_ap,
                    acc,
                    f1_macro,
                    f1_micro,
                    jaccard_macro,
                    f1_micro,
                    rocauc_macro,
                    rocauc_micro,
                    mlrap,
                    cov_error,
                    num_labels
                    ]]
    
    cols = [
            "macro_ap",
            "micro_ap",
            "acc",
            "f1_macro",
            "f1_micro",
            "jaccard_macro",
            "jaccard_micro",
            "rocauc_macro",
            "rocauc_micro",
            "mlrap",
            "cov_error",
            "num_labels"
            ]
    
    metrics_avg = pd.DataFrame(metrics_avg, columns=cols)

    return metrics_dict, metrics_perclass, metrics_avg