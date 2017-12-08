import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from numba import jit
import datetime


@jit
def eval_gini(y_true, y_prob):
    """
    Original author CPMP
    https://www.kaggle.com/cpmpml/extremely-fast-gini-computation
    """
    y_true = np.asarray(y_true)
    y_true = y_true[np.argsort(y_prob)]
    ntrue = 0
    gini = 0
    delta = 0
    n = len(y_true)
    for i in range(n - 1, -1, -1):
        y_i = y_true[i]
        ntrue += y_i
        gini += y_i * delta
        delta += 1 - y_i
    gini = 1 - 2 * gini / (ntrue * (n - ntrue))
    return gini

if __name__ == '__main__':
    curr_date = datetime.datetime.now()
    trn = pd.read_csv('../../input/train.csv', index_col=0)
    oof_preds = np.zeros(len(trn))
    folds = StratifiedKFold(5, True, 15)
    mean_score = 0.0
    for i_fold, (trn_idx, val_idx) in enumerate(folds.split(trn.target, trn.target)):

        curr_oof = pd.read_csv('val_ffm_%d_preds.txt' % (i_fold + 1), header=None)
        curr_oof.columns = ['target']
        oof_preds[val_idx] = curr_oof.target.values
        curr_score = eval_gini(trn.iloc[val_idx].target.values, oof_preds[val_idx])
        mean_score += curr_score / folds.n_splits
        print(curr_score)

    oof_score = eval_gini(trn.target, oof_preds)
    print("Full OOF score = %.6f / %.6f" % (oof_score, mean_score))

    filename = "../output_preds/"
    filename += "libffm_" + str(int(1e6 * oof_score)) + "_"
    filename += curr_date.strftime("%Y_%m_%d_%Hh%M") + "_"

    trn["libffm_oof"] = oof_preds
    trn[["libffm_oof"]].to_csv(filename + "oof.csv", index=True, float_format="%.9f")

    sub = pd.read_csv('../../input/sample_submission.csv')
    print(len(sub))
    outputs = pd.read_csv('output.txt', header=None)
    outputs.columns = ['target']
    print(len(outputs))

    sub.target = outputs.target.ravel()
    sub.to_csv(filename + 'sub.csv',index=False)
