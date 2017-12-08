import pandas as pd
import numpy as np
from collections import defaultdict
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder
from numba import jit
from itertools import combinations
from scipy.sparse import hstack as csr_hstack
import time
import gc
from multiprocessing import *
from numba import jit
import argparse

"""
Create 2way feature interactions and check score improvement against 5 CV
This code has been widely inspired by Kaz-Anova scripts available at 
https://github.com/kaz-Anova/StackNet/tree/master/example/example_amazon
"""


@jit
def eval_gini(y_true, y_prob):
    """
    Original author CMPM
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


def check_combination_scores(combs, trn_df, sub_df, X_trn, X_sub, y, grand_auc):
    """
    Method that checks combination score improvement for a given set of feature combinations
    :param combs: Combinations to be checked
    :param trn_df: raw training dataset
    :param sub_df: raw submission dataset
    :param X_trn: csr matrix containing training data
    :param X_sub: csr matrix containing submission data
    :param y: target data
    :param grand_auc: benchmark score without feature combination
    :return:
    """
    # Create 5 CV folds
    kfolder = StratifiedKFold(n_splits=5, shuffle=True, random_state=15)
    # Go through combinations
    start = time.time()
    nb_comb = len(combs)
    kept_combs_ = []
    for n_c, (f1, f2) in enumerate(combs):
        # Create combination and add to sparse matrices
        name1 = f1 + "_plus_" + f2
        dsp_comb = "('%s','%s')" % (f1, f2)
        c_display = "%-45s | %4d " % (dsp_comb, n_c + 1)
        print('current feature %60s %4d on %4d in %5.1f'
              % (name1, n_c + 1, nb_comb, (time.time() - start) / 60), end='', flush=True)
        print('\r' * len(c_display), end='')

        trn_df[name1] = trn_df[f1].apply(lambda x: str(x)) + "_" + trn_df[f2].apply(lambda x: str(x))
        sub_df[name1] = sub_df[f1].apply(lambda x: str(x)) + "_" + sub_df[f2].apply(lambda x: str(x))
        # Label Encode
        lbl = LabelEncoder()
        lbl.fit(list(trn_df[name1].values) + list(sub_df[name1].values))
        trn_df[name1] = lbl.transform(list(trn_df[name1].values))
        sub_df[name1] = lbl.transform(list(sub_df[name1].values))

        # Transform and stack
        one = OneHotEncoder(handle_unknown='ignore')
        X_trn2 = csr_hstack((X_trn, one.fit_transform(trn_df[[name1]]))).tocsr()
        X_sub2 = csr_hstack((X_sub, one.transform(sub_df[[name1]]))).tocsr()

        # Check 5CV score improvement
        mean_auc = 0
        oof = np.zeros(len(y))
        for train_index, test_index in kfolder.split(y, y):
            X_train, X_cv = X_trn2[train_index], X_trn2[test_index]
            y_train, y_cv = np.array(y)[train_index], np.array(y)[test_index]

            model = SGDClassifier(loss='log',
                                  penalty='l2',
                                  alpha=0.0005,
                                  max_iter=50,
                                  random_state=1,
                                  n_jobs=-1)

            model.fit(X_train, y_train)
            oof[test_index] = model.predict_proba(X_cv)[:, 1]
            c_display += "| %.6f " % eval_gini(y_cv, oof[test_index])
            mean_auc += roc_auc_score(y_cv, oof[test_index])

        mean_auc = eval_gini(y, oof)
        c_display += "| %.6f | %5.1f " % (eval_gini(y, oof), (time.time() - start) / 60)

        # Check score improvement
        if mean_auc > grand_auc + 0.00001:
            print(c_display)
            kept_combs_.append((f1, f2, mean_auc))

        del trn_df[name1]
        del sub_df[name1]
        del X_trn2
        del X_sub2
        gc.collect()

    return kept_combs_


def create_2way_interractions(path="../../input/"):
    """
    Create feature combination and parallelize combination score improvement
    :param path: path to the raw datasets
    :return: None
    """
    trn_df = pd.read_csv(path + "train.csv")
    sub_df = pd.read_csv(path + "test.csv")

    # Transform floats to categories
    for f in ["ps_reg_03", "ps_car_12", "ps_car_13", "ps_car_14"]:
        full_f = pd.concat([trn_df[f], sub_df[f]], axis=0)
        full_cut = np.array(pd.cut(full_f, 10, labels=False))
        trn_df[f] = full_cut[:len(trn_df)]
        sub_df[f] = full_cut[len(trn_df):]
        del full_f
        del full_cut

    y = np.array(trn_df['target'])

    to_drop = [f for f in trn_df if "_calc" in f]
    trn_df.drop(["target", "id"] + to_drop, axis=1, inplace=True)
    sub_df.drop(["id"] + to_drop, axis=1, inplace=True)

    # For OneHotEncoder
    trn_df.replace(-1, 999, inplace=True)
    sub_df.replace(-1, 999, inplace=True)

    # Compute initial csr matrix
    one = OneHotEncoder(handle_unknown='ignore')
    X_trn = one.fit_transform(trn_df.values)
    X_sub = one.transform(sub_df.values)

    # Compute benchmark AUC
    kfolder = StratifiedKFold(n_splits=5, shuffle=True, random_state=15)
    grand_auc = 0
    i = 0
    oof = np.zeros(len(y))
    for train_index, test_index in kfolder.split(y, y):
        X_train, X_cv = X_trn[train_index], X_trn[test_index]
        y_train, y_cv = np.array(y)[train_index], np.array(y)[test_index]

        model = SGDClassifier(loss='log',
                              penalty='l2',
                              # alpha=0.0000225,
                              alpha=0.0005,
                              max_iter=50,
                              random_state=1,
                              n_jobs=-1)

        model.fit(X_train, y_train)
        oof[test_index] = model.predict_proba(X_cv)[:, 1]
        print(" fold %d/%d auc %f " % (i + 1, kfolder.n_splits, eval_gini(y_cv, oof[test_index])))
        i += 1

    grand_auc = eval_gini(y, oof)
    print("grand GINI is %f " % grand_auc)

    # Now go through combinations
    columns = trn_df.columns.values
    all_combinations = [(f1, f2) for f1, f2 in combinations(columns, 2)]
    # Create processing pools
    nb_cpus = 2
    p = Pool(nb_cpus)
    import functools
    kept_combs = p.map(functools.partial(check_combination_scores,
                                         trn_df=trn_df,
                                         sub_df=sub_df,
                                         X_trn=X_trn,
                                         X_sub=X_sub,
                                         y=y,
                                         grand_auc=grand_auc),
                       np.array_split(all_combinations, nb_cpus))
    p.close()
    p.join()

    all_combs = []
    for comb in kept_combs:
        all_combs += comb

    return all_combs


def check_2way_interactions(all_combs):
    interactions = dict()
    for i, elements in enumerate(all_combs):
        f1 = elements[0]
        f2 = elements[1]
        score = float(elements[-1])

        interactions[i] = {"f1": f1, "f2": f2, "score": score}
        print("%-20s + %-20s : %.6f" % (f1, f2, score))

    # Create associated DataFrame
    inter_df = pd.DataFrame.from_dict(interactions, orient="index")
    inter_df.sort_values(by="score", ascending=False, inplace=True)
    print(inter_df.head(30))

    path = "../../input/"
    trn_df = pd.read_csv(path + "train.csv")
    y = trn_df.target
    sub_df = pd.read_csv(path + "test.csv")

    # Transform floats to categories
    for f in ["ps_reg_03", "ps_car_12", "ps_car_13", "ps_car_14"]:
        full_f = pd.concat([trn_df[f], sub_df[f]], axis=0)
        full_cut = np.array(pd.cut(full_f, 50, labels=False))
        trn_df[f] = full_cut[:len(trn_df)]
        sub_df[f] = full_cut[len(trn_df):]
        del full_f
        del full_cut

    to_drop = [f for f in trn_df if "_calc" in f]
    trn_df.drop(["target", "id"] + to_drop, axis=1, inplace=True)
    sub_df.drop(["id"] + to_drop, axis=1, inplace=True)

    # For OneHotEncoder
    trn_df.replace(-1, 999, inplace=True)
    sub_df.replace(-1, 999, inplace=True)

    # Compute initial csr matrix
    one = OneHotEncoder(handle_unknown='ignore')
    X_trn = one.fit_transform(trn_df.values)
    X_sub = one.transform(sub_df.values)

    # Compute benchmark AUC
    kfolder = StratifiedKFold(n_splits=5, shuffle=True, random_state=15)
    grand_auc = 0
    i = 0
    benchmark_folds = []
    for train_index, test_index in kfolder.split(y, y):
        X_train, X_cv = X_trn[train_index], X_trn[test_index]
        y_train, y_cv = np.array(y)[train_index], np.array(y)[test_index]

        model = SGDClassifier(loss='log',
                              penalty='l2',
                              alpha=0.0005,
                              max_iter=50,
                              random_state=1,
                              n_jobs=-1)

        model.fit(X_train, y_train)

        preds = model.predict_proba(X_cv)[:, 1]
        auc = roc_auc_score(y_cv, preds)
        benchmark_folds.append(auc)
        print(" fold %d/%d auc %f " % (i + 1, kfolder.n_splits, auc))
        grand_auc += auc
        i += 1

    grand_auc /= kfolder.n_splits
    print("grand AUC is %f " % (grand_auc))

    # Now go through combinations
    columns = trn_df.columns.values
    start = time.time()
    nb_combs = len(inter_df)
    for n_c, (f1, f2) in enumerate(inter_df[["f1", "f2"]].values):
        name1 = f1 + "_plus_" + f2
        print('current feature %60s %4d on %4d in %5.1f'
              % (name1, n_c + 1, nb_combs, (time.time() - start) / 60), end='')
        print('\r' * 99, end='', flush=True)
        trn_df[name1] = trn_df[f1].apply(lambda x: str(x)) + "_" + trn_df[f2].apply(lambda x: str(x))
        sub_df[name1] = sub_df[f1].apply(lambda x: str(x)) + "_" + sub_df[f2].apply(lambda x: str(x))
        # Label Encode
        lbl = LabelEncoder()
        lbl.fit(list(trn_df[name1].values) + list(sub_df[name1].values))
        trn_df[name1] = lbl.transform(list(trn_df[name1].values))

        # Transform and stack
        one = OneHotEncoder(handle_unknown='ignore')
        X_trn2 = csr_hstack((X_trn, one.fit_transform(trn_df[[name1]]))).tocsr()

        mean_auc = 0
        candidate_folds = []
        for train_index, test_index in kfolder.split(y, y):
            X_train, X_cv = X_trn2[train_index], X_trn2[test_index]
            y_train, y_cv = np.array(y)[train_index], np.array(y)[test_index]

            model = SGDClassifier(loss='log',
                                  penalty='l2',
                                  alpha=0.0005,
                                  max_iter=50,
                                  random_state=1,
                                  n_jobs=-1)

            model.fit(X_train, y_train)

            preds = model.predict_proba(X_cv)[:, 1]
            mean_auc += roc_auc_score(y_cv, preds)
            candidate_folds.append(roc_auc_score(y_cv, preds))

        mean_auc /= kfolder.n_splits

        # if np.sum(np.sign(np.array(candidate_folds) - np.array(benchmark_folds))) == 5:
        #     display = "('" + f1 + "', '" + f2 + "')"
        #     print("%s  # is KEPT for all folds improvt AUC %.6f and fold 2 = %.6f" % (display, mean_auc, candidate_folds[1]))
        #     X_trn = csr_hstack((X_trn, one.fit_transform(trn_df[[name1]]))).tocsr()
        #     del benchmark_folds
        #     benchmark_folds = candidate_folds.copy()

        if mean_auc > grand_auc + 1e-6:
            display = "('" + f1 + "', '" + f2 + "')"
            print("%s  # is KEPT for score improvement AUC %.6f and fold 2 = %.6f" % (display, mean_auc, candidate_folds[1]))
            X_trn = csr_hstack((X_trn, one.fit_transform(trn_df[[name1]]))).tocsr()
            grand_auc = mean_auc

        del candidate_folds

        # else:
        #     pos_folds = np.sum(np.sign(np.array(candidate_folds) - np.array(benchmark_folds)))
        #     print("Only %2d folds improved ")
        #

        # if mean_auc > grand_auc + 0.00001:
        #     display = "('" + f1 + "', '" + f2 + "')"
        #     print("%-60s # is KEPT AUC %.6f                   " % (display, mean_auc))
        #     X_trn = csr_hstack((X_trn, one.fit_transform(trn_df[[name1]]))).tocsr()
        #     grand_auc = mean_auc
        #     # X_sub = csr_hstack((X_sub, one.transform(sub_df[[name1]]))).tocsr()

        del trn_df[name1]
        del sub_df[name1]
        del X_trn2
        # del X_sub2
        gc.collect()

    return inter_df


if __name__ == "__main__":
    print("Creating combinations")
    combs = create_2way_interractions()
    # combs = [
    #     ('ps_ind_01', 'ps_ind_02_cat', 0.270332),
    #     ('ps_ind_01', 'ps_ind_03', 0.271731 ),
    #     ('ps_ind_11_bin', 'ps_car_11_cat', 0.271710),
    #     ('ps_ind_11_bin', 'ps_car_11', 0.271093),
    #     ('ps_ind_01', 'ps_ind_05_cat', 0.271422),
    #     ('ps_ind_01', 'ps_ind_06_bin', 0.270876),
    #     ('ps_ind_11_bin', 'ps_car_13', 0.271264),
    #     ('ps_ind_01', 'ps_ind_08_bin', 0.271209),
    #     ('ps_ind_11_bin', 'ps_car_15', 0.271348),
    #     ('ps_ind_01', 'ps_ind_09_bin', 0.271515),
    #     ('ps_ind_12_bin', 'ps_ind_15', 0.271548),
    # ]
    print("Checking combinations")
    check_2way_interactions(combs)


