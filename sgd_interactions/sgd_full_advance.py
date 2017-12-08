"""
SGD with all data as dummies
Combinations are optimized so that all folds' score improves when combination is added
"""
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from scipy.sparse import hstack as csr_hstack
import gc
import time
from collections import defaultdict
from scipy.sparse import save_npz, load_npz
from numba import jit
import datetime
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output

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


create_comb = True

def main():
    trn_df = pd.read_csv("../../input/train.csv", index_col=0)
    sub_df = pd.read_csv("../../input/test.csv", index_col=0)
    kfolder = StratifiedKFold(n_splits=5, shuffle=True, random_state=15)
    y = np.array(trn_df['target'])
    del trn_df["target"]
    if create_comb:
        combs = [
            # 5 CV results all folds advance
            ('ps_reg_01', 'ps_car_02_cat'),  # is KEPT AUC 0.631060 and fold 2 = 0.626759    1 on  346 in   0.0
            ('ps_reg_01', 'ps_car_04_cat'),  # is KEPT AUC 0.631534 and fold 2 = 0.627195    2 on  346 in   1.2
            ('ps_reg_01', 'ps_reg_02'),  # is KEPT AUC 0.631971 and fold 2 = 0.628210g_02   23 on  346 in  24.1
            ('ps_ind_02_cat', 'ps_reg_02'),  # is KEPT AUC 0.632570 and fold 2 = 0.629278   35 on  346 in  37.2
            ('ps_ind_01', 'ps_reg_02'),  # is KEPT AUC 0.632840 and fold 2 = 0.629652g_02   68 on  346 in  73.0
            ('ps_ind_02_cat', 'ps_ind_03'),  # is KEPT AUC 0.634706 and fold 2 = 0.631872   76 on  346 in  81.8
            ('ps_ind_01', 'ps_car_15'),  # is KEPT AUC 0.636025 and fold 2 = 0.633185r_15   88 on  346 in  95.2
            ('ps_ind_02_cat', 'ps_ind_04_cat'),  # is KEPT AUC 0.636377 and fold 2 = 0.6334025 on  346 in 137.1
            ('ps_ind_01', 'ps_car_01_cat'),  # is KEPT AUC 0.636804 and fold 2 = 0.633405  136 on  346 in 150.1
            ('ps_car_04_cat', 'ps_car_06_cat'),  # is KEPT AUC 0.637084 and fold 2 = 0.6339277 on  346 in 151.4
            ('ps_ind_05_cat', 'ps_car_01_cat'),  # is KEPT AUC 0.637335 and fold 2 = 0.6341959 on  346 in 177.3
            ('ps_ind_05_cat', 'ps_ind_06_bin'),  # is KEPT AUC 0.637492 and fold 2 = 0.6342003 on  346 in 182.1
            ('ps_ind_01', 'ps_ind_03'),  # is KEPT AUC 0.637686 and fold 2 = 0.634441d_03  164 on  346 in 183.4
            ('ps_ind_06_bin', 'ps_car_07_cat'),  # is KEPT AUC 0.637889 and fold 2 = 0.6347347 on  346 in 187.1
            ('ps_ind_03', 'ps_car_04_cat'),  # is KEPT AUC 0.637950 and fold 2 = 0.634756  168 on  346 in 188.4
            ('ps_ind_05_cat', 'ps_car_02_cat'),  # is KEPT AUC 0.638043 and fold 2 = 0.6348340 on  346 in 214.9
            ('ps_ind_13_bin', 'ps_ind_15'),  # is KEPT AUC 0.638118 and fold 2 = 0.634986  192 on  346 in 217.4
            ('ps_ind_17_bin', 'ps_car_03_cat'),  # is KEPT AUC 0.638308 and fold 2 = 0.6352191 on  346 in 228.5
            ('ps_ind_03', 'ps_ind_11_bin'),  # is KEPT AUC 0.638345 and fold 2 = 0.635244  206 on  346 in 234.8
            ('ps_ind_02_cat', 'ps_car_08_cat') , # is KEPT AUC 0.638466 and fold 2 = 0.6353033 on  346 in 256.1
            ('ps_calc_08', 'ps_calc_18_bin') , # is KEPT AUC 0.638582 and fold 2 = 0.635307253 on  346 in 293.5
            ('ps_car_07_cat', 'ps_calc_08'),  # is KEPT AUC 0.638683 and fold 2 = 0.635378 327 on  346 in 386.6
        ]
        z = defaultdict(int)
        for f1, f2 in combs:
            z[f1] += 1
            z[f2] += 1
        the_keys = sorted(z, key=lambda x: z[x])[::-1]
        for key in the_keys:
            print("%-20s : %5d" % (key, z[key]))

        # trn_df = trn_df[train_features + ["target"]]
        # sub_df = sub_df[train_features]

        f_drop = ["ps_reg_03", "ps_car_13", "ps_car_14", "ps_car_12"]
        for f in f_drop:
            if -1 in np.unique(trn_df[f]):
                trn_df[f + "_miss"] = pd.Series(trn_df[f] == -1).astype(int)
                sub_df[f + "_miss"] = pd.Series(sub_df[f] == -1).astype(int)

        print("MinMaxScale")
        # for f in f_drop:
        skl = MinMaxScaler()
        skl.fit(pd.concat([trn_df[f_drop], sub_df[f_drop]], axis=0))
        trn_df[f_drop] = skl.transform(trn_df[f_drop])
        sub_df[f_drop] = skl.transform(sub_df[f_drop])

        # f_drop = [f for f in sub_df.columns if "_cat" not in f]

        # trn_df.drop(f_drop, axis=1, inplace=True)
        # sub_df.drop(f_drop, axis=1, inplace=True)

        # For OneHotEncoder
        trn_df.replace(-1, 999, inplace=True)
        sub_df.replace(-1, 999, inplace=True)

        columns = trn_df.columns.values
        columns = [columns[k] for k in range(0, len(columns))]  # we exclude the first column

        # Compute initial csr matrix
        print("Create initial sparse matrix")
        one = OneHotEncoder(handle_unknown='ignore')
        X_trn = one.fit_transform(trn_df.values)
        X_sub = one.transform(sub_df.values)

        # Add combinations and label encode
        print("Adding combinations to datasets")
        for f1, f2 in combs:
            # Create combination feature
            start = time.time()
            name1 = f1 + "_plus_" + f2
            print('Adding %40s' % (name1))
            trn_df[name1] = trn_df[f1].apply(lambda x: str(x)) + "_" + trn_df[f2].apply(lambda x: str(x))
            sub_df[name1] = sub_df[f1].apply(lambda x: str(x)) + "_" + sub_df[f2].apply(lambda x: str(x))
            # print("combinations created in %5.1f" % ((time.time() - start) / 60))

            # Label Encode combination
            lbl = LabelEncoder()
            lbl.fit(list(trn_df[name1].values) + list(sub_df[name1].values))
            trn_df[name1] = lbl.transform(list(trn_df[name1].values))
            sub_df[name1] = lbl.transform(list(sub_df[name1].values))
            # print("Factorization done in %5.1f" % ((time.time() - start) / 60))

            # OneHot Encode and add to the spare matrix
            one = OneHotEncoder(handle_unknown='ignore')
            # print("TRN stack in %5.1f" % ((time.time() - start) / 60))

            X_trn = csr_hstack((X_trn, one.fit_transform(trn_df[[name1]])))  # .tocsr()
            # print("TRN stack in %5.1f" % ((time.time() - start) / 60))

            X_sub = csr_hstack((X_sub, one.transform(sub_df[[name1]])))  # .tocsr()
            # print("Sub stack in %5.1f" % ((time.time() - start) / 60))

            # Free up Memory
            del trn_df[name1]
            del sub_df[name1]
            gc.collect()

        # Now transform coo matrix in csr
        print("Transform coo to csr")
        X_trn = X_trn.tocsr()
        gc.collect()
        X_sub = X_sub.tocsr()
        gc.collect()

        # print("Transforming submission dataset")
        # ohe_features = [f for f in trn_df if f not in f_drop]
        # print(ohe_features)
        # one = OneHotEncoder(handle_unknown='ignore')
        # one.fit(np.array(trn_df[ohe_features]))
        # X_trn = one.transform(np.array(trn_df[ohe_features]))
        # X_sub = one.transform(np.array(sub_df[ohe_features]))
        print(type(X_trn))
        print(np.sum(X_sub.sum(axis=0) == 0))
        print(np.sum(X_trn.sum(axis=0) == 0))
        z = np.arange(X_sub.shape[1])
        zz = np.array((X_sub.sum(axis=0) != 0))[0]
        null_sub_cols = z[zz]

        # remove 0 columns in trn and sub
        print("Remove columns not in both trn and sub")
        X_trn = X_trn[:, null_sub_cols]
        X_sub = X_sub[:, null_sub_cols]

        # Add float features
        # print("Add floats")
        # X_trn = csr_hstack((trn_df[f_drop].values, X_trn)).tocsr()
        # X_sub = csr_hstack((sub_df[f_drop].values, X_sub)).tocsr()

        # Make sure shapes are identical
        print("Transformed shapes : ", X_trn.shape, X_sub.shape)
        print(np.sum(X_sub.sum(axis=0) == 0))
        print(np.sum(X_trn.sum(axis=0) == 0))

        save_npz("ohe_trn_sgd_full_adv.npz", X_trn)
        save_npz("ohe_sub_sgd_full_adv.npz", X_sub)

    else:
        X_trn = load_npz("ohe_trn_sgd_full_adv.npz")
        X_sub = load_npz("ohe_sub_sgd_full_adv.npz")

    for alpha in [.002]:
        i = 0  # iterator counter
        sub_df["target"] = 0
        oof_preds = np.zeros(len(trn_df))
        trn_score = 0
        for train_index, test_index in kfolder.split(y, y):
            # X_train, X_cv = np.array(X)[train_index], np.array(X)[test_index]
            # y_train, y_cv = np.array(y)[train_index], np.array(y)[test_index]
            # one = OneHotEncoder(handle_unknown='ignore')
            # one.fit(X_train)
            # X_train = one.transform(X_train)
            # # print(X_train.shape)
            # X_cv = one.transform(X_cv)
            model = SGDClassifier(loss='log',
                                  penalty='l2',
                                  # alpha=0.0000225,
                                  alpha=alpha,  # 0.0005,
                                  max_iter=100,
                                  random_state=1,
                                  class_weight="balanced",
                                  n_jobs=1)

            X_train, X_cv = X_trn[train_index], X_trn[test_index]
            y_train, y_cv = np.array(y)[train_index], np.array(y)[test_index]
            model.fit(X_train, y_train)
            oof_preds[test_index] = model.predict_proba(X_cv)[:, 1]
            trn_auc = eval_gini(y_train, model.predict_proba(X_train)[:, 1])
            trn_score += trn_auc / kfolder.n_splits
            val_auc = eval_gini(y_cv, oof_preds[test_index])
            print("fold %d/%d auc %.6f / %.6f " % (i + 1, kfolder.n_splits, trn_auc, val_auc))
            i += 1
            # sub_df["target"] += model.predict_proba(X_sub)[:, 1] / kfolder.n_splits
            del X_train
            del X_cv
            gc.collect()

        oof_score = eval_gini(y, oof_preds)
        print("Full OOF score = %.6f / TRN %.6f" % (oof_score, trn_score))
        trn_df["sgd_comb"] = oof_preds

        filename = "../output_preds/sgd_comb_" + str(alpha) + "_"
        filename += str(int(1e6 * oof_score)) + "_"
        filename += curr_date.strftime("%Y_%m_%d_%Hh%M")
        trn_df[["sgd_comb"]].to_csv(filename + "_oof.csv", float_format="%.9f")

    # Fit complete model
    for alpha in [.002]:
        model = SGDClassifier(loss='log',
                              penalty='l2',
                              # alpha=0.0000225,
                              alpha=alpha,  # 0.0005,
                              max_iter=100,
                              random_state=1,
                              class_weight="balanced",
                              n_jobs=1)
        print("Training full")
        model.fit(X_trn, y)
        sub_df["target"] = model.predict_proba(X_sub)[:, 1]
        print("Full training score : %.6f" % eval_gini(y, model.predict_proba(X_trn)[:, 1]))

        print("Predicting submission")
        filename = "../output_preds/sgd_comb_" + str(alpha) + "_"
        filename += str(int(1e6 * oof_score)) + "_"
        filename += curr_date.strftime("%Y_%m_%d_%Hh%M")
        sub_df[["target"]].to_csv(filename + "_sub.csv", float_format="%.9f")


if __name__ == "__main__":
    curr_date = datetime.datetime.now()
    gc.enable()
    main()

# 5 CV results all folds advance
# ('ps_reg_01', 'ps_car_02_cat')  # is KEPT AUC 0.631060 and fold 2 = 0.626759    1 on  346 in   0.0
# ('ps_reg_01', 'ps_car_04_cat')  # is KEPT AUC 0.631534 and fold 2 = 0.627195    2 on  346 in   1.2
# ('ps_reg_01', 'ps_reg_02')  # is KEPT AUC 0.631971 and fold 2 = 0.628210g_02   23 on  346 in  24.1
# ('ps_ind_02_cat', 'ps_reg_02')  # is KEPT AUC 0.632570 and fold 2 = 0.629278   35 on  346 in  37.2
# ('ps_ind_01', 'ps_reg_02')  # is KEPT AUC 0.632840 and fold 2 = 0.629652g_02   68 on  346 in  73.0
# ('ps_ind_02_cat', 'ps_ind_03')  # is KEPT AUC 0.634706 and fold 2 = 0.631872   76 on  346 in  81.8
# ('ps_ind_01', 'ps_car_15')  # is KEPT AUC 0.636025 and fold 2 = 0.633185r_15   88 on  346 in  95.2
# ('ps_ind_02_cat', 'ps_ind_04_cat')  # is KEPT AUC 0.636377 and fold 2 = 0.6334025 on  346 in 137.1
# ('ps_ind_01', 'ps_car_01_cat')  # is KEPT AUC 0.636804 and fold 2 = 0.633405  136 on  346 in 150.1
# ('ps_car_04_cat', 'ps_car_06_cat')  # is KEPT AUC 0.637084 and fold 2 = 0.6339277 on  346 in 151.4
# ('ps_ind_05_cat', 'ps_car_01_cat')  # is KEPT AUC 0.637335 and fold 2 = 0.6341959 on  346 in 177.3
# ('ps_ind_05_cat', 'ps_ind_06_bin')  # is KEPT AUC 0.637492 and fold 2 = 0.6342003 on  346 in 182.1
# ('ps_ind_01', 'ps_ind_03')  # is KEPT AUC 0.637686 and fold 2 = 0.634441d_03  164 on  346 in 183.4
# ('ps_ind_06_bin', 'ps_car_07_cat')  # is KEPT AUC 0.637889 and fold 2 = 0.6347347 on  346 in 187.1
# ('ps_ind_03', 'ps_car_04_cat')  # is KEPT AUC 0.637950 and fold 2 = 0.634756  168 on  346 in 188.4
# ('ps_ind_05_cat', 'ps_car_02_cat')  # is KEPT AUC 0.638043 and fold 2 = 0.6348340 on  346 in 214.9
# ('ps_ind_13_bin', 'ps_ind_15')  # is KEPT AUC 0.638118 and fold 2 = 0.634986  192 on  346 in 217.4
# ('ps_ind_17_bin', 'ps_car_03_cat')  # is KEPT AUC 0.638308 and fold 2 = 0.6352191 on  346 in 228.5
# ('ps_ind_03', 'ps_ind_11_bin')  # is KEPT AUC 0.638345 and fold 2 = 0.635244  206 on  346 in 234.8
# ('ps_ind_02_cat', 'ps_car_08_cat')  # is KEPT AUC 0.638466 and fold 2 = 0.6353033 on  346 in 256.1
# ('ps_calc_08', 'ps_calc_18_bin')  # is KEPT AUC 0.638582 and fold 2 = 0.635307253 on  346 in 293.5
# ('ps_car_07_cat', 'ps_calc_08')  # is KEPT AUC 0.638683 and fold 2 = 0.635378 327 on  346 in 386.6
