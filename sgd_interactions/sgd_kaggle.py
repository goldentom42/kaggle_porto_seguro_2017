# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in

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

"""
Folds : 0.286612 0.278048 0.282642 0.290505 0.286672 
OOF : 0.284752
Public LB : 0.277
"""

# print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
gc.enable()

curr_date = datetime.datetime.now()

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

    # Transform floats to bins
    for f in ["ps_reg_03", "ps_car_12", "ps_car_13", "ps_car_14"]:
        full_f = pd.concat([trn_df[f], sub_df[f]], axis=0)
        full_cut = np.array(pd.cut(full_f, 50, labels=False))
        trn_df[f] = full_cut[:len(trn_df)]
        sub_df[f] = full_cut[len(trn_df):]
        del full_f
        del full_cut

    # to_drop = [f for f in trn_df if "_calc" in f]
    # trn_df.drop(to_drop, axis=1, inplace=True)
    # sub_df.drop(to_drop, axis=1, inplace=True)

    idx = np.arange(len(trn_df))
    # np.random.shuffle(idx)
    # trn_df = trn_df.iloc[idx]

    y = np.array(trn_df['target'])
    del trn_df["target"]

    kfolder = StratifiedKFold(n_splits=5, shuffle=True, random_state=15)

    if create_comb:
        combs = [
            # 5 CV results all folds advance
            ('ps_ind_02_cat', 'ps_ind_03'),  # is KEPT for score improvement AUC 0.636061 and fold 2 = 0.632844
            ('ps_reg_01', 'ps_car_04_cat'),  # is KEPT for score improvement AUC 0.637524 and fold 2 = 0.634597
            ('ps_reg_01', 'ps_car_02_cat'),  # is KEPT for score improvement AUC 0.637741 and fold 2 = 0.635010
            ('ps_ind_08_bin', 'ps_reg_01'),  # is KEPT for score improvement AUC 0.637912 and fold 2 = 0.634944
            ('ps_ind_18_bin', 'ps_reg_01'),  # is KEPT for score improvement AUC 0.638110 and fold 2 = 0.635189
            ('ps_reg_01', 'ps_car_10_cat'),  # is KEPT for score improvement AUC 0.638140 and fold 2 = 0.635201
            ('ps_reg_01', 'ps_reg_03'),  # is KEPT for score improvement AUC 0.638208 and fold 2 = 0.635520 6.4
            ('ps_ind_16_bin', 'ps_reg_01'),  # is KEPT for score improvement AUC 0.638219 and fold 2 = 0.635779
            ('ps_ind_01', 'ps_reg_01'),  # is KEPT for score improvement AUC 0.638226 and fold 2 = 0.63588114.8
            ('ps_ind_01', 'ps_car_15'),  # is KEPT for score improvement AUC 0.639228 and fold 2 = 0.63674921.0
            ('ps_ind_05_cat', 'ps_reg_01'),  # is KEPT for score improvement AUC 0.639352 and fold 2 = 0.636897
            ('ps_reg_01', 'ps_car_01_cat'),  # is KEPT for score improvement AUC 0.639380 and fold 2 = 0.637160
            ('ps_ind_03', 'ps_reg_01'),  # is KEPT for score improvement AUC 0.639540 and fold 2 = 0.63757528.5
            ('ps_reg_02', 'ps_car_02_cat'),  # is KEPT for score improvement AUC 0.639804 and fold 2 = 0.638095
            ('ps_ind_02_cat', 'ps_reg_02'),  # is KEPT for score improvement AUC 0.639951 and fold 2 = 0.638577
            ('ps_ind_01', 'ps_reg_02'),  # is KEPT for score improvement AUC 0.640029 and fold 2 = 0.63863736.2
            ('ps_reg_01', 'ps_car_14'),  # is KEPT for score improvement AUC 0.640031 and fold 2 = 0.63893338.5
            ('ps_reg_02', 'ps_car_12'),  # is KEPT for score improvement AUC 0.640067 and fold 2 = 0.63872340.7
            ('ps_reg_02', 'ps_car_07_cat'),  # is KEPT for score improvement AUC 0.640143 and fold 2 = 0.638890
            ('ps_ind_05_cat', 'ps_reg_02'),  # is KEPT for score improvement AUC 0.640198 and fold 2 = 0.639023
            ('ps_ind_17_bin', 'ps_reg_02'),  # is KEPT for score improvement AUC 0.640224 and fold 2 = 0.639101
            ('ps_ind_08_bin', 'ps_reg_02'),  # is KEPT for score improvement AUC 0.640277 and fold 2 = 0.639216
            ('ps_ind_04_cat', 'ps_reg_02'),  # is KEPT for score improvement AUC 0.640305 and fold 2 = 0.639410
            ('ps_reg_02', 'ps_car_08_cat'),  # is KEPT for score improvement AUC 0.640321 and fold 2 = 0.639528
            ('ps_ind_01', 'ps_car_01_cat'),  # is KEPT for score improvement AUC 0.640702 and fold 2 = 0.639495
            ('ps_ind_01', 'ps_car_09_cat'),  # is KEPT for score improvement AUC 0.640827 and fold 2 = 0.639394
            ('ps_ind_02_cat', 'ps_ind_04_cat'),  # is KEPT for score improvement AUC 0.641109 and fold 2 = 0.639496
            ('ps_reg_03', 'ps_car_09_cat'),  # is KEPT for score improvement AUC 0.641313 and fold 2 = 0.639649
            ('ps_ind_17_bin', 'ps_car_15'),  # is KEPT for score improvement AUC 0.641339 and fold 2 = 0.639497
            ('ps_ind_03', 'ps_ind_04_cat'),  # is KEPT for score improvement AUC 0.641443 and fold 2 = 0.639560
            ('ps_ind_02_cat', 'ps_car_15'),  # is KEPT for score improvement AUC 0.641549 and fold 2 = 0.639283
            ('ps_car_14', 'ps_car_15'),  # is KEPT for score improvement AUC 0.641585 and fold 2 = 0.63931683.8
            ('ps_ind_03', 'ps_ind_09_bin'),  # is KEPT for score improvement AUC 0.641659 and fold 2 = 0.639103
            ('ps_car_09_cat', 'ps_car_13'),  # is KEPT for score improvement AUC 0.641829 and fold 2 = 0.639222
            ('ps_car_01_cat', 'ps_car_06_cat'),  # is KEPT for score improvement AUC 0.641994 and fold 2 = 0.639155
            ('ps_ind_03', 'ps_ind_07_bin'),  # is KEPT for score improvement AUC 0.642012 and fold 2 = 0.639380
            ('ps_ind_03', 'ps_car_11'),  # is KEPT for score improvement AUC 0.642157 and fold 2 = 0.63966993.4
            ('ps_ind_03', 'ps_ind_05_cat'),  # is KEPT for score improvement AUC 0.642254 and fold 2 = 0.639855
            ('ps_ind_05_cat', 'ps_car_01_cat'),  # is KEPT for score improvement AUC 0.642425 and fold 2 = 0.639982
            ('ps_car_11_cat', 'ps_car_15'),  # is KEPT for score improvement AUC 0.642475 and fold 2 = 0.640263
            ('ps_ind_01', 'ps_car_11'),  # is KEPT for score improvement AUC 0.642532 and fold 2 = 0.64041601.9
            ('ps_ind_05_cat', 'ps_ind_06_bin'),  # is KEPT for score improvement AUC 0.642662 and fold 2 = 0.640411
            ('ps_ind_08_bin', 'ps_car_11_cat'),  # is KEPT for score improvement AUC 0.642768 and fold 2 = 0.640404
            ('ps_reg_03', 'ps_car_02_cat'),  # is KEPT for score improvement AUC 0.642792 and fold 2 = 0.640319
            ('ps_ind_01', 'ps_car_05_cat'),  # is KEPT for score improvement AUC 0.642818 and fold 2 = 0.640481
            ('ps_ind_08_bin', 'ps_reg_03'),  # is KEPT for score improvement AUC 0.642863 and fold 2 = 0.640493
            ('ps_ind_09_bin', 'ps_car_01_cat'),  # is KEPT for score improvement AUC 0.643008 and fold 2 = 0.640440
            ('ps_ind_09_bin', 'ps_car_03_cat'),  # is KEPT for score improvement AUC 0.643081 and fold 2 = 0.640417
            ('ps_ind_15', 'ps_car_08_cat'),  # is KEPT for score improvement AUC 0.643294 and fold 2 = 0.640742
            ('ps_ind_01', 'ps_ind_03'), # is KEPT for score improvement AUC 0.643474 and fold 2 = 0.640990
            ('ps_ind_15', 'ps_ind_18_bin'),  # is KEPT for score improvement AUC 0.643589 and fold 2 = 0.641175
            ('ps_car_07_cat', 'ps_car_12'),  # is KEPT for score improvement AUC 0.643734 and fold 2 = 0.641282
            ('ps_ind_16_bin', 'ps_car_15'),  # is KEPT for score improvement AUC 0.643736 and fold 2 = 0.641386
            ('ps_ind_07_bin', 'ps_ind_15'),  # is KEPT for score improvement AUC 0.643736 and fold 2 = 0.641397
            ('ps_car_01_cat', 'ps_car_07_cat'),  # is KEPT for score improvement AUC 0.643852 and fold 2 = 0.641656
            ('ps_car_03_cat', 'ps_car_13'),  # is KEPT for score improvement AUC 0.643919 and fold 2 = 0.641811
            ('ps_ind_02_cat', 'ps_car_09_cat'),  # is KEPT for score improvement AUC 0.643922 and fold 2 = 0.641710
            ('ps_car_02_cat', 'ps_car_13'),  # is KEPT for score improvement AUC 0.643987 and fold 2 = 0.641723
            ('ps_ind_17_bin', 'ps_car_03_cat'),  # is KEPT for score improvement AUC 0.644090 and fold 2 = 0.641913
            ('ps_ind_01', 'ps_ind_18_bin'),  # is KEPT for score improvement AUC 0.644186 and fold 2 = 0.641955
            ('ps_ind_04_cat', 'ps_reg_03'),  # is KEPT for score improvement AUC 0.644274 and fold 2 = 0.642394
            ('ps_ind_18_bin', 'ps_car_03_cat'),  # is KEPT for score improvement AUC 0.644362 and fold 2 = 0.642484
            ('ps_ind_09_bin', 'ps_car_11_cat'),  # is KEPT for score improvement AUC 0.644396 and fold 2 = 0.642689
            ('ps_ind_06_bin', 'ps_car_07_cat'),  # is KEPT for score improvement AUC 0.644480 and fold 2 = 0.642792
            ('ps_ind_03', 'ps_reg_03'),  # is KEPT for score improvement AUC 0.644573 and fold 2 = 0.64310591.2
            ('ps_ind_09_bin', 'ps_ind_15'),  # is KEPT for score improvement AUC 0.644666 and fold 2 = 0.643009
            ('ps_car_07_cat', 'ps_car_14'),  # is KEPT for score improvement AUC 0.644715 and fold 2 = 0.643210
            ('ps_car_02_cat', 'ps_car_06_cat'),  # is KEPT for score improvement AUC 0.644722 and fold 2 = 0.643225
            ('ps_ind_01', 'ps_car_03_cat'),  # is KEPT for score improvement AUC 0.644799 and fold 2 = 0.643437
            ('ps_car_11_cat', 'ps_car_14'),  # is KEPT for score improvement AUC 0.644836 and fold 2 = 0.643388
            ('ps_ind_17_bin', 'ps_car_13'),  # is KEPT for score improvement AUC 0.644875 and fold 2 = 0.643358
            ('ps_ind_05_cat', 'ps_ind_07_bin'),  # is KEPT for score improvement AUC 0.644890 and fold 2 = 0.643295
            ('ps_reg_03', 'ps_car_04_cat'),  # is KEPT for score improvement AUC 0.644915 and fold 2 = 0.643232
            ('ps_ind_03', 'ps_ind_08_bin'),  # is KEPT for score improvement AUC 0.644918 and fold 2 = 0.643325
            ('ps_ind_01', 'ps_ind_17_bin'),  # is KEPT for score improvement AUC 0.644946 and fold 2 = 0.643309
            ('ps_ind_01', 'ps_car_04_cat'),  # is KEPT for score improvement AUC 0.644996 and fold 2 = 0.643403
            ('ps_ind_03', 'ps_ind_11_bin'),  # is KEPT for score improvement AUC 0.645008 and fold 2 = 0.643433
            ('ps_car_01_cat', 'ps_car_04_cat'),  # is KEPT for score improvement AUC 0.645049 and fold 2 = 0.643559
            ('ps_ind_03', 'ps_ind_10_bin'),  # is KEPT for score improvement AUC 0.645052 and fold 2 = 0.643580
            ('ps_car_04_cat', 'ps_car_06_cat'),  # is KEPT for score improvement AUC 0.645091 and fold 2 = 0.643765
            ('ps_ind_15', 'ps_car_11_cat'),  # is KEPT for score improvement AUC 0.645176 and fold 2 = 0.643751
            ('ps_ind_03', 'ps_ind_13_bin'),  # is KEPT for score improvement AUC 0.645177 and fold 2 = 0.643778
            ('ps_ind_16_bin', 'ps_car_13'),  # is KEPT for score improvement AUC 0.645241 and fold 2 = 0.643861
            ('ps_car_06_cat', 'ps_car_08_cat'),  # is KEPT for score improvement AUC 0.645315 and fold 2 = 0.643860
            ('ps_ind_04_cat', 'ps_ind_06_bin'),  # is KEPT for score improvement AUC 0.645396 and fold 2 = 0.643918
            ('ps_ind_14', 'ps_ind_15'),  # is KEPT for score improvement AUC 0.645399 and fold 2 = 0.64391558.8
            ('ps_ind_05_cat', 'ps_car_07_cat'),  # is KEPT for score improvement AUC 0.645459 and fold 2 = 0.643955
            ('ps_ind_10_bin', 'ps_car_13'),  # is KEPT for score improvement AUC 0.645476 and fold 2 = 0.643948
            ('ps_ind_18_bin', 'ps_car_05_cat'),  # is KEPT for score improvement AUC 0.645479 and fold 2 = 0.643761
            ('ps_ind_03', 'ps_car_04_cat'),  # is KEPT for score improvement AUC 0.645495 and fold 2 = 0.643683
            ('ps_ind_03', 'ps_car_02_cat'),  # is KEPT for score improvement AUC 0.645500 and fold 2 = 0.643660
            ('ps_ind_13_bin', 'ps_car_13'),  # is KEPT for score improvement AUC 0.645510 and fold 2 = 0.643645
            ('ps_car_10_cat', 'ps_car_13'),  # is KEPT for score improvement AUC 0.645533 and fold 2 = 0.643654
            ('ps_ind_11_bin', 'ps_car_13'),  # is KEPT for score improvement AUC 0.645537 and fold 2 = 0.643643
            ('ps_reg_03', 'ps_car_08_cat'),  # is KEPT for score improvement AUC 0.645538 and fold 2 = 0.643722
            ('ps_ind_03', 'ps_car_10_cat'),  # is KEPT for score improvement AUC 0.645548 and fold 2 = 0.643743
            ('ps_reg_03', 'ps_car_05_cat'),  # is KEPT for score improvement AUC 0.645560 and fold 2 = 0.643735
            ('ps_ind_18_bin', 'ps_car_06_cat'),  # is KEPT for score improvement AUC 0.645577 and fold 2 = 0.643725
            ('ps_ind_05_cat', 'ps_car_02_cat'),  # is KEPT for score improvement AUC 0.645647 and fold 2 = 0.643774
            ('ps_ind_01', 'ps_ind_05_cat'),  # is KEPT for score improvement AUC 0.645665 and fold 2 = 0.643863
            ('ps_car_01_cat', 'ps_car_09_cat'),  # is KEPT for score improvement AUC 0.645727 and fold 2 = 0.643877
            ('ps_ind_04_cat', 'ps_ind_07_bin'),  # is KEPT for score improvement AUC 0.645773 and fold 2 = 0.643951
            ('ps_ind_18_bin', 'ps_car_09_cat'),  # is KEPT for score improvement AUC 0.645799 and fold 2 = 0.644063
            ('ps_ind_15', 'ps_ind_17_bin'),  # is KEPT for score improvement AUC 0.645804 and fold 2 = 0.643843
            ('ps_ind_08_bin', 'ps_car_14'),  # is KEPT for score improvement AUC 0.645838 and fold 2 = 0.644075
            ('ps_ind_05_cat', 'ps_ind_18_bin'),  # is KEPT for score improvement AUC 0.645871 and fold 2 = 0.644028
            ('ps_ind_04_cat', 'ps_car_07_cat'),  # is KEPT for score improvement AUC 0.645893 and fold 2 = 0.644039
            ('ps_ind_05_cat', 'ps_car_04_cat'),  # is KEPT for score improvement AUC 0.645901 and fold 2 = 0.644062
            ('ps_ind_05_cat', 'ps_car_12'),  # is KEPT for score improvement AUC 0.645922 and fold 2 = 0.644130
            ('ps_ind_06_bin', 'ps_car_01_cat'),  # is KEPT for score improvement AUC 0.645929 and fold 2 = 0.644207
            ('ps_ind_06_bin', 'ps_car_02_cat'),  # is KEPT for score improvement AUC 0.646067 and fold 2 = 0.644416
            ('ps_ind_13_bin', 'ps_car_06_cat'),  # is KEPT for score improvement AUC 0.646072 and fold 2 = 0.644443
            ('ps_ind_11_bin', 'ps_reg_03'),  # is KEPT for score improvement AUC 0.646074 and fold 2 = 0.644451
            ('ps_ind_11_bin', 'ps_car_06_cat'),  # is KEPT for score improvement AUC 0.646075 and fold 2 = 0.644473
            ('ps_ind_10_bin', 'ps_car_06_cat'),  # is KEPT for score improvement AUC 0.646075 and fold 2 = 0.644487
            ('ps_ind_18_bin', 'ps_car_12'),  # is KEPT for score improvement AUC 0.646092 and fold 2 = 0.644387
            ('ps_ind_16_bin', 'ps_car_09_cat'),  # is KEPT for score improvement AUC 0.646148 and fold 2 = 0.644510
            ('ps_ind_08_bin', 'ps_car_07_cat'),  # is KEPT for score improvement AUC 0.646156 and fold 2 = 0.644521
            ('ps_ind_05_cat', 'ps_car_08_cat'),  # is KEPT for score improvement AUC 0.646166 and fold 2 = 0.644549
            ('ps_ind_16_bin', 'ps_car_07_cat'),  # is KEPT for score improvement AUC 0.646169 and fold 2 = 0.644528
            ('ps_ind_08_bin', 'ps_ind_12_bin'),  # is KEPT for score improvement AUC 0.646204 and fold 2 = 0.644587
            ('ps_car_10_cat', 'ps_car_11'),  # is KEPT for score improvement AUC 0.646228 and fold 2 = 0.644559
            ('ps_ind_06_bin', 'ps_car_06_cat'),  # is KEPT for score improvement AUC 0.646240 and fold 2 = 0.644334
            ('ps_ind_14', 'ps_car_03_cat'),  # is KEPT for score improvement AUC 0.646243 and fold 2 = 0.644321
            ('ps_ind_16_bin', 'ps_car_11'),  # is KEPT for score improvement AUC 0.646254 and fold 2 = 0.644423
            ('ps_ind_14', 'ps_car_10_cat'),  # is KEPT for score improvement AUC 0.646266 and fold 2 = 0.644415
            ('ps_ind_05_cat', 'ps_ind_11_bin'),  # is KEPT for score improvement AUC 0.646267 and fold 2 = 0.644420
            ('ps_ind_11_bin', 'ps_car_03_cat'),  # is KEPT for score improvement AUC 0.646268 and fold 2 = 0.644414
            ('ps_ind_10_bin', 'ps_car_03_cat'),  # is KEPT for score improvement AUC 0.646269 and fold 2 = 0.644411
            ('ps_ind_02_cat', 'ps_ind_17_bin'),  # is KEPT for score improvement AUC 0.646279 and fold 2 = 0.644409
        ]
        z = defaultdict(int)
        for f1, f2 in combs:
            z[f1] += 1
            z[f2] += 1
        the_keys = sorted(z, key=lambda x: z[x])[::-1]
        for key in the_keys:
            print("%-20s : %5d" % (key, z[key]))

        # For OneHotEncoder
        trn_df.replace(-1, 999, inplace=True)
        sub_df.replace(-1, 999, inplace=True)

        # columns = trn_df.columns.values
        # columns = [columns[k] for k in range(0, len(columns))]  # we exclude the first column

        # Compute initial csr matrix
        print("Create initial sparse matrix")
        one = OneHotEncoder(handle_unknown='ignore')
        X_trn = one.fit_transform(trn_df.values)
        X_sub = one.transform(sub_df.values)

        print(X_trn.shape, X_sub.shape)
        min_samples = 200
        # Add combinations and label encode
        print("Adding combinations to datasets")
        floats = ["ps_reg_03", "ps_car_12", "ps_car_13", "ps_car_14"]
        for f1, f2 in combs:
            if (f1 in floats) or (f2 in floats):
                continue
            # Create combination feature
            start = time.time()
            name1 = f1 + "_plus_" + f2
            # print('Adding %40s' % (name1))
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
            X_trn2 = one.fit_transform(trn_df[[name1]])
            X_sub2 = one.transform(sub_df[[name1]])
            idx = np.arange(X_trn2.shape[1])
            not_null_cols = np.array((X_sub2.sum(axis=0) >= min_samples))[0]
            not_null_idx = idx[not_null_cols]
            print("%-50s nb_cols %4d not_null_cols %4d"
                  % (name1, X_trn2.shape[1], len(not_null_idx)))


            X_trn = csr_hstack((X_trn, X_trn2[:, not_null_cols]))  # .tocsr()
            X_sub = csr_hstack((X_sub, X_sub2[:, not_null_cols]))  # .tocsr()
            # print("Sub stack in %5.1f" % ((time.time() - start) / 60))
            # print(X_trn.shape, X_sub.shape)
            # Free up Memory
            del trn_df[name1]
            del sub_df[name1]
            gc.collect()

        # Check X_trn columns
        idx = np.arange(X_trn.shape[1])
        not_null_cols = np.array((X_trn.sum(axis=0) >= min_samples))[0]
        not_null_idx = idx[not_null_cols]

        # Now transform coo matrix in csr
        print("Transform coo to csr")
        X_trn = X_trn.tocsr()
        gc.collect()
        X_sub = X_sub.tocsr()
        gc.collect()

        X_trn = X_trn[:, not_null_idx]
        X_sub = X_sub[:, not_null_idx]
        print(X_trn.shape, X_sub.shape)

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

        save_npz("ohe_trn_sgd.npz", X_trn)
        save_npz("ohe_sub_sgd.npz", X_sub)

    else:
        X_trn = load_npz("ohe_trn_sgd.npz")
        X_sub = load_npz("ohe_sub_sgd.npz")
        X_trn = X_trn[idx, :]

    print("weight alpha trn1 trn2 ")
    for weight in [.5]:  # , .2, .1, .05]:
        if weight == "balanced":
            class_weight = "balanced"
        else:
            class_weight = {0: 1 / (2 * (1 - weight)), 1: 1 / (2 * weight)}
        for alpha in [0.002]:  # [0.05, 0.02, 0.01, 0.0075, 0.005, 0.002, 0.001, 0.0005]:
            # print("%6.2f " % weight, end="", flush=True)
            # print("%6.4f " % alpha, end="", flush=True)
            # 0.01 has not been tested yet
            # fold 1/5 auc 0.297859 / 0.281374
            # fold 2/5 auc 0.298414 / 0.276374
            # fold 3/5 auc 0.298232 / 0.280176
            # fold 4/5 auc 0.296282 / 0.284552
            # fold 5/5 auc 0.297491 / 0.280663
            # Full OOF score = 0.280560 / TRN 0.297656 <= alpha = 0.05
            # fold 1/5 auc 0.315465 / 0.288203
            # fold 2/5 auc 0.316312 / 0.284223
            # fold 3/5 auc 0.316593 / 0.286668
            # fold 4/5 auc 0.314000 / 0.292540
            # fold 5/5 auc 0.315481 / 0.288370
            # Full OOF score = 0.287763 / TRN 0.315570 <= alpha = 0.02

            # fold 1/5 auc 0.337434 / 0.293383
            # fold 2/5 auc 0.338820 / 0.289128
            # fold 3/5 auc 0.339537 / 0.290123
            # fold 4/5 auc 0.336459 / 0.297890
            # fold 5/5 auc 0.338274 / 0.292622
            # Full OOF score = 0.292138 / TRN 0.338105  <= alpha = 0.0075
            # fold 1/5 auc 0.347326 / 0.294294
            # fold 2/5 auc 0.348994 / 0.289611
            # fold 3/5 auc 0.349795 / 0.290089
            # fold 4/5 auc 0.346575 / 0.298601
            # fold 5/5 auc 0.348563 / 0.292693
            # Full OOF score = 0.292528 / TRN 0.348251 <= alpha = 0.005
            # fold 1/5 auc 0.370660 / 0.292644
            # fold 2/5 auc 0.373167 / 0.286688
            # fold 3/5 auc 0.373732 / 0.286645
            # fold 4/5 auc 0.370409 / 0.296236
            # fold 5/5 auc 0.372758 / 0.288687
            # Full OOF score = 0.289670 / TRN 0.372145 <= alpha = 0.002
            # fold 1/5 auc 0.388288 / 0.287554
            # fold 2/5 auc 0.391486 / 0.280518
            # fold 3/5 auc 0.391499 / 0.280987
            # fold 4/5 auc 0.388641 / 0.290535
            # fold 5/5 auc 0.391054 / 0.281943
            # Full OOF score = 0.283592 / TRN 0.390194  <= alpha = 0.001
            # fold 1/5 auc 0.404207 / 0.279557
            # fold 2/5 auc 0.408386 / 0.270202
            # fold 3/5 auc 0.407610 / 0.272203
            # fold 4/5 auc 0.406041 / 0.280789
            # fold 5/5 auc 0.408007 / 0.272231
            # Full OOF score = 0.273735 / TRN 0.406850 <= alpha = 0.0005
            # Training full
            # Full training score : 0.361550
            # Predicting submission
            #
            # Process finished with exit code 0


            # print("Seed %2d " % seed, end="", flush=True)
            kfolder = StratifiedKFold(n_splits=5, shuffle=True, random_state=15)
            i = 0  # iterator counter
            sub_df["target"] = 0
            oof_preds = np.zeros(len(trn_df))
            trn_score = 0
            for train_index, test_index in kfolder.split(y, y):

                model = SGDClassifier(loss='log',
                                      penalty='l2',
                                      # alpha=0.0000225,
                                      alpha=alpha,  # 0.0005,
                                      max_iter=50,
                                      random_state=1,
                                      class_weight=class_weight,
                                      n_jobs=1)

                X_train, X_cv = X_trn[train_index], X_trn[test_index]
                y_train, y_cv = np.array(y)[train_index], np.array(y)[test_index]
                model.fit(X_train, y_train)
                oof_preds[test_index] = model.predict_proba(X_cv)[:, 1]
                trn_auc = eval_gini(y_train, model.predict_proba(X_train)[:, 1])
                trn_score += trn_auc / kfolder.n_splits
                val_auc = eval_gini(y_cv, oof_preds[test_index])
                # print("fold %d/%d auc %.6f / %.6f " % (i + 1, kfolder.n_splits, trn_auc, val_auc))
                # print("%8.6f %8.6f " % (trn_auc, val_auc), end="", flush=True)
                print("%8.6f " % val_auc, end="", flush=True)

                i += 1
                # sub_df["target"] += model.predict_proba(X_sub)[:, 1] / kfolder.n_splits
                del X_train
                del X_cv
                gc.collect()

            oof_score = eval_gini(y, oof_preds)
            # print("Full OOF score = %.6f / TRN %.6f" % (oof_score, trn_score))
            # print("%8.6f %8.6f" % (oof_score, trn_score))
            print("%8.6f " % (oof_score))

            trn_df["sgd_comb"] = oof_preds

            filename = "../output_preds/sgd_comb_" + str(weight) + "_" + str(alpha) + "_"
            filename += str(int(1e6 * oof_score)) + "_"
            filename += curr_date.strftime("%Y_%m_%d_%Hh%M")
            trn_df[["sgd_comb"]].to_csv(filename + "_oof.csv", float_format="%.9f")

            # Fit complete model
            model = SGDClassifier(loss='log',
                                  penalty='l2',
                                  # alpha=0.0000225,
                                  alpha=alpha,  # 0.0005,
                                  max_iter=50,
                                  random_state=1,
                                  class_weight=class_weight,
                                  n_jobs=1)
            print("Training full")
            model.fit(X_trn, y)
            sub_df["target"] = model.predict_proba(X_sub)[:, 1]
            print("Full training score : %.6f" % eval_gini(y, model.predict_proba(X_trn)[:, 1]))

            print("Predicting submission")
            sub_df[["target"]].to_csv(filename + "_sub.csv", float_format="%.9f")


if __name__ == "__main__":
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
