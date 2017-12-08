import abc
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, KFold
from typing import Callable
import datetime
from numba import jit
from lightgbm import LGBMClassifier
import time
from py_ml_utils.feature_transformer import TargetAverageTransformation
import gc
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# __all__ = "Level1Model"

def gini(actual, pred):
    assert (len(actual) == len(pred))
    # Put actual, pred and index in a matrix
    all = np.asarray(np.c_[actual, pred, np.arange(len(actual))], dtype=np.float)
    # Sort the matrix by multiple keys
    # first key is negative pred (sort descending)
    # second key is index
    all = all[np.lexsort((all[:, 2], -1 * all[:, 1]))]
    # Sum all actual values
    totalLosses = all[:, 0].sum()

    giniSum = all[:, 0].cumsum().sum() / totalLosses

    giniSum -= (len(actual) + 1) / 2.

    return giniSum / len(actual)


def gini_normalized(a, p):
    return gini(a, p) / gini(a, a)

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


def gini_xgb(preds, dtrain):
    labels = dtrain.get_label()
    gini_score = eval_gini(labels, preds)
    return [('gini', gini_score)]


class Level1Model(object):
    train_features = [
        "ps_car_13",  # : 1571.65 / shadow  609.23
        "ps_reg_03",  # : 1408.42 / shadow  511.15
        "ps_ind_05_cat",  # : 1387.87 / shadow   84.72
        "ps_ind_03",  # : 1219.47 / shadow  230.55
        "ps_ind_15",  # :  922.18 / shadow  242.00
        "ps_reg_02",  # :  920.65 / shadow  267.50
        "ps_car_14",  # :  798.48 / shadow  549.58
        "ps_car_12",  # :  731.93 / shadow  293.62
        "ps_car_01_cat",  # :  698.07 / shadow  178.72
        "ps_car_07_cat",  # :  694.53 / shadow   36.35
        "ps_ind_17_bin",  # :  620.77 / shadow   23.15
        "ps_car_03_cat",  # :  611.73 / shadow   50.67
        "ps_reg_01",  # :  598.60 / shadow  178.57
        "ps_car_15",  # :  593.35 / shadow  226.43
        "ps_ind_01",  # :  547.32 / shadow  154.58
        "ps_ind_16_bin",  # :  475.37 / shadow   34.17
        "ps_ind_07_bin",  # :  435.28 / shadow   28.92
        "ps_car_06_cat",  # :  398.02 / shadow  212.43
        "ps_car_04_cat",  # :  376.87 / shadow   76.98
        "ps_ind_06_bin",  # :  370.97 / shadow   36.13
        "ps_car_09_cat",  # :  214.12 / shadow   81.38
        "ps_car_02_cat",  # :  203.03 / shadow   26.67
        "ps_ind_02_cat",  # :  189.47 / shadow   65.68
        "ps_car_11",  # :  173.28 / shadow   76.45
        "ps_car_05_cat",  # :  172.75 / shadow   62.92
        "ps_calc_09",  # :  169.13 / shadow  129.72
        "ps_calc_05",  # :  148.83 / shadow  120.68
        "ps_ind_08_bin",  # :  140.73 / shadow   27.63
        "ps_car_08_cat",  # :  120.87 / shadow   28.82
        "ps_ind_09_bin",  # :  113.92 / shadow   27.05
        "ps_ind_04_cat",  # :  107.27 / shadow   37.43
        "ps_ind_18_bin",  # :   77.42 / shadow   25.97
        "ps_ind_12_bin",  # :   39.67 / shadow   15.52
        "ps_ind_14",  # :   37.37 / shadow   16.65
        "ps_car_11_cat",
    ]
    combs = [
        ('ps_reg_01', 'ps_car_02_cat'),
        ('ps_reg_01', 'ps_car_04_cat'),
        ("ps_ind_01", "ps_car_15"),  #")  # 0.260193
        ("ps_ind_01", "ps_ind_04_cat"),  #")  # 0.259795
        ("ps_ind_02_cat", "ps_ind_04_cat"),  #")  # 0.259361
        ("ps_ind_04_cat", "ps_car_11"),  #")  # 0.259172
        ("ps_ind_04_cat", "ps_car_09_cat"),  #")  # 0.258886
        ("ps_ind_03", "ps_ind_04_cat"),  # ")  # 0.259107
        # ("ps_ind_02_cat", "ps_ind_03"),
        # ("ps_ind_07_bin", "ps_reg_02"),
        # ("ps_car_04_cat", "ps_reg_02"),
    ]

    def __init__(self, strat=True, splits=5, random_state=15, submit=False, mean_sub=False, metric=None):
        # type: (bool, int, int, bool, bool, Callable) -> None
        self.curr_date = datetime.datetime.now()
        self._submit = submit
        self._id = ""
        self.trn = None
        self.target = None
        self.sub = None
        self.model = None
        self.metric = metric
        self.mean_submission = mean_sub
        if strat:
            self._folds = StratifiedKFold(n_splits=splits, shuffle=True, random_state=random_state)
        else:
            self._folds = KFold(n_splits=splits, shuffle=True, random_state=random_state)
        self.set_model()

    def set_model(self):
        self.model = LGBMClassifier(
            boosting_type="rf",
            n_estimators=400,
            learning_rate=1,
            num_leaves=2024,
            max_depth=10,
            max_bin=255,
            objective="binary",
            min_split_gain=0.01,
            min_child_weight=1,
            min_child_samples=30,
            subsample=.632,
            colsample_bytree=.6,
            scale_pos_weight=1,
            reg_alpha=1e-4,
            reg_lambda=1e-4,
            random_state=0,
            n_jobs=-1
        )

    @property
    def do_submission(self):
        return self._submit

    @property
    def id(self):
        return self._get_id()

    @abc.abstractmethod
    def _get_id(self):
        self._id = "rfc_init_data_"
        if self._id == "":
            raise ValueError("Id is not set for class " + str(type(self)))
        return self._id

    def read_data(self):
        self.trn = pd.read_csv("../../input/train.csv", index_col=0)
        self.target = self.trn["target"]
        del self.trn["target"]
        if self.do_submission:
            self.sub = pd.read_csv("../../input/test.csv", index_col=0)

    def add_missing_value_dummies(self):
        # type: (pd.DataFrame, Union(pd.DataFrame, None)) -> (pd.DataFrame, Optional[DataFrame])
        for f in self.trn.columns:
            if -1 in np.unique(self.trn[f]):
                self.trn[f + "_miss"] = pd.Series(self.trn[f] == -1).astype(int)
                if self.sub is not None:
                    self.sub[f + "_miss"] = pd.Series(self.sub[f] == -1).astype(int)

    def prepare_data(self):
        noisy_features = list(set(self.trn.columns) - set(self.train_features))
        # self.add_missing_value_dummies()
        # self.trn, self.sub = ps_fe.add_combinations(self.trn, self.sub, self.combs)

        # Remove noisy features
        self.trn.drop(noisy_features, axis=1, inplace=True)
        if self.do_submission:
            self.sub.drop(noisy_features, axis=1, inplace=True)

        print(self.trn.columns)

    def predict_oof_and_submission(self):

        self.read_data()
        self.prepare_data()

        if self.model is None:
            raise ValueError("Model is not set for class " + str(type(self)))
        if self.target is None:
            raise ValueError("Model is not set for class " + str(type(self)))
        if self.trn is None:
            raise ValueError("Model is not set for class " + str(type(self)))
        if (self.sub is None) and self.do_submission:
            raise ValueError("Model is not set for class " + str(type(self)))

        # Prepare predictors
        oof_preds = np.zeros(len(self.trn))
        if self.sub is not None:
            sub_preds = np.zeros(len(self.sub))
        # Go through folds
        start = time.time()

        train_features = list(self.trn.columns)
        full_cats = [f for f in self.trn.columns
                     if (len(np.unique(self.trn[f])) < 20) & (len(np.unique(self.trn[f])) > 2)]
        idx = self.trn.index
        start = time.time()

        # for n_c, (f1, f2) in enumerate([("ps_reg_03", "ps_car_13")]):   # enumerate(combinations(full_cats, 2)):
        #     name1 = f1 + "_p_" + f2
        #     # print('current feature %60s %4d in %5.1f'
        #     #       % (name1, n_c + 1, (time.time() - start) / 60), end='')
        #     # print('\r' * 75, end='')
        #     self.trn[name1] = self.trn[f1].apply(lambda x: str(x)) + "_" + self.trn[f2].apply(lambda x: str(x))
        #     # if sub is not None:
        #     #     sub[name1] = sub[f1].apply(lambda x: str(x)) + "_" + sub[f2].apply(lambda x: str(x))
        #     # Label Encode
        #     lbl = LabelEncoder()
        #     lbl.fit(self.trn[name1].values)
        #     self.trn[name1] = lbl.transform(list(self.trn[name1].values))

        # train_features.append(name1)
        print(train_features)
        f_cats = [f for f in self.trn.columns
                  if (len(np.unique(self.trn[f])) < 20) & (len(np.unique(self.trn[f])) > 2)]
        benchmark = 0.271948
        for n_c, (f1, f2) in enumerate(self.combs):
            if (f1 not in self.trn) or (f2 not in self.trn):
                continue

            name1 = f1 + "_p_" + f2
            print('current feature %60s %4d in %5.1f'
                  % (name1, n_c + 1, (time.time() - start) / 60))
            if self.do_submission:
                self.trn[name1] = self.trn[f1].apply(lambda x: str(x)) + "_" + self.trn[f2].apply(lambda x: str(x))
                self.sub[name1] = self.sub[f1].apply(lambda x: str(x)) + "_" + self.sub[f2].apply(lambda x: str(x))
                # lbl = LabelEncoder()
                # lbl.fit(list(self.trn[name1].values) + list(self.sub[name1].values))
                # self.trn[name1] = lbl.transform(list(self.trn[name1].values))
                # self.sub[name1] = lbl.transform(list(self.sub[name1].values))
                self.trn[name1], indexer = pd.factorize(self.trn[name1], sort=True)
                self.sub[name1] = indexer.get_indexer(self.sub[name1])
                print("Unknown values in submission : %6d" % (np.sum(self.sub[name1] == -1)))
            else:
                self.trn[name1] = self.trn[f1].apply(lambda x: str(x)) + "_" + self.trn[f2].apply(lambda x: str(x))
                self.trn[name1], _ = pd.factorize(self.trn[name1])

            train_features.append(name1)

        mean_regul = 200
        scores = []
        for i_fold, (trn_idx, val_idx) in enumerate(self._folds.split(self.target, self.target)):
            # Split data
            trn_x, trn_y = self.trn.loc[idx[trn_idx], train_features].copy(), self.target.iloc[trn_idx]
            val_x, val_y = self.trn.loc[idx[val_idx], train_features].copy(), self.target.iloc[val_idx]

            for f in f_cats:
                # print("Encoding %s" % f)
                # Compute mean on training data
                overall = trn_y.mean()
                # Compute groupby mean and count on training data
                z = pd.concat([trn_x, trn_y], axis=1)
                f_mn = z.groupby(f).target.mean()
                f_cn = z.groupby(f).target.count()
                # Map these on the entire dataset
                means = trn_x[f].map(f_mn)
                counts = trn_x[f].map(f_cn)
                trn_x[f] = (means * counts + mean_regul * overall) / (counts + mean_regul)
                trn_x[f].fillna(overall, inplace=True)
                means = val_x[f].map(f_mn)
                counts = val_x[f].map(f_cn)
                val_x[f] = (means * counts + mean_regul * overall) / (counts + mean_regul)
                val_x[f].fillna(overall, inplace=True)
                if self.do_submission and self.mean_submission:
                    means = self.sub[f].map(f_mn)
                    counts = self.sub[f].map(f_cn)
                    self.sub[f] = (means * counts + mean_regul * overall) / (counts + mean_regul)
                    self.sub[f].fillna(overall, inplace=True)
                del z

            # Fit model
            self.model.fit(trn_x.values,
                           trn_y.values)
            # Predict OOF
            oof_preds[val_idx] = self.model.predict_proba(val_x.values)[:, 1]

            # Predict SUB if mean is requested
            if (self.sub is not None) and self.mean_submission:
                sub_preds += self.model.predict_proba(self.sub.values)[:, 1] / self._folds.n_splits

            # Print results of current fold
            print("%.6f | " % self.metric(val_y, oof_preds[val_idx]), end="", flush=True)
            scores.append(self.metric(val_y, oof_preds[val_idx]))

            del trn_x
            del val_x
            gc.collect()

        # display OOF result
        oof_score = self.metric(self.target, oof_preds)
        elapsed = (time.time() - start) / 60
        print("OOF  %.6f + %.6f in %5.1f" % (oof_score, np.std(scores), elapsed))

        # if oof_score > benchmark:
        #     print("Kept")
        #     # benchmark = oof_score
        # else:
        #     print("dropped")
        #     del self.trn[name1]
        #     del self.sub[name1]
        #     train_features.remove(name1)

        # Check if we need to fit the model on the full dataset
        if (self.sub is not None) and not self.mean_submission:
            # Perform target encoding
            for f in f_cats:
                # print("Encoding %s" % f)
                # Compute mean on training data
                overall = self.target.mean()
                # Compute groupby mean and count on training data
                z = pd.concat([self.trn, self.target], axis=1)
                f_mn = z.groupby(f).target.mean()
                f_cn = z.groupby(f).target.count()
                # Map these on the entire dataset
                means = self.trn[f].map(f_mn)
                counts = self.trn[f].map(f_cn)
                self.trn[f] = (means * counts + mean_regul * overall) / (counts + mean_regul)
                self.trn[f].fillna(overall, inplace=True)
                means = self.sub[f].map(f_mn)
                counts = self.sub[f].map(f_cn)
                self.sub[f] = (means * counts + mean_regul * overall) / (counts + mean_regul)
                self.sub[f].fillna(overall, inplace=True)
                del z

            # Fit model
            self.model.fit(self.trn, self.target)
            # Compute prediction for submission
            sub_preds = self.model.predict_proba(self.sub.values)[:, 1]

        if self.do_submission:
            filename = "../output_preds/" + self.id
            filename += str(int(1e6 * oof_score)) + "_"
            filename += self.curr_date.strftime("%Y_%m_%d_%Hh%M")

            # Save OOF predictions for stacking
            self.trn[self.id] = oof_preds
            self.trn[[self.id]].to_csv(filename + "_oof.csv", float_format="%.9f")

            # Save submission prediction for stacking or submission
            self.sub["target"] = sub_preds
            self.sub[["target"]].to_csv(filename + "_sub.csv", float_format="%.9f")


if __name__ == '__main__':
    gc.enable()
    model = Level1Model(strat=True,
                        splits=5,
                        random_state=15,
                        submit=True,
                        mean_sub=False,
                        metric=eval_gini)

    model.predict_oof_and_submission()


# Test combinations
# Label Encoder : with benchmark test
# current feature                                    ps_reg_01_p_ps_car_02_cat    1 in   0.0
# 0.275051 | 0.265494 | 0.272681 | 0.273839 | 0.274257 | OOF  0.272223 + 0.003471 in   3.5
# Kept
# current feature                                    ps_reg_01_p_ps_car_04_cat    2 in   3.5
# 0.275519 | 0.266060 | 0.273078 | 0.273830 | 0.272922 | OOF  0.272254 + 0.003244 in   7.2
# Kept
# current feature                                        ps_ind_01_p_ps_car_15    3 in   7.2
# 0.275771 | 0.266303 | 0.272929 | 0.275050 | 0.274999 | OOF  0.272969 + 0.003485 in  11.0
# Kept
# current feature                                    ps_ind_01_p_ps_ind_04_cat    4 in  11.0
# 0.275810 | 0.265908 | 0.273926 | 0.273871 | 0.275022 | OOF  0.272872 + 0.003574 in  14.8
# dropped
# current feature                                ps_ind_02_cat_p_ps_ind_04_cat    5 in  14.8
# 0.276928 | 0.266552 | 0.274359 | 0.274681 | 0.275447 | OOF  0.273554 + 0.003631 in  18.6
# Kept
# current feature                                    ps_ind_04_cat_p_ps_car_11    6 in  18.6
# 0.275303 | 0.265959 | 0.274552 | 0.273791 | 0.275930 | OOF  0.273066 + 0.003645 in  22.3
# dropped
# current feature                                ps_ind_04_cat_p_ps_car_09_cat    7 in  22.3
# 0.275535 | 0.265775 | 0.275323 | 0.273735 | 0.275721 | OOF  0.273174 + 0.003788 in  26.0
# dropped
# current feature                                    ps_ind_03_p_ps_ind_04_cat    8 in  26.0
# 0.276986 | 0.267458 | 0.277185 | 0.274725 | 0.277108 | OOF  0.274638 + 0.003732 in  29.8
# Kept
# current feature                                    ps_ind_02_cat_p_ps_ind_03    9 in  29.8
# 0.279677 | 0.266594 | 0.277040 | 0.278453 | 0.277956 | OOF  0.275875 + 0.004752 in  33.5
# Kept
# current feature                                    ps_ind_07_bin_p_ps_reg_02   10 in  33.5
# 0.278132 | 0.267577 | 0.275852 | 0.275596 | 0.276055 | OOF  0.274579 + 0.003646 in  37.4
# dropped
# current feature                                    ps_car_04_cat_p_ps_reg_02   11 in  37.4
# 0.277668 | 0.267513 | 0.276269 | 0.276610 | 0.276127 | OOF  0.274768 + 0.003702 in  41.4
# dropped

# pd.factorize sort =False and benchmark test
# current feature                                    ps_reg_01_p_ps_car_02_cat    1 in   0.0
# Unknown values in submission :      1
# 0.274888 | 0.265395 | 0.272222 | 0.273334 | 0.274163 | OOF  0.271951 + 0.003420 in   3.5
# Kept
# current feature                                    ps_reg_01_p_ps_car_04_cat    2 in   3.5
# Unknown values in submission :      0
# 0.274475 | 0.265390 | 0.273057 | 0.273570 | 0.273008 | OOF  0.271872 + 0.003297 in   7.1
# dropped
# current feature                                        ps_ind_01_p_ps_car_15    3 in   7.1
# Unknown values in submission :      0
# 0.274692 | 0.264960 | 0.272988 | 0.273822 | 0.273068 | OOF  0.271874 + 0.003527 in  10.6
# dropped
# current feature                                    ps_ind_01_p_ps_ind_04_cat    4 in  10.6
# Unknown values in submission :      3
# 0.275206 | 0.264680 | 0.273082 | 0.274254 | 0.273385 | OOF  0.272094 + 0.003794 in  14.5
# Kept
# current feature                                ps_ind_02_cat_p_ps_ind_04_cat    5 in  14.5
# Unknown values in submission :      6
# 0.275987 | 0.264977 | 0.273616 | 0.275304 | 0.275688 | OOF  0.273069 + 0.004151 in  18.7
# Kept
# current feature                                    ps_ind_04_cat_p_ps_car_11    6 in  18.7
# Unknown values in submission :      0
# 0.277223 | 0.265596 | 0.273237 | 0.274433 | 0.274383 | OOF  0.272937 + 0.003916 in  22.3
# dropped
# current feature                                ps_ind_04_cat_p_ps_car_09_cat    7 in  22.3
# Unknown values in submission :      2
# 0.277420 | 0.264151 | 0.273533 | 0.275065 | 0.274651 | OOF  0.272928 + 0.004585 in  26.1
# dropped
# current feature                                    ps_ind_03_p_ps_ind_04_cat    8 in  26.1
# Unknown values in submission :     15
# 0.276578 | 0.265604 | 0.274326 | 0.274738 | 0.276190 | OOF  0.273446 + 0.004032 in  29.8
# Kept
# current feature                                    ps_ind_02_cat_p_ps_ind_03    9 in  29.8
# Unknown values in submission :      0
# 0.275718 | 0.264798 | 0.275207 | 0.274207 | 0.275749 | OOF  0.273058 + 0.004206 in  33.6
# dropped
# current feature                                    ps_ind_07_bin_p_ps_reg_02   10 in  33.6
# Unknown values in submission :      0
# 0.274174 | 0.264437 | 0.273586 | 0.273216 | 0.275418 | OOF  0.272118 + 0.003936 in  37.3
# dropped
# current feature                                    ps_car_04_cat_p_ps_reg_02   11 in  37.3
# Unknown values in submission :     10
# 0.272822 | 0.264778 | 0.274689 | 0.273782 | 0.276702 | OOF  0.272512 + 0.004094 in  41.0
# dropped

# Label encoder no benchmark test
# current feature                                    ps_reg_01_p_ps_car_02_cat    1 in   0.0
# 0.275051 | 0.265494 | 0.272681 | 0.273839 | 0.274257 | OOF  0.272223 + 0.003471 in   4.1
# Kept
# current feature                                    ps_reg_01_p_ps_car_04_cat    2 in   4.1
# 0.275519 | 0.266060 | 0.273078 | 0.273830 | 0.272922 | OOF  0.272254 + 0.003244 in   8.3
# Kept
# current feature                                        ps_ind_01_p_ps_car_15    3 in   8.3
# 0.275771 | 0.266303 | 0.272929 | 0.275050 | 0.274999 | OOF  0.272969 + 0.003485 in  12.3
# Kept
# current feature                                    ps_ind_01_p_ps_ind_04_cat    4 in  12.3
# 0.275810 | 0.265908 | 0.273926 | 0.273871 | 0.275022 | OOF  0.272872 + 0.003574 in  16.2
# Kept
# current feature                                ps_ind_02_cat_p_ps_ind_04_cat    5 in  16.2
# 0.275951 | 0.265757 | 0.275008 | 0.273909 | 0.275748 | OOF  0.273236 + 0.003826 in  20.0
# Kept
# current feature                                    ps_ind_04_cat_p_ps_car_11    6 in  20.0
# 0.276312 | 0.265771 | 0.273566 | 0.276794 | 0.276469 | OOF  0.273741 + 0.004169 in  23.8
# Kept
# current feature                                    ps_ind_03_p_ps_ind_04_cat    7 in  23.8
# 0.278588 | 0.267058 | 0.277303 | 0.277055 | 0.279215 | OOF  0.275800 + 0.004465 in  27.7
# Kept
# current feature                                    ps_ind_02_cat_p_ps_ind_03    8 in  27.7
# 0.281206 | 0.269422 | 0.277097 | 0.279039 | 0.279802 | OOF  0.277256 + 0.004163 in  31.7
# Kept
# current feature                                       ps_ind_01_p_ps_calc_07    9 in  31.7

# pd.factorize sort = True
# current feature                                    ps_reg_01_p_ps_car_02_cat    1 in   0.0
# Unknown values in submission :      1
# 0.275051 | 0.265494 | 0.272681 | 0.273839 | 0.274257 | OOF  0.272223 + 0.003471 in   4.6
# Kept
# current feature                                    ps_reg_01_p_ps_car_04_cat    2 in   4.6
# Unknown values in submission :      0
# 0.275519 | 0.266060 | 0.273078 | 0.273830 | 0.272922 | OOF  0.272254 + 0.003244 in   8.3
# Kept
# current feature                                        ps_ind_01_p_ps_car_15    3 in   8.3
# Unknown values in submission :      0
# 0.275771 | 0.266303 | 0.272929 | 0.275050 | 0.274999 | OOF  0.272969 + 0.003485 in  12.3
# Kept
# current feature                                    ps_ind_01_p_ps_ind_04_cat    4 in  12.3
# Unknown values in submission :      3
# 0.275810 | 0.265908 | 0.273926 | 0.273871 | 0.275022 | OOF  0.272872 + 0.003574 in  16.3
# dropped
# current feature                                ps_ind_02_cat_p_ps_ind_04_cat    5 in  16.3
# Unknown values in submission :      6
# 0.276928 | 0.266552 | 0.274359 | 0.274681 | 0.275447 | OOF  0.273554 + 0.003631 in  20.4
# Kept
# current feature                                    ps_ind_04_cat_p_ps_car_11    6 in  20.4
# Unknown values in submission :      0
# 0.275303 | 0.265959 | 0.274552 | 0.273791 | 0.275930 | OOF  0.273066 + 0.003645 in  25.0
# dropped
# current feature                                ps_ind_04_cat_p_ps_car_09_cat    7 in  25.0
# Unknown values in submission :      2
# 0.275535 | 0.265775 | 0.275323 | 0.273735 | 0.275721 | OOF  0.273174 + 0.003788 in  28.8
# dropped
# current feature                                    ps_ind_03_p_ps_ind_04_cat    8 in  28.8
# Unknown values in submission :     15
# 0.276986 | 0.267458 | 0.277185 | 0.274725 | 0.277108 | OOF  0.274638 + 0.003732 in  32.6
# Kept
# current feature                                    ps_ind_02_cat_p_ps_ind_03    9 in  32.6
# Unknown values in submission :      0
# 0.279677 | 0.266594 | 0.277040 | 0.278453 | 0.277956 | OOF  0.275875 + 0.004752 in  36.4
# Kept
# current feature                                    ps_ind_07_bin_p_ps_reg_02   10 in  36.4
# Unknown values in submission :      0
# 0.278132 | 0.267577 | 0.275852 | 0.275596 | 0.276055 | OOF  0.274579 + 0.003646 in  40.3
# dropped
# current feature                                    ps_car_04_cat_p_ps_reg_02   11 in  40.3
# Unknown values in submission :     10
# 0.277668 | 0.267513 | 0.276269 | 0.276610 | 0.276127 | OOF  0.274768 + 0.003702 in  44.3
# dropped



