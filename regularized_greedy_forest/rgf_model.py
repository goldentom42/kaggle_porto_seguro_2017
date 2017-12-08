import abc
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, KFold
from typing import Callable
import datetime
from numba import jit
import time
from py_ml_utils.feature_transformer import TargetAverageTransformation
import gc
from rgf.sklearn import RGFClassifier

# __all__ = "Level1Model"


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
    ]
    combs = [
        ('ps_reg_01', 'ps_car_02_cat'),
        ('ps_reg_01', 'ps_car_04_cat'),
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
        self.model = RGFClassifier(max_leaf=1000,  # 1000,
                                   algorithm="RGF",  # RGF_Sib, RGF_Opt
                                   loss="Log",
                                   l2=0.01,
                                   sl2=0.01,
                                   normalize=False,
                                   min_samples_leaf=10,
                                   n_iter=None,
                                   opt_interval=100,
                                   learning_rate=.5,
                                   calc_prob="sigmoid",
                                   n_jobs=-1,
                                   memory_policy="generous",
                                   verbose=0
                                   )

    @property
    def do_submission(self):
        return self._submit

    @property
    def id(self):
        return self._get_id()

    @abc.abstractmethod
    def _get_id(self):
        self._id = "rgf_full_feat_"
        if self._id == "":
            raise ValueError("Id is not set for class " + str(type(self)))
        return self._id

    def read_data(self):
        self.trn = pd.read_csv("../../input/train.csv", index_col=0)
        self.target = self.trn["target"]
        del self.trn["target"]
        if self.do_submission:
            self.sub = pd.read_csv("../../input/test.csv", index_col=0)

    def add_combinations(self):
        # type: (...) -> (pd.DataFrame, Optional[DataFrame])
        start = time.time()
        for n_c, (f1, f2) in enumerate(self.combs):
            name1 = f1 + "_plus_" + f2
            print('current feature %60s %4d in %5.1f'
                  % (name1, n_c + 1, (time.time() - start) / 60), end='')
            print('\r' * 75, end='')
            self.trn[name1] = self.trn[f1].apply(lambda x: str(x)) + "_" + self.trn[f2].apply(lambda x: str(x))
            if self.do_submission:
                self.sub[name1] = self.sub[f1].apply(lambda x: str(x)) + "_" + self.sub[f2].apply(lambda x: str(x))
                self.trn[name1], indexer = pd.factorize(self.trn[name1])
                self.sub[name1] = indexer.get_indexer(self.sub[name1])
            else:
                self.trn[name1], _ = pd.factorize(self.trn[name1])

    def prepare_data(self):
        noisy_features = list(set(self.trn.columns) - set(self.train_features))

        # Bin continuous variables before One-Hot Encoding
        for f in ["ps_reg_03", "ps_car_12", "ps_car_13", "ps_car_14"]:
            full_f = pd.concat([self.trn[f], self.sub[f]], axis=0)
            full_cut = np.array(pd.cut(full_f, 50, labels=False))
            self.trn[f] = full_cut[:len(self.trn)]
            self.sub[f] = full_cut[len(self.trn):]
            del full_f
            del full_cut

        self.add_combinations()
        # Remove noisy features
        self.trn.drop(noisy_features, axis=1, inplace=True)
        if self.do_submission:
            self.sub.drop(noisy_features, axis=1, inplace=True)

        print(self.trn.columns)

    def predict_oof_and_submission(self):

        self.read_data()
        self.prepare_data()
        pos_ratio = .3
        class_weight = {0: 1 / (2 * (1 - pos_ratio)), 1: 1 / (2 * pos_ratio)}

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
        f_cats = [f for f in self.trn.columns if "_cat" in f]
        for i_fold, (trn_idx, val_idx) in enumerate(self._folds.split(self.target, self.target)):
            # Split data
            trn_x, trn_y = self.trn.iloc[trn_idx].copy(), self.target.iloc[trn_idx]
            val_x, val_y = self.trn.iloc[val_idx].copy(), self.target.iloc[val_idx]

            # Compute target averages
            for f in f_cats:
                ft = TargetAverageTransformation(feature_name=f,
                                                 average=TargetAverageTransformation.MEAN,
                                                 min_samples_leaf=200,
                                                 smoothing=10,
                                                 noise_level=0)
                trn_x[f + "_avg"] = ft.fit_transform(data=trn_x, target=trn_y)
                val_x[f + "_avg"] = ft.transform(data=val_x)
                if self.do_submission:
                    self.sub[f + "_avg"] = ft.transform(data=self.sub)
            # Fit model
            eval_sets = [(trn_x.values, trn_y.values),
                         (val_x.values, val_y.values)]
            sample_weight = trn_y.apply(lambda x: class_weight[x]).values

            self.model.fit(trn_x.values,
                           trn_y.values)
            # Predict OOF
            oof_preds[val_idx] = self.model.predict_proba(val_x.values)[:, 1]

            # Predict SUB if mean is requested
            if (self.sub is not None) and self.mean_submission:
                sub_preds += self.model.predict_proba(self.sub.values)[:, 1] / self._folds.n_splits

            # Print results of current fold
            print("Fold %2d score : %.6f in [%5.1f]"
                  % (i_fold + 1,
                     self.metric(val_y, oof_preds[val_idx]),
                     (time.time() - start) / 60))
            del trn_x
            del val_x
            gc.collect()

        # display OOF result
        oof_score = self.metric(self.target, oof_preds)
        print("Full OOF score : %.6f" % oof_score)

        # Check if we need to fit the model on the full dataset
        if (self.sub is not None) and not self.mean_submission:
            # Compute target averages
            for f in f_cats:
                ft = TargetAverageTransformation(feature_name=f,
                                                 average=TargetAverageTransformation.MEAN,
                                                 min_samples_leaf=200,
                                                 smoothing=10,
                                                 noise_level=0)
                self.trn[f + "_avg"] = ft.fit_transform(data=self.trn, target=self.target)
                self.sub[f + "_avg"] = ft.transform(data=self.sub)
            # Fit model
            self.model.fit(self.trn, self.target)
            # Compute prediction for submission
            sub_preds = self.model.predict_proba(self.sub)[:, 1]

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