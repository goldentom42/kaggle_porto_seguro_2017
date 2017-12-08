import abc
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler
from sklearn.linear_model import RidgeClassifier
from typing import Callable
import datetime
# from data_fe_utils import PortoSeguroFeatureEngineering
from numba import jit
# from xgboost import XGBClassifier
import time
from scipy.sparse import hstack as csr_hstack

__all__ = "Level1Model"

# TODO create this script
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
        self.trn_csr = None
        self.sub_csr = None
        if strat:
            self._folds = StratifiedKFold(n_splits=splits, shuffle=True, random_state=random_state)
        else:
            self._folds = KFold(n_splits=splits, shuffle=True, random_state=random_state)
        self.set_model()

    def set_model(self):
        self.model = RidgeClassifier(
            alpha=3000,  # Was 1000
            normalize=False,
            max_iter=1000,
            class_weight="balanced",  # {0: 1, 1: 2},
            random_state=1,
            solver="sag",
            tol=1e-3,
            copy_X=False,
        )
        # self.model.fit()

    @property
    def do_submission(self):
        return self._submit

    @property
    def id(self):
        return self._get_id()

    @abc.abstractmethod
    def _get_id(self):
        self._id = "ridge_dummies"
        if self._id == "":
            raise ValueError("Id is not set for class " + str(type(self)))
        return self._id

    def read_data(self):
        self.trn = pd.read_csv("../../input/train.csv", index_col=0)
        self.target = self.trn["target"]
        del self.trn["target"]
        if self.do_submission:
            self.sub = pd.read_csv("../../input/test.csv", index_col=0)

    def prepare_data(self):
        self.trn = self.trn[self.train_features]
        if self.do_submission:
            self.sub = self.sub[self.train_features]

        for f in ["ps_reg_03", "ps_car_12", "ps_car_13", "ps_car_14"]:
            full_f = pd.concat([self.trn[f], self.sub[f]], axis=0)
            full_cut = np.array(pd.cut(full_f, 20, labels=False))
            self.trn[f] = full_cut[:len(self.trn)]
            self.sub[f] = full_cut[len(self.trn):]
            del full_f
            del full_cut

        # Transform low card f to
        high_card_f = []
        binary_f = []
        for f in self.trn.columns:
            card = len(np.unique(self.trn[f]))
            one = OneHotEncoder(handle_unknown='ignore')

            if (card > 2) & (card < 110):
                print("Encoding %s" % f)
                if self.trn_csr is None:
                    self.trn_csr = one.fit_transform(self.trn[[f]].replace(-1, 99999))
                    if self.do_submission:
                        self.sub_csr = one.transform(self.sub[[f]].replace(-1, 99999))
                else:
                    self.trn_csr = csr_hstack((self.trn_csr, one.fit_transform(self.trn[[f]].replace(-1, 99999))))
                    if self.do_submission:
                        self.sub_csr = csr_hstack((self.sub_csr, one.transform(self.sub[[f]].replace(-1, 99999))))
            elif card <= 2:
                binary_f.append(f)
            else:
                high_card_f.append(f)

        # Add binary data
        print("Add binary feats : ", binary_f)
        self.trn_csr = csr_hstack((self.trn_csr, self.trn[binary_f]))
        if self.do_submission:
            self.sub_csr = csr_hstack((self.sub_csr, self.sub[binary_f]))

        # Add High card data
        # We need to scale those features
        print("Add high card feats : ", high_card_f)
        # skl = StandardScaler()
        # if not self.do_submission:
        #     self.trn_csr = csr_hstack((self.trn_csr, skl.fit_transform(self.trn[high_card_f].values)))
        # else:
        #     skl.fit(np.vstack((self.trn[high_card_f].values, self.sub[high_card_f].values)))
        #     self.trn_csr = csr_hstack((self.trn_csr, skl.transform(self.trn[high_card_f].values)))
        #     self.sub_csr = csr_hstack((self.sub_csr, skl.transform(self.sub[high_card_f].values)))

        print("Transform to csr")
        self.trn_csr = self.trn_csr.tocsr()
        print("CSR shape = ", self.trn_csr.shape)
        if self.do_submission:
            self.sub_csr = self.sub_csr.tocsr()

        print(self.trn_csr.sum(axis=0) < 100)

        self.sub_csr_not_enough = np.array(self.sub_csr.sum(axis=0) <= 100)[0, :]
        self.sub_csr_occurences = np.array(self.sub_csr.sum(axis=0))[0, :]
        print(self.sub_csr_occurences.shape)
        print(self.sub_csr_not_enough)

    def predict_oof_and_submission(self):

        self.read_data()
        self.prepare_data()
        pos_ratio = .5
        class_weight = {0: 1 / (2 * (1 - pos_ratio)), 1: 1 / (2 * pos_ratio)}
        coefs = np.zeros((self.trn_csr.shape[1], self._folds.n_splits))

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
        if self.do_submission:
            sub_preds = np.zeros(len(self.sub))
        # Go through folds
        start = time.time()
        for i_fold, (trn_idx, val_idx) in enumerate(self._folds.split(self.target, self.target)):
            # Fit model
            self.model.fit(self.trn_csr[trn_idx],
                           self.target.values[trn_idx])
            coefs[:, i_fold] = self.model.coef_
            print(self.model.coef_[0, self.sub_csr_not_enough])
            print(self.sub_csr_occurences[self.sub_csr_not_enough])
            # Predict OOF
            oof_preds[val_idx] = self.model.decision_function(self.trn_csr[val_idx])

            # Predict SUB if mean is requested
            if (self.sub is not None) and self.mean_submission:
                sub_preds += self.model.decision_function(self.sub_csr) / self._folds.n_splits

            # Print results of current fold
            print("Fold %2d score : %.6f in [%5.1f]"
                  % (i_fold + 1,
                     self.metric(self.target.values[val_idx], oof_preds[val_idx]),
                     (time.time() - start) / 60))

        # display OOF result
        oof_score = self.metric(self.target, oof_preds)
        print("Full OOF score : %.6f" % oof_score)

        # Check if we need to fit the model on the full dataset
        if (self.sub is not None) and not self.mean_submission:
            # Fit model
            self.model.fit(self.trn_csr, self.target)
            # Compute prediction for submission
            sub_preds = self.model.decision_function(self.sub_csr)
            # Make sure coefs are not crazy
            coefs = np.abs(np.array(self.model.coef_)[0, :])
            sub_occ = np.array(self.sub_csr.sum(axis=0))[0, :]
            trn_occ = np.array(self.trn_csr.sum(axis=0))[0, :]
            sortation = np.argsort(coefs)[::-1]
            for s in sortation:
                print("%6d %6d %.5f" % (trn_occ[s], sub_occ[s], coefs[s]))

        if self.do_submission:
            filename = "../output_preds/" + self.id + "_"
            filename += str(int(1e6 * oof_score)) + "_"
            filename += self.curr_date.strftime("%Y_%m_%d_%Hh%M")

            # Save OOF predictions for stacking
            self.trn[self.id] = 1 / (1 + np.exp(- oof_preds))
            self.trn[[self.id]].to_csv(filename + "_oof.csv", float_format="%.9f")

            # Save submission prediction for stacking or submission
            self.sub["target"] = 1 / (1 + np.exp(- sub_preds))
            self.sub[["target"]].to_csv(filename + "_sub.csv", float_format="%.9f")


if __name__ == '__main__':
    model = Level1Model(strat=True,
                        splits=5,
                        random_state=15,
                        submit=True,
                        mean_sub=False,
                        metric=eval_gini)

    model.predict_oof_and_submission()
