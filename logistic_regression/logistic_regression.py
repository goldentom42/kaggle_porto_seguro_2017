import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler
from numba import jit
import datetime
from typing import Optional, Union


@jit
def eval_gini(y_true, y_prob):
    """
    Original author CPMP : https://www.kaggle.com/cpmpml
    In kernel : https://www.kaggle.com/cpmpml/extremely-fast-gini-computation
    """
    y_true = np.asarray(y_true)
    y_true = y_true[np.argsort(y_prob)]
    ntrue = 0
    gini = 0
    delta = 0
    n = len(y_true)
    for i in range(n-1, -1, -1):
        y_i = y_true[i]
        ntrue += y_i
        gini += y_i * delta
        delta += 1 - y_i
    gini = 1 - 2 * gini / (ntrue * (n - ntrue))
    return gini


def get_file_id():
    return "logreg"


def read_data(submit_=False):
    if submit_:
        return pd.read_csv("../../input/train.csv", index_col=0), pd.read_csv("../../input/test.csv", index_col=0)
    else:
        return pd.read_csv("../../input/train.csv", index_col=0), None


def add_feature_weighting_averages(trn=None, sub=None):
    # type: (pd.DataFrame, Optional[pd.DataFrame]) -> (pd.DataFrame, Optional[pd.DataFrame])
    trn["ps_calc_12_*_ps_ind_03"] = trn["ps_calc_12"] * (trn["ps_ind_03"] + 1)
    trn["ps_calc_12_*_ps_ind_03_*_ps_reg_02"] = trn["ps_calc_12"] * (trn["ps_ind_03"] + 1) * trn["ps_reg_02"]
    trn["ps_car_13_miss_ps_reg_03"] = trn["ps_car_13"] * pd.Series(trn["ps_reg_03"] == -1).replace(0, -1)
    trn["ps_car_15_*_miss_ps_reg_03"] = trn["ps_car_15"] * pd.Series(trn["ps_reg_03"] == -1).replace(0, -1)
    trn["ps_calc_12_*_ps_car_13"] = trn["ps_car_13"] * trn["ps_calc_12"]
    trn["ps_car_13_*_miss_car_03_cat"] = trn["ps_car_13"] * pd.Series(trn["ps_car_03_cat"] == -1).replace(0, -1)

    if sub is not None:
        sub["ps_calc_12_*_ps_ind_03"] = sub["ps_calc_12"] * (sub["ps_ind_03"] + 1)
        sub["ps_calc_12_*_ps_ind_03_*_ps_reg_02"] = sub["ps_calc_12"] * (sub["ps_ind_03"] + 1) * sub["ps_reg_02"]
        sub["ps_car_13_miss_ps_reg_03"] = sub["ps_car_13"] * pd.Series(sub["ps_reg_03"] == -1).replace(0, -1)
        sub["ps_car_15_*_miss_ps_reg_03"] = sub["ps_car_15"] * pd.Series(sub["ps_reg_03"] == -1).replace(0, -1)
        sub["ps_calc_12_*_ps_car_13"] = sub["ps_car_13"] * sub["ps_calc_12"]
        sub["ps_car_13_*_miss_car_03_cat"] = sub["ps_car_13"] * pd.Series(sub["ps_car_03_cat"] == -1).replace(0, -1)

    return trn, sub


def add_missing_value_dummies(trn=None, sub=None):
    # type: (pd.DataFrame, Union(pd.DataFrame, None)) -> (pd.DataFrame, Optional[pd.DataFrame])
    for f in trn.columns:
        if -1 in np.unique(trn[f]):
            trn[f + "_miss"] = pd.Series(trn[f] == -1).astype(int)
            if sub is not None:
                sub[f + "_miss"] = pd.Series(sub[f] == -1).astype(int)
    return trn, sub


def recon(reg):
    """
    Extracts Munipalities and Federal District unit
    @author : Pascal Nagel: https://www.kaggle.com/pnagel
    in kaggle kernel : https://www.kaggle.com/pnagel/reconstruction-of-ps-reg-03
    """
    integer = int(np.round((40 * reg) ** 2))  # gives 2060 for our example
    federal = -1
    for f in range(28):
        if (integer - f) % 27 == 0:
            federal = f
    municipality = (integer - federal) // 27
    return federal, municipality


def add_municipalities(trn=None, sub=None):
    """ Add Municipalities """
    # type: (pd.DataFrame, pd.Series) -> (pd.DataFrame, Union(pd.DataFrame, None))
    print("Adding Municipalities")
    trn['ps_reg_M'] = trn['ps_reg_03'].apply(lambda x: recon(x)[1])
    if sub is not None:
        sub['ps_reg_M'] = sub['ps_reg_03'].apply(lambda x: recon(x)[1])

    return trn, sub


def prepare_data(trn, sub):
    # Retrieve target and remove from train
    tgt = trn.target
    trn.drop(["target"], axis=1, inplace=True)
    # Display target rebalancing weights
    print("Standard rebalancing weights : ", len(tgt) / (2 * np.bincount(tgt)))
    # display Missing value recap
    # Add missing value dummies
    trn, sub = add_missing_value_dummies(trn, sub)
    # Add Municipalities
    trn, sub = add_municipalities(trn, sub)
    # Add Mileage and cylinder unit
    # trn, sub = ps_fe.compute_mileage_cylinder(trn, sub)
    # Add Feature weighting
    trn, sub = add_feature_weighting_averages(trn, sub)

    # All mean     :
    # All median   :
    # All Frequent :
    # miss_features = [f for f in trn.columns if -1 in np.unique(trn[f])]
    # for f in miss_features:
    #     val_inf = MeanMissingValueInferer(feature_name=f, missing_value=-1)
    #     trn[f] = val_inf.infer(trn)

    # Bin continuous variables before One-Hot Encoding
    for f in ["ps_reg_03", "ps_car_12", "ps_car_13", "ps_car_14"]:
        full_f = pd.concat([trn[f], sub[f]], axis=0)
        full_cut = np.array(pd.cut(full_f, 50, labels=False))
        trn[f] = full_cut[:len(trn)]
        sub[f] = full_cut[len(trn):]
        del full_f
        del full_cut

    # Scale all features between 0 and 1
    features = trn.columns
    skl = MinMaxScaler()
    skl_feats = features
    if sub is None:
        trn = trn.loc[:, features]
        trn[skl_feats] = skl.fit_transform(trn[skl_feats])
    else:
        trn = trn.loc[:, features]
        sub = sub.loc[:, features]
        trn[skl_feats] = skl.fit_transform(trn[skl_feats])
        sub[skl_feats] = skl.transform(sub[skl_feats])

    # Return datasets and target
    return trn, sub, tgt


def predict_proba(trn=None, sub=None, target=None, submit=False):
    # Set date
    curr_date = datetime.datetime.now()

    # define folds
    n_splits = 5
    folds = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=15)

    # Init predictions
    oof_preds = np.empty(len(trn))

    # Init feature importance
    imp_df = pd.DataFrame()
    imp_df["feature"] = trn_df.columns

    # define estimator
    # l2, liblinear, balanced, C=1.0 : 0.244756 + 0.003221 (std)
    # l2, liblinear, balanced, C=0.1 : 0.244874 / 0.244876 + 0.003083 (std)
    reg = LogisticRegression(C=.1,
                             penalty="l2",
                             class_weight="balanced",
                             random_state=0,
                             solver="liblinear",
                             verbose=0,
                             max_iter=300)
    print(reg)
    oof_score = 0
    for pos_ratio in [0.03]:
        class_weight = {0: 1 / (2 * (1 - pos_ratio)), 1: 1 / (2 * pos_ratio)}
        print("POS RATIO = %.3f" % pos_ratio, class_weight)
        # reg.set_params(class_weight=class_weight)
        # Run through folds
        trn_scores = []
        val_scores = []
        for fold_, (trn_idx, val_idx) in enumerate(folds.split(trn.values, target.values)):
            # Split data
            trn_dat, trn_tgt = trn.iloc[trn_idx], target.iloc[trn_idx]
            val_dat, val_tgt = trn.iloc[val_idx], target.iloc[val_idx]
            # Fit estimator
            reg.fit(trn_dat, trn_tgt)
            # Get predictions
            trn_preds = reg.predict_proba(trn_dat)[:, 1]
            oof_preds[val_idx] = reg.predict_proba(val_dat)[:, 1]
            # Update importances
            if hasattr(reg, "feature_importances_"):
                imp_df["imp" + str(fold_ + 1)] = reg.feature_importances_
            if hasattr(reg, "coef_"):
                imp_df["imp" + str(fold_ + 1)] = reg.coef_[0]

            # Display current score
            print("Fold %2d TRN %.6f VAL %.6f"
                  % (fold_ + 1, eval_gini(trn_tgt, trn_preds), eval_gini(val_tgt, oof_preds[val_idx])))

            # Keep current score for mean and deviation computation
            trn_scores.append(eval_gini(trn_tgt, trn_preds))
            val_scores.append(eval_gini(val_tgt, oof_preds[val_idx]))

        # Display full OOF score, mean and standard dev of folds
        oof_score = eval_gini(target, oof_preds)
        trn_mean = float(np.mean(trn_scores))  # Return type defined as ndarray but a scalar is returned
        trn_dev = float(np.std(trn_scores))  # Return type defined as ndarray but a scalar is returned
        mean_scores = float(np.mean(val_scores))  # Return type defined as ndarray but a scalar is returned
        dev_scores = float(np.std(val_scores))  # Return type defined as ndarray but a scalar is returned
        print("Full = TRN %.6f + %.6f | VAL %.6f / %.6f + %.6f"
              % (trn_mean, trn_dev, oof_score, mean_scores, dev_scores))

    # Create submission predictions
    if submit:
        reg.fit(trn, target)
        sub["target"] = reg.predict_proba(sub.values)[:, 1]

        # Create filename
        filename = "../output_preds/" + get_file_id() + "_"
        filename += str(int(1e6 * oof_score)) + "_"
        filename += curr_date.strftime("%Y_%m_%d_%Hh%M") + "_"

        sub[["target"]].to_csv(filename + "sub.csv", index=True, float_format='%.9f')

        # Save OOF predictions
        trn[get_file_id() + "_oof"] = oof_preds
        trn["target"] = target
        trn[[get_file_id() + "_oof", "target"]].to_csv(filename + "oof.csv", index=True, float_format='%.9f')

    imp_df.to_csv(get_file_id() + "_importances.csv", index=False, sep=";", float_format="%.9f")


if __name__ == '__main__':
    submit = True
    trn_df, sub_df = read_data(submit_=submit)
    trn_df, sub_df, target = prepare_data(trn=trn_df, sub=sub_df)
    predict_proba(trn_df, sub_df, target, submit)
