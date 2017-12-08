import pandas as pd
import numpy as np
from lightgbm import LGBMClassifier
from sklearn.model_selection import StratifiedKFold
from py_ml_utils.feature_transformer import DummyTransformation, FrequencyTransformation, TargetAverageTransformation
import datetime
import gc

"""
Position 11 / 1268
LeaderBoard : 0.283 | TRN 0.370678 + 0.001065 | VAL 0.284162 / 0.284211 + 0.004834
LGB average of submission predictions over 5 folds and 479 rounds
Training data is transformed as follows:
- binary data      : raw, frequency and target average(mean, 200, 10)
- categorical data : raw, dummy, frequency and target average(mean, 200, 10)
- float data       : no particular processing
- combinations : (ps_ind_04_cat, ps_car_01_cat)
                 (ps_ind_05_cat, ps_car_01_cat) 
                 (ps_car_13_1dec, ps_ind_05_cat)
                 are target averaged and transformed to frequency
reg = LGBMClassifier(
            objective="binary",
            learning_rate=.03, # .03
            num_leaves=20,
            n_estimators=n_estimators,
            colsample_bytree=.9,
            subsample=.9,
            min_split_gain=1,
            min_child_samples=10,
            min_child_weight=5,
            reg_lambda=3,
            reg_alpha=2
        )
"""

FILE_ID = "lgbm_clf_best"


def gini(actual, pred):
    assert (len(actual) == len(pred))
    # Put actual, pred and index in a matrix
    all_ = np.asarray(np.c_[actual, pred, np.arange(len(actual))], dtype=np.float)
    # Sort the matrix by multiple keys
    # first key is negative pred (sort descending)
    # second key is index
    all_ = all_[np.lexsort((all_[:, 2], -1 * all_[:, 1]))]
    # Sum all actual values
    total_losses = all_[:, 0].sum()

    gini_sum = all_[:, 0].cumsum().sum() / total_losses

    gini_sum -= (len(actual) + 1) / 2.

    return gini_sum / len(actual)


def gini_normalized(a, p):
    return gini(a, p) / gini(a, a)


# Create an XGBoost-compatible metric from Gini
def gini_xgb(preds, dtrain):
    labels = dtrain.get_label()
    gini_score = gini_normalized(labels, preds)
    return [('gini', gini_score)]


def gini_lgb(truth, preds):
    # labels = train_data.get_label()
    return 'gini', gini_normalized(truth, preds), True


def prepare_dataset(submit=False):
    # Read the data
    trn_df = pd.read_csv("../../input/train.csv", index_col=0)
    sub_df = pd.read_csv("../../input/test.csv", index_col=0)

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

    trn_df = trn_df[train_features + ["target"]]
    sub_df = sub_df[train_features]

    # Get categorical data
    f_cats = [cat for cat in sub_df.columns if "_cat" in cat]
    print("Number of categorical data : ", len(f_cats))
    # Get binary data
    f_bins = [b for b in sub_df.columns if "_bin" in b]
    print("Number of binary data      : ", len(f_bins))
    # Get float features
    f_flt = [f for f in sub_df.columns if f not in f_bins + f_cats]
    print("Number of float data       : ", len(f_flt))

    # Transform floats to bins
    for f in ["ps_reg_03", "ps_car_12", "ps_car_13", "ps_car_14"]:
        full_f = pd.concat([trn_df[f], sub_df[f]], axis=0)
        full_cut = np.array(pd.cut(full_f, 50, labels=False))
        trn_df[f] = full_cut[:len(trn_df)]
        sub_df[f] = full_cut[len(trn_df):]
        del full_f
        del full_cut

    # Adding feature interaction
    comb = [("ps_ind_04_cat", "ps_car_01_cat"),
            ("ps_ind_05_cat", "ps_car_01_cat")]

    for f1, f2 in comb:
        f_comb = f1 + "_" + f2 + "_cmb"
        # For some reason apply is a lot faster than astype. apply takes 230ms where atype takes 1.3s
        # for reference .to_string() takes even longer 5s on average...
        trn_df[f_comb] = trn_df[f1].apply(lambda x: str(x)) + "|" + trn_df[f2].apply(lambda x: str(x))
        sub_df[f_comb] = sub_df[f1].apply(lambda x: str(x)) + "|" + sub_df[f2].apply(lambda x: str(x))

    f_comb = [f for f in sub_df.columns if "_cmb" in f]
    print("Number of combination data  : ", len(f_comb))

    # Make categories frequencies
    f_dum = []
    f_freq = []
    f_avg = []
    for f in f_cats + f_bins:
        if len(trn_df[f].unique()) > 2:
            tf = DummyTransformation(feature_name=f)
            trn_dum = tf.fit_transform(trn_df)
            sub_dum = tf.transform(sub_df)
            trn_df = pd.concat([trn_df, trn_dum], axis=1)
            sub_df = pd.concat([sub_df, sub_dum], axis=1)
            f_dum += sorted(set(trn_dum.columns).intersection(set(sub_dum.columns)))

        tf = TargetAverageTransformation(feature_name=f,
                                         average=TargetAverageTransformation.MEAN,
                                         min_samples_leaf=200,
                                         smoothing=10)
        trn_df[f + "_avg"] = tf.fit_transform(trn_df, target=trn_df.target)
        sub_df[f + "_avg"] = tf.transform(sub_df)
        f_avg.append(f + "_avg")

        tf = FrequencyTransformation(feature_name=f)
        trn_df[f + "_freq"] = tf.fit_transform(trn_df)
        sub_df[f + "_freq"] = tf.transform(sub_df)
        f_freq.append(f + "_freq")

    for f in f_comb:
        tf = TargetAverageTransformation(feature_name=f,
                                         average=TargetAverageTransformation.MEAN,
                                         min_samples_leaf=200,
                                         smoothing=10)
        trn_df[f + "_avg"] = tf.fit_transform(trn_df, target=trn_df.target)
        sub_df[f + "_avg"] = tf.transform(sub_df)
        f_avg.append(f + "_avg")

        # Replace combination by its frequency
        tf = FrequencyTransformation(feature_name=f)
        trn_df[f] = tf.fit_transform(trn_df)
        sub_df[f] = tf.transform(sub_df)

    return (trn_df[["target"] + f_cats + f_bins + f_flt + f_freq + f_dum + f_avg + f_comb],
            sub_df[f_cats + f_bins + f_flt + f_freq + f_dum + f_avg + f_comb])


def main(submit=False, curr_date_=None):
    metric = gini_normalized

    trn_df, sub_df = prepare_dataset(submit)

    print(trn_df.shape, sub_df.shape)

    # Get target out of train data
    target = trn_df.target
    del trn_df["target"]

    # Create folds
    n_splits = 5
    n_estimators = 500
    folds = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=15)

    # Go though folds
    reg = LGBMClassifier(
        objective="binary",
        learning_rate=.03,  # .03
        num_leaves=20,
        n_estimators=n_estimators,
        colsample_bytree=.8,
        subsample=.8,
        min_split_gain=0,
        min_child_samples=10,
        min_child_weight=5,
        reg_lambda=0,
        reg_alpha=0,
        scale_pos_weight=1.0

    )

    oof_preds = np.empty(len(trn_df))
    sub_preds = np.zeros(len(sub_df))

    imp_df = pd.DataFrame()
    imp_df["feature"] = trn_df.columns
    lgb_evals = np.zeros((n_estimators, n_splits))

    trn_scores = []
    val_scores = []
    # go through folds
    print("Scores = ", end="")
    for fold_, (trn_idx, val_idx) in enumerate(folds.split(trn_df.values, target.values)):
        # Split data
        trn_dat, trn_tgt = trn_df.values[trn_idx], target.values[trn_idx]
        val_dat, val_tgt = trn_df.values[val_idx], target.values[val_idx]
        # print("fitting ", str(type(reg)))
        # Fit LightGBM
        reg.fit(trn_dat, trn_tgt,
                eval_set=[(trn_dat, trn_tgt), (val_dat, val_tgt)],
                eval_names=["trn", "val"],
                eval_metric="auc",  # gini_lgb,
                categorical_feature="auto",
                verbose=50,
                early_stopping_rounds=None)

        # Retrieve evaluation scores to compute best iteration
        lgb_evals[:, fold_] = reg.evals_result_["val"]["auc"]

        trn_preds = reg.predict_proba(trn_dat)[:, 1]
        oof_preds[val_idx] = reg.predict_proba(val_dat)[:, 1]

        # Update importances
        imp_df["imp" + str(fold_ + 1)] = reg.feature_importances_

        # Display current score
        print(" %.6f | " % metric(val_tgt, oof_preds[val_idx]), end="")

        # Keep current score for mean and deviation computation
        trn_scores.append(gini_normalized(trn_tgt, trn_preds))
        val_scores.append(gini_normalized(val_tgt, oof_preds[val_idx]))

        del trn_dat
        del val_dat
        del trn_tgt
        del val_tgt
        gc.collect()
    # Display full OOF score, mean and standard dev of folds
    oof_score = metric(target, oof_preds)
    trn_mean = float(np.mean(trn_scores))  # Return type defined as ndarray but a scalar is returned
    trn_dev = float(np.std(trn_scores))  # Return type defined as ndarray but a scalar is returned
    mean_scores = float(np.mean(val_scores))  # Return type defined as ndarray but a scalar is returned
    dev_scores = float(np.std(val_scores))  # Return type defined as ndarray but a scalar is returned
    print("Full = TRN %.6f + %.6f | VAL %.6f / %.6f + %.6f"
          % (trn_mean, trn_dev, oof_score, mean_scores, dev_scores))

    # Compute lgb best round and displays 3 best rounds
    mean_auc = np.mean(lgb_evals, axis=1)
    bests = np.argsort(mean_auc)[::-1][:3]
    scores = mean_auc[bests]
    print("Best rounds and scores : ", [(bests[i], scores[i]) for i in range(3)])

    # Create submission predictions
    reg.fit(trn_df, target,
            eval_set=[(trn_df, target)],
            eval_metric="auc",
            verbose=50,
            early_stopping_rounds=None)

    sub_df["target"] = reg.predict_proba(sub_df.values)[:, 1]

    # Create filename
    filename = "../output_preds/" + FILE_ID + "_sub_"
    filename += str(int(1e6 * oof_score)) + "_"
    filename += curr_date_.strftime("%Y_%m_%d_%Hh%M") + ".csv"

    sub_df[["target"]].to_csv(filename, index=True, float_format="%.9f")

    # Save OOF predictions
    trn_df[FILE_ID + "_oof"] = oof_preds
    trn_df["target"] = target
    filename = "../output_preds/" + FILE_ID + "_oof_"
    filename += str(int(1e6 * oof_score)) + "_"
    filename += curr_date_.strftime("%Y_%m_%d_%Hh%M") + ".csv"
    trn_df[[FILE_ID + "_oof", "target"]].to_csv(filename, index=True, float_format="%.9f")

    imp_df.to_csv(FILE_ID + "_importances.csv", index=False, sep=";")

if __name__ == '__main__':
    gc.enable()
    curr_date = datetime.datetime.now()
    main(submit=True, curr_date_=curr_date)
