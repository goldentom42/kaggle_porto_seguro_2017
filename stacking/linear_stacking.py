"""
Linear Stacking solution scoring 0.29116 on Private LB

The solution uses the following predictions
ftrl_proximal_269092_oof.csv / ftrl_proximal_269092_sub.csv
keras_2layers_dummies_276637_oof.csv / keras_2layers_dummies_276637_sub.csv
keras_shallow_dummies_273395_oof.csv / keras_shallow_dummies_273395_sub.csv
lgbm_clf_best_oof_286141_XXX.csv / lgbm_clf_best_sub_286141_XXX.csv
libffm_281648_XXX_oof.csv / libffm_281648_XXX_sub.csv
logreg_250686_XXX_oof.csv / logreg_250686_XXX_sub.csv
rfc_init_data_276747_XXX_oof.csv / rfc_init_data_276747_XXX_sub.csv
rgf_full_feat_283579_XXX_oof.csv / rgf_full_feat_283579_XXX_sub.csv
ridge_dummies_268192_XXX_oof.csv / ridge_dummies_268192_XXX_sub.csv
sgd_comb_0.002_278725_XXX_oof.csv / sgd_comb_0.002_278725_XXX_sub.csv
xgb_full_feat286265_XXX_oof.csv / xgb_full_feat286265_XXX_sub.csv

The best local scoring SGD was not kept due to a clear overfit.
So the second best SGD is used.

The linear stacker uses its standard algorithm letting weights being negative.
For information here are the different stacking results:
                                           5-CV    PublicLB   PrivateLB
- Swapping algo :                        0.29095 |  0.28750 |   0.29089
- Standard algo with positive weights :  0.28979 |  0.28628 |   0.29052
- Standard algo with negative weights :  0.29100 |  0.28794 |   0.29114

Private LB score is pretty close to local 5CV results

"""
import pandas as pd
import numpy as np
from numba import jit
from linear_stacker import BinaryClassificationLinearPredictorStacker
from linear_stacker import LinearPredictorStacker
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
import datetime


@jit  # for more info please visit https://numba.pydata.org/
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
    for i in range(n-1, -1, -1):
        y_i = y_true[i]
        ntrue += y_i
        gini += y_i * delta
        delta += 1 - y_i
    gini = 1 - 2 * gini / (ntrue * (n - ntrue))
    return gini


def main():
    # Get files
    from subprocess import check_output
    files = check_output(["ls", "../output_preds"]).decode("utf8")
    oof_files = [file for file in files.split('\n') if ("_oof" in file) and ("7z" not in file)]
    sub_files = [file.replace("_oof", "_sub") for file in oof_files]
    oof_files, sub_files

    oof_preds = None
    target = None
    sub_preds = None
    for file in oof_files:

        # read file and index by id
        oof = pd.read_csv("../output_preds/" + file, index_col=0)

        if target is None:
            target = oof["target"].copy()

        # drop target from oof
        if "target" in oof.columns:
            oof.drop(["target"], axis=1, inplace=True)

        # Update OOF predictions
        if oof_preds is None:
            oof_preds = oof.copy()
        else:
            oof_preds = pd.concat([oof_preds, oof], axis=1)

        del oof

    print(oof_preds.head())

    sub_preds = None

    for file in sub_files:

        # read file and index by id
        sub = pd.read_csv("../output_preds/" + file, index_col=0)

        # Update OOF predictions
        if sub_preds is None:
            sub_preds = sub.copy()
        else:
            sub_preds = pd.concat([sub_preds, sub], axis=1)

        del sub

    sub_preds.columns = oof_preds.columns

    curr_date = datetime.datetime.now()

    stk = BinaryClassificationLinearPredictorStacker(
        metric=eval_gini,
        algorithm=LinearPredictorStacker.STANDARD,
        max_iter=200,
        verbose=2,
        normed_weights=True,
        maximize=True,
        seed=1,
        # eps=1e-5
    )

    # Create folds - should be identical to OOF construction
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=15)
    # Init stacker OOF and submission probabilities
    stk_oof = np.zeros(len(oof_preds))
    stk_sub = np.zeros(len(sub_preds))
    weights = np.zeros((oof_preds.shape[1], folds.n_splits))
    # Go through folds
    for fold_, (trn_idx, val_idx) in enumerate(folds.split(oof_preds.values, target.values)):
        # Split data
        trn_x, trn_y = oof_preds.iloc[trn_idx], target.iloc[trn_idx]
        val_x, val_y = oof_preds.iloc[val_idx], target.iloc[val_idx]

        # Fit the stacker
        stk.fit(trn_x, trn_y)

        # Looks like there is a bug in BinaryPredictor
        # So compute with weights directly
        z_trn = np.zeros(len(oof_preds.iloc[trn_idx]))
        z_val = np.zeros(len(oof_preds.iloc[val_idx]))
        z_sub = np.zeros(len(sub_preds))
        for i, w in enumerate(stk.weights):
            z_trn += w * oof_preds.iloc[trn_idx].values[:, i]
            z_val += w * oof_preds.iloc[val_idx].values[:, i]
            z_sub += w * sub_preds.values[:, i]
        # Linear predictor has very different response output for each fold
        # So we need to regularize to get a stable output across folds
        regul = LogisticRegression()
        regul.fit(z_trn.reshape(-1, 1), target.iloc[trn_idx])
        stk_oof[val_idx] = regul.predict_proba(z_val.reshape(-1, 1))[:, 1]
        stk_sub += regul.predict_proba(z_sub.reshape(-1, 1))[:, 1] / 5
        print("Fold %2d oof score: %.6f" % (fold_ + 1, eval_gini(target.iloc[val_idx], stk_oof[val_idx])))

        weights[:, fold_] = stk.weights

    oof_score = eval_gini(target, stk_oof)
    print("STK FULL OOF: %.6f" % oof_score)

    # Display weights
    w_df = pd.DataFrame(weights, columns=["f" + str(i + 1) for i in range(folds.n_splits)])
    w_df.insert(0, "oof_preds", oof_preds.columns)
    w_df["coef_"] = w_df.mean(axis=1)

    sub_preds["target"] = stk_sub  # stacker.predict_proba(sub_preds)

    filename = "linear_stacking_" + str(stk.algo) + "_"
    filename += str(int(oof_score * 100000)) + "_"
    filename += curr_date.strftime("%Y_%m_%d_%Hh%M")
    sub_preds[["target"]].to_csv(filename + "_sub.csv", index=True, float_format="%.9f")

    print("weights : ")
    print (w_df)


if __name__ == '__main__':
    main()