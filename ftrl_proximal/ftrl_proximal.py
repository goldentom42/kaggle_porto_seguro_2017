"""
Based on supernova's code : https://www.kaggle.com/supernova117
https://www.kaggle.com/supernova117/ftrl-with-validation-and-auc
Modified by olivier : https://www.kaggle.com/ogrellier

It scores 0.267 on Public LB with complete fit on training dataset
The mean submission of each CV fold gives a score of 0.266 on Public LB
"""

from datetime import datetime
from math import exp, log, sqrt
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss, roc_auc_score
from numba import jit
from multiprocessing import *
import gc

np.random.seed(4689571)


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


class FTRLProximal(object):
    """ Our main algorithm: Follow the regularized leader - proximal

        In short,
        this is an adaptive-learning-rate sparse logistic-regression with
        efficient reg_alpha-reg_lambda-regularization

        Reference:
        http://www.eecs.tufts.edu/~dsculley/papers/ad-click-prediction.pdf
    """

    def __init__(self, id_name="id", target_name="target",
                 alpha=1e-2, beta=1.0, reg_alpha=1e-5,
                 reg_lambda=1.0, dim_expo=24,
                 interaction=False, n_jobs=2):
        # parameters
        self.alpha = alpha
        self.beta = beta
        self.reg_alpha = reg_alpha
        self.reg_lambda = reg_lambda

        self.id = id_name
        self.target_name = target_name

        # Check n_jobs
        if n_jobs == -1:
            self.cpus = cpu_count()
        else:
            self.cpus = min(n_jobs, cpu_count())

        # feature related parameters
        self.D = 2 ** dim_expo
        self.interaction = interaction

        # model
        # n: squared sum of past gradients
        # z: weights
        # w: lazy weights
        self.n = np.zeros(self.D)  # [0.] * self.D
        # self.z = [random() for k in range(D)]  # [0.] * D
        self.z = np.random.uniform(0.0, 1.0, self.D)
        self.w = {}

    def _logloss(self, p, y):
        """ FUNCTION: Bounded logloss

            INPUT:
                p: our prediction
                y: real answer

            OUTPUT:
                logarithmic loss of p given y
        """

        p = max(min(p, 1. - 10e-15), 10e-15)
        return -log(p) if y == 1. else -log(1. - p)

    def hash_features(self, df):
        features = [f for f in df if f not in [self.id, self.target_name]]
        for f in features:
            df[f] = df[f].apply(lambda x: abs(hash(f + "_" + str(x)) % self.D))
        return df

    def multi_hash(self, df):
        """ Hash data
                Using multi_transform as defined by the1owl: https://www.kaggle.com/the1owl
                in kernel :
                :param df:
                :return:
                """
        p = Pool(self.cpus)
        df = p.map(self.hash_features, np.array_split(df, self.cpus))
        df = pd.concat(df, axis=0, ignore_index=True).reset_index(drop=True)
        p.close()
        p.join()
        return df

    def multi_predict(self, np_dat):
        preds = np.zeros(len(np_dat))
        for i, row in enumerate(np_dat):
            preds[i], _ = self._predict(row)
        return preds

    def fit(self, data, target, epochs=1, eval_x=None, eval_y=None, verbose=50000):
        """ Fit method """
        # First hash data
        hash_x = self.multi_hash(data)

        # Remove id if in the columns
        if self.id in hash_x:
            hash_x.drop(self.id, axis=1, inplace=True)

        if eval_x is not None:
            val_hash_x = self.multi_hash(eval_x)
            # Remove id if in the columns
            if self.id in val_hash_x:
                val_hash_x.drop(self.id, axis=1, inplace=True)

        # Run all epochs
        start = datetime.now()
        for e_ in range(epochs):
            # Go through all samples to train
            loss_train = 0.
            idx = np.arange(len(hash_x))
            np.random.shuffle(idx)
            y = target.values[idx]
            for count, x in enumerate(hash_x.values[idx]):
                # Train on current sample
                p = self.train(x, y[count])
                # Compute cumulated loss
                loss_train += self._logloss(p, y[count])

                if (eval_x is not None) and ((count + 1) % verbose == 0):
                    # Compute validation losses
                    p = Pool(self.cpus)
                    oof_v = p.map(self.multi_predict, np.array_split(val_hash_x.values, self.cpus))
                    val_preds = np.hstack(oof_v)
                    p.close()
                    p.join()
                    val_logloss = log_loss(eval_y, val_preds)
                    val_auc = roc_auc_score(eval_y, val_preds)
                    # Display current training and validation losses
                    # t_logloss stands for current train_logloss, v for valid
                    print('time_used:%s\tepoch: %-4drows:%d\tt_logloss:%.5f\tv_logloss:%.5f\tv_auc:%.6f'
                          % (datetime.now() - start, e_, count + 1, loss_train / count, val_logloss, val_auc))
                    del val_preds
                    del oof_v
                    gc.collect()
                elif (count + 1) % verbose == 0:
                    # Display current training and validation losses
                    # t_logloss stands for current train_logloss, v for valid
                    print('time_used:%s\tepoch: %-4drows:%d\tt_logloss:%.5f'
                          % (datetime.now() - start, e_, count + 1, loss_train / count))
        del hash_x
        gc.collect()
        return None

    def predict_proba(self, data):
        # First hash data
        hash_x = self.multi_hash(data)

        # Remove id if in the columns
        if self.id in hash_x:
            hash_x.drop(self.id, axis=1, inplace=True)

        # Compute predictions
        p = Pool(cpu_count())
        oof_v = p.map(self.multi_predict, np.array_split(hash_x.values, cpu_count()))
        preds = np.hstack(oof_v)
        p.close()
        p.join()

        del oof_v
        del hash_x
        gc.collect()

        return preds

    def _indices(self, x):
        """ A helper generator that yields the indices in x

            The purpose of this generator is to make the following
            code a bit cleaner when doing feature interaction.
        """
        # first yield index of the bias term
        yield 0

        # then yield the normal indices
        for index in x:
            yield index

        # now yield interactions (if applicable)
        if self.interaction:
            # D = self.D
            L = len(x)

            x = sorted(x)
            for i in range(L):
                for j in range(i + 1, L):
                    # one-hot encode interactions with hash trick
                    yield abs(hash(str(x[i]) + '_' + str(x[j]))) % self.D

    def _predict(self, x):
        """
        Get probability estimation for input x
        The input is expected to be hashed
        outputs the probability and individual values
        """

        # compute probability
        w = {}

        # wTx is the inner product of w and x
        wTx = 0.
        # print(x)
        for i in self._indices(x):
            # sign = -1. if z[i] < 0 else 1.  # get sign of z[i]

            # build w on the fly using z and n, hence the name - lazy weights
            # we are doing this at prediction instead of update time is because
            # this allows us for not storing the complete w
            # if sign * z[i] <= reg_alpha:
            if abs(self.z[i]) <= self.reg_alpha:
                # w[i] vanishes due to reg_alpha regularization
                w[i] = 0.
            else:
                # apply prediction time reg_alpha, reg_lambda regularization to z and get w
                w[i] = (np.sign(self.z[i]) * self.reg_alpha - self.z[i]) / \
                       ((self.beta + sqrt(self.n[i])) / self.alpha + self.reg_lambda)

            wTx += w[i]

        # bounded sigmoid function, this is the probability estimation
        proba = 1. / (1. + exp(-max(min(wTx, 35.), -35.)))

        return proba, w

    def train(self, x, y):
        # Compute prediction
        p, w = self._predict(x)

        # Update weights
        # gradient under logloss
        g = p - y

        # update z and n
        for i in self._indices(x):
            sigma = (sqrt(self.n[i] + g * g) - sqrt(self.n[i])) / self.alpha
            self.z[i] += g - sigma * w[i]
            self.n[i] += g * g

        return p


def main():
    trn = pd.read_csv("../../input/train.csv")
    sub = pd.read_csv("../../input/test.csv")
    y = trn.target
    del trn["target"]

    to_drop = [f for f in trn.columns if "_calc" in f]
    trn.drop(to_drop, axis=1, inplace=True)
    sub.drop(to_drop, axis=1, inplace=True)

    for f in ["ps_reg_03", "ps_car_12", "ps_car_13", "ps_car_14"]:
        full_f = pd.concat([trn[f], sub[f]], axis=0)
        full_cut = np.array(pd.cut(full_f, 20, labels=False))
        trn[f] = full_cut[:len(trn)]
        sub[f] = full_cut[len(trn):]
        del full_f
        del full_cut

    folds = StratifiedKFold(5, True, 15)
    oof_preds = np.zeros(len(trn))
    for trn_idx, val_idx in folds.split(y, y):
        model = FTRLProximal(
            id_name="id",
            target_name="target",
            alpha=.01,
            beta=1.,
            reg_alpha=1e-5,
            reg_lambda=1,
            dim_expo=24,
            interaction=False,
            n_jobs=-1
        )

        model.fit(trn.iloc[trn_idx], y.iloc[trn_idx],
                  epochs=3,
                  eval_x=trn.iloc[val_idx],
                  eval_y=y.iloc[val_idx])

        oof_preds[val_idx] = model.predict_proba(trn.iloc[val_idx])

        print("Curr Fold score = %.6f" % eval_gini(y.iloc[val_idx], oof_preds[val_idx]))

    model = FTRLProximal(
        id_name="id",
        target_name="target",
        alpha=.01,
        beta=1.,
        reg_alpha=1e-5,
        reg_lambda=1,
        dim_expo=24,
        interaction=False,
        n_jobs=-1
    )

    model.fit(trn, y, epochs=3)

    sub_preds = model.predict_proba(sub)

    oof_score = eval_gini(y, oof_preds)
    print("Full oof score : %.6f" % oof_score)

    # Save OOF and submission predictions
    trn["ftrl_proximal"] = oof_preds
    trn["target"] = y

    sub["target"] = sub_preds
    filename = "../output_preds/ftrl_proximal_nomean_" + str(int(oof_score * 1e6))
    sub[["id", "target"]].to_csv(filename + "_sub.csv", index=False, float_format="%.9f")
    trn[["id", "target", "ftrl_proximal"]].to_csv(filename + "_oof.csv", index=False, float_format="%.9f")


if __name__ == '__main__':
    gc.enable()

    main()
    # import cProfile
    # cProfile.run('main()')
