import numpy as np
np.random.seed(20)
from tensorflow import set_random_seed
set_random_seed(21)
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers.advanced_activations import PReLU
from numba import jit
import gc
from sklearn.preprocessing import OneHotEncoder

"""
This is a one layer, thus shallow NN with PReLU activation
The input is a One-Hot-Encoded version of the dataset
Continuous input is binned using pd.cut 
The output is a 2-dimension softmax 
This takes some time...
It scores 0.73 on PLB
"""


@jit
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
    for i in range(n - 1, -1, -1):
        y_i = y_true[i]
        ntrue += y_i
        gini += y_i * delta
        delta += 1 - y_i
    gini = 1 - 2 * gini / (ntrue * (n - ntrue))
    return gini


def batch_generator(data, target, batch_size, shuffle):
    # chenglong code for fiting from generator
    # (https://www.kaggle.com/c/talkingdata-mobile-user-demographics/forums/t/22567/neural-network-for-sparse-matrices)
    # Compute number of batches
    number_of_batches = np.ceil(data.shape[0] / batch_size)
    # Initialize count of batches created
    counter = 0
    # Create sample data index
    sample_index = np.arange(data.shape[0])
    # Shuffle sample index if required
    if shuffle:
        np.random.shuffle(sample_index)
    while True:
        # Return a batch of data going over the sample index
        batch_index = sample_index[batch_size * counter:batch_size * (counter + 1)]
        batch_data = data[batch_index, :].toarray()
        batch_target = target[batch_index]
        counter += 1
        yield batch_data, batch_target
        # If all data has been used then go over it again with shuffle if required
        if (counter == number_of_batches):
            if shuffle:
                np.random.shuffle(sample_index)
            counter = 0


def batch_generatorp(X, batch_size, shuffle):
    number_of_batches = X.shape[0] / np.ceil(X.shape[0] / batch_size)
    counter = 0
    sample_index = np.arange(X.shape[0])
    while True:
        batch_index = sample_index[batch_size * counter:batch_size * (counter + 1)]
        X_batch = X[batch_index, :].toarray()
        counter += 1
        yield X_batch
        if counter == number_of_batches:
            counter = 0


def nn_model(input_dim):
    nn = Sequential()
    nn.add(Dense(100, input_shape=(input_dim,), kernel_initializer='he_normal'))
    nn.add(PReLU())
    nn.add(Dropout(0.2))
    nn.add(Dense(2, kernel_initializer='he_normal', activation='softmax'))
    nn.compile(loss='binary_crossentropy', optimizer='adagrad')
    return nn


if __name__ == '__main__':
    # Enable Garbage collection
    gc.enable()
    # Read data
    trn = pd.read_csv('../../input/train.csv', index_col=0)
    sub = pd.read_csv('../../input/test.csv', index_col=0)
    # Get target
    target = trn.target
    # Create Dual target for NN softmax
    dual_y = pd.concat([1 - target, target], axis=1)
    del trn["target"]
    # Remove supposedly noisy features
    trn.drop([f for f in trn if "_calc" in f], axis=1, inplace=True)
    sub.drop([f for f in sub if "_calc" in f], axis=1, inplace=True)
    # Output initial shapes
    print(trn.shape, sub.shape)

    # Bin continuous variables before One-Hot Encoding
    for f in ["ps_reg_03", "ps_car_12", "ps_car_13", "ps_car_14"]:
        full_f = pd.concat([trn[f], sub[f]], axis=0)
        full_cut = np.array(pd.cut(full_f, 20, labels=False))
        trn[f] = full_cut[:len(trn)]
        sub[f] = full_cut[len(trn):]
        del full_f
        del full_cut
    # Need to remove negative values before calling OneHotEncoder (so funny)
    trn.replace(-1, 99999, inplace=True)
    sub.replace(-1, 99999, inplace=True)
    # OHE
    one = OneHotEncoder(handle_unknown='ignore')
    trn_csr = one.fit_transform(trn.values)
    sub_csr = one.transform(sub.values)
    print("CSR shapes : ", trn_csr.shape, sub_csr.shape)
    # Keep ids to output files at the end of the script
    oof_ids = target.to_frame()
    sub_ids = sub[["ps_reg_03"]]
    # Del initial DataFrames and collect
    del trn
    del sub
    gc.collect()
    # Display model summary
    model = nn_model(trn_csr.shape[1])
    model.summary()
    del model
    # Create folds
    n_folds = 5
    folds = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=15)
    # Go through folds
    oof_preds = np.zeros(trn_csr.shape[0])
    sub_preds = np.zeros(sub_csr.shape[0])
    n_epochs = 20
    n_bags = 5
    for i_fold, (trn_idx, val_idx) in enumerate(folds.split(target, target)):
        trn_x, val_x = trn_csr[trn_idx], trn_csr[val_idx]
        trn_y, val_y = dual_y.values[trn_idx], dual_y.values[val_idx]
        for bag in range(n_bags):
            model = nn_model(trn_x.shape[1])
            batch_size = 512
            model.fit_generator(
                generator=batch_generator(data=trn_x,
                                          target=trn_y,
                                          batch_size=batch_size,
                                          shuffle=True),  # was True
                steps_per_epoch=np.ceil(trn_x.shape[0] / batch_size),
                epochs=n_epochs,
                validation_data=batch_generator(data=val_x,
                                                target=val_y,
                                                batch_size=batch_size,
                                                shuffle=False),
                validation_steps=np.ceil(val_x.shape[0] / batch_size),
                verbose=1,
            )

            oof_preds[val_idx] += model.predict_generator(
                generator=batch_generatorp(val_x, batch_size, False),
                steps=np.ceil(val_x.shape[0] / batch_size),
            )[:, 1] / n_bags
        # Print current fold results
        print(eval_gini(val_y[:, 1], oof_preds[val_idx]))

    oof_score = eval_gini(target, oof_preds)
    print("Full OOF score : %.6f" % oof_score)

    print("Training and modeling submission")
    for bag in range(n_bags):
        model = nn_model(trn_csr.shape[1])
        batch_size = 512
        model.fit_generator(
            generator=batch_generator(data=trn_csr,
                                      target=dual_y.values,
                                      batch_size=batch_size,
                                      shuffle=True),  # was True
            steps_per_epoch=np.ceil(trn_csr.shape[0] / batch_size),
            epochs=n_epochs,
            verbose=0,
        )

        sub_preds += model.predict_generator(
                generator=batch_generatorp(sub_csr, batch_size, False),
                steps=np.ceil(sub_csr.shape[0] / batch_size),
            )[:, 1] / n_bags

    filename = "../output_preds/"
    filename += "keras_shallow_dummies_" + str(int(oof_score * 1e6))
    oof_ids["keras_dummies"] = oof_preds
    oof_ids[["keras_dummies", "target"]].to_csv(filename + "_oof.csv",
                                                index=True, float_format="%.9f")
    sub_ids["target"] = sub_preds
    sub_ids[["target"]].to_csv(filename + "_sub.csv", index=True, float_format="%.9f")