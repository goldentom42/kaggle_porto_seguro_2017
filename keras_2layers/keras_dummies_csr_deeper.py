import numpy as np
np.random.seed(20)
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers.advanced_activations import PReLU
from numba import jit
import gc
from sklearn.preprocessing import OneHotEncoder
from keras import regularizers

"""
Using Theano backend.
(595212, 37) (892816, 37)
CSR shape :  (595212, 311)
CSR shape :  (892816, 311)
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
dense_1 (Dense)              (None, 50)                15600     
_________________________________________________________________
p_re_lu_1 (PReLU)            (None, 50)                50        
_________________________________________________________________
dense_2 (Dense)              (None, 25)                1275      
_________________________________________________________________
p_re_lu_2 (PReLU)            (None, 25)                25        
_________________________________________________________________
dense_3 (Dense)              (None, 2)                 52        
=================================================================
Total params: 17,002
Trainable params: 17,002
Non-trainable params: 0
_________________________________________________________________
Fold  0 Bag  0 : 0.278590
Fold  0 Bag  1 : 0.277824
Fold  0 Bag  2 : 0.278867
Fold  0 Bag  3 : 0.278958
Fold  0 Bag  4 : 0.278940
Fold 0 full gini : 0.278940
Fold  1 Bag  0 : 0.268462
Fold  1 Bag  1 : 0.272647
Fold  1 Bag  2 : 0.274431
Fold  1 Bag  3 : 0.274345
Fold  1 Bag  4 : 0.274597
Fold 1 full gini : 0.274597
Fold  2 Bag  0 : 0.273806
Fold  2 Bag  1 : 0.277419
Fold  2 Bag  2 : 0.278587
Fold  2 Bag  3 : 0.279234
Fold  2 Bag  4 : 0.279902
Fold 2 full gini : 0.279902
Fold  3 Bag  0 : 0.272706
Fold  3 Bag  1 : 0.274947
Fold  3 Bag  2 : 0.277518
Fold  3 Bag  3 : 0.277394
Fold  3 Bag  4 : 0.277877
Fold 3 full gini : 0.277877
Fold  4 Bag  0 : 0.275586
Fold  4 Bag  1 : 0.277467
Fold  4 Bag  2 : 0.278024
Fold  4 Bag  3 : 0.278376
Fold  4 Bag  4 : 0.278679
Fold 4 full gini : 0.278679
Full OOF score : 0.276637  <= This is quite different from average of folds 

It scores 0.275 on PLB
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
        if counter == number_of_batches:
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
    model = Sequential()
    model.add(Dense(50, input_shape=(input_dim,),
                    kernel_initializer='he_normal',
                    kernel_regularizer=regularizers.l2(0.0001)
                    ))
    model.add(PReLU())
    # model.add(Dropout(0.2))
    model.add(Dense(25, kernel_initializer='he_normal',
                    kernel_regularizer=regularizers.l2(0.0001)
                    ))
    model.add(PReLU())
    # model.add(Dropout(0.1))
    model.add(Dense(2, kernel_initializer='he_normal', activation='softmax'))
    # Best in stability out of the box are adagrad, adam and nadam
    # adagrad and nadam have the best overfit
    model.compile(loss='binary_crossentropy', optimizer="adagrad")
    return model


if __name__ == '__main__':

    gc.enable()
    trn = pd.read_csv('../../input/train.csv', index_col=0)
    sub = pd.read_csv('../../input/test.csv', index_col=0)

    target = trn.target
    dual_y = pd.concat([1 - target, target], axis=1)

    del trn["target"]

    # trn = trn[trn.columns[:5]]
    trn.drop([f for f in trn if "_calc" in f], axis=1, inplace=True)
    sub.drop([f for f in sub if "_calc" in f], axis=1, inplace=True)

    print(trn.shape, sub.shape)

    n_folds = 5
    folds = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=15)

    # Transform data to dummies and csr_matrix
    for f in ["ps_reg_03", "ps_car_12", "ps_car_13", "ps_car_14"]:
        full_f = pd.concat([trn[f], sub[f]], axis=0)
        full_cut = np.array(pd.cut(full_f, 20, labels=False))
        trn[f] = full_cut[:len(trn)]
        sub[f] = full_cut[len(trn):]
        del full_f
        del full_cut

    one = OneHotEncoder(handle_unknown='ignore')
    # Need to remove negative values before calling OneHotEncoder (so funny)
    trn.replace(-1, 99999, inplace=True)
    sub.replace(-1, 99999, inplace=True)

    trn_csr = one.fit_transform(trn.values)
    sub_csr = one.transform(sub.values)

    print("CSR shape : ", trn_csr.shape)
    print("CSR shape : ", sub_csr.shape)

    oof_ids = target.to_frame()
    sub_ids = sub[["ps_reg_03"]]

    del trn
    del sub

    gc.collect()
    n_epochs = 20  # 20
    n_bags = 5  # 5
    # Go through folds
    oof_preds = np.zeros(trn_csr.shape[0])

    model = nn_model(trn_csr.shape[1])
    model.summary()
    del model

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
                verbose=0,
            )

            oof_preds[val_idx] += model.predict_generator(
                generator=batch_generatorp(val_x, batch_size, False),
                steps=np.ceil(val_x.shape[0] / batch_size),
            )[:, 1] / n_bags

            print("Fold %2d Bag %2d : %.6f"
                  % (i_fold, bag,  eval_gini(val_y[:, 1], oof_preds[val_idx])))

        # Print current fold results
        print("Fold %d full gini : %.6f"
              % (i_fold, eval_gini(val_y[:, 1], oof_preds[val_idx])))

        # break

    oof_score = eval_gini(target, oof_preds)
    print("Full OOF score : %.6f" % oof_score)

    print("Training and modeling submission")
    sub_preds = np.zeros(sub_csr.shape[0])
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
    filename += "keras_2layers_dummies_" + str(int(oof_score * 1e6))
    oof_ids["keras_2layers_dummies"] = oof_preds
    oof_ids[["keras_2layers_dummies", "target"]].to_csv(filename + "_oof.csv", index=True, float_format="%.9f")
    sub_ids["target"] = sub_preds
    sub_ids[["target"]].to_csv(filename + "_sub.csv", index=True, float_format="%.9f")