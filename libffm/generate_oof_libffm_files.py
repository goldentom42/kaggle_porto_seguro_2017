import math
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

trn = pd.read_csv('../../input/train.csv')
# test = pd.read_csv('../input/test.csv')
# test.insert(1, 'target', 0)
print(trn.shape)
# print(test.shape)

# x = pd.concat([trn, test])
# x = x.reset_index(drop=True)
unwanted = trn.columns[trn.columns.str.startswith('ps_calc_')]
trn.drop(unwanted, inplace=True, axis=1)

features = trn.columns[2:]
categories = []
for c in features:
    trnno = len(trn.loc[:trn.shape[0], c].unique())
    print(c, trnno)

trn.loc[:, 'ps_reg_03'] = pd.cut(trn['ps_reg_03'], 50, labels=False)
trn.loc[:, 'ps_car_12'] = pd.cut(trn['ps_car_12'], 50, labels=False)
trn.loc[:, 'ps_car_13'] = pd.cut(trn['ps_car_13'], 50, labels=False)
trn.loc[:, 'ps_car_14'] = pd.cut(trn['ps_car_14'], 50, labels=False)
# trn.loc[:, 'ps_car_15'] = pd.cut(trn['ps_car_15'], 50, labels=False)
trn.loc[:, 'ps_car_15'] = (trn['ps_car_15'] ** 2).astype(int)

# Always good to shuffle for SGD type optimizers
# trn = trn.sample(frac=1).reset_index(drop=True)

trn.drop('id', inplace=True, axis=1)

categories = trn.columns[1:]
numerics = []

folds = StratifiedKFold(5, True, 15)
for i_fold, (trn_idx, val_idx) in enumerate(folds.split(trn.target, trn.target)):

    # if i_fold > 1:
    #     break

    train = trn.iloc[trn_idx]
    test = trn.iloc[val_idx]

    currentcode = len(numerics)
    catdict = {}
    catcodes = {}

    for x in numerics:
        catdict[x] = 0

    for x in categories:
        catdict[x] = 1

    noofrows = train.shape[0]
    noofcolumns = len(features)
    with open("trn_ffm_%d.txt" % (i_fold + 1), "w") as text_file:
        for n, r in enumerate(range(noofrows)):
            if ((n % 100000) == 0):
                print('Row', n)
            datastring = ""
            datarow = train.iloc[r].to_dict()
            datastring += str(int(datarow['target']))

            for i, x in enumerate(catdict.keys()):
                # Check if wa have a numerical value
                if (catdict[x] == 0):
                    datastring = datastring + " " + str(i) + ":" + str(i) + ":" + str(datarow[x])
                else:
                    # We have a categorical value
                    # Check categorical feature has a dict in place
                    if x not in catcodes:
                        # Dict not in place so create it
                        catcodes[x] = {}
                        # Increment the current code
                        currentcode += 1
                        # Put the code in the dict for current value
                        # dict[feature][value] = code
                        catcodes[x][datarow[x]] = currentcode
                    elif datarow[x] not in catcodes[x]:
                        # value for feature is not encoded yet so increment the code and add value in the dict
                        currentcode += 1
                        catcodes[x][datarow[x]] = currentcode
                    # Output feature number:value code:1
                    code = catcodes[x][datarow[x]]
                    datastring = datastring + " " + str(i) + ":" + str(int(code)) + ":1"
            datastring += '\n'
            text_file.write(datastring)

    noofrows = test.shape[0]
    noofcolumns = len(features)
    with open("val_ffm_%d.txt" % (i_fold + 1), "w") as text_file:
        for n, r in enumerate(range(noofrows)):
            if n % 100000 == 0:
                print('Row', n)
            datastring = ""
            datarow = test.iloc[r].to_dict()
            datastring += str(int(datarow['target']))

            for i, x in enumerate(catdict.keys()):
                if (catdict[x] == 0):
                    datastring = datastring + " " + str(i) + ":" + str(i) + ":" + str(datarow[x])
                else:
                    if (x not in catcodes):
                        catcodes[x] = {}
                        currentcode += 1
                        catcodes[x][datarow[x]] = currentcode
                    elif (datarow[x] not in catcodes[x]):
                        currentcode += 1
                        catcodes[x][datarow[x]] = currentcode

                    code = catcodes[x][datarow[x]]
                    datastring = datastring + " " + str(i) + ":" + str(int(code)) + ":1"
            datastring += '\n'
            text_file.write(datastring)
