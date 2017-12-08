import math
import numpy as np
import pandas as pd

"""
generates train and test files in libffm format
This has been taken from Scirpus on Kaggle 
"""

train = pd.read_csv('../../input/train.csv')
test = pd.read_csv('../../input/test.csv')
test.insert(1, 'target', 0)
print(train.shape)
print(test.shape)

x = pd.concat([train, test])
x = x.reset_index(drop=True)
unwanted = x.columns[x.columns.str.startswith('ps_calc_')]
x.drop(unwanted, inplace=True, axis=1)

features = x.columns[2:]
categories = []
for c in features:
    trainno = len(x.loc[:train.shape[0], c].unique())
    testno = len(x.loc[train.shape[0]:, c].unique())
    print(c, trainno, testno)

x.loc[:, 'ps_reg_03'] = pd.cut(x['ps_reg_03'], 50, labels=False)
x.loc[:, 'ps_car_12'] = pd.cut(x['ps_car_12'], 50, labels=False)
x.loc[:, 'ps_car_13'] = pd.cut(x['ps_car_13'], 50, labels=False)
x.loc[:, 'ps_car_14'] = pd.cut(x['ps_car_14'], 50, labels=False)

x.loc[:, 'ps_car_15'] = (x['ps_car_15'] ** 2).astype(int)

test = x.loc[train.shape[0]:].copy()
train = x.loc[:train.shape[0]].copy()

# Always good to shuffle for SGD type optimizers
train = train.sample(frac=1).reset_index(drop=True)

train.drop('id', inplace=True, axis=1)
test.drop('id', inplace=True, axis=1)

categories = train.columns[1:]
numerics = []

currentcode = len(numerics)
catdict = {}
catcodes = {}

for x in numerics:
    catdict[x] = 0

for x in categories:
    catdict[x] = 1

noofrows = train.shape[0]
noofcolumns = len(features)
with open("alltrainffm.txt", "w") as text_file:
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
with open("alltestffm.txt", "w") as text_file:
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
