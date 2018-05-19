from __future__ import division
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing
from sklearn.preprocessing import Imputer
import pandas as pd
import numpy as np
from sklearn.cross_validation import KFold
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier as RF
from sklearn import tree

# read data
df_transaction = pd.read_csv('transactions.csv', error_bad_lines=False)
df_transaction.head()
df_submission = pd.read_csv('sample_submission.csv')
df_train = pd.read_csv('train.csv')
df_is_chrun = [df_submission,df_train]
df_is_chrun = pd.concat(df_is_chrun)
df_is_chrun['is_churn'].sum()
df_is_chrun = [df_submission,df_train]
df_is_chrun = pd.concat(df_is_chrun)
df_is_chrun['is_churn'].sum()
df_log = pd.read_csv('user_logs.csv')
df_member = pd.read_csv('members.csv')
df_merge = pd.merge(df_member, df_is_chrun, how= 'left', on = 'msno')
df_merge = pd.merge(df_merge, df_transaction, how= 'left', on = 'msno')
df_merge = df_merge.drop_duplicates()
df_merge = df_merge.dropna()
df_merge.head()
df_merge['sex'] = np.where(df_merge['gender'] == 'male',1,0)
df_merge[['is_churn']] = df_merge[['is_churn']].astype(int)
df_merge_drop = df_merge_drop.reset_index()
df_merge_drop = df_merge_drop.fillna(method='ffill')
df_merge_drop.head()
y = df_merge_drop.columns
tree = tree.DecisionTreeClassifier()

y = df_merge['is_churn']
drop_cols = ['city','bd','registered_via','msno', 'gender','registration_init_time',
         'transaction_date','membership_expire_date','is_churn','is_auto_renew','is_cancel','sex']
df_merge_drop = df_merge.drop(drop_cols,axis=1)
df_merge_drop = df_merge_drop.dropna() 
df_merge_drop.head()

# define X
X = df_merge_drop.as_matrix().astype(np.float32)

# preprocessing
# standardize features
scaler = preprocessing.StandardScaler()
X = scaler.fit_transform(X)
imp = Imputer(missing_values='NaN', strategy='mean', axis=1)
X = imp.fit_transform(X)

# cleaninig nan and missing value
df_merge_drop.dropna(inplace=True)
indices_to_keep = ~df_merge_drop.isin([np.nan, np.inf, -np.inf]).any(1)

from sklearn.cross_validation import KFold

def acc(y_true,y_pred):
    return np.mean(y_true == y_pred)

	
# cross_validation
def cross_validation(X,y,clf_class,**kwargs):
    kFold = KFold(len(y), n_folds=5, shuffle=True)
    y_pred=y.copy()
    
    for train_index, test_index in kFold:
        X_train, X_test = X[train_index], X[test_index]
        y_train = y[train_index]
        clf = clf_class(**kwargs)
        clf.fit(X_train,y_train)
        y_pred[test_index] = clf.predict(x_test)
        return y_pred
		
# run_algorithm
print("svm: %.3f" % acc(y, cross_validation(X,y,SVC)))
print("rf:%.3f" % acc(y, cross_validation(X,y,RF)))
print("DT:%.3f" % acc(y, cross_validation(X,y,tree)))

import matplotlib.pyplot as plt
import numpy as np
import time
import csv
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.models import Sequential

df_transaction = pd.read_csv('transactions.csv', error_bad_lines=False)
df_transaction.head()
df_submission = pd.read_csv('sample_submission.csv')
df_train = pd.read_csv('train.csv')
df_is_chrun = [df_submission,df_train]
df_is_chrun = pd.concat(df_is_chrun)
df_is_chrun['is_churn'].sum()
df_is_chrun = [df_submission,df_train]
df_is_chrun = pd.concat(df_is_chrun)
df_is_chrun['is_churn'].sum()
df_log = pd.read_csv('user_logs.csv')
df_member = pd.read_csv('members.csv')
df_merge = pd.merge(df_member, df_is_chrun, how= 'left', on = 'msno')
df_merge = pd.merge(df_merge, df_transaction, how= 'left', on = 'msno')
df_merge = df_merge.drop_duplicates()
df_merge = df_merge.dropna()
df_merge.head()
df_merge['sex'] = np.where(df_merge['gender'] == 'male',1,0)
df_merge[['is_churn']] = df_merge[['is_churn']].astype(int)
df_merge_drop = df_merge_drop.reset_index()
df_merge_drop = df_merge_drop.fillna(method='ffill')
df_merge_drop.head()
y = df_merge_drop.columns
tree = tree.DecisionTreeClassifier()

y = df_merge['is_churn']
drop_cols = ['city','bd','registered_via','msno', 'gender','registration_init_time',
         'transaction_date','membership_expire_date','is_churn','is_auto_renew','is_cancel','sex']
df_merge_drop = df_merge.drop(drop_cols,axis=1)
df_merge_drop = df_merge_drop.dropna() 
df_merge_drop.head()

# define X
X = df_merge_drop.as_matrix().astype(np.float32)

def build_model():
    model = Sequential()
    layers = [243, 243, 243, 1]

    model.add(LSTM(
        layers[1],
        input_shape=(None, layers[0]),
        return_sequences=True))
    model.add(Dropout(0.2))

    model.add(LSTM(
        layers[2],
        return_sequences=False))
    model.add(Dropout(0.2))

    model.add(Dense(
        layers[3]))
    model.add(Activation("linear"))

    start = time.time()
    model.compile(loss="mse", optimizer="rmsprop")
    return model


def run_network(model=None, data=None):
    global_start_time = time.time()
    epochs = 1
    ratio = 0.5
    sequence_length = 50

    if model is None:
        model = build_model()

    try:
        model.fit(
            X, y,
            batch_size=512, nb_epoch=epochs, validation_split=0.05)
        predicted = model.predict(X_test)
        predicted = np.reshape(predicted, (predicted.size,))
    except KeyboardInterrupt:
        print 'Training duration (s) : ', time.time() - global_start_time
        return model, y_test, 0

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(y_test[:100])
    plt.plot(predicted[:100])
    plt.show()
    
    return model, y_test, predicted