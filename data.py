import sys

import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectPercentile, f_classif, RFE, chi2
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier


def clean_data():
    df = pd.read_csv('NSDUH06_16_04242018.csv')
    # if ircigage or iralcage has missing value, regard as -1
    # for other missing value: regard as 0
    tmp = df.fillna({'ircigage': -1, 'iralcage': -1})
    clean = tmp.fillna(0)
    clean.to_csv('NSDUH06_16_clean.csv', index=False)


def survey_data():
    """
        1. Pain medication misuse behaviors classification
        (1- yes; 0-no)
        pnrnmflag (if ever misuse pain medication)
        pnrnmyr (if misuse pain medication last year)
        pnrnmmon (if misuse pain medication last month)

        2. Heroin misuse behavior classification
        (1- yes; 0-no)
        herflag (if ever use heroin -including past year)
        heryr  (if last year use heroin)
        hermon (if last month use heroin)
        """
    df = pd.read_csv('NSDUH06_16_clean.csv')
    df = df.drop(columns=['sryr'])
    # print('index: ', len(df))
    # print('column: ', df.columns.size)
    return df


def percentile_selection(goal):
    df = survey_data()
    df_X = df.drop(columns=[goal])
    # df.loc[:, goal].values
    data_X = df_X.as_matrix().astype('int32')
    data_y = df[goal].values.astype('int32')
    selector = SelectPercentile(f_classif, percentile=10)
    score = selector.fit(data_X, data_y).scores_
    rank = np.argsort(score)[::-1]
    print(goal, 'per score', score)
    print(goal, 'per rank', rank)


def pca_selection(goal):
    df = survey_data()
    df_X = df.drop(columns=[goal])
    data_X = df_X.as_matrix().astype('int32')
    data_y = df[goal].values.astype('int32')
    selector = PCA()
    score = selector.fit(data_X).explained_variance_ratio_
    rank = np.argsort(score)[::-1]
    print(goal, 'pca score', score)
    print(goal, 'pca rank', rank)


def rfe_selection(goal):
    df = survey_data()
    df_X = df.drop(columns=[goal])
    data_X = df_X.as_matrix().astype('int32')
    data_y = df[goal].values.astype('int32')
    selector = RFE(LogisticRegression())
    score = selector.fit(data_X, data_y).ranking_
    rank = np.argsort(score)
    print(goal, 'rfe score', score)
    print(goal, 'rfe rank', rank)


def training_split(goal):
    df = survey_data()
    df_X = df.drop(columns=[goal])
    data_X = df_X.as_matrix().astype('int32')
    data_y = df[goal].values.astype('int32')
    X_train, X_test, y_train, y_test = train_test_split(data_X, data_y, test_size=0.4, random_state=3)
    return X_train, X_test, y_train, y_test


def cross_validation(goal):
    index = {'pnrnmflag': 36, 'pnrnmyr': 37, 'pnrnmmon': 38, 'herflag': 18, 'heryr': 19, 'hermon': 20}
    df = survey_data()
    raw = df.as_matrix().astype('int32')

    estimators = [KNeighborsClassifier(n_neighbors=5, weights='uniform', metric='cityblock'),
                  DecisionTreeClassifier(random_state=3),
                  RandomForestClassifier(n_estimators=20, random_state=3),
                  LogisticRegression()]

    for estimator in estimators:
        all = list()
        k_fold = KFold(10)
        for k, (train, test) in enumerate(k_fold.split(raw)):
            X_train = np.delete(raw[train], index[goal], axis=1)
            X_test = np.delete(raw[test], index[goal], axis=1)
            y_train = raw[train][:, index[goal]]
            y_test = raw[test][:, index[goal]]
            model = estimator
            model.fit(X_train, y_train)
            y_test_predicted = list()
            for sample in X_test:
                value = model.predict([sample])[0]
                y_test_predicted.append(value)
            acc = metrics.accuracy_score(y_test, y_test_predicted)
            print(goal, 'DecisionTreeCV', k, acc)
            all.append(acc)
        print(goal, 'DecisionTreeCV-mean', np.asarray(all).mean())


def knn_classification(goal, neighbors):
    # distance 'euclidean', 'minkowski'
    X_train, X_test, y_train, y_test = training_split(goal)
    model = KNeighborsClassifier(n_neighbors=neighbors, weights='uniform', metric='cityblock')
    model.fit(X_train, y_train)
    y_test_predicted = list()
    for sample in X_test:
        value = model.predict([sample])[0]
        y_test_predicted.append(value)
    acc = metrics.accuracy_score(y_test, y_test_predicted)
    print(goal, 'cityblock', neighbors, acc)


def tree_classification(goal):
    X_train, X_test, y_train, y_test = training_split(goal)
    model = DecisionTreeClassifier(random_state=3)
    model.fit(X_train, y_train)
    y_test_predicted = list()
    for sample in X_test:
        value = model.predict([sample])[0]
        y_test_predicted.append(value)
    acc = metrics.accuracy_score(y_test, y_test_predicted)
    print(goal, 'DecisionTree', acc)


def forest_classification(goal, tree=10):
    X_train, X_test, y_train, y_test = training_split(goal)
    model = RandomForestClassifier(n_estimators=tree, random_state=3)
    model.fit(X_train, y_train)
    y_test_predicted = list()
    for sample in X_test:
        value = model.predict([sample])[0]
        y_test_predicted.append(value)
    acc = metrics.accuracy_score(y_test, y_test_predicted)
    print(goal, tree, acc)


def logistic_classification(goal):
    X_train, X_test, y_train, y_test = training_split(goal)
    model = LogisticRegression()
    model.fit(X_train, y_train)
    y_test_predicted = list()
    for sample in X_test:
        value = model.predict([sample])[0]
        y_test_predicted.append(value)
    acc = metrics.accuracy_score(y_test, y_test_predicted)
    print(goal, 'Logistic', acc)
