"""
Created by Roman Polishchenko at 2/6/19
2 course, comp math
Taras Shevchenko National University of Kyiv
email: roma.vinn@gmail.com
"""

from time import time
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegressionCV
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (RandomForestClassifier, ExtraTreesClassifier,
                              BaggingClassifier, VotingClassifier)

from Parser.parser import parse_folder, prepare_data, Parser
from Visualization.visualization import tsne_plot, scatter, simple_plot


class Classifier:
    """
    Implementation of classifier for Game bot detection task.

    This class had been written, considering results showed up in 'classifier.ipynb'.
    """
    # exclude K-NN when work with BaggingClassifier
    METHODS = {'LogReg': LogisticRegressionCV(cv=5, max_iter=1000),
               'NBayes': GaussianNB(),
               'DTree': DecisionTreeClassifier(),
               'RForest': RandomForestClassifier(n_estimators=150),
               'ETrees': ExtraTreesClassifier(n_estimators=100, bootstrap=True),
               "KNN": KNeighborsClassifier(n_neighbors=7),
               'Vote': VotingClassifier(estimators=[('lr', LogisticRegressionCV(cv=5, max_iter=1000)),
                                                    ('nb', GaussianNB()),
                                                    # ('kn', KNeighborsClassifier(n_neighbors=7)),
                                                    ('dt', DecisionTreeClassifier()),
                                                    ('et', ExtraTreesClassifier(n_estimators=100, bootstrap=True)),
                                                    ('rf', RandomForestClassifier(n_estimators=150))],
                                        voting='soft')}

    def __init__(self, method='Vote', boosted=False):
        self.clf = Classifier.METHODS.get(method)
        if self.clf is None:
            self.clf = Classifier.METHODS['Vote']

        if boosted:
            self.clf = BaggingClassifier(self.clf)

        self.x_train = pd.DataFrame()
        self.y_train = pd.DataFrame()
        self.x_test = pd.DataFrame()
        self.y_test = pd.DataFrame()
        self.prediction = None

    def get_data(self, data, target, test_size=0.25):
        """
        Getting data, splitting it for train/test.

        :param data: pd.DataFrame
        :param target: pd.Series
        :param test_size: size of test subset
        :return: None
        """
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(data,
                                                                                target,
                                                                                test_size=test_size)
        return self

    def fit(self):
        """
        Fit on x_train, y_train data
        :return: self
        """
        self.clf.fit(self.x_train, self.y_train)
        return self

    def predict(self):
        """
        Return prediction for x_test

        :return: prediction
                    pd.DataFrame
        """
        self.prediction = self.clf.predict(self.x_test)
        return self.prediction

    def score(self):
        """
        Return accuracy
        :return: accuracy
        """
        return self.clf.score(self.x_test, self.y_test)

    def confusion_matrix(self):
        """
        Return the confusion matrix for our results.

        :return: confusion matrix
                    np.array
        """
        if self.prediction is None:
            self.predict()
        return confusion_matrix(self.y_test, self.prediction)


def test(method, features=('gpm', 'tbc', 'epm', 'aht')):
    # test function on "../logs_new" data
    with open('result-{}.txt'.format(method), 'w') as f:
        for time_window in range(5000, 30001, 5000):
            print('Working with time window equals to {} ms...'.format(time_window))
            begin = time()
            df = parse_folder('../logs_new', step=time_window)
            print('[{} ms] Data was parsing {} seconds'.format(time_window,
                                                               round(time() - begin, 2)), file=f)
            features, targets = prepare_data(df, features=features)
            clf = Classifier(method=method, boosted=False).get_data(features, targets).fit()
            print('Score:', clf.score(), file=f)
            print('Confusion matrix:\n{}\n'.format(clf.confusion_matrix()), file=f)


def test_ant():
    # test function on i.antipiev logs
    with open('../logs_ant/result_boosted.txt', 'w') as f:
        for time_window in range(5000, 30001, 5000):
            print('Working with time window equals to {} ms...'.format(time_window), file=f)
            bot_parser = Parser('../logs_ant/log_bot_Game.csv', '../logs_ant/log_bot_Touch.csv', step=time_window)
            human_parser = Parser('../logs_ant/log_human_Game.csv', '../logs_ant/log_human_Touch.csv', step=time_window)
            bot_parser.set_label(1)
            human_parser.set_label(0)
            data = pd.concat([bot_parser.df, human_parser.df], ignore_index=True)
            features, targets = prepare_data(data)
            clf = Classifier(boosted=True).get_data(features, targets, test_size=0.25).fit()
            print('Score:', clf.score(), file=f)
            print('Confusion matrix:\n{}\n'.format(clf.confusion_matrix()), file=f)


def search_params():
    """
    Selecting best params for our best model - VotingClassifier
    :return: dict {clf__param: best value}
    """
    df = parse_folder('../logs_new', step=5000)
    features, targets = prepare_data(df)
    clf = Classifier()
    params = {'kn__n_neighbors': [3, 5, 7, 9], 'et__bootstrap': [True, False],
              'rf__n_estimators': [50, 100, 150], 'et__n_estimators': [50, 100, 150]}
    grid = GridSearchCV(estimator=clf.clf, cv=5, scoring='accuracy', n_jobs=-1, param_grid=params)
    grid.fit(features, targets)
    return grid.best_params_
    #  {'et__bootstrap': True, 'et__n_estimators': 100, 'kn__n_neighbors': 7, 'rf__n_estimators': 150}


def human_vs_human_2():
    # Human vs human classification
    with open('human_vs_human_2.txt', 'w') as f:
        for time_window in range(5000, 30001, 5000):
            print('Working with {} second window...'.format(time_window))
            h1_logs = Parser('../logs/human_game_logs_1.csv', '../logs/human_touch_logs_1.csv', step=time_window)
            h2_logs = Parser('../logs_new/human/session_2019-01-27_15-01-22/gamelog_Game_2019-01-27_15-01-22.csv',
                             '../logs_new/human/session_2019-01-27_15-01-22/gamelog_Touch_2019-01-27_15-01-22.csv',
                             step=time_window)

            h1_logs.set_label(1)
            h2_logs.set_label(2)

            data = pd.concat([h1_logs.df, h2_logs.df], ignore_index=True)
            features, targets = prepare_data(data)
            if time_window == 20000:
                scatter(features, targets, save='h_2')
                simple_plot(features, targets, save='h_2')
                tsne_plot(features, targets, perp=10, save='h_2')
            clf = Classifier(boosted=False).get_data(features, targets, test_size=0.25).fit()
            print('Score:', clf.score(), file=f)
            print('Confusion matrix:\n{}\n'.format(clf.confusion_matrix()), file=f)


def human_vs_human_3():
    # Human vs human vs human classification
    with open('human_vs_human_3.txt', 'w') as f:
        for time_window in range(5000, 30001, 5000):
            print('Working with {} second window...'.format(time_window))
            h1_logs = Parser('../logs/human_game_logs_1.csv', '../logs/human_touch_logs_1.csv', step=time_window)
            h2_logs = Parser('../logs_new/human/session_2019-01-27_15-01-22/gamelog_Game_2019-01-27_15-01-22.csv',
                             '../logs_new/human/session_2019-01-27_15-01-22/gamelog_Touch_2019-01-27_15-01-22.csv',
                             step=time_window)
            h3_logs = Parser('../logs_new/human/session_2019-01-25_14-10-15/gamelog_Game_2019-01-25_14-10-15.csv',
                             '../logs_new/human/session_2019-01-25_14-10-15/gamelog_Touch_2019-01-25_14-10-15.csv',
                             step=time_window)

            h1_logs.set_label(1)
            h2_logs.set_label(2)
            h3_logs.set_label(3)

            data = pd.concat([h1_logs.df, h2_logs.df, h3_logs.df], ignore_index=True)
            features, targets = prepare_data(data)

            if time_window == 20000:
                scatter(features, targets, save='h_3')
                simple_plot(features, targets, save='h_3')
                tsne_plot(features, targets, perp=10, save='h_3')

            clf = Classifier(boosted=False).get_data(features, targets, test_size=0.25).fit()
            print('Score:', clf.score(), file=f)
            print('Confusion matrix:\n{}\n'.format(clf.confusion_matrix()), file=f)


if __name__ == '__main__':
    # search_params()
    # test_ant()

    test('Vote')
    human_vs_human_2()
    human_vs_human_3()
