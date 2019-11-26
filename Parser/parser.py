"""
Created by Roman Polishchenko
2 course, comp math
Taras Shevchenko National University of Kyiv
email: roma.vinn@gmail.com
"""
import pandas as pd
import numpy as np
import os
from time import time
from sklearn import preprocessing


class Parser:
    """
    Class, that implements data parser from two log files.
    """
    GAME_LOG_HEADER = ['Time',
                       'Event',
                       'X-coord',
                       'Y-coord']
    TOUCH_LOG_HEADER = ['Timestamp-sec',
                        'Timestamp-ms',
                        'touch-event',
                        'X-coord',
                        'Y-coord',
                        'Contact-major',
                        'Contact-minor']
    FEATURES = dict()

    def __init__(self, game_log_path, touch_log_path, step=10000, update=True):
        """
        :param game_log_path: path to game_log file
        :param touch_log_path: path to touch_log file
        :param step: splitting step in ms
        """
        Parser.FEATURES = {'apm': self.apm,
                           'epm': self.epm,
                           'gpm': self.gpm,
                           'aht': self.aht,
                           'dev': Parser.dev,
                           'tln': Parser.tln,
                           'dbc': Parser.dbc,
                           'tbc': self.tbc
                           }

        self.step = step
        self.game_log = pd.read_csv(game_log_path, header=None, names=Parser.GAME_LOG_HEADER)
        self.touch_log = pd.read_csv(touch_log_path, header=None, names=Parser.TOUCH_LOG_HEADER)

        self._uniform_time()

        self.stop_time = self._get_stop_time()

        self.df = pd.DataFrame(columns=Parser.FEATURES.keys())
        if update:
            self.update_df()

    def _uniform_time(self):
        """
        Adding column "Time" to self.touch_log, that is matched to
        the same column in self.game_log.
        :return: None
        """
        first_touch_row = self.touch_log[:1]
        first_game_row = self.game_log[:1]
        start_time = first_touch_row['Timestamp-sec'] + first_touch_row['Timestamp-ms'] / 1000

        start_time = start_time - first_game_row['Time'] / 1000

        self.touch_log['Time'] = self.touch_log['Timestamp-sec'] + self.touch_log['Timestamp-ms'] / 1000
        self.touch_log['Time'] -= start_time[0]

        self.touch_log['Time'] = round(self.touch_log['Time'] * 1000)
        # remove first "ghost" clicks
        self.touch_log = self.touch_log[1:]
        self.game_log = self.game_log[1:]

    def _get_stop_time(self):
        """
        Getting last time in ms, when some action were observed.
        :return: stop_time [int]
        """
        stop_time = self.game_log[-1:]['Time']
        return stop_time[stop_time.keys()[0]]

    def set_step(self, new_step):
        self.step = new_step

    def set_label(self, label):
        self.df['label'] = label

    def update_df(self):
        # taking the subsets of data according to the "time windows" they relate to
        def _get_tmp_touch(pointer):
            criteria1 = self.touch_log['Time'] >= pointer
            criteria2 = self.touch_log['Time'] < pointer + self.step
            tmp_touch = self.touch_log[criteria1 & criteria2]
            return tmp_touch

        def _get_tmp_game(pointer):
            criteria1 = self.game_log['Time'] >= pointer
            criteria2 = self.game_log['Time'] < pointer + self.step
            tmp_game = self.game_log[criteria1 & criteria2]
            return tmp_game

        self.df = pd.concat([pd.DataFrame([[func(_get_tmp_game(pointer), _get_tmp_touch(pointer))
                                           for func in Parser.FEATURES.values()]],
                                          columns=Parser.FEATURES.keys())
                             for pointer in range(0, self.stop_time, self.step)],
                            ignore_index=True)
        self.df = pd.DataFrame(preprocessing.normalize(self.df))
        self.df.columns = self.FEATURES.keys()

    # FEATURES

    def apm(self, _game_log, _touch_log):
        """
        Actions (clicks) per minute.
        :param _game_log: game log data frame
        :param _touch_log: touch log data frame
        :return: apm
        """
        try:
            clicks = _game_log['Event'].value_counts()[' Touch']
        except KeyError:
            clicks = 0
        return clicks * 60000 / self.step

    def gpm(self, _game_log, _touch_log):
        """
        Gained coins per minute.
        :param _game_log: game log data frame
        :param _touch_log: touch log data frame
        :return: gpm
        """
        try:
            coins = _game_log['Event'].value_counts()[' Drop']
        except KeyError:
            coins = 0
        return coins * 60000 / self.step

    def epm(self, _game_log, _touch_log):
        """
        Gained experience per minute.
        :param _game_log: game log data frame
        :param _touch_log: touch log data frame
        :return: epm
        """
        try:
            exp = _game_log['Event'].value_counts()[' Fight']
        except KeyError:
            exp = 0
        return exp * 10 * 60000 / self.step

    def aht(self, _game_log, _touch_log):
        """
        Average holding time per minute.
        :param _game_log: game log data frame
        :param _touch_log: touch log data frame
        :return: aht
        """
        holding_time = list()

        try:
            # if previous click started in previous "time window"
            if _touch_log['touch-event'].iloc[0] == 0:
                flag = True
            else:
                flag = False
        # if there wasn`t any data points in concrete "time window" -> return 0
        except IndexError:
            return 0

        begin_time = 0
        for index, row in _touch_log.iterrows():
            # when the finger is down and current touch hadn`t started
            if row['touch-event'] == 2 and not flag:
                flag = True
                begin_time = row['Time'] % self.step
            # when the finger is up and current touch had started before
            elif row['touch-event'] == 0 and flag:
                flag = False
                holding_time.append(row['Time'] % self.step - begin_time)

        # if current click not finished in current "time window"
        if flag:
            holding_time.append(self.step - begin_time % self.step)

        # if there wasn`t any data points in concrete "time window" -> return 0
        if not holding_time:
            return 0
        else:
            return np.array(holding_time).mean()

    @staticmethod
    def dev(_game_log, _touch_log):
        """
        Deviation of clicks.
        :param _game_log: game log data frame
        :param _touch_log: touch log data frame
        :return: dev
        """
        dev = np.std(_touch_log[['X-coord', 'Y-coord']])

        # if there wasn`t any data points in concrete "time window" -> return 0
        res = 0
        if not np.any(np.isnan(dev)):
            res = np.linalg.norm(dev)
        return res

    @staticmethod
    def tln(_game_log, _touch_log):
        """
        Trajectory length.
        :param _game_log: game log data frame
        :param _touch_log: touch log data frame
        :return: tln
        """
        tln = 0

        # Lets say, that starting coordinates are (1010, 540), because S8
        # resolution is 1080x2020 and the game is in landscape mode
        curr_x = 1010
        curr_y = 540

        for index, row in _game_log.iterrows():
            if row['Event'] == ' Touch':
                next_x = row['X-coord']
                next_y = row['Y-coord']

                tln += np.sqrt((next_x - curr_x)**2 + (next_y - curr_y)**2)

                curr_x = next_x
                curr_y = next_y

        return tln

    @staticmethod
    def dbc(_game_log, _touch_log):
        """
        Average distance between clicks.
        :param _game_log: game log data frame
        :param _touch_log: touch log data frame
        :return: dbc
        """
        distances = list()
        dbc = 0

        # Lets say, that starting coordinates are (1010, 540), because S8
        # resolution is 1080x2020 and the game is in landscape mode
        curr_x = 1010
        curr_y = 540
        for index, row in _game_log.iterrows():
            if row['Event'] == ' Touch':
                next_x = row['X-coord']
                next_y = row['Y-coord']

                distances.append(np.sqrt((next_x - curr_x)**2 + (next_y - curr_y)**2))

                curr_x = next_x
                curr_y = next_y

        # at least one click was made
        if distances:
            dbc = np.mean(distances)

        return dbc

    def tbc(self, _game_log, _touch_log):
        """
        Average time between clicks.
        :param _game_log: game log data frame
        :param _touch_log: touch log data frame
        :return: tbc
        """
        tbc = [0]

        for index, row in _game_log.iterrows():
            if row['Event'] == ' Touch':
                tbc.append(row['Time'] % self.step - tbc[-1])

        return np.mean(tbc)


def parse_folder(path, classes=('human', 'bot'), step=10000, save=False):
    """
    Parse logs from folder that must be organized in particular way:

    log_folder
        bot
            session_1
                gamelog_Game_1.csv
                gamelog_Touch_1.csv
            session_2
            ...
            session_n
        human
            session_1
                gamelog_Game_1.csv
                gamelog_Touch_1.csv
            session_2
            ...
            session_m
        other_instance
            ...
                ...

    :param path: path to log folder
    :param classes: tuple of class names (in resulting df: label == it`s number in tuple)
    :param step: split step (time window size)
    :param save: True – save the data to the file, False (default) – don`t save
    :return: resulting DataFrame
    """
    instances = os.listdir(path)
    data = list()
    for label, cls in enumerate(classes):
        if cls not in instances:
            raise 'There is no logs for class {}'.format(cls)
        # get all sessions, filtering excess folders, if they are
        sessions = list(filter(lambda x: 'session' in x, os.listdir(path + '/' + cls)))
        for session in sessions:
            rel_path = path + '/' + cls + '/' + session + '/'
            files = sorted(os.listdir(rel_path))
            game_log_path = rel_path + files[0]
            touch_log_path = rel_path + files[1]
            parser = Parser(game_log_path, touch_log_path, step=step)
            parser.set_label(label)
            data.append(parser.df)
    data = pd.concat(data, ignore_index=True)
    if save:
        data.to_csv('{}-{}-data.csv'.format('-'.join(classes), step))
    return data


def prepare_data(data, features=('gpm', 'tbc', 'epm', 'aht')):
    """
    Preparing data: splitting it on features/targets.
    :param data: pd.DataFrame with column 'label' contains labels for data points
    :param features: wanted features to use in classification
    :return: Features data frame and targets series
                (pd.DataFrame, pd.Series)
    """
    features = data[list(features)]
    targets = data['label']
    return features, targets


def test():
    bot_parser = Parser('../logs/bot_game_logs_1.csv', '../logs/bot_touch_logs_1.csv')
    human_parser = Parser('../logs/human_game_logs_1.csv', '../logs/human_touch_logs_1.csv')

    bot_parser.set_label(1)
    human_parser.set_label(0)
    data = pd.concat([bot_parser.df, human_parser.df], ignore_index=True)
    # print(data)
    data.to_csv(path_or_buf='resulting_df.csv')


# test
if __name__ == '__main__':

    pd.set_option("display.max_rows", 200)
    pd.set_option('display.max_columns', 10)
    pd.set_option('display.width', 200)

    # test()
    begin = time()
    logs = parse_folder('../logs_new', step=30000, save=True)
    print('Time elapsed: {}s'.format(time() - begin))
    print(logs)
    print(logs.info())
