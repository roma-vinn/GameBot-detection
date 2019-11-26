"""
Created by Roman Polishchenko at 2/7/19
2 course, comp math
Taras Shevchenko National University of Kyiv
email: roma.vinn@gmail.com
"""

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from pandas.plotting import scatter_matrix


def scatter(x: pd.DataFrame, y: pd.Series, save=None):
    """
    Build a scatter matrix.
    :param x: data
    :param y: targets
    :param save: if not None, then save plot to 'scatter_matrix_[save].png' file
    """
    scatter_mat = pd.plotting.scatter_matrix(x, c=y, marker='o',
                                             s=40, hist_kwds={'bins': 15},
                                             figsize=(15, 15))
    if save:
        plt.savefig('scatter_matrix_{}.png'.format(save))
    return scatter_mat


def box_plots(x: pd.DataFrame, y: pd.Series, save=None):
    """
    Build a box plot for each feature.
    :param x: data
    :param y: targets
    :param save: if not None, then save plot to 'box_plot_[save].png' file
    """
    targets = y.unique()
    f, ax = plt.subplots(len(x.columns), len(targets), figsize=(5, 20), sharey='col')
    plt.subplots_adjust(hspace=0.5)
    for index_x, feature in enumerate(x.columns):
        for index_y, target in enumerate(targets):
            subplot = ax[index_x][index_y]
            subplot.set_title(target)
            x[y == target][feature].plot.box(ax=subplot, sharey=True)
    if save:
        plt.savefig('box_plot_{}.png'.format(save))
    plt.show()


def simple_plot(x: pd.DataFrame, y: pd.Series, save=None):
    """
    Build a simple comparative plot for each feature.
    :param x: data
    :param y: targets
    :param save: if not None, then save plot to 'simple_plot_[save].png' file
    """

    targets = y.unique()
    instances_data = []
    for target in targets:
        _data = x[y == target]
        _data.index = range(len(_data))
        instances_data.append((target, _data))
    features = x.columns

    plt.figure(figsize=(15, 15))
    for index, feature in enumerate(features):
        plt.subplot(len(features), 1, index + 1)
        for target, instance_data in instances_data:
            plt.plot(instance_data[feature], label=target)

        plt.ylabel(feature)
        plt.legend(loc='best')

    if save:
        plt.savefig('simple_plot_{}.png'.format(save))

    plt.show()


def tsne_plot(x: pd.DataFrame, y: pd.Series, perp=25, save=None):
    """
    Build a t-SNE plot for all features.
    :param x: data
    :param y: targets
    :param perp: perplexity
    :param save: if not None, then save plot to 'tsne_plot_[save].png' file
    """
    tsne_res = TSNE(perplexity=perp).fit_transform(x)
    xs = tsne_res[:, 0]
    ys = tsne_res[:, 1]
    targets = y.unique()
    for target in targets:
        plt.scatter(xs[y == target], ys[y == target], label=target)
    plt.legend(loc='best')

    if save:
        plt.savefig('tsne_plot_{}.png'.format(save))

    plt.show()
