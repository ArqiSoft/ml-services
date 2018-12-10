"""
Package for make plots and maps needed for 'human-validate'
training and prediction processes
"""

import glob
import math
import os
import re

import matplotlib
import numpy as np
import pandas
import seaborn as sns
from keras.callbacks import Callback
from sklearn.metrics import r2_score, confusion_matrix, roc_curve, auc

from MLLogger import BaseMLLogger

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import interp

LOGGER = BaseMLLogger(log_name='sds_ml_plotter_logger')
# plots types names
ROC_PLOT = 'roc_plot'
CONFUSION_MATRIX = 'confusion_matrix'
REGRESSION_RESULT_TEST = 'regression_results_test'
REGRESSION_RESULT_TRAIN = 'regression_results_train'
RADAR_PLOT = 'radar_plot'
THUMBNAIL_IMAGE = 'thumbnail_image'
TRAIN_COLOR = 'blue'
TEST_COLOR = 'red'
VALIDATION_COLOR = 'green'


class BatchLogger(Callback):
    def __init__(self, display):
        """
        Method for initialize BatchLogger

        :param display: display progress every 5%
        :type display: WHICH TYPE?
        """
        # TODO check docstring

        self.seen = 0
        self.display = display

    def on_batch_end(self, batch, logs=None):
        """
        Method for what?

        :param batch:
        :param logs:
        :return:
        """
        # TODO check docstring

        logs = logs or {}
        total_steps = self.params['samples'] * (self.params['epochs'] - 1)
        self.seen += logs.get('size', 0)
        percent_check = total_steps * self.display / 100

        if self.seen % percent_check == 0:
            all_param = (100 * self.seen / total_steps, logs.get('loss'))
            print('{}% Done  Loss: {:.4}'.format(all_param[0], all_param[1]))


def plot_train_history(model, model_name, sub_folder):
    """
    Function that plots and renders history of the model's training
    (loss changes on epoch)

    :param model: model training history from keras
    :param model_name: name of the model
    :param sub_folder: working directory
    :return: path to the image of plot loss vs epochs
    """
    # TODO check docstring

    lw = 2
    plt.figure(figsize=(8, 6))
    plt.plot(
        model.epoch, model.history['loss'], c='b', label='Train loss', lw=lw)
    plt.plot(
        model.epoch, model.history['val_loss'], c='g',
        label='Validation loss', lw=lw
    )
    # plt.ylim([0.0, max(model.history['loss'])])
    # plt.xlim([0, model.epoch[-1]])
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(os.path.join(
        sub_folder, '{}_history_plot'.format(model_name)
    ))
    img_path = os.path.abspath(os.path.join(
        sub_folder, '{}_history_plot.png'.format(model_name)
    ))
    plt.close()

    return img_path


def plot_train_history_reg(model, model_name, sub_folder, metric='loss'):
    """
    Method for plot train history reg (what is reg?)

    :param model: model training history from keras
    :param model_name:
    :param sub_folder:
    :param metric: metric to plot, default is 'loss'
        which is 'mean_squared_error'
        (optional is 'mean_absolute_error' or 'coeff_determination' )
    :type metric: str
    :return: plot loss vs epochs
    """
    # TODO check docstring

    lw = 2
    plt.figure(figsize=(8, 6))
    if metric == 'loss':
        metric_name = 'mean squared error'
        plt.ylim([0.0, 10.0])
    else:
        metric_name = metric.replace('_', ' ')
    if metric == 'coeff_determination':
        plt.ylim([0.0, 1.0])
    plt.plot(
        model.epoch, model.history[metric], c='b',
        label='Train {}'.format(metric_name), lw=lw
    )
    plt.plot(
        model.epoch, model.history['val_{}'.format(metric)], c='g',
        label='Validation {}'.format(metric_name), lw=lw
    )
    # plt.xlim([0, model.epoch[-1]])
    plt.xlabel('Epochs')
    plt.ylabel(metric_name)
    plt.legend()
    plt.savefig(os.path.join(
        sub_folder, '{}_reg_history_plot.png'.format(model_name.split('.')[0])
    ))
    img_path = os.path.abspath(os.path.join(
        sub_folder, '{}_reg__history_plot.png'.format(model_name)
    ))
    plt.close()

    return img_path


def roc_plot(classifier, fold_name=None, lw=2):
    # TODO make docstring
    model_name = classifier.model_name

    # plot two ROC: train, test and train only
    fig = plt.figure(figsize=(13, 6))

    # Train and test ROC
    fig.add_subplot(1, 2, 1)
    # plot 50%
    plt.plot([0, 1], [0, 1], linestyle='--', lw=lw, color='k')
    # plot train ROC
    fpr_tr, tpr_tr, thresholds_tr = roc_curve(
        classifier.y_train['value'], classifier.predict_probas['train'])

    metrics = classifier.metrics[fold_name]
    roc_auc_train = float(metrics[('train', 'AUC')])
    plt.plot(
        fpr_tr, tpr_tr, lw=lw, color=TRAIN_COLOR,
        label='ROC {} (area = {:0.2f})'.format('train', roc_auc_train)
    )

    # plot test ROC
    if classifier.x_test.shape[0] > 0:
        fpr_ts, tpr_ts, thresholds_ts = roc_curve(
            classifier.y_test['value'], classifier.predict_probas['test'])
        roc_auc_test = float(metrics[('test', 'AUC')])
        plt.plot(
            fpr_ts, tpr_ts, lw=lw + 1, color=TEST_COLOR,
            label='ROC {} (area = {:0.2f})'.format('test', roc_auc_test)
        )
    if classifier.y_valid is not None:
        fpr_val, tpr_val, thresholds_val = roc_curve(
            classifier.y_valid['value'],
            classifier.predict_probas['validation']
        )
        roc_auc_valid = float(metrics[('validation', 'AUC')])
        plt.plot(
            fpr_val, tpr_val, lw=lw + 1, color=VALIDATION_COLOR,
            label='ROC {} (area = {:0.2f})'.format('validation', roc_auc_valid)
        )

    plt.xlim([-0.01, 1.01])
    plt.ylim([-0.01, 1.01])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(
        'Receiver operating characteristic for\n {} classifier K-fold {}\n'.format(
            model_name, fold_name
        )
    )
    plt.legend(loc='lower right')

    # Train ROC
    fig.add_subplot(1, 2, 2)
    # plot 50%
    plt.plot([0, 1], [0, 1], linestyle='--', lw=lw, color='k')
    # plot train ROC
    plt.plot(fpr_tr, tpr_tr, lw=lw, color=TRAIN_COLOR,
             label='ROC (area = {:0.2f})'.format(roc_auc_train))
    plt.xlim([-0.01, 1.01])
    plt.ylim([-0.01, 1.01])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(
        'Receiver operating characteristic for\n {} classifier K-fold {}\n'.format(
            model_name, fold_name
        )
    )
    plt.legend(loc='lower right')
    plt.tight_layout()

    # make image name, based on fold name
    if fold_name:
        fold_name = 'fold_{}_'.format(fold_name)
    img_file_name = '{}_{}ROC_plot.png'.format(
        '_'.join(model_name.split()), fold_name)
    # make image filepath, with using classifier subfolder
    img_path = os.path.abspath(
        os.path.join(classifier.sub_folder, img_file_name)
    )

    # save image
    plt.savefig(img_path)
    plt.close()

    return img_path


def multi_roc_plot(classifier, fold_number=None, lw=2):
    # TODO make docstring
    # TODO 200 strings for 1 method?
    model_name = classifier.model_name
    n_subsets = 1 + int(len(classifier.x_test) > 0) + int(
        classifier.y_valid is not None)
    n_classes = classifier.y_train.shape[1]

    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    fig = plt.figure(figsize=(6 * n_subsets, 6))

    fpr['micro'], tpr['micro'], _ = roc_curve(
        classifier.y_train.ravel(), classifier.predict_probas['train'].ravel())
    roc_auc['micro'] = auc(fpr['micro'], tpr['micro'])

    fig.add_subplot(1, n_subsets, 1)
    plt.plot([0, 1], [0, 1], linestyle='--', lw=lw, color='k')

    plt.plot(
        fpr['micro'], tpr['micro'], lw=lw, color=TRAIN_COLOR,
        label='micro-average train ROC curve (area = {0:0.2f})'.format(
            roc_auc['micro'])
    )

    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(
            classifier.y_train[:, i], classifier.predict_probas['train'][:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
        plt.plot(
            fpr[i], tpr[i], lw=lw,
            label='train ROC curve of class {0} (area = {1:0.2f})'.format(
                i, roc_auc[i])
        )

    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
    mean_tpr = np.zeros_like(all_fpr)

    for i in range(n_classes):
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])

    mean_tpr /= n_classes
    fpr['macro'] = all_fpr
    tpr['macro'] = mean_tpr
    roc_auc['macro'] = auc(fpr['macro'], tpr['macro'])

    plt.plot([0, 1], [0, 1], linestyle='--', lw=lw, color='k')
    plt.plot(
        fpr['macro'], tpr['macro'], lw=lw,
        label='macro-average train ROC curve (area = {0:0.2f})'.format(
            roc_auc['macro'])
    )

    plt.xlim([-0.01, 1.01])
    plt.ylim([-0.01, 1.01])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(
        'Receiver operating characteristic for\n {} train K-fold {}\n'.format(
            model_name, fold_number
        )
    )
    plt.legend(loc='lower right')

    if len(classifier.x_test) > 0:
        fpr['micro'], tpr['micro'], _ = roc_curve(
            classifier.y_test.ravel(),
            classifier.predict_probas['test'].ravel()
        )
        roc_auc['micro'] = auc(fpr['micro'], tpr['micro'])

        fig.add_subplot(1, n_subsets, 2)

        plt.plot([0, 1], [0, 1], linestyle='--', lw=lw, color='k')
        plt.plot(
            fpr['micro'], tpr['micro'], lw=lw, color=TEST_COLOR,
            label='micro-average test ROC curve (area = {0:0.2f})'.format(
                roc_auc['micro'])
        )

        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(
                classifier.y_test[:, i],
                classifier.predict_probas['test'][:, i]
            )
            roc_auc[i] = auc(fpr[i], tpr[i])
            plt.plot(
                fpr[i], tpr[i], lw=lw,
                label='test ROC curve of class {0} (area = {1:0.2f})'.format(
                    i, roc_auc[i])
            )

        all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
        mean_tpr = np.zeros_like(all_fpr)

        for i in range(n_classes):
            mean_tpr += interp(all_fpr, fpr[i], tpr[i])

        mean_tpr /= n_classes
        fpr['macro'] = all_fpr
        tpr['macro'] = mean_tpr
        roc_auc['macro'] = auc(fpr['macro'], tpr['macro'])

        plt.plot([0, 1], [0, 1], linestyle='--', lw=lw, color='k')
        plt.plot(
            fpr['macro'], tpr['macro'], lw=lw,
            label='macro-average test ROC curve (area = {0:0.2f})'.format(
                roc_auc['macro'])
        )

        plt.xlim([-0.01, 1.01])
        plt.ylim([-0.01, 1.01])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(
            'Receiver operating characteristic for\n {} test K-fold {}\n'.format(
                model_name, fold_number
            )
        )
        plt.legend(loc='lower right')

    if classifier.y_valid is not None:
        fpr['micro'], tpr['micro'], _ = roc_curve(
            classifier.y_valid.ravel(),
            classifier.predict_probas['validation'].ravel()
        )
        roc_auc['micro'] = auc(fpr['micro'], tpr['micro'])

        fig.add_subplot(1, n_subsets, 3)

        plt.plot([0, 1], [0, 1], linestyle='--', lw=lw, color='k')
        plt.plot(
            fpr['micro'], tpr['micro'], lw=lw, color=VALIDATION_COLOR,
            label='micro-average validation ROC curve (area = {0:0.2f})'.format(
                roc_auc['micro'])
        )

        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(
                classifier.y_valid[:, i],
                classifier.predict_probas['validation'][:, i]
            )
            roc_auc[i] = auc(fpr[i], tpr[i])
            plt.plot(
                fpr[i], tpr[i], lw=lw,
                label='validation ROC curve of class {0} (area = {1:0.2f})'.format(i, roc_auc[i])
            )

        all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
        mean_tpr = np.zeros_like(all_fpr)

        for i in range(n_classes):
            mean_tpr += interp(all_fpr, fpr[i], tpr[i])

        mean_tpr /= n_classes
        fpr['macro'] = all_fpr
        tpr['macro'] = mean_tpr
        roc_auc['macro'] = auc(fpr['macro'], tpr['macro'])

        plt.plot([0, 1], [0, 1], linestyle='--', lw=lw, color='k')
        plt.plot(
            fpr['macro'], tpr['macro'], lw=lw,
            label='macro-average validation ROC curve (area = {0:0.2f})'.format(roc_auc['macro'])
        )

        plt.xlim([-0.01, 1.01])
        plt.ylim([-0.01, 1.01])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(
            'Receiver operating characteristic for\n {} validation K-fold {}\n'.format(
                model_name, fold_number
            )
        )
        plt.legend(loc='lower right')

    model_name = '_'.join(model_name.split())

    plt.tight_layout()

    fold_name = None
    if fold_number:
        fold_name = 'fold_{}_'.format(fold_number)

    img_file_name = '{}_{}ROC_plot.png'.format(model_name, fold_name)
    img_path = os.path.abspath(
        os.path.join(classifier.sub_folder, img_file_name)
    )
    plt.savefig(img_path)
    plt.close()

    return img_path


def plot_heat_maps_cm(
        confusion_matrix_train, model_name=None, confusion_matrix_test=None,
        confusion_matrix_validation=None, fold_number=None
):
    """
    Method for plots confusion matrix heat map

    :param confusion_matrix_validation: confusion matrix validation
    :param model_name: name of trained model for which make plots
    :param confusion_matrix_train: confusion matrix trained
    :param confusion_matrix_test: confusion matrix test
    :return:
    """

    plot_count = 1
    if confusion_matrix_validation is not None:
        plot_count += 1
    if confusion_matrix_test is not None:
        plot_count += 1

    plot_number = 1
    figure = plt.figure(figsize=(13, 4))

    add_confusion_matrix_subplot(
        figure, confusion_matrix_train, plot_number, plot_count,
        'train', model_name, fold_number=fold_number
    )

    if confusion_matrix_validation is not None:
        plot_number += 1
        add_confusion_matrix_subplot(
            figure, confusion_matrix_validation, plot_number, plot_count,
            'validation', model_name, fold_number=fold_number
        )

    if confusion_matrix_test is not None:
        plot_number += 1
        add_confusion_matrix_subplot(
            figure, confusion_matrix_test, plot_number, plot_count,
            'test', model_name, fold_number=fold_number
        )

    return plt


def add_confusion_matrix_subplot(
        figure, confusion_matrix, plot_number, plot_count, dataset_type,
        model_name, fold_number=None
):
    """
    Method to add confusion matrix heat map to figure

    :param figure: figure object with heat maps
    :param confusion_matrix: heat map to add
    :param plot_number: column number
    :param plot_count: count of heat maps on figure
    :param dataset_type: test, train, validation heat map
    :param model_name: name of model
    :param fold_number: number of fold
    :return: updated figure
    """

    figure.add_subplot(1, plot_count, plot_number)
    sns.heatmap(
        confusion_matrix, annot=True,
        annot_kws={'fontsize': 12}, fmt='d'
    )
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    fold_name = None
    if fold_number:
        fold_name = 'K-fold {}'.format(fold_number)

    plt.title('Confusion matrix for {} dataset\n'
              '{} model {}'.format(dataset_type, model_name, fold_name))

    return figure


def plot_cm_final(classifier, metrics, fold_name=None):
    """
    Method for plotting final confusion matrixex for train, test, valid sets
    (if test, valid sets exists)
    Save plot to file and return it absolute path

    :param classifier: trained model fold (or CV model)
    :param metrics: confusion matrix metrics for train, test, validation sets
    :param fold_name: number of fold or 'mean'
    :return: path to confusion matrix plot
    :type metrics: dict
    :type fold_name: str
    :rtype: str
    """

    # make confusion matrix plot
    plt = plot_heat_maps_cm(
        metrics['train'], model_name=classifier.model_name,
        confusion_matrix_test=metrics['test'],
        confusion_matrix_validation=metrics['validation'],
        fold_number=fold_name
    )
    plt.tight_layout()

    # make plot filename
    if fold_name:
        fold_name = 'fold_{}_'.format(fold_name)
    img_file_name = '{}_{}confusion.png'.format(
        '_'.join(classifier.model_name.split()), fold_name)
    # make plot filepath
    img_path = os.path.abspath(
        os.path.join(classifier.sub_folder, img_file_name)
    )

    # save plot
    plt.savefig(img_path)
    plt.close()

    return img_path


def plot_multi_cm_final(
        classifier, reg=False, cut_off=np.log(770), dnn=True,
        cv=False, cl_tr=None, cl_ts=None, y_val=None, cl_val=None,
        fold_number=None
):
    """
     Method for plotting final confusion matrix

    :param classifier:
    :param reg:
    :param cut_off:
    :param batch_size:
    :param dnn:
    :param cv:
    :param cl_tr:
    :param cl_ts:
    :param x_val:
    :param y_val:
    :param cl_val:
    :return:
    """

    # TODO fill docstring

    from learner.models import onehot_decoded

    y_val = classifier.y_valid

    if dnn:
        if cv:
            y_pred_tr = cl_tr
            y_pred_ts = cl_ts
            y_pred_val = cl_val
        else:
            y_pred_tr = classifier.predict_probas['train']
            if len(classifier.x_test) > 0:
                y_pred_ts = classifier.predict_probas['test']
            if len(y_val) > 0:
                y_pred_val = classifier.predict_probas['validation']

    elif reg:
        if len(classifier.x_test) > 0:
            y_pred_ts = [(1 if x <= cut_off else 0) for x in classifier.predict_probas['test']]
            print(y_pred_ts)

        if len(y_val) > 0:
            y_pred_val = [(1 if x <= cut_off else 0) for x in classifier.predict_probas['validation']]
            print(y_pred_val)

        y_pred_tr = [(1 if x <= cut_off else 0) for x in classifier.predict_probas['train']]
        print(y_pred_tr)

    else:
        y_pred_tr = classifier.predict_probas['train']
        if len(classifier.x_test) > 0:
            y_pred_ts = classifier.predict_probas['test']
        if len(y_val) > 0:
            y_pred_val = classifier.predict_probas['validation']

    if len(classifier.x_test) > 0:
        cm_ts = confusion_matrix(
            onehot_decoded(classifier.y_test), onehot_decoded(y_pred_ts))
    else:
        cm_ts = None

    if len(y_val) > 0:
        cm_val = confusion_matrix(
            onehot_decoded(y_val), onehot_decoded(y_pred_val))
    else:
        cm_val = None

    cm_tr = confusion_matrix(
        onehot_decoded(classifier.y_train), onehot_decoded(y_pred_tr))
    model_name = '_'.join(classifier.model_name.split())
    plt = plot_heat_maps_cm(
        cm_tr, model_name=classifier.model_name,
        confusion_matrix_test=cm_ts, confusion_matrix_validation=cm_val,
        fold_number=fold_number
    )
    plt.tight_layout()

    fold_name = None
    if fold_number:
        fold_name = 'fold_{}_'.format(fold_number)

    img_file_name = '{}_{}confusion.png'.format(model_name, fold_name)
    img_path = os.path.abspath(
        os.path.join(classifier.sub_folder, img_file_name)
    )
    plt.savefig(img_path)
    plt.close()

    return img_path


def rotate(origin, point, angle):
    """
    Rotate a point counterclockwise by a given angle around a given origin.
    The angle should be given in radians.

    :param origin:
    :param point: point coordinates in?
    :param angle: rotating angle in radians
    :type point: list?
    :return:
    """
    # TODO check docstring
    ox, oy = origin
    px, py = point

    qx = ox + math.cos(angle) * (px - ox) - math.sin(angle) * (py - oy)
    qy = oy + math.sin(angle) * (px - ox) + math.cos(angle) * (py - oy)

    return qx, qy


def scale_back(x):
    """
    Method for convert 0, 1 to -1, 1

    :param x:
    :type x: int?
    :return:
    """
    # TODO checkdocstring

    old_range = (1 - 0)
    new_range = (1 - -1)

    return round((((x - 0) * new_range) / old_range) + -1, 1)


def radar_plot(path_to_csv, subfolder, nbits, titlename=None):
    """
    Makes a radar plot from information in scv file
    df have index labels as test or training set,
    first column labeling metric for the values
    in that row, with the columns being the algorithm.

    :param path_to_csv: path to csv file with metrics
    :param subfolder: folder to write graphs
    :param nbits: count of bits
    :param titlename: title name for radar plot
    :return: radar plot path
    """

    # Make dataframe
    gl = glob.glob(path_to_csv)[0]
    df = pandas.read_csv(gl, index_col=0)

    # Rename df columns
    model_name = df.columns[0]
    df.columns = df.columns.map(
        lambda x: re.sub(r'layers', r' layers,', str(x)))
    df.columns = df.columns.map(
        lambda x: re.sub(r'(\w)([A-Z])', r'\1 \2', str(x)))
    df.columns = df.columns.map(lambda x: re.sub(r'_', r'', str(x)))
    df.columns = df.columns.map(lambda x: re.sub(r'D NN', r'DNN', str(x)))
    df.columns = df.columns.map(lambda x: re.sub(r'DN N', r'DNN', str(x)))

    metric_converter = {'Cohen_Kappa': 'Cohen\'s Kappa',
                        'Matthews_corr': 'MCC',
                        'f1-score': 'F1-Score'}

    folds_metrics = []
    for metric in df[model_name]:
        if metric_converter.get(metric, None):
            folds_metrics.append(metric_converter[metric])
        else:
            folds_metrics.append(metric)

    df[model_name] = folds_metrics

    plt.style.use('seaborn-bright')

    train = df[df.index == 'Train']
    test = df[df.index == 'Test']
    valid = df[df.index == 'Validation']

    list_of_datasets = [train]
    train_index = 0
    test_index = None
    valid_index = None
    if len(valid) != 0:
        list_of_datasets.append(valid)
        valid_index = 1

    if len(test) != 0:
        list_of_datasets.append(test)
        test_index = 2

    styles = [
        ['r', '-'], ['g', '-'], ['b', '-'], ['y', '-'], ['c', '-'], ['m', '-'],
        ['k', '-'], ['r', '--'], ['g', '-.'], ['b', ':'], ['y', '-.'],
        ['c', ':'], ['m', '--'], ['k', '-.']
    ]

    model_nums = dict(zip(df.columns.values[1:], styles))

    hep_points = [
        [0, 1],
        [math.cos((3 * math.pi / 14)), math.sin((3 * math.pi / 14))],
        [math.cos((math.pi / 14)), -math.sin((math.pi / 14))],
        [math.cos((5 * math.pi / 14)), -math.sin((5 * math.pi / 14))],
        [-math.cos((5 * math.pi / 14)), -math.sin((5 * math.pi / 14))],
        [-math.cos((math.pi / 14)), -math.sin((math.pi / 14))],
        [-math.cos((3 * math.pi / 14)), math.sin((3 * math.pi / 14))]
    ]

    number_of_sets = len(list_of_datasets)
    fig, axarr = plt.subplots(
        1, number_of_sets, figsize=(number_of_sets*10, 10))

    for ax in axarr:

        # plot the background circles
        # the first largest having the background
        # color, the rest transparent
        circle = plt.Circle(
            (0, 0), 1, facecolor='bisque', alpha=0.2, edgecolor='grey')

        circles = [circle]
        n = 1
        for i in range(5):
            circle = plt.Circle(
                (0, 0), n, fill=None, alpha=0.6, edgecolor='black')
            n -= 0.2
            circles.append(circle)

        for circle in circles:
            ax.add_artist(circle)

        metrics = train[model_name].tolist()
        # plot the dashed lined separating the metrics
        for pt in hep_points:
            ax.plot(
                [0, pt[0]], [0, pt[1]], 'k--', lw=2, alpha=0.4)

        # plot scales and metrics for each of the heptagon slices
        for i, metric in enumerate(metrics):
            hep_x, hep_y = rotate((0, 0),
                                  (hep_points[i][0], hep_points[i][1]),
                                  (2 * math.pi / 7) / 2)

            # the right needs to be offset a little bit farther
            ax.text(
                (1.2 * hep_x if metric != 'Cohen\'s Kappa' else 1.3 * hep_x),
                (1.2 * hep_y if metric != 'Cohen\'s Kappa' else 1.3 * hep_y),
                metrics[i],
                verticalalignment='bottom',
                horizontalalignment='center', fontsize=14
            )

            for value in [0.2, 0.4, 0.6, 0.8]:
                ax.text(
                    value * hep_x, value * hep_y,
                    str(value) if metric != 'MCC' else str(scale_back(value)),
                    verticalalignment='bottom',
                    horizontalalignment='center'
                )

        ax.set_ylim([-1.25, 1.25])
        ax.set_xlim([-1.25, 1.25])
        ax.axis('off')

        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_xticks([])
        ax.set_yticks([])
        # plot the test and training data on different axes
        for i, data_set in enumerate(list_of_datasets):
            for model, data in data_set.iloc[:, 1:].iteritems():

                # scale metrics if needed
                pts = data.apply(
                    lambda x: x / 100 if float(x) > 1 else float(x)
                ).tolist()
                xs = []
                ys = []

                for j, pt in enumerate(pts):
                    x, y = rotate(
                        (0, 0),
                        (pt * hep_points[j][0], pt * hep_points[j][1]),
                        (2 * math.pi / 7) / 2
                    )
                    axarr[i].scatter(x, y, s=3, color='k', label=model)
                    xs.append(x)
                    ys.append(y)

                axarr[i].plot(
                    xs + xs[:1], ys + ys[:1],
                    color=str(model_nums.get(model)[0]),
                    ls=str(model_nums.get(model)[1]),
                    label=model, alpha=0.6
                )

    axarr[train_index].set_title('Train', fontsize=16)
    if test_index:
        axarr[test_index].set_title('Test', fontsize=16)
    if valid_index:
        axarr[valid_index].set_title('Validation', fontsize=16)

    models = data_set.iloc[:, 1:].columns.tolist()
    handles = [
        plt.plot(
            [], [], c=str(model_nums.get(model)[0]),
            ls=str(model_nums.get(model)[1])
        )[0] for model in models
    ]
    ncol = ((df.shape[1] - 1) // 3) + 1
    legend = fig.legend(
        handles, models, loc=8, prop={'size': 18},
        ncol=ncol, bbox_to_anchor=(0.4, -0.02)
    )
    frame = legend.get_frame()
    frame.set_facecolor('white')

    fig.suptitle(
        '{}\nNumber of bins = {}'.format(titlename, nbits), fontsize=18)

    radar_plot_path = os.path.abspath(
        os.path.join(subfolder, 'radar_plot.png'))
    plt.savefig(radar_plot_path, bbox_inches='tight')
    plt.close()

    return radar_plot_path


def plot_actual_predicted(regressor, ds_type='test', fold_name=None):
    """
    Function plots Actual values vs predicted values for regression models

    :param regressor: string, name of y variable
    :param ds_type:
    :param fold_name: number of fold for which making plot or 'mean'
    :return: image absolute path
    :type ds_type: str
    :type fold_name: str
    :rtype: str
    """

    if ds_type == 'test':
        y = regressor.y_test['value']
        y_predicted = regressor.predict_classes['test']
    elif ds_type == 'train':
        y = regressor.y_train['value']
        y_predicted = regressor.predict_classes['train']
    elif ds_type == 'validation':
        y = regressor.y_valid['value']
        y_predicted = regressor.predict_classes['validation']
    else:
        raise ValueError('Unknown ds_type: {}'.format(ds_type))

    r2_results = r2_score(y, y_predicted)

    plt.figure(figsize=(10, 8))
    sns.regplot(np.ravel(y_predicted), np.ravel(y), ci=False)

    model_name = regressor.model_name
    name_reg = regressor.prediction_target
    plt.legend(['{} regression'.format(model_name), name_reg], loc=2)
    plt.xlabel('Predicted values of  {}'.format(name_reg))
    plt.ylabel('Actual values of {}'.format(name_reg))
    plot_title = 'Predicted vs Actual values for {} dataset, fold {}\n'.format(
        ds_type, fold_name)
    plot_title += 'coefficient of determination R^2 = {:.4f}'.format(
        r2_results)
    plt.title(plot_title)

    # make image filename
    if fold_name:
        fold_name = 'fold_{}_'.format(fold_name)
    img_file_name = '{}_{}_{}_{}regression_plot.png'.format(
        model_name, name_reg, ds_type, fold_name)
    # make image filepath
    img_path = os.path.join(regressor.sub_folder, img_file_name)

    # save image
    plt.savefig(img_path)
    plt.close()

    return img_path


def plot_reg_results_from_df(
        y_pred, sub_folder, model_name, name_reg, y_reg, ds_type='test'
):
    """
    Function plots Actual values vs predicted values for regression models

    :param y_pred:
    :param sub_folder:
    :param model_name: name of the model
    :param name_reg: name of y variable
    :param y_reg: pre-processed dataseries, dataframes
        or numpy arrays suitable for this model
    :param ds_type:
    :type model_name: str
    :type name_reg: str
    :return: {'R2_squared' : r2_results},
        where r2_results is a coefficient of determination R^2
    :rtype: dict
    """
    # TODO check docstring

    r2_results = r2_score(y_reg, y_pred)
    plt.figure(figsize=(10, 8))
    # need to swap X and Y and use numpy.squeeze to fix problem
    sns.regplot(
        np.ravel(np.array(y_pred)), np.ravel(np.array(y_reg)), ci=False)
    plt.legend(['{} regression'.format(model_name), name_reg], loc=2)
    plt.xlabel('Predicted values of  {}'.format(name_reg))
    plt.ylabel('Actual values of {}'.format(name_reg))
    plot_title = 'Predicted vs Actual values for {} dataset\n'.format(ds_type)
    plot_title += ' coefficient of determination R^2 = {:.4f}'.format(
        r2_results)
    plt.title(plot_title)
    plt.savefig(os.path.join(
        sub_folder, '{}_{}_reg_plot.png'.format(model_name, name_reg)))
    img_path = os.path.abspath(os.path.join(
        sub_folder, '{}_reg_plot.png'.format(model_name)))
    plt.close()

    return img_path


def distribution_plot(regressor, model_name=''):
    """
    Method which make distribution plot for regression model,
    add plot path to regressor and save to file

    :param regressor: regressor trainer object
    :param model_name: name of model for which make distribution plot
    :type model_name: str
    :return: path to distribution plot file
    """

    # make path to new distribution plot
    path_to_distribution_plot = os.path.join(
        regressor.sub_folder, '{}_train_test_distribution.png'.format(
            model_name
        )
    )
    # make plot
    plt.figure(figsize=(8, 8))
    plt.hist(
        regressor.y_train['value'], bins=20, label='Train',
        color=TRAIN_COLOR
    )
    if regressor.test_size != 0:
        plt.hist(
            regressor.y_test['value'], bins=20, label='Test',
            color=TEST_COLOR
        )
    plt.legend()
    # add title to plot
    plt.title('\nHistogram of {} values'.format(regressor.prediction_target))
    # add axis labels
    plt.xlabel(regressor.prediction_target)
    plt.ylabel('Frequency')
    # add plot path to regressor
    regressor.template_tags['distribution_plots'].append(
        path_to_distribution_plot)
    # save plot to file
    plt.savefig(path_to_distribution_plot)
    plt.close()

    return path_to_distribution_plot


def plot_regression_thumbnail(regressor):
    """
    Method to make thumbnail image for trained regression model based on mean
    metrics
    Returns absolute path to generated image

    :param regressor: trained regression model object
    :return: thumbnail image full path
    :rtype: str
    """

    # bottom numbers and text size
    text_font_size = 45

    y_test = regressor.y_test['value'][:, 0]
    y_train = regressor.y_train['value'][:, 0]
    model_name = regressor.model_name
    name_reg = regressor.prediction_target

    # set figure size 1200x1200
    figure = plt.figure(figsize=(12, 12))

    # make test and train plots
    # get predicted values
    y_predicted_test = regressor.predict_classes['test']
    y_predicted_train = regressor.predict_classes['train']
    # make plots
    sns.regplot(
        y_predicted_train, y_train, ci=False, color=TRAIN_COLOR)
    sns.regplot(
        y_predicted_test, y_test, ci=False, color=TEST_COLOR)
    # calculate R2 values
    r2_results_test = r2_score(y_test, y_predicted_test)
    r2_results_train = r2_score(y_train, y_predicted_train)
    # add texts to picture
    figure.text(
        0.95, 0.21, 'R2 score', verticalalignment='bottom',
        horizontalalignment='right', fontsize=text_font_size
    )
    figure.text(
        0.65, 0.11, 'Train', verticalalignment='bottom', color=TRAIN_COLOR,
        horizontalalignment='right', fontsize=text_font_size
    )
    figure.text(
        0.95, 0.11, 'Test', verticalalignment='bottom', color=TEST_COLOR,
        horizontalalignment='right', fontsize=text_font_size
    )
    figure.text(
        0.65, 0.01, '{:.02f}'.format(r2_results_train), color=TRAIN_COLOR,
        verticalalignment='bottom', horizontalalignment='right',
        fontsize=text_font_size
    )
    figure.text(
        0.95, 0.01, '{:.02f}'.format(r2_results_test), color=TEST_COLOR,
        verticalalignment='bottom', horizontalalignment='right',
        fontsize=text_font_size
    )
    # fix figure formatting
    frame = plt.gca()
    frame.axes.xaxis.set_visible(False)
    frame.axes.yaxis.set_visible(False)
    figure.patch.set_visible(False)
    plt.box(on=None)
    # save picture
    img_path = os.path.join(
        regressor.sub_folder,
        ('{}_{}_{}.jpg'.format(model_name, name_reg, THUMBNAIL_IMAGE))
    )
    plt.savefig(img_path)
    plt.close()

    return img_path


def plot_classification_thumbnail(classifier, lw=2):
    """
    Method to make thumbnail image for trained classifier model based on mean
    metrics
    Returns absolute path to generated image

    :param classifier: trained classifier model object
    :return: thumbnail image full path
    :rtype: str
    """

    text_font_size = 45
    model_name = classifier.model_name
    # plot two ROC: train, test and train only
    figure = plt.figure(figsize=(12, 12))

    metrics = classifier.metrics['mean']
    # Train and test ROC
    # plot 50%
    plt.plot([0, 1], [0, 1], linestyle='--', lw=lw, color='k')
    # plot train ROC
    fpr_tr, tpr_tr, thresholds_tr = roc_curve(
        classifier.y_train['value'], classifier.predict_probas['train'])
    roc_auc_train = float(metrics[('train', 'AUC')])
    plt.plot(fpr_tr, tpr_tr, lw=lw, color=TRAIN_COLOR)

    # add texts to picture
    figure.text(
        0.95, 0.21, 'ROC AUC', fontsize=text_font_size,
        verticalalignment='bottom', horizontalalignment='right',
    )
    figure.text(
        0.65, 0.11, 'Train', fontsize=text_font_size, color=TRAIN_COLOR,
        verticalalignment='bottom', horizontalalignment='right',
    )
    figure.text(
        0.65, 0.01, '{:.02f}'.format(roc_auc_train), fontsize=text_font_size,
        verticalalignment='bottom', horizontalalignment='right',
        color=TRAIN_COLOR,
    )

    # plot test ROC
    if classifier.x_test.shape[0] > 0:
        fpr_ts, tpr_ts, thresholds_ts = roc_curve(
            classifier.y_test['value'], classifier.predict_probas['test'])
        roc_auc_test = float(metrics[('test', 'AUC')])
        plt.plot(fpr_ts, tpr_ts, lw=lw + 1, color=TEST_COLOR)
        figure.text(
            0.95, 0.11, 'Test', verticalalignment='bottom', color=TEST_COLOR,
            horizontalalignment='right', fontsize=text_font_size
        )
        figure.text(
            0.95, 0.01, '{:.02f}'.format(roc_auc_test), color=TEST_COLOR,
            verticalalignment='bottom', horizontalalignment='right',
            fontsize=text_font_size
        )

    if classifier.y_valid.shape[0] > 0:
        fpr_val, tpr_val, thresholds_val = roc_curve(
            classifier.y_valid['value'], classifier.predict_probas['validation'])
        roc_auc_valid = float(metrics[('validation', 'AUC')])
        plt.plot(fpr_val, tpr_val, lw=lw + 1, color=VALIDATION_COLOR)
        figure.text(
            0.35, 0.11, 'Valid', verticalalignment='bottom',
            horizontalalignment='right', fontsize=text_font_size,
            color=VALIDATION_COLOR
        )
        figure.text(
            0.35, 0.01, '{:.02f}'.format(roc_auc_valid),
            verticalalignment='bottom', horizontalalignment='right',
            color=VALIDATION_COLOR, fontsize=text_font_size
        )

    # fix plot formatting
    plt.xlim([-0.01, 1.01])
    plt.ylim([-0.01, 1.01])
    frame = plt.gca()
    frame.axes.xaxis.set_visible(False)
    frame.axes.yaxis.set_visible(False)
    plt.tight_layout()
    plt.box(on=None)
    # save picture
    model_name = '_'.join(model_name.split())
    image_file_name = '{}_{}.jpg'.format(model_name, THUMBNAIL_IMAGE)
    img_path = os.path.abspath(
        os.path.join(classifier.sub_folder, image_file_name))
    plt.savefig(img_path)
    plt.close()

    return img_path
