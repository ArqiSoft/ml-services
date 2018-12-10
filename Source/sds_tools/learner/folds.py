"""
Module contain classes and methods to simplify folds calculations
(classification, regression, classic, dnn, multi, cv etc)
Each fold can train model, calculate merics, plot graphs.
This classes used in trainers with CV folds
CV fold can calculate mean metrics, plot mean graphs
"""
import json
import zipfile

import numpy
from sklearn.metrics import (
    cohen_kappa_score, precision_score, recall_score,
    roc_curve, auc, f1_score, accuracy_score, r2_score, matthews_corrcoef,
    mean_squared_error, mean_absolute_error, confusion_matrix
)

from learner.models import onehot_encoded, onehot_decoded
from learner.plotters import (
    roc_plot, plot_cm_final, plot_classification_thumbnail, multi_roc_plot,
    plot_multi_cm_final, plot_actual_predicted, plot_regression_thumbnail
)
from general_helper import NUMPY_PROCESSOR_DTYPES


class BaseCVFolds:
    def __init__(self, sub_folder=None):
        self.models = []
        self.model_name = ''
        self.path_to_model_file = ''
        self.training_time = 0
        self.path_to_model = ''
        self.sub_folder = sub_folder
        self.model_type = ''
        self.model = None
        self.x_train = numpy.ndarray([])
        self.x_test = numpy.ndarray([])
        self.x_valid = None
        self.y_train = numpy.ndarray([])
        self.y_test = numpy.ndarray([])
        self.y_valid = None
        self.predict_classes = dict()
        self.predict_probas = dict()
        self.metrics = dict()
        self.prediction_target = ''
        # single fold class definition
        # set it in child class carefully
        self.fold_class = BaseFold

    def calculate_metrics(self):
        """
        Method to calculate general metrics for each fold in CV trained model
        Metrics calculations depend on model type (classifier or regressor) so
        method can be changed in child classes

        :return: calculated metrics
        :rtype: dict
        """

        # arrays initialization
        number_of_models = len(self.models)
        train_set_size = self.x_train.shape[0]
        test_set_size = self.x_test.shape[0]
        tmp_predict_classes_train = numpy.ndarray(
            (train_set_size, number_of_models))
        tmp_predict_classes_test = numpy.ndarray(
            (test_set_size, number_of_models))
        tmp_predict_classes_valid = numpy.array([])
        tmp_y_valid = numpy.array(
            [],
            dtype=NUMPY_PROCESSOR_DTYPES
        )
        metrics = dict()

        # all model folds loop
        for number, model in enumerate(self.models):
            # calculate fold metrics
            fold_name = str(number + 1)
            metrics[fold_name] = model.calculate_metrics(fold_name)

            # calculate mean metrics
            self.model = model.model
            predict_classes = calculate_predicted_values(self)

            # predicted values
            tmp_predict_classes_train[:, number] = predict_classes['train']
            if predict_classes['test'].shape[0] > 0:
                tmp_predict_classes_test[:, number] = predict_classes['test']

            tmp_predict_classes_valid = numpy.append(
                tmp_predict_classes_valid, model.predict_classes['validation'])
            tmp_y_valid = numpy.append(tmp_y_valid, model.y_valid)

        # calculate mean predicted values
        # used later to calculate mean metrics
        self.predict_classes = {
            'train': tmp_predict_classes_train.mean(axis=1),
            'test': tmp_predict_classes_test.mean(axis=1),
            'validation': tmp_predict_classes_valid
        }

        self.y_valid = tmp_y_valid

        return metrics

    def make_plots(self):
        """
        Method fo make plots for all folds and mean
        Return generated plots paths as dict

        :return: plots paths
        :rtype: dict
        """

        plots = dict()
        for number, model in enumerate(self.models):
            fold_name = str(number + 1)
            plots[fold_name] = model.make_plots(fold_name)

        plots['mean'] = self.make_mean_plots()

        return plots

    def compress_models(self):
        """
        Method for compress all folds models to one archive
        Return path to archive

        :return: path to models archive
        :rtype: str
        """

        # set path to archive
        path_to_archive = '{}/{}.zip'.format(
            self.sub_folder, '_'.join(self.model_name.split()))

        # add models files to archive
        all_models_archive = zipfile.ZipFile(path_to_archive, 'w')
        for model in self.models:
            model_path = model.path_to_model_file
            all_models_archive.write(
                model_path, model_path.split('/')[-1],
                compress_type=zipfile.ZIP_DEFLATED
            )

        all_models_archive.close()

        return path_to_archive

    def make_dataframes(self, model_trainer):
        """
        Method to set dataframes, based on trainer dataframes

        :param model_trainer: model trainer object which start folds training,
            metrics calculations, plotting etc
        """

        self.x_train = model_trainer.x_train
        self.x_test = model_trainer.x_test
        self.x_valid = model_trainer.x_train
        self.y_train = model_trainer.y_train
        self.y_test = model_trainer.y_test
        self.y_valid = model_trainer.y_train

    def calculate_mean_metrics(self):
        """
        Method to calculate mean metrics, should be defined in child class
        depends on model type (classifier or regressor)
        """
        pass

    def make_mean_plots(self):
        """
        Method to make mean plots, should be defined in child class
        depends on model type (classifier or regressor)
        """
        pass


class ClassicClassifierCVFolds(BaseCVFolds):
    def __init__(self, sub_folder=None):
        super().__init__(sub_folder=sub_folder)
        self.fold_class = ClassicClassifierFold

    def calculate_metrics(self):
        """
        Classifier method to calculate metrics.
        Add predict probas calculations which needed to classification metrics

        :return: calculated metrics
        :rtype: dict
        """

        # calculate base metrics
        self.metrics = super().calculate_metrics()

        # define classifier arrays
        number_of_models = len(self.models)
        train_set_size = self.x_train.shape[0]
        test_set_size = self.x_test.shape[0]
        tmp_predict_probas_train = numpy.ndarray(
            (train_set_size, number_of_models))
        tmp_predict_probas_test = numpy.ndarray(
            (test_set_size, number_of_models))
        tmp_predict_probas_valid = numpy.array([])

        # models folds loop
        for number, model in enumerate(self.models):
            self.model = model.model
            predict_probas = make_predict_proba(self)

            tmp_predict_probas_train[:, number] = predict_probas['train']
            if predict_probas['test'] is not None:
                tmp_predict_probas_test[:, number] = predict_probas['test']
            tmp_predict_probas_valid = numpy.append(
                tmp_predict_probas_valid, model.predict_probas['validation'])

        # calculate and add mean classifier metrics
        self.predict_probas = {
            'train': tmp_predict_probas_train.mean(axis=1),
            'test': tmp_predict_probas_test.mean(axis=1),
            'validation': tmp_predict_probas_valid
        }
        self.metrics['mean'] = self.calculate_mean_metrics()

        return self.metrics

    def calculate_mean_metrics(self):
        """
        Method for calculate mean folds metrics
        Return mean metrics as dict

        :return: mean metrics
        :rtype: dict
        """

        # there is not bool values in brackets!
        self.predict_classes['train'] = (self.predict_classes['train'] > 0.5).astype(int)
        self.predict_classes['test'] = (self.predict_classes['test'] > 0.5).astype(int)
        self.predict_classes['validation'] = (self.predict_classes['validation'] > 0.5).astype(int)
        mean_metrics = calculate_classification_metrics(self)

        return mean_metrics

    def make_mean_plots(self):
        """
        Method for make mean plots
        Return mean plots paths as dict

        :return: path to mean plots
        :rtype: dict
        """

        plots = make_classifier_plots(self, 'mean')

        return plots


class ClassicRegressorCVFolds(BaseCVFolds):
    def __init__(self, sub_folder=None):
        super().__init__(sub_folder=sub_folder)
        self.fold_class = ClassicRegressorFold

    def calculate_metrics(self):
        """
        Regressor method to calculate metrics.
        Mean metrics calculation added

        :return: calculated metrics
        :rtype: dict
        """

        # calculate base metric
        self.metrics = super().calculate_metrics()
        # calculate regressor mean metrics
        self.metrics['mean'] = self.calculate_mean_metrics()

        return self.metrics

    def calculate_mean_metrics(self):
        """
        Method for calculate mean folds metrics
        Return mean metrics as dict

        :return: mean metrics
        :rtype: dict
        """

        mean_metrics = dict()
        mean_metrics.update(calculate_regression_train(self))
        if self.y_test.shape[0] != 0:
            mean_metrics.update(calculate_regression_test(self))
        if self.y_valid.shape[0] != 0:
            mean_metrics.update(calculate_regression_valid(self))

        return mean_metrics

    def make_mean_plots(self):
        """
        Method for make mean plots
        Return mean plots paths as dict

        :return: path to mean plots
        :rtype: dict
        """

        plots = make_regression_plots(self, 'mean')

        return plots


class ClassicMultiRegressorCVFolds(BaseCVFolds):
    def __init__(self, sub_folder=None):
        super().__init__(sub_folder=sub_folder)
        self.fold_class = ClassicMultiregressorFold

    def calculate_metrics(self):
        """
        Method for calculate all training metrcis for all folds and mean metric
        Return training metrics as dict

        :return: training metrics
        :rtype: dict
        """

        tmp_predict_classes_train = []
        tmp_predict_classes_test = []
        tmp_predict_classes_valid = []
        tmp_y_valid = []

        metrics = dict()
        for number, model in enumerate(self.models):
            # calculate fold metrics
            fold_name = str(number + 1)
            metrics[fold_name] = model.calculate_metrics(fold_name)

            # calculate mean metrics
            self.model = model.model
            predict_classes = calculate_predicted_values(self)

            tmp_predict_classes_train.append(
                pandas.DataFrame(predict_classes['train']))
            tmp_predict_classes_test.append(
                pandas.DataFrame(predict_classes['test']))

            tmp_predict_classes_valid.append(
                pandas.DataFrame(model.predict_classes['validation']))
            tmp_y_valid.append(pandas.DataFrame(model.y_valid))

        self.predict_classes = {
            'train': pandas.concat(
                tmp_predict_classes_train).groupby(level=0).mean().as_matrix(),
            'test': pandas.concat(
                tmp_predict_classes_test).groupby(level=0).mean().as_matrix(),
            'validation': pandas.concat(tmp_predict_classes_valid).as_matrix(),
        }

        self.y_train = pandas.DataFrame(self.y_train)
        self.y_test = pandas.DataFrame(self.y_test)
        self.y_valid = pandas.concat(tmp_y_valid)

        metrics['mean'] = self.calculate_multi_mean_metrics()
        self.metrics = metrics

        return metrics

    def calculate_multi_mean_metrics(self):

        mean_metrics = dict()
        for n_task in range(self.y_train.shape[1]):
            mean_metrics.update(calculate_multi_rmse(self, n_task))
            mean_metrics.update(calculate_multi_r2(self, n_task))
            mean_metrics.update(calculate_multi_mae(self, n_task))
        # metrics.update(calculate_metrics_cv(self))

        return mean_metrics

    def make_mean_plots(self):
        """
        Method for make mean plots
        Return mean plots paths as dict

        :return: path to mean plots
        :rtype: dict
        """

        plots = make_regression_plots(self, 'mean')

        return plots


class ClassicMultiClassifierCVFolds(BaseCVFolds):
    def __init__(self, sub_folder=None):
        super().__init__(sub_folder=sub_folder)
        self.fold_class = ClassicMultiClassifierFold

    def calculate_metrics(self):
        """
        Method for calculate all training metrcis for all folds and mean metric
        Return training metrics as dict

        :return: training metrics
        :rtype: dict
        """

        tmp_predict_classes_train = []
        tmp_predict_classes_test = []
        tmp_predict_classes_valid = []
        tmp_y_valid = []

        metrics = dict()
        for number, model in enumerate(self.models):
            # calculate fold metrics
            fold_name = str(number + 1)
            metrics[fold_name] = model.calculate_metrics(fold_name)

            # calculate mean metrics
            self.model = model.model
            predict_classes = calculate_predicted_values(self)

            tmp_predict_classes_train.append(
                pandas.DataFrame(predict_classes['train']))
            tmp_predict_classes_test.append(
                pandas.DataFrame(predict_classes['test']))

            tmp_predict_classes_valid.append(
                pandas.DataFrame(model.predict_classes['validation']))
            tmp_y_valid.append(pandas.DataFrame(model.y_valid))

        self.predict_classes = {
            'train': pandas.concat(
                tmp_predict_classes_train).groupby(level=0).mean().as_matrix(),
            'test': pandas.concat(
                tmp_predict_classes_test).groupby(level=0).mean().as_matrix(),
            'validation': pandas.concat(tmp_predict_classes_valid).as_matrix(),
        }

        self.y_train = onehot_encoded(self.y_train)
        self.y_test = onehot_encoded(self.y_test)
        self.y_valid = pandas.concat(tmp_y_valid)

        metrics['mean'] = calculate_multi_classification_metrics(self)
        self.metrics = metrics

        return metrics

    def make_mean_plots(self):
        """
        Method for make mean plots
        Return mean plots paths as dict

        :return: path to mean plots
        :rtype: dict
        """

        plots = make_multi_classifier_plots(self, 'mean')

        return plots


class DNNMultiRegressorCVFolds(BaseCVFolds):
    def __init__(self, sub_folder=None):
        super().__init__(sub_folder=sub_folder)
        self.fold_class = DNNMultiregressorFold

    def calculate_metrics(self):
        """
        Method for calculate all training metrcis for all folds and mean metric
        Return training metrics as dict

        :return: training metrics
        :rtype: dict
        """

        tmp_predict_classes_train = []
        tmp_predict_classes_test = []
        tmp_predict_classes_valid = []
        tmp_y_valid = []

        metrics = dict()
        for number, model in enumerate(self.models):
            # calculate fold metrics
            fold_name = str(number + 1)
            metrics[fold_name] = model.calculate_metrics(fold_name)

            # calculate mean metrics
            self.model = model.model
            predict_classes = calculate_predicted_values(self)

            tmp_predict_classes_train.append(
                pandas.DataFrame(predict_classes['train']))
            tmp_predict_classes_test.append(
                pandas.DataFrame(predict_classes['test']))

            tmp_predict_classes_valid.append(
                pandas.DataFrame(model.predict_classes['validation']))
            tmp_y_valid.append(pandas.DataFrame(model.y_valid))

        self.predict_classes = {
            'train': pandas.concat(
                tmp_predict_classes_train).groupby(level=0).mean().as_matrix(),
            'test': pandas.concat(
                tmp_predict_classes_test).groupby(level=0).mean().as_matrix(),
            'validation': pandas.concat(tmp_predict_classes_valid).as_matrix(),
        }

        self.y_train = pandas.DataFrame(self.y_train)
        self.y_test = pandas.DataFrame(self.y_test)
        self.y_valid = pandas.concat(tmp_y_valid)

        metrics['mean'] = self.calculate_multi_mean_metrics()
        self.metrics = metrics

        return metrics

    def calculate_multi_mean_metrics(self):

        mean_metrics = dict()
        for n_task in range(self.y_train.shape[1]):
            mean_metrics.update(calculate_multi_rmse(self, n_task))
            mean_metrics.update(calculate_multi_r2(self, n_task))
            mean_metrics.update(calculate_multi_mae(self, n_task))
        # metrics.update(calculate_metrics_cv(self))

        return mean_metrics

    def make_mean_plots(self):
        """
        Method for make mean plots
        Return mean plots paths as dict

        :return: path to mean plots
        :rtype: dict
        """

        plots = make_regression_plots(self, 'mean')

        return plots

    def make_dataframes(self, model_trainer):
        self.x_train = model_trainer.x_train.as_matrix()
        self.x_test = model_trainer.x_test.as_matrix()
        self.x_valid = model_trainer.x_train.as_matrix()
        self.y_train = model_trainer.y_train.as_matrix()
        self.y_test = model_trainer.y_test.as_matrix()
        self.y_valid = model_trainer.y_train.as_matrix()


class DNNMultiClassifierCVFolds(BaseCVFolds):
    def __init__(self, sub_folder=None):
        super().__init__(sub_folder=sub_folder)
        self.fold_class = DNNMultiClassifierFold

    def calculate_metrics(self):
        number_of_models = len(self.models)
        train_set_size = self.x_train.shape[0]
        test_set_size = self.x_test.shape[0]

        tmp_y_valid = numpy.empty((0, self.y_train.shape[1]), float)

        tmp_predict_probas_train = numpy.ndarray(
            (number_of_models, train_set_size, self.y_train.shape[1]))
        tmp_predict_probas_test = numpy.ndarray(
            (number_of_models, test_set_size, self.y_train.shape[1]))
        tmp_predict_probas_valid = numpy.empty(
            (0, self.y_train.shape[1]), float)

        metrics = dict()
        for number, model in enumerate(self.models):
            # calculate fold metrics
            fold_name = str(number + 1)
            metrics[fold_name] = model.calculate_metrics(fold_name)

            # calculate mean metrics
            self.model = model.model
            # predict_classes = calculate_predicted_values(self)
            predict_probas = make_multiclass_predict_proba(self)

            # predicted values
            tmp_predict_probas_train[number] = predict_probas['train']
            if predict_probas['test'] is not None:
                tmp_predict_probas_test[number] = predict_probas['test']
            tmp_predict_probas_valid = numpy.vstack(
                (tmp_predict_probas_valid, model.predict_probas['validation']))
            tmp_y_valid = numpy.vstack((tmp_y_valid, model.y_valid))

        self.predict_probas = {
            'train': tmp_predict_probas_train.mean(axis=0),
            'test': tmp_predict_probas_test.mean(axis=0),
            'validation': tmp_predict_probas_valid
        }

        self.y_valid = tmp_y_valid
        self.metrics = metrics
        self.metrics['mean'] = self.calculate_mean_metrics()

        return metrics

    def make_mean_plots(self):
        """
        Method for make mean plots
        Return mean plots paths as dict

        :return: path to mean plots
        :rtype: dict
        """

        plots = make_multi_classifier_plots(self, 'mean')

        return plots

    def make_dataframes(self, model_trainer):
        self.x_train = model_trainer.x_train
        self.x_test = model_trainer.x_test
        self.x_valid = model_trainer.x_train
        self.y_train = onehot_encoded(model_trainer.y_train['value'])
        self.y_test = onehot_encoded(model_trainer.y_test['value'])
        self.y_valid = onehot_encoded(model_trainer.y_train['value'])


class DNNClassifierCVFolds(ClassicClassifierCVFolds):
    def __init__(self, sub_folder=None):
        super().__init__(sub_folder=sub_folder)
        self.fold_class = DNNClassifierFold


class DNNRegressorCVFolds(ClassicRegressorCVFolds):
    def __init__(self, sub_folder=None):
        super().__init__(sub_folder=sub_folder)
        self.fold_class = ClassicRegressorFold


class BaseFold:
    def __init__(
            self, model=None, model_type=None, x_train=None, x_test=None,
            x_valid=None, y_train=None, y_test=None, y_valid=None,
            sub_folder=None, prediction_target=None
    ):
        self.model = model
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        self.x_valid = x_valid
        self.y_valid = y_valid
        self.path_to_model_file = ''
        self.training_time = 0
        self.model_name = ''
        self.model_type = model_type
        self.metrics = dict()
        self.plots = list()
        self.predict_classes = dict()
        self.predict_probas = dict()
        self.sub_folder = sub_folder
        self.prediction_target = prediction_target

    def calculate_metrics(self, fold_name):
        """
        Method to calculate single fold metrics
        Define it in child class depends on trained model type

        :param fold_name: current fold number
        :type fold_name: str
        """
        pass

    def make_plots(self, fold_name):
        """
        Method to plot single fold graphs
        Define it in child class depends on trained model type

        :param fold_name: current fold number
        :type fold_name: str
        """
        pass


class ClassicClassifierFold(BaseFold):
    def __init__(
            self, model=None, model_type=None, x_train=None, x_test=None,
            x_valid=None, y_train=None, y_test=None, y_valid=None,
            sub_folder=None, prediction_target=None
    ):
        super().__init__(
            model=model, model_type=model_type, x_train=x_train, x_test=x_test,
            x_valid=x_valid, y_train=y_train, y_test=y_test, y_valid=y_valid,
            sub_folder=sub_folder, prediction_target=prediction_target
        )

    def calculate_metrics(self, fold_name):
        """
        Method to calculate single classifier fold metrics
        Update metrics attribute with metrics (as dict)

        :param fold_name: current fold number
        :type fold_name: str
        :return: fold metrics
        :rtype: dict
        """

        # calculate predicted values for fold
        self.predict_classes = calculate_predicted_values(self)
        # additionally calculate predict probas for classifier
        self.predict_probas = make_predict_proba(self)

        # check metrics usage
        # for example we can not using test metrics, so need to catch that case
        for key, value in self.predict_probas.items():
            if value is not None:
                tmp_value = value
            else:
                tmp_value = None

            self.predict_probas[key] = tmp_value

        # calculate metrics
        metrics = calculate_classification_metrics(self)
        self.metrics[fold_name] = metrics

        return metrics

    def make_plots(self, fold_name):
        """
        Method to make single classifier fold plots
        Update plots attribute with plots paths (as dict)

        :param fold_name: current fold number
        :type fold_name: str
        :return: paths to single classifier fold plots
        :rtype: dict
        """

        plots = make_classifier_plots(self, fold_name)
        self.plots = plots

        return plots


class ClassicRegressorFold(BaseFold):
    def __init__(
            self, model=None, model_type=None, x_train=None, x_test=None,
            x_valid=None, y_train=None, y_test=None, y_valid=None,
            sub_folder=None, prediction_target=None
    ):
        super().__init__(
            model=model, model_type=model_type, x_train=x_train, x_test=x_test,
            x_valid=x_valid, y_train=y_train, y_test=y_test, y_valid=y_valid,
            sub_folder=sub_folder, prediction_target=prediction_target
        )

    def calculate_metrics(self, fold_name):
        """
        Method to calculate single regressor fold metrics
        Update metrics attribute with metrics (as dict)

        :param fold_name: current fold number
        :type fold_name: str
        :return: fold metrics
        :rtype: dict
        """

        # calculate predicted values for fold
        self.predict_classes = calculate_predicted_values(self)
        # calculate metrics for fold
        metrics = regression_metrics(self)
        self.metrics[fold_name] = metrics

        return metrics

    def make_plots(self, fold_name):
        """
        Method to make single regressor fold plots
        Update plots attribute with plots paths (as dict)

        :param fold_name: current fold number or 'mean'
        :type fold_name: str
        :return: paths to single regressor fold plots
        :rtype: dict
        """

        plots = make_regression_plots(self, fold_name)
        self.plots = plots

        return plots


class DNNClassifierFold(ClassicClassifierFold):
    def __init__(
            self, model=None, model_type=None, x_train=None, x_test=None,
            x_valid=None, y_train=None, y_test=None, y_valid=None,
            sub_folder=None, prediction_target=None
    ):
        super().__init__(
            model=model, model_type=model_type, x_train=x_train, x_test=x_test,
            x_valid=x_valid, y_train=y_train, y_test=y_test, y_valid=y_valid,
            sub_folder=sub_folder, prediction_target=prediction_target
        )

    def calculate_metrics(self, fold_name):
        """
        Method to calculate single DNN classifier fold metrics
        Update metrics attribute with metrics (as dict)

        :param fold_name: current fold number
        :type fold_name: str
        :return: fold metrics
        :rtype: dict
        """

        # calculate predicted values for fold
        self.predict_classes = calculate_predicted_values(self)
        # additionally calculate predict probas for classifier
        self.predict_probas = make_predict_proba(self)

        # normalize values, should be 0 or 1 for classification
        self.predict_classes['train'] = (
            self.predict_classes['train'] > 0.5).astype(int).reshape(-1)
        self.predict_classes['test'] = (
            self.predict_classes['test'] > 0.5).astype(int).reshape(-1)
        self.predict_classes['validation'] = (
            self.predict_classes['validation'] > 0.5).astype(int).reshape(-1)

        # check metrics usage
        # for example we can not using test metrics, so need to catch that case
        for key, value in self.predict_probas.items():
            if value is not None:
                tmp_value = value
            else:
                tmp_value = None

            self.predict_probas[key] = tmp_value

        # calculate metrics
        metrics = calculate_classification_metrics(self)
        self.metrics[fold_name] = metrics

        return metrics


class DNNMultiregressorFold(BaseFold):
    def calculate_metrics(self, fold_name):
        """
        Method for calculate all training metrcis for current fold for DNN
        Return training metrics as dict

        :param fold_name: number of current fold
        :return: training metrics
        :rtype: dict
        """

        # transform datasets to being able DNN prediction
        self.x_train = self.x_train.as_matrix()
        self.x_valid = self.x_valid.as_matrix()
        self.x_test = self.x_test.as_matrix()

        self.predict_classes = calculate_predicted_values(self)

        metrics = milti_regression_metrics(self)

        self.metrics[fold_name] = metrics

        return metrics


class DNNMultiClassifierFold(BaseFold):
    def __init__(
            self, model=None, model_type=None, x_train=None, x_test=None,
            x_valid=None, y_train=None, y_test=None, y_valid=None,
            sub_folder=None, prediction_target=None
    ):
        super().__init__(
            model=model, model_type=model_type, x_train=x_train, x_test=x_test,
            x_valid=x_valid, y_train=y_train, y_test=y_test, y_valid=y_valid,
            sub_folder=sub_folder, prediction_target=prediction_target
        )

    def calculate_metrics(self, fold_name):
        # calculate predicted values for fold
        self.predict_classes = calculate_multiclass_predicted_values(self)
        # additionally calculate predict probas for classifier
        self.predict_probas = make_multiclass_predict_proba(self)

        for key, value in self.predict_probas.items():
            if value is not None:
                tmp_value = value
            else:
                tmp_value = None

            self.predict_probas[key] = tmp_value

        # calculate metrics
        metrics = calculate_multi_classification_metrics(self)
        self.metrics[fold_name] = metrics

        return metrics

    def make_plots(self, fold_name):
        """
        Method fo make plots for current fold
        Return generated plots paths as dict

        :param fold_name: number of current fold
        :return: plots paths
        :rtype: dict
        """

        plots = make_multi_classifier_plots(self, fold_name)
        self.plots = plots

        return plots


class ClassicMultiregressorFold(BaseFold):
    def calculate_metrics(self, fold_name):
        """
        Method for calculate all training metrcis for current fold for DNN
        Return training metrics as dict

        :param fold_name: number of current fold
        :return: training metrics
        :rtype: dict
        """

        # transform datasets to being able DNN prediction
        self.x_train = self.x_train.as_matrix()
        self.x_valid = self.x_valid.as_matrix()
        self.x_test = self.x_test.as_matrix()

        self.predict_classes = calculate_predicted_values(self)

        metrics = milti_regression_metrics(self)

        self.metrics[fold_name] = metrics

        return metrics


class ClassicMultiClassifierFold(BaseFold):
    def __init__(
            self, model=None, model_type=None, x_train=None, x_test=None,
            x_valid=None, y_train=None, y_test=None, y_valid=None,
            sub_folder=None, prediction_target=None
    ):
        super().__init__(
            model=model, model_type=model_type, x_train=x_train, x_test=x_test,
            x_valid=x_valid, y_train=y_train, y_test=y_test, y_valid=y_valid,
            sub_folder=sub_folder, prediction_target=prediction_target
        )

    def make_plots(self, fold_name):
        """
        Method fo make plots for current fold
        Return generated plots paths as dict

        :param fold_name: number of current fold
        :return: plots paths
        :rtype: dict
        """

        plots = make_multi_classifier_plots(self, fold_name)
        self.plots = plots

        return plots


def make_classifier_plots(classifier, fold_name):
    """
    Method for make all classification plots by using classifier attributes
    Return plots paths as dict

    :param classifier: classifying model object
    :param fold_name: plot fold number
    :return: plots paths
    :type classifier: ClassicClassifierCVFolds, DNNClassifierCVFolds,
            DNNClassifierFold, ClassicClassifierFold
    :type fold_name: str
    :rtype: dict
    """

    # make roc plot
    path_to_roc = roc_plot(classifier, fold_name=fold_name)
    # make confusion matrix plot
    metrics = confusion_matrix_metrics(classifier)
    path_to_cm = plot_cm_final(classifier, metrics, fold_name=fold_name)

    # make thumbnail plot based on mean metrics
    if fold_name == 'mean':
        thumbnail_plot_path = plot_classification_thumbnail(classifier)
    else:
        thumbnail_plot_path = None

    # add plots paths to model data dict
    return {
        'roc_plot_path': path_to_roc,
        'cm_plot_path': path_to_cm,
        'thumbnail_plot_path': thumbnail_plot_path
    }


def make_multi_classifier_plots(classifier, fold_name):
    """
    Method for make all classification plots by using classifier attributes
    Return plots paths as dict

    :param classifier: classifying model object
    :param fold_name: plot fold number
    :return: plots paths
    :type classifier: ClassicClassifierCVFolds, DNNClassifierCVFolds,
            DNNClassifierFold, ClassicClassifierFold
    :type fold_name: str
    :rtype: dict
    """

    # make roc plot
    path_to_roc = multi_roc_plot(classifier, fold_number=fold_name)
    # make confusion matrix plot
    path_to_cm = plot_multi_cm_final(
        classifier, dnn=True, fold_number=fold_name)

    # add plots paths to model data dict
    return {
        'roc_plot_path': path_to_roc,
        'cm_plot_path': path_to_cm,
        # 'thumbnail_plot_path': thumbnail_plot_path
    }


def make_regression_plots(regressor, fold_name):
    """
    Method for make all regression plots by using regressor attributes
    Return plots paths as dict

    :param regressor: regression model object
    :param fold_name: fold number or 'mean'
    :return: plots paths
    :type regressor: ClassicRegressorFold, DNNRegressorCVFolds
    :type fold_name: str
    :rtype: dict
    """

    # get plots paths
    path_to_reg_results_test = plot_actual_predicted(
        regressor, ds_type='test', fold_name=fold_name
    )
    path_to_reg_results_train = plot_actual_predicted(
        regressor, ds_type='train', fold_name=fold_name
    )
    path_to_reg_results_valid = plot_actual_predicted(
        regressor, ds_type='validation', fold_name=fold_name
    )

    # make thumbnail plot based on mean metrics
    if fold_name == 'mean':
        thumbnail_plot_path = plot_regression_thumbnail(regressor)
    else:
        thumbnail_plot_path = None

    # add plots paths to model data dict
    return {
        'regression_results_test': path_to_reg_results_test,
        'regression_results_train': path_to_reg_results_train,
        'regression_results_valid': path_to_reg_results_valid,
        'thumbnail_plot_path': thumbnail_plot_path
    }


def calculate_classification_metrics(classifier):
    """
    Method for calculate all classification metrcis using classifier attributes
    Return metrics as dict

    :param classifier: classifying model object
    :return: classification metrics
    :type classifier: ClassicClassifierCVFolds, DNNClassifierCVFolds,
            DNNClassifierFold, ClassicClassifierFold
    :rtype: dict
    """

    # train set classification metrics calculation
    accuracy_train = accuracy_score(
        classifier.y_train['value'], classifier.predict_classes['train'])
    f1_score_train = f1_score(
        classifier.y_train['value'], classifier.predict_classes['train'])
    fpr_tr, tpr_tr, thresholds_tr = roc_curve(
        classifier.y_train['value'], classifier.predict_probas['train'])
    roc_auc_train = auc(fpr_tr, tpr_tr)
    cohen_kappa_train = cohen_kappa_score(
        classifier.y_train['value'], classifier.predict_classes['train'])
    matthews_corr_train = matthews_corrcoef(
        classifier.y_train['value'], classifier.predict_classes['train'])
    precision_train = precision_score(
        classifier.y_train['value'], classifier.predict_classes['train'])
    recall_train = recall_score(
        classifier.y_train['value'], classifier.predict_classes['train'])
    # update metrics dict with train set metrics values
    metrics = {
        ('train', 'AUC'): roc_auc_train,
        ('train', 'ACC'): accuracy_train,
        ('train', 'f1-score'): f1_score_train,
        ('train', 'Cohen_Kappa'): cohen_kappa_train,
        ('train', 'Matthews_corr'): matthews_corr_train,
        ('train', 'Precision'): precision_train,
        ('train', 'Recall'): recall_train
    }

    # test set classification metrics calculation
    if classifier.x_test.shape[0] > 0:
        accuracy_test = accuracy_score(
            classifier.y_test['value'], classifier.predict_classes['test'])
        f1_score_test = f1_score(
            classifier.y_test['value'], classifier.predict_classes['test'])
        fpr_ts, tpr_ts, thresholds_ts = roc_curve(
            classifier.y_test['value'], classifier.predict_probas['test'])
        roc_auc_test = auc(fpr_ts, tpr_ts)
        cohen_kappa_test = cohen_kappa_score(
            classifier.y_test['value'], classifier.predict_classes['test'])
        matthews_corr_test = matthews_corrcoef(
            classifier.y_test['value'], classifier.predict_classes['test'])
        precision_test = precision_score(
            classifier.y_test['value'], classifier.predict_classes['test'])
        recall_test = recall_score(
            classifier.y_test['value'], classifier.predict_classes['test'])
        # update metrics dict with test set metrics values
        metrics['test', 'AUC'] = roc_auc_test
        metrics['test', 'ACC'] = accuracy_test
        metrics['test', 'f1-score'] = f1_score_test
        metrics['test', 'Cohen_Kappa'] = cohen_kappa_test
        metrics['test', 'Matthews_corr'] = matthews_corr_test
        metrics['test', 'Precision'] = precision_test
        metrics['test', 'Recall'] = recall_test

    # validation set metrics calculation
    if classifier.y_valid is not None:
        y_valid = classifier.y_valid['value']
        accuracy_validation = accuracy_score(
            y_valid, classifier.predict_classes['validation'])
        f1_score_validation = f1_score(
            y_valid, classifier.predict_classes['validation'])
        fpr_val, tpr_val, thresholds_val = roc_curve(
            y_valid, classifier.predict_probas['validation'])
        roc_auc_validation = auc(fpr_val, tpr_val)
        cohen_kappa_validation = cohen_kappa_score(
            y_valid, classifier.predict_classes['validation'])
        matthews_corr_validation = matthews_corrcoef(
            y_valid, classifier.predict_classes['validation'])
        precision_validation = precision_score(
            y_valid, classifier.predict_classes['validation'])
        recall_validation = recall_score(
            y_valid, classifier.predict_classes['validation'])
        # update metrics dict with validation set metrics values
        metrics['validation', 'AUC'] = roc_auc_validation
        metrics['validation', 'ACC'] = accuracy_validation
        metrics['validation', 'f1-score'] = f1_score_validation
        metrics['validation', 'Cohen_Kappa'] = cohen_kappa_validation
        metrics['validation', 'Matthews_corr'] = matthews_corr_validation
        metrics['validation', 'Precision'] = precision_validation
        metrics['validation', 'Recall'] = recall_validation

    return metrics


def calculate_multi_classification_metrics(classifier):
    y_train_true = onehot_decoded(classifier.y_train)
    y_train_pred = onehot_decoded(classifier.predict_probas['train'])

    accuracy_train = accuracy_score(y_train_true, y_train_pred)
    f1_score_train = f1_score(y_train_true, y_train_pred, average='weighted')
    cohen_kappa_train = cohen_kappa_score(y_train_true, y_train_pred)
    matthews_corr_train = matthews_corrcoef(y_train_true, y_train_pred)
    precision_train = precision_score(
        y_train_true, y_train_pred, average='weighted')
    recall_train = recall_score(y_train_true, y_train_pred, average='weighted')

    metrics = {
        ('train', 'ACC'): accuracy_train,
        ('train', 'f1-score'): f1_score_train,
        ('train', 'Cohen_Kappa'): cohen_kappa_train,
        ('train', 'Matthews_corr'): matthews_corr_train,
        ('train', 'Precision'): precision_train,
        ('train', 'Recall'): recall_train
    }

    # test set classification metrics calculation
    if len(classifier.x_test) > 0:
        y_test_true = onehot_decoded(classifier.y_test)
        y_test_pred = onehot_decoded(classifier.predict_probas['test'])

        accuracy_test = accuracy_score(y_test_true, y_test_pred)
        f1_score_test = f1_score(y_test_true, y_test_pred, average='weighted')
        cohen_kappa_test = cohen_kappa_score(y_test_true, y_test_pred)
        matthews_corr_test = matthews_corrcoef(y_test_true, y_test_pred)
        precision_test = precision_score(
            y_test_true, y_test_pred, average='weighted')
        recall_test = recall_score(
            y_test_true, y_test_pred, average='weighted')

        metrics['test', 'ACC'] = accuracy_test
        metrics['test', 'f1-score'] = f1_score_test
        metrics['test', 'Cohen_Kappa'] = cohen_kappa_test
        metrics['test', 'Matthews_corr'] = matthews_corr_test
        metrics['test', 'Precision'] = precision_test
        metrics['test', 'Recall'] = recall_test

    if classifier.y_valid is not None:
        y_valid_true = onehot_decoded(classifier.y_valid)
        y_valid_pred = onehot_decoded(classifier.predict_probas['validation'])

        accuracy_validation = accuracy_score(y_valid_true, y_valid_pred)
        f1_score_validation = f1_score(
            y_valid_true, y_valid_pred, average='weighted')
        cohen_kappa_validation = cohen_kappa_score(y_valid_true, y_valid_pred)
        matthews_corr_validation = matthews_corrcoef(
            y_valid_true, y_valid_pred)
        precision_validation = precision_score(
            y_valid_true, y_valid_pred, average='weighted')
        recall_validation = recall_score(
            y_valid_true, y_valid_pred, average='weighted')

        metrics['validation', 'ACC'] = accuracy_validation
        metrics['validation', 'f1-score'] = f1_score_validation
        metrics['validation', 'Cohen_Kappa'] = cohen_kappa_validation
        metrics['validation', 'Matthews_corr'] = matthews_corr_validation
        metrics['validation', 'Precision'] = precision_validation
        metrics['validation', 'Recall'] = recall_validation

    return metrics


def milti_regression_metrics(multiregressor):
    """
    Method to calculate all regression metrics.
    Return calculated metrics as dict

    :return: regression metrics
    :rtype: dict
    """

    metrics = dict()

    for n_task in range(multiregressor.y_train.shape[1]):
        metrics.update(calculate_multi_rmse(multiregressor, n_task))
        metrics.update(calculate_multi_r2(multiregressor, n_task))
        metrics.update(calculate_multi_mae(multiregressor, n_task))
    # metrics.update(calculate_metrics_cv(self))

    return metrics


def regression_metrics(regressor):
    """
    Method to calculate regressor metrics for train,
    test, validation sets (if test and validations sets exists)
    Metrics (such as R2, RMSE etc) should be defined inside
    calculate_regression_...() methods
    Return calculated metrics for each exists set

    :param regressor: trained fold (or CV model) object
    :return: calculated metrics for trained regression model
    :type regressor: DNNRegressorCVFolds, ClassicRegressorFold
    :rtype: dict
    """

    # train set always exists
    metrics = dict()
    metrics.update(calculate_regression_train(regressor))
    # if test set exists
    if regressor.y_test.shape[0] != 0:
        metrics.update(calculate_regression_test(regressor))
    # if validation set exists
    if regressor.y_valid.shape[0] != 0:
        metrics.update(calculate_regression_valid(regressor))

    return metrics


def calculate_predicted_values(model_trainer):
    """
    Method for calculate predicted values for trained model
    Using to calculate metrics, compare with true values
    Return predicted values as dict

    :param model_trainer: fold (or CV model) object
    :return: predicted values
    :type model_trainer:
            DNNRegressorCVFolds, ClassicRegressorFold,
            ClassicClassifierCVFolds, DNNClassifierCVFolds,
            DNNClassifierFold, ClassicClassifierFold
    :rtype: dict
    """

    classes_test = numpy.array([])
    classes_validation = numpy.array([])
    model = model_trainer.model

    # validation set exists
    if model_trainer.x_valid.shape[0] > 0:
        classes_validation = model.predict(model_trainer.x_valid['value'])
        # catch classic calculations case, reshape array to 1D
        if len(classes_validation.shape) > 1:
            classes_validation = classes_validation.reshape(-1)

    # test set exists
    if model_trainer.x_test.shape[0] > 0:
        classes_test = model.predict(model_trainer.x_test['value'])
        # catch classic calculations case, reshape array to 1D
        if len(classes_test.shape) > 1:
            classes_test = classes_test.reshape(-1)

    # train set should be always exists
    classes_train = model.predict(model_trainer.x_train['value'])
    # catch classic calculations case, reshape array to 1D
    if len(classes_train.shape) > 1:
        classes_train = classes_train.reshape(-1)

    # return predicted values for single fold (or CV model mean values)
    # for each set type (train, test, validation)
    return {
        'test': classes_test,
        'train': classes_train,
        'validation': classes_validation
    }


def make_multiclass_predict_proba(model_trainer):
    """
    Method for calculate predicted values for trained model
    Using to calculate metrics, compare with true values
    Return predicted values as dict

    :param model_trainer:
    :return: predicted values
    :type model_trainer: TrainedModel, TrainedModelDNN, TrainedModelCV
    :rtype: dict
    """

    probas_test = numpy.array([])
    probas_validation = numpy.array([])
    model = model_trainer.model

    if model_trainer.x_valid.shape[0] > 0:
        probas_validation = model.predict(model_trainer.x_valid['value'])
        row_sums = probas_validation.sum(axis=1)
        probas_validation = probas_validation / row_sums[:, numpy.newaxis]

    if model_trainer.x_test.shape[0] > 0:
        probas_test = model.predict(model_trainer.x_test['value'])
        row_sums = probas_test.sum(axis=1)
        probas_test = probas_test / row_sums[:, numpy.newaxis]

    probas_train = model.predict(model_trainer.x_train['value'])
    row_sums = probas_train.sum(axis=1)
    probas_train = probas_train / row_sums[:, numpy.newaxis]

    return {
        'test': probas_test,
        'train': probas_train,
        'validation': probas_validation
    }


def calculate_multiclass_predicted_values(model_trainer):
    """
    Method for calculate predicted values for trained model
    Using to calculate metrics, compare with true values
    Return predicted values as dict

    :param model_trainer:
    :return: predicted values
    :type model_trainer: TrainedModel, TrainedModelDNN, TrainedModelCV
    :rtype: dict
    """

    classes_test = numpy.array([])
    classes_validation = numpy.array([])
    model = model_trainer.model

    if model_trainer.x_valid.shape[0] > 0:
        classes_validation = model.predict(model_trainer.x_valid['value'])

    if model_trainer.x_test.shape[0] > 0:
        classes_test = model.predict(model_trainer.x_test['value'])

    classes_train = model.predict(model_trainer.x_train['value'])

    return {
        'test': classes_test,
        'train': classes_train,
        'validation': classes_validation
    }


def make_predict_proba(classifier):
    """
    Method for calculate predict proba values for trained classification model
    Using to calculate classification metrics only
    Return predicted values as dict

    :param classifier: trained classifying model object
    :return: predict proba values
    :type classifier: ClassicClassifierCVFolds, DNNClassifierCVFolds,
            DNNClassifierFold, ClassicClassifierFold
    :rtype: dict
    """

    proba_test = None
    proba_validation = None
    model = classifier.model

    # validation set exists
    if classifier.x_valid.shape[0] != 0:
        proba_validation = model.predict_proba(classifier.x_valid['value'])
        # catch classic calculations case, reshape array to 1D
        if proba_validation.shape[1] > 1:
            proba_validation = proba_validation[:, 1]
        # catch DNN calculations case, reshape array to 1D
        else:
            proba_validation = proba_validation.reshape(-1)

    # test set exists
    if classifier.x_test.shape[0] > 0:
        proba_test = model.predict_proba(classifier.x_test['value'])
        # catch classic calculations case, reshape array to 1D
        if proba_test.shape[1] > 1:
            proba_test = proba_test[:, 1]
        # catch DNN calculations case, reshape array to 1D
        else:
            proba_test = proba_test.reshape(-1)

    # calculate train set
    proba_train = model.predict_proba(classifier.x_train['value'])
    # catch classic calculations case, reshape array to 1D
    if proba_train.shape[1] > 1:
        proba_train = proba_train[:, 1]
    # catch DNN calculations case, reshape array to 1D
    else:
        proba_train = proba_train.reshape(-1)

    # return predicted probas for single fold (or CV model mean values)
    # for each set type (train, test, validation)
    return {
        'test': proba_test,
        'train': proba_train,
        'validation': proba_validation
    }


def calculate_regression_train(regressor):
    """
    Method to calculate regressor metrics for train set,
    calculate RMSE, R2, MAE metrics

    :param regressor: trained regression model object (or CV model)
    :return: rmse metrics
    :type regressor: ClassicRegressorCVFolds,
            ClassicRegressorFold, DNNRegressorCVFolds
    :rtype: dict
    """

    train_mse_tmp = mean_squared_error(
        regressor.y_train['value'], regressor.predict_classes['train'])
    train_mae_tmp = mean_absolute_error(
        regressor.y_train['value'], regressor.predict_classes['train'])
    train_r2_tmp = r2_score(
        regressor.y_train['value'], regressor.predict_classes['train'])

    return {
        ('train', 'RMSE'): train_mse_tmp ** 0.5,
        ('train', 'MAE'): train_mae_tmp,
        ('train', 'R2'): train_r2_tmp,
    }


def calculate_regression_test(regressor):
    """
    Method to calculate regressor metrics for test set (if test set exists),
    calculate RMSE, R2, MAE metrics

    :param regressor: trained regression model object (or CV model)
    :return: rmse metrics
    :type regressor: ClassicRegressorCVFolds,
            ClassicRegressorFold, DNNRegressorCVFolds
    :rtype: dict
    """

    test_mse_tmp = mean_squared_error(
        regressor.y_test['value'], regressor.predict_classes['test'])
    test_mae_tmp = mean_absolute_error(
        regressor.y_test['value'], regressor.predict_classes['test'])
    test_r2_tmp = r2_score(
        regressor.y_test['value'], regressor.predict_classes['test'])

    return {
        ('test', 'MAE'): test_mae_tmp,
        ('test', 'R2'): test_r2_tmp,
        ('test', 'RMSE'): test_mse_tmp ** 0.5,
    }


def calculate_multi_rmse(regressor, n_task):
    """
    Method which calculate root mean squared error value for trained model
    Using regressor attributes
    Return RMSE metrics as dict for train and test datasets

    :param regressor: trained regression model object
    :param n_task:
    :type regressor: TrainedModel, TrainedModelDNN, TrainedModelCV
    :return: rmse metrics
    :rtype: dict
    """

    # calculate mse metric
    test_mse_tmp = mean_squared_error(
        regressor.y_test.values[:, n_task],
        regressor.predict_classes['test'][:, n_task]
    )
    train_mse_tmp = mean_squared_error(
        regressor.y_train.values[:, n_task],
        regressor.predict_classes['train'][:, n_task]
    )

    # convert mse to rmse
    return {
        (str(n_task), 'train', 'RMSE'): train_mse_tmp ** 0.5,
        (str(n_task), 'test', 'RMSE'): test_mse_tmp ** 0.5,
    }


def calculate_multi_mae(regressor, n_task):
    """
    Method which calculate mean absolute error value for trained model
    Using regressor attributes
    Return MAE metrics as dict for train and test datasets

    :param regressor: trained regression model object
    :param n_task:
    :type regressor: TrainedModel, TrainedModelDNN, TrainedModelCV
    :return: rmse metrics
    :rtype: dict
    """

    # calculate mae metric
    test_mae_tmp = mean_absolute_error(
        regressor.y_test.values[:, n_task],
        regressor.predict_classes['test'][:, n_task]
    )
    train_mae_tmp = mean_absolute_error(
        regressor.y_train.values[:, n_task],
        regressor.predict_classes['train'][:, n_task]
    )

    return {
        (str(n_task), 'train', 'MAE'): train_mae_tmp,
        (str(n_task), 'test', 'MAE'): test_mae_tmp,
    }


def calculate_multi_r2(regressor, n_task):
    """
    Method which calculate R^2 value for trained model
    Using regressor attributes
    Return R2 metrics as dict for train and test datasets

    :param regressor: trained regression model object
    :param n_task:
    :type regressor: TrainedModel, TrainedModelDNN, TrainedModelCV
    :return: rmse metrics
    :rtype: dict
    """

    # calculate R^2 metric
    test_r2_tmp = r2_score(
        regressor.y_test.values[:, n_task],
        regressor.predict_classes['test'][:, n_task]
    )
    train_r2_tmp = r2_score(
        regressor.y_train.values[:, n_task],
        regressor.predict_classes['train'][:, n_task]
    )

    # return R^2 metrics as dict
    return {
        (str(n_task), 'train', 'R2'): train_r2_tmp,
        (str(n_task), 'test', 'R2'): test_r2_tmp,
    }


def calculate_regression_valid(regressor):
    """
    Method which calculate cross-validation metrics for trained model,
    if validation set exists
    Return cross-validation metrics as dict

    :param regressor: trained regression model object (or CV model)
    :return: rmse metrics
    :type regressor: ClassicRegressorCVFolds,
            ClassicRegressorFold, DNNRegressorCVFolds
    :rtype: dict
    """

    # calculate metrics
    cv_r2 = r2_score(
        regressor.y_valid['value'], regressor.predict_classes['validation'])
    cv_mse = mean_squared_error(
        regressor.y_valid['value'], regressor.predict_classes['validation'])
    cv_mae = mean_absolute_error(
        regressor.y_valid['value'], regressor.predict_classes['validation'])

    return {
        ('validation', 'R2'): cv_r2,
        ('validation', 'RMSE'): cv_mse ** 0.5,
        ('validation', 'MAE'): cv_mae
    }


def confusion_matrix_metrics(classifier):
    """
    Method to calculate confusion matrix metrics for trained classifier model
    Using to write metrics to csv file or make confusion matrics plots

    :param classifier: trained classifying model (fold or CV model)
    :return: confusion matrix metrics
    :rtype: dict
    """

    test = None
    valid = None
    train = confusion_matrix(
        classifier.y_train['value'], classifier.predict_classes['train'])

    # test set exists
    if classifier.x_test.shape[0] > 0:
        y_pred_ts = classifier.predict_classes['test']
        test = confusion_matrix(classifier.y_test['value'], y_pred_ts)

    # validation set exists
    if classifier.y_valid.shape[0] > 0:
        y_pred_val = classifier.predict_classes['validation']
        valid = confusion_matrix(classifier.y_valid['value'], y_pred_val)

    return {
        'train': train,
        'test': test,
        'validation': valid
    }


def write_confusion_matrix_metrics(metrics, metrics_file_path):
    # TODO make docstring there
    # TODO make readable variables names there
    tp = [
        metrics['train'][0, 0],
        metrics['validation'][0, 0],
        metrics['test'][0, 0]
    ]
    tn = [
        metrics['train'][1, 1],
        metrics['validation'][1, 1],
        metrics['test'][1, 1]
    ]
    fp = [
        metrics['train'][1, 0],
        metrics['validation'][1, 0],
        metrics['test'][1, 0]
    ]
    fn = [
        metrics['train'][0, 1],
        metrics['validation'][0, 1],
        metrics['test'][0, 1]
    ]
    npv = [
        tn[0] / (tn[0] + fn[0]),
        tn[1] / (tn[1] + fn[1]),
        tn[2] / (tn[2] + fn[2])
    ]
    spec = [
        tn[0] / (tn[0] + fp[0]),
        tn[1] / (tn[1] + fp[1]),
        tn[2] / (tn[2] + fp[2])
    ]
    formatted_metrics = {
        'tp__tr_val_ts': tp, 'tn__tr_val_ts': tn, 'fp__tr_val_ts': fp,
        'fn__tr_val_ts': fn, 'npv__tr_val_ts': npv, 'spec__tr_val_ts': spec
    }

    with open(metrics_file_path, 'w') as outfile:
        json.dump(formatted_metrics, outfile)
