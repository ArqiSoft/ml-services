"""
Module which contain basic trainer class with all basic variables definitions,
calculations and logic. Inherit from this class each time when you want to add
new trainer class. DO NOT use this class to train models, use 'child' classes.
"""

import copy
import json
import ntpath
import operator
import os
import uuid
import zipfile
from collections import OrderedDict
from time import gmtime, strftime, time

import numpy
import pandas
import shap
from keras.callbacks import ModelCheckpoint
from keras.models import load_model
from scipy import sparse, spatial
from scipy.linalg import inv
from sklearn import model_selection
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesRegressor, IsolationForest
from sklearn.externals import joblib
from sklearn.feature_selection import (
    SelectKBest, f_classif, VarianceThreshold, f_regression
)
from sklearn.neighbors import KernelDensity
from sklearn.preprocessing import RobustScaler, StandardScaler, MinMaxScaler
from xhtml2pdf import pisa

from MLLogger import BaseMLLogger
from general_helper import (
    DENSITY_MODEL_FILENAME, TRAIN_MEAN_FILENAME, SCALER_FILENAME, get_distance,
    DISTANCE_MATRIX_FILENAME, TMP_from_numpy_by_field_name, make_directory,
    post_data_to_blob, get_multipart_object, coeff_determination,
    single_fold_selector, K_MEANS_FILENAME
)
from learner.algorithms import (
    ALGORITHM, OPTIMIZER_FUNCTION, CLASSIC, MULTI_REGRESSOR,
    algorithm_code_by_name, CLASSIFIER, REGRESSOR, algorithm_name_by_code,
    model_type_by_name, TRAIN_FUNCTION, ADDITIONAL_ARGUMENTS, DNN,
    HYPERPARAMETERES_HYPEROPT, HYPERPARAMETERES_SKOPT, MULTI_CLASSIFIER,
    algorithm_method_by_name
)
from learner.folds import (
    BaseCVFolds, ClassicClassifierCVFolds, ClassicMultiClassifierCVFolds,
    ClassicMultiRegressorCVFolds, ClassicRegressorCVFolds, DNNRegressorCVFolds,
    DNNClassifierCVFolds, DNNMultiClassifierCVFolds, DNNMultiRegressorCVFolds
)
from learner.models import (
    class_weight_to_dict, onehot_encoded, train_dnn_multi_valid, optimizer
)
from learner.plotters import (
    REGRESSION_RESULT_TEST, REGRESSION_RESULT_TRAIN, ROC_PLOT, CONFUSION_MATRIX
)
from messages import trained_model_info_message

from report_helper.TMP_text import (
    QMRF_REPORT, TRAINING_CSV_METRICS, report_text_classifier,
    report_text_regressor
)
from report_helper.html_render import create_report_html, QMRF_TEMPLATE
from report_helper.qmrf_report_constants import (
    make_initial_context, update_user_context, make_model_context
)

LOGGER = BaseMLLogger(log_name='model_trainer', log_file_name='model_trainer')
SKOPT_OPTIMIZERS = ['forest', 'gbrt', 'gauss']
NOT_SKOPT_OPTIMIZERS = ['parzen']


class Trainer:
    def __init__(
            self, filename, prediction_target, dataframe, scale=None,
            test_set_size=0.2, fptypes=None, n_split=4, output_path=None,
            sub_folder=None, seed=7, pca=None, var_threshold=None, k_best=None,
            z_score=None, outliers_from_test=None,  outliers_from_train=None,
            manual_test_set=None, subsample_size=1.0, hyperopt_optimize=False,
            n_iter_optimize=100, opt_method='default', dnn_layers=None,
            dnn_nodes=None
    ):
        """
        Initialize any model trainer class with general training parameters

        :param filename:
        :param prediction_target: name of trained property
        :param dataframe: prepared dataframe, using to training model
        :param scale: scaler type name
            It should be 'robust', 'standard', 'minmax' or None
        :param test_set_size: size of test set should be in interval [0, 1)
        :param fptypes: fingerprints set
        :param n_split:
        :param output_path:
        :param sub_folder: folder path to write trained model files to
        :param seed:
        :param pca:
        :param var_threshold:
        :type filename: str
        :type prediction_target: str
        :type scale: str
        :type test_set_size: float
        :type fptypes: list
        :type n_split: int
        :type output_path: str
        :type sub_folder: str
        """
        self.n_iter_optimize = n_iter_optimize
        self.opt_method = opt_method
        self.manual_test_set = manual_test_set
        self.prediction_target = prediction_target
        self.n_split = n_split
        self.seed = seed
        self.fptypes = fptypes
        self.scale = scale
        self.pca = pca
        self.threshold = var_threshold
        self.test_size = test_set_size
        self.train_size = 1 - test_set_size
        self.metrics = {}
        self.path_to_scaler = None
        self.path_to_density_model = None
        self.path_to_distance_matrix = None
        self.path_to_train_mean = None
        self.path_to_k_means = None
        self.subsample_size = subsample_size
        self.hyperopt_optimize = hyperopt_optimize
        self.n_iter_optimize = n_iter_optimize
        self.model_type = self.model_type or CLASSIFIER
        self.dataframe = self.apply_dataframe(dataframe)
        self.parameter_dataframe = TMP_from_numpy_by_field_name(
            self.dataframe, 'name', [prediction_target]
        ).reshape(self.dataframe.shape[0], 1)
        self.filename = ntpath.basename(filename)
        self.template_tags = self.make_template_tags()
        self.sub_folder = self.make_sub_folder(sub_folder, output_path)
        self.bins = self.get_bins_number()
        self.x_train = numpy.array([])
        self.x_test = numpy.array([])
        self.y_train = numpy.array([])
        self.y_test = numpy.array([])
        self.y_valid = numpy.array([])
        self.x_valid = numpy.array([])
        self.mean_centroid = 0
        self.vi_file_path = str()
        # prepare dataset indexes for stratified K-fold split or single fold
        self.cv = self.make_stratified_k_fold()
        self.cv_valid_ids = []
        self.csv_formatter = '{:.04f}'.format
        self.plots = dict()
        self.density_mean = 0
        self.density_std = 0
        self.distance_mean = 0
        self.distance_std = 0
        self.train_mean = 0.0
        self.train_shape = 0
        self.modi = 0.0
        self.linspace_size = 5
        self.manual_test_set = manual_test_set
        self.cv_model = BaseCVFolds()
        self.correct_dataframe_with_z_score(z_score)
        self.correct_data_features_values(k_best)
        self.update_template_tags()
        self.split_dataframe_by_subsample_size()
        self.split_dataframes()
        self.apply_outliers(outliers_from_train, 'x_train', 'y_train')
        self.apply_outliers(outliers_from_test, 'x_test', 'y_test')
        self.correct_by_threshold()
        self.correct_by_pca()
        self.apply_cross_validation()
        self.path_to_scaler = self.apply_scaler()
        self.model_name = 'no model name'
        self.model_method = 'no model method'

    def train_model(self, algorithm_code):
        """
        Basic train model method. Train model with given parameters by
        algorithm code, check training time int minutes, save model to file.
        Returns training metrics such as model object, path to model file,
        training time and trained model name
        Expand it in child class

        :param algorithm_code: code of algorithm
        :type algorithm_code: str
        :return: basic model training metrics
        :rtype: dict
        """

        # get model name by code
        self.model_name = algorithm_name_by_code(algorithm_code)
        self.model_method = algorithm_method_by_name(self.model_name)
        # get all folds training time
        training_time_minutes = self.train_folds_models()
        # save all folds models to files
        self.store_folds_trainings_data(training_time_minutes)
        # calculate mean metrics and each fold metrics
        self.calculate_metrics()

        return self.cv_model

    def make_training_function(self, optimal_parameters):
        training_function = ALGORITHM[TRAIN_FUNCTION][self.model_name](
            optimal_parameters)

        return training_function

    def make_training_parameters_grid(self):
        if self.opt_method == 'default':
            training_parameters_grid = ALGORITHM[OPTIMIZER_FUNCTION][self.model_name](self)
        elif self.opt_method in NOT_SKOPT_OPTIMIZERS:
            training_parameters_grid = ALGORITHM[HYPERPARAMETERES_HYPEROPT][self.model_name]
        elif self.opt_method in SKOPT_OPTIMIZERS:
            training_parameters_grid = ALGORITHM[HYPERPARAMETERES_SKOPT][self.model_name]
        else:
            raise ValueError(
                'unknown optimizer: {}'.format(self.opt_method))
        return training_parameters_grid

    def make_optimal_parameters(self, training_parameters_grid):
        if self.opt_method != 'default':
            optimal_parameters = optimizer(
                self, ALGORITHM[TRAIN_FUNCTION][self.model_name],
                training_parameters_grid
            )
        else:
            optimal_parameters = training_parameters_grid

        return optimal_parameters

    def train_folds_models(self):
        """
        Method to train models fold by fold.
        Update cv_model attribute with trained cv model folds.
        Return time spent to train all folds.

        :return: training time (minutes)
        :rtype: float
        """

        # get model name from algorithm code
        self.cv_model.model_name = self.model_name
        self.cv_model.model_type = model_type_by_name(self.model_name)

        # train model with timer
        start_training = time()

        training_parameters_grid = self.make_training_parameters_grid()
        optimal_parameters = self.make_optimal_parameters(
            training_parameters_grid)
        training_function = self.make_training_function(optimal_parameters)
        arguments = ALGORITHM[ADDITIONAL_ARGUMENTS][self.model_name](self)
        self.train_k_fold_models(
            training_function, arguments, optimal_parameters)

        return (time() - start_training) / 60

    def store_folds_trainings_data(self, training_time_minutes):
        """
        Method to dump (save to file) trained folds model (*.sav or *.h5) files

        :param training_time_minutes: time spent to train model (minutes)
        :type training_time_minutes: float
        """

        # write trained models data to files
        for number, model in enumerate(self.cv_model.models):
            model_filename = self.make_model_filename(number + 1)
            path_to_model = os.path.join(self.sub_folder, model_filename)
            joblib.dump(model.model, path_to_model)
            model.path_to_model_file = path_to_model
            model.training_time = training_time_minutes
            model.model_name = self.model_name

    def calculate_metrics(self):
        """
        Method to calculate folds metrics, mean metrics for trainer model
        Store calculated metrics at 'metrics' class attribute
        """

        # define CV model attributes
        self.cv_model.prediction_target = self.prediction_target
        self.cv_model.make_dataframes(self)
        self.cv_model.sub_folder = self.sub_folder
        # write classification metrics
        self.metrics[self.model_name] = self.cv_model.calculate_metrics()

    def train_k_fold_models(
            self, training_method, arguments, optimal_parameters
    ):
        """
        Method for train all folds of chosen model training method

        :param training_method: model training function
        :param arguments: additional model arguments for "fit"
        :param optimal_parameters: optimal parameters for training
        :type arguments: dict
        """

        for train_ids, valid_ids in self.cv_valid_ids:
            # define using subdataframes
            x_train = self.x_train[train_ids]
            y_train = self.y_train[train_ids]
            x_valid = self.x_train[valid_ids]
            y_valid = self.y_train[valid_ids]

            # copy training method to avoid chain "fit"
            model = copy.copy(training_method).fit(
                x_train['value'], y_train['value'], **arguments
            )

            # define trained model object
            trained_model = self.cv_model.fold_class(
                model=model, model_type=self.cv_model.model_type,
                x_test=self.x_test, y_test=self.y_test,
                x_train=x_train, y_train=y_train,
                x_valid=x_valid, y_valid=y_valid,
                prediction_target=self.prediction_target,
                sub_folder=self.sub_folder
            )
            self.cv_model.models.append(trained_model)

    def make_model_dict(self, body):
        """
        Method which make prepared model data for POST request.

        :param body: input with train model command rabbitmq message
        :type body: dict
        :return: prepared model data
        :rtype: dict
        """

        # make model data dict
        model_data_dict = {
            'ModelInfo': json.dumps(trained_model_info_message(body, self)),
            'FileType': 'MachineLearningModel',
            'SkipOsdrProcessing': 'true',
            'correlationId': body['CorrelationId']
        }

        return model_data_dict

    def post_model(self, body, oauth):
        """
        Method which POST basic model train data, such as model object
        Expand it in child class

        :param body: body of message received from RabbitMQ queue
        :param oauth: using in ml service OAuth2Session object
        :type body: dict
        :return: POST response
        """

        # model data dict
        model_data_dict = self.make_model_dict(body)
        path_to_archive = self.cv_model.compress_models()
        path_to_archive = self.compress_additional_files(path_to_archive)
        # make model multipart encoded object
        multipart_model = get_multipart_object(
            body, path_to_archive, 'application/x-spss-sav',
            additional_fields=model_data_dict
        )

        # send, http POST request to blob storage api with model data
        response_model = post_data_to_blob(oauth, multipart_model)

        return response_model

    def post_applicability_domain(self, body, oauth):
        """
        Method which make and POST applicability domain values,
        matrixes and models

        :param body: body of message received from RabbitMQ queue
        :param oauth: using in ml service OAuth2Session object
        :type body: dict
        """

        # calculate all values, matrixes and models of applicability domain
        self.make_applicability_domain()

        # POST density model to blob storage
        density_model_response = self.post_density_model(body, oauth)
        # POST distance matrix to blob storage
        distance_matrix_response = self.post_distance_matrix(body, oauth)
        # POST train mean vector as file to blob storage
        train_mean_response = self.post_train_mean(body, oauth)

        # save all blob ids
        density_model_blob_id = None
        if density_model_response:
            density_model_blob_id = density_model_response.json()[0]
        body['DensityModelBlobId'] = density_model_blob_id

        distance_matrix_blob_id = None
        if distance_matrix_response:
            distance_matrix_blob_id = distance_matrix_response.json()[0]
        body['DistanceMatrixBlobId'] = distance_matrix_blob_id

        train_mean_blob_id = None
        if train_mean_response:
            train_mean_blob_id = train_mean_response.json()[0]
        body['TrainMeanBlobId'] = train_mean_blob_id

    def post_distance_matrix(self, body, oauth):
        """
        Method for POST distance matrix to blob storage.
        Return POST response

        :param body: body of message received from RabbitMQ queue
        :param oauth: using in ml service OAuth2Session object
        :type body: dict
        :return: POST response
        """

        path_to_distance_matrix = self.path_to_distance_matrix
        if not path_to_distance_matrix:
            return None

        distance_matrix_info = {
            'ParentId': body['CurrentModelId'],
            'SkipOsdrProcessing': 'true',
            'FileInfo': json.dumps({
                'fileType': 'Distance_matrix'
            })
        }
        multipart_distance_matrix = get_multipart_object(
            body, path_to_distance_matrix, 'application/x-spss-sav',
            additional_fields=distance_matrix_info
        )
        response_distance_matrix = post_data_to_blob(
            oauth, multipart_distance_matrix)

        return response_distance_matrix

    def post_train_mean(self, body, oauth):
        """
        Method for POST train mean vector as file to blob storage.
        Return POST response

        :param body: body of message received from RabbitMQ queue
        :param oauth: using in ml service OAuth2Session object
        :type body: dict
        :return: POST response
        """

        path_to_train_mean = self.path_to_train_mean
        if not path_to_train_mean:
            return None
        # prepare metadata
        train_mean_info = {
            'ParentId': body['CurrentModelId'],
            'SkipOsdrProcessing': 'true',
            'FileInfo': json.dumps({
                'fileType': 'Train_mean'
            })
        }
        # prepare file with metadata as multipart object
        multipart_train_mean = get_multipart_object(
            body, path_to_train_mean, 'application/x-spss-sav',
            additional_fields=train_mean_info
        )
        # send train mean with metadata
        response_train_mean = post_data_to_blob(
            oauth, multipart_train_mean)

        return response_train_mean

    def post_density_model(self, body, oauth):
        """
        Method for POST density model to blob storage.
        Return POST response

        :param body: body of message received from RabbitMQ queue
        :param oauth: using in ml service OAuth2Session object
        :type body: dict
        :return: POST response
        """

        path_to_density_model = self.path_to_density_model
        if not path_to_density_model:
            return None
        # prepare metadata
        density_model_info = {
            'ParentId': body['CurrentModelId'],
            'SkipOsdrProcessing': 'true',
            'FileInfo': json.dumps({
                'fileType': 'Density_model'
            })
        }
        # prepare file with metadata as multipart object
        multipart_density_model = get_multipart_object(
            body, path_to_density_model, 'application/x-spss-sav',
            additional_fields=density_model_info
        )
        # send density model with metadata
        response_density_model = post_data_to_blob(
            oauth, multipart_density_model)

        return response_density_model

    def make_applicability_domain(self):
        """
        Method for calculate applicability domain for current trainer.
        Save calculation results to files and add files paths to
        trainer attributes
        """

        self.estimate_density()
        self.estimate_distance()
        self.k_mean_clusters()
        self.calculate_trainer_modi()

    def correct_by_pca(self):
        """
        Procedure that performs simple dimensionality reduction of the feature
        vector using Principle Components Analysis.
        """
        if self.pca:
            pc = PCA(n_components=self.pca)
            pc.fit(self.x_train)
            self.x_train = pc.transform(self.x_train)
            if self.test_size > 0:
                self.x_test = pc.transform(self.x_test)
            self.bins = self.pca

    def correct_by_threshold(self):
        """
        Procedure that discards low-variance features
        (within a certain threshold) from the feature vector
        """
        if not self.threshold:
            return

        selector = VarianceThreshold(threshold=self.threshold)
        selector = selector.fit(self.x_train)
        self.x_train = selector.transform(self.x_train)
        if self.test_size != 0:
            self.x_test = selector.transform(self.x_test)
        self.bins = self.x_train.shape[1]
        threshold_selector_filename = '{}_var_threshold_selector.sav'.format(
            self.filename.split('.')[0])
        path_to_threshold_selector = os.path.join(
            self.sub_folder, threshold_selector_filename
        )
        joblib.dump(selector, path_to_threshold_selector)

    def correct_dataframe_with_z_score(self, z_score):
        """
        Procedure that discards observations from the data
        by performing z-score test of the end-point value (Y - vector)
        """
        if not z_score:
            return

        valuename = self.prediction_target
        mean_value = self.dataframe[valuename].mean()
        numerator = self.dataframe[valuename] - mean_value
        denominator = self.dataframe[valuename].std(ddof=0)
        drop_index = self.dataframe[numerator / denominator >= z_score].index
        self.dataframe = self.dataframe.drop(drop_index)

    def apply_scaler(self):
        """
        Method that applies chosen scaler to the feature vectors of the dataset
        :return: path to scaler object
        :rtype: str
        """
        if not self.scale or self.scale == 'not use scaler values':
            return None

        if self.scale == 'robust':
            scaler = RobustScaler()

        elif self.scale == 'standard':
            scaler = StandardScaler()

        elif self.scale == 'minmax':
            scaler = MinMaxScaler()

        else:
            raise TypeError('Unknown scale: {}'.format(self.scale))

        scaler = scaler.fit(self.x_train['value'])
        tmp_x_train = scaler.transform(self.x_train['value'])

        for row in range(0, self.x_train.shape[0]):
            for column in range(0, self.x_train.shape[1]):
                self.x_train[row][column]['value'] = tmp_x_train[row][column]

        if self.test_size != 0 or self.manual_test_set is not None:
            tmp_x_test = scaler.transform(self.x_test['value'])
            for row in range(0, self.x_test.shape[0]):
                for column in range(0, self.x_test.shape[1]):
                    self.x_test[row][column]['value'] = tmp_x_test[row][
                        column]

        path_to_scaler = os.path.join(self.sub_folder, SCALER_FILENAME)

        return joblib.dump(scaler, path_to_scaler)[0]

    def post_scaler(self, body, oauth):
        """
        Method for POST scaler model to blob storage.
        Add scaler model blob id to body

        :param body: body of message received from RabbitMQ queue
        :param oauth: using in ml service OAuth2Session object
        :type body: dict
        """

        scaler_path = self.path_to_scaler
        if not scaler_path:
            return None
        # prepare scaler metadata
        scaler_info = {
            'ParentId': body['CurrentModelId'],
            'SkipOsdrProcessing': 'true',
            'FileInfo': json.dumps({
                'fileType': 'Scaler'
            })
        }
        # prepare scaler with metadata as multipart object
        multipart_scaler = get_multipart_object(
            body, scaler_path, 'application/x-spss-sav',
            additional_fields=scaler_info
        )
        # send scaler with metadata to OSDR blob storage
        response_scaler = post_data_to_blob(oauth, multipart_scaler)

        scaler_blob_id = None
        if response_scaler:
            scaler_blob_id = response_scaler.json()[0]
        body['ScalerBlobId'] = scaler_blob_id

    def make_sub_folder(self, sub_folder, output_path):
        """
        Method which make subfolder (which contain model's files) if not exist
        Use child's class subfolder name creator. Return subfolder full path

        :param sub_folder: subfolder name or None, trainer init argument
        :param output_path: output path or None, trainer init argument
        :type sub_folder: str
        :type output_path: str
        :return: subfolder full path
        :rtype: str
        """

        if not sub_folder:
            folder_path = self.make_sub_folder_path()
        else:
            folder_path = sub_folder

        if output_path:
            folder_path = os.path.join(output_path, folder_path)

        make_directory(folder_path)

        return folder_path

    def make_perfomance_csv(self):
        """
        Method which make csv file with all needed metrics for model and
        write it to csv file

        :return: path to csv file with metrics
        :rtype: str
        """

        # format trained models metrics
        self.apply_metrics_formatter()

        # for fold_number, fold_metrics in self.metrics.items():
        # get model perfomance metrics
        reformatted_metrics = OrderedDict()
        for metric_name, fold_metrics in self.metrics.items():
            for fold_name, fold_metric in fold_metrics.items():
                header_fold_name = '{} CV fold {}'.format(
                    metric_name, fold_name)
                if fold_name == 'mean':
                    header_fold_name = '{} CV {}'.format(
                        metric_name, fold_name)

                reformatted_key = '{} {}'.format(metric_name, fold_name)
                reformatted_metrics[reformatted_key] = fold_metric
                if fold_name == 'mean':
                    reformatted_metrics.move_to_end(
                        reformatted_key, last=False)
                # add metrics header
                reformatted_metrics[reformatted_key].update({
                    ('Dataset', 'Evaluation metrics'): header_fold_name
                })

        # sort rows
        model_performance = pandas.DataFrame.from_dict(
            reformatted_metrics
        ).rename(
            index={'Dataset': '0', 'train': '1', 'test': '3',
                   'validation': '2'}
        ).sort_index().rename(
            index={'1': 'Train', '3': 'Test', '2': 'Validation',
                   '0': 'Dataset'}
        )

        # format model perfomance metrics
        self.template_tags['metrics'] += model_performance.to_html(
            header=False
        ).replace(
            'th>', 'td>'
        ).replace(
            '<th', '<td'
        ).replace(
            'style="text-align: right;"', ''
        ).replace(
            'valign="top"', ''
        )
        # write perfomance metrics to file
        csv_file_name = '{}_{}.csv'.format(
            self.cv_model.model_name, self.filename.split('.')[0])
        path_to_csv = os.path.join(self.sub_folder, csv_file_name)
        model_performance.to_csv(path_to_csv)

        return path_to_csv

    def post_performance(self, body, oauth):
        """
        Method for write training metrics to csv file and
        POST csv file to blob storage.
        Return path to created csv file

        :param body: body of message received from RabbitMQ queue
        :param oauth: using in ml service OAuth2Session object
        :type body: dict
        :return: path to csv file with training metrics
        :rtype: str
        """

        csv_file_path = self.make_perfomance_csv()
        # prepare metrics metadata
        csv_metrics_info = {
            'FileInfo': json.dumps({
                'modelName': self.cv_model.model_name,
                'fileType': TRAINING_CSV_METRICS
            }),
            'ParentId': body['CurrentModelId']
        }
        # prepare training metrics with metadata as multipart object
        multipart_csv = get_multipart_object(
            body, csv_file_path, 'text/csv',
            additional_fields=csv_metrics_info
        )
        # send training metrics with metadata to OSDR blob storage
        response = post_data_to_blob(oauth, multipart_csv)
        body['NumberOfGenericFiles'] += 1

        LOGGER.info(
            'CSV sended with status code: {} blob id: {}'.format(
                response.status_code, response.json()[0])
        )

        return csv_file_path

    def apply_metrics_formatter(self):
        """
        Apply needed formatter to float numbers for all metrics
        Redefine self.csv_formatter in childs if wont to change float format
        """

        for fold_metrics in self.metrics.values():
            for metrics_set in fold_metrics.values():
                for metric_name, metric_value in metrics_set.items():
                    metrics_set[metric_name] = self.csv_formatter(metric_value)

    def post_qmrf_report(self, body, oauth):
        """
        Method which create and POST to blob storage training QMRF report

        :param body: body of message received from RabbitMQ queue
        :param oauth: oauth object
        :type body: dict
        :return: POST response
        """

        # generate QMRF report pdf file and get path to that file
        path_to_pdf_qmrf_report = self.make_qmrf_report(oauth, body)
        # define QMRF report file metadata
        qmrf_report_info = {
            'FileInfo': json.dumps({
                'modelName': self.cv_model.model_name,
                'fileType': QMRF_REPORT
            }),
            'ParentId': body['CurrentModelId']
        }
        # make QMRF pdf report multipart encoded object
        multipart_qmrf_report = get_multipart_object(
            body, path_to_pdf_qmrf_report, 'application/pdf',
            additional_fields=qmrf_report_info
        )
        # POST http request to blob storage api with QMRF report
        response = post_data_to_blob(oauth, multipart_qmrf_report)
        body['NumberOfGenericFiles'] += 1

        return response

    def make_qmrf_report(self, oauth, body):
        """
        Method which make training QMRF pdf report

        :param oauth: body of message received from RabbitMQ queue
        :param body: oauth object
        :type body: dict
        :return: path to QMRF report pdf file
        :rtype: str
        """

        # make QMRF report as HTML and get path
        qmrf_context = make_initial_context()
        if oauth and body:
            update_user_context(oauth, body['UserId'], qmrf_context)
        if body:
            make_model_context(self, qmrf_context, body)
        formatted_model_name = '_'.join(self.cv_model.model_name.split())
        html_report_path = os.path.join(
            self.sub_folder, '{}_{}'.format(
                formatted_model_name, QMRF_TEMPLATE)
        )
        path_to_html_qmrf_report = create_report_html(
            qmrf_context, QMRF_TEMPLATE, html_report_path)

        # convert HTML report content to pdf and write to file
        pdf_report_path = os.path.join(
            self.sub_folder, '{}_QMRF_report.pdf'.format(formatted_model_name))
        pisa.CreatePDF(
            open(path_to_html_qmrf_report, 'r'), open(pdf_report_path, 'wb'),
            path=os.path.abspath(self.sub_folder)
        )
        path_to_pdf_qmrf_report = os.path.abspath(pdf_report_path)

        return path_to_pdf_qmrf_report

    def apply_dataframe(self, dataframe):
        """
        Remove nan values from initial dataframe
        nan values can come from bad fingerprints/descroptors or bad molecules

        :param dataframe: almost prepared dataset which have applied
            all fingerprints\descriptors, radius and size
        :return: dataframe with cleared nan values
        :type dataframe: numpy.ndarray
        :rtype: numpy.ndarray
        """

        return dataframe[~numpy.isnan(dataframe['value']).any(axis=1)]

    def apply_cross_validation(self):
        """
        Method for split dataframe train sets (x and y) to folds by ids
        Update trainer attribute with splitted ids
        """

        # indexes for new train dataset depends on model type
        if self.model_type == MULTI_REGRESSOR:
            bins = numpy.linspace(0, self.y_train.shape[0], self.linspace_size)
            if self.manual_test_set is not None:
                y_train = numpy.digitize(self.y_train, bins)[:, 0]
            else:
                y_train = numpy.digitize(self.y_train, bins)[:, 0]
        elif (
            self.model_type == CLASSIFIER or
            self.model_type == MULTI_CLASSIFIER
        ):
            y_train = self.y_train[:, 0]['value']
        elif self.model_type == REGRESSOR:
            bins = numpy.linspace(0, self.y_train.shape[0], self.linspace_size)
            y_train = numpy.digitize(self.y_train['value'], bins)[:, 0]
        else:
            raise ValueError(
                'Unknow model type: {}'.format(self.model_type))
        # split dataframe by folds
        for train_id, valid_id in self.cv(self.x_train['value'], y_train):
            self.cv_valid_ids.append([train_id.tolist(), valid_id.tolist()])

    def make_stratified_k_fold(self):
        """
        Method for define stratified K fold function, using training parameters

        :return: stratified K fold function
        """

        # single fold training
        if self.n_split == 1:
            k_fold_function = single_fold_selector
        # multiple folds training
        else:
            k_fold_function = model_selection.StratifiedKFold(
                shuffle=True, n_splits=self.n_split, random_state=0
            ).split

        return k_fold_function

    def apply_outliers(self, outlier, x_attribute, y_attribute):
        """
        Experimental function that discards outliers
        (feature vector - based) from the data
        using Isolation Forest anomaly detection method.
        :param outlier:
        :param x_attribute:
        :param y_attribute:
        """
        # TODO review the code
        if not outlier:
            return

        iso_for = IsolationForest().fit(self.x_train)
        outliers_train = []
        x_value = getattr(self, x_attribute)
        y_value = getattr(self, y_attribute)

        for i in range(x_value.shape[0]):
            if iso_for.predict(x_value[i, :].reshape(1, -1)) == -1:
                outliers_train.append(i)

        x_value = numpy.delete(x_value, outliers_train, axis=0)
        y_value = numpy.delete(y_value, outliers_train, axis=0)
        setattr(self, x_attribute, x_value)
        setattr(self, y_attribute, y_value)

    def make_centroid(self):
        """
        :return: mean - centroid of train data, VI - inverted covariance matrix
        """
        # TODO make docstring there
        self.mean_centroid = numpy.mean(self.x_train, axis=0)
        vi_filename = '{}/{}_vi_matrix'.format(self.sub_folder, uuid.uuid1())
        covariance = numpy.cov(self.x_train.T)
        vi = inv(covariance)
        vi.tofile(vi_filename)
        self.vi_file_path = vi_filename

    def make_template_tags(self):
        """
        Method which make basic template_tags for trainer
        Expand it on 'child' class

        :return: template_tags with basic keys
        :rtype: dict
        """

        template_tags = dict()
        template_tags['bin_plots'] = []
        template_tags['dataset'] = 'Parameter {} from dataset file {}'.format(
            self.prediction_target, self.filename)
        template_tags['dataset_info'] = []
        template_tags['metrics'] = ''

        return template_tags

    def make_sub_folder_path(self):
        """
        Method which make basic subfolder path, which contain all model's files
        Expand that method in child class

        :return: datetime string as basic subfolder name
        :rtype: str
        """

        date_time = '{}_{}'.format(
            uuid.uuid1(), strftime('%Y_%m_%d__%H_%M_%S', gmtime())
        )
        return date_time

    def make_model_filename(self, k_fold_number):
        """
        Method which make basic filename for new trained model

        :return: filename for new trained model
        :rtype: str
        """

        model_splited_filename = '_'.join(self.model_name.split())

        return '{}_k_fold_{}.sav'.format(model_splited_filename, k_fold_number)

    def get_bins_number(self):
        """
        Method which calculate and return bins number

        :return: bins number
        :rtype: int
        """

        return self.dataframe.shape[1] - 1

    def split_dataframes(self):
        """
        Method to split dataframe to test/train subdataframes. It splits
        x dataframes correspond to y dataframes by indexes. It depends on
        test dataset size and manual test set value.
        For example test_size=0.2 mean that test set would contain 200
        molecules, train set 800 molecules when initial dataset contain 1000
        molecules. Test set size does not matter if manual test set exists.
        It uses manual test set fully, independ of test set size
        """

        # x values from source dataframe
        initial_x_dataframe = self.dataframe[:, 0:self.bins]

        # no tests sets, only train sets (fully source dataframe)
        if self.manual_test_set is None and self.test_size == 0:
            self.x_train = initial_x_dataframe
            self.y_train = self.parameter_dataframe

        # test set from source dataframe (molecules count * test_size)
        # train set from source dataframe (molecules count * (1 - test_size))
        elif self.manual_test_set is None and self.test_size != 0:
            self.x_train, self.x_test, self.y_train, self.y_test = model_selection.train_test_split(
                initial_x_dataframe,
                self.parameter_dataframe,
                train_size=self.train_size,
                test_size=self.test_size,
                random_state=self.seed,
                shuffle=True
            )

        # test set from manual test set dataframe (fully)
        # train set from source dataframe (fully source dataframe)
        elif self.manual_test_set is not None:
            self.manual_test_set = self.manual_test_set[~numpy.isnan(self.manual_test_set['value']).any(axis=1)]
            self.y_train = self.parameter_dataframe
            self.x_train = initial_x_dataframe
            self.y_test = TMP_from_numpy_by_field_name(
                self.manual_test_set, 'name', [self.prediction_target]
            ).reshape(self.manual_test_set.shape[0], 1)
            self.x_test = self.manual_test_set[:, 0:self.bins]

        else:
            raise ValueError(
                'Wrong choice. Test size: {} Manual test set: {}'.format(
                    self.test_size, self.manual_test_set)
            )

    def split_dataframe_by_subsample_size(self):
        """
        this functions finds and removes part of major class examples
        in order to balance dataset true\false ratio
        change name to drop_major_subsample
        """

        # Randomly remove a fraction of major class samples
        if self.subsample_size and 0 <= self.subsample_size <= 1:
            all_indexes = self.dataframe[:, 0]['molecule_number']

            remove_indexes = numpy.random.choice(
                numpy.array(all_indexes),
                int(all_indexes.shape[0] * (1. - self.subsample_size)),
                replace=False
            )
            self.dataframe = numpy.delete(self.dataframe, remove_indexes, 0)
            self.parameter_dataframe = numpy.delete(
                self.parameter_dataframe, remove_indexes, 0)
        else:
            raise ValueError(
                'subsample of {} is impossible'.format(self.subsample_size))

    def post_plots(self, body, oauth):
        """
        Implement this method in child class

        :param body: body of message received from RabbitMQ queue
        :param oauth: oauth object
        """
        pass

    def make_report_text(self):
        """
        Implement this method in child class
        """
        pass

    def correct_data_features_values(self, k_best):
        """
        Implement this method in child class
        """
        pass

    def update_template_tags(self):
        """
        Implement this method in child class
        """
        pass

    def estimate_density(self):
        """
        Build model for mutivariate kernel density estimation.
        Takes x_train dataframe as input, builds a kernel density model
        with optimised parameters for input vectors,
        and computes density for all input vectors.
        """

        if self.scale is None:
            bandwidth = numpy.logspace(-1, 2, 20)
        else:
            bandwidth = numpy.linspace(0.1, 0.5, 5)

        x_train = self.x_train['value']
        # find best parameters for KD
        grid = model_selection.GridSearchCV(
            KernelDensity(), {'bandwidth': bandwidth}, cv=3)
        grid.fit(x_train)

        # compute KD for x_train
        bw = grid.best_params_
        dens_model = KernelDensity(bw['bandwidth']).fit(x_train)

        samples = dens_model.score_samples(x_train)

        dens_mean = numpy.mean(samples)
        dens_std = numpy.std(samples)

        path_to_model = os.path.join(self.sub_folder, DENSITY_MODEL_FILENAME)
        joblib.dump(dens_model, path_to_model)

        self.path_to_density_model = path_to_model
        self.density_mean = dens_mean
        self.density_std = dens_std

    def estimate_distance(self):
        """
        Takes x_train dataframe as input,
        calculates Mahalonobis distance between whole input
        dataset and each input vector from the dataset
        """

        x_train = self.x_train['value']
        centroid = x_train - numpy.tile(
            numpy.mean(x_train, axis=0), (x_train.shape[0], 1)
        )
        train_mean = numpy.mean(x_train, axis=0)
        self.train_shape = x_train.shape[0] - 1
        x_train = numpy.asarray(x_train)
        dist_list = numpy.apply_along_axis(
            get_distance, 1, x_train, centroid=centroid, train_mean=train_mean,
            train_shape=self.train_shape
        )

        dist_mean = numpy.mean(dist_list)
        dist_std = numpy.std(dist_list)

        matrix_file_path = os.path.join(
            self.sub_folder, DISTANCE_MATRIX_FILENAME)
        train_mean_file_path = os.path.join(
            self.sub_folder, TRAIN_MEAN_FILENAME)

        numpy.savetxt(train_mean_file_path, train_mean)

        centroid = sparse.csr_matrix(
            numpy.asarray(centroid).astype(dtype='float32'))
        sparse.save_npz(matrix_file_path, centroid)

        self.path_to_distance_matrix = matrix_file_path
        self.path_to_train_mean = train_mean_file_path
        self.distance_mean = dist_mean
        self.distance_std = dist_std

    def k_mean_clusters(self, n=50):
        """
        use kmeans to clusterise dataset
        :param n: number of clusters
        """

        means = shap.kmeans(self.x_train['value'], n)
        file_path = os.path.join(self.sub_folder, K_MEANS_FILENAME)
        joblib.dump(means, file_path)
        self.path_to_k_means = file_path

    def calculate_trainer_modi(self):
        pass

    def compress_additional_files(self, path_to_archive):
        """
        Method to compress additional files (such as scaler, train mean vector,
        density model, distance matrix) to archive with model folds

        :param path_to_archive: path to folds archive to add additional files
        :return: path to archive with folds and additional files
        :type path_to_archive: str
        :rtype: str
        """

        # set archive path file to append files
        archive_file = zipfile.ZipFile(path_to_archive, 'a')
        # add scaler
        archive_file.write(
            self.path_to_scaler, self.path_to_scaler.split('/')[-1],
            compress_type=zipfile.ZIP_DEFLATED
        )
        # add train mean
        archive_file.write(
            self.path_to_train_mean,
            self.path_to_train_mean.split('/')[-1],
            compress_type=zipfile.ZIP_DEFLATED
        )
        # add density model
        archive_file.write(
            self.path_to_density_model,
            self.path_to_density_model.split('/')[-1],
            compress_type=zipfile.ZIP_DEFLATED
        )
        # add distance matrix
        archive_file.write(
            self.path_to_distance_matrix,
            self.path_to_distance_matrix.split('/')[-1],
            compress_type=zipfile.ZIP_DEFLATED
        )
        # add k means
        archive_file.write(
            self.path_to_k_means,
            self.path_to_k_means.split('/')[-1],
            compress_type=zipfile.ZIP_DEFLATED
        )

        archive_file.close()

        return path_to_archive

    def make_info_json(self):
        """
        Method to generate json file, using in models uploader
        Create json file in model folder
        Using to simplify models metadata generation after training
        """

        info_dict = {
            'ModelName': None,
            'SourceFileName': None,
            'SourceBlobId': None,
            'SourceBucket': None,
            'ModelBucket': None,
            'Method': algorithm_code_by_name(self.cv_model.model_name),
            'MethodDisplayName': '{}, {}'.format(
                self.cv_model.model_name, self.model_type.capitalize()),
            'ClassName': self.prediction_target,
            'SubSampleSize': None,
            'TestDatasetSize': self.test_size,
            'KFold': self.n_split,
            'Fingerprints': self.fptypes,
            'UserId': None,
            'ModelType': self.model_type,
            'ModelId': str(uuid.uuid1()),
            'Bins': self.bins,
            'Scaler': self.scale,
            'DensityMean': float(self.density_mean),
            'DensityStd': float(self.density_std),
            'DistanceMean': float(self.distance_mean),
            'DistanceStd': float(self.distance_std),
            'TrainShape': self.train_shape,
            'Modi': self.modi,
            'Dataset': {
                'Title': None,
                'Description': None
            },
            'Property': {
                'Category': None,
                'Name': None,
                'Units': None,
                'Description': None
            }
        }

        json_file_path = '{}/{}.json'.format(
            self.sub_folder, '_'.join(self.cv_model.model_name.split())
        )
        json_file = open(json_file_path, 'w')
        json_file.write(json.dumps(info_dict))
        json_file.close()


class ClassicClassifier(Trainer):
    def __init__(
            self, filename, classname, dataframe, test_set_size=0.2, seed=7,
            subsample_size=1.0, n_split=4, fptype=None, output_path=None,
            sub_folder=None, k_best=None, scale=None, pca=None, z_score=None,
            var_threshold=None, outliers_from_test=None, manual_test_set=None,
            outliers_from_train=None, hyperopt_optimize=False,
            opt_method='default', n_iter_optimize=100
    ):
        self.model_type = CLASSIFIER
        super().__init__(
            filename, classname, dataframe, output_path=output_path, seed=seed,
            test_set_size=test_set_size, fptypes=fptype, n_split=n_split,
            scale=scale, sub_folder=sub_folder, pca=pca, z_score=z_score,
            var_threshold=var_threshold, outliers_from_test=outliers_from_test,
            k_best=k_best, outliers_from_train=outliers_from_train,
            manual_test_set=manual_test_set, subsample_size=subsample_size,
            hyperopt_optimize=hyperopt_optimize, opt_method=opt_method,
            n_iter_optimize=n_iter_optimize

        )
        self.cv_model = ClassicClassifierCVFolds()
        self.class_weight_train = list()
        self.class_weight_test = list()
        self.make_class_weights()

    def post_plots(self, body, oauth):
        """
        Method which make plots, then POST plots to blob storage

        :param body: rabbitmq message body
        :type body: dict
        :param oauth: OAuth2 object
        """

        # make needed for model plots
        self.plots = self.cv_model.make_plots()

        for k_fold_number, plots in self.plots.items():
            # make ROC plot multipart encoded object
            roc_info = {
                'FileInfo': json.dumps({
                    'modelName': self.cv_model.model_name,
                    'fileType': ROC_PLOT
                }),
                'ParentId': body['CurrentModelId']
            }
            multipart_roc = get_multipart_object(
                body, plots['roc_plot_path'], 'image/png',
                additional_fields=roc_info
            )
            # send, http POST request to blob storage api with ROC plot
            post_data_to_blob(oauth, multipart_roc)
            body['NumberOfGenericFiles'] += 1

            # make confusion matrix multipart encoded object
            confusion_matrix_info = {
                'FileInfo': json.dumps({
                    'modelName': self.cv_model.model_name,
                    'fileType': CONFUSION_MATRIX
                }),
                'ParentId': body['CurrentModelId']
            }
            multipart_cm = get_multipart_object(
                body, plots['cm_plot_path'], 'image/png',
                additional_fields=confusion_matrix_info
            )
            # send, http POST request to blob storage api with confusion matrix
            post_data_to_blob(oauth, multipart_cm)
            body['NumberOfGenericFiles'] += 1

            # update template tags with model's plots
            self.template_tags['confusion_matrixes'].append(
                plots['cm_plot_path'])
            self.template_tags['roc_plots'].append(
                plots['roc_plot_path'])

    def calculate_trainer_modi(self):
        # TODO make docstring there
        n_act = 0
        n_n_act = 0
        n_act_same = 0
        n_n_act_same = 0
        dataframe = self.dataframe['value']
        for row_1 in dataframe:
            row_1 = numpy.nan_to_num(row_1)
            tmp_dist_list = []
            tmp_rows_dict = {}
            counter = 0
            for row_2 in dataframe:
                row_2 = numpy.nan_to_num(row_2)
                if numpy.array_equal(row_1, row_2):
                    continue
                else:
                    dist = spatial.distance.euclidean(row_1[:-1], row_2[:-1])
                    tmp_dist_list.append(dist)
                    tmp_rows_dict[counter] = row_2
                    counter += 1

            ind = tmp_dist_list.index(min(tmp_dist_list))
            nearest_neighbour = tmp_rows_dict[ind]
            if row_1[-1] == 1:
                n_act += 1
                if nearest_neighbour[-1] == 1:
                    n_act_same += 1
            else:
                n_n_act += 1
                if nearest_neighbour[-1] == 0:
                    n_n_act_same += 1

        self.modi = 0.5 * ((n_act_same / n_act) + (n_n_act_same / n_n_act))

    def make_class_weights(self):
        """
        I dont think we even need to set_class weight for test
        but i will put a dummy just for now
        """
        # TODO check this module
        class_weight_train_dict = class_weight_to_dict(
            self.y_train[:, 0]['value'])
        for index in self.y_train[:, 0]['value']:
            self.class_weight_train.append(class_weight_train_dict[index])

        if self.test_size != 0:
            class_weight_test_dict = class_weight_to_dict(
                self.y_test[:, 0]['value'])
            for index in self.y_test[:, 0]['value']:
                self.class_weight_test.append(class_weight_test_dict[index])

    def correct_data_features_values(self, k_best):
        """
        Function that performs feature selection based on the importance of the features
        Very controversial feature selection technique.
        :param k_best: amount of the best features to retain
        """
        if k_best:
            test = SelectKBest(score_func=f_classif, k=k_best)
            test.fit(self.df_features, self.parameter_dataframe)
            self.df_features = test.transform(self.df_features)
            self.dataframe = self.df_features.join(
                self.parameter_dataframe, on=None, how='left',
                lsuffix='', rsuffix='', sort=False
            )

    def make_sub_folder_path(self):
        """
        Method which make sub folder path for classifier, based on init
        parameters, which will contain all model training data.
        Path will contain datetime and fingerprint data.

        :return: path to sub folder
        :rtype: str
        """

        # get current datetime value
        date_time = super().make_sub_folder_path()
        # convert used fingerprints parameters to string
        if self.fptypes:
            all_fptypes_parameters = []
            for fptype in self.fptypes:
                fptype_parameters = '_'.join(map(str, list(fptype.values())))
                all_fptypes_parameters.append(fptype_parameters)

            fptype_parameters_as_string = '_'.join(all_fptypes_parameters)
        else:
            fptype_parameters_as_string = ''
        # construct full sub folder path
        folder_path = 'classification_{}_{}_{}'.format(
            self.filename.split('.')[0], date_time,
            fptype_parameters_as_string
        )

        return folder_path

    def update_template_tags(self):
        """
        Method which update basic template tags with keys needed for classifier
        trainer
        """

        self.template_tags['radar_plots'] = []
        self.template_tags['confusion_matrixes'] = []
        self.template_tags['roc_plots'] = []
        self.template_tags['dataset'] += ' for classifier model training'

    def make_report_text(self):
        """
        Method which call report text generator for classifier.
        Generator will update classifier template tags with needed for
        training report parameters
        """

        report_text_classifier(self)


class ClassicRegressor(Trainer):
    def __init__(
            self, filename, valuename, dataframe, manual_test_set=None,
            test_set_size=0.2, n_split=4, seed=0, fptype=None, scale=None,
            outliers_from_test=None, k_best=None, outliers_from_train=None,
            output_path=None, sub_folder=None, log=False, top_important=None,
            pca=None, var_threshold=None, z_score=None, subsample_size=1.0,
            hyperopt_optimize=False, n_iter_optimize=100, opt_method='default'
    ):

        self.top_important = top_important
        self.outliers_from_test = outliers_from_test
        self.model_type = REGRESSOR
        super().__init__(
            filename, valuename, dataframe, output_path=output_path, seed=seed,
            test_set_size=test_set_size, fptypes=fptype, n_split=n_split,
            scale=scale, sub_folder=sub_folder, pca=pca, z_score=z_score,
            var_threshold=var_threshold, outliers_from_test=outliers_from_test,
            k_best=k_best, outliers_from_train=outliers_from_train,
            manual_test_set=manual_test_set, subsample_size=subsample_size,
            hyperopt_optimize=hyperopt_optimize, opt_method=opt_method,
            n_iter_optimize=n_iter_optimize
        )
        self.cv_model = ClassicRegressorCVFolds()
        self.log = log
        self.correct_by_top_important()
        self.correct_by_log()
        self.csv_formatter = '{:.4E}'.format

    def calculate_trainer_modi(self):
        self.modi = 0.75

    def correct_by_log(self):
        """
        Procedure that turns all the end-point values (Y) to
        logY.
        """
        if self.log:
            self.y_train = numpy.log(self.y_train)
            self.y_test = numpy.log(self.y_test)

    def correct_by_top_important(self):
        """
        Another controversial feature selection procedure,
        based on the regression decision trees.
        """
        if not self.top_important:
            return

        model = ExtraTreesRegressor()
        model.fit(self.x_train, self.y_train)
        rating_dict = {}
        feat_imp = model.feature_importances_
        for i in range(0, len(feat_imp)):
            rating_dict[i] = feat_imp[i]
        rating_list = sorted(rating_dict.items(), key=operator.itemgetter(1))
        rating_list.reverse()
        indices_list = [x[0] for x in rating_list[:self.top_important]]
        self.x_train = self.x_train[:, indices_list]
        if self.test_size != 0:
            self.x_test = self.x_test[:, indices_list]
        self.bins = self.top_important
        joblib.dump(
            model, '{}/{}_top_important.sav'.format(
                self.sub_folder, self.filename.split('.')[0]
            )
        )

    def correct_data_features_values(self, k_best):
        """
        Clone.
        """
        if k_best:
            model = SelectKBest(score_func=f_regression, k=k_best)
            model = model.fit(self.x_train, self.y_train)
            self.x_train = model.transform(self.x_train)
            self.x_test = model.transform(self.x_test)
            self.bins = k_best

    def make_sub_folder_path(self):
        """
        Method which make sub folder path for regressor, based on init
        parameters, which will contain all model training data.
        Path will contain datetime, fingerprint, scaled, top important,
        outliers and pca parametrs data.

        :return: path to sub folder
        :rtype: str
        """

        # get current datetime value
        date_time = super().make_sub_folder_path()
        # convert used fingerprints parameters to string
        if self.fptypes:
            all_fptypes_parameters = []
            for fptype in self.fptypes:
                fptype_parameters = '_'.join(map(str, list(fptype.values())))
                all_fptypes_parameters.append(fptype_parameters)

            fptype_parameters_string = '_'.join(all_fptypes_parameters)
        else:
            fptype_parameters_string = ''
        # convert other used parameters to string
        scaled = 'scaled' if self.scale else ''
        top = 'top_{}'.format(self.top_important) if self.top_important else ''
        outliers = 'outliers' if self.outliers_from_test else ''
        pca = 'pca_{}'.format(self.pca) if self.pca else ''
        # construct full sub folder path
        folder_path = 'regression_{}_{}_{}_{}_{}_{}_{}'.format(
            self.filename.split('.')[0], date_time, fptype_parameters_string,
            scaled, top, outliers, pca
        )

        return folder_path

    def update_template_tags(self):
        """
        Method which update basic template tags with keys needed for regressor
        trainer
        """

        self.template_tags['distribution_plots'] = []
        self.template_tags['dataset'] += ' for regression model training'
        self.template_tags['regression_result_test'] = []
        self.template_tags['regression_result_train'] = []

    def make_report_text(self):
        """
        Method which call report text generator for regressor.
        Generator will update regressor template tags with needed for
        training report parameters
        """

        report_text_regressor(self)

    def post_plots(self, body, oauth):
        """
        Method which make plots for model, then POST plots to blob storage

        :param body: rabbitmq message body
        :param oauth: OAuth2 object
        :type body: dict
        """

        # make needed for model plots
        self.plots = self.cv_model.make_plots()
        for k_fold_number, plots in self.plots.items():
            # make test regression results plot multipart encoded object
            regression_results_test_info = {
                'FileInfo': json.dumps({
                    'modelName': self.cv_model.model_name,
                    'fileType': REGRESSION_RESULT_TEST
                }),
                'ParentId': body['CurrentModelId']
            }
            regression_results_test = get_multipart_object(
                body, plots['regression_results_test'], 'image/png',
                additional_fields=regression_results_test_info
            )
            # POST to blob storage api with test regression results
            post_data_to_blob(oauth, regression_results_test)
            body['NumberOfGenericFiles'] += 1

            # make train regression results plot multipart encoded object
            regression_results_train_info = {
                'FileInfo': json.dumps({
                    'modelName': self.cv_model.model_name,
                    'fileType': REGRESSION_RESULT_TRAIN
                }),
                'ParentId': body['CurrentModelId']
            }
            regression_results_train = get_multipart_object(
                body, plots['regression_results_train'], 'image/png',
                additional_fields=regression_results_train_info
            )
            # POST to blob storage api with train regression results
            post_data_to_blob(oauth, regression_results_train)
            body['NumberOfGenericFiles'] += 1

            # update template tags with model's plots
            self.template_tags['regression_result_test'].append(
                plots['regression_results_test'])
            self.template_tags['regression_result_train'].append(
                plots['regression_results_train'])


class DNNClassifier(ClassicClassifier):
    def __init__(
            self, filename, classname, dataframe, test_set_size=0.2, seed=7,
            subsample_size=1.0, n_split=4, fptype=None, output_path=None,
            sub_folder=None, k_best=None, scale=None, opt_method='default',
            manual_test_set=None, n_iter_optimize=100
    ):
        self.model_type = CLASSIFIER
        super().__init__(
            filename, classname, dataframe, test_set_size=test_set_size,
            seed=seed, subsample_size=subsample_size, n_split=n_split,
            fptype=fptype, output_path=output_path, sub_folder=sub_folder,
            k_best=k_best, scale=scale, manual_test_set=manual_test_set,
            opt_method=opt_method, n_iter_optimize=n_iter_optimize
        )
        self.cv_model = DNNClassifierCVFolds()

    def store_folds_trainings_data(self, training_time_minutes):
        """
        Method to dump (save to file) trained folds model (*.sav or *.h5) files

        :param training_time_minutes: time spent to train model (minutes)
        :type training_time_minutes: float
        """

        # write trained models data to files
        for number, model in enumerate(self.cv_model.models):
            model.training_time = training_time_minutes
            model.model_name = self.model_name

    def train_k_fold_models(
            self, training_function, arguments, optimal_parameters
    ):
        """
        Method for train all folds of chosen model

        :param training_function: model training function
        :param arguments: additional model arguments for "fit"
        :param optimal_parameters: optimal parameters for training
        :type arguments: dict
        """

        fold_number = 1
        for train_ids, valid_ids in self.cv_valid_ids:
            # define using subdataframes

            x_train = self.x_train[train_ids]
            y_train = self.y_train[train_ids]
            x_valid = self.x_train[valid_ids]
            y_valid = self.y_train[valid_ids]

            # change validation data type if used
            if 'validation_data' in arguments.keys():
                arguments['validation_data'] = (
                    x_valid['value'], y_valid['value']
                )

            # set path to save best DNN model of current fold
            path_to_model = os.path.join(
                self.sub_folder, 'TMP_CLASSIFIER_DNN_FOLD_{}.h5'.format(
                    fold_number
                )
            )
            checkpointer = ModelCheckpoint(
                filepath=path_to_model, verbose=1, save_best_only=True)
            arguments['callbacks'].append(checkpointer)

            training_function_copy = self.make_training_function(
                optimal_parameters)

            # copy training method to avoid chain "fit"
            training_function_copy.fit(
                x_train['value'], y_train['value'], **arguments
            )
            model = load_model(
                path_to_model,
                custom_objects={'coeff_determination': coeff_determination}
            )

            # define trained model object
            trained_model = self.cv_model.fold_class(
                model=model, model_type=self.cv_model.model_type,
                x_test=self.x_test, y_test=self.y_test,
                x_train=x_train, y_train=y_train,
                x_valid=x_valid, y_valid=y_valid,
                prediction_target=self.prediction_target,
                sub_folder=self.sub_folder
            )
            trained_model.path_to_model_file = path_to_model
            self.cv_model.models.append(trained_model)

            fold_number += 1
            arguments['callbacks'].remove(checkpointer)

            del training_function_copy
            del model

    def make_sub_folder_path(self):
        """
        Method which make sub folder path for classifier, based on init
        parameters, which will contain all model training data.
        Path will contain datetime and fingerprint data.

        :return: path to sub folder
        :rtype: str
        """

        classifying_folder_path = super().make_sub_folder_path()
        folder_path = 'DNN_{}'.format(classifying_folder_path)

        return folder_path


class DNNRegressor(ClassicRegressor):
    def __init__(
            self, filename, valuename, dataframe, manual_test_set=None,
            test_set_size=0.2, n_split=4, seed=0, fptype=None, scale=None,
            outliers_from_test=None, k_best=None, outliers_from_train=None,
            output_path=None, sub_folder=None, log=False, top_important=None,
            pca=None, var_threshold=None, z_score=None, subsample_size=1.0,
            n_iter_optimize=100, opt_method='default'
    ):
        self.model_type = REGRESSOR
        super().__init__(
            filename, valuename, dataframe, manual_test_set=manual_test_set,
            test_set_size=test_set_size, n_split=n_split, seed=seed, pca=pca,
            fptype=fptype, scale=scale, outliers_from_test=outliers_from_test,
            k_best=k_best, outliers_from_train=outliers_from_train, log=log,
            output_path=output_path, sub_folder=sub_folder, z_score=z_score,
            top_important=top_important, var_threshold=var_threshold,
            subsample_size=subsample_size, opt_method=opt_method,
            n_iter_optimize=n_iter_optimize
        )
        self.cv_model = DNNRegressorCVFolds()

    def store_folds_trainings_data(self, training_time_minutes):
        """
        Method to dump (save to file) trained folds model (*.sav or *.h5) files

        :param training_time_minutes: time spent to train model (minutes)
        :type training_time_minutes: float
        """

        # write trained models data to files
        for number, model in enumerate(self.cv_model.models):
            model.training_time = training_time_minutes
            model.model_name = self.model_name

    def train_k_fold_models(
            self, training_function, arguments, optimal_parameters
    ):
        """
        Method for train all folds of chosen model

        :param training_function: model training function
        :param arguments: additional model arguments for "fit"
        :param optimal_parameters: optimal parameters for training
        :type arguments: dict
        """

        fold_number = 1
        for train_ids, valid_ids in self.cv_valid_ids:
            # define using subdataframes
            x_train = self.x_train[train_ids]
            y_train = self.y_train[train_ids]
            x_valid = self.x_train[valid_ids]
            y_valid = self.y_train[valid_ids]

            # change validation data type if used
            if 'validation_data' in arguments.keys():
                arguments['validation_data'] = (
                    x_valid['value'], y_valid['value']
                )

            # set path to save best DNN model of current fold
            path_to_model = os.path.join(
                self.sub_folder, 'TMP_REGRESSION_DNN_FOLD_{}.h5'.format(
                    fold_number
                )
            )
            checkpointer = ModelCheckpoint(
                filepath=path_to_model, monitor='val_loss', verbose=1,
                save_best_only=True
            )
            arguments['callbacks'].append(checkpointer)

            training_function_copy = self.make_training_function(
                optimal_parameters)

            # copy training method to avoid chain "fit"
            training_function_copy.fit(
                x_train['value'], y_train['value'], **arguments
            )
            model = load_model(
                path_to_model,
                custom_objects={'coeff_determination': coeff_determination}
            )

            # define trained model object
            trained_model = self.cv_model.fold_class(
                model=model, model_type=self.cv_model.model_type,
                x_test=self.x_test, y_test=self.y_test,
                x_train=x_train, y_train=y_train,
                x_valid=x_valid, y_valid=y_valid,
                prediction_target=self.prediction_target,
                sub_folder=self.sub_folder
            )
            trained_model.path_to_model_file = path_to_model
            self.cv_model.models.append(trained_model)

            fold_number += 1
            arguments['callbacks'].remove(checkpointer)

            del training_function_copy
            del model

    def make_sub_folder_path(self):
        """
        Method which make sub folder path for regressor, based on init
        parameters, which will contain all model training data.
        Path will contain datetime, fingerprint, scaled, top important,
        outliers and pca parametrs data.

        :return: path to sub folder
        :rtype: str
        """

        regression_folder_path = super().make_sub_folder_path()
        folder_path = 'DNN_{}'.format(regression_folder_path)

        return folder_path


class ClassicMultiClassifier(Trainer):
    def __init__(
            self, filename, classname, dataframe, test_set_size=0.2, seed=7,
            major_subsample=0.2, n_split=4, fptype=None, output_path=None,
            sub_folder=None, k_best=None, scale=None
    ):

        super().__init__(
            filename, classname, dataframe, output_path=output_path, seed=seed,
            test_set_size=test_set_size, fptypes=fptype, n_split=n_split,
            scale=scale, sub_folder=sub_folder
        )
        self.cv_model = ClassicMultiClassifierCVFolds()

    def post_plots(self, body, oauth):
        """
        Method which make plots, then POST plots to blob storage

        :param body: rabbitmq message body
        :param oauth: OAuth2 object
        """

        # make needed for model plots
        self.plots = self.cv_model.make_plots()

        for k_fold_number, plots in self.plots.items():
            # make ROC plot multipart encoded object
            roc_info = {
                'FileInfo': json.dumps({
                    'modelName': self.cv_model.model_name,
                    'fileType': ROC_PLOT
                }),
                'ParentId': body['CurrentModelId']
            }
            multipart_roc = get_multipart_object(
                body, plots['roc_plot_path'], 'image/png',
                additional_fields=roc_info
            )
            # send, http POST request to blob storage api with ROC plot
            post_data_to_blob(oauth, multipart_roc)
            body['NumberOfGenericFiles'] += 1

            # make confusion matrix multipart encoded object
            confusion_matrix_info = {
                'FileInfo': json.dumps({
                    'modelName': self.cv_model.model_name,
                    'fileType': CONFUSION_MATRIX
                }),
                'ParentId': body['CurrentModelId']
            }
            multipart_cm = get_multipart_object(
                body, plots['cm_plot_path'], 'image/png',
                additional_fields=confusion_matrix_info
            )
            # send, http POST request to blob storage api with confusion matrix
            post_data_to_blob(oauth, multipart_cm)
            body['NumberOfGenericFiles'] += 1

            # update template tags with model's plots
            self.template_tags['confusion_matrixes'].append(
                plots['cm_plot_path'])
            self.template_tags['roc_plots'].append(
                plots['roc_plot_path'])

    def correct_data_features_values(self, k_best):
        """
        another clone.
        :param k_best:
        """
        if k_best:
            test = SelectKBest(score_func=f_classif, k=k_best)
            test.fit(self.df_features, self.parameter_dataframe)
            self.df_features = pandas.DataFrame(
                test.transform(self.df_features))
            self.dataframe = self.df_features.join(
                self.parameter_dataframe, on=None, how='left',
                lsuffix='', rsuffix='', sort=False
            )

    def make_sub_folder_path(self):
        """
        Method which make sub folder path for classifier, based on init
        parameters, which will contain all model training data.
        Path will contain datetime and fingerprint data.

        :return: path to sub folder
        :rtype: str
        """

        # get current datetime value
        date_time = super().make_sub_folder_path()
        # convert used fingerprints parameters to string
        if self.fptypes:
            all_fptypes_parameters = []
            for fptype in self.fptypes:
                fptype_parameters = '_'.join(map(str, list(fptype.values())))
                all_fptypes_parameters.append(fptype_parameters)

            fptype_parameters_as_string = '_'.join(all_fptypes_parameters)
        else:
            fptype_parameters_as_string = ''
        # construct full sub folder path
        folder_path = 'classification_{}_{}_{}'.format(
            self.filename.split('.')[0], date_time,
            fptype_parameters_as_string
        )

        return folder_path

    def update_template_tags(self):
        """
        Method which update basic template tags with keys needed for classifier
        trainer
        """

        self.template_tags['radar_plots'] = []
        self.template_tags['confusion_matrixes'] = []
        self.template_tags['roc_plots'] = []
        self.template_tags['dataset'] += ' for classifier model training'

    def make_report_text(self):
        """
        Method which call report text generator for classifier.
        Generator will update classifier template tags with needed for
        training report parameters
        """

        report_text_classifier(self)


class ClassicMultiRegressor(Trainer):
    def __init__(
            self, filename, valuename, dataframe, manual_test_set=None,
            test_set_size=0.2, n_split=4, seed=0, fptype=None, scale=None,
            outliers_from_test=None, k_best=None, outliers_from_train=None,
            output_path=None, sub_folder=None, log=False, top_important=None,
            pca=None, var_threshold=None, z_score=None
    ):

        self.top_important = top_important
        self.outliers_from_test = outliers_from_test
        self.pca = pca
        super().__init__(
            filename, valuename, dataframe, output_path=output_path, seed=seed,
            test_set_size=test_set_size, fptypes=fptype, n_split=n_split,
            scale=scale, sub_folder=sub_folder
        )
        self.cv_model = ClassicMultiRegressorCVFolds()
        self.model_type = MULTI_REGRESSOR
        self.model_algorithm = CLASSIC
        self.manual_test_set = manual_test_set
        self.threshold = var_threshold
        self.log = log

        self.sub_folder = self.make_sub_folder(sub_folder, output_path)
        make_directory(self.sub_folder)

        self.update_template_tags()
        # self.correct_dataframe_with_z_score(z_score)
        self.sub_dataframe_value = numpy.vstack(
            (self.dataframe[self.prediction_target],)
        ).T
        self.bins = dataframe.shape[1] - len(valuename)
        self.split_dataframes()
        self.csv_formatter = '{:.4E}'.format

        self.apply_cross_validation()

        self.path_to_scaler = self.apply_scaler()

    def correct_by_log(self):
        """
        another clone.
        """
        if self.log:
            self.y_train = numpy.log(self.y_train)
            self.y_test = numpy.log(self.y_test)

    def correct_by_top_important(self):
        """
        another clone
        """
        if not self.top_important:
            return

        model = ExtraTreesRegressor()
        model.fit(self.x_train, self.y_train)
        rating_dict = {}
        feat_imp = model.feature_importances_
        for i in range(0, len(feat_imp)):
            rating_dict[i] = feat_imp[i]
        rating_list = sorted(rating_dict.items(), key=operator.itemgetter(1))
        rating_list.reverse()
        indices_list = [x[0] for x in rating_list[:self.top_important]]
        self.x_train = self.x_train[:, indices_list]
        if self.test_size != 0:
            self.x_test = self.x_test[:, indices_list]
        self.bins = self.top_important
        joblib.dump(
            model, '{}/{}_top_important.sav'.format(
                self.sub_folder, self.filename.split('.')[0]
            )
        )

    def correct_data_features_values(self, k_best):
        """
        clone
        """
        # TODO MAKE DOCSTRINGS THERE
        if k_best:
            model = SelectKBest(score_func=f_regression, k=k_best)
            model = model.fit(self.x_train, self.y_train)
            self.x_train = model.transform(self.x_train)
            self.x_test = model.transform(self.x_test)
            self.bins = k_best

    def apply_outliers(self, outlier, x_attribute, y_attribute):
        """
        another clone
        :param outlier:
        :param x_attribute:
        :param y_attribute:
        """
        if not outlier:
            return

        iso_for = IsolationForest().fit(self.x_train)
        outliers_train = []
        x_value = getattr(self, x_attribute)
        y_value = getattr(self, y_attribute)

        for i in range(x_value.shape[0]):
            if iso_for.predict(x_value[i, :].reshape(1, -1)) == -1:
                outliers_train.append(i)

        x_value = numpy.delete(x_value, outliers_train, axis=0)
        y_value = numpy.delete(y_value, outliers_train, axis=0)
        setattr(self, x_attribute, x_value)
        setattr(self, y_attribute, y_value)

    def split_dataframes(self):
        """
        """
        # TODO MAKE DOCSTRINGS THERE

        if self.test_size == 0:
            self.x_train = pandas.DataFrame(self.dataframe.ix[:, 0:(self.bins)])#.as_matrix())
            self.y_train = pandas.DataFrame(self.sub_dataframe_value)

        else:
            self.x_train, self.x_test, self.y_train, self.y_test = model_selection.train_test_split(
                self.dataframe.ix[:, 0:(self.bins)],
                self.dataframe[self.prediction_target],
                train_size=self.train_size,
                test_size=self.test_size,
                random_state=self.seed,
                shuffle=True
            )
            # self.y_train = numpy.vstack(
            #     (yx_train_new[self.prediction_target],)
            # ).T
            # self.x_train = yx_train_new.ix[:, :self.bins].as_matrix()
            # self.y_test = numpy.vstack(
            #     (yx_test_new[self.prediction_target],)
            # ).T
            # self.x_test = yx_test_new.ix[:, :self.bins].as_matrix()

        if self.manual_test_set is not None:
            self.manual_test_set = self.manual_test_set.apply(
                pandas.to_numeric, errors='coerce'
            ).dropna(axis=0, how='any').reset_index(drop=True)
            self.y_train = pandas.DataFrame(self.sub_dataframe_value)
            self.x_train = pandas.DataFrame(self.dataframe.ix[:, 0:(self.bins)])# .as_matrix()
            self.y_test = pandas.DataFrame(numpy.vstack(
                (self.manual_test_set[self.prediction_target],)
            ).T)
            self.x_test = pandas.DataFrame(self.manual_test_set.ix[:, :self.bins])# .as_matrix()
            self.test_size = 1

    def make_sub_folder_path(self):
        """
        Method which make sub folder path for regressor, based on init
        parameters, which will contain all model training data.
        Path will contain datetime, fingerprint, scaled, top important,
        outliers and pca parametrs data.

        :return: path to sub folder
        :rtype: str
        """

        # get current datetime value
        date_time = super().make_sub_folder_path()
        # convert used fingerprints parameters to string
        if self.fptypes:
            all_fptypes_parameters = []
            for fptype in self.fptypes:
                fptype_parameters = '_'.join(map(str, list(fptype.values())))
                all_fptypes_parameters.append(fptype_parameters)

            fptype_parameters_string = '_'.join(all_fptypes_parameters)
        else:
            fptype_parameters_string = ''
        # convert other used parameters to string
        scaled = 'scaled' if self.scale else ''
        top = 'top_{}'.format(self.top_important) if self.top_important else ''
        outliers = 'outliers' if self.outliers_from_test else ''
        pca = 'pca_{}'.format(self.pca) if self.pca else ''
        # construct full sub folder path
        folder_path = 'regression_{}_{}_{}_{}_{}_{}_{}'.format(
            self.filename.split('.')[0], date_time, fptype_parameters_string,
            scaled, top, outliers, pca
        )

        return folder_path

    def correct_dataframe_with_z_score(self, z_score):
        """
        and yet another clone
        """
        # TODO MAKE DOCSTRING THERE
        if not z_score:
            return

        valuename = self.prediction_target
        mean_value = self.dataframe[valuename].mean()
        numerator = self.dataframe[valuename] - mean_value
        denominator = self.dataframe[valuename].std(ddof=0)
        drop_index = self.dataframe[numerator / denominator >= z_score].index
        self.dataframe = self.dataframe.drop(drop_index)

    def update_template_tags(self):
        """
        Method which update basic template tags with keys needed for regressor
        trainer
        """

        self.template_tags['distribution_plots'] = []
        self.template_tags['dataset'] += ' for regression model training'
        self.template_tags['regression_result_test'] = []
        self.template_tags['regression_result_train'] = []

    def make_report_text(self):
        """
        Method which call report text generator for regressor.
        Generator will update regressor template tags with needed for
        training report parameters
        """

        report_text_regressor(self)

    def post_plots(self, body, oauth):
        """
        Method which make plots for model, then POST plots to blob storage

        :param body: rabbitmq message body
        :param oauth: OAuth2 object
        """

        # make needed for model plots
        self.plots = self.cv_model.make_plots()
        for k_fold_number, plots in self.plots.items():
            # make test regression results plot multipart encoded object
            regression_results_test_info = {
                'FileInfo': json.dumps({
                    'modelName': self.cv_model.model_name,
                    'fileType': REGRESSION_RESULT_TEST
                }),
                'ParentId': body['CurrentModelId']
            }
            regression_results_test = get_multipart_object(
                body, plots['regression_results_test'], 'image/png',
                additional_fields=regression_results_test_info
            )
            # POST to blob storage api with test regression results
            post_data_to_blob(oauth, regression_results_test)
            body['NumberOfGenericFiles'] += 1

            # make train regression results plot multipart encoded object
            regression_results_train_info = {
                'FileInfo': json.dumps({
                    'modelName': self.cv_model.model_name,
                    'fileType': REGRESSION_RESULT_TRAIN
                }),
                'ParentId': body['CurrentModelId']
            }
            regression_results_train = get_multipart_object(
                body, plots['regression_results_train'], 'image/png',
                additional_fields=regression_results_train_info
            )
            # POST to blob storage api with train regression results
            post_data_to_blob(oauth, regression_results_train)
            body['NumberOfGenericFiles'] += 1

            # update template tags with model's plots
            self.template_tags['regression_result_test'].append(
                plots['regression_results_test'])
            self.template_tags['regression_result_train'].append(
                plots['regression_results_train'])


class DNNMultiClassifier(ClassicMultiClassifier):
    def __init__(
            self, filename, classname, dataframe, test_set_size=0.2, seed=7,
            major_subsample=0.2, n_split=4, fptype=None, output_path=None,
            sub_folder=None, k_best=None, scale=None, x_valid=None,
            y_valid=None
    ):
        self.model_type = MULTI_CLASSIFIER
        super().__init__(
            filename, classname, dataframe, test_set_size=test_set_size,
            seed=seed, major_subsample=major_subsample, n_split=n_split,
            fptype=fptype, output_path=output_path, sub_folder=sub_folder,
            k_best=k_best, scale=scale
        )
        self.cv_model = DNNMultiClassifierCVFolds()
        self.model_algorithm = DNN
        self.x_valid = x_valid
        self.y_valid = y_valid

    def store_folds_trainings_data(self, model_name, training_time_minutes):
        for number, model in enumerate(self.cv_model.models):
            model.training_time = training_time_minutes
            model.model_name = model_name

    def train_folds_models(self, model_name):
        # get model name from algorithm code
        self.cv_model.model_name = model_name
        self.cv_model.model_type = model_type_by_name(model_name)
        # train model with timer
        start_training = time()

        optimal_parameters = ALGORITHM[OPTIMIZER_FUNCTION][model_name](self)
        optimal_parameters['num_labels'] = onehot_encoded(
            self.y_train['value']).shape[1]
        training_function = ALGORITHM[TRAIN_FUNCTION][model_name](
            optimal_parameters)
        arguments = ALGORITHM[ADDITIONAL_ARGUMENTS][model_name](self)
        self.train_k_fold_models(
            training_function, arguments, optimal_parameters)

        return (time() - start_training) / 60

    def train_k_fold_models(
            self, training_function, arguments, optimal_parameters
    ):
        """
        Method for train all folds of chosen model

        :param training_function: model training function
        :param arguments: additional model arguments for "fit"
        :param optimal_parameters: optimal parameters for training
        """

        fold_number = 1
        optimal_parameters = ALGORITHM[OPTIMIZER_FUNCTION][
            self.cv_model.model_name](self)
        optimal_parameters['num_labels'] = onehot_encoded(
            self.y_train['value']).shape[1]
        for train_ids, valid_ids in self.cv_valid_ids:
            # define using subdataframes
            x_train = self.x_train[train_ids]
            y_train = self.y_train[train_ids]
            x_valid = self.x_train[valid_ids]
            y_valid = self.y_train[valid_ids]

            # change validation data type if used
            if 'validation_data' in arguments.keys():
                arguments['validation_data'] = (
                    x_valid['value'], onehot_encoded(y_valid['value'])
                )

            # set path to save best DNN model of current fold
            path_to_model = os.path.join(
                self.sub_folder, 'TMP_CLASSIFIER_DNN_FOLD_{}.h5'.format(
                    fold_number
                )
            )
            checkpointer = ModelCheckpoint(
                filepath=path_to_model, verbose=1, save_best_only=True)
            arguments['callbacks'].append(checkpointer)

            training_function_copy = train_dnn_multi_valid(optimal_parameters)

            # copy training method to avoid chain "fit"
            training_function_copy.fit(
                x_train['value'], onehot_encoded(y_train['value']), **arguments
            )
            model = load_model(
                path_to_model,
                custom_objects={'coeff_determination': coeff_determination}
            )

            # define trained model object
            trained_model = self.cv_model.fold_class(
                model=model, model_type=self.cv_model.model_type,
                x_test=self.x_test, x_train=x_train, x_valid=x_valid,
                y_test=onehot_encoded(self.y_test['value']),
                y_train=onehot_encoded(y_train['value']),
                y_valid=onehot_encoded(y_valid['value']),
                prediction_target=self.prediction_target,
                sub_folder=self.sub_folder
            )
            trained_model.path_to_model_file = path_to_model
            self.cv_model.models.append(trained_model)

            fold_number += 1
            arguments['callbacks'].remove(checkpointer)

            del training_function_copy
            del model

    def make_sub_folder_path(self):
        """
        Method which make sub folder path for classifier, based on init
        parameters, which will contain all model training data.
        Path will contain datetime and fingerprint data.

        :return: path to sub folder
        :rtype: str
        """

        classifying_folder_path = super().make_sub_folder_path()
        folder_path = 'DNN_{}'.format(classifying_folder_path)

        return folder_path


class DNNMultiRegressor(ClassicMultiRegressor):
    def __init__(
            self, filename, valuename, dataframe, manual_test_set=None,
            test_set_size=0.2, n_split=4, seed=0, fptype=None, scale=None,
            outliers_from_test=None, k_best=None, outliers_from_train=None,
            output_path=None, sub_folder=None, log=False, top_important=None,
            pca=None, var_threshold=None, z_score=None
    ):

        super().__init__(
            filename, valuename, dataframe, manual_test_set=manual_test_set,
            test_set_size=test_set_size, n_split=n_split, seed=seed, pca=pca,
            fptype=fptype, scale=scale, outliers_from_test=outliers_from_test,
            k_best=k_best, outliers_from_train=outliers_from_train, log=log,
            output_path=output_path, sub_folder=sub_folder, z_score=z_score,
            top_important=top_important, var_threshold=var_threshold
        )
        self.cv_model = DNNMultiRegressorCVFolds()
        self.model_algorithm = DNN

    def train_model(self, algorithm_code):
        """
        Basic train model method. Train model with given parameters by
        algorithm code, check training time int minutes, save model to file.
        Returns training metrics such as model object, path to model file,
        training time and trained model name
        Expand it in child class

        :param algorithm_code: code of algorithm
        :return: basic model training metrics
        :rtype: dict
        """

        # get model name from algorithm code
        model_name = algorithm_name_by_code(algorithm_code)
        self.cv_model.model_name = model_name
        self.cv_model.model_type = model_type_by_name(model_name)
        # train model with timer
        start_training = time()

        optimal_parameters = ALGORITHM[OPTIMIZER_FUNCTION][model_name](self)
        optimal_parameters['num_labels'] = self.y_train.values.shape[1]
        training_function = ALGORITHM[TRAIN_FUNCTION][model_name](
            optimal_parameters)
        arguments = ALGORITHM[ADDITIONAL_ARGUMENTS][model_name](self)
        self.train_k_fold_models(training_function, arguments)

        training_time_minutes = (time() - start_training) / 60
        # write trained models data to files
        for number, model in enumerate(self.cv_model.models):
            model.training_time = training_time_minutes
            model.model_name = model_name

        # define CV model attributes
        self.cv_model.prediction_target = self.prediction_target
        self.cv_model.make_dataframes(self)
        self.cv_model.sub_folder = self.sub_folder
        # write classification metrics
        self.metrics[model_name] = self.cv_model.calculate_metrics()

        return self.cv_model

    def train_k_fold_models(self, training_function, arguments):
        """
        Method for train all folds of chosen model

        :param training_function: model training function
        :param arguments: additional model arguments for "fit"
        """

        fold_number = 1
        for train_ids, valid_ids in self.cv_valid_ids:
            # define using subdataframes
            x_train = self.x_train.iloc[train_ids]
            y_train = self.y_train.iloc[train_ids]
            x_valid = self.x_train.iloc[valid_ids]
            y_valid = self.y_train.iloc[valid_ids]

            # change validation data type if used
            if 'validation_data' in arguments.keys():
                arguments['validation_data'] = (
                    x_valid.as_matrix(), y_valid.as_matrix()
                )

            # set path to save best DNN model of current fold
            path_to_model = os.path.join(
                self.sub_folder, 'TMP_REGRESSION_DNN_FOLD_{}.h5'.format(
                    fold_number
                )
            )
            checkpointer = ModelCheckpoint(
                filepath=path_to_model, monitor='val_loss', verbose=1,
                save_best_only=True
            )
            arguments['callbacks'].append(checkpointer)

            # copy training method to avoid chain "fit"
            copy.copy(training_function).fit(
                x_train.as_matrix(), y_train.as_matrix(), **arguments
            )
            model = load_model(
                path_to_model,
                custom_objects={'coeff_determination': coeff_determination}
            )

            # define trained model object
            trained_model = self.cv_model.fold_class(
                model=model, model_type=self.cv_model.model_type,
                x_test=self.x_test, y_test=self.y_test,
                x_train=x_train, y_train=y_train,
                x_valid=x_valid, y_valid=y_valid,
                prediction_target=self.prediction_target,
                sub_folder=self.sub_folder
            )
            trained_model.path_to_model_file = path_to_model
            self.cv_model.models.append(trained_model)

            fold_number += 1
            arguments['callbacks'].remove(checkpointer)

    def make_sub_folder_path(self):
        """
        Method which make sub folder path for regressor, based on init
        parameters, which will contain all model training data.
        Path will contain datetime, fingerprint, scaled, top important,
        outliers and pca parametrs data.

        :return: path to sub folder
        :rtype: str
        """

        regression_folder_path = super().make_sub_folder_path()
        folder_path = 'DNN_{}'.format(regression_folder_path)

        return folder_path
