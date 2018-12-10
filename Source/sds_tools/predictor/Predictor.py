"""
Module with predictor class which can predict properties and create prediction
csv file
"""

import os
import time

import numpy
import shap
from rdkit import Chem

from MLLogger import BaseMLLogger
from general_helper import make_directory, numpy_to_csv, get_distance
from learner.algorithms import CLASSIFIER, REGRESSOR

from processor import sdf_to_csv

try:
    BLOB_URL = '{}/blobs'.format(os.environ['OSDR_BLOB_SERVICE_URL'])
except KeyError:
    BLOB_URL = ''

TEMP_FOLDER = os.environ['OSDR_TEMP_FILES_FOLDER']
# create temporary folder if it not exist
make_directory(TEMP_FOLDER)
LOGGER = BaseMLLogger(
    log_name='predictor_logger', log_file_name='predictor-logger')


class MLPredictor(object):
    def __init__(self, parameters, dataframe=None):
        """
        Creation predictor object from given parameters (user input) and logger
        (if defined)

        :param parameters: user input with prediction parameters
        :param dataframe: prepared dataframe
        """

        self.dataset_file_name = parameters['DatasetFileName']
        self.primary_field = parameters['ClassName']
        self.fptype = parameters['Fingerprints']
        self.scaler = parameters['Models']['scaler']
        self.model_type = parameters['ModelType']
        self.molecules = parameters['Molecules']
        start_timer = time.time()
        self.dataframe = self.make_dataframe(dataframe)
        LOGGER.info('MAKING DATAFRAME: {} sec'.format(time.time() - start_timer))
        self.dataframe_size = self.dataframe.shape[1]
        self.models = parameters['Models']
        self.prediction = numpy.ndarray(
            (len(self.molecules), 6),
            dtype=[('name', 'U30'), ('molecule_number', 'i4'), ('value', 'U40')]
        )
        self.molecules_properties = list()
        self.molecules_distances = list()
        self.molecules_densities = list()
        self.molecules_shap_values = list()
        self.csv_formatter = self.make_formatter()
        self.density_model = parameters['Models']['density_model']
        self.distance_matrix = parameters['Models']['distance_matrix']
        self.density_mean = float(parameters['DensityMean'])
        self.density_std = float(parameters['DensityStd'])
        self.distance_mean = float(parameters['DistanceMean'])
        self.distance_std = float(parameters['DistanceStd'])
        self.train_mean = parameters['Models']['train_mean']
        self.train_shape = parameters['TrainShape']
        self.modi = parameters['Modi']
        self.k_means = parameters['Models']['k_means']
        self.model_code = parameters['ModelCode']

    def make_prediction(self):
        """
        Method which make property prediction for dataset with trained models.
        If you use more than one model results will be average of all models.
        Append predicted properties to molecules_properties list.
        """

        # make name of column with predicted values
        if self.model_type == CLASSIFIER:
            predict_column_name = 'Probability of {}'.format(
                self.primary_field)
        elif self.model_type == REGRESSOR:
            predict_column_name = self.primary_field
        else:
            # throw exception if model type unknown
            raise TypeError(
                'Unknown model type: {}'.format(self.model_type))

        prediction_data = dict()
        # loop by all trained models used for prediction
        for model_number in range(0, self.models['models_number']):
            start_timer = time.time()
            LOGGER.info('LOAD MODEL FOLD: {} sec'.format(
                time.time() - start_timer))

            # loop by all molecules in dataset
            with self.models['sessions'][model_number].as_default():
                for molecule_number in range(0, len(self.molecules)):
                    # get x values from dataset features
                    start_timer = time.time()
                    molecule_x_values = self.get_x_values(molecule_number)
                    LOGGER.info('MOLECULE X VALUES: {} sec'.format(
                        time.time() - start_timer))
                    # check descriptor created
                    if not isinstance(molecule_x_values, numpy.ndarray):
                        continue
                    start_timer = time.time()
                    with self.models['graphs'][model_number].as_default():
                        # regression modelling
                        if self.model_type == REGRESSOR:
                            # prediction function different for dnn and classic
                            prediction_function = get_regression_function(
                                self.models['models'][model_number],
                                self.model_code
                            )

                            # value which predicted for current molecule
                            # with using current model
                            predicted_value = prediction_function(
                                molecule_x_values)[0]

                            # except case when have array in prediction
                            if isinstance(predicted_value, numpy.ndarray):
                                predicted_value = predicted_value[0]

                        # classifying modelling
                        elif self.model_type == CLASSIFIER:
                            # prediction function different for dnn and classic
                            prediction_function = get_classifier_function(
                                self.models['models'][model_number],
                                self.models['models_names'][model_number]
                            )
                            # TODO magic there
                            if '.h5' in self.models['models_names'][model_number]:
                                prediction_value_index = 0
                            else:
                                prediction_value_index = 1
                            # value which predicted for current molecule
                            # with using current model
                            predicted_value = prediction_function(
                                molecule_x_values)[0][prediction_value_index]
                        else:
                            # throw exception if model type unknown
                            raise TypeError(
                                'Unknown model type: {}'.format(self.model_type))
                        if self.k_means:
                            explainer = shap.KernelExplainer(
                                prediction_function, self.k_means)
                            shap_values = explainer.shap_values(
                                molecule_x_values, nsamples=100)[0]
                        else:
                            shap_values = []

                    LOGGER.info('PREDICTION: {} sec'.format(
                        time.time() - start_timer))
                    if molecule_number not in prediction_data.keys():
                        prediction_data[molecule_number] = {
                            'property': list(),
                            'distance': None,
                            'density': None,
                            'shap_values': list()
                        }

                    prediction_data[molecule_number]['property'].append(
                        predicted_value)
                    prediction_data[molecule_number]['shap_values'].append(
                        shap_values)

                    start_timer = time.time()
                    if ((prediction_data[molecule_number]['distance'] is None or
                        prediction_data[molecule_number]['density'] is None)
                        and
                            (self.density_model and self.distance_matrix)
                    ):
                        distance, density = self.check_mol(molecule_x_values)
                        prediction_data[molecule_number]['distance'] = distance
                        prediction_data[molecule_number]['density'] = density

                    LOGGER.info('AD CALCULATION: {} sec'.format(
                        time.time() - start_timer))

        start_timer = time.time()
        # loop by all molecules in dataset
        for molecule_number in range(0, len(self.molecules)):
            if molecule_number in prediction_data.keys():
                mean_modelled_property = get_mean_value(
                    prediction_data[molecule_number]['property'])
                mean_distance = prediction_data[molecule_number]['distance']
                mean_density = prediction_data[molecule_number]['density']
                mean_shap_values = numpy.mean(
                    prediction_data[molecule_number]['shap_values'], axis=0)
            else:
                mean_modelled_property = None
                mean_distance = None
                mean_density = None
                mean_shap_values = None

            # if mean value == None, then something wrong with molecule
            if mean_modelled_property is None:
                mean_modelled_property = 'Cant_predict_property'
            else:
                mean_modelled_property = self.csv_formatter(
                    mean_modelled_property)

            # if mean value == None, then something wrong with molecule
            if mean_distance is None:
                mean_distance = 'Cant_calculate_distance'
            else:
                if mean_distance > 0.5:
                    mean_distance = 'Inside'
                else:
                    mean_distance = 'Outside'

            # if mean value == None, then something wrong with molecule
            if mean_density is None:
                mean_density = 'Cant_calculate_density'
            else:
                if mean_density > 0.5:
                    mean_density = 'Inside'
                else:
                    mean_density = 'Outside'

            self.molecules_properties.append((
                predict_column_name, molecule_number, mean_modelled_property
            ))
            self.molecules_distances.append((
                'Applicability domain distance', molecule_number, mean_distance
            ))
            self.molecules_densities.append((
                'Applicability domain density', molecule_number, mean_density
            ))
            self.molecules_shap_values.append((
                'Shap values', molecule_number, mean_shap_values
            ))

        LOGGER.info('DUMMY SORTING: {} sec'.format(
            time.time() - start_timer))
        start_timer = time.time()
        self.store_prediction()
        LOGGER.info('STORE PREDICTION: {} sec'.format(
            time.time() - start_timer))

    def get_x_values(self, molecule_number):
        """
        Method which take x vector for molecule in dataframe by molecule number

        :param molecule_number: molecule number in dataframe
        :type molecule_number: int
        :return: x vector for molecule or None if molecule not in dataframe
        """

        try:
            molecule_x_values = self.dataframe[molecule_number].reshape(1, -1)
        except KeyError:
            molecule_x_values = None

        return molecule_x_values

    def check_mol(self, x_predict):
        """

        :param x_predict:
        :return:
        """
        # TODO make docstring there
        dens_mean = self.density_mean
        dens_std = self.density_std
        distance_matrix = self.distance_matrix
        dist_mean = self.distance_mean
        dist_std = self.distance_std
        dens_model = self.density_model
        dist = get_distance(
            x_predict[0], distance_matrix, self.train_mean, self.train_shape)
        if dist > dist_mean + 3*dist_std:
            distance = 0
        else:
            distance = 1

        dens = abs(dens_model.score_samples(x_predict)[0])
        if dens < dens_mean - 3*dens_std:
            density = 0
        else:
            density = 1

        return distance, density

    def store_prediction(self):
        """
        Method which add needed parameters to prediction object and
        write it to csv file which path contain dataset filename and datetime

        :return: file path to created csv file
        :rtype: str
        """

        # fill column with Id
        # self.prediction['Id'] = list(range(0, len(self.molecules)))
        # fill column with SMILES
        smiles = list()
        modis = list()
        ids = list()
        for molecule_number, molecule in enumerate(self.molecules):
            smile = Chem.MolToSmiles(molecule, isomericSmiles=True)
            if not smile:
                smile = 'Bad_SMILES'
            smiles.append(('Compound SMILES', molecule_number, smile))
            modis.append(('MODI', molecule_number, self.modi))
            ids.append(('Id', molecule_number, molecule_number))

        self.prediction[:, 0] = ids
        self.prediction[:, 1] = smiles
        self.prediction[:, 2] = self.molecules_properties
        self.prediction[:, 3] = self.molecules_distances
        self.prediction[:, 4] = self.molecules_densities
        self.prediction[:, 5] = modis

    def write_prediction_to_csv(self):
        """
        Method to write stored prediction to csv file

        :return: path to csv file with prediction
        :rtype: str
        """

        # save prediction results to csv file
        dataset_file_name = self.get_dataset_filename()
        csv_path = '{}/{}_{}_prediction.csv'.format(
            TEMP_FOLDER, dataset_file_name,
            time.strftime('%Y_%m_%d__%H_%M_%S', time.gmtime())
        )

        numpy_to_csv(self.prediction, csv_path)

        return csv_path

    def get_dataset_filename(self):
        """
        Method which get dataset filename from dataset full path

        :return: dataset filename
        :rtype: str
        """

        return self.dataset_file_name.split('/')[-1]

    def make_dataframe(self, dataframe):
        """
        Method which make ndarray dataframe from molecules dataset

        :return: ndarray dataframe with dataset molecules
        """

        # make initial dataframe
        if dataframe is None:
            dataframe = sdf_to_csv(
                self.dataset_file_name, self.fptype, molecules=self.molecules
            )

        rows_to_delete = numpy.where(
            numpy.isnan(dataframe['value']).any(axis=1))
        dataframe = numpy.delete(dataframe, rows_to_delete, axis=0)

        for index in sorted(rows_to_delete[0], reverse=True):
            LOGGER.info(index)
            del self.molecules[index]

        # apple scaler to dataframe if it used on training
        if self.scaler:
            dataframe = self.scaler.transform(dataframe['value'])
        else:
            dataframe = dataframe["value"]
        return dataframe

    def make_formatter(self):
        """
        Method for make string formatter for predicted values.
        Formatter depends on training type

        :return: formatter object for predictor by type of trained model
        """

        # for trained classification model
        if self.model_type == CLASSIFIER:
            formatter = '{:.04f}'.format
        # for trained regression model
        elif self.model_type == REGRESSOR:
            formatter = '{:.4E}'.format
        else:
            # throw exception if model type unknown
            raise TypeError(
                'Unknown model type: {}'.format(self.model_type))

        return formatter


def get_mean_value(predicted_values):
    """
    Method for calculate mean prediction value by all predicted value

    :param predicted_values: all predicted values
    :type predicted_values: list
    :return: mean prediction value
    :rtype: float
    """

    if len(predicted_values) == 0:
        return None

    sum_value = 0
    for value in predicted_values:
        sum_value += value

    return sum_value / len(predicted_values)


def get_classifier_function(model, model_name):
    """
    Method which return classifying prediction function.
    It different for classic and DNN models

    :param model: trained classifying object
    :param model_name: name of trained classifying model file
    :type model_name: str
    :return: prediction function for trained classifying model
    """

    # DNN model
    if '.h5' in model_name:
        prediction_function = model.predict
    # classic model
    else:
        prediction_function = model.predict_proba

    return prediction_function


def get_regression_function(model, model_code):
    """
    Method which return prediction function for trained regression model

    :param model: trained model object
    :return: regression predictor function
    """

    return model.predict
