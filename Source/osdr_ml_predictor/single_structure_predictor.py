import os
import uuid
from time import time

import requests

from MLLogger import BaseMLLogger
from exception_handler import MLExceptionHandler
from general_helper import (
    get_model_info, get_oauth, fetch_token, prepare_prediction_parameters,
    molecules_from_mol_strings, logging_exception_message, cache_model,
    molecules_from_smiles, prepare_prediction_files, MODELS_IN_MEMORY_CACHE,
    clear_models_folder
)
from learner.algorithms import algorithm_code_by_name
from mass_transit.MTMessageProcessor import PureConsumer, PurePublisher
from mass_transit.mass_transit_constants import (
    PREDICT_SINGLE_STRUCTURE, SINGLE_STRUCTURE_PREDICTED
)
from messages import single_structure_property_predicted
from predictor.Predictor import MLPredictor

API_MODELS_ENTITIES_URL = os.environ['API_MODELS_ENTITIES_URL']
LOGGER = BaseMLLogger(
    log_name='logger', log_file_name='sds-ml-single-structure-predictor')
os.environ['OAUTHLIB_INSECURE_TRANSPORT'] = '1'
# TODO make it better! move to other place (general_helper.py?)
# set hardcoded unitless property units
# using to remove units if it unitless
UNITLESS = 'Unitless'


@MLExceptionHandler(logger=LOGGER)
def callback(body):
    """
    Pika callback function used by single structure predictor.
    Make list of json with prediction data for each model prediction.

    :param body: RabbitMQ MT message's body
    """

    # start total prediction time counter
    start_prediction_total = time()
    models_data = []
    oauth = get_oauth()
    # try to reformat molecule, if using "\\" instead of "\" in mol string
    if '\\n' in body['Structure']:
        body['Structure'] = body['Structure'].replace('\\n', '\n')

    # set molecules converter function depends on input structure type
    molecules_converter_method = None
    if body['Format'] == 'MOL':
        molecules_converter_method = molecules_from_mol_strings
    elif body['Format'] == 'SMILES':
        molecules_converter_method = molecules_from_smiles

    # read molecules from input mol string or SMILES
    molecules = None
    exception_message = None
    try:
        molecules = molecules_converter_method([body['Structure']])
    except:
        # log error traceback
        logging_exception_message(LOGGER)
        exception_message = 'Get molecule from molstring exception'
    # get models ids, models blob ids, models buckets from input message
    fetch_token(oauth)
    model_id, models_blob_ids, models_buckets = get_models_from_body_message(
        body)
    # make prediction for all models
    for model_id, model_blob_id, model_bucket in zip(model_id, models_blob_ids, models_buckets):
        # start current prediction counter
        start_current_prediction = time()

        start_timer = time()
        # define exception message for current prediction
        if not exception_message:
            exception_message = None

        # initialize prediction parameters
        prediction_parameters = dict()
        prediction_parameters['DatasetFileName'] = 'Single Structure Predict'
        prediction_parameters['Molecules'] = molecules

        # get current model info
        fetch_token(oauth)
        blob_model_info = get_model_info(oauth, model_blob_id, model_bucket)
        blob_model_info['ModelBlobId'] = model_blob_id
        blob_model_info['ModelBucket'] = model_bucket
        blob_model_info['ModelCode'] = algorithm_code_by_name(
            blob_model_info['Method'])

        # add prediction data to json
        # add training parameters
        # add dataset parameters
        predicted_data = {
            'id': model_id,
            'trainingParameters': training_parameters(blob_model_info),
            'property': property_parameters(blob_model_info),
            'dataset': dataset(blob_model_info),
            'reportId': str(uuid.uuid1())
        }

        fetch_token(oauth)
        LOGGER.info('INITIAL PREPARING: {} sec'.format(time() - start_timer))
        try:
            # update prediction parameters
            prepare_prediction_parameters(
                oauth, prediction_parameters, blob_model_info)
        except:
            # log error traceback
            logging_exception_message(LOGGER)
            if not exception_message:
                exception_message = 'Prepare prediction parameters exception'

        start_timer = time()
        try:
            # make prediction
            ml_predictor = MLPredictor(prediction_parameters)
            ml_predictor.make_prediction()
            prediction = ml_predictor.prediction
        except:
            # log error traceback
            logging_exception_message(LOGGER)
            if not exception_message:
                exception_message = 'Predictor exception'
        LOGGER.info('PREDICTION TIME: {} sec'.format(time() - start_timer))

        # add error to json if something wrong
        if exception_message:
            predicted_data['error'] = error(exception_message)
        else:
            predicted_data['result'] = result(prediction)
            predicted_data['applicabilityDomain'] = applicability_domain(
                prediction)

        # stop current prediction counter
        current_prediction_time_seconds = time() - start_current_prediction
        predicted_data['predictionElapsedTime'] = int(
            current_prediction_time_seconds * 1000)

        models_data.append(predicted_data)

    # stop total predictions counter
    total_prediction_time_seconds = time() - start_prediction_total
    body['Data'] = {
        'predictionElapsedTime': int(total_prediction_time_seconds * 1000),
        'models': models_data
    }
    # add prediction consensus data to sending message
    consensus_data = consensus(models_data)
    if consensus_data:
        body['Data'].update({'consensus': consensus_data})

    # send (publish) properties predicted message to OSDR
    prediction_created_event = single_structure_property_predicted(body)
    properties_predicted_publisher = PurePublisher(
        SINGLE_STRUCTURE_PREDICTED)
    properties_predicted_publisher.publish(prediction_created_event)

    return None


def consensus(models_data):
    """
    Define consensus results for single structure predictor.
    Set consensus value, mean average value of all predictions
    Set consensus units, same as first model units using for prediction,
    or remove units key

    :param models_data: all predicted property models data
    :type models_data: list
    :return: consensus value and units
    :rtype: dict
    """

    consensus_value = 0
    counter = 0
    for model_data in models_data:
        if 'result' in model_data.keys():
            consensus_value += float(model_data['result']['value'])
            counter += 1

    if counter != 0:
        consensus_value = consensus_value / counter
    else:
        return None

    consensus_data = {
        'value': str(consensus_value),
    }

    units = correct_units(models_data[0]['property']['units'])
    if units:
        consensus_data.update({'units': units})

    return consensus_data


def preload_ssp_models():
    """
    Method to preload all SSP models to memory, model should have SSP target.
    Using to spped up predictions later
    Add preloaded SSP models to MODELS_IN_MEMORY_CACHE general_helper.py module
    """

    oauth = get_oauth()

    # set default GET request header and parameters values
    headers = {
        'Accept': 'application/json'
    }
    params = (
        ('$filter', 'Targets eq \'SSP\''),
        ('PageNumber', '1'),
        ('PageSize', '100'),
    )

    # GET all SSP models information from OSDR web api
    response = requests.get(
        API_MODELS_ENTITIES_URL, headers=headers, params=params, verify=False)

    # loop to preload all SSP models
    for model_data in response.json():
        fetch_token(oauth)
        # get model blob id and bucket from response
        model_blob_id = model_data['blob']['id']
        model_bucket = model_data['blob']['bucket']
        # get model info
        blob_model_info = get_model_info(oauth, model_blob_id, model_bucket)
        blob_model_info['ModelBlobId'] = model_blob_id
        blob_model_info['ModelBucket'] = model_bucket
        # preset parameters using to prediction
        prediction_parameters = dict()
        prepare_prediction_files(oauth, prediction_parameters, blob_model_info)
        # add model to cache, if it not in cache
        if model_blob_id not in MODELS_IN_MEMORY_CACHE.keys():
            try:
                MODELS_IN_MEMORY_CACHE[model_blob_id] = cache_model(
                    prediction_parameters['ModelsFolder'])
            except FileNotFoundError:
                LOGGER.error(
                    'Cant preload SSP model with blob id: {}'.format(
                        model_blob_id
                    )
                )


def get_models_from_body_message(body):
    """
    Method to get model id, model blob id, model bucket from input rabbitmq
    message.

    :param body: rabbitmq message body
    :type body: dict
    :return: models ids, models blob ids, models, buckets
    """

    model_id = []
    models_blob_ids = []
    models_buckets = []

    for model_ids in body['Models']:
        model_id.append(model_ids['Id'])
        models_blob_ids.append(model_ids['Blob']['id'])
        models_buckets.append(model_ids['Blob']['bucket'])

    return model_id, models_blob_ids, models_buckets


def error(exception_message):
    """
    Define json prediction error for single structure predictor

    :param exception_message: single structure prediction error message
    :type exception_message: str
    :return: prediction error
    :rtype: dict
    """

    return {
        'error': exception_message
    }


def result(prediction):
    """
    Define json prediction results for single structure predictor

    :param prediction: structure prediction
    :return: prediction results
    :rtype: dict
    """

    return {
        'value': float(prediction[0][2][2])
    }


def dataset(blob_model_info):
    """
    Define json dataset parameters for single structure predictor

    :param blob_model_info: model info
    :type blob_model_info: dict
    :return: dataset parameters
    :rtype: dict
    """

    if 'Description' in blob_model_info['Dataset'].keys():
        description = blob_model_info['Dataset']['Description']
    elif 'DatasetDescription' in blob_model_info.keys():
        description = blob_model_info['DatasetDescription']
    else:
        description = None

    return {
        'title': blob_model_info['Dataset']['Title'],
        'description': description
    }


def property_parameters(blob_model_info):
    """
    Define json property parameters for single structure predictor

    :param blob_model_info: model info
    :type blob_model_info: dict
    :return: property parameters
    :rtype: dict
    """

    return {
        'category': blob_model_info['Property']['Category'],
        'name': blob_model_info['Property']['Name'],
        'units': correct_units(blob_model_info['Property']['Units']),
        'description': blob_model_info['Property']['Description']
    }


def correct_units(units):
    """
    Method to correct property units. If property have UNITLESS units,
    set units to None

    :param units: property units
    :type units: str
    :return: property units or None if units == UNITLESS
    :rtype: str
    """

    corrected_units = None
    if units != UNITLESS:
        corrected_units = units

    return corrected_units


def training_parameters(blob_model_info):
    """
    Define json training parameters for single structure predictor

    :param blob_model_info: model info
    :type blob_model_info: dict
    :return: training parameters
    :rtype: dict
    """

    return {
        'method': blob_model_info['ModelCode'],
        'fingerprints': blob_model_info['Fingerprints'],
        'name': blob_model_info['ModelName'],
        'scaler': blob_model_info['Scaler'],
        'kFold': blob_model_info['KFold'],
        'testDatasetSize': blob_model_info['TestDatasetSize'],
        'subSampleSize': blob_model_info['SubSampleSize'],
        'className': blob_model_info['ClassName'],
        'modi': blob_model_info['Modi']
    }


def applicability_domain(prediction):
    """
    Define json applicability domain parameters for single structure predictor

    :param prediction: structure prediction
    :return: applicability domain parameters
    :rtype: dict
    """

    return {
        'distance': prediction[0][3][2],
        'density': prediction[0][4][2],
    }


def format_predicted_value(value):
    """
    Method to format predicted value. Currently useless

    :param value: predicted value
    :type value: float
    :return: formatted prediction value
    :rtype: str
    """

    return str(value)


if __name__ == '__main__':
    clear_models_folder()
    try:
        preload_ssp_models()
        LOGGER.info('SSP models preloaded')
    except:
        LOGGER.error('SSP models not loaded')
    try:
        PREFETCH_COUNT = int(os.environ['OSDR_RABBIT_MQ_ML_SINGLE_STRUCTURE_PREDICTOR_PREFETCH_COUNT'])
    except KeyError:
        PREFETCH_COUNT = 1
        LOGGER.error('Prefetch count not defined. Set it to 1')

    PREDICT_SINGLE_STRUCTURE['event_callback'] = callback
    SINGLE_STRUCTURE_PROPERTY_PREDICTOR_CONSUMER = PureConsumer(
        PREDICT_SINGLE_STRUCTURE, infinite_consuming=True,
        prefetch_count=PREFETCH_COUNT
    )
    SINGLE_STRUCTURE_PROPERTY_PREDICTOR_CONSUMER.start_consuming()
