"""
Module to make chemical predictions. Take messages from RabbitMQ queue,
process it and send data to linked blob storage.
"""
import os

import requests
import shutil

from MLLogger import BaseMLLogger
from learner.algorithms import algorithm_code_by_name
from messages import property_predicted, prediction_failed
from exception_handler import MLExceptionHandler
from general_helper import (
    fetch_token, get_multipart_object, post_data_to_blob, get_oauth,
    get_dataset, get_model_info, prepare_prediction_parameters,
    get_molecules_from_sdf_bytes, MODELS_IN_MEMORY_CACHE, clear_models_folder
)
from mass_transit.MTMessageProcessor import PureConsumer, PurePublisher
from mass_transit.mass_transit_constants import (
    PREDICTION_FAILED, PREDICT_PROPERTIES, PROPERTIES_PREDICTED
)
from predictor.Predictor import MLPredictor

os.environ['OAUTHLIB_INSECURE_TRANSPORT'] = '1'
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
CLIENT_ID = os.environ['OSDR_ML_MODELER_CLIENT_ID']
TEMP_FOLDER = os.environ['OSDR_TEMP_FILES_FOLDER']
BLOB_URL = '{}/blobs'.format(os.environ['OSDR_BLOB_SERVICE_URL'])
BLOB_VERSION_URL = '{}/version'.format(os.environ['OSDR_BLOB_SERVICE_URL'])

LOGGER = BaseMLLogger(log_name='logger', log_file_name='sds-ml-predictor-core')

LOGGER.info('Checking BLOB service: {}'.format(BLOB_VERSION_URL))
RESPONSE = requests.get(BLOB_VERSION_URL, verify=False)
LOGGER.info('BLOB version received: {}'.format(RESPONSE.text))


@MLExceptionHandler(
    logger=LOGGER, fail_publisher=PREDICTION_FAILED,
    fail_message_constructor=prediction_failed
)
def callback(body):
    """
    Pika callback function used by ml predictor.
    Make file with predicted properties by picked model.
    Send file to blob storage for OSDR

    :param body: RabbitMQ MT message's body
    """

    oauth = get_oauth()
    prediction_parameters = dict()
    fetch_token(oauth)
    # update prediction parameters
    # add dataset as bytes
    # add dataset file name, just for report message
    prediction_parameters['Dataset'], prediction_parameters['DatasetFileName'] = get_dataset(oauth, body)
    model_info = get_model_info(
        oauth, body['ModelBlobId'], body['ModelBucket'])

    fetch_token(oauth)
    model_info['ModelBlobId'] = body['ModelBlobId']
    model_info['ModelBucket'] = body['ModelBucket']
    model_info['ModelCode'] = algorithm_code_by_name(model_info['Method'])
    # update prediction parameters with model paramers, such as density matrix,
    # distance model, MODI, fingerprints etc
    prepare_prediction_parameters(
        oauth, prediction_parameters, model_info)

    # update prediction parameters
    # add list of molecules
    prediction_parameters['Molecules'] = get_molecules_from_sdf_bytes(
        prediction_parameters['Dataset'])

    # define predictor object using prediction parameters
    # make prediction for all molecules
    ml_predictor = MLPredictor(prediction_parameters)
    ml_predictor.make_prediction()

    # send prediction result to OSDR
    # write prediction to csv
    prediction_csv_path = ml_predictor.write_prediction_to_csv()
    fetch_token(oauth)
    # prepare multipart object
    multipart_csv = get_multipart_object(body, prediction_csv_path, 'text/csv')
    # POST data to blob storage
    response_csv = post_data_to_blob(oauth, multipart_csv)

    # get prediction blob id and publish message in properties predicted queue
    prediction_created_event = property_predicted(
        body, os.path.basename(prediction_csv_path), response_csv.json()[0])
    properties_predicted_publisher = PurePublisher(PROPERTIES_PREDICTED)
    properties_predicted_publisher.publish(prediction_created_event)

    # remove prediction file from temporary folder
    os.remove(prediction_csv_path)
    # remove temporary models folder
    shutil.rmtree(prediction_parameters['ModelsFolder'], ignore_errors=True)
    # clear memory
    del MODELS_IN_MEMORY_CACHE[model_info['ModelBlobId']]

    return None


if __name__ == '__main__':
    clear_models_folder()
    try:
        PREFETCH_COUNT = int(os.environ['OSDR_RABBIT_MQ_ML_PREDICTOR_PREFETCH_COUNT'])
    except KeyError:
        PREFETCH_COUNT = 1
        LOGGER.error('Prefetch count not defined. Set it to 1')

    PREDICT_PROPERTIES['event_callback'] = callback
    ML_PREDICTOR_CONSUMER = PureConsumer(
        PREDICT_PROPERTIES, infinite_consuming=True,
        prefetch_count=PREFETCH_COUNT
    )
    ML_PREDICTOR_CONSUMER.start_consuming()
