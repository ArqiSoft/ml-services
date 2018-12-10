"""
Module to make chemical models. Take messages from RabbitMQ queue,
process it and send data to linked blob storage.
"""

import json
import os
import shutil

import requests

from MLLogger import BaseMLLogger
from exception_handler import MLExceptionHandler
from general_helper import (
    get_oauth, fetch_token, get_multipart_object, post_data_to_blob,
    get_file_info_from_blob, make_stream_from_sdf, validate_subsample_size,
    logging_exception_message, validate_kfold, validate_test_datatset_size
)
from learner.algorithms import (
    model_type_by_code, CLASSIFIER, REGRESSOR, algorithm_name_by_code,
    ALGORITHM, TRAINER_CLASS
)
from learner.fingerprints import validate_fingerprints
from learner.plotters import radar_plot, distribution_plot, THUMBNAIL_IMAGE
from mass_transit.MTMessageProcessor import PurePublisher, PureConsumer
from mass_transit.mass_transit_constants import (
    TRAIN_MODEL, TRAINING_FAILED, MODEL_TRAINED, MODEL_TRAINING_STARTED,
    MODEL_THUMBNAIL_GENERATED
)
from messages import (
    model_trained_message, utc_now_str, model_training_start_message,
    thumbnail_generated_message, training_failed
)
from processor import sdf_to_csv
from report_helper.TMP_text import MODEL_PDF_REPORT
from report_helper.html_render import make_pdf_report

LOGGER = BaseMLLogger(log_name='logger', log_file_name='sds-ml-modeler')

LOGGER.info('Configuring from environment variables')
os.environ['OAUTHLIB_INSECURE_TRANSPORT'] = '1'
CLIENT_ID = os.environ['OSDR_ML_MODELER_CLIENT_ID']
BLOB_URL = '{}/blobs'.format(os.environ['OSDR_BLOB_SERVICE_URL'])
BLOB_VERSION_URL = '{}/version'.format(os.environ['OSDR_BLOB_SERVICE_URL'])
TEMP_FOLDER = os.environ['OSDR_TEMP_FILES_FOLDER']

LOGGER.info('Configured')
LOGGER.info('Checking BLOB service: {}'.format(BLOB_VERSION_URL))
RESPONSE = requests.get(BLOB_VERSION_URL, verify=False)
LOGGER.info('BLOB version received: {}'.format(RESPONSE.text))


@MLExceptionHandler(
    logger=LOGGER, fail_publisher=TRAINING_FAILED,
    fail_message_constructor=training_failed
)
def train_model(body):
    """
    Pika callback function used by ml modeller. Make plots files, metrics files
    and model file if success.

    :param body: RabbitMQ MT message's body
    """

    # prepared needed for calculations variables
    body['CurrentModelId'] = body['CorrelationId']
    oauth = get_oauth()

    if not body['Method']:
        raise ValueError('Empty method in model trainer')

    body['Method'] = body['Method'].lower()
    method_code = body['Method']
    body['Name'] = '_'.join(algorithm_name_by_code(method_code).split())

    # publish training started event to OSDR
    fetch_token(oauth)
    publish_start_training_event(body)

    # validate input data
    # raise error with invalid parameter
    validate_user_input(body)

    # make dataframe using input parameters
    model_type = model_type_by_code(method_code)
    try:
        dataframe = make_dataframe(model_type, body, oauth)
    except:
        # log error traceback
        logging_exception_message(LOGGER)
        raise Exception('Make dataframe exception')

    # calculate scaler model
    model_trainer = make_model_trainer(method_code, body, dataframe)
    LOGGER.info('Sending scaler file to {}/{}'.format(BLOB_URL, CLIENT_ID))
    fetch_token(oauth)

    # make applicability domain files and calculate values
    LOGGER.info('Calculate applicability domain')
    model_trainer.make_applicability_domain()

    # generic files for current training model, except model (*.sav) files
    body['NumberOfGenericFiles'] = 0
    # train chosen model
    LOGGER.info('Start model training')
    try:
        model_training_data = conduct_training(
            model_trainer, method_code, body, oauth)
    except:
        # log error traceback
        logging_exception_message(LOGGER)
        raise Exception('Model training exception')

    # POST model's files (images, csvs etc)
    try:
        # POST classifier model files
        if model_type == CLASSIFIER:
            post_classifier(
                model_trainer, body, oauth,
                model_training_data['path_to_csv_file']
            )
        # POST regressor model files
        elif model_type == REGRESSOR:
            post_regressor(model_trainer, body, oauth)
    except:
        # log error traceback
        logging_exception_message(LOGGER)
        raise Exception('Post generic data exception')

    # update template tags with all needed data for model's report
    model_trainer.make_report_text()

    # make pdf report for trained model
    pdf_path = make_pdf_report(
        model_trainer.sub_folder, model_trainer.template_tags,
        model_name=body['Name']
    )
    LOGGER.info('Sending pdf report to {}/{}'.format(BLOB_URL, CLIENT_ID))

    # POST pdf report to blob storage
    fetch_token(oauth)
    model_report_info = {
        'FileInfo': json.dumps({
            'modelName': model_trainer.cv_model.model_name,
            'fileType': MODEL_PDF_REPORT
        }),
        'ParentId': body['CurrentModelId']
    }
    multipart_pdf = get_multipart_object(
        body, pdf_path, 'application/pdf', additional_fields=model_report_info)
    post_data_to_blob(oauth, multipart_pdf)
    body['NumberOfGenericFiles'] += 1

    # remove temporary directory
    shutil.rmtree(model_trainer.sub_folder, ignore_errors=True)

    # send model trained message to OSDR via rabbitmq
    model_trained = model_trained_message(
        body, model_training_data, model_trainer)
    model_trained_message_publisher = PurePublisher(MODEL_TRAINED)
    model_trained_message_publisher.publish(model_trained)

    LOGGER.info('Finished Calculations!')

    return None


def conduct_training(
        model_trainer, method_code, training_parameters, oauth
):
    """
    Method for train model, calculate metrics, plot graphs (and thumbnail) and
    POST all data to blob storage

    :param model_trainer: model trainer object
    :param method_code: code of training model using in OSDR
    :param training_parameters: model training parameters
    :param oauth: using in ml service OAuth2Session object
    :type model_trainer: ClassicClassifier, ClassicRegressor, DNNClassifier,
        DNNRegressor
    :type method_code: str
    :type training_parameters: dict
    :return: variables and blob ids using in OSDR
        (for rabbitmq model trained message)
    :rtype: dict
    """

    # add additional keys for training parameters, depends on method_code
    update_current_model_data(training_parameters, method_code)

    # train model, using trainer
    # add models, metrics, plots, etc as trainer attributes
    model_trainer.train_model(method_code)

    # send, http POST request to blob storage api with model data
    fetch_token(oauth)
    response_model = model_trainer.post_model(training_parameters, oauth)
    model_blob_id = response_model.json()[0]
    LOGGER.info('Send model to {}/{}'.format(BLOB_URL, CLIENT_ID))
    # change model trained flag
    # using in model trained or training failed rabbitmq messages to OSDR
    training_parameters['IsModelTrained'] = True
    # post QMRF report generated for current model
    model_trainer.post_qmrf_report(training_parameters, oauth)

    # get model file's variables, using in OSDR
    file_data = get_file_info_from_blob(oauth, model_blob_id).json()
    file_length = file_data['length']
    file_md5 = file_data['mD5']

    # POST plots with model metrics and metrics to blob storage
    path_to_csv_file = model_trainer.post_performance(
        training_parameters, oauth)
    model_trainer.post_plots(training_parameters, oauth)
    post_thumbnail(model_trainer, training_parameters, oauth)

    return {
        'model_blob_id': model_blob_id,
        'model_file_length': file_length,
        'model_file_md5': file_md5,
        'path_to_csv_file': path_to_csv_file,
        'bins_number': model_trainer.bins
    }


def post_thumbnail(model_trainer, training_parameters, oauth):
    """
    Method which send (publish) message with model's thumbnail data to OSDR

    :param model_trainer: model trainer object
    :param training_parameters: model training parameters
    :param oauth: using in ml service OAuth2Session object
    :type model_trainer: ClassicClassifier, ClassicRegressor, DNNClassifier,
        DNNRegressor
    :type training_parameters: dict
    """

    # get path to thumbnail image from model trainer
    path_to_thumbnail = model_trainer.plots['mean']['thumbnail_plot_path']
    # generate thumbnail info (using in blob storage)
    thumbnail_info = {
        'FileInfo': json.dumps({
            'modelName': model_trainer.cv_model.model_name,
            'fileType': THUMBNAIL_IMAGE
        }),
        'SkipOsdrProcessing': 'true',
        'ParentId': training_parameters['CurrentModelId']
    }
    # prepare thumbnail multipart object for send
    multipart_thumbnail = get_multipart_object(
        training_parameters, path_to_thumbnail, 'image/png',
        additional_fields=thumbnail_info
    )
    # POST prepared thumbnail object (image and info) to blob storage
    # get response from blob storage
    response = post_data_to_blob(oauth, multipart_thumbnail)
    # send (publish) thumbnail blob id in OSDR via rabbitmq
    thumbnail_blob_id = response.json()[0]
    publish_thumbnail_generated(training_parameters, thumbnail_blob_id)


def publish_thumbnail_generated(training_parameters, thumbnail_blob_id):
    """
    Method for send (publish) thumbnail generated event to OSDR.
    Send thumbnail blob id and model trainig parameters (model id etc)

    :param training_parameters: model training parameters
    :param thumbnail_blob_id: model thumbnail image's blob id
    :type training_parameters: dict
    """

    # generate message body
    thumbnail_generated = thumbnail_generated_message(
        training_parameters, thumbnail_blob_id)
    # make publisher object
    model_thumbnail_generated_publisher = PurePublisher(
        MODEL_THUMBNAIL_GENERATED)
    # send (publish) rabbitmq message, thumbnail generated event
    model_thumbnail_generated_publisher.publish(thumbnail_generated)
    # change thumbnail generated flag
    # using in thumbnail generated rabbitmq message to OSDR
    training_parameters['IsThumbnailGenerated'] = True


def update_current_model_data(training_parameters, method_code):
    """
    Method to add additional fields to training parameters,
    which using in rabbitmq messages to OSDR

    :param training_parameters: model training parameters
    :param method_code: code of training model using in OSDR
    :type training_parameters: dict
    """

    # set method code value
    training_parameters['CurrentMethodNumber'] = method_code
    # set training started time
    training_parameters['StarModelingTime'] = utc_now_str()
    # set model trained flag
    # using for model trained or training failed messages to OSDR
    training_parameters['IsModelTrained'] = False


def publish_start_training_event(body):
    """
    Method for send (publish) start training event to OSDR via rabbitmq

    :param body: RabbitMQ MT message's body
    :type body: dict
    """

    # generate rabbitmq message
    start_training_publisher = PurePublisher(MODEL_TRAINING_STARTED)
    # create valid message for send
    message_body = model_training_start_message(body)
    # send message
    start_training_publisher.publish(message_body)


def post_classifier(classifier, training_parameters, oauth, metrics_path):
    """
    Method for POST special for classifier model files, such as radar plot

    :param classifier: model trainer object
    :param training_parameters: model training parameters
    :param oauth: using in ml service OAuth2Session object
    :param metrics_path: path to model's metrics csv file
    :type classifier: ClassicClassifier, DNNClassifier
    :type training_parameters: dict
    """

    LOGGER.info('Creating radar_plot')
    # get nbits number from model trainer
    nbits = classifier.bins

    # generate radar plot image
    # get path to radar plot image
    path_to_radar_plot = radar_plot(
        metrics_path, classifier.sub_folder, nbits,
        titlename=training_parameters['Name']
    )
    # update model trainer template tags
    # using on pdf report creation
    classifier.template_tags['radar_plots'].append(path_to_radar_plot)
    # generate radar plot info for blob storage
    radar_plot_info = {
        'FileInfo': json.dumps({
            'modelName': classifier.cv_model.model_name,
        }),
        'ParentId': training_parameters['CurrentModelId']
    }
    # make radar plot multipart encoded object
    multipart_radar_plot = get_multipart_object(
        training_parameters, path_to_radar_plot, 'image/png',
        additional_fields=radar_plot_info
    )
    # log sending multipart encoded object
    LOGGER.info('Sending radar plot to {}/{}'.format(BLOB_URL, CLIENT_ID))
    training_parameters['NumberOfGenericFiles'] += 1
    # send, http POST request to blob storage api with radar plot
    post_data_to_blob(oauth, multipart_radar_plot)


def post_regressor(regresor, training_parameters, oauth):
    """
    Method for POST special for regresor model files,
    such as distribution plot

    :param regresor: model trainer object
    :param training_parameters: model training parameters
    :param oauth: using in ml service OAuth2Session object
    :type regresor: ClassicRegressor, DNNRegressor
    :type training_parameters: dict
    """

    LOGGER.info('Creating distribution_plot')

    # generate distribution plot
    # get path to distribution plot
    path_to_distribution = distribution_plot(
        regresor, model_name=training_parameters['Name'])
    # generate distribution plot info for blob storage
    distribution_plot_info = {
        'FileInfo': json.dumps({
            'modelName': regresor.cv_model.model_name,
        }),
        'ParentId': training_parameters['CurrentModelId']
    }
    # make distribution plot multipart encoded object
    multipart_distribution_plot = get_multipart_object(
        training_parameters, path_to_distribution, 'image/png',
        additional_fields=distribution_plot_info
    )
    # log sending multipart encoded object
    LOGGER.info('Sending distribution plot to {}/{}'.format(
        BLOB_URL, CLIENT_ID))
    training_parameters['NumberOfGenericFiles'] += 1
    # send, http POST request to blob storage api with distribution plot
    post_data_to_blob(oauth, multipart_distribution_plot)


def make_dataframe(model_type, training_parameters, oauth):
    """
    Method for make dataframe using training parameters values.
    Download sdf file from blob storage and convert it to dataframe

    :param model_type: type of training model, classifier or regressor
    :param training_parameters: model training parameters
    :param oauth: using in ml service OAuth2Session object
    :type model_type: str
    :type training_parameters: dict
    :return: prepared dataframe with needed training target column
    """

    # download sdf file from blob storage
    # make stream from downloaded object
    stream = make_stream_from_sdf(training_parameters, oauth)
    # define using local variables
    filename = training_parameters['SourceFileName']
    classname = training_parameters['ClassName']
    fptype = training_parameters['Fingerprints']

    # create dataframe using training parameters and dataframe
    LOGGER.info('Creating Fingerprints for molecules...')
    if model_type == CLASSIFIER:
        dataframe = sdf_to_csv(
            filename, fptype, class_name_list=classname, stream=stream
        )
    elif model_type == REGRESSOR:
        dataframe = sdf_to_csv(
            filename, fptype, value_name_list=classname, stream=stream
        )
    else:
        # raise error if using unknown model type
        # you should use defined (known) model types global variables only
        LOGGER.error('Unknown model type: {}'.format(model_type))
        raise TypeError('Unknown model type: {}'.format(model_type))

    LOGGER.info('Fingerprints created.')

    return dataframe


def make_model_trainer(method_code, training_parameters, dataframe):
    """
    Method for define model trainer object, using later for training model.
    Define by using training parameters and dataframe

    :param method_code: code of training model using in OSDR
    :param training_parameters: model training parameters
    :param dataframe: prepared dataframe
    :type method_code: str
    :type training_parameters: dict
    :return: model trainer object
    :rtype: ClassicClassifier, ClassicRegressor, DNNClassifier, DNNRegressor
    """

    filename = training_parameters['SourceFileName']
    classname = training_parameters['ClassName']
    test_set_size = training_parameters['TestDatasetSize']
    subsample_size = training_parameters['SubSampleSize']
    n_split = training_parameters['KFold']

    if 'optimizationMethod' in training_parameters['HyperParameters'].keys():
        optimization_key = 'optimizationMethod'
    elif 'OptimizationMethod' in training_parameters['HyperParameters'].keys():
        optimization_key = 'OptimizationMethod'
    else:
        raise KeyError('No optimization method key')

    if 'numberOfIterations' in training_parameters['HyperParameters'].keys():
        iterations_key = 'numberOfIterations'
    elif 'NumberOfIterations' in training_parameters['HyperParameters'].keys():
        iterations_key = 'NumberOfIterations'
    else:
        raise KeyError('No optimization method key')

    optimization_method = training_parameters['HyperParameters'][
        optimization_key]
    number_of_iterations = training_parameters['HyperParameters'][
        iterations_key]

    # define scaler name
    # None if not scaler in training parameters
    scaler = None
    if training_parameters['Scaler']:
        scaler = training_parameters['Scaler'].lower()

    # define training method name by code
    method_name = algorithm_name_by_code(method_code)

    model_trainer = ALGORITHM[TRAINER_CLASS][method_name](
        filename, classname, dataframe, test_set_size=test_set_size,
        output_path=TEMP_FOLDER, n_split=n_split, scale=scaler,
        subsample_size=subsample_size, n_iter_optimize=number_of_iterations,
        opt_method=optimization_method
    )

    return model_trainer


def validate_user_input(body):
    """
    Method for validate model training message.
    Check kfold, subsample size, test dataset size and fingerprints variables
    and keys. Raise exception if some value/key invalid

    :param body: rabbit mq message with user's input for train model
    """

    # define required body's keys
    required_keys = [
        'KFold', 'SubSampleSize', 'TestDatasetSize', 'Fingerprints'
    ]
    # check all required keys in body
    for key in required_keys:
        if key not in body.keys():
            raise Exception(
                'User input have not required parameter: {}'.format(key))

    # check k-fold
    k_fold = body['KFold']
    validate_kfold(k_fold)
    # check subsample size
    subsample_size = body['SubSampleSize']
    validate_subsample_size(subsample_size)
    # check test dataset size
    test_dataset_size = body['TestDatasetSize']
    validate_test_datatset_size(test_dataset_size)
    # check fingerprints
    fingerprints = body['Fingerprints']
    validate_fingerprints(fingerprints)


if __name__ == '__main__':
    try:
        PREFETCH_COUNT = int(
            os.environ['OSDR_RABBIT_MQ_ML_MODELER_PREFETCH_COUNT'])
    except KeyError:
        PREFETCH_COUNT = 1
        LOGGER.error('Prefetch count not defined. Set it to 1')

    TRAIN_MODEL['event_callback'] = train_model
    TRAIN_MODELS_COMMAND_CONSUMER = PureConsumer(
        TRAIN_MODEL, infinite_consuming=True, prefetch_count=PREFETCH_COUNT
    )
    TRAIN_MODELS_COMMAND_CONSUMER.start_consuming()
