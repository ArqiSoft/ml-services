import json
import os
import time
import unittest
import uuid

import redis
import requests

from MLLogger import BaseMLLogger
from general_helper import (
    get_oauth, fetch_token, get_multipart_object, post_data_to_blob,
    get_file_info_from_blob
)
from learner.algorithms import (
    CODES, DNN_REGRESSOR, DNN_CLASSIFIER, ELASTIC_NETWORK, LOGISTIC_REGRESSION
)
from learner.algorithms import NAIVE_BAYES
from mass_transit.MTMessageProcessor import MTPublisher
from mass_transit.mass_transit_constants import (
    MODELER_FAIL_TEST, MODEL_TRAINED_TEST, PREDICTOR_FAIL_TEST,
    PROPERTIES_PREDICTED_TEST, GENERATE_REPORT_TEST, OPTIMIZE_TRAINING_TEST,
    OPTIMIZE_TRAINING_FAIL_TEST, PREDICT_SINGLE_STRUCTURE_TEST,
    FEATURE_VECTORS_CALCULATOR_TEST, FEATURE_VECTORS_CALCULATOR_FAIL_TEST)

TEMP_FOLDER = os.environ['OSDR_TEMP_FILES_FOLDER']
LOGGER = BaseMLLogger(log_name='ml_test_logger', log_file_name='ml_test')
REDIS_CLIENT = redis.StrictRedis(host='redis', db=0)

MODELER_FAIL_FLAG = False
CLASSIC_CLASSIFICATION_NAIVE_TRAINED_FLAG = False
CLASSIC_REGRESSION_TRAINED_FLAG = False
PREDICTOR_FAIL_FLAG = False
CLASSIC_CLASSIFICATION_PREDICTED_FLAG = False
REGRESSOR_TRAINING_OPTIMIZED = False
CLASSIFIER_TRAINING_OPTIMIZED = False
NAIVE_BAYES_MODEL_BLOB_ID = None
LOGISTIC_REGRESSION_MODEL_BLOB_ID = None
CLASSIC_CLASSIFICATION_MODEL_BUCKET = None
CLASSIC_CLASSIFICATION_FILES_BLOB_IDS = []
ELASTIC_NETWORK_MODEL_BLOB_ID = None
CLASSIC_REGRESSION_FILES_BLOB_IDS = []
TMP_GLOBAL_VARIABLE = []
TRAINING_OPTIMIZER_FAIL_FLAG = False
CLASSIFICATION_SINGLE_STRUCTURE_PREDICTED_FLAG = False
FEATURE_VECTORS_CALCULATED_FLAG = False
FEATURE_VECTORS_CALCULATION_FAILED_FLAG = False
CLASSIC_CLASSIFICATION_LOGISTIC_TRAINED_FLAG = False
BLOB_URL = '{}/blobs'.format(os.environ['OSDR_BLOB_SERVICE_URL'])
CLIENT_ID = os.environ['OSDR_ML_MODELER_CLIENT_ID']


class TestTMP(unittest.TestCase):
    def setUp(self):
        """
        Method for do some needed stuff before test starts
        """

        os.environ['OAUTHLIB_INSECURE_TRANSPORT'] = '1'
        # initialise test client for flask FAR application
        # and set it to internal TestCase variable
        self.blob_version_url = '{}/version'.format(
            os.environ['OSDR_BLOB_SERVICE_URL'])
        self.temp_folder = os.environ['OSDR_TEMP_FILES_FOLDER']
        self.test_file_name = 'ML_test.txt'
        self.oauth = get_oauth()
        self.test_file = 'DNN_data_solubility.sdf'
        self.test_file_cif = 'test.cif'
        self.parent_id = 'c1cc0000-5d8b-0015-d72b-08d52f3ea2a9'
        self.user_id = '8d76f88c-fc99-45ca-8951-74d3a5fda263'
        self.source_blob_id = get_blob_id(self)
        self.model_bucket = os.environ['OSDR_ML_MODELER_CLIENT_ID']

    def test_blob_connection(self):
        """
        Test for blob storage API container connection
        """

        # logging test start
        LOGGER.info('Blob connection test start')

        response = requests.get(self.blob_version_url, verify=False)

        # if blob storage available, 200 status code
        self.assertEqual(200, response.status_code)

    def test_posting_data_to_blob(self):
        """
        Test for post_data_to_blob method. Used in ML services a lot
        """

        # logging test start
        LOGGER.info('Post data to blob test start')

        # create test file in file system
        make_test_file = open(
            os.path.join(self.temp_folder, self.test_file_name), 'a')
        make_test_file.write('123 312 222\n')
        make_test_file.close()

        # make metadata for POST
        message_object = {
            'ParentId': self.parent_id,
            'UserId': self.user_id
        }
        # make file data POST
        data_file_path = os.path.join(self.temp_folder, self.test_file_name)
        data_file_type = 'text/txt'
        # make multipart object, which used for simplifying POSTing
        multipart_object = get_multipart_object(
            message_object, data_file_path, data_file_type)

        # POST data to blob storage
        fetch_token(self.oauth)
        response = post_data_to_blob(self.oauth, multipart_object)

        # if post successful, 200 status code
        self.assertEqual(200, response.status_code)

    def test_classic_classification_model_training_logistic_regression(self):
        """
        Test for 'model train' workflow
        """

        # logging test start
        LOGGER.info('Model trained test start')
        # create MassTransit emulation dict
        MODEL_TRAINED_TEST['event_callback'] = classic_classification_train_logistic_regression
        MODEL_TRAINED_TEST['command_callback'] = classic_classification_train_logistic_regression
        model_trained_publisher = MTPublisher(MODEL_TRAINED_TEST)
        # create valid message for send
        message_body = {
            'SourceBlobId': self.source_blob_id,
            'SourceBucket': self.user_id,
            'ParentId': self.parent_id,
            'Name': 'test_classic_classification_logistic_regression',
            'Method': CODES[LOGISTIC_REGRESSION],
            'ClassName': 'Soluble',
            'SubSampleSize': 1.0,
            'TestDatasetSize': 0.2,
            'KFold': 2,
            'Fingerprints': [
                {'Type': 'DESC'},
                {'Type': 'MACCS'},
                {'Type': 'AVALON', 'Size': 512},
                {'Type': 'FCFC', 'Size': 512, 'Radius': 3}
            ],
            'CorrelationId': 'c1cc0000-5d8b-0015-2aab-08d52f3ea202',
            'UserId': self.user_id,
            'Scaler': 'Standard',
            'HyperParameters': {
                'optimizationMethod': 'default',
                'numberOfIterations': 100
            },
            'DnnLayers': None,
            'DnnNeurons': None
        }
        # send message
        fetch_token(self.oauth)
        model_trained_publisher.publish(message_body)
        # wait while model training
        time.sleep(10)

        self.assertEqual(True, CLASSIC_CLASSIFICATION_LOGISTIC_TRAINED_FLAG)

    def test_classic_classification_model_training_naive_bayes(self):
        """
        Test for 'model train' workflow
        """

        # logging test start
        LOGGER.info('Model trained test start')
        # create MassTransit emulation dict
        MODEL_TRAINED_TEST['event_callback'] = classic_classification_train_naive_bayes
        MODEL_TRAINED_TEST['command_callback'] = classic_classification_train_naive_bayes
        model_trained_publisher = MTPublisher(MODEL_TRAINED_TEST)
        # create valid message for send
        message_body = {
            'SourceBlobId': self.source_blob_id,
            'SourceBucket': self.user_id,
            'ParentId': self.parent_id,
            'Name': 'test_classic_classification_naive_bayes',
            'Method': CODES[NAIVE_BAYES],
            'ClassName': 'Soluble',
            'SubSampleSize': 1.0,
            'TestDatasetSize': 0.2,
            'KFold': 5,
            'Fingerprints': [
                {'Type': 'DESC'},
                {'Type': 'MACCS'},
                {'Type': 'AVALON', 'Size': 512},
                {'Type': 'FCFC', 'Size': 512, 'Radius': 3}
            ],
            'CorrelationId': 'c1cc0000-5d8b-0015-2aab-08d52f3ea201',
            'UserId': self.user_id,
            'Scaler': 'Standard',
            'HyperParameters': {
                'optimizationMethod': 'default',
                'numberOfIterations': 100
            },
            'DnnLayers': None,
            'DnnNeurons': None
        }
        # send message
        fetch_token(self.oauth)
        model_trained_publisher.publish(message_body)
        # wait while model training
        time.sleep(10)

        self.assertEqual(True, CLASSIC_CLASSIFICATION_NAIVE_TRAINED_FLAG)

    def test_classic_classification_properties_predicted(self):
        """
        Test for 'properties successful predicted' workflow
        """

        # logging test start
        LOGGER.info('Properties predicted test start')
        # create MassTransit emulation dict
        PROPERTIES_PREDICTED_TEST['event_callback'] = properties_predicted
        PROPERTIES_PREDICTED_TEST['command_callback'] = properties_predicted
        properties_predicted_publisher = MTPublisher(PROPERTIES_PREDICTED_TEST)
        # create valid message for send
        message_body = {
            'ParentId': self.parent_id,
            'FolderId': self.user_id,
            'FolderName': 'test_prediction',
            'DatasetFileName': 'test',
            'DatasetBlobId': self.source_blob_id,
            'DatasetBucket': self.user_id,
            'ModelBlobId': NAIVE_BAYES_MODEL_BLOB_ID,
            'ModelBucket': CLASSIC_CLASSIFICATION_MODEL_BUCKET,
            'Id': 'd397a097-68eb-464a-86b2-abd2315ddebf',
            'UserId': self.user_id,
            'CorrelationId': 'c1cc1000-5d8b-0015-16ec-08d52f4161fe'
        }
        # send message
        fetch_token(self.oauth)
        properties_predicted_publisher.publish(message_body)

        self.assertEqual(True, CLASSIC_CLASSIFICATION_PREDICTED_FLAG)

    def test_dnn_regression_model_trained_event(self):
        """
        Test for 'DNN regression model successful trained' workflow
        """

        # logging test start
        LOGGER.info('DNN regression model trained test start')
        # create MassTransit emulation dict
        MODEL_TRAINED_TEST['event_callback'] = callback_models_trained
        MODEL_TRAINED_TEST['command_callback'] = callback_models_trained
        model_trained_publisher = MTPublisher(MODEL_TRAINED_TEST)
        # create valid message for send
        message_body = {
            'SourceBlobId': self.source_blob_id,
            'SourceBucket': self.user_id,
            'ParentId': self.parent_id,
            'Name': 'test_dnn_regression',
            'Method': CODES[DNN_REGRESSOR],
            'ClassName': 'logS',
            'SubSampleSize': 1.0,
            'TestDatasetSize': 0.2,
            'KFold': 3,
            'Fingerprints': [{'Type': 'ECFP', 'Radius': 2, 'Size': 128}],
            'CorrelationId': 'c1cc0000-5d8b-0015-2aab-08d52f3ea201',
            'UserId': self.user_id,
            'Scaler': 'Standard',
            'HyperParameters': {
                'optimizationMethod': 'default',
                'numberOfIterations': 100
            },
            'DnnLayers': 2,
            'DnnNeurons': 128
        }

        # send message
        fetch_token(self.oauth)
        model_trained_publisher.publish(message_body)

        # self.assertEqual(True, CLASSIC_CLASSIFICATION_TRAINED_FLAG)

    def test_dnn_classifying_model_trained_event(self):
        """
        Test for 'DNN classifying model successful trained' workflow
        """

        # logging test start
        LOGGER.info('DNN classifying model trained test start')
        # create MassTransit emulation dict
        MODEL_TRAINED_TEST['event_callback'] = callback_models_trained
        MODEL_TRAINED_TEST['command_callback'] = callback_models_trained
        model_trained_publisher = MTPublisher(MODEL_TRAINED_TEST)
        # create valid message for send
        message_body = {
            'SourceBlobId': self.source_blob_id,
            'SourceBucket': self.user_id,
            'ParentId': self.parent_id,
            'Name': 'test_dnn_classifying',
            'SourceFileName': 'test.sdf',
            'Method': CODES[DNN_CLASSIFIER],
            'ClassName': 'Soluble',
            'SubSampleSize': 1.0,
            'TestDatasetSize': 0.2,
            'KFold': 2,
            'Fingerprints': [{'Type': 'ECFP', 'Radius': 2, 'Size': 512}],
            'CorrelationId': 'c1cc0000-5d8b-0015-2aab-08d52f3ea221',
            'UserId': self.user_id,
            'Scaler': 'Standard',
            'HyperParameters': {
                'optimizationMethod': 'default',
                'numberOfIterations': 100
            },
            'DnnLayers': 2,
            'DnnNeurons': 128
        }

        # send message
        fetch_token(self.oauth)
        model_trained_publisher.publish(message_body)

        # self.assertEqual(True, CLASSIC_CLASSIFICATION_TRAINED_FLAG)

    @unittest.skip('does not realised')
    def test_properties_predicted_dnn(self):
        self.test_file = 'dnn_model_dataset.h5'
        dnn_dataset_blob_id = get_blob_id(self)
        self.test_file = 'dnn_model.h5'
        dnn_blob_id = get_blob_id(self)

        LOGGER.info('Properties predicted test start')
        PROPERTIES_PREDICTED_TEST[
            'event_callback'] = properties_predicted
        PROPERTIES_PREDICTED_TEST[
            'command_callback'] = properties_predicted
        message_body = {
            'ParentId': self.parent_id,
            'FolderId': self.user_id,
            'FolderName': 'test_dnn.prediction',
            'DatasetFileName': 'test_dnn',
            'DatasetBlobId': dnn_dataset_blob_id,
            'DatasetBucket': self.user_id,
            'ModelBlobId': dnn_blob_id,
            'ModelBucket': CLASSIC_CLASSIFICATION_MODEL_BUCKET,
            'FingerprintRadius': 3,
            'FingerprintSize': 1024,
            'FingerprintType': 1,
            'PrimaryIdField': 'Soluble',
            'Id': 'd397a097-68eb-464a-86b2-abd2315daebf',
            'UserId': self.user_id,
            'CorrelationId': 'c1cc0000-5d8b-0015-16ec-08d52f4162fe'
        }

        properties_predicted_publisher = MTPublisher(
            PROPERTIES_PREDICTED_TEST)
        properties_predicted_publisher.publish(message_body)

    def test_modeler_fail_event(self):
        """
        Test for 'modeling failed' workflow
        """

        # logging test start
        LOGGER.info('Modeler fail test start')
        # create MassTransit emulation dict
        MODELER_FAIL_TEST['event_callback'] = modeler_fail
        MODELER_FAIL_TEST['command_callback'] = modeler_fail
        fail_modeler_publisher = MTPublisher(MODELER_FAIL_TEST)
        # create invalid message for send
        message_body = {'123': 123}
        # send message
        fetch_token(self.oauth)
        fail_modeler_publisher.publish(message_body)

        self.assertEqual(True, MODELER_FAIL_FLAG)

    def test_predictor_fail_event(self):
        """
        Test for 'properties prediction failed' workflow
        """

        # logging test start
        LOGGER.info('Predictor fail test start')
        # create MassTransit emulation dict
        PREDICTOR_FAIL_TEST['event_callback'] = predictor_fail
        PREDICTOR_FAIL_TEST['command_callback'] = predictor_fail
        fail_predictor_publisher = MTPublisher(PREDICTOR_FAIL_TEST)
        # create invalid message for send
        message_body = {'123': 123}
        # send message
        fetch_token(self.oauth)
        fail_predictor_publisher.publish(message_body)

        self.assertEqual(True, PREDICTOR_FAIL_FLAG)

    def test_classifier_training_optimizer(self):
        """
        Test for classic classifier training optimizer
        """

        # logging test start
        LOGGER.info('Classic classification training optimizer test start')
        # create MassTransit emulation dict
        OPTIMIZE_TRAINING_TEST['event_callback'] = regression_training_optimized
        OPTIMIZE_TRAINING_TEST['command_callback'] = regression_training_optimized
        training_optimized_publisher = MTPublisher(OPTIMIZE_TRAINING_TEST)

        # create valid message for send
        message_body = {
            'CorrelationId': 'c1ca0000-5d8b-0015-16ec-08d52f4161fe',
            'Id': 'd397a097-68eb-464a-86b2-abd2315ddebf',
            'UserId': self.user_id,
            'SourceFileName': 'FileNameThere',
            'TargetFolderId': self.parent_id,
            'TimeStamp': '00',
            'SourceBlobId': self.source_blob_id,
            'SourceBucket': self.user_id,
            'Methods': [CODES[NAIVE_BAYES], CODES[LOGISTIC_REGRESSION]],
            'ClassName': 'Soluble',
            'HyperParameters': {
                'NumberOfIterations': 100,
                'OptimizationMethod': 'default'
            },
            'DnnLayers': None,
            'DnnNeurons': None
        }
        # send message
        fetch_token(self.oauth)
        training_optimized_publisher.publish(message_body)

        self.assertEqual(True, REGRESSOR_TRAINING_OPTIMIZED)

    def test_classic_classification_report_generation(self):
        """
        Test for 'generate report' workflow
        """

        # logging test start
        LOGGER.info('Model trained test start')
        # create MassTransit emulation dict
        GENERATE_REPORT_TEST[
            'event_callback'] = classic_classification_report_generation
        GENERATE_REPORT_TEST[
            'command_callback'] = classic_classification_report_generation
        generate_report_publisher = MTPublisher(GENERATE_REPORT_TEST)
        models_blob_ids = [
            NAIVE_BAYES_MODEL_BLOB_ID,
            LOGISTIC_REGRESSION_MODEL_BLOB_ID
        ]
        # create valid message for send
        message_body = {
            'CorrelationId': 'c1cc0000-5d8b-0015-16ac-08d52f4161fe',
            'ParentId': self.parent_id,
            'TimeStamp': 'time stamp there',
            'Models': create_models_data(models_blob_ids),
            'UserId': self.user_id
        }
        # send message
        fetch_token(self.oauth)
        generate_report_publisher.publish(message_body)

        self.assertEqual(True, True)

    def test_regression_model_trained_event(self):
        """
        Test for 'classic regression model successful trained' workflow
        """

        # logging test start
        LOGGER.info('Regression model trained test start')
        # create MassTransit emulation dict
        MODEL_TRAINED_TEST['event_callback'] = classic_regression_train
        MODEL_TRAINED_TEST['command_callback'] = classic_regression_train
        model_trained_publisher = MTPublisher(MODEL_TRAINED_TEST)
        # create valid message for send
        message_body = {
            'CorrelationId': 'c1cc0000-5d8b-0015-2aab-08d52f3ea203',
            'SourceBlobId': self.source_blob_id,
            'SourceBucket': self.user_id,
            'ParentId': self.parent_id,
            'Name': 'test_regression',
            'Method': CODES[ELASTIC_NETWORK],
            'ClassName': 'logS',
            'SubSampleSize': 1.0,
            'TestDatasetSize': 0.2,
            'KFold': 2,
            'Fingerprints': [{'Type': 'ECFP', 'Radius': 4, 'Size': 512}],
            'UserId': self.user_id,
            'Scaler': 'Standard',
            'HyperParameters': {
                'OptimizationMethod': 'default',
                'NumberOfIterations': 100
            },
            'DnnLayers': None,
            'DnnNeurons': None
        }
        # send message
        fetch_token(self.oauth)
        model_trained_publisher.publish(message_body)
        # wait while model training
        time.sleep(10)

        self.assertEqual(True, CLASSIC_REGRESSION_TRAINED_FLAG)

    def test_regression_properties_predicted(self):
        """
        Test for 'regression classic properties successful predicted' workflow
        """

        # logging test start
        LOGGER.info('Regresion properties predicted test start')
        # create MassTransit emulation dict
        PROPERTIES_PREDICTED_TEST['event_callback'] = properties_predicted
        PROPERTIES_PREDICTED_TEST['command_callback'] = properties_predicted
        properties_predicted_publisher = MTPublisher(PROPERTIES_PREDICTED_TEST)

        # create valid message for send
        message_body = {
            'ParentId': self.parent_id,
            'FolderId': self.user_id,
            'FolderName': 'test_regression_prediction',
            'DatasetFileName': 'test',
            'DatasetBlobId': self.source_blob_id,
            'DatasetBucket': self.user_id,
            'ModelBlobId': ELASTIC_NETWORK_MODEL_BLOB_ID,
            'ModelBucket': CLASSIC_CLASSIFICATION_MODEL_BUCKET,
            'Id': 'd397a097-68eb-464a-86b2-abd2315ddebf',
            'UserId': self.user_id,
            'CorrelationId': 'c1cc0000-5d8b-0015-16ec-08d52f4161fe'
        }
        # send message
        fetch_token(self.oauth)
        properties_predicted_publisher.publish(message_body)

        self.assertEqual(True, CLASSIC_CLASSIFICATION_PREDICTED_FLAG)

    def test_training_optimizer_fail(self):
        """
        Test for classic regressor training optimizer
        """

        # logging test start
        LOGGER.info('Training optimizer failed test start')
        # create MassTransit emulation dict
        OPTIMIZE_TRAINING_FAIL_TEST[
            'event_callback'] = training_optimizer_fail
        OPTIMIZE_TRAINING_FAIL_TEST[
            'command_callback'] = training_optimizer_fail
        properties_predicted_publisher = MTPublisher(
            OPTIMIZE_TRAINING_FAIL_TEST)

        # create valid message for send
        message_body = {
            'Methods': []
        }
        # send message
        fetch_token(self.oauth)
        properties_predicted_publisher.publish(message_body)

        self.assertEqual(True, TRAINING_OPTIMIZER_FAIL_FLAG)

    def test_classic_classifier_single_properties_prediction(self):
        """
        Test for classic regressor training optimizer
        """

        # logging test start
        LOGGER.info('Classic regression training optimizer test start')
        # create MassTransit emulation dict
        PREDICT_SINGLE_STRUCTURE_TEST[
            'event_callback'] = classification_single_structure_predicted
        PREDICT_SINGLE_STRUCTURE_TEST[
            'command_callback'] = classification_single_structure_predicted
        single_structure_property_predicted_publisher = MTPublisher(
            PREDICT_SINGLE_STRUCTURE_TEST)

        # create valid message for send
        message_body = {
            'CorrelationId': 'c1cc0000-5d8b-0015-16ec-08d52f4161fb',
            'Id': 'd397a097-68eb-464a-86b2-abd2315ddeba',
            'Structure': '\n  Ketcher  5 71818102D 1   1.00000     0.00000     0\n\n 13 13  0     0  0            999 V2000\n   -2.3818   -0.6252    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0\n   -1.5159   -0.1254    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0\n   -1.5157    0.8745    0.0000 O   0  0  0  0  0  0  0  0  0  0  0  0\n   -0.6501   -0.6254    0.0000 O   0  0  0  0  0  0  0  0  0  0  0  0\n    0.2159   -0.1256    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0\n    0.2162    0.8747    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0\n    1.0824    1.3752    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0\n    1.9483    0.8753    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0\n    1.9479   -0.1251    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0\n    1.0818   -0.6256    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0\n    1.0820   -1.6254    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0\n    1.9479   -2.1252    0.0000 O   0  0  0  0  0  0  0  0  0  0  0  0\n    0.2161   -2.1254    0.0000 O   0  0  0  0  0  0  0  0  0  0  0  0\n  1  2  1  0  0  0  0\n  2  3  2  0  0  0  0\n  2  4  1  0  0  0  0\n  4  5  1  0  0  0  0\n  5  6  2  0  0  0  0\n  6  7  1  0  0  0  0\n  7  8  2  0  0  0  0\n  8  9  1  0  0  0  0\n  9 10  2  0  0  0  0\n  5 10  1  0  0  0  0\n 10 11  1  0  0  0  0\n 11 12  1  0  0  0  0\n 11 13  2  0  0  0  0\nM  END\n',
            'Format': 'MOL',
            'PropertyName': 'Soluble',
            'Models': [
                {
                    'Id': str(uuid.uuid1()),
                    'Blob': {
                        'id': LOGISTIC_REGRESSION_MODEL_BLOB_ID,
                        'bucket': CLIENT_ID
                    }
                },
                {
                    'Id': str(uuid.uuid1()),
                    'Blob': {
                        'id': NAIVE_BAYES_MODEL_BLOB_ID,
                        'bucket': CLIENT_ID
                    }
                }
            ]
        }
        # send message
        fetch_token(self.oauth)
        single_structure_property_predicted_publisher.publish(message_body)

        self.assertEqual(True, CLASSIFICATION_SINGLE_STRUCTURE_PREDICTED_FLAG)

    def test_feature_vectors_calculator_sdf(self):
        """
        Test for sdf file feature vectors calculator
        """

        # logging test start
        LOGGER.info('Feature vectors calculator test start')
        # create MassTransit emulation dict
        FEATURE_VECTORS_CALCULATOR_TEST['event_callback'] = feature_vectors_calculated
        FEATURE_VECTORS_CALCULATOR_TEST['command_callback'] = feature_vectors_calculated
        feature_vectors_calculated_publisher = MTPublisher(FEATURE_VECTORS_CALCULATOR_TEST)
        # create valid message for send
        message_body = {
            'CorrelationId': 'c1cc1111-5d8b-0015-16ec-08d52f4261fb',
            'Fingerprints': [
                {'Type': 'DESC'},
                {'Type': 'FCFC', 'Size': 512, 'Radius': 3}
            ],
            'FileType': 'sdf'
        }
        file_to_send = open(self.test_file, 'rb')
        REDIS_CLIENT.set(
            '{}-file'.format(message_body['CorrelationId']),
            file_to_send.read()
        )
        file_to_send.close()
        # send message
        fetch_token(self.oauth)
        feature_vectors_calculated_publisher.publish(message_body)
        time.sleep(10)
        self.assertEqual(True, FEATURE_VECTORS_CALCULATED_FLAG)

    def test_feature_vectors_calculator_cif(self):
        """
        Test for cif file feature vectors calculator
        """

        # logging test start
        LOGGER.info('Feature vectors calculator test start')
        # create MassTransit emulation dict
        FEATURE_VECTORS_CALCULATOR_TEST['event_callback'] = feature_vectors_calculated
        FEATURE_VECTORS_CALCULATOR_TEST['command_callback'] = feature_vectors_calculated
        feature_vectors_calculated_publisher = MTPublisher(FEATURE_VECTORS_CALCULATOR_TEST)
        # create valid message for send
        message_body = {
            'CorrelationId': 'c1cc1111-5d8b-0015-16ec-08d52f4261af',
            'Fingerprints': [
                {'Type': 'universal	'}
            ],
            'FileType': 'cif'
        }
        file_to_send = open(self.test_file_cif, 'rb')
        REDIS_CLIENT.set(
            '{}-file'.format(message_body['CorrelationId']),
            file_to_send.read()
        )
        file_to_send.close()
        # send message
        fetch_token(self.oauth)
        feature_vectors_calculated_publisher.publish(message_body)
        time.sleep(10)
        self.assertEqual(True, FEATURE_VECTORS_CALCULATED_FLAG)

    def test_feature_vectors_calculator_failed(self):
        """
        Test for feature vectors calculator failed case
        """

        # logging test start
        LOGGER.info('Feature vectors calculator failed case test start')
        # create MassTransit emulation dict
        FEATURE_VECTORS_CALCULATOR_FAIL_TEST['event_callback'] = feature_vectors_calculation_failed
        FEATURE_VECTORS_CALCULATOR_FAIL_TEST['command_callback'] = feature_vectors_calculation_failed
        feature_vectors_calculation_failed_publisher = MTPublisher(FEATURE_VECTORS_CALCULATOR_FAIL_TEST)

        # create valid message for send
        message_body = {
            'CorrelationId': 'c1cc1111-5d8b-0015-16ec-08d52f4161fb',
            'Fingerprints': [
                {'Type': 'DESC'},
                {'Type': 'FCFC', 'Size': 512, 'Radius': 3}
            ],
            'FileType': 'sdf'
        }

        REDIS_CLIENT.set(
            '{}-file'.format(message_body['CorrelationId']),
            'empty'
        )

        # send message
        fetch_token(self.oauth)
        feature_vectors_calculation_failed_publisher.publish(message_body)

        self.assertEqual(True, FEATURE_VECTORS_CALCULATION_FAILED_FLAG)

    def run(self, result=None):
        """ Stop after first error """
        if result.errors:
            raise Exception(result.errors[0][1])
        elif result.failures:
            raise Exception(result.failures[0][1])
        else:
            super().run(result)


def get_single_structure():
    structure_file_name = 'single_structure'
    structure_file = open(structure_file_name, 'r')
    structure = structure_file.readlines()[0]
    structure_file.close()

    return structure


def callback_models_trained(body):
    global CLASSIC_CLASSIFICATION_TRAINED_FLAG

    CLASSIC_CLASSIFICATION_TRAINED_FLAG = True

    return None


def predictor_fail(body):
    """
    Callback function which work when properties prediction failed

    :param body: MassTransit message
    """

    global PREDICTOR_FAIL_FLAG

    PREDICTOR_FAIL_FLAG = True

    return None


def create_models_data(models_blob_ids):
    models = []

    for model_blob_id in models_blob_ids:
        model_info = {
            'bucket': os.environ['OSDR_ML_MODELER_CLIENT_ID'],
            'blobId': model_blob_id,
            'genericFiles': get_blob_ids_of_model_generic_files(model_blob_id)
        }

        models.append(model_info)

    return models


def get_blob_ids_of_model_generic_files(model_blob_id):
    oauth = get_oauth()
    fetch_token(oauth)

    file_info = get_file_info_from_blob(oauth, model_blob_id).json()
    if 'ModelInfo' in file_info['metadata'].keys():
        info_key = 'ModelInfo'
    elif 'modelInfo' in file_info['metadata'].keys():
        info_key = 'modelInfo'
    else:
        raise KeyError('No model info')

    model_id = json.loads(file_info['metadata'][info_key])['ModelId']

    file_to_read = open('{}/{}'.format(TEMP_FOLDER, model_id), 'r')
    lines = file_to_read.readlines()
    file_to_read.close()

    generic_files_ids = []
    for line in lines:
        generic_file_ids = json.loads(line.replace('\'', '"'))
        if generic_file_ids['parent_id'] != model_id:
            continue

        generic_files_ids.append(generic_file_ids['blob_id'])

    return generic_files_ids


def classification_single_structure_predicted(body):
    global CLASSIFICATION_SINGLE_STRUCTURE_PREDICTED_FLAG

    CLASSIFICATION_SINGLE_STRUCTURE_PREDICTED_FLAG = True

    return None


def feature_vectors_calculated(body):
    global FEATURE_VECTORS_CALCULATED_FLAG

    if REDIS_CLIENT.get('{}-csv'.format(body['CorrelationId'])):
        FEATURE_VECTORS_CALCULATED_FLAG = True

    LOGGER.info('FEATURES VECTORS CALCULATOR BODY: {}'.format(body))

    return None


def feature_vectors_calculation_failed(body):
    global FEATURE_VECTORS_CALCULATION_FAILED_FLAG

    FEATURE_VECTORS_CALCULATION_FAILED_FLAG = True
    LOGGER.info('FEATURES VECTORS CALCULATOR FAILED BODY: {}'.format(body))

    return None


def classification_training_optimized(body):
    global CLASSIFIER_TRAINING_OPTIMIZED

    LOGGER.info('CLASSIFICATION OPTIMIZED BODY: {}'.format(body))
    CLASSIFIER_TRAINING_OPTIMIZED = True

    return None


def regression_training_optimized(body):
    global REGRESSOR_TRAINING_OPTIMIZED

    LOGGER.info('REGRESSION OPTIMIZED BODY: {}'.format(body))
    REGRESSOR_TRAINING_OPTIMIZED = True

    return None


def properties_predicted(body):
    """
    Callback function which work when properties prediction success

    :param body: MassTransit message
    """

    global CLASSIC_CLASSIFICATION_PREDICTED_FLAG

    CLASSIC_CLASSIFICATION_PREDICTED_FLAG = True

    return None


def classic_classification_report_generation(body):
    LOGGER.info('REPORT GENERATION TEST BODY: {}'.format(body))
    LOGGER.info('GLOBAL VARIABLE: {}'.format(TMP_GLOBAL_VARIABLE))

    return None


def classic_classification_train_logistic_regression(body):
    """
    Callback function which work when modelling success

    :param body: MassTransit message
    """
    global CLASSIC_CLASSIFICATION_LOGISTIC_TRAINED_FLAG
    global CLASSIC_CLASSIFICATION_FILES_BLOB_IDS
    global LOGISTIC_REGRESSION_MODEL_BLOB_ID

    oauth = get_oauth()
    fetch_token(oauth)

    list_of_ids = get_blob_ids_of_model_generic_files(body['BlobId'])
    CLASSIC_CLASSIFICATION_FILES_BLOB_IDS.extend(list_of_ids)
    LOGISTIC_REGRESSION_MODEL_BLOB_ID = body['BlobId']
    CLASSIC_CLASSIFICATION_LOGISTIC_TRAINED_FLAG = True

    return None


def classic_classification_train_naive_bayes(body):
    """
    Callback function which work when modelling success

    :param body: MassTransit message
    """
    global CLASSIC_CLASSIFICATION_NAIVE_TRAINED_FLAG
    global NAIVE_BAYES_MODEL_BLOB_ID
    global CLASSIC_CLASSIFICATION_MODEL_BUCKET
    global CLASSIC_CLASSIFICATION_FILES_BLOB_IDS

    oauth = get_oauth()
    fetch_token(oauth)

    list_of_ids = get_blob_ids_of_model_generic_files(body['BlobId'])
    NAIVE_BAYES_MODEL_BLOB_ID = body['BlobId']
    CLASSIC_CLASSIFICATION_MODEL_BUCKET = body['Bucket']
    CLASSIC_CLASSIFICATION_FILES_BLOB_IDS.extend(list_of_ids)

    CLASSIC_CLASSIFICATION_NAIVE_TRAINED_FLAG = True

    return None


def classic_regression_train(body):
    """
    Callback function which work when modelling success

    :param body: MassTransit message
    """

    global CLASSIC_REGRESSION_TRAINED_FLAG
    global ELASTIC_NETWORK_MODEL_BLOB_ID
    global CLASSIC_REGRESSION_FILES_BLOB_IDS

    oauth = get_oauth()
    fetch_token(oauth)

    list_of_ids = get_blob_ids_of_model_generic_files(body['BlobId'])
    ELASTIC_NETWORK_MODEL_BLOB_ID = body['BlobId']
    CLASSIC_REGRESSION_FILES_BLOB_IDS.extend(list_of_ids)

    CLASSIC_REGRESSION_TRAINED_FLAG = True

    return None


def modeler_fail(body):
    """
    Callback function which work when modelling failed

    :param body: MassTransit message
    """

    global MODELER_FAIL_FLAG

    MODELER_FAIL_FLAG = True

    return None


def training_optimizer_fail(body):
    global TRAINING_OPTIMIZER_FAIL_FLAG

    TRAINING_OPTIMIZER_FAIL_FLAG = True

    return None


def get_blob_id(test_object):
    """
    Method for add file to blob storage and get file's blob id

    :param test_object: TestCase class object
    :return: saved file blob id
    """

    fetch_token(test_object.oauth)

    # make file's metadata
    metadata = {
        'ParentId': test_object.parent_id,
        'UserId': test_object.user_id
    }
    # make path to file independent of OS
    path_to_file = test_object.test_file
    # prepare multipart object to POST
    multipart_file = get_multipart_object(metadata, path_to_file, 'text/sdf')
    # POST file with metadata to blob storage
    response = post_data_to_blob(
        test_object.oauth, multipart_file, bucket_id=test_object.user_id)

    return response.json()[0]


if __name__ == '__main__':
    # start API's endpoints tests
    unittest.main()
