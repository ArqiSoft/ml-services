"""
Module which contains all possible events for ML training, prediction,
report generation and failure messages
"""
from datetime import datetime

from general_helper import CLIENT_ID
from learner.algorithms import algorithm_name_by_code

# formatter using for model training metadata
METADATA_FORMATTER = '{:.4E}'.format


def model_trained_message(body, trained_model_data, model_trainer):
    """
    Method for make message as dict with model trained data
    which would be send to rabbitmq ml model trained queue

    :param body: body of message received from RabbitMQ queue
    :param trained_model_data: model id, images ids, qmrf id, csv id,
        model file length, model file md5
    :param model_trainer: model trainer object, any child of Trainer
    :type body: dict
    :type trained_model_data: dict
    :return: model trained message
    :rtype: dict
    """

    return {
        'CorrelationId': body['CorrelationId'],
        'Id': body['CurrentModelId'],
        'BlobId': trained_model_data['model_blob_id'],
        'Bucket': CLIENT_ID,
        'UserId': body['UserId'],
        'TimeStamp': utc_now_str(),
        'NumberOfGenericFiles': body['NumberOfGenericFiles'],
        'Property': create_property(),
        'Dataset': create_dataset(),
        'PropertyCategory': None,
        'PropertyName': None,
        'PropertyUnits': None,
        'PropertyDescription': None,
        'DatasetTitle': None,
        'DatasetDescription': None,
        'Modi': model_trainer.modi,
        'AvailableSingleStructurePrediction': False,
        'MethodDisplayName': algorithm_name_by_code(body['Method']),
        'Metadata': trained_model_info_message(body, model_trainer)
    }


def training_failed(error, body):
    """
    Method for make message as dict with data
    which would be send to rabbitmq if model training failed

    :param error: error readable message
    :param body: body of message received from RabbitMQ queue
    :type error: str
    :type body: dict
    :return: model training message
    :rtype: dict
    """

    current_model_id = '00000000-0000-0000-0000-000000000000'
    if 'CurrentModelId' in body.keys():
        current_model_id = body['CurrentModelId']

    number_of_generic_files = 0
    if 'NumberOfGenericFiles' in body.keys():
        number_of_generic_files = body['NumberOfGenericFiles']

    is_model_trained = False
    if 'IsModelTrained' in body.keys():
        is_model_trained = body['IsModelTrained']

    is_thumbnail_generated = False
    if 'IsThumbnailGenerated' in body.keys():
        is_thumbnail_generated = body['IsThumbnailGenerated']

    return {
        'CorrelationId': body['CorrelationId'],
        'Id': current_model_id,
        'NumberOfGenericFiles': number_of_generic_files,
        'IsModelTrained': is_model_trained,
        'IsThumbnailGenerated': is_thumbnail_generated,
        'Message': error,
        'UserId': body['UserId'],
        'TimeStamp': utc_now_str(),
    }


def prediction_failed(error, body):
    """
    Method for make message as dict with data
    which would be send to rabbitmq if property prediction failed

    :param error: error readable message
    :param body: body of message received from RabbitMQ queue
    :type error: str
    :type body: dict
    :return: prediction failed message
    :rtype: dict
    """

    return {
        'CorrelationId': body['CorrelationId'],
        'Id': body['Id'],
        'Message': error,
        'UserId': body['UserId'],
        'TimeStamp': utc_now_str()
    }


def training_report_generation_failed(error, body):
    """
    Method for make message as dict with data
    which would be send to rabbitmq if training report generation failed

    :param error: error readable message
    :param body: body of message received from RabbitMQ queue
    :type error: str
    :type body: dict
    :return: report generation failed message
    :rtype: dict
    """

    number_of_generic_files = 0
    if 'NumberOfGenericFiles' in body.keys():
        number_of_generic_files = body['NumberOfGenericFiles']

    return {
        'CorrelationId': body['CorrelationId'],
        'NumberOfGenericFiles': number_of_generic_files,
        'Message': error,
        'TimeStamp': utc_now_str(),
    }


def training_optimization_failed(error, body):
    """
    Method for make message as dict with data
    which would be send to rabbitmq if training optimization failed

    :param error: error readable message
    :param body: body of message received from RabbitMQ queue
    :type error: str
    :type body: dict
    :return: training optimization failed message
    :rtype: dict
    """

    return {
        'CorrelationId': body['CorrelationId'],
        'Id': body['Id'],
        'UserId': body['UserId'],
        'Message': error
    }


def property_predicted(body, filename, file_blob_id):
    """
    Method for make message as dict with data
    which would be send to rabbitmq ml properties predicted queue

    :param body: body of message received from RabbitMQ queue
    :param filename: name of saved file with prediction
    :param file_blob_id: uploaded prediction file blob storage id
    :type body: dict
    :type filename: str
    :return: property predicted message
    :rtype: dict
    """

    return {
        'Id': body['Id'],
        'CorrelationId': body['CorrelationId'],
        'FileName': filename,
        'FileBlobId': file_blob_id,
        'FileBucket': CLIENT_ID,
        'UserId': body['UserId'],
        'TimeStamp': utc_now_str(),
        'Version': '0'
    }


def trained_model_info_message(body, model_trainer):
    """
    Method for make metadata which would be added to model entry in database

    :param body: body of message received from RabbitMQ queue
    :param model_trainer: model trainer object, any child of Trainer
    :type body: dict
    :return: model info dict for metadata
    :rtype: dict
    """

    return {
        'ModelName': body['Name'],
        'SourceFileName': body['SourceFileName'],
        'SourceBlobId': body['SourceBlobId'],
        'SourceBucket': body['SourceBucket'],
        'ModelBucket': body['ParentId'],
        'Method': algorithm_name_by_code(body['Method']),
        'ClassName': body['ClassName'],
        'SubSampleSize': body['SubSampleSize'],
        'TestDatasetSize': body['TestDatasetSize'],
        'KFold': body['KFold'],
        'Fingerprints': body['Fingerprints'],
        'UserId': body['UserId'],
        'ModelType': model_trainer.model_type,
        'ModelId': body['CurrentModelId'],
        'Bins': model_trainer.bins,
        'Scaler': body['Scaler'],
        'DensityMean': METADATA_FORMATTER(model_trainer.density_mean),
        'DensityStd': METADATA_FORMATTER(model_trainer.density_std),
        'DistanceMean': METADATA_FORMATTER(model_trainer.distance_mean),
        'DistanceStd': METADATA_FORMATTER(model_trainer.distance_std),
        'TrainShape': model_trainer.train_shape,
        'Property': create_property(),
        'Dataset': create_dataset(),
        'Modi': model_trainer.modi
    }


def create_property():
    """
    Method to make property data using for trained model

    :return: using for training property data
    :rtype: dict
    """

    return {
        'Category': None,
        'Name': None,
        'Units': 'Unitless',
        'Description': None
    }


def create_dataset():
    """
    Method to make dataset data using for trained model

    :return: using for training dataset data
    :rtype: dict
    """

    return {
        'Title': None,
        'Description': None,
    }


def model_training_start_message(body):
    """
    Method for make message with data which would be send
    to rabbitmq ml start training queue

    :param body: body of message received from RabbitMQ queue
    :type body: dict
    :return: model training start message
    :rtype: dict
    """

    return {
        'CorrelationId': body['CorrelationId'],
        'Id': body['CurrentModelId'],
        'UserId': body['UserId'],
        'ModelName': algorithm_name_by_code(body['Method']),
        'TimeStamp': utc_now_str()
    }


def feature_vectors_calculated_message(body):
    """
    Method for make message with data which would be send
    to rabbitmq features vectors calculated queue, if calculation success

    :param body: body of message received from RabbitMQ queue
    :type body: dict
    :return: features vectors calculated message
    :rtype: dict
    """

    return {
        'CorrelationId': body['CorrelationId'],
        'Structures': body['Structures'],
        'Columns': body['Columns'],
        'Failed': body['Failed']
    }


def feature_vectors_calculation_failed(error, body):
    """
    Method for make message as dict with data
    which would be send to rabbitmq if features vectors calculation failed

    :param error: error readable message
    :param body: body of message received from RabbitMQ queue
    :type error: str
    :type body: dict
    :return: features vectors calculation failed message
    :rtype: dict
    """

    return {
        'CorrelationId': body['CorrelationId'],
        'Message': error
    }


def model_training_optimized(body):
    """
    Method for make message with data which would be send
    to rabbitmq ml optimize training queue

    :param body: body of message received from RabbitMQ queue
    :type body: dict
    :return: training optimized message
    :rtype: dict
    """

    return {
        'CorrelationId': body['CorrelationId'],
        'SubSampleSize': body['SubSampleSize'],
        'TestDataSize': body['TestDataSize'],
        'Scaler': body['Scaler'],
        'KFold': body['KFold'],
        'Fingerprints': body['Fingerprints'],
        'Id': body['Id'],
        'UserId': body['UserId'],
        'HyperParameters': {
            'OptimizationMethod': body['OptimizationMethod'],
            'NumberOfIterations': body['NumberOfIterations']
        }
    }


def thumbnail_generated_message(body, thumbnail_blob_id):
    """
    Method for make message as dict with data
    which would be send to rabbitmq thumbnail generated queue

    :param body: body of message received from RabbitMQ queue
    :param thumbnail_blob_id: thimbnail blob id
    :type body: dict
    :return: thumbnail generated message
    :rtype: dict
    """

    return {
        'CorrelationId': body['CorrelationId'],
        'Id': body['CurrentModelId'],
        'BlobId': thumbnail_blob_id,
        'Bucket': CLIENT_ID,
        'TimeStamp': utc_now_str()
    }


def training_report_generated_message(body):
    """
    Method for make message as dict with data
    which would be send to rabbitmq training report generated queue

    :param body: body of message received from RabbitMQ queue
    :type body: dict
    :return: 'training report generated' message
    :rtype: dict
    """

    return {
        'CorrelationId': body['CorrelationId'],
        'NumberOfGenericFiles': body['NumberOfGenericFiles'],
        'UserId': body['UserId'],
        'TimeStamp': utc_now_str()
    }


def single_structure_property_predicted(body):
    """
    Method for make message as dict with data which would be send to rabbitmq
    ml single structure property predicted queue

    :param body: body of message received from RabbitMQ queue
    :type body: dict
    :return: single structure property predicted message
    :rtype: dict
    """

    return {
        'CorrelationId': body['CorrelationId'],
        'Id': body['Id'],
        'Data': body['Data']
    }


def utc_now_str():
    """
    Method for get string with datetime now in utc format

    :return: datetime now in utc format as string
    :rtype: str
    """

    return str(datetime.utcnow())
