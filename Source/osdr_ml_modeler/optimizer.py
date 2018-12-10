import csv
import json
import os
import shutil
from collections import OrderedDict
from time import time

import numpy
import redis
from sklearn import model_selection

from MLLogger import BaseMLLogger
from exception_handler import MLExceptionHandler
from general_helper import (
    get_oauth, make_stream_from_sdf, make_directory, get_multipart_object,
    post_data_to_blob, fetch_token
)
from learner.algorithms import (
    CLASSIFIER, REGRESSOR, model_type_by_code, NAIVE_BAYES, ELASTIC_NETWORK,
    TRAINER_CLASS, ALGORITHM, CODES
)
from mass_transit.MTMessageProcessor import PureConsumer, PurePublisher
from mass_transit.mass_transit_constants import (
    OPTIMIZE_TRAINING, TRAINING_OPTMIZATION_FAILED, TRAINING_OPTIMIZED
)
from messages import training_optimization_failed, model_training_optimized
from processor import sdf_to_csv

os.environ['OAUTHLIB_INSECURE_TRANSPORT'] = '1'
BLOB_URL = '{}/blobs'.format(os.environ['OSDR_BLOB_SERVICE_URL'])
REDIS_CLIENT = redis.StrictRedis(host='redis', db=0)

TEMP_FOLDER = os.environ['OSDR_TEMP_FILES_FOLDER']
LOGGER = BaseMLLogger(
    log_name='logger', log_file_name='sds-ml-training-optimizer')
try:
    EXPIRATION_TIME = int(os.environ['REDIS_EXPIRATION_TIME_SECONDS'])
except KeyError:
    EXPIRATION_TIME = 12*60*60  # 12 hours
    LOGGER.error('Max thread number not defined. Set it to 1')
OPTIMIZER_FORMATTER = '{:.04f}'.format
# set optimizer fingerprints sets
# will found optimal set from this list, and use it later for training model
# all other sets will be shown on optimizer report and on training report
BASE_FINGERPRINTS = [
    [
        {'Type': 'DESC'}, {'Type': 'AVALON', 'Size': 512},
        {'Type': 'ECFP', 'Radius': 3, 'Size': 128},
        {'Type': 'FCFC', 'Radius': 2, 'Size': 256}
    ], [
        {'Type': 'MACCS'}, {'Type': 'AVALON', 'Size': 256},
        {'Type': 'ECFP', 'Radius': 4, 'Size': 1024},
        {'Type': 'FCFC', 'Radius': 4, 'Size': 256}
    ], [
        {'Type': 'DESC'}, {'Type': 'AVALON', 'Size': 256},
        {'Type': 'ECFP', 'Radius': 4, 'Size': 512},
        {'Type': 'FCFC', 'Radius': 2, 'Size': 128}
    ], [
        {'Type': 'DESC'}, {'Type': 'MACCS'},
        {'Type': 'ECFP', 'Radius': 2, 'Size': 128},
        {'Type': 'FCFC', 'Radius': 4, 'Size': 256}
    ], [
        {'Type': 'DESC'}, {'Type': 'ECFP', 'Radius': 3, 'Size': 1024},
        {'Type': 'FCFC', 'Radius': 4, 'Size': 256}
    ], [
        {'Type': 'DESC'}, {'Type': 'ECFP', 'Radius': 2, 'Size': 512},
        {'Type': 'FCFC', 'Radius': 2, 'Size': 512}
    ], [
        {'Type': 'DESC'}, {'Type': 'MACCS'},
        {'Type': 'ECFP', 'Radius': 2, 'Size': 1024},
        {'Type': 'FCFC', 'Radius': 3, 'Size': 512}
    ], [
        {'Type': 'ECFP', 'Radius': 2, 'Size': 512},
        {'Type': 'FCFC', 'Radius': 3, 'Size': 128}
    ], [
        {'Type': 'DESC'}, {'Type': 'MACCS'},
        {'Type': 'ECFP', 'Radius': 3, 'Size': 512},
        {'Type': 'FCFC', 'Radius': 2, 'Size': 128}
    ], [
        {'Type': 'DESC'}, {'Type': 'AVALON', 'Size': 128},
        {'Type': 'ECFP', 'Radius': 3, 'Size': 512}
    ], [
        {'Type': 'DESC'}, {'Type': 'AVALON', 'Size': 128},
        {'Type': 'ECFP', 'Radius': 2, 'Size': 128},
        {'Type': 'FCFC', 'Radius': 2, 'Size': 128}
    ], [
        {'Type': 'DESC'}, {'Type': 'MACCS'}, {'Type': 'AVALON', 'Size': 512},
        {'Type': 'FCFC', 'Radius': 4, 'Size': 128}
    ]
]


@MLExceptionHandler(
    logger=LOGGER, fail_publisher=TRAINING_OPTMIZATION_FAILED,
    fail_message_constructor=training_optimization_failed
)
def find_optimal_parameters(body):
    """
    Pika callback function used by ml optimizer
    Find optimal training fingerprints set for input dataset
    Using only 1000 (by default) or less structures from input dataset
    Send overall optimizing result to Redis, to use it in ml training report

    :param body: RabbitMQ MT message's body
    :type body: dict
    """

    oauth = get_oauth()

    # check input methods
    if not body['Methods']:
        raise ValueError('Empty Methods')

    # calculate metrics for each fingerprints set
    metrics, target_metric = fingerprints_grid_search(
        oauth, body, BASE_FINGERPRINTS)
    # send all metrics to redis
    # later use it to add to training report
    REDIS_CLIENT.setex(
        'optimizer_metrics_{}'.format(body['CorrelationId']), EXPIRATION_TIME,
        json.dumps(metrics)
    )
    # find best fingerprints set
    optimal_fingerprints = sorted(
        metrics.values(), key=lambda value: value['metrics'][target_metric],
        reverse=True
    )[0]['fptype']
    # set other default 'optimal' parameters for training model
    body['SubSampleSize'] = 1.0
    body['TestDataSize'] = 0.3
    body['Scaler'] = 'MinMax'
    body['KFold'] = 5
    body['Fingerprints'] = optimal_fingerprints
    body['OptimizationMethod'] = 'default'
    body['NumberOfIterations'] = 100

    # make optimizer metrics csv and post it to blob storage
    formatted_metrics = TMP_TMP(
        metrics, model_type_by_code(body['Methods'][0].lower()))
    csv_path = '{}/ml_optimizer/{}/optimizing.csv'.format(
        TEMP_FOLDER, body['CorrelationId'])
    write_optimized_metrics_to_csv(formatted_metrics, csv_path)
    multipart_model = get_multipart_object(
        body, csv_path, 'application/x-spss-sav',
        additional_fields={'ParentId': body['TargetFolderId']}
    )

    # send optimizer metrics csv file to blob storage
    fetch_token(oauth)
    response = post_data_to_blob(oauth, multipart_model)
    LOGGER.info('Optimizer csv status code: {}'.format(response.status_code))

    # send best fingerprints set and 'optimal' parameters to training model
    training_optimized = model_training_optimized(body)
    training_optimized_message_publisher = PurePublisher(TRAINING_OPTIMIZED)
    training_optimized_message_publisher.publish(training_optimized)

    # clear current optimization folder
    shutil.rmtree(
        '{}/ml_optimizer/{}'.format(TEMP_FOLDER, body['CorrelationId']),
        ignore_errors=True
    )


def write_optimized_metrics_to_csv(metrics, csv_file_path):
    csv_formatted_metrics = OrderedDict()
    for key, value in metrics.items():
        if 'fingerprints' not in csv_formatted_metrics.keys():
            csv_formatted_metrics['fingerprints'] = dict()

        column_name = key
        if key == '0':
            column_name = 'Fingerprints set'
        csv_formatted_metrics['fingerprints'][column_name] = column_name

        for sub_key, sub_value in value.items():
            if sub_key not in csv_formatted_metrics.keys():
                csv_formatted_metrics[sub_key] = dict()

            csv_formatted_metrics[sub_key][column_name] = sub_value

    with open(csv_file_path, 'w') as f:
        w = csv.DictWriter(f, csv_formatted_metrics.keys())

        subkeys = csv_formatted_metrics['fingerprint_processing_time'].keys()
        for row_key in subkeys:
            row_dict = dict()
            for key, value in csv_formatted_metrics.items():
                row_dict[key] = value[row_key]

            w.writerow(row_dict)


def fingerprints_as_string(fingerprints):
    """
    Method to formatting fingerprints list to human readable string value

    :param fingerprints: fingerprints set as list
    :type fingerprints: list
    :return: fingerprints set as string
    :rtype: str
    """

    all_fingerprints_string = []
    # loop all fingerprints values in list
    for fingerprint in fingerprints:
        fingerprint_string = '{}'.format(fingerprint['Type'])

        if 'Radius' in fingerprint.keys():
            fingerprint_string += ' {} radius'.format(fingerprint['Radius'])

        if 'Size' in fingerprint.keys():
            fingerprint_string += ' {} size'.format(fingerprint['Size'])

        all_fingerprints_string.append(fingerprint_string)

    return ', '.join(all_fingerprints_string)


def fingerprints_grid_search(
        oauth, body, fingerprints, subsample_size=1000
):
    """
    Function for searching of optimal combination of fingerprints.
    subsample_size molecules are extracted from initial dataset and used
    for training of multiple models with varying combinations of fingerprints.

    :param oauth:
    :param body:
    :param fingerprints: list of fingerprints' combinations
    :param subsample_size: number of objects that will be used to train model
    :return: dict with fingerprints' metrics and statistics
    """

    # make folder for current optimization
    optimizer_folder = '{}/ml_optimizer/{}'.format(
        TEMP_FOLDER, body['CorrelationId'])
    make_directory(optimizer_folder)

    # download and save sdf file
    stream = make_stream_from_sdf(body, oauth)
    filename = body['SourceFileName']
    temporary_sdf_filename = '{}/tmp_{}.sdf'.format(optimizer_folder, filename)
    temporary_sdf_file = open(temporary_sdf_filename, 'wb')
    temporary_sdf_file.write(stream.getvalue())
    temporary_sdf_file.close()

    # extract sample (which have subsample_size) from source dataset
    prediction_target = body['ClassName']
    mode = model_type_by_code(body['Methods'][0].lower())
    sample_file_name = extract_sample_dataset(
        input_file_name=temporary_sdf_filename, subsample_size=subsample_size,
        prediction_target=prediction_target, mode=mode
    )

    # define classifier and regressor models for optimizing
    if mode == CLASSIFIER:
        model_code = NAIVE_BAYES
        target_metric = 'test__AUC'
    elif mode == REGRESSOR:
        model_code = ELASTIC_NETWORK
        target_metric = 'test__R2'
    else:
        raise ValueError('Unknown node: {}'.format(mode))

    # loop all base fingerprints sets to find best set
    metrics = dict()
    for fingerprint_number, fptype in enumerate(fingerprints):

        # make dataframe depends on fingerprint set
        # and model type (classifier or regressor)
        start_fps_processing = time()
        if mode == CLASSIFIER:
            dataframe = sdf_to_csv(
                sample_file_name, fptype=fptype,
                class_name_list=prediction_target
            )
        elif mode == REGRESSOR:
            dataframe = sdf_to_csv(
                sample_file_name, fptype=fptype,
                value_name_list=prediction_target
            )
        else:
            raise ValueError('Unknown mode: {}'.format(mode))
        fps_processing_time_seconds = time() - start_fps_processing

        # train model
        start_current_training = time()
        classic_classifier = ALGORITHM[TRAINER_CLASS][model_code](
            sample_file_name, prediction_target, dataframe, subsample_size=1.0,
            test_set_size=0.2, seed=0, fptype=fptype, scale='minmax',
            n_split=1, output_path=optimizer_folder
        )
        classic_classifier.train_model(CODES[model_code])
        current_training_time_seconds = time() - start_current_training

        # add formatted model's metrics and times to heap
        formatted_metrics = format_metrics(
            classic_classifier.metrics[model_code]['mean'])
        metrics.update({
            fingerprint_number: {
                'fptype': fptype,
                'metrics': formatted_metrics,
                'fingerprint_processing_time': fps_processing_time_seconds,
                'prediction_time': current_training_time_seconds
            }
        })

    return metrics, target_metric


def extract_sample_dataset(
        input_file_name, subsample_size, prediction_target, mode
):
    """
    Function for generation of subsampled dataset
    and writing a corresponding file

    :param input_file_name: name of input file
    :param subsample_size:
            number of structures that will be used to train model
    :param prediction_target: name of the target variable
    :param mode: classification or regression
    :return: name of subsampled file
    """

    prediction_target = '<' + prediction_target + '>'
    valid_list = extract_sample_mols(
        input_file_name, mode, subsample_size=subsample_size,
        prediction_target=prediction_target
    )
    sample_file_name = write_sample_sdf(input_file_name, valid_list)

    return sample_file_name


def write_sample_sdf(input_file_name, valid_list):
    """
    Function for writing a temporary file with a subset of pre-selected
    structures

    :param input_file_name: name of input file
    :param valid_list: list of indexes of pre-selected structures
    :return: name of subsampled file
    """

    sample_file_name = '{}_sample.sdf'.format(input_file_name.split('.')[0])
    sample_file = open(sample_file_name, 'w')
    mol = []
    i = 0

    for line in open(input_file_name):
        mol.append(line)
        if line[:4] == '$$$$':
            i += 1
            if i in valid_list:
                for mol_line in mol:
                    sample_file.write(mol_line)
                valid_list.remove(i)
                mol = []
            else:
                mol = []

    sample_file.close()

    return sample_file_name


def extract_sample_mols(
        input_file_name, mode, prediction_target='', n_bins=20,
        critical_ratio=0.05, subsample_size=1000,
):
    """
    Function for generation of list of indexes. The subset of structures with
    the corresponding indexes will be used for the following model's training.

    :param input_file_name: name of input file
    :param mode: classification or regression
    :param prediction_target: name of the target variable
    :param n_bins: number of bins that will be used to split dataset
        (in a stratified manner) in regression mode
    :param critical_ratio: minimal fraction of minor class objects.
        If actual value is less than critical_ratio,
        major/minor classes ratio will be changed to critical_ratio
    :param subsample_size:
            number of structures that will be used to train model
    :return: list of indexes
    """

    counter = 0
    values_list = list()
    mol_numbers = list()

    with open(input_file_name, 'r') as infile:
        for line in infile:
            if prediction_target in line:
                values_list.append(next(infile, '').strip())

            if line[:4] == '$$$$':
                mol_numbers.append(counter)
                counter += 1

    mol_numbers = numpy.array(mol_numbers)

    if mol_numbers.size <= subsample_size:
        valid_list = mol_numbers
    else:
        if mode == CLASSIFIER:
            temp_values_list = []
            for value in values_list:
                try:
                    temp_value = value.upper()
                    if temp_value == 'TRUE':
                        temp_values_list.append(1)
                    elif temp_value == 'FALSE':
                        temp_values_list.append(0)
                    else:
                        temp_values_list.append(int(temp_value))
                except (AttributeError, ValueError):
                    temp_values_list.append(None)

            values_list = numpy.array(temp_values_list, dtype=int)
            true_class_indexes = numpy.argwhere(values_list == 1).flatten()
            false_class_indexes = numpy.argwhere(values_list == 0).flatten()

            if true_class_indexes.size > false_class_indexes.size:
                major_class_indexes = true_class_indexes
                minor_class_indexes = false_class_indexes
            else:
                major_class_indexes = false_class_indexes
                minor_class_indexes = true_class_indexes

            if minor_class_indexes.size < subsample_size * critical_ratio:
                new_num_major_indexes = subsample_size - minor_class_indexes.size
                valid_list = numpy.hstack((
                    minor_class_indexes,
                    numpy.random.choice(
                        major_class_indexes, new_num_major_indexes, replace=False
                    )
                ))
            else:
                if minor_class_indexes.size/mol_numbers.size > critical_ratio:
                    train_fraction = subsample_size / mol_numbers.size
                    new_num_minor_indexes = train_fraction * minor_class_indexes.size
                    new_num_major_indexes = train_fraction * major_class_indexes.size
                    valid_list = (numpy.hstack((
                        numpy.random.choice(
                            minor_class_indexes, round(new_num_minor_indexes),
                            replace=False
                        ),
                        numpy.random.choice(
                            major_class_indexes, round(new_num_major_indexes),
                            replace=False
                        )
                    )))
                else:
                    valid_list = numpy.hstack((
                        numpy.random.choice(
                            minor_class_indexes,
                            round(subsample_size * critical_ratio),
                            replace=False
                        ),
                        numpy.random.choice(
                            major_class_indexes,
                            round(subsample_size * (1 - critical_ratio)),
                            replace=False
                        )
                    ))

        elif mode == REGRESSOR:
            values_list = numpy.array(values_list, dtype=float)
            percentiles = numpy.percentile(
                values_list, numpy.linspace(0, 100, n_bins + 1))
            falls_into = numpy.searchsorted(percentiles, values_list)
            falls_into[falls_into == 0] = 1
            x_train, x_test, y_train, y_test = model_selection.train_test_split(
                mol_numbers, falls_into, stratify=falls_into,
                train_size=subsample_size
            )

            valid_list = x_train
        else:
            raise ValueError('Unknown mode: {}'.format(mode))

    return valid_list.tolist()


def format_metrics(metrics):
    """
    Method to return dict with formatted metrics keys.
    From tuple of strings to string with dunder ('__') between values

    :param metrics: unformatted metrics. keys looks like ('test', 'AUC')
    :type metrics: dict
    :return: formatted metrics. keys looks like 'test__AUC'
    :rtype: dict
    """

    formatted_metrics = dict()
    for key, value in metrics.items():
        formatted_metrics['{}__{}'.format(key[0], key[1])] = value

    return formatted_metrics


def TMP_TMP(optimal_metrics_dict, model_type):
    # prepare metrics table headers
    if model_type == CLASSIFIER:
        formatted_metrics = OrderedDict({
            '0': {
                'fingerprint_processing_time': 'FP computation time, sec',
                'test__ACC': 'Test ACC',
                'test__AUC': 'Test AUC',
                'test__Matthews_corr': 'Test Matthews corr coeff',
                'prediction_time': 'training time, sec'
            }
        })
    elif model_type == REGRESSOR:
        formatted_metrics = OrderedDict({
            '0': {
                'fingerprint_processing_time': 'FP computation time, sec',
                'test__R2': 'Test R2',
                'test__RMSE': 'Test RMSE',
                'prediction_time': 'training time, sec'
            }
        })
    else:
        raise ValueError('Unknown model type: {}'.format(model_type))

    # fill metrics table values, correspond by header
    if model_type == CLASSIFIER:
        for model_number, model_data in optimal_metrics_dict.items():
            fingerprints_string = fingerprints_as_string(model_data['fptype'])
            formatted_metrics[fingerprints_string] = {
                'fingerprint_processing_time': OPTIMIZER_FORMATTER(
                    model_data['fingerprint_processing_time']),
                'test__ACC': OPTIMIZER_FORMATTER(
                    model_data['metrics']['test__ACC']),
                'test__AUC': OPTIMIZER_FORMATTER(
                    model_data['metrics']['test__AUC']),
                'test__Matthews_corr': OPTIMIZER_FORMATTER(
                    model_data['metrics']['test__Matthews_corr']),
                'prediction_time': OPTIMIZER_FORMATTER(
                    model_data['prediction_time'])
            }
    elif model_type == REGRESSOR:
        for model_number, model_data in optimal_metrics_dict.items():
            fingerprints_string = fingerprints_as_string(model_data['fptype'])
            formatted_metrics[fingerprints_string] = {
                'fingerprint_processing_time': OPTIMIZER_FORMATTER(
                    model_data['fingerprint_processing_time']),
                'test__R2': OPTIMIZER_FORMATTER(
                    model_data['metrics']['test__R2']),
                'test__RMSE': OPTIMIZER_FORMATTER(
                    model_data['metrics']['test__RMSE']),
                'prediction_time': OPTIMIZER_FORMATTER(
                    model_data['prediction_time'])
            }
    else:
        raise ValueError('Unknown model type: {}'.format(model_type))

    return formatted_metrics


if __name__ == '__main__':
    try:
        PREFETCH_COUNT = int(
            os.environ['OSDR_RABBIT_MQ_ML_OPTIMIZER_PREFETCH_COUNT'])
    except KeyError:
        PREFETCH_COUNT = 1
        LOGGER.error('Prefetch count not defined. Set it to 1')

    OPTIMIZE_TRAINING['event_callback'] = find_optimal_parameters
    TRAIN_MODELS_COMMAND_CONSUMER = PureConsumer(
        OPTIMIZE_TRAINING, infinite_consuming=True,
        prefetch_count=PREFETCH_COUNT
    )
    TRAIN_MODELS_COMMAND_CONSUMER.start_consuming()
