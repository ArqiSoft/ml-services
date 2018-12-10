"""
Module which contain models training report generator function
"""

import csv
import json
import os
import shutil
import uuid
from collections import OrderedDict

import redis
from pandas import DataFrame

from MLLogger import BaseMLLogger
from exception_handler import MLExceptionHandler
from general_helper import (
    get_file_from_blob, get_oauth, fetch_token, get_file_info_from_blob,
    get_multipart_object, post_data_to_blob, logging_exception_message,
    make_directory
)
from learner.algorithms import CLASSIFIER, REGRESSOR
from learner.plotters import radar_plot
from mass_transit.MTMessageProcessor import PureConsumer, PurePublisher
from mass_transit.mass_transit_constants import (
    GENERATE_REPORT, REPORT_GENERATED, TRAINING_REPORT_GENERATION_FAILED
)
from messages import (
    training_report_generated_message, training_report_generation_failed
)
from report_helper.TMP_text import (
    TRAINING_CSV_METRICS, ALL_MODELS_TRAINING_CSV_METRICS
)
from report_helper.html_render import make_pdf_report

os.environ['OAUTHLIB_INSECURE_TRANSPORT'] = '1'
TEMP_FOLDER = os.environ['OSDR_TEMP_FILES_FOLDER']
REDIS_CLIENT = redis.StrictRedis(host='redis', db=0)
LOGGER = BaseMLLogger(
    log_name='logger', log_file_name='sds-ml-training-reporter')
OPTIMIZER_FORMATTER = '{:.04f}'.format


@MLExceptionHandler(
    logger=LOGGER, fail_publisher=TRAINING_REPORT_GENERATION_FAILED,
    fail_message_constructor=training_report_generation_failed
)
def generate_training_report(body):
    """
    Pika callback function used by training report generator.
    Make plots files, general metrics csv file and report file if success.

    :param body: RabbitMQ MT message's body
    """

    oauth = get_oauth()
    fetch_token(oauth)

    # define using variables for ml reporter
    model = body['Models'][0]
    model_blob_id = model['blobId']
    file_info = get_file_info_from_blob(oauth, model_blob_id).json()
    if 'ModelInfo' in file_info['metadata'].keys():
        info_key = 'ModelInfo'
    elif 'modelInfo' in file_info['metadata'].keys():
        info_key = 'modelInfo'
    else:
        raise KeyError('No model info')
    model_info = json.loads(file_info['metadata'][info_key])
    body['Bins'] = model_info['Bins']
    model_name = model_info['ModelName']
    model_type = model_info['ModelType']
    base_folder = '{}/general_training_report_{}'.format(
        TEMP_FOLDER, uuid.uuid1())
    make_directory(base_folder)
    LOGGER.info('MODEL INFO: {}'.format(model_info))
    LOGGER.info('MODEL NAME: {}'.format(model_name))
    LOGGER.info('MODEL TYPE: {}'.format(model_type))

    # generate general metrics dict
    all_models_metrics = []
    for model in body['Models']:
        # if something wrong with model file from blob storage
        if 'genericFiles' not in model.keys():
            raise TypeError('Empty model\'s generic files blob ids')

        for file_blob_id in model['genericFiles']:
            file_info = get_file_info_from_blob(oauth, file_blob_id).json()
            if 'fileInfo' in file_info['metadata'].keys():
                fileinfo_key = 'fileInfo'
            elif 'FileInfo' in file_info['metadata'].keys():
                fileinfo_key = 'FileInfo'
            else:
                raise KeyError('No fileInfo key in: {}'.format(
                    file_info['metadata'].keys()))

            file_info = json.loads(file_info['metadata'][fileinfo_key])

            if 'fileType' in file_info.keys():
                filetype_key = 'fileType'
            elif 'FileType' in file_info.keys():
                filetype_key = 'FileType'
            else:
                filetype_key = None

            if filetype_key and file_info[filetype_key] == TRAINING_CSV_METRICS:
                csv_blob_id = file_blob_id
                csv_model_metrics = get_file_from_blob(
                    csv_blob_id, oauth).content
                all_models_metrics.append(csv_model_metrics.decode())

            LOGGER.info('CURRENT MODEL INFO: {}'.format(file_info))

    LOGGER.info('ALL MODELS METRICS: {}'.format(all_models_metrics))
    # write general metrics data to csv file
    csv_files_names = write_csvs_files(all_models_metrics)
    general_csv_dict = merge_csv_files(csv_files_names)
    rows = make_general_csv_rows(general_csv_dict)
    general_csv_file_path = write_rows_to_csv_file(rows, base_folder)
    metrics = html_metrics_from_dict(general_csv_dict)

    fetch_token(oauth)
    # make csv info for blob storage
    general_csv_info = {
        'FileInfo': json.dumps({
            'modelName': model_name,
            'fileType': ALL_MODELS_TRAINING_CSV_METRICS
        }),
        'SkipOsdrProcessing': 'true'
    }
    # make multipart object prepared to POST to blob storage
    # include csv file and file info
    multipart_general_csv = get_multipart_object(
        body, general_csv_file_path, 'text/csv',
        additional_fields=general_csv_info
    )
    # POST metrcis csv file to blob storage
    post_data_to_blob(oauth, multipart_general_csv)

    # create general images
    body['NumberOfGenericFiles'] = 0
    path_to_radar_plot = None
    try:
        if model_type == CLASSIFIER:
            LOGGER.info('Creating radar_plot')
            nbits = body['Bins']
            path_to_radar_plot = radar_plot(
                general_csv_file_path, base_folder, nbits,
                titlename=model_name
            )
            # make radar plot multipart encoded object
            multipart_radar_plot = get_multipart_object(
                body, path_to_radar_plot, 'image/png',
                additional_fields={'correlationId': body['CorrelationId']}
            )

            # send, http POST request to blob storage api with radar plot
            post_data_to_blob(oauth, multipart_radar_plot)
            body['NumberOfGenericFiles'] += 1
    except:
        # log error traceback
        logging_exception_message(LOGGER)
        raise Exception('Post generic data exception')

    optimizer_metrics = REDIS_CLIENT.get(
        'optimizer_metrics_{}'.format(body['CorrelationId']))

    if optimizer_metrics:
        optimizer_metrics = html_optimal_metrics_from_dict(
            json.loads(optimizer_metrics), model_type)

    # add metrics and images to pdf report file
    context = {
        'metrics': metrics,
        'radar_plots': [path_to_radar_plot],
        'optimizer': optimizer_metrics
    }
    pdf_path = make_pdf_report(base_folder, context, model_name='general')
    fetch_token(oauth)
    multipart_general_csv = get_multipart_object(
        body, pdf_path, 'application/pdf',
        additional_fields={'correlationId': body['CorrelationId']}
    )
    post_data_to_blob(oauth, multipart_general_csv)
    body['NumberOfGenericFiles'] += 1

    # remove temporary directory
    shutil.rmtree(base_folder, ignore_errors=True)

    report_generated = training_report_generated_message(body)
    model_report_generated_message_publisher = PurePublisher(REPORT_GENERATED)
    model_report_generated_message_publisher.publish(report_generated)

    LOGGER.info('Report generated!')

    return None


def html_optimal_metrics_from_dict(optimal_metrics_dict, model_type):
    """
    Method to convert training metrics from dict to html table

    :param optimal_metrics_dict: training metrics
    :param model_type: type of model, regression or classification
    :type optimal_metrics_dict: dict
    :type model_type: str
    :return: training metrics as html
    :rtype: str
    """

    formatted_metrics = TMP_TMP(optimal_metrics_dict, model_type)

    # make pandas dataframe from dict
    metrics_as_pandas = DataFrame.from_dict(
        formatted_metrics, orient='index'
    ).sort_index().rename(
        index={'0': 'Descriptors/Fingerprints'}
    )

    # convert metrics from OrderedDict to html
    # remove "bad" symbols from html metrics
    metrics = metrics_as_pandas.to_html(
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

    return metrics


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


def html_metrics_from_dict(general_metrics_dict):
    """
    Method to convert general training metrics from dict to html

    :param general_metrics_dict: general training metrics
    :type general_metrics_dict: dict
    :return: general training metrics as html
    :rtype: str
    """

    nuber_of_models = len(general_metrics_dict[('', '')])
    del general_metrics_dict[('', '')]
    # make formatted metrics dict from base dict
    reformatted_metrics = OrderedDict()
    for metric_name, metric_values in general_metrics_dict.items():
        for model_index in range(0, nuber_of_models):

            if model_index not in reformatted_metrics.keys():
                reformatted_metrics[model_index] = {
                    metric_name: metric_values[model_index]
                }

            reformatted_metrics[model_index].update({
                metric_name: metric_values[model_index]
            })

    metrics = DataFrame.from_dict(reformatted_metrics)

    # convert metrics from OrderedDict to html
    # remove "bad" symbols from html metrics
    metrics_as_html = metrics.to_html(
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
    return metrics_as_html


def write_csvs_files(csv_contents):
    """
    Method for write all downloaded metrics to csv files.

    :param csv_contents: list of all downloaded metrics as bytes
    :return: files names of written csv's file with metrcis
    :rtype: list
    """

    csv_files_names = []

    for csv_content in csv_contents:
        scv_file_name = '{}/{}.csv'.format(TEMP_FOLDER, uuid.uuid1())
        file_to_write = open(scv_file_name, 'w')
        file_to_write.write(csv_content)
        file_to_write.close()
        csv_files_names.append(scv_file_name)

    return csv_files_names


def merge_csv_files(csv_files_names):
    """
    Method for make one general metrics csv file from folds and mean metrics.
    Read metrics files name by name (csv_files_names) and merge metrcis to one
    general metrcis dict.
    Return general metrics dict

    :param csv_files_names: names of all folds and mean metrcis files
    :type csv_files_names: list
    :return: general metrcics dict
    :rtype: OrderedDict
    """

    general_csv_dict = OrderedDict()
    for csv_file_name in csv_files_names:
        mean_index = None
        csv_file = open(csv_file_name, newline='')
        csv_reader = csv.reader(csv_file, delimiter=',', quotechar='|')
        for row in csv_reader:
            if not mean_index:
                for index, row_data in enumerate(row):
                    if 'mean' in row_data:
                        mean_index = index

            key = (row[0], row[1])
            value = [row[mean_index]]

            if key not in general_csv_dict.keys():
                general_csv_dict[key] = []

            general_csv_dict[key].extend(value)

        csv_file.close()
        os.remove(csv_file_name)

    return general_csv_dict


def make_general_csv_rows(general_csv_dict):
    """
    Method for make list of metrics from general metrics dict.
    Rows using in general metrics writer

    :param general_csv_dict: dict with all metrics
    :type general_csv_dict: dict
    :return: all metrics as rows
    :rtype: list
    """

    rows = []
    for key, value in general_csv_dict.items():
        row = [key[0], key[1]]
        row.extend(value)
        rows.append(row)

    return rows


def write_rows_to_csv_file(rows, write_folder):
    """
    Method for write general metrics rows to csv file.
    Return path to written file

    :param rows: general csv metrics rows
    :param write_folder: folder to write path
    :type rows: list
    :type write_folder: str
    :return: path to general csv metrics file
    """

    output_csv_name = '{}/{}_general_report.csv'.format(
        write_folder, uuid.uuid1())
    csvfile_to_write = open(output_csv_name, 'a', newline='')
    csv_writer = csv.writer(csvfile_to_write, delimiter=',')
    for row in rows:
        csv_writer.writerow(row)

    csvfile_to_write.close()

    return output_csv_name


if __name__ == '__main__':
    try:
        PREFETCH_COUNT = int(
            os.environ['OSDR_RABBIT_MQ_ML_TRAINING_REPORTER_PREFETCH_COUNT'])
    except KeyError:
        PREFETCH_COUNT = 1
        LOGGER.error('Prefetch count not defined. Set it to 1')

    GENERATE_REPORT['event_callback'] = generate_training_report
    GENERATE_TRAINING_REPORT_COMMAND_CONSUMER = PureConsumer(
        GENERATE_REPORT, infinite_consuming=True, prefetch_count=PREFETCH_COUNT
    )
    GENERATE_TRAINING_REPORT_COMMAND_CONSUMER.start_consuming()
