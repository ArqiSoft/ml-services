import os

import numpy
import redis
from rdkit import Chem

from learner import algorithms
from MLLogger import BaseMLLogger
from exception_handler import MLExceptionHandler
from general_helper import (
    logging_exception_message, get_molecules_from_sdf_bytes, numpy_to_csv,
    get_inchi_key, make_directory
)
from mass_transit.MTMessageProcessor import PureConsumer, PurePublisher
from mass_transit.mass_transit_constants import (
    CALCULATE_FEATURE_VECTORS, FEATURE_VECTORS_CALCULATED,
    FEATURE_VECTORS_CALCULATION_FAILED
)
from messages import (
    feature_vectors_calculated_message, feature_vectors_calculation_failed
)
from processor import sdf_to_csv
from structure_featurizer import generate_csv

LOGGER = BaseMLLogger(
    log_name='logger', log_file_name='sds-feature-vector-calculator')

os.environ['OAUTHLIB_INSECURE_TRANSPORT'] = '1'
REDIS_CLIENT = redis.StrictRedis(host='redis', db=0)
TEMP_FOLDER = os.environ['OSDR_TEMP_FILES_FOLDER']
# make temporary folder if it does not exists
make_directory(TEMP_FOLDER)

try:
    EXPIRATION_TIME = int(os.environ['REDIS_EXPIRATION_TIME_SECONDS'])
except KeyError:
    EXPIRATION_TIME = 12*60*60  # 12 hours
    LOGGER.error('Max thread number not defined. Set it to 1')


@MLExceptionHandler(
    logger=LOGGER, fail_publisher=FEATURE_VECTORS_CALCULATION_FAILED,
    fail_message_constructor=feature_vectors_calculation_failed
)
def calculate_feature_vectors(body):
    try:
        data_file_as_bytes_array = REDIS_CLIENT.get(
            '{}-file'.format(body['CorrelationId']))
    except:
        # log error traceback
        logging_exception_message(LOGGER)
        error_message = 'Can\'t get sdf file from redis'
        raise Exception(error_message)

    if body['FileType'] == 'sdf':
        csv_file_path = process_sdf(body, data_file_as_bytes_array)
    elif body['FileType'] == 'cif':
        path_to_cif_file = '{}/{}.cif'.format(TEMP_FOLDER, body['CorrelationId'])
        cif_file = open(path_to_cif_file, 'wb')
        cif_file.write(data_file_as_bytes_array)
        cif_file.close()

        if 'type' in body['Fingerprints'][0].keys():
            type_key = 'type'
        elif 'Type' in body['Fingerprints'][0].keys():
            type_key = 'Type'
        else:
            raise KeyError(
                'No type key in fingerprints: {}'.format(body['Fingerprints']))

        descriptors = [x[type_key].strip() for x in body['Fingerprints']]
        csv_file_path = '{}/{}.csv'.format(TEMP_FOLDER, body['CorrelationId'])
        shape = generate_csv([path_to_cif_file], descriptors, csv_file_path)

        body['Structures'] = shape[0]
        body['Columns'] = shape[1]
        body['Failed'] = 0
    else:
        raise ValueError('Unknown file type: {}'.format(body['FileType']))

    file_to_send = open(csv_file_path, 'rb')

    try:
        REDIS_CLIENT.setex(
            '{}-csv'.format(body['CorrelationId']), EXPIRATION_TIME,
            file_to_send.read()
        )
    except:
        # log error traceback
        logging_exception_message(LOGGER)
        error_message = 'Can\'t send csv file to redis'
        raise Exception(error_message)

    file_to_send.close()

    # os.remove(csv_file_path)

    feature_vectors_calculator_publisher = feature_vectors_calculated_message(
        body)
    model_trained_message_publisher = PurePublisher(FEATURE_VECTORS_CALCULATED)
    model_trained_message_publisher.publish(
        feature_vectors_calculator_publisher)

    return None


def process_sdf(body, sdf_as_bytes_array):
    fingerprints = list()
    for fingerprint in body['Fingerprints']:
        new_fingerprint = dict()
        for key, value in fingerprint.items():
            new_fingerprint[key.capitalize()] = value

        fingerprints.append(new_fingerprint)

    body['Fingerprints'] = fingerprints
    molecules = get_molecules_from_sdf_bytes(sdf_as_bytes_array)
    errors_list = [[]]
    try:
        data_frame = sdf_to_csv(
            '', body['Fingerprints'],
            find_classes=True, find_values=True, molecules=molecules,
            processing_errors=errors_list
        )
    except:
        # log error traceback
        logging_exception_message(LOGGER)
        error_message = 'Can\'t make dataframe using this sdf file'
        raise Exception(error_message)

    smiles_list = list()
    inchies = list()
    for molecule_number, molecule in enumerate(molecules):
        smiles = Chem.MolToSmiles(molecule, isomericSmiles=True)
        inchi = get_inchi_key(molecule)
        smiles_list.append(('SMILES', molecule_number, smiles))
        inchies.append(('InChiKey', molecule_number, inchi))

    smiles_numpy_array = numpy.array(
        smiles_list,
        dtype=[('name', 'U17'), ('molecule_number', 'i4'), ('value', 'U40')]
    )
    inchi_numpy_array = numpy.array(
        inchies,
        dtype=[('name', 'U17'), ('molecule_number', 'i4'), ('value', 'U40')]
    )
    errors_numpy_array = numpy.array(
        errors_list[0],
        dtype=[('name', 'U17'), ('molecule_number', 'i4'), ('value', 'U40')]
    )
    data_frame = data_frame.astype(
        [('name', 'U10'), ('molecule_number', 'i4'), ('value', 'U40')])

    data_frame = numpy.insert(data_frame, 0, inchi_numpy_array, axis=1)
    data_frame = numpy.insert(data_frame, 0, smiles_numpy_array, axis=1)
    data_frame = numpy.insert(
        data_frame, data_frame.shape[1], errors_numpy_array, axis=1)

    csv_file_path = '{}/{}.csv'.format(TEMP_FOLDER, body['CorrelationId'])

    try:
        numpy_to_csv(data_frame, csv_file_path)
    except:
        # log error traceback
        logging_exception_message(LOGGER)
        error_message = 'Can\'t convert dataframe to csv file'
        raise Exception(error_message)

    body['Structures'] = data_frame.shape[0]
    body['Columns'] = data_frame.shape[1]
    body['Failed'] = 0

    return csv_file_path


if __name__ == '__main__':
    try:
        PREFETCH_COUNT = int(
            os.environ[
                'OSDR_RABBIT_MQ_FEATURE_VECTOR_CALCULATOR_PREFETCH_COUNT'
            ]
        )
    except KeyError:
        PREFETCH_COUNT = 1
        LOGGER.error('Prefetch count not defined. Set it to 1')

    CALCULATE_FEATURE_VECTORS['event_callback'] = calculate_feature_vectors
    FEATURE_VECTOR_CALCULATOR_CONSUMER = PureConsumer(
        CALCULATE_FEATURE_VECTORS, infinite_consuming=True,
        prefetch_count=PREFETCH_COUNT
    )
    FEATURE_VECTOR_CALCULATOR_CONSUMER.start_consuming()
