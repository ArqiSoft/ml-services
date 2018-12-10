"""
Module which contain base methods for ml services
"""
import glob
import io
import json
import os
import shutil
import sys
import traceback
import uuid
import zipfile
from time import time

import keras
import numpy
import requests
import tensorflow
from oauthlib.oauth2 import BackendApplicationClient
from rdkit import Chem
from requests_oauthlib import OAuth2Session
from requests_toolbelt import MultipartEncoder
from scipy import sparse
from sklearn.externals import joblib

from MLLogger import BaseMLLogger

os.environ['OAUTHLIB_INSECURE_TRANSPORT'] = '1'
LOGGER = BaseMLLogger(log_name='logger', log_file_name='sds-ml-logger')
# define modules default global variables
CLIENT_ID = None
CLIENT_SECRET = None
SCOPE = None
TOKEN_URL = None
BLOB_URL = None
TEMP_FOLDER = os.getcwd()
OSDR_API_URL = 'https://api.dev.dataledger.io/osdr/v1/api'
SCALER_FILENAME = 'scaler.sav'
DENSITY_MODEL_FILENAME = 'density_model.sav'
DISTANCE_MATRIX_FILENAME = 'distance_matrix.npz'
TRAIN_MEAN_FILENAME = 'train_mean.npy'
K_MEANS_FILENAME = 'k_means.ksav'
MODEL_ADDITIONAL_FILES = [
    SCALER_FILENAME, DENSITY_MODEL_FILENAME, TRAIN_MEAN_FILENAME,
    DISTANCE_MATRIX_FILENAME, K_MEANS_FILENAME
]
MODELS_IN_MEMORY_CACHE = dict()
NUMPY_PROCESSOR_DTYPES = [
    ('name', 'U15'), ('molecule_number', 'i8'), ('value', 'f8')
]

try:
    TEMP_FOLDER = os.environ['OSDR_TEMP_FILES_FOLDER']
except KeyError as undefined_key:
    LOGGER.error(
        'Temporary folder path not defined. Use default value: {}'.format(
            TEMP_FOLDER
        )
    )

try:
    CLIENT_ID = os.environ['OSDR_ML_MODELER_CLIENT_ID']
    CLIENT_SECRET = os.environ['OSDR_ML_MODELER_CLIENT_SECRET']
    SCOPE = ['api', 'osdr-api']
    TOKEN_URL = os.environ['OSDR_BLOB_SERVICE_TOKEN_URL']
    BLOB_URL = '{}/blobs'.format(os.environ['OSDR_BLOB_SERVICE_URL'])
    OSDR_API_URL = os.environ['OSDR_API_URL']
except KeyError as undefined_key:
    LOGGER.error('Environment variables not defined. Use default values')
    LOGGER.error('Undefined key: {}'.format(undefined_key))

if not os.path.exists(TEMP_FOLDER):
    os.makedirs(TEMP_FOLDER)


def post_data_to_blob(
        oauth, multipart_object, blob_url=BLOB_URL, bucket_id=CLIENT_ID
):
    """
    Method for send POST http request
    to oauth session with multipart_object in body

    :param oauth: using in ml service OAuth2Session object
    :param multipart_object: using in ml service MultipartEncoder object
    :param blob_url: blob storage url
    :param bucket_id: bucket id value to POST in
    :return: response of oauth http POST operation
    """

    post_response = oauth.post(
        '{}/{}'.format(blob_url, bucket_id),
        headers={'Content-Type': multipart_object.content_type},
        data=multipart_object, verify=False
    )

    return post_response


def get_file_info_from_blob(
        oauth, blob_id, blob_url=BLOB_URL, bucket_id=CLIENT_ID
):
    """
    Method to get any file info from blob storage, by blob bucket and id,
    if file and info exist

    :param oauth: using in ml service OAuth2Session object
    :param blob_id: blob storage id of needed file
    :param blob_url: blob storage api URL
    :param bucket_id: blob bucket with needed file
    :return: response of GET request with file info, status code etc
    """

    url = '{}/{}/{}/info'.format(blob_url, bucket_id, blob_id)
    response = oauth.get(url)

    return response


def get_user_info_from_osdr(oauth, user_id, osdr_api_url=OSDR_API_URL):
    # TODO make docstring there
    url = '{}/users/{}/public-info'.format(osdr_api_url, user_id)
    response = oauth.get(url, verify=False)

    return response


def get_file_from_blob(
        file_blob_id, oauth, blob_url=BLOB_URL, bucket_id=CLIENT_ID
):
    file_url = '{}/{}/{}'.format(blob_url, bucket_id, file_blob_id)
    response = oauth.get(file_url, verify=False)

    return response


def delete_data_from_blob(oauth, blob_bucket_id, blob_id=CLIENT_ID):
    """
    Method for delete entry from blob storage

    :param oauth: using in ml service OAuth2Session object
    :param blob_bucket_id: id of blob bucket from which we want delete entry
    :param blob_id: blob storage id of entry which we want to delete
    :return: deletion operation status code
    """

    delete_response = oauth.delete(
        '{}/{}/{}'.format(BLOB_URL, blob_bucket_id, blob_id))

    return delete_response


def fetch_token(oauth):
    """
    Method for fetching oauth token and logging that operation result

    :param oauth: using in ml service OAuth2Session object
    :return: oauth token
    """

    headers = {
        'Content-Type': 'application/x-www-form-urlencoded',
    }
    data = [
        ('grant_type', 'client_credentials'),
        ('client_id', CLIENT_ID),
        ('client_secret', CLIENT_SECRET),
    ]
    LOGGER.info('Token URL: {}'.format(TOKEN_URL))
    LOGGER.info('Token headers: {}'.format(headers))
    LOGGER.info('Token data: {}'.format(data))
    response = requests.post(TOKEN_URL, headers=headers, data=data)

    try:
        token = response.json()
        oauth.token = token
        LOGGER.info('Status code: {}'.format(response.status_code))
        LOGGER.info('Token: {}'.format(token))
    except json.JSONDecodeError:
        raise Exception(
            'Fetch token exception. URL: {} , ID: {} , secret: {}, status code: {}'.format(
                TOKEN_URL, CLIENT_ID, CLIENT_SECRET, response.status_code
            )
        )

    return token


def get_multipart_object(body, file_path, file_type, additional_fields=None):
    """
    Method which create and return MultipartEncoder object,
    which would be posting later in ml service

    :param body: rabbitmq message as dict, for take ParentId, UserId
    :param file_path: path to file which would be encoded
    :param file_type: type of file which would be encoded
    :param additional_fields: additional fields, which you want to add
    :return: encoded MultipartEncoder object
    """

    # base ID fields of multipart object as dict
    if additional_fields and 'ParentId' in additional_fields.keys():
        parent_id = additional_fields['ParentId']
    else:
        parent_id = body['ParentId']
    dict_with_fields = {
        'parentId': parent_id,
        'userId': body['UserId']
    }
    # base file field of multipart object
    file_dict = {
        'file': (os.path.basename(file_path), open(file_path, 'rb'), file_type)
    }
    # add additional fields to created multipart object
    if additional_fields:
        dict_with_fields.update(additional_fields)

    # file(s) must loaded after metadata. metadata after file will be erased
    dict_with_fields.update(file_dict)

    multipart_object = MultipartEncoder(fields=dict_with_fields)

    return multipart_object


def get_oauth():
    """
    Method which create and return OAuth2Session object for ml service

    :return: OAuth2Session object
    """

    client = BackendApplicationClient(client_id=CLIENT_ID)
    oauth = OAuth2Session(client=client, scope=SCOPE)
    oauth.verify = False

    return oauth


def make_directory(directory_path):
    """
    Method for create directory if it not exist

    :param directory_path: path to directory which we want to create
    :type directory_path: str
    """

    if not os.path.isdir(directory_path):
        os.makedirs(directory_path)


def validate_kfold(k_fold):
    """
    Method for validate k-fold value, raise exception if invalid value
    Value should be int, and in interval [2, 10]

    :param k_fold: k-fold value
    :type k_fold: float
    """

    # check k-fold type, should be int
    if not isinstance(k_fold, int):
        raise Exception('User input KFold should be int')
    # check k-fold in interval [2, 10]
    if not 2 <= k_fold <= 10:
        raise Exception('User input KFold should be in interval [2, 10]')


def validate_subsample_size(subsample_size):
    """
    Method for validate subsample size value, raise exception if invalid value
    Value should be float, and in interval (0.1, 1]

    :param subsample_size: subsample_size value
    :type subsample_size: float
    """

    # check subsample size type, should be float
    if not isinstance(subsample_size, float):
        raise Exception('User input SubSampleSize should be float')
    # check subsample size in interval (0.1, 1]
    if not 0.1 < subsample_size <= 1.0:
        raise Exception(
            'User input SubSampleSize should be in interval (0.1, 1]')


def validate_test_datatset_size(test_dataset_size):
    """
    Method for validate test dataset size value, raise exception if invalid
    Value should be float, and in interval [0, 0.5]

    :param test_dataset_size: test_dataset_size value
    :type test_dataset_size: float
    """

    # check test dataset size size in interval [0, 0.5]
    if not 0.0 <= test_dataset_size <= 0.5:
        raise Exception(
            'User input TestDatasetSize should be in interval [0.0, 0.5]')


def make_stream_from_sdf(body, oauth):
    sdf_url = '{}/{}/{}'.format(
        BLOB_URL, body['SourceBucket'], body['SourceBlobId'])
    fetch_token(oauth)
    LOGGER.info('Loading Dataset from: {}'.format(sdf_url))
    sdf = oauth.get(sdf_url, verify=False)
    stream = io.BytesIO(sdf.content)

    LOGGER.info('SDF: {}'.format(sdf))
    LOGGER.info('SDF URL: {}'.format(sdf_url))
    LOGGER.info('SDF HEADERS: {}'.format(sdf.headers))

    dataset_file_name = dict(sdf.headers)['Content-Disposition']
    dataset_file_name = dataset_file_name.split('filename=')[-1]
    body['SourceFileName'] = dataset_file_name

    return stream


def logging_exception_message(logger):
    """
    Method for catch exception message and write it to log

    :param logger: used logger
    """

    exception_type, exception, traceback_text = sys.exc_info()
    # traceback message object as string
    traceback_message = '\n'.join(
        traceback.format_tb(traceback_text))
    # make error traceback message
    exception_message = 'Internal server error.\n'
    exception_message += '{}{}: {}\n'.format(
        traceback_message, exception_type.__name__, exception
    )
    logger.error(exception_message)


def get_file_as_bytes_by_url(oauth, get_url):
    if not get_url:
        return None

    # get scaler from blobstorage
    get_response = oauth.get(get_url, verify=False)

    # get file as binary object from GET response body
    return get_response.content


def make_blob_url(parameters, bucket_key, blob_id_key, blob_url=BLOB_URL):
    if not parameters[blob_id_key]:
        return None

    return '{}/{}/{}'.format(
        blob_url, parameters[bucket_key], parameters[blob_id_key])


def write_file_to_temporary_folder(
        file_as_bytes, file_name, temporary_folder=TEMP_FOLDER
):
    make_directory(temporary_folder)

    temporary_file_path = '{}/{}_{}'.format(
        temporary_folder, uuid.uuid1(), file_name)

    file_to_write = open(temporary_file_path, 'wb')
    file_to_write.write(file_as_bytes)
    file_to_write.close()

    return temporary_file_path


def write_model_to_temporary_folder(oauth, model_url, write_path_folder):
    """
    Method which get and write model file to temporary folder.
    Extract model files to temporary folder if model have few files.
    Files takes from linked blob storage (BLOB_URL variable)

    :return: temporary folder name
    """

    # get model file from blob storage
    # make GET request to blob storage
    get_response = oauth.get(model_url, verify=False)
    # get file as binary object from GET response body
    model_file = get_response.content
    # get filename
    model_file_name = dict(get_response.headers)['Content-Disposition']
    model_file_name = model_file_name.split('filename=')[-1]
    if '"' in model_file_name:
        model_file_name = model_file_name.replace('"', '')

    # make temporary folder with model file
    model_file_path = write_file_to_temporary_folder(
        model_file, model_file_name, temporary_folder=write_path_folder)

    # get model file extension
    file_extension = model_file_name.split('.')[-1]

    # if model contain only one file
    if file_extension == 'h5' or file_extension == 'sav':
        pass
    # if model contain few files in archive
    elif file_extension == 'zip':
        # extract model file from archive to temporary directory
        archive_name = model_file_path
        models_archive = zipfile.ZipFile(archive_name)
        models_archive.extractall(write_path_folder)
        models_archive.close()

        # remove archive
        os.remove(model_file_path)
    else:
        # throw exception if file extension unknown
        raise TypeError(
            'Unknown model extension: {}'.format(file_extension))

    return write_path_folder


def get_dataset(oauth, body):
    """
    Method which get dataset from blobstorage or local storage,
    if run_mode set to 'local', as binary object

    :return: dataset as binary object, and dataset filename
    """

    dataset_url = make_blob_url(body, 'DatasetBucket', 'DatasetBlobId')

    # get dataset from blobstorage
    get_response = oauth.get(dataset_url, verify=False)
    # get file as binary object from GET response body
    dataset = get_response.content
    # get filename
    dataset_file_name = dict(get_response.headers)['Content-Disposition']
    dataset_file_name = dataset_file_name.split('filename=')[-1]

    return dataset, dataset_file_name


def get_model_info(oauth, model_id, model_bucket):
    """
    Method to get model information from blob storage by using model blob id
    and model blob bucket

    :param oauth: OAuth2 object
    :param model_id: model entry blob id uuid
    :param model_bucket: model bucket
    :return: model information from blob storage
    :type model_id: str
    :type model_bucket: str
    :rtype: dict
    """

    blob_model_info_as_string = get_file_info_from_blob(
        oauth, model_id, bucket_id=model_bucket
    ).json()['metadata']

    if 'ModelInfo' in blob_model_info_as_string.keys():
        info_key = 'ModelInfo'
    elif 'modelInfo' in blob_model_info_as_string.keys():
        info_key = 'modelInfo'
    else:
        raise KeyError('No model info')

    return json.loads(blob_model_info_as_string[info_key])


def prepare_prediction_parameters(oauth, prediction_parameters, model_info):
    """
    Prepare all needed files and parameters (key/value pair) to make
    successfully predictions. Does not matter SSP or classic prediction.
    Upload model files to cache to use it later

    :param oauth: OAuth2 object
    :param prediction_parameters: prediction parameters
            (dict using for making predictions)
    :param model_info: information about model,
            in general loaded from blobstorage
    :type prediction_parameters: dict
    :type model_info: dict
    """

    prepare_prediction_files(oauth, prediction_parameters, model_info)
    prepare_model_info(prediction_parameters, model_info)

    if model_info['ModelBlobId'] not in MODELS_IN_MEMORY_CACHE.keys():
        MODELS_IN_MEMORY_CACHE[model_info['ModelBlobId']] = cache_model(
            prediction_parameters['ModelsFolder'])

    prediction_parameters['Models'] = MODELS_IN_MEMORY_CACHE[
        model_info['ModelBlobId']]


def prepare_prediction_files(oauth, prediction_parameters, model_info):
    """
    Method to upload trained model from blobstorage to HDD, using model info
    Save loaded model to temporary folder with using model blob id, so
    all models would be 'unique'

    :param oauth: OAuth2 object
    :param prediction_parameters: prediction parameters
            (dict using for making predictions)
    :param model_info: information about model,
            in general loaded from blobstorage
    :type prediction_parameters: dict
    :type model_info: dict
    """

    models_folder = '{}/SSP_temp_models/{}'.format(
        TEMP_FOLDER, model_info['ModelBlobId'])

    if not os.path.exists(models_folder):
        model_url = make_blob_url(
            model_info, 'ModelBucket', 'ModelBlobId')
        models_folder = write_model_to_temporary_folder(
            oauth, model_url, models_folder)

    prediction_parameters['ModelsFolder'] = models_folder


def prepare_model_info(prediction_parameters, model_info):
    """
    Method to update prediction parameters by using model info

    :param prediction_parameters: prediction parameters
            (dict using for making predictions)
    :param model_info: information about model,
            in general loaded from blobstorage
    :type prediction_parameters: dict
    :type model_info: dict
    """

    prediction_parameters['Fingerprints'] = model_info['Fingerprints']
    prediction_parameters['ClassName'] = model_info['ClassName']
    prediction_parameters['DensityMean'] = model_info['DensityMean']
    prediction_parameters['DensityStd'] = model_info['DensityStd']
    prediction_parameters['DistanceMean'] = model_info['DistanceMean']
    prediction_parameters['DistanceStd'] = model_info['DistanceStd']
    prediction_parameters['TrainShape'] = model_info['TrainShape']
    prediction_parameters['Modi'] = model_info['Modi']
    prediction_parameters['ModelType'] = model_info['ModelType']
    prediction_parameters['ModelCode'] = model_info['ModelCode']


def cache_model(models_folder_path):
    """
    Method to load model files (with needed objects sich as graph or session),
    additional models files (such as scaler, density model etc) from HDD to
    in-memory storage.
    Using to fast access in predictions or something else

    :param models_folder_path: path to folder with models and additional files
    :return: dict with linked models data ('in-memory storage' object), which
            can be used later for fast predictions or something else
    :type models_folder_path: str
    :rtype: dict
    """

    cached_files = {
        'models': list(),
        'models_names': list(),
        'graphs': list(),
        'sessions': list(),
        'models_number': 0,
        'distance_matrix': None,
        'scaler': None,
        'train_mean': None,
        'density_model': None,
        'k_means': None
    }

    cache_model_files(cached_files, models_folder_path)
    cache_additional_files(cached_files, models_folder_path)

    return cached_files


def cache_model_files(cached_files, models_folder_path):
    """
    Method to store model files (*.sav or *h5) in memory
    Also load graphs and sessions to storage, it 'must have' to using model
    Process data from HDD to memory storage ONLY if model NOT in storage yet

    :param cached_files: in-memory storage
    :param models_folder_path: path to folder with models files
    :type cached_files: dict
    :type models_folder_path: str
    """

    start_timer = time()
    for model_file_path in glob.glob('{}/*'.format(models_folder_path)):
        model_file_name = model_file_path.split('/')[-1]
        if model_file_name in MODEL_ADDITIONAL_FILES:
            continue

        model, graph, session = prepare_model_by_path(model_file_path)
        cached_files['models_names'].append(model_file_name)
        cached_files['models'].append(model)
        cached_files['graphs'].append(graph)
        cached_files['sessions'].append(session)
        cached_files['models_number'] += 1
    LOGGER.info('MODELS LOAD: {} sec'.format(time() - start_timer))


def cache_additional_files(cached_files, models_folder_path):
    """
    Method to store additional model files (such as scaler, density model etc)
    in memory. Process data from HDD to memory storage

    :param cached_files: in-memory storage
    :param models_folder_path: path to folder with additional models files
    :type cached_files: dict
    :type models_folder_path: str
    """

    start_timer = time()
    cached_files['train_mean'] = numpy.loadtxt(
        '{}/{}'.format(models_folder_path, TRAIN_MEAN_FILENAME))
    LOGGER.info('TRAIN MEAN LOAD: {} sec'.format(time() - start_timer))

    start_timer = time()
    distance_matrix_path = '{}/{}'.format(
        models_folder_path, DISTANCE_MATRIX_FILENAME)
    cached_files['distance_matrix'] = sparse.load_npz(
        distance_matrix_path
    ).todense()
    LOGGER.info('DISTANCE MATRIX LOAD: {} sec'.format(time() - start_timer))

    start_timer = time()
    cached_files['density_model'] = joblib.load(
        '{}/{}'.format(models_folder_path, DENSITY_MODEL_FILENAME))
    LOGGER.info('DENSITY MODEL LOAD: {} sec'.format(time() - start_timer))

    start_timer = time()
    cached_files['scaler'] = None
    scaler_path = '{}/{}'.format(models_folder_path, SCALER_FILENAME)
    if scaler_path:
        cached_files['scaler'] = joblib.load(scaler_path)
    LOGGER.info('SCALER LOAD: {} sec'.format(time() - start_timer))

    start_timer = time()
    cached_files['k_means'] = joblib.load(
        '{}/{}'.format(models_folder_path, K_MEANS_FILENAME))
    LOGGER.info('K MEANS LOAD: {} sec'.format(time() - start_timer))


def prepare_model_by_path(model_path):
    """
    Method to load model in memory (*.sav or *h5 file) with needed graph and
    session values. Clear graph and session before load each model.
    Return all needed objects to make predictions using loaded model, such as
    loaded model, graph and session objects

    :param model_path:
    :return: loaded model, graph for model and session for model
    :type model_path: str
    """

    # dnn file extension
    if '.h5' in model_path:
        keras.backend.clear_session()
        model = keras.models.load_model(
            model_path,
            custom_objects={'coeff_determination': coeff_determination}
        )

        graph = tensorflow.get_default_graph()
        session = keras.backend.get_session()

    # classic file extension
    elif '.sav' in model_path:
        model = joblib.load(model_path)
        graph = tensorflow.get_default_graph()
        session = keras.backend.get_session()

    # unknown extension
    else:
        raise ValueError('Unknown model file name: {}'.format(model_path))

    return model, graph, session


def get_molecules_from_sdf_bytes(dataset):
    """
    Method which make RDKit molecules from dataset bytes-object

    :param dataset: bytearray with molecules
    :return: list of RDKit molecules
    :type dataset: bytearray
    :rtype: list
    """

    stream = io.BytesIO(dataset)
    supplier = Chem.ForwardSDMolSupplier(stream)
    molecules = [x for x in supplier if x]

    return molecules


def molecules_from_mol_strings(strings_list):
    """
    Method to make RDKit molecules from mol strings
    Return RDKit molecules list

    :param strings_list: list of mol strings
    :return: RDKit molecules list
    :type strings_list: list
    :rtype: list
    """

    molecules = []
    for string in strings_list:
        molecules.append(Chem.MolFromMolBlock(string))

    return molecules


def molecules_from_smiles(smiles_list):
    """
    Method to make RDKit molecules from smiles
    Return RDKit molecules list

    :param smiles_list: list of SMILES strings
    :return: RDKit molecules list
    :type smiles_list: list
    :rtype: list
    """

    molecules = []
    for smiles in smiles_list:
        molecules.append(Chem.MolFromSmiles(smiles))

    return molecules


def clear_models_folder(temporary_folder=TEMP_FOLDER):
    """
    Method to clear temporary models folder before upload models
    Remove all in folder!!

    :param temporary_folder: path to temporary folder
    :type temporary_folder: str
    """

    # make ssp folder path
    models_folder = '{}/SSP_temp_models'.format(temporary_folder)
    # remove temporary directory
    shutil.rmtree(models_folder, ignore_errors=True)


def TMP_from_numpy_by_field_name(ndarray, field_name, equal_values):
    """
    Temporary method to get columns from numpy array using column name and
    filter values

    :param ndarray: numpy ndarray with 'value' and 'name' fields,
            which should be saved to csv file
    :param field_name: name of field to search in
    :param equal_values: values to search in ndarray by field name
    :type field_name: str
    :type equal_values: list
    :return: filtered numpy ndarray
    """

    filtered_list = list()
    for value in equal_values:
        filtered_values = ndarray[numpy.where(ndarray[field_name] == value)]
        if len(filtered_values) > 0:
            filtered_list.append(filtered_values)

    return numpy.array(filtered_list)


def numpy_to_csv(numpy_array, csv_path):
    """
    Temporary method to save numpy ndarray 'value' to cvs file with headers
    Using csv_path to save file

    :param numpy_array: numpy ndarray with 'value' and 'name' fields,
            which should be saved to csv file
    :param csv_path: csv file path to save numpy array
    :type csv_path: str
    """

    with open(csv_path, 'w') as csv_file:
        csv_file.write(','.join(numpy_array[0]['name']) + '\n')
        numpy.savetxt(csv_file, numpy_array['value'], fmt='%s', delimiter=',')


def get_inchi_key(molecule):
    """
    Method to get InChi from rdkit 'molecule' entity

    :param molecule: rdkit 'molecule' entity
    :return: inchi_key text string
    :rtype: str
    """

    inchi = Chem.MolToInchi(molecule)
    inchi_key = Chem.InchiToInchiKey(inchi)

    return inchi_key


def coeff_determination(y_true, y_predicted):
    """
    Method for calculate determination coefficient

    :param y_true: true y value(s)
    :param y_predicted: predicted y value(s)
    :return: determination coefficient
    """

    ss_res = keras.backend.sum(keras.backend.square(y_true - y_predicted))
    ss_tot = keras.backend.sum(
        keras.backend.square(y_true - keras.backend.mean(y_true)))

    return 1 - ss_res/(ss_tot + keras.backend.epsilon())


def single_fold_selector(x, y):
    """
    Method to get all indexes from x array.
    Like stratified split but for single fold only

    :param x: array of x values
    :param y: array of y values
    :type x: numpy.ndarray
    :type y: numpy.ndarray
    :return: indexes for train and valid sets
    :rtype: list
    """

    return [[numpy.arange(x.shape[0]), numpy.arange(x.shape[0])]]


def get_distance(x_predict, centroid, train_mean, train_shape):
    """

    :param x_predict: feature vector for prediction
    :param centroid: centroid of train set
    :param train_mean: mean value of train set
    :param train_shape: shape of train set
    :return: Mahalanobis distance between feature vector and x_train
    """
    # TODO description there
    variance_covariance_matrix = numpy.matmul(
        numpy.transpose(centroid), centroid
    ) / train_shape
    transposed_covariance_matrix = numpy.transpose(variance_covariance_matrix)
    Mahalanobis_distance = numpy.sqrt(
        numpy.matmul(
            numpy.matmul(
                x_predict - train_mean, transposed_covariance_matrix
            ),
            numpy.transpose(x_predict - train_mean)
        )
    )

    return Mahalanobis_distance
