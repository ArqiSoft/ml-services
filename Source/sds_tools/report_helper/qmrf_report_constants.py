"""
Method which contain data and methods for create QMRF training report
"""

import datetime

from general_helper import get_user_info_from_osdr
from learner.algorithms import (
    model_type_by_name, algorithm_help_by_name, CLASSIFIER, REGRESSOR
)


TITLE = 'title'
CONTENT = 'content'

QSAR_IDENTIFIER = {
    TITLE: {
        1: 'QSAR_identifier',
        1.1: 'QSAR_identifier (title)',
        1.2: 'Other related models',
        1.3: 'Software coding the model'
    },
    CONTENT: {
        1: 'QSAR_identifier content',
        1.1: 'QSAR_identifier (title) content',
        1.2: 'No related models',
        1.3: 'Python\n'
             'Python is a powerful programming language\n'
             'https://www.python.org/\n\n'
             'RDKit\n'
             'RDKit is a chemestry library for Python\n'
             'http://www.rdkit.org/\n\n'
             'Tensorflow\n'
             'Tensorflow is a machine learning library\n'
             'https://www.tensorflow.org/\n\n'
             'Keras\n'
             'Keras is a deep learning library for Python\n'
             'https://keras.io/'
    }
}

GENERAL_INFORATION = {
    TITLE: {
        2: 'General information',
        2.1: 'Date of QMRF',
        2.2: 'QMRF author(s) and contact deatils',
        2.3: 'Date of QMRF update(s)',
        2.4: 'QMRF update(s)',
        2.5: 'Model developer(s) and contact details',
        2.6: 'Date of model development and/or publications',
        2.7: 'Reference(s) to main scientific papers and/or software package',
        2.8: 'Availability of information about the model',
        2.9: 'Availability of another QMRF for exactly the same model'
    },
    CONTENT: {
        2: 'General information content',
        2.1: datetime.datetime.now().date().strftime('%d %B %Y'),
        2.2: 'QMRF author(s) and contact deatils content',
        2.3: 'HARDCODE!!!! Empty field',
        2.4: 'HARDCODE!!!! Empty field',
        2.5: 'SCIENCE DATA SOFTWARE, LLC, Rockville, MD, United States, science@your-company.com',
        2.6: datetime.datetime.now().date().strftime('%d %B %Y'),
        2.7: 'Reference(s) to main scientific papers and/or software package content',
        2.8: 'Availability of information about the model content',
        2.9: 'Not to date'
    }
}

OECD_PRINCIPLE_1 = {
    TITLE: {
        3: 'Defining the endpoint - OECD Principle 1',
        3.1: 'Species',
        3.2: 'Endpoint',
        3.3: 'Comment on endpoint',
        3.4: 'Endpoint units',
        3.5: 'Dependent variable',
        3.6: 'Experimental protocol',
        3.7: 'Endpoint data quality and variability',
    },
    CONTENT: {
        3: 'Defining the endpoint - OECD Principle 1 content',
        3.1: 'Not applicable',
        3.2: 'HARDCODE!!!! Endpoint content',
        3.3: 'HARDCODE!!!! Comment on endpoint content',
        3.4: 'HARDCODE!!!! Endpoint units content',
        3.5: 'Dependent variable content',
        3.6: 'HARDCODE!!!! Experimental protocol content',
        3.7: 'SEMIHARDCODE!!!! Endpoint data quality and variability content',
    }
}

OECD_PRINCIPLE_2 = {
    TITLE: {
        4: 'Defining the algorithm - OECD Principle 2',
        4.1: 'Type of model',
        4.2: 'Explicit algorithm',
        4.3: 'Descriptors in the model',
        4.4: 'Descriptor selection',
        4.5: 'Algorithm and descriptor generation',
        4.6: 'Software name and version for descriptor generation',
        4.7: 'Chemicals/Descriptors ratio',
    },
    CONTENT: {
        4: 'Defining the algorithm - OECD Principle 2 content',
        4.1: 'Type of model content',
        4.2: 'Explicit algorithm content',
        4.3: 'Descriptors in the model content',
        4.4: 'Not available',
        4.5: 'Algorithm and descriptor generation content',
        4.6: 'RDKIT v.2017.03.03\nhttp://www.rdkit.org',
        4.7: 'Chemicals/Descriptors ratio content',
    }
}

OECD_PRINCIPLE_3 = {
    TITLE: {
        5: 'Defining the applicability domain - OECD Principle 3',
        5.1: 'Description of the applicability domain of the model',
        5.2: 'Method used to assess the applicability domain',
        5.3: 'Software name and version for applicability domain assessment',
        5.4: 'Limits of applicability',
    },
    CONTENT: {
        5: 'Defining the applicability domain - OECD Principle 3 content',
        5.1: 'HARDCODE!!!! Description of the applicability domain of the model content',
        5.2: 'HARDCODE!!!! Method used to assess the applicability domain content',
        5.3: 'OSDR 0.14',
        5.4: 'HARDCODE!!!! Limits of applicability content',
    }
}

OECD_PRINCIPLE_4_INTERNAL_VALIDATION = {
    TITLE: {
        6: 'Internal validation - OECD Principle 4',
        6.1: 'Availability of the training set',
        6.2: 'Available information for the training set',
        6.3: 'Data for each descriptor variable for the training set',
        6.4: 'Data for the dependent variable for the training set',
        6.5: 'Other information about the training set',
        6.6: 'Pre-processing of data before modelling',
        6.7: 'Statistics for goodness-of-fit',
        6.8: 'Robustness - Statistics obtained by leave-one-out cross-validation',
        6.9: 'Robustness - Statistics obtained by leave-many-out cross-validation',
        6.10: 'Robustness - Statistics obtained by Y-scrambling',
        6.11: 'Robustness - Statistics obtained by bootstrap',
        6.12: 'Robustness - Statistics obtained by other methods'
    },
    CONTENT: {
        6: 'Internal validation - OECD Principle 4 content',
        6.1: 'Yes',
        6.2: 'CAS RN: Yes\n'
             'Chemical Name: Yes\n'
             'Smiles: Yes\n'
             'Formula: No\n'
             'INChI: Yes\n'
             'MOL file: Yes',
        6.3: 'All',
        6.4: 'All',
        6.5: 'Other information about the training set content',
        6.6: 'No preprocessing of the values.',
        6.7: 'Statistics for goodness-of-fit content',
        6.8: 'HARDCODE!!!! Robustness - Statistics obtained by leave-one-out cross-validation content',
        6.9: 'HARDCODE!!!! Robustness - Statistics obtained by leave-many-out cross-validation content',
        6.10: 'HARDCODE!!!! Robustness - Statistics obtained by Y-scrambling content',
        6.11: 'HARDCODE!!!! Robustness - Statistics obtained by bootstrap content',
        6.12: 'HARDCODE!!!! Robustness - Statistics obtained by other methods content'
    }
}

OECD_PRINCIPLE_4_EXTERNAL_VALIDATION = {
    TITLE: {
        7: 'External validation - OECD Principle 4',
        7.1: 'Availability of the external validation set',
        7.2: 'Available information for the external validation set',
        7.3: 'Data for each descriptor variable for the external validation set',
        7.4: 'Data for the dependent variable for the external validation set',
        7.5: 'Other information about the external validation set',
        7.6: 'Experimental design of test set',
        7.7: 'Predictivity - Statistics obtained by external validation',
        7.8: 'Predictivity - Assessment of the external validation set',
        7.9: 'Comments on the external validation of the model',
    },
    CONTENT: {
        7: 'External validation - OECD Principle 4 content',
        7.1: 'Yes',
        7.2: 'CAS RN: Yes\n'
             'Chemical Name: Yes\n'
             'Smiles: Yes\n'
             'Formula: No\n'
             'INChI: Yes\n'
             'MOL file: Yes',
        7.3: 'All',
        7.4: 'All',
        7.5: 'Other information about the external validation set content',
        7.6: 'Experimental design of test set content',
        7.7: 'Predictivity - Statistics obtained by external validation content',
        7.8: 'HARDCODE!!!! Predictivity - Assessment of the external validation set content',
        7.9: 'The choice of proportions between the training set and the validation\n'
             'set as well as the splitting method helped in accurately evaluating the\n'
             'model and covering most of the training set chemical space. This goal\n'
             'was accomplished without the need to do a structural sampling that\n'
             'usually shows over-optimistic evaluation of the predictivity or a\n'
             'complete random selection that risks biasing the evaluation towards a\n'
             'certain region of the chemical space.',
    }
}

OECD_PRINCIPLE_5 = {
    TITLE: {
        8: 'Providing a mechanistic interpretation - OECD Principle 5',
        8.1: 'Mechanistic basis of the model',
        8.2: 'A priori or a posteriori mechanistic interpretation',
        8.3: 'Other information about the mechanistic interpretation',
    },
    CONTENT: {
        8: 'Providing a mechanistic interpretation - OECD Principle 5 content',
        8.1: 'HARDCODE!!!! Mechanistic basis of the model content',
        8.2: 'HARDCODE!!!! A priori or a posteriori mechanistic interpretation content',
        8.3: 'HARDCODE!!!! Other information about the mechanistic interpretation content',
    }
}

MISCELLANEOUS_INFORMATION = {
    TITLE: {
        9: 'Miscellaneous information',
        9.1: 'Comments',
        9.2: 'Bibliography',
        9.3: 'Supporting information',
    },
    CONTENT: {
        9: 'Miscellaneous information content',
        9.1: 'HARDCODE!!!! Comments content',
        9.2: 'HARDCODE!!!! Bibliography content',
        9.3: 'HARDCODE!!!! Supporting information content',
    }
}

SUMMARY = {
    TITLE: {
        10: 'Summary (JRC QSAR Model Database)',
        10.1: 'QMRF number',
        10.2: 'Publication date',
        10.3: 'Keywords',
        10.4: 'Comments'
    },
    CONTENT: {
        10: 'Summary (JRC QSAR Model Database) content',
        10.1: 'QMRF number content',
        10.2: 'HARDCODE???? Current date there? Publication date content',
        10.3: 'HARDCODE!!!! Keywords content',
        10.4: 'HARDCODE!!!! Comments content'
    }
}

ALL_GROUPS = [
    QSAR_IDENTIFIER, GENERAL_INFORATION, OECD_PRINCIPLE_1, OECD_PRINCIPLE_2,
    OECD_PRINCIPLE_3, OECD_PRINCIPLE_4_INTERNAL_VALIDATION,
    OECD_PRINCIPLE_4_EXTERNAL_VALIDATION, OECD_PRINCIPLE_5,
    MISCELLANEOUS_INFORMATION, SUMMARY
]


def make_initial_context():
    # TODO tmp
    initial_content = {'context': {}}
    for group in ALL_GROUPS:
        for number, title in group[TITLE].items():
            initial_content['context'][number] = {}

            formatted_title = '{}.{}'.format(number, title)
            content = None

            if isinstance(number, float):
                formatted_title += ':'
                content = group[CONTENT][number]

            initial_content['context'][number]['title'] = formatted_title
            initial_content['context'][number]['content'] = content

    return initial_content


def update_user_context(oauth, user_id, base_content):
    # TODO tmp
    try:
        response = get_user_info_from_osdr(oauth, user_id)
    except:
        response = None

    if response and response.status_code == 200:
        user_info = response.json()
    else:
        user_info = {
            'firstName': 'unknown',
            'lastName': 'unknown'
        }
    # user_info = dict()
    # user_info['firstName'] = 'User first name'
    # user_info['lastName'] = 'last name'
    user_info['affilation'] = 'user\'s affilation there'
    user_info['email'] = 'user@email.there'

    TMP_user_info = '{} {}, {}, {}'.format(
        user_info['firstName'], user_info['lastName'],
        user_info['affilation'], user_info['email']
    )

    base_content['context'][2.2]['content'] = TMP_user_info


def make_model_context(model_trainer, base_content, body):
    # TODO tmp
    current_mean_model_performance = model_trainer.metrics[
        model_trainer.cv_model.model_name]['mean']

    base_content['context'][1.1]['content'] = get_qsar_identifier(
        model_trainer)
    base_content['context'][2.8]['content'] = algorithm_help_by_name(
        model_trainer.cv_model.model_name)
    base_content['context'][3.2]['content'] = get_endpoint_content(
        model_trainer)
    # base_content['context'][3.4]['content'] = 'Endpoint units'
    base_content['context'][3.5]['content'] = model_trainer.prediction_target
    base_content['context'][3.7]['content'] = get_original_quality(
        model_trainer)
    base_content['context'][4.1]['content'] = model_trainer.cv_model.model_name
    base_content['context'][4.3]['content'] = get_descriptors(body)
    base_content['context'][4.5]['content'] = get_descriptors(body)
    base_content['context'][4.7]['content'] = get_descriptors_rate(
        model_trainer)
    base_content['context'][6.5]['content'] = get_molecules_counter(
        model_trainer, 'train')
    base_content['context'][6.7]['content'] = get_performance(
        current_mean_model_performance, 'train')
    base_content['context'][7.5]['content'] = get_molecules_counter(
        model_trainer, 'test')
    base_content['context'][7.7]['content'] = get_performance(
        current_mean_model_performance, 'test')


def get_descriptors(body):
    # TODO tmp
    fingerprints_list = []
    for fingerprint in body['Fingerprints']:
        fingerprints_list.append(fingerprint['Type'])

    fingerprints = ', '.join(fingerprints_list)
    rdkit_link = 'http://www.rdkit.org/docs/GettingStartedInPython.html#list-of-available-descriptors'
    TMP_message = 'Used descriptors/fingerprints: {}.\n\n{}'.format(
        fingerprints, rdkit_link)

    return TMP_message


def get_endpoint_content(model_trainer):
    # TODO tmp
    model_name = model_trainer.cv_model.model_name
    model_type = model_type_by_name(model_name)
    prediction_target = model_trainer.prediction_target
    if model_type == CLASSIFIER:
        TMP_message = 'classification of {} probability'.format(
            prediction_target)
    elif model_type == REGRESSOR:
        TMP_message = 'regression of {} variable'.format(prediction_target)
    else:
        raise ValueError('Unknown model type: {}'.format(model_type))

    return TMP_message


def get_descriptors_rate(model_trainer):
    # TODO tmp
    descriptors = model_trainer.dataframe.shape[1] - 1
    molecules_training_set = len(model_trainer.x_train)
    descriptors_rate = molecules_training_set / descriptors

    TMP_message = '{} chemicals (trainingset)/{} descriptors= {:.02f}'.format(
        molecules_training_set, descriptors, descriptors_rate
    )

    return TMP_message


def get_original_quality(model_trainer):
    # TODO tmp
    counter = model_trainer.dataframe.shape[0]
    TMP_message = 'SEMIHARDCODE!!!! can get numbers\n' \
                  ' The original data collected from the sdf file database ({} chemicals)'.format(counter)

    return TMP_message


def get_performance(current_model_performance, TMP_flag):
    # TODO tmp
    TMP_message = 'Performance in {}:\n'.format(TMP_flag)
    if ((TMP_flag, 'RMSE') in current_model_performance.keys() and
        (TMP_flag, 'R2') in current_model_performance.keys()):

        TMP_message += 'R2={}\n'.format(
            current_model_performance[(TMP_flag, 'RMSE')])
        TMP_message += 'RMSE={}'.format(
            current_model_performance[(TMP_flag, 'R2')])

    TMP_message = 'Performance in {}:\n'.format(TMP_flag)

    for key, value in current_model_performance.items():
        if TMP_flag not in key:
            continue

        metric_name = key[1]
        TMP_message += '{}={}\n'.format(metric_name, value)

    return TMP_message


def get_molecules_counter(model_trainer, TMP_flag):
    # TODO tmp
    if TMP_flag == 'test':
        counter = len(model_trainer.x_test)
    elif TMP_flag == 'train':
        counter = len(model_trainer.x_train)
    else:
        raise ValueError('Unknown set: {}'.format(TMP_flag))

    TMP_message = 'The {} set consists of {} chemicals.'.format(
        TMP_flag, counter)

    return TMP_message


def get_qsar_identifier(model_trainer):
    # TODO tmp
    model_name = model_trainer.cv_model.model_name
    model_type = model_type_by_name(model_name)
    training_target = model_trainer.prediction_target

    TMP_message = 'OSDR-model {} for {} variable'.format(
        model_type, training_target)

    return TMP_message
