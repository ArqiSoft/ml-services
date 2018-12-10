"""
Module which contain methods for make classifying or regression reports text
Contain needed constants with reports and metrics names
"""

TRAINING_CSV_METRICS = 'csv_report'
QMRF_REPORT = 'qmrf_report'
MODEL_PDF_REPORT = 'current_model_pdf_report'
ALL_MODELS_TRAINING_CSV_METRICS = 'all_models_csv_metrics'


def report_text_classifier(classifier):
    """
    Method for make data for classifying training report
    Add data to classifier attribute 'dataset_info'

    :param classifier: classifier object
    """

    classifier.template_tags['dataset_info'].append('Original dataset:')
    major_class = 1
    minor_class = 0
    original_dataset = list(classifier.parameter_dataframe)

    classifier.template_tags['dataset_info'].append(
        'Major class is: {} sample size: {}'.format(
            major_class, original_dataset.count(major_class)
        )
    )

    classifier.template_tags['dataset_info'].append(
        'Minor class is: {} sample size: {}'.format(
            minor_class, original_dataset.count(minor_class)
        )
    )

    classifier.template_tags['dataset_info'].append(
        'Original major class sample size is: {}'.format(
            len(original_dataset)))


def report_text_regressor(regressor):
    """
    Method for make data for regression training report
    Add data to regressor attribute 'dataset_info'

    :param regressor: regressor object
    """

    regressor.template_tags['dataset_info'].append('Original dataset:')

    regressor.template_tags['dataset_info'].append(
        regressor.parameter_dataframe)

    # regressor.template_tags['dataset_info'].append(
    #     regressor.df_features)
