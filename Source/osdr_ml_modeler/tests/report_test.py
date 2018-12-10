import os
import unittest

from learner.algorithms import (
    CODES, TRAINER_CLASS, ALGORITHM, RANDOM_FOREST_CLASSIFIER,
    NEIGHBORS_REGRESSOR
)
from learner.plotters import radar_plot, distribution_plot
from processor import sdf_to_csv


class TestReport(unittest.TestCase):
    def setUp(self):
        self.fingerprints = [
            {'Type': 'ENUM2CAN'},
            {'Type': 'PATTERN', 'Size': 512},
            {'Type': 'FCFC_CHIRALITY', 'Size': 512, 'Radius': 3},
            {'Type': 'ECFP', 'Size': 512, 'Radius': 3}
        ]
        self.test_set_size = 0.2
        self.n_split = 2
        self.sdf_file_path = '{}/resources/DNN_data_solubility.sdf'.format(
            os.path.dirname(os.path.abspath(__file__)))
        self.temporary_folder = os.environ['OSDR_TEMP_FILES_FOLDER']

    def test_random_forest_classifier_report(self):
        valuename = 'Soluble'
        dataframe = sdf_to_csv(
            self.sdf_file_path, self.fingerprints, class_name_list=valuename)
        classic_classifier = ALGORITHM[TRAINER_CLASS][
            RANDOM_FOREST_CLASSIFIER
        ](
            self.sdf_file_path, valuename, dataframe, subsample_size=0.2,
            test_set_size=self.test_set_size, seed=0, fptype=self.fingerprints,
            scale='robust', output_path=self.temporary_folder,
            n_split=self.n_split
        )
        classic_classifier.train_model(CODES[RANDOM_FOREST_CLASSIFIER])
        classic_classifier.make_applicability_domain()
        plots = classic_classifier.cv_model.make_plots()
        path_to_csv = classic_classifier.make_perfomance_csv()
        path_to_qmrf_report = classic_classifier.make_qmrf_report(
            None, None).split('/')[-1]
        path_to_archive = classic_classifier.cv_model.compress_models()
        path_to_archive = classic_classifier.compress_additional_files(
            path_to_archive).split('/')[-1]
        path_to_radar_plot = radar_plot(
            path_to_csv, classic_classifier.sub_folder,
            classic_classifier.bins, titlename='Random Forest'
        ).split('/')[-1]
        path_to_csv = path_to_csv.split('/')[-1]

        prepare_plots(plots)
        true_plots = {
            '1': {
                'roc_plot_path': 'Random_Forest_Classifier_fold_1_ROC_plot.png',
                'cm_plot_path': 'Random_Forest_Classifier_fold_1_confusion.png',
                'thumbnail_plot_path': None
            },
            '2': {
                'roc_plot_path': 'Random_Forest_Classifier_fold_2_ROC_plot.png',
                'cm_plot_path': 'Random_Forest_Classifier_fold_2_confusion.png',
                'thumbnail_plot_path': None
            },
            'mean': {
                'roc_plot_path': 'Random_Forest_Classifier_fold_mean_ROC_plot.png',
                'cm_plot_path': 'Random_Forest_Classifier_fold_mean_confusion.png',
                'thumbnail_plot_path': 'Random_Forest_Classifier_thumbnail_image.jpg'
            }
        }

        self.assertDictEqual(plots, true_plots)
        self.assertEqual(path_to_radar_plot, 'radar_plot.png')
        self.assertEqual(
            path_to_csv, 'Random Forest Classifier_DNN_data_solubility.csv')
        self.assertEqual(
            path_to_qmrf_report, 'Random_Forest_Classifier_QMRF_report.pdf')
        self.assertEqual(path_to_archive, 'Random_Forest_Classifier.zip')

    def test_neighbors_regressor_report(self):
        valuename = 'logS'
        dataframe = sdf_to_csv(
            self.sdf_file_path, self.fingerprints, value_name_list=valuename)
        classic_regressor = ALGORITHM[TRAINER_CLASS][NEIGHBORS_REGRESSOR](
            self.sdf_file_path, valuename, dataframe, scale='minmax', seed=0,
            test_set_size=self.test_set_size, fptype=self.fingerprints,
            output_path=self.temporary_folder, n_split=self.n_split,
            subsample_size=0.2
        )
        classic_regressor.train_model(CODES[NEIGHBORS_REGRESSOR])
        classic_regressor.make_applicability_domain()
        plots = classic_regressor.cv_model.make_plots()
        path_to_csv = classic_regressor.make_perfomance_csv().split('/')[-1]
        path_to_qmrf_report = classic_regressor.make_qmrf_report(
            None, None).split('/')[-1]
        path_to_archive = classic_regressor.cv_model.compress_models()
        path_to_archive = classic_regressor.compress_additional_files(
            path_to_archive).split('/')[-1]
        path_to_distrubution_plot = distribution_plot(
            classic_regressor, model_name='Nearest Neighbors').split('/')[-1]

        prepare_plots(plots)
        true_plots = {
            '1': {
                'regression_results_test': 'K Neighbors Regressor_logS_test_fold_1_regression_plot.png',
                'regression_results_train': 'K Neighbors Regressor_logS_train_fold_1_regression_plot.png',
                'regression_results_valid': 'K Neighbors Regressor_logS_validation_fold_1_regression_plot.png',
                'thumbnail_plot_path': None
            },
            '2': {
                'regression_results_test': 'K Neighbors Regressor_logS_test_fold_2_regression_plot.png',
                'regression_results_train': 'K Neighbors Regressor_logS_train_fold_2_regression_plot.png',
                'regression_results_valid': 'K Neighbors Regressor_logS_validation_fold_2_regression_plot.png',
                'thumbnail_plot_path': None
            },
            'mean': {
                'regression_results_test': 'K Neighbors Regressor_logS_test_fold_mean_regression_plot.png',
                'regression_results_train': 'K Neighbors Regressor_logS_train_fold_mean_regression_plot.png',
                'regression_results_valid': 'K Neighbors Regressor_logS_validation_fold_mean_regression_plot.png',
                'thumbnail_plot_path': 'K Neighbors Regressor_logS_thumbnail_image.jpg'
            }
        }

        self.assertDictEqual(plots, true_plots)
        self.assertEqual(
            path_to_distrubution_plot,
            'Nearest Neighbors_train_test_distribution.png'
        )
        self.assertEqual(
            path_to_csv, 'K Neighbors Regressor_DNN_data_solubility.csv')
        self.assertEqual(
            path_to_qmrf_report, 'K_Neighbors_Regressor_QMRF_report.pdf')
        self.assertEqual(path_to_archive, 'K_Neighbors_Regressor.zip')

    def run(self, result=None):
        """ Stop after first error """
        if result.errors:
            raise Exception(result.errors[0][1])
        elif result.failures:
            raise Exception(result.failures[0][1])
        else:
            super().run(result)


def prepare_plots(plots):
    for fold, plots_values in plots.items():
        for plot_type, plot_path in plots_values.items():
            if plot_path:
                plots[fold][plot_type] = plot_path.split('/')[-1]


if __name__ == '__main__':
    unittest.main()
