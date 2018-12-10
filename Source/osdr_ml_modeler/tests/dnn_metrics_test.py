import os
import unittest

from learner.algorithms import (
    CODES,  TRAINER_CLASS, ALGORITHM, DNN_CLASSIFIER, DNN_REGRESSOR
)
from processor import sdf_to_csv


class TestDNNMetrics(unittest.TestCase):
    def setUp(self):
        self.fingerprints = [
            {'Type': 'CAN2ENUM'},
            {'Type': 'RDK', 'Size': 512},
            {'Type': 'LAYERED', 'Size': 512},
            {'Type': 'FCFC', 'Size': 512, 'Radius': 3}
        ]
        self.test_set_size = 0.2
        self.n_split = 2
        self.sdf_file_path = '{}/resources/DNN_data_solubility.sdf'.format(
            os.path.dirname(os.path.abspath(__file__)))
        self.temporary_folder = os.environ['OSDR_TEMP_FILES_FOLDER']

    def test_dnn_classification_metrics(self):
        """
        Test for DNN classification metrics
        """

        valuename = 'Soluble'
        dataframe = sdf_to_csv(
            self.sdf_file_path, self.fingerprints, class_name_list=valuename)
        dnn_classifier = ALGORITHM[TRAINER_CLASS][DNN_CLASSIFIER](
            self.sdf_file_path, valuename, dataframe, subsample_size=1.0,
            test_set_size=self.test_set_size, seed=0, fptype=self.fingerprints,
            scale='minmax', output_path=self.temporary_folder,
            n_split=self.n_split
        )
        dnn_classifier.train_model(CODES[DNN_CLASSIFIER])

        metrics = dnn_classifier.metrics[DNN_CLASSIFIER]['mean']
        true_metrics = {
            ('train', 'AUC'): 0.99,
            ('train', 'ACC'): 0.97,
            ('train', 'f1-score'): 0.98,
            ('train', 'Cohen_Kappa'): 0.84,
            ('train', 'Matthews_corr'): 0.85,
            ('train', 'Precision'): 0.96,
            ('train', 'Recall'): 0.99,
            ('test', 'AUC'): 0.96,
            ('test', 'ACC'): 0.94,
            ('test', 'f1-score'): 0.96,
            ('test', 'Cohen_Kappa'): 0.67,
            ('test', 'Matthews_corr'): 0.69,
            ('test', 'Precision'): 0.94,
            ('test', 'Recall'): 0.99,
            ('validation', 'AUC'): 0.89,
            ('validation', 'ACC'): 0.92,
            ('validation', 'f1-score'): 0.95,
            ('validation', 'Cohen_Kappa'): 0.55,
            ('validation', 'Matthews_corr'): 0.57,
            ('validation', 'Precision'): 0.93,
            ('validation', 'Recall'): 0.98
        }

        self.assertDictAlmostEqual(metrics, true_metrics, delta=0.3)

    def test_dnn_regression_metrics(self):
        """
        Test for DNN regression metrics
        """

        valuename = 'logS'
        dataframe = sdf_to_csv(
            self.sdf_file_path, self.fingerprints, value_name_list=valuename)
        dnn_regressor = ALGORITHM[TRAINER_CLASS][DNN_REGRESSOR](
            self.sdf_file_path, valuename, dataframe, scale='minmax', seed=0,
            test_set_size=self.test_set_size, fptype=self.fingerprints,
            output_path=self.temporary_folder, n_split=self.n_split,
            subsample_size=1.0
        )
        dnn_regressor.train_model(CODES[DNN_REGRESSOR])

        metrics = dnn_regressor.metrics[DNN_REGRESSOR]['mean']
        true_metrics = {
            ('train', 'RMSE'): 0.60,
            ('train', 'MAE'): 0.46,
            ('train', 'R2'): 0.91,
            ('test', 'MAE'): 0.62,
            ('test', 'R2'): 0.84,
            ('test', 'RMSE'): 0.82,
            ('validation', 'R2'): 0.81,
            ('validation', 'RMSE'): 0.87,
            ('validation', 'MAE'): 0.67
        }

        self.assertDictAlmostEqual(metrics, true_metrics, delta=0.3)

    def run(self, result=None):
        """ Stop after first error """
        if result.errors:
            raise Exception(result.errors[0][1])
        elif result.failures:
            raise Exception(result.failures[0][1])
        else:
            super().run(result)

    def assertDictAlmostEqual(self, dict_1, dict_2, delta=None):
        self.assertEqual(dict_1.keys(), dict_2.keys())

        for key, value_1 in dict_1.items():
            value_2 = dict_2[key]
            self.assertAlmostEqual(value_1, value_2, delta=delta)


if __name__ == '__main__':
    unittest.main()
