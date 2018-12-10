import os
import unittest

from learner.algorithms import (
    CODES, TRAINER_CLASS, ALGORITHM, RANDOM_FOREST_REGRESSOR,
    LOGISTIC_REGRESSION
)
from processor import sdf_to_csv


class TestTestSetSize(unittest.TestCase):
    def setUp(self):
        self.fingerprints = [
            {'Type': 'DESC'},
            {'Type': 'MACCS'},
            {'Type': 'AVALON', 'Size': 512},
            {'Type': 'FCFC', 'Size': 512, 'Radius': 3}
        ]
        self.test_set_size = 0
        self.n_split = 2
        self.sdf_file_path = '{}/resources/DNN_data_solubility.sdf'.format(
            os.path.dirname(os.path.abspath(__file__)))
        self.temporary_folder = os.environ['OSDR_TEMP_FILES_FOLDER']

    def test_logistic_regression_test_size_zero(self):
        """
        Test for classic classification (logistic regression) test size zero
        """

        classname = 'Soluble'
        dataframe = sdf_to_csv(
            self.sdf_file_path, self.fingerprints, class_name_list=classname)
        classic_classifier = ALGORITHM[TRAINER_CLASS][LOGISTIC_REGRESSION](
            self.sdf_file_path, classname, dataframe, subsample_size=1.0,
            test_set_size=self.test_set_size, seed=0, fptype=self.fingerprints,
            scale='minmax', output_path=self.temporary_folder,
            n_split=self.n_split
        )
        classic_classifier.train_model(CODES[LOGISTIC_REGRESSION])

        metrics = classic_classifier.metrics[LOGISTIC_REGRESSION]['mean']
        true_metrics = {
            ('train', 'AUC'): 0.99,
            ('train', 'ACC'): 0.97,
            ('train', 'f1-score'): 0.98,
            ('train', 'Cohen_Kappa'): 0.84,
            ('train', 'Matthews_corr'): 0.84,
            ('train', 'Precision'): 0.97,
            ('train', 'Recall'): 0.99,
            ('validation', 'AUC'): 0.97,
            ('validation', 'ACC'): 0.95,
            ('validation', 'f1-score'): 0.97,
            ('validation', 'Cohen_Kappa'): 0.70,
            ('validation', 'Matthews_corr'): 0.73,
            ('validation', 'Precision'): 0.94,
            ('validation', 'Recall'): 0.99
        }

        self.assertDictAlmostEqual(metrics, true_metrics, delta=0.1)

    def test_random_forest_regressor_test_size_zero(self):
        """
        Test for classic regression (random forest regressor) test size zero
        """

        valuename = 'logS'
        dataframe = sdf_to_csv(
            self.sdf_file_path, self.fingerprints, value_name_list=valuename)
        classic_regressor = ALGORITHM[TRAINER_CLASS][RANDOM_FOREST_REGRESSOR](
            self.sdf_file_path, valuename, dataframe, scale='minmax', seed=0,
            test_set_size=self.test_set_size, fptype=self.fingerprints,
            output_path=self.temporary_folder, n_split=self.n_split,
            subsample_size=1.0
        )
        classic_regressor.train_model(CODES[RANDOM_FOREST_REGRESSOR])

        metrics = classic_regressor.metrics[RANDOM_FOREST_REGRESSOR]['mean']
        true_metrics = {
            ('train', 'RMSE'): 0.42,
            ('train', 'MAE'): 0.32,
            ('train', 'R2'): 0.95,
            ('validation', 'R2'): 0.90,
            ('validation', 'RMSE'): 0.63,
            ('validation', 'MAE'): 0.48
        }

        self.assertDictAlmostEqual(metrics, true_metrics, delta=0.1)

    def assertDictAlmostEqual(self, dict_1, dict_2, delta=None):
        self.assertEqual(dict_1.keys(), dict_2.keys())

        for key, value_1 in dict_1.items():
            value_2 = dict_2[key]
            self.assertAlmostEqual(value_1, value_2, delta=delta)

    def run(self, result=None):
        """ Stop after first error """
        if result.errors:
            raise Exception(result.errors[0][1])
        elif result.failures:
            raise Exception(result.failures[0][1])
        else:
            super().run(result)


if __name__ == '__main__':
    unittest.main()
