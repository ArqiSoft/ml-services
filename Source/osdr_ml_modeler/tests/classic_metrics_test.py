import os
import unittest

from learner.algorithms import (
    CODES, NAIVE_BAYES, ELASTIC_NETWORK, TRAINER_CLASS, ALGORITHM
)
from processor import sdf_to_csv


class TestClassicMetrics(unittest.TestCase):
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

    def test_naive_bayes_metrics(self):
        """
        Test for classic classification metrics (naive Bayes)
        """

        valuename = 'Soluble'
        dataframe = sdf_to_csv(
            self.sdf_file_path, self.fingerprints, class_name_list=valuename)
        classic_classifier = ALGORITHM[TRAINER_CLASS][NAIVE_BAYES](
            self.sdf_file_path, valuename, dataframe, subsample_size=1.0,
            test_set_size=self.test_set_size, seed=0, fptype=self.fingerprints,
            scale='robust', output_path=self.temporary_folder,
            n_split=self.n_split
        )
        classic_classifier.train_model(CODES[NAIVE_BAYES])

        metrics = classic_classifier.metrics[NAIVE_BAYES]['mean']
        true_metrics = {
            ('train', 'AUC'): 0.85,
            ('train', 'ACC'): 0.81,
            ('train', 'f1-score'): 0.88,
            ('train', 'Cohen_Kappa'): 0.41,
            ('train', 'Matthews_corr'): 0.47,
            ('train', 'Precision'): 0.97,
            ('train', 'Recall'): 0.8,
            ('test', 'AUC'): 0.89,
            ('test', 'ACC'): 0.81,
            ('test', 'f1-score'): 0.88,
            ('test', 'Cohen_Kappa'): 0.46,
            ('test', 'Matthews_corr'): 0.53,
            ('test', 'Precision'): 0.98,
            ('test', 'Recall'): 0.79,
            ('validation', 'AUC'): 0.83,
            ('validation', 'ACC'): 0.81,
            ('validation', 'f1-score'): 0.88,
            ('validation', 'Cohen_Kappa'): 0.40,
            ('validation', 'Matthews_corr'): 0.45,
            ('validation', 'Precision'): 0.97,
            ('validation', 'Recall'): 0.81
        }

        self.assertDictAlmostEqual(metrics, true_metrics, delta=0.1)

    def test_elastic_net_metrics(self):
        """
        Test for classic regression metrics (elastic net)
        """

        valuename = 'logS'
        dataframe = sdf_to_csv(
            self.sdf_file_path, self.fingerprints, value_name_list=valuename)
        classic_regressor = ALGORITHM[TRAINER_CLASS][ELASTIC_NETWORK](
            self.sdf_file_path, valuename, dataframe, scale='minmax', seed=0,
            test_set_size=self.test_set_size, fptype=self.fingerprints,
            output_path=self.temporary_folder, n_split=self.n_split,
            subsample_size=1.0
        )
        classic_regressor.train_model(CODES[ELASTIC_NETWORK])

        metrics = classic_regressor.metrics[ELASTIC_NETWORK]['mean']
        true_metrics = {
            ('train', 'RMSE'): 0.51,
            ('train', 'MAE'): 0.39,
            ('train', 'R2'): 0.93,
            ('test', 'MAE'): 0.55,
            ('test', 'R2'): 0.87,
            ('test', 'RMSE'): 0.74,
            ('validation', 'R2'): 0.86,
            ('validation', 'RMSE'): 0.75,
            ('validation', 'MAE'): 0.58
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
