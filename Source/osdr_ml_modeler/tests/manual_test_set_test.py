import os
import unittest

from learner.algorithms import (
    CODES, TRAINER_CLASS, ALGORITHM, SUPPORT_VECTOR_MACHINE_CLASSIFIER,
    SUPPORT_VECTOR_MACHINE_REGRESSOR
)
from processor import sdf_to_csv


class TestManualTestSet(unittest.TestCase):
    def setUp(self):
        self.fingerprints = [
            {'Type': 'EGR'},
            {'Type': 'MACCS'},
            {'Type': 'AVALON', 'Size': 512},
            {'Type': 'ECFC', 'Size': 512, 'Radius': 3}
        ]
        self.test_set_size = 0
        self.n_split = 2
        self.sdf_file_path = '{}/resources/DNN_data_solubility_notest.sdf'.format(
            os.path.dirname(os.path.abspath(__file__)))
        self.manual_test_file_path = '{}/resources/DNN_data_solubility_test.sdf'.format(
            os.path.dirname(os.path.abspath(__file__)))
        self.temporary_folder = os.environ['OSDR_TEMP_FILES_FOLDER']

    def test_svm_classifier_manual_test_set(self):
        """
        Test for classic classification (SVM classifier) manual test set
        """

        classname = 'Soluble'
        dataframe = sdf_to_csv(
            self.sdf_file_path, self.fingerprints, class_name_list=classname)
        manual_test_dataframe = sdf_to_csv(
            self.manual_test_file_path, self.fingerprints,
            class_name_list=classname
        )
        classic_classifier = ALGORITHM[TRAINER_CLASS][
            SUPPORT_VECTOR_MACHINE_CLASSIFIER
        ](
            self.sdf_file_path, classname, dataframe, subsample_size=1.0,
            test_set_size=self.test_set_size, seed=0, fptype=self.fingerprints,
            scale='standard', output_path=self.temporary_folder,
            n_split=self.n_split, manual_test_set=manual_test_dataframe
        )
        classic_classifier.train_model(
            CODES[SUPPORT_VECTOR_MACHINE_CLASSIFIER])

        metrics = classic_classifier.metrics[
            SUPPORT_VECTOR_MACHINE_CLASSIFIER]['mean']
        true_metrics = {
            ('train', 'AUC'): 0.99,
            ('train', 'ACC'): 0.99,
            ('train', 'f1-score'): 0.99,
            ('train', 'Cohen_Kappa'): 0.95,
            ('train', 'Matthews_corr'): 0.96,
            ('train', 'Precision'): 0.99,
            ('train', 'Recall'): 0.99,
            ('test', 'AUC'): 0.95,
            ('test', 'ACC'): 0.93,
            ('test', 'f1-score'): 0.96,
            ('test', 'Cohen_Kappa'): 0.64,
            ('test', 'Matthews_corr'): 0.66,
            ('test', 'Precision'): 0.93,
            ('test', 'Recall'): 0.98,
            ('validation', 'AUC'): 0.94,
            ('validation', 'ACC'): 0.93,
            ('validation', 'f1-score'): 0.96,
            ('validation', 'Cohen_Kappa'): 0.59,
            ('validation', 'Matthews_corr'): 0.63,
            ('validation', 'Precision'): 0.93,
            ('validation', 'Recall'): 0.99
        }

        self.assertDictAlmostEqual(metrics, true_metrics, delta=0.1)

    def test_svm_regressor_manual_test_set(self):
        """
        Test for classic regression (SVM regressor) manual test set
        """

        valuename = 'logS'
        dataframe = sdf_to_csv(
            self.sdf_file_path, self.fingerprints, value_name_list=valuename)
        manual_test_dataframe = sdf_to_csv(
            self.manual_test_file_path, self.fingerprints,
            value_name_list=valuename
        )
        classic_regressor = ALGORITHM[TRAINER_CLASS][
            SUPPORT_VECTOR_MACHINE_REGRESSOR
        ](
            self.sdf_file_path, valuename, dataframe, seed=0,
            test_set_size=self.test_set_size, fptype=self.fingerprints,
            output_path=self.temporary_folder, n_split=self.n_split,
            manual_test_set=manual_test_dataframe, subsample_size=1.0
        )
        classic_regressor.train_model(CODES[SUPPORT_VECTOR_MACHINE_REGRESSOR])

        metrics = classic_regressor.metrics[
            SUPPORT_VECTOR_MACHINE_REGRESSOR]['mean']
        true_metrics = {
            ('train', 'RMSE'): 0.54,
            ('train', 'MAE'): 0.38,
            ('train', 'R2'): 0.93,
            ('test', 'MAE'): 0.64,
            ('test', 'R2'): 0.78,
            ('test', 'RMSE'): 0.88,
            ('validation', 'R2'): 0.75,
            ('validation', 'RMSE'): 1.0,
            ('validation', 'MAE'): 0.68
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
