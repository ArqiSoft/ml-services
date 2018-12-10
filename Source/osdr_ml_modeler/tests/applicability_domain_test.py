import os
import unittest

from learner.algorithms import (
    ELASTIC_NETWORK, TRAINER_CLASS, ALGORITHM, NAIVE_BAYES
)
from processor import sdf_to_csv


class TestApplicabilityDomain(unittest.TestCase):
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

    def test_classifier_applicability_domain(self):
        valuename = 'Soluble'
        dataframe = sdf_to_csv(
            self.sdf_file_path, self.fingerprints, class_name_list=valuename)
        classic_classifier = ALGORITHM[TRAINER_CLASS][NAIVE_BAYES](
            self.sdf_file_path, valuename, dataframe, subsample_size=1.0,
            test_set_size=self.test_set_size, seed=0, fptype=self.fingerprints,
            scale='minmax', output_path=self.temporary_folder,
            n_split=self.n_split
        )
        classic_classifier.make_applicability_domain()

        self.assertAlmostEqual(
            classic_classifier.distance_mean, 5.5480266, delta=0.01)
        self.assertAlmostEqual(
            classic_classifier.distance_std, 2.28519, delta=0.01)
        self.assertAlmostEqual(
            classic_classifier.density_mean, 1916.112310059458, delta=0.01)
        self.assertAlmostEqual(
            classic_classifier.density_std, 0.08123426215633249, delta=0.01)
        self.assertAlmostEqual(classic_classifier.modi, 0.79, delta=0.01)
        self.assertEqual(classic_classifier.train_shape, 1295)

    def test_regressor_applicability_domain(self):
        valuename = 'logS'
        dataframe = sdf_to_csv(
            self.sdf_file_path, self.fingerprints, value_name_list=valuename)
        classic_regressor = ALGORITHM[TRAINER_CLASS][ELASTIC_NETWORK](
            self.sdf_file_path, valuename, dataframe, scale='minmax', seed=0,
            test_set_size=self.test_set_size, fptype=self.fingerprints,
            output_path=self.temporary_folder, n_split=self.n_split,
            subsample_size=1.0
        )
        classic_regressor.make_applicability_domain()

        self.assertAlmostEqual(
            classic_regressor.distance_mean, 5.5480266, delta=0.01)
        self.assertAlmostEqual(
            classic_regressor.distance_std, 2.28519, delta=0.01)
        self.assertAlmostEqual(
            classic_regressor.density_mean, 1916.112310059458, delta=0.01)
        self.assertAlmostEqual(
            classic_regressor.density_std, 0.08123426215633249, delta=0.01)
        self.assertAlmostEqual(classic_regressor.modi, 0.75, delta=0.01)
        self.assertEqual(classic_regressor.train_shape, 1295)

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
