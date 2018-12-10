import os
import unittest

from learner.algorithms import (
    TRAINER_CLASS, ALGORITHM, ELASTIC_NETWORK, ADA_BOOST_DECISION_TREE
)
from learner.models import (
    elastic_network_hyperparameters, ada_boost_classifier_hyperparameters,
    ada_boost_classifier_hyperparameters_skopt,
    elastic_network_hyperparameters_skopt
)
from processor import sdf_to_csv


class TestHyperparameters(unittest.TestCase):
    def setUp(self):
        self.fingerprints = [
            {'Type': 'CAN2CAN'},
            {'Type': 'ECFC_CHIRALITY', 'Size': 512, 'Radius': 3},
            {'Type': 'ATOM_PAIRS', 'Size': 512, 'Radius': 3},
            {'Type': 'FCFP', 'Size': 512, 'Radius': 3}
        ]
        self.test_set_size = 0.2
        self.n_split = 2
        self.sdf_file_path = '{}/resources/DNN_data_solubility.sdf'.format(
            os.path.dirname(os.path.abspath(__file__)))
        self.temporary_folder = os.environ['OSDR_TEMP_FILES_FOLDER']

    def test_elastic_network_hyperparameters(self):
        """
        Test for classic classification hyperparameters (elastic network)
        """

        valuename = 'logS'
        dataframe = sdf_to_csv(
            self.sdf_file_path, self.fingerprints, value_name_list=valuename)
        elastic_network_regressor = ALGORITHM[TRAINER_CLASS][ELASTIC_NETWORK](
            self.sdf_file_path, valuename, dataframe, subsample_size=1.0,
            test_set_size=self.test_set_size, seed=0, fptype=self.fingerprints,
            scale='minmax', output_path=self.temporary_folder,
            n_split=self.n_split, opt_method='parzen'
        )
        elastic_network_regressor.model_name = ELASTIC_NETWORK
        parameters = elastic_network_regressor.make_training_parameters_grid()
        self.assertEquals(elastic_network_hyperparameters, parameters)

    def test_ada_boost_hyperparameters(self):
        """
        Test for classic classification hyperparameters (Ada boost)
        """

        classname = 'Soluble'
        dataframe = sdf_to_csv(
            self.sdf_file_path, self.fingerprints, class_name_list=classname)
        ada_boost = ALGORITHM[TRAINER_CLASS][ADA_BOOST_DECISION_TREE](
            self.sdf_file_path, classname, dataframe, subsample_size=1.0,
            test_set_size=self.test_set_size, seed=0, fptype=self.fingerprints,
            scale='minmax', output_path=self.temporary_folder,
            n_split=self.n_split, opt_method='parzen'
        )
        ada_boost.model_name = ADA_BOOST_DECISION_TREE
        parameters = ada_boost.make_training_parameters_grid()
        self.assertEquals(ada_boost_classifier_hyperparameters, parameters)

    def test_elastic_network_skopt_hyperparameters(self):
        """
        Test for classic classification skopt hyperparameters (elastic network)
        """

        valuename = 'logS'
        dataframe = sdf_to_csv(
            self.sdf_file_path, self.fingerprints, value_name_list=valuename)
        elastic_network_regressor = ALGORITHM[TRAINER_CLASS][ELASTIC_NETWORK](
            self.sdf_file_path, valuename, dataframe, subsample_size=1.0,
            test_set_size=self.test_set_size, seed=0, fptype=self.fingerprints,
            scale='minmax', output_path=self.temporary_folder,
            n_split=self.n_split, opt_method='forest'
        )
        elastic_network_regressor.model_name = ELASTIC_NETWORK
        parameters = elastic_network_regressor.make_training_parameters_grid()
        self.assertEquals(elastic_network_hyperparameters_skopt, parameters)

    def test_ada_boost_skopt_hyperparameters(self):
        """
        Test for classic classification skopt hyperparameters (Ada boost)
        """

        classname = 'Soluble'
        dataframe = sdf_to_csv(
            self.sdf_file_path, self.fingerprints, class_name_list=classname)
        ada_boost = ALGORITHM[TRAINER_CLASS][ADA_BOOST_DECISION_TREE](
            self.sdf_file_path, classname, dataframe, subsample_size=1.0,
            test_set_size=self.test_set_size, seed=0, fptype=self.fingerprints,
            scale='minmax', output_path=self.temporary_folder,
            n_split=self.n_split, opt_method='gauss'
        )
        ada_boost.model_name = ADA_BOOST_DECISION_TREE
        parameters = ada_boost.make_training_parameters_grid()
        self.assertEquals(
            ada_boost_classifier_hyperparameters_skopt, parameters)

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
