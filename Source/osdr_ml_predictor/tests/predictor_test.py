import os
import unittest

from general_helper import (
    molecules_from_smiles, cache_model_files, cache_additional_files
)
from learner.algorithms import (
    REGRESSOR, CLASSIFIER, NAIVE_BAYES, ELASTIC_NETWORK, XGBOOST_CLASSIFIER,
    XGBOOST_REGRESSOR
)
from predictor.Predictor import MLPredictor


class TestPrediction(unittest.TestCase):
    def setUp(self):
        molecule_smiles = 'C[C@H](CCCC(C)C)[C@H]1CC[C@@H]2[C@@]1(CC[C@H]3[C@H]2CC=C4[C@@]3(CC[C@@H](C4)O)C)C'
        self.molecules = molecules_from_smiles([molecule_smiles])
        self.cached_files = {
            'models': list(),
            'models_names': list(),
            'graphs': list(),
            'sessions': list(),
            'models_number': 0,
            'distance_matrix': None,
            'scaler': None,
            'train_mean': None,
            'density_model': None
        }

    def test_naive_bayes_single_molecule_prediction(self):
        models_folder = '{}/resources/naive_bayes_model'.format(
            os.path.dirname(os.path.abspath(__file__)))
        cache_model_files(self.cached_files, models_folder)
        cache_additional_files(self.cached_files, models_folder)

        parameters = {
            'DatasetFileName': 'datasetname',
            'ClassName': 'Soluble',
            'Fingerprints': [
                {
                    'Type': 'DESC'
                },
                {
                    'Type': 'MACCS'
                },
                {
                    'Type': 'AVALON',
                    'Size': 512
                },
                {
                    'Type': 'FCFC',
                    'Size': 512,
                    'Radius': 3
                }
            ],
            'ModelType': CLASSIFIER,
            'Molecules': self.molecules,
            'Models': self.cached_files,
            'DensityMean': -320.786410317521,
            'DensityStd': 0.06777013324099566,
            'DistanceMean': 1560.512541620144,
            'DistanceStd': 2626.242934683793,
            'TrainShape': 1035,
            'Modi': 0.8346316555598516,
            'ModelCode': NAIVE_BAYES
        }

        naive_bayes_predictor = MLPredictor(parameters)
        naive_bayes_predictor.make_prediction()

        self.assertAlmostEqual(
            float(naive_bayes_predictor.prediction[0][2][2]), 0.0, delta=0.1)

    def test_elastic_network_single_molecule_prediction(self):
        models_folder = '{}/resources/elastic_network_model'.format(
            os.path.dirname(os.path.abspath(__file__)))
        cache_model_files(self.cached_files, models_folder)
        cache_additional_files(self.cached_files, models_folder)

        parameters = {
            'DatasetFileName': 'datasetname',
            'ClassName': 'logS',
            'Fingerprints': [
                {
                    'Type': 'DESC'
                },
                {
                    'Type': 'MACCS'
                },
                {
                    'Type': 'AVALON',
                    'Size': 512
                },
                {
                    'Type': 'FCFC',
                    'Size': 512,
                    'Radius': 3
                }
            ],
            'ModelType': REGRESSOR,
            'Molecules': self.molecules,
            'Models': self.cached_files,
            'DensityMean': 1916.3329055822746,
            'DensityStd': 0.06787869754582834,
            'DistanceMean': 5.589669704437256,
            'DistanceStd': 2.2714009284973145,
            'TrainShape': 1035,
            'Modi': 0,
            'ModelCode': ELASTIC_NETWORK
        }

        elastic_network = MLPredictor(parameters)
        elastic_network.make_prediction()

        self.assertAlmostEqual(
            float(elastic_network.prediction[0][2][2]), -7.0, delta=0.1)

    def test_xgboost_classifier_single_molecule_prediction(self):
        models_folder = '{}/resources/xgboost_classifier_model'.format(
            os.path.dirname(os.path.abspath(__file__)))
        cache_model_files(self.cached_files, models_folder)
        cache_additional_files(self.cached_files, models_folder)

        parameters = {
            'DatasetFileName': 'datasetname',
            'ClassName': 'Soluble',
            'Fingerprints': [
                {
                    'Type': 'DESC'
                },
                {
                    'Type': 'MACCS'
                },
                {
                    'Type': 'AVALON',
                    'Size': 512
                },
                {
                    'Type': 'FCFC',
                    'Size': 512,
                    'Radius': 3
                }
            ],
            'ModelType': CLASSIFIER,
            'Molecules': self.molecules,
            'Models': self.cached_files,
            'DensityMean': -320.09921176182047,
            'DensityStd': 0.04294549666657096,
            'DistanceMean': 1593.5023161947988,
            'DistanceStd': 2865.44598716839,
            'TrainShape': 518,
            'Modi': 0.8181268251981644,
            'ModelCode': XGBOOST_CLASSIFIER
        }

        xgboost_classifier = MLPredictor(parameters)
        xgboost_classifier.make_prediction()

        self.assertAlmostEqual(
            float(xgboost_classifier.prediction[0][2][2]), 0.25, delta=0.1)

    def test_xgboost_regressor_single_molecule_prediction(self):
        models_folder = '{}/resources/xgboost_regressor_model'.format(
            os.path.dirname(os.path.abspath(__file__)))
        cache_model_files(self.cached_files, models_folder)
        cache_additional_files(self.cached_files, models_folder)

        parameters = {
            'DatasetFileName': 'datasetname',
            'ClassName': 'logS',
            'Fingerprints': [
                {
                    'Type': 'DESC'
                },
                {
                    'Type': 'MACCS'
                },
                {
                    'Type': 'AVALON',
                    'Size': 512
                },
                {
                    'Type': 'FCFC',
                    'Size': 512,
                    'Radius': 3
                }
            ],
            'ModelType': REGRESSOR,
            'Molecules': self.molecules,
            'Models': self.cached_files,
            'DensityMean': 1917.024270189101,
            'DensityStd': 0.060691652384931655,
            'DistanceMean': 6.273035970923933,
            'DistanceStd': 2.8182187428844445,
            'TrainShape': 207,
            'Modi': 0.75,
            'ModelCode': XGBOOST_REGRESSOR
        }

        xgboost_regressor = MLPredictor(parameters)
        xgboost_regressor.make_prediction()

        self.assertAlmostEqual(
            float(xgboost_regressor.prediction[0][2][2]), -6.67, delta=0.1)

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
