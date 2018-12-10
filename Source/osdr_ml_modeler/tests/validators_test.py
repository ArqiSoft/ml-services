import unittest

from general_helper import (
    validate_kfold, validate_test_datatset_size, validate_subsample_size
)
from learner.fingerprints import validate_fingerprints


class TestValidators(unittest.TestCase):

    def test_k_fold_validator(self):
        k_fold = 5
        validate_kfold(k_fold)

    def test_subsample_size_validator(self):
        subsample_size = 0.2
        validate_subsample_size(subsample_size)

    def test_dataset_size_validator(self):
        test_dataset_size = 0.3
        validate_test_datatset_size(test_dataset_size)

    def test_fingerprints_validator(self):
        fingerprints = [
            {'Type': 'ECFP', 'Size': 512, 'Radius': 3},
            {'Type': 'ENUM2CAN'},
            {'Type': 'PATTERN', 'Size': 512},
        ]
        validate_fingerprints(fingerprints)

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
