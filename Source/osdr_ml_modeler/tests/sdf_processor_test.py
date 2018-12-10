import unittest

from general_helper import molecules_from_mol_strings
from processor import sdf_to_csv


class TestSdfProcessor(unittest.TestCase):
    def setUp(self):
        self.fingerprints = [
            {'Type': 'ENUM2CAN'},
            {'Type': 'PATTERN', 'Size': 512},
            {'Type': 'FCFC_CHIRALITY', 'Size': 512, 'Radius': 3},
            {'Type': 'ECFP', 'Size': 512, 'Radius': 3}
        ]
        self.molstring = '\n  Ketcher  5 71818102D 1   1.00000     0.00000     0\n\n 13 13  0     0  0            999 V2000\n   -2.3818   -0.6252    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0\n   -1.5159   -0.1254    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0\n   -1.5157    0.8745    0.0000 O   0  0  0  0  0  0  0  0  0  0  0  0\n   -0.6501   -0.6254    0.0000 O   0  0  0  0  0  0  0  0  0  0  0  0\n    0.2159   -0.1256    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0\n    0.2162    0.8747    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0\n    1.0824    1.3752    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0\n    1.9483    0.8753    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0\n    1.9479   -0.1251    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0\n    1.0818   -0.6256    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0\n    1.0820   -1.6254    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0\n    1.9479   -2.1252    0.0000 O   0  0  0  0  0  0  0  0  0  0  0  0\n    0.2161   -2.1254    0.0000 O   0  0  0  0  0  0  0  0  0  0  0  0\n  1  2  1  0  0  0  0\n  2  3  2  0  0  0  0\n  2  4  1  0  0  0  0\n  4  5  1  0  0  0  0\n  5  6  2  0  0  0  0\n  6  7  1  0  0  0  0\n  7  8  2  0  0  0  0\n  8  9  1  0  0  0  0\n  9 10  2  0  0  0  0\n  5 10  1  0  0  0  0\n 10 11  1  0  0  0  0\n 11 12  1  0  0  0  0\n 11 13  2  0  0  0  0\nM  END\n'

    def test_sdf_processor(self):
        molecules = molecules_from_mol_strings([self.molstring])
        dataframe = sdf_to_csv(
            '', self.fingerprints, find_classes=True, find_values=True,
            molecules=molecules
        )

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
