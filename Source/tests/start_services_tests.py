import os
import unittest
import shutil

# set environment variables for tests
TEMP_FOLDER = 'temp_folder'
os.environ['OSDR_TEMP_FILES_FOLDER'] = TEMP_FOLDER
os.environ['OSDR_LOG_FOLDER'] = '{}/logs'.format(TEMP_FOLDER)

# DO NOT REMOVE THIS IMPORTS! ITS TESTS LIST!!
from osdr_ml_modeler.tests.sdf_processor_test import TestSdfProcessor
from osdr_ml_modeler.tests.validators_test import TestValidators
from osdr_ml_modeler.tests.applicability_domain_test import TestApplicabilityDomain
from osdr_ml_modeler.tests.manual_test_set_test import TestManualTestSet
from osdr_ml_modeler.tests.classic_metrics_test import TestClassicMetrics
from osdr_ml_modeler.tests.test_set_size_test import TestTestSetSize
from osdr_ml_predictor.tests.predictor_test import TestPrediction
from osdr_ml_modeler.tests.classic_hyperparameters_test import TestHyperparameters
from osdr_ml_modeler.tests.dnn_metrics_test import TestDNNMetrics
from osdr_ml_modeler.tests.report_test import TestReport


if __name__ == '__main__':
    unittest.main(exit=False)
    shutil.rmtree(TEMP_FOLDER, ignore_errors=True)
