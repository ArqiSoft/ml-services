"""
Constants which emulate .NET interfaces for Mass Transit
"""


def event_callback(message):
    return message


TRAIN_MODEL = dict()
TRAIN_MODEL['name'] = 'TRAIN_MODEL'
TRAIN_MODEL['need_upper'] = True
TRAIN_MODEL['publisher_queue_name'] = 'TrainModel'
TRAIN_MODEL['event_json_file'] = 'json_schemas/train_model.json'
TRAIN_MODEL['event_callback'] = event_callback

TRAINING_FAILED = dict()
TRAINING_FAILED['name'] = 'TRAINING_FAILED'
TRAINING_FAILED['need_upper'] = True
TRAINING_FAILED['publisher_queue_name'] = 'TrainingFailed'
TRAINING_FAILED['event_json_file'] = 'json_schemas/training_failed.json'
TRAINING_FAILED['event_callback'] = event_callback

MODEL_TRAINED = dict()
MODEL_TRAINED['name'] = 'MODEL_TRAINED'
MODEL_TRAINED['need_upper'] = True
MODEL_TRAINED['publisher_queue_name'] = 'ModelTrained'
MODEL_TRAINED['event_json_file'] = 'json_schemas/model_trained.json'
MODEL_TRAINED['event_callback'] = event_callback

PREDICT_PROPERTIES = dict()
PREDICT_PROPERTIES['name'] = 'ML_PREDICTOR_CONSUMER'
PREDICT_PROPERTIES['need_upper'] = True
PREDICT_PROPERTIES['publisher_queue_name'] = 'PredictProperties'
PREDICT_PROPERTIES[
    'event_json_file'] = 'json_schemas/predictor_command.json'
PREDICT_PROPERTIES['event_callback'] = event_callback

PROPERTIES_PREDICTED = dict()
PROPERTIES_PREDICTED['name'] = 'ML_PREDICTOR_PUBLISHER'
PROPERTIES_PREDICTED['need_upper'] = True
PROPERTIES_PREDICTED['publisher_queue_name'] = 'PropertiesPredicted'
PROPERTIES_PREDICTED['event_json_file'] = 'json_schemas/predictor_event.json'
PROPERTIES_PREDICTED['event_callback'] = event_callback

PREDICTION_FAILED = dict()
PREDICTION_FAILED['name'] = 'ML_PREDICTOR_FAIL'
PREDICTION_FAILED['need_upper'] = True
PREDICTION_FAILED['publisher_queue_name'] = 'PropertiesPredictionFailed'
PREDICTION_FAILED['event_json_file'] = 'json_schemas/predictor_fail.json'
PREDICTION_FAILED['event_callback'] = event_callback

MODELER_FAIL_TEST = dict()
MODELER_FAIL_TEST['name'] = 'ML_MODELER_FAIL_TEST'
MODELER_FAIL_TEST['need_upper'] = True
MODELER_FAIL_TEST['publisher_queue_name'] = 'ModelTrainingFailed'
MODELER_FAIL_TEST['event_json_file'] = 'json_schemas/training_failed.json'
MODELER_FAIL_TEST['event_callback'] = event_callback
MODELER_FAIL_TEST['command_json_file'] = 'json_schemas/train_model.json'
MODELER_FAIL_TEST['command_callback'] = event_callback

MODEL_TRAINED_TEST = dict()
MODEL_TRAINED_TEST['name'] = 'ML_MODEL_TRAINED_TEST'
MODEL_TRAINED_TEST['need_upper'] = True
MODEL_TRAINED_TEST['publisher_queue_name'] = 'ModelTrained'
MODEL_TRAINED_TEST['event_json_file'] = 'json_schemas/model_trained.json'
MODEL_TRAINED_TEST['event_callback'] = event_callback
MODEL_TRAINED_TEST['command_json_file'] = 'json_schemas/train_model.json'
MODEL_TRAINED_TEST['command_callback'] = event_callback

PREDICTOR_FAIL_TEST = dict()
PREDICTOR_FAIL_TEST['name'] = 'ML_PREDICTOR_FAIL_TEST'
PREDICTOR_FAIL_TEST['need_upper'] = True
PREDICTOR_FAIL_TEST['publisher_queue_name'] = 'PropertiesPredictionFailed'
PREDICTOR_FAIL_TEST['event_json_file'] = 'json_schemas/predictor_fail.json'
PREDICTOR_FAIL_TEST['event_callback'] = event_callback
PREDICTOR_FAIL_TEST[
    'command_json_file'] = 'json_schemas/predictor_command.json'
PREDICTOR_FAIL_TEST['command_callback'] = event_callback

PROPERTIES_PREDICTED_TEST = dict()
PROPERTIES_PREDICTED_TEST['name'] = 'ML_PROPERTIES_PREDICTED_TEST'
PROPERTIES_PREDICTED_TEST['need_upper'] = True
PROPERTIES_PREDICTED_TEST['publisher_queue_name'] = 'PropertiesPredicted'
PROPERTIES_PREDICTED_TEST[
    'event_json_file'] = 'json_schemas/predictor_event.json'
PROPERTIES_PREDICTED_TEST['event_callback'] = event_callback
PROPERTIES_PREDICTED_TEST[
    'command_json_file'] = 'json_schemas/predictor_command.json'
PROPERTIES_PREDICTED_TEST['command_callback'] = event_callback

MODEL_TRAINING_STARTED = dict()
MODEL_TRAINING_STARTED['name'] = 'MODEL_TRAINING_STARTED'
MODEL_TRAINING_STARTED['need_upper'] = True
MODEL_TRAINING_STARTED['publisher_queue_name'] = 'ModelTrainingStarted'
MODEL_TRAINING_STARTED[
    'event_json_file'] = 'json_schemas/model_training_started.json'
MODEL_TRAINING_STARTED['event_callback'] = event_callback

GENERATE_REPORT_TEST = dict()
GENERATE_REPORT_TEST['name'] = 'GENERATE_REPORT_TEST'
GENERATE_REPORT_TEST['need_upper'] = True
GENERATE_REPORT_TEST['publisher_queue_name'] = 'ReportGenerated'
GENERATE_REPORT_TEST['event_json_file'] = 'json_schemas/report_generated.json'
GENERATE_REPORT_TEST['event_callback'] = event_callback
GENERATE_REPORT_TEST['command_json_file'] = 'json_schemas/generate_report.json'
GENERATE_REPORT_TEST['command_callback'] = event_callback

GENERATE_REPORT = dict()
GENERATE_REPORT['name'] = 'GENERATE_REPORT'
GENERATE_REPORT['need_upper'] = True
GENERATE_REPORT['publisher_queue_name'] = 'GenerateReport'
GENERATE_REPORT['event_json_file'] = 'json_schemas/generate_report.json'
GENERATE_REPORT['event_callback'] = event_callback

REPORT_GENERATED = dict()
REPORT_GENERATED['name'] = 'REPORT_GENERATED'
REPORT_GENERATED['need_upper'] = True
REPORT_GENERATED['publisher_queue_name'] = 'ReportGenerated'
REPORT_GENERATED['event_json_file'] = 'json_schemas/report_generated.json'
REPORT_GENERATED['event_callback'] = event_callback

MODEL_THUMBNAIL_GENERATED = dict()
MODEL_THUMBNAIL_GENERATED['name'] = 'MODEL_THUMBNAIL_GENERATED'
MODEL_THUMBNAIL_GENERATED['need_upper'] = True
MODEL_THUMBNAIL_GENERATED['publisher_queue_name'] = 'ModelThumbnailGenerated'
MODEL_THUMBNAIL_GENERATED[
    'event_json_file'] = 'json_schemas/model_thumbnail_generated.json'
MODEL_THUMBNAIL_GENERATED['event_callback'] = event_callback

BLOB_LOADED_TEST = dict()
BLOB_LOADED_TEST['name'] = 'BLOB_LOADED_TEST'
BLOB_LOADED_TEST['need_upper'] = True
BLOB_LOADED_TEST['publisher_queue_name'] = 'BlobLoaded'
BLOB_LOADED_TEST['event_json_file'] = 'json_schemas/blob_loaded.json'
BLOB_LOADED_TEST['event_callback'] = event_callback

TRAINING_REPORT_GENERATION_FAILED = dict()
TRAINING_REPORT_GENERATION_FAILED['name'] = 'TRAINING_REPORT_GENERATION_FAILED'
TRAINING_REPORT_GENERATION_FAILED['need_upper'] = True
TRAINING_REPORT_GENERATION_FAILED[
    'publisher_queue_name'] = 'ReportGenerationFailed'
TRAINING_REPORT_GENERATION_FAILED[
    'event_json_file'] = 'json_schemas/training_report_generation_failed.json'
TRAINING_REPORT_GENERATION_FAILED['event_callback'] = event_callback

OPTIMIZE_TRAINING = dict()
OPTIMIZE_TRAINING['name'] = 'OPTIMIZE_TRAINING'
OPTIMIZE_TRAINING['need_upper'] = True
OPTIMIZE_TRAINING['publisher_queue_name'] = 'OptimizeTraining'
OPTIMIZE_TRAINING['event_json_file'] = 'json_schemas/optimize_training.json'
OPTIMIZE_TRAINING['event_callback'] = event_callback

TRAINING_OPTMIZATION_FAILED = dict()
TRAINING_OPTMIZATION_FAILED['name'] = 'TRAINING_OPTMIZATION_FAILED'
TRAINING_OPTMIZATION_FAILED['need_upper'] = True
TRAINING_OPTMIZATION_FAILED['publisher_queue_name'] = 'TrainingOptimizationFailed'
TRAINING_OPTMIZATION_FAILED[
    'event_json_file'] = 'json_schemas/training_optimization_failed.json'
TRAINING_REPORT_GENERATION_FAILED['event_callback'] = event_callback

TRAINING_OPTIMIZED = dict()
TRAINING_OPTIMIZED['name'] = 'TRAINING_OPTIMIZED'
TRAINING_OPTIMIZED['need_upper'] = True
TRAINING_OPTIMIZED['publisher_queue_name'] = 'TrainingOptimized'
TRAINING_OPTIMIZED['event_json_file'] = 'json_schemas/training_optimized.json'
TRAINING_OPTIMIZED['event_callback'] = event_callback

OPTIMIZE_TRAINING_TEST = dict()
OPTIMIZE_TRAINING_TEST['name'] = 'OPTIMIZE_TRAINING_TEST'
OPTIMIZE_TRAINING_TEST['need_upper'] = True
OPTIMIZE_TRAINING_TEST['publisher_queue_name'] = 'TrainingOptimized'
OPTIMIZE_TRAINING_TEST['event_json_file'] = 'json_schemas/training_optimized.json'
OPTIMIZE_TRAINING_TEST['event_callback'] = event_callback
OPTIMIZE_TRAINING_TEST['command_json_file'] = 'json_schemas/optimize_training.json'
OPTIMIZE_TRAINING_TEST['command_callback'] = event_callback

OPTIMIZE_TRAINING_FAIL_TEST = dict()
OPTIMIZE_TRAINING_FAIL_TEST['name'] = 'OPTIMIZE_TRAINING_TEST'
OPTIMIZE_TRAINING_FAIL_TEST['need_upper'] = True
OPTIMIZE_TRAINING_FAIL_TEST['publisher_queue_name'] = 'TrainingOptimizationFailed'
OPTIMIZE_TRAINING_FAIL_TEST['event_json_file'] = 'json_schemas/training_optimization_failed.json'
OPTIMIZE_TRAINING_FAIL_TEST['event_callback'] = event_callback
OPTIMIZE_TRAINING_FAIL_TEST['command_json_file'] = 'json_schemas/optimize_training.json'
OPTIMIZE_TRAINING_FAIL_TEST['command_callback'] = event_callback

PREDICT_SINGLE_STRUCTURE = dict()
PREDICT_SINGLE_STRUCTURE['name'] = 'SINGLE_STRUCTURE_PREDICT_PROPERTY'
PREDICT_SINGLE_STRUCTURE['need_upper'] = True
PREDICT_SINGLE_STRUCTURE['publisher_queue_name'] = 'PredictStructure'
PREDICT_SINGLE_STRUCTURE[
    'event_json_file'] = 'json_schemas/single_structure_predict_property.json'
PREDICT_SINGLE_STRUCTURE['event_callback'] = event_callback

SINGLE_STRUCTURE_PREDICTED = dict()
SINGLE_STRUCTURE_PREDICTED['name'] = 'SINGLE_STRUCTURE_PROPERTY_PREDICTED'
SINGLE_STRUCTURE_PREDICTED['need_upper'] = True
SINGLE_STRUCTURE_PREDICTED['publisher_queue_name'] = 'PredictedResultReady'
SINGLE_STRUCTURE_PREDICTED['event_json_file'] = 'json_schemas/single_structure_property_predicted.json'
SINGLE_STRUCTURE_PREDICTED['event_callback'] = event_callback

PREDICT_SINGLE_STRUCTURE_TEST = dict()
PREDICT_SINGLE_STRUCTURE_TEST['name'] = 'PREDICT_SINGLE_STRUCTURE_TEST'
PREDICT_SINGLE_STRUCTURE_TEST['need_upper'] = True
PREDICT_SINGLE_STRUCTURE_TEST['publisher_queue_name'] = 'PredictedResultReady'
PREDICT_SINGLE_STRUCTURE_TEST['event_json_file'] = 'json_schemas/single_structure_property_predicted.json'
PREDICT_SINGLE_STRUCTURE_TEST['event_callback'] = event_callback
PREDICT_SINGLE_STRUCTURE_TEST['command_json_file'] = 'json_schemas/single_structure_predict_property.json'
PREDICT_SINGLE_STRUCTURE_TEST['command_callback'] = event_callback

CALCULATE_FEATURE_VECTORS = dict()
CALCULATE_FEATURE_VECTORS['name'] = 'CALCULATE_FEATURE_VECTORS'
CALCULATE_FEATURE_VECTORS['need_upper'] = True
CALCULATE_FEATURE_VECTORS['publisher_queue_name'] = 'CalculateFeatureVectors'
CALCULATE_FEATURE_VECTORS['event_json_file'] = 'json_schemas/calculate_feature_vectors.json'
CALCULATE_FEATURE_VECTORS['event_callback'] = event_callback

FEATURE_VECTORS_CALCULATED = dict()
FEATURE_VECTORS_CALCULATED['name'] = 'FEATURE_VECTORS_CALCULATED'
FEATURE_VECTORS_CALCULATED['need_upper'] = True
FEATURE_VECTORS_CALCULATED['publisher_queue_name'] = 'FeatureVectorsCalculated'
FEATURE_VECTORS_CALCULATED['event_json_file'] = 'json_schemas/feature_vectors_calculated.json'
FEATURE_VECTORS_CALCULATED['event_callback'] = event_callback

FEATURE_VECTORS_CALCULATION_FAILED = dict()
FEATURE_VECTORS_CALCULATION_FAILED['name'] = 'FEATURE_VECTORS_CALCULATION_FAILED'
FEATURE_VECTORS_CALCULATION_FAILED['need_upper'] = True
FEATURE_VECTORS_CALCULATION_FAILED['publisher_queue_name'] = 'FeatureVectorsCalculationFailed'
FEATURE_VECTORS_CALCULATION_FAILED['event_json_file'] = 'json_schemas/feature_vectors_calculation_failed.json'
FEATURE_VECTORS_CALCULATION_FAILED['event_callback'] = event_callback

FEATURE_VECTORS_CALCULATOR_TEST = dict()
FEATURE_VECTORS_CALCULATOR_TEST['name'] = 'FEATURE_VECTORS_CALCULATOR_TEST'
FEATURE_VECTORS_CALCULATOR_TEST['need_upper'] = True
FEATURE_VECTORS_CALCULATOR_TEST['publisher_queue_name'] = 'FeatureVectorsCalculated'
FEATURE_VECTORS_CALCULATOR_TEST['event_json_file'] = 'json_schemas/feature_vectors_calculated.json'
FEATURE_VECTORS_CALCULATOR_TEST['event_callback'] = event_callback
FEATURE_VECTORS_CALCULATOR_TEST['command_json_file'] = 'json_schemas/calculate_feature_vectors.json'
FEATURE_VECTORS_CALCULATOR_TEST['command_callback'] = event_callback

FEATURE_VECTORS_CALCULATOR_FAIL_TEST = dict()
FEATURE_VECTORS_CALCULATOR_FAIL_TEST['name'] = 'FEATURE_VECTORS_CALCULATOR_TEST'
FEATURE_VECTORS_CALCULATOR_FAIL_TEST['need_upper'] = True
FEATURE_VECTORS_CALCULATOR_FAIL_TEST['publisher_queue_name'] = 'FeatureVectorsCalculationFailed'
FEATURE_VECTORS_CALCULATOR_FAIL_TEST['event_json_file'] = 'json_schemas/feature_vectors_calculation_failed.json'
FEATURE_VECTORS_CALCULATOR_FAIL_TEST['event_callback'] = event_callback
FEATURE_VECTORS_CALCULATOR_FAIL_TEST['command_json_file'] = 'json_schemas/calculate_feature_vectors.json'
FEATURE_VECTORS_CALCULATOR_FAIL_TEST['command_callback'] = event_callback
