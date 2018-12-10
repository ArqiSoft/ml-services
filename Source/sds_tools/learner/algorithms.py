"""
Module which contain definition for all algorithms used for training model
"""

from learner.models import (
    bayes_model, logistic_regression_model, ada_boost_model, bayes_optimizer,
    logistic_regression_optimizer, random_forest_classifier_model,
    random_forest_regressor_model, xgboost_classifier, train_dnn_multi_valid,
    kernel_ridge_regressor_optimizer, dnn_multi_regressor_optimizer,
    support_vector_machine_model_regressor, kneighbors_model_classifier,
    elastic_network_opimizer, support_vector_machine_model_classifier,
    elastic_net_cv_model, xgboost_regressor, kneighbors_model_regressor,
    ada_boost_optimizer, random_forest_optimizer, compile_dnn_reg_cv_new,
    support_vector_machine_optimizer, kneighbors_optimizer, xgboost_optimizer,
    epmty_arguments, dnn_classifier_arguments, dnn_multi_classifier_arguments,
    dnn_multi_classifier_optimizer, dnn_multi_regressor_arguments,
    train_dnn_multi_reg, kneighbors_classifier_hyperparameters,
    random_forest_classifier_hyperparameters, kernel_ridge_regressor_model,
    random_forest_regressor_hyperparameters, dnn_regressor_arguments,
    kneighbors_regressor_hyperparameters, train_dnn_valid_new,
    support_vector_machine_classifier_hyperparameters, dnn_regressor_optimizer,
    support_vector_machine_regressor_hyperparameters, dnn_classifier_optimizer,
    xgboost_classifier_hyperparameters, xgboost_regressor_hyperparameters,
    kernel_ridge_regressor_hyperparameters, ada_boost_classifier_hyperparameters,
    elastic_network_hyperparameters, elastic_network_hyperparameters_skopt,
    random_forest_classifier_hyperparameters_skopt, random_forest_regressor_hyperparameters_skopt,
    kneighbors_classifier_hyperparameters_skopt, kneighbors_regressor_hyperparameters_skopt,
    support_vector_machine_classifier_hyperparameters_skopt, xgboost_classifier_hyperparameters_skopt,
    support_vector_machine_regressor_hyperparameters_skopt, xgboost_regressor_hyperparameters_skopt,
    dnn_regressor_hyperparameters_hyperopt, dnn_regressor_hyperparameters_skopt,
    dnn_classifier_hyperparameters_hyperopt, dnn_classifier_hyperparameters_skopt,
    ada_boost_classifier_hyperparameters_skopt,
)


# all using algorithms names
NAIVE_BAYES = 'Naive Bayes'
LOGISTIC_REGRESSION = 'Logistic Regression'
ADA_BOOST_DECISION_TREE = 'Ada Boost Decision Tree'
RANDOM_FOREST_CLASSIFIER = 'Random Forest Classifier'
RANDOM_FOREST_REGRESSOR = 'Random Forest Regressor'
SUPPORT_VECTOR_MACHINE_CLASSIFIER = 'Support Vector Machine Classifier'
SUPPORT_VECTOR_MACHINE_REGRESSOR = 'Support Vector Machine Regressor'
KERNEL_RIDGE_REGRESSOR = 'Kernel Ridge Regressor'
NEIGHBORS_CLASSIFIER = 'K Neighbors Classifier'
NEIGHBORS_REGRESSOR = 'K Neighbors Regressor'
XGBOOST_CLASSIFIER = 'XGBoost Classifier'
XGBOOST_REGRESSOR = 'XGBoost Regressor'
ELASTIC_NETWORK = 'Elastic Network Regressor'
DNN_REGRESSOR = 'DNN Regressor'
DNN_CLASSIFIER = 'DNN Classifier'
DNN_MULTI_CLASSIFIER = 'DNN Multiclass Classifier'
DNN_MULTI_REGRESSOR = 'DNN Multitask Regressoor'
# algorithm types
CLASSIFIER = 'classifier'
MULTI_CLASSIFIER = 'multi_classifier'
MULTI_REGRESSOR = 'multi_regressor'
REGRESSOR = 'regressor'
# algorithm methods
CLASSIC = 'classic'
DNN = 'DNN'
# constants defining algorithms key
ALGORITHM_CODE = 'code of algorithm'
TRAIN_FUNCTION = 'model training function'
TYPE = 'algorithm type'
METHOD = 'algorithm_method'
HELP = 'help message for algorithm'
ADDITIONAL_ARGUMENTS = 'additional arguments generator function'
HYPERPARAMETERES_HYPEROPT = 'hyperparameters of algorithm (hyperopt)'
HYPERPARAMETERES_SKOPT = 'hyperparameters of algorithm (scikit-optimize)'
OPTIMIZER_FUNCTION = 'parameters optimizer function for model'
TRAINER_CLASS = 'models trainers classes'
# code of all training methods
# equals Method for incoming message TrainModel
CODES = {
    NAIVE_BAYES: 'naivebayes',
    LOGISTIC_REGRESSION: 'linearregression',
    ADA_BOOST_DECISION_TREE: 'decisiontree',
    RANDOM_FOREST_CLASSIFIER: 'randomforestclassifier',
    RANDOM_FOREST_REGRESSOR: 'randomforestregressor',
    SUPPORT_VECTOR_MACHINE_CLASSIFIER: 'supportvectormachineclassifier',
    SUPPORT_VECTOR_MACHINE_REGRESSOR: 'supportvectormachineregressor',
    KERNEL_RIDGE_REGRESSOR: 'kernelridgeregressor',
    NEIGHBORS_CLASSIFIER: 'nearestneighborsclassifier',
    XGBOOST_CLASSIFIER: 'extremegradientboostingclassifier',
    NEIGHBORS_REGRESSOR: 'nearestneighborsregressor',
    XGBOOST_REGRESSOR: 'extremegradientboostingregressor',
    ELASTIC_NETWORK: 'elasticnet',
    DNN_REGRESSOR: 'dnnregressor',
    DNN_CLASSIFIER: 'dnnclassifier',
    DNN_MULTI_CLASSIFIER: 'dnnmulticlassifier',
    DNN_MULTI_REGRESSOR: 'dnnmultiregressor',
}

# all algorithms useful properties
ALGORITHM = {
    # codes of algorithms, same as in input OSDR message
    ALGORITHM_CODE: {
        CODES[NAIVE_BAYES]: NAIVE_BAYES,
        CODES[LOGISTIC_REGRESSION]: LOGISTIC_REGRESSION,
        CODES[ADA_BOOST_DECISION_TREE]: ADA_BOOST_DECISION_TREE,
        CODES[RANDOM_FOREST_CLASSIFIER]: RANDOM_FOREST_CLASSIFIER,
        CODES[RANDOM_FOREST_REGRESSOR]: RANDOM_FOREST_REGRESSOR,
        CODES[SUPPORT_VECTOR_MACHINE_CLASSIFIER]: SUPPORT_VECTOR_MACHINE_CLASSIFIER,
        CODES[SUPPORT_VECTOR_MACHINE_REGRESSOR]: SUPPORT_VECTOR_MACHINE_REGRESSOR,
        CODES[KERNEL_RIDGE_REGRESSOR]: KERNEL_RIDGE_REGRESSOR,
        CODES[NEIGHBORS_CLASSIFIER]: NEIGHBORS_CLASSIFIER,
        CODES[XGBOOST_CLASSIFIER]: XGBOOST_CLASSIFIER,
        CODES[NEIGHBORS_REGRESSOR]: NEIGHBORS_REGRESSOR,
        CODES[XGBOOST_REGRESSOR]: XGBOOST_REGRESSOR,
        CODES[ELASTIC_NETWORK]: ELASTIC_NETWORK,
        CODES[DNN_REGRESSOR]: DNN_REGRESSOR,
        CODES[DNN_CLASSIFIER]: DNN_CLASSIFIER,
        CODES[DNN_MULTI_CLASSIFIER]: DNN_MULTI_CLASSIFIER,
        CODES[DNN_MULTI_REGRESSOR]: DNN_MULTI_REGRESSOR,
    },
    # functions used for train model with algorithm
    TRAIN_FUNCTION: {
        NAIVE_BAYES: bayes_model,
        LOGISTIC_REGRESSION: logistic_regression_model,
        ADA_BOOST_DECISION_TREE: ada_boost_model,
        RANDOM_FOREST_CLASSIFIER: random_forest_classifier_model,
        RANDOM_FOREST_REGRESSOR: random_forest_regressor_model,
        SUPPORT_VECTOR_MACHINE_CLASSIFIER: support_vector_machine_model_classifier,
        SUPPORT_VECTOR_MACHINE_REGRESSOR: support_vector_machine_model_regressor,
        KERNEL_RIDGE_REGRESSOR: kernel_ridge_regressor_model,
        NEIGHBORS_CLASSIFIER: kneighbors_model_classifier,
        NEIGHBORS_REGRESSOR: kneighbors_model_regressor,
        XGBOOST_CLASSIFIER: xgboost_classifier,
        XGBOOST_REGRESSOR: xgboost_regressor,
        ELASTIC_NETWORK: elastic_net_cv_model,
        DNN_REGRESSOR: compile_dnn_reg_cv_new,
        DNN_CLASSIFIER: train_dnn_valid_new,
        DNN_MULTI_CLASSIFIER: train_dnn_multi_valid,
        DNN_MULTI_REGRESSOR: train_dnn_multi_reg,
    },
    # optimization functions for each model
    # optimized parameters using before "fit" model
    OPTIMIZER_FUNCTION: {
        NAIVE_BAYES: bayes_optimizer,
        LOGISTIC_REGRESSION: logistic_regression_optimizer,
        ADA_BOOST_DECISION_TREE: ada_boost_optimizer,
        RANDOM_FOREST_CLASSIFIER: random_forest_optimizer,
        RANDOM_FOREST_REGRESSOR: random_forest_optimizer,
        SUPPORT_VECTOR_MACHINE_CLASSIFIER: support_vector_machine_optimizer,
        SUPPORT_VECTOR_MACHINE_REGRESSOR: support_vector_machine_optimizer,
        KERNEL_RIDGE_REGRESSOR: kernel_ridge_regressor_optimizer,
        NEIGHBORS_CLASSIFIER: kneighbors_optimizer,
        NEIGHBORS_REGRESSOR: kneighbors_optimizer,
        XGBOOST_CLASSIFIER: xgboost_optimizer,
        XGBOOST_REGRESSOR: xgboost_optimizer,
        ELASTIC_NETWORK: elastic_network_opimizer,
        DNN_REGRESSOR: dnn_regressor_optimizer,
        DNN_CLASSIFIER: dnn_classifier_optimizer,
        DNN_MULTI_CLASSIFIER: dnn_multi_classifier_optimizer,
        DNN_MULTI_REGRESSOR: dnn_multi_regressor_optimizer,
    },
    # type of algorithm model. regressor or classifier model is able
    TYPE: {
        CODES[NAIVE_BAYES]: CLASSIFIER,
        CODES[LOGISTIC_REGRESSION]: CLASSIFIER,
        CODES[ADA_BOOST_DECISION_TREE]: CLASSIFIER,
        CODES[RANDOM_FOREST_CLASSIFIER]: CLASSIFIER,
        CODES[RANDOM_FOREST_REGRESSOR]: REGRESSOR,
        CODES[SUPPORT_VECTOR_MACHINE_CLASSIFIER]: CLASSIFIER,
        CODES[SUPPORT_VECTOR_MACHINE_REGRESSOR]: REGRESSOR,
        CODES[KERNEL_RIDGE_REGRESSOR]: REGRESSOR,
        CODES[NEIGHBORS_CLASSIFIER]: CLASSIFIER,
        CODES[XGBOOST_CLASSIFIER]: CLASSIFIER,
        CODES[NEIGHBORS_REGRESSOR]: REGRESSOR,
        CODES[XGBOOST_REGRESSOR]: REGRESSOR,
        CODES[ELASTIC_NETWORK]: REGRESSOR,
        CODES[DNN_REGRESSOR]: REGRESSOR,
        CODES[DNN_CLASSIFIER]: CLASSIFIER,
        CODES[DNN_MULTI_CLASSIFIER]: MULTI_CLASSIFIER,
        CODES[DNN_MULTI_REGRESSOR]: MULTI_REGRESSOR,
    },
    # type of model training method. DNN or classic model is able
    METHOD: {
        CODES[NAIVE_BAYES]: CLASSIC,
        CODES[LOGISTIC_REGRESSION]: CLASSIC,
        CODES[ADA_BOOST_DECISION_TREE]: CLASSIC,
        CODES[RANDOM_FOREST_CLASSIFIER]: CLASSIC,
        CODES[RANDOM_FOREST_REGRESSOR]: CLASSIC,
        CODES[SUPPORT_VECTOR_MACHINE_CLASSIFIER]: CLASSIC,
        CODES[SUPPORT_VECTOR_MACHINE_REGRESSOR]: CLASSIC,
        CODES[KERNEL_RIDGE_REGRESSOR]: CLASSIC,
        CODES[NEIGHBORS_CLASSIFIER]: CLASSIC,
        CODES[XGBOOST_CLASSIFIER]: CLASSIC,
        CODES[NEIGHBORS_REGRESSOR]: CLASSIC,
        CODES[XGBOOST_REGRESSOR]: CLASSIC,
        CODES[ELASTIC_NETWORK]: CLASSIC,
        CODES[DNN_REGRESSOR]: DNN,
        CODES[DNN_CLASSIFIER]: DNN,
        CODES[DNN_MULTI_CLASSIFIER]: DNN,
        CODES[DNN_MULTI_REGRESSOR]: DNN,
    },
    # help text, which explain how that algorithm work
    HELP: {
        NAIVE_BAYES: 'Naive Bayes help text',
        LOGISTIC_REGRESSION: 'Logistic Regression help text',
        ADA_BOOST_DECISION_TREE: 'Ada Boost DT help text',
        RANDOM_FOREST_CLASSIFIER: 'Random Forest classifier help text',
        RANDOM_FOREST_REGRESSOR: 'Random Forest regressor help text',
        SUPPORT_VECTOR_MACHINE_CLASSIFIER: 'SVM classifier help text',
        SUPPORT_VECTOR_MACHINE_REGRESSOR: 'SVM regressor help text',
        KERNEL_RIDGE_REGRESSOR: 'KRR help text',
        NEIGHBORS_CLASSIFIER: 'kNN classifier help text',
        NEIGHBORS_REGRESSOR: 'kNN regressor help text',
        XGBOOST_CLASSIFIER: 'XGBoost classifier help text',
        XGBOOST_REGRESSOR: 'XGBoost regressor help text',
        ELASTIC_NETWORK: 'Elastic network help text',
        DNN_REGRESSOR: 'DNN regressor help text',
        DNN_CLASSIFIER: 'DNN classifier help text',
        DNN_MULTI_CLASSIFIER: 'DNN multiclass classifier help text',
        DNN_MULTI_REGRESSOR: 'DNN multitask regressor help text',
    },
    # additional arguments for each model
    # using whe "fit" model
    ADDITIONAL_ARGUMENTS: {
        NAIVE_BAYES: epmty_arguments,
        LOGISTIC_REGRESSION: epmty_arguments,
        ADA_BOOST_DECISION_TREE: epmty_arguments,
        RANDOM_FOREST_REGRESSOR: epmty_arguments,
        RANDOM_FOREST_CLASSIFIER: epmty_arguments,
        SUPPORT_VECTOR_MACHINE_CLASSIFIER: epmty_arguments,
        SUPPORT_VECTOR_MACHINE_REGRESSOR: epmty_arguments,
        KERNEL_RIDGE_REGRESSOR: epmty_arguments,
        NEIGHBORS_CLASSIFIER: epmty_arguments,
        NEIGHBORS_REGRESSOR: epmty_arguments,
        XGBOOST_CLASSIFIER: epmty_arguments,
        XGBOOST_REGRESSOR: epmty_arguments,
        ELASTIC_NETWORK: epmty_arguments,
        DNN_REGRESSOR: dnn_regressor_arguments,
        DNN_CLASSIFIER: dnn_classifier_arguments,
        DNN_MULTI_CLASSIFIER: dnn_multi_classifier_arguments,
        DNN_MULTI_REGRESSOR: dnn_multi_regressor_arguments,
    },
    HYPERPARAMETERES_HYPEROPT: {
        ADA_BOOST_DECISION_TREE: ada_boost_classifier_hyperparameters,
        RANDOM_FOREST_CLASSIFIER: random_forest_classifier_hyperparameters,
        RANDOM_FOREST_REGRESSOR: random_forest_regressor_hyperparameters,
        NEIGHBORS_CLASSIFIER: kneighbors_classifier_hyperparameters,
        NEIGHBORS_REGRESSOR: kneighbors_regressor_hyperparameters,
        SUPPORT_VECTOR_MACHINE_CLASSIFIER: support_vector_machine_classifier_hyperparameters,
        SUPPORT_VECTOR_MACHINE_REGRESSOR: support_vector_machine_regressor_hyperparameters,
        KERNEL_RIDGE_REGRESSOR: kernel_ridge_regressor_hyperparameters,
        XGBOOST_CLASSIFIER: xgboost_classifier_hyperparameters,
        XGBOOST_REGRESSOR: xgboost_regressor_hyperparameters,
        ELASTIC_NETWORK: elastic_network_hyperparameters,
        DNN_REGRESSOR: dnn_regressor_hyperparameters_hyperopt,
        DNN_CLASSIFIER: dnn_classifier_hyperparameters_hyperopt,
    },
    HYPERPARAMETERES_SKOPT: {
        ADA_BOOST_DECISION_TREE: ada_boost_classifier_hyperparameters_skopt,
        RANDOM_FOREST_CLASSIFIER: random_forest_classifier_hyperparameters_skopt,
        RANDOM_FOREST_REGRESSOR: random_forest_regressor_hyperparameters_skopt,
        NEIGHBORS_CLASSIFIER: kneighbors_classifier_hyperparameters_skopt,
        NEIGHBORS_REGRESSOR: kneighbors_regressor_hyperparameters_skopt,
        SUPPORT_VECTOR_MACHINE_CLASSIFIER: support_vector_machine_classifier_hyperparameters_skopt,
        SUPPORT_VECTOR_MACHINE_REGRESSOR: support_vector_machine_regressor_hyperparameters_skopt,
        KERNEL_RIDGE_REGRESSOR: kernel_ridge_regressor_hyperparameters,
        XGBOOST_CLASSIFIER: xgboost_classifier_hyperparameters_skopt,
        XGBOOST_REGRESSOR: xgboost_regressor_hyperparameters_skopt,
        ELASTIC_NETWORK: elastic_network_hyperparameters_skopt,
        DNN_REGRESSOR: dnn_regressor_hyperparameters_skopt,
        DNN_CLASSIFIER: dnn_classifier_hyperparameters_skopt,
    }
}


def algorithm_name_by_code(algorithm_code):
    """
    Method which return algorithm name by algorithm code.
    algorithm_code MUST contain any 'intable' type

    :param algorithm_code: code of algorithm
    :return: algorithm name or 'Unknown algorithm'
        if algorithm code not exist in algorithm dict
    """

    algorithm_name = 'Unknown algorithm'
    if algorithm_code in ALGORITHM[ALGORITHM_CODE]:
        algorithm_name = ALGORITHM[ALGORITHM_CODE][algorithm_code]

    return algorithm_name


def algorithm_code_by_name(algorithm_name):
    """
    Method which return algorithm code by algorithm name.
    algorithm_name MUST contain any 'intable' name

    :param algorithm_name: name of algorithm
    :return: algorithm type code by algorithm name or None
    """

    algorithm_code = None
    for number, name in ALGORITHM[ALGORITHM_CODE].items():
        if name == algorithm_name:
            algorithm_code = number
            break

    return algorithm_code


def model_type_by_code(algorithm_code):
    """
    Method which return algorithm type by algorithm code.
    algorithm_code MUST contain any 'intable' type

    :param algorithm_code: code of algorithm
    :return: algorithm type name by algorithm code or None
    """

    # invalid algorithm code case
    if algorithm_code not in ALGORITHM[ALGORITHM_CODE].keys():
        return None

    return ALGORITHM[TYPE][algorithm_code]


def model_type_by_name(model_name):
    """
    Method which return algorithm type by model name.
    algorithm_name MUST contain any 'intable' type

    :param model_name: model name
    :return: algorithm type name by model name or None
    """

    algorithm_code = algorithm_code_by_name(model_name)

    return model_type_by_code(algorithm_code)


def algorithm_method_by_code(algorithm_code):
    """
    Method which return algorithm training type by algorithm code.
    algorithm_code MUST contain any 'intable' type

    :param algorithm_code: code of algorithm
    :return: algorithm training type name by algorithm code or None
    """

    if algorithm_code not in ALGORITHM[ALGORITHM_CODE].keys():
        return None

    return ALGORITHM[METHOD][algorithm_code]


def algorithm_method_by_name(model_name):
    algorithm_code = algorithm_code_by_name(model_name)

    return algorithm_method_by_code(algorithm_code)


def algorithm_help_by_name(algorithm_name):
    """
    Method which return algorithm help text by algorithm name.
    algorithm_name MUST contain any 'intable' type

    :param algorithm_name: algorithm name for which want to get help
    :return: help string for given algorithm
    """

    return ALGORITHM[HELP][algorithm_name]


def set_trainers_classes_to_algorithms():
    """
    Add classes to all algorithms used in ML training
    If you add new algorithm, add link to class there
    """
    # avoid cross import
    from learner.model_trainers import (
        ClassicClassifier, ClassicRegressor, DNNRegressor, DNNClassifier,
        DNNMultiClassifier, DNNMultiRegressor
    )

    ALGORITHM[TRAINER_CLASS] = {
        NAIVE_BAYES: ClassicClassifier,
        LOGISTIC_REGRESSION: ClassicClassifier,
        ADA_BOOST_DECISION_TREE: ClassicClassifier,
        RANDOM_FOREST_CLASSIFIER: ClassicClassifier,
        RANDOM_FOREST_REGRESSOR: ClassicRegressor,
        SUPPORT_VECTOR_MACHINE_CLASSIFIER: ClassicClassifier,
        SUPPORT_VECTOR_MACHINE_REGRESSOR: ClassicRegressor,
        KERNEL_RIDGE_REGRESSOR: ClassicRegressor,
        NEIGHBORS_CLASSIFIER: ClassicClassifier,
        XGBOOST_CLASSIFIER: ClassicClassifier,
        NEIGHBORS_REGRESSOR: ClassicRegressor,
        XGBOOST_REGRESSOR: ClassicRegressor,
        ELASTIC_NETWORK: ClassicRegressor,
        DNN_REGRESSOR: DNNRegressor,
        DNN_CLASSIFIER: DNNClassifier,
        DNN_MULTI_CLASSIFIER: DNNMultiClassifier,
        DNN_MULTI_REGRESSOR: DNNMultiRegressor,
    }

# add algoritmhs classes on first call only
if TRAINER_CLASS not in ALGORITHM.keys():
    set_trainers_classes_to_algorithms()
