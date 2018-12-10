import matplotlib
import pandas as pd
import xgboost
from keras.constraints import maxnorm

from MLLogger import BaseMLLogger
from general_helper import coeff_determination

matplotlib.use('Agg')
from learner.plotters import BatchLogger

import numpy as np

# Load sklearn methods
from sklearn import (
    model_selection, utils, linear_model, svm, naive_bayes,
    ensemble, neighbors, kernel_ridge
)
from sklearn.metrics import (
    r2_score, roc_auc_score, make_scorer, matthews_corrcoef, mean_squared_error
)
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from scipy.stats import randint, uniform
from keras.callbacks import ReduceLROnPlateau, EarlyStopping
import tensorflow as tf
from keras.models import Sequential
from keras.layers import (
    Dense, Dropout, Activation, BatchNormalization, PReLU, ELU,
    ThresholdedReLU, Flatten
)
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.regularizers import l2
from keras.initializers import RandomNormal
from keras.optimizers import (
    SGD, Adam, Nadam, RMSprop, Adagrad, Adadelta, Adamax
)
from keras.layers.advanced_activations import LeakyReLU

from operator import itemgetter
from hyperopt import hp, fmin, tpe, STATUS_OK, Trials, STATUS_FAIL
from hyperopt.pyll import scope
from skopt import gp_minimize, gbrt_minimize, forest_minimize
from skopt.space import Categorical, Real, Integer


GPU_XGBOOST_VERSION = '0.72'
LOGGER = BaseMLLogger(log_name='logger')


def onehot_encoded(array):
    """

    :param array:
    :return:
    """
    # TODO docstring there
    label_encoder = LabelEncoder()
    integer_encoded = label_encoder.fit_transform(array.ravel())
    onehot_encoder = OneHotEncoder(sparse=False)
    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
    return np.array(onehot_encoder.fit_transform(integer_encoded), dtype=int)


def onehot_decoded(array):
    # TODO docstring there
    return np.argmax(array, axis=1)


def class_weight_to_dict(y_class):
    """
    :param y_class: 1D array, labels
    :return: balanced class weight dictionary
    """
    # TODO add description to docstring

    # compute class weight
    class_weight = utils.compute_class_weight('balanced', [0, 1], y_class)
    class_weight_dict = {}
    for index in range(len(class_weight)):
        class_weight_dict[index] = class_weight[index]

    return class_weight_dict


def epmty_arguments(trainer_object):
    """
    Method which generate empty arguments for all training models,
    which havent it's own additional arguments

    :param trainer_object: trained model
    :return: additional naive bayes model arguments
    :rtype: dict
    """

    return {}


def TMP_model_dnn(
    nh_layers=3, input_dim=1024, num_hidden=None, num_labels=1,
    drop_out=0.5, input_drop_out=0.2, beta=0.000, activation='selu',
    device='/gpu:0', activation_out='linear', k_constraint=3, batch_norm=True,
    kernel_initializer='lecun_normal'
):
    """
    Function that creates and returns a keras SEQUENTIAL neural network
    N layers DNN model with drop out layers for each hidden layer.
    All the default parameters are for
    a 3 layer DNN model

    :param nh_layers: number of hidden layers
    :param input_dim: number of features in the dataset
    :param num_hidden: hidden layers size as a list
    :param num_labels: number of labels
    :param drop_out: the same drop out applied for all hidden layers
    :param input_drop_out:
    :param beta: l2 regularization
    :param l_rate: initial learning rate
    :param momentum: used only for SGD optimizer
    :param metric_name:
    :param activation: hidden layers activation function
    :param device: gpu:0 (can accept cpu:0 or gpu:1)
    :param activation_out: out layer activation function
    :param model_summary: show the model summary, default is True)
    :param optimizer: optimizer name
    :param determination_function:
    :param k_constraint:
    :param batch_norm:
    :return: keras DNN model
    """

    num_hidden = num_hidden or [1024, 1024, 1024]

    if len(num_hidden) != nh_layers:
        raise Exception('The number of layers nh_layers are not matched to the length of num_hidden')
    if activation in ['relu', 'tanh', 'selu','LeakyReLU', 'PReLU', 'ELU', 'ThresholdedReLU']:
        act = activation
    else:
        raise Exception('I can\'t use this activation function: {} to compile a DNN'.format(activation))

    with tf.device(('/{}'.format(device))):
        # create model
        model = Sequential(name='model_{}_layers'.format(nh_layers))
        model.add(Dropout(
            input_drop_out, input_shape=(input_dim, ), name='Input_dropout'
        ))
        model.add(Dense(
            num_hidden[0],name='Dense_1',
            kernel_regularizer=l2(beta), kernel_initializer=kernel_initializer,
            kernel_constraint=maxnorm(k_constraint)
        ))
        model.add(get_activation_function(activation))
        if batch_norm:
            model.add(BatchNormalization())
        model.add(Dropout(drop_out, name='DropOut_1'))
        for idx in range(nh_layers - 1):
            model.add(Dense(
                num_hidden[idx + 1], kernel_regularizer=l2(beta),
                kernel_constraint=maxnorm(k_constraint),
                kernel_initializer=kernel_initializer,
                name='Dense_{}'.format(idx + 2)
            ))
            model.add(get_activation_function(activation))
            if batch_norm:
                model.add(BatchNormalization())
            model.add(Dropout(drop_out, name='DropOut_{}'.format(idx + 2)))

        model.add(Dense(
            num_labels, activation=activation_out,
            name='Output',kernel_initializer='lecun_normal'
        ))

    return model


def lw_autoencoder_model(
        input_dim=512, drop_out=0.5, device='/gpu:0', l_rate=0.01
):
    """

    :param input_dim: input vector length
    :param drop_out: dropout
    :param device: CPU or GPU
    :param l_rate: learning rate
    :return:
    """
    # TODO docstring there
    encoders_dims = [input_dim,500,500,2000,10]
    init_stddev = 0.01

    layer_wise_autoencoders = []
    encoders = []
    decoders = []

    with tf.device(('/{}'.format(device))):
        for i in range(1, len(encoders_dims)):
            encoder_activation = 'linear' if i == (len(encoders_dims) - 1) else 'relu'
            encoder = Dense(
                encoders_dims[i], activation=encoder_activation,
                input_shape=(encoders_dims[i - 1],),
                kernel_initializer=RandomNormal(
                    mean=0.0, stddev=init_stddev, seed=None
                ),
                bias_initializer='zeros', name='encoder_dense_%d' % i
            )
            encoders.append(encoder)

            decoder_index = len(encoders_dims) - i
            decoder_activation = 'linear' if i == 1 else 'relu'
            decoder = Dense(
                encoders_dims[i - 1], activation=decoder_activation,
                kernel_initializer=RandomNormal(
                    mean=0.0, stddev=init_stddev, seed=None
                ),
                bias_initializer='zeros',
                name='decoder_dense_%d' % decoder_index
            )
            decoders.append(decoder)
            autoencoder = Sequential([
                Dropout(drop_out, input_shape=(encoders_dims[i - 1],),
                        name='encoder_dropout_%d' % i),
                encoder,
                Dropout(drop_out, name='decoder_dropout_%d' % decoder_index),
                decoder
            ])
            autoencoder.compile(
                loss='mse', optimizer=SGD(lr=l_rate, decay=0, momentum=0.9))
            layer_wise_autoencoders.append(autoencoder)

        # build the end-to-end autoencoder for finetuning
        # Note that at this point dropout is discarded
        encoder = Sequential(encoders)
        encoder.compile(
            loss='mse', optimizer=SGD(lr=l_rate, decay=0, momentum=0.9))
        decoders.reverse()
        autoencoder = Sequential(encoders + decoders)
        autoencoder.compile(
            loss='mse', optimizer=SGD(lr=l_rate, decay=0, momentum=0.9))
    return {
        'model':autoencoder,
        'layer_wise_autoencoders':layer_wise_autoencoders,
        'encoder':encoder,
        'encoders':encoders
    }


def CNN_model_reg(
        graph_size=30, num_channels=2, drop_out=0.5, num_filters=[32],
        kernel_size=(5, 5), num_labels=1, device='/gpu:0', l_rate=0.01,
        beta=0.001, momentum=0.9, metric_name='mean_absolute_error',
        optimizer='Nadam', determination_function=coeff_determination,
        k_constraint=3
):
    """

    :param graph_size:
    :param num_channels:
    :param drop_out:
    :param num_filters:
    :param kernel_size:
    :param num_labels:
    :param device:
    :param l_rate:
    :param beta:
    :param momentum:
    :param metric_name:
    :param optimizer:
    :param determination_function:
    :param k_constraint:
    :return:
    """
    # TODO docstring there
    if num_filters is None:
        num_filters = [32, 64]
    with tf.device(('/{}'.format(device))):
        model = Sequential()
        model.add(Conv2D(
            filters=32, kernel_size=kernel_size, padding='same',
            input_shape=(graph_size, graph_size, num_channels),
            activation='relu', name='Input_Conv', kernel_regularizer=l2(beta),
            kernel_constraint=maxnorm(k_constraint)
        ))
        model.add(Conv2D(
            filters=32, kernel_size=kernel_size, padding='same',
            activation='relu', kernel_regularizer=l2(beta),
            kernel_constraint=maxnorm(k_constraint)
        ))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(drop_out))

        for i in num_filters:
            model.add(Conv2D(
                filters=i, kernel_size=kernel_size, padding='same',
                activation='relu', kernel_regularizer=l2(beta),
                kernel_constraint=maxnorm(k_constraint)
            ))
            model.add(Conv2D(
                filters=i, kernel_size=kernel_size, padding='same',
                activation='relu', kernel_regularizer=l2(beta),
                kernel_constraint=maxnorm(k_constraint)
            ))
            model.add(MaxPooling2D(pool_size=kernel_size))
            model.add(Dropout(drop_out))

        model.add(Flatten())
        model.add(Dense(
            512, activation='relu', kernel_regularizer=l2(beta),
            kernel_constraint=maxnorm(k_constraint)
        ))
        model.add(Dropout(drop_out))
        model.add(Dense(
            num_labels, activation='relu', name='Output',
            kernel_regularizer=l2(beta),
            kernel_constraint=maxnorm(k_constraint)
        ))

        opt = get_full_optimizer(optimizer, l_rate, momentum)
        model.summary()

        model.compile(
            loss=metric_name, optimizer=opt,
            metrics=[metric_name, determination_function]
        )
    return model


def bayes_optimizer(classifier):
    """
    Method for calculate optimal parameters for naive bayes model

    :param classifier: trained classifying model
    :return: optimal parameters for bayes model
    :rtype: dict
    """

    return {'alpha': 1.0}


def bayes_model(optimal_parameters):
    """
    Return best Bernoulli Naive Bayes Classifier
    base on stratified k-fold cross validation

    :return: best Bernoulli Naive Bayes Classifier
    """

    # build model
    clf_nb = naive_bayes.BernoulliNB().set_params(**optimal_parameters)

    return clf_nb


def logistic_regression_optimizer(classifier):
    """
    Method for calculate optimal parameters for logistic regression model

    :param classifier: trained classifying model
    :return: optimal parameters for logistic regression model
    :rtype: dict
    """

    # define tuning parameters
    # number of L2 penalties to use
    n_alphas_tmp = 5
    alphas_tmp = np.logspace(-5, -1, n_alphas_tmp)
    # define scorer function
    new_scorer = make_scorer(matthews_corrcoef)
    # Logistic Regression Classifier, L2 penalty
    clf_lr_l2 = linear_model.LogisticRegressionCV(
        Cs=alphas_tmp, cv=10, solver='sag', penalty='l2', max_iter=5000,
        random_state=None, verbose=False, scoring=new_scorer
    ).fit(
        classifier.x_train['value'], classifier.y_train['value'][:, 0]
    )

    best_parameters = {
        'penalty': clf_lr_l2.penalty,
        'dual': clf_lr_l2.dual,
        'tol': clf_lr_l2.tol,
        'C': clf_lr_l2.C_[0],
        'fit_intercept': clf_lr_l2.fit_intercept,
        'intercept_scaling': clf_lr_l2.intercept_scaling,
        'class_weight': clf_lr_l2.class_weight,
        'random_state': clf_lr_l2.random_state,
        'solver': clf_lr_l2.solver,
        'max_iter': clf_lr_l2.max_iter,
        'multi_class': clf_lr_l2.multi_class,
        'verbose': clf_lr_l2.verbose,
        'n_jobs': clf_lr_l2.n_jobs
    }

    return best_parameters


def logistic_regression_model(optimal_parameters):
    """
    Linear LogisticRegression Classifier with cross-validation
    Return best Linear Regression stratified k-fold cross validation model

    :return: Linear Regression Classifier
    """

    # Logistic Regression Classifier, L2 penalty
    clf_lr_l2 = linear_model.LogisticRegression().set_params(
        **optimal_parameters)

    return clf_lr_l2


def ada_boost_optimizer(model_trainer):
    """
    Method for calculate optimal parameters for ada boost model

    :param model_trainer: trained classifying model
    :return: optimal parameters for ada boost model
    :rtype: dict
    """

    clf_dt = ensemble.AdaBoostClassifier(
        random_state=model_trainer.seed)
    scorer = matthews_corrcoef
    # define tuning parameters
    tuning_parameters = [{
        'n_estimators': [10,25,50,100,200],
        'learning_rate': [0.9],
        'random_state': [model_trainer.seed]
    }]
    # define scorer function
    new_scorer = make_scorer(scorer)
    # build a model
    model_clf_search = model_selection.GridSearchCV(
        clf_dt, tuning_parameters, cv=10, scoring=new_scorer
    ).fit(
        model_trainer.x_train['value'], model_trainer.y_train['value'][:, 0]
    )

    return model_clf_search.best_params_


def ada_boost_model(optimal_parameters):
    # build model
    clf_ab = ensemble.AdaBoostClassifier().set_params(**optimal_parameters)

    return clf_ab


# dictionary of AdaBoost hyperparameters' ranges for hyperparameter
# optimization with hyperopt package
ada_boost_classifier_hyperparameters = {
    'n_estimators': scope.int(
        hp.qloguniform('n_estimators_', np.log(1e1), np.log(3e3), 1)),
    'learning_rate': hp.loguniform(
        'learning_rate_', np.log(1e-3), np.log(1e1)),
}


# dictionary of AdaBoost hyperparameters' ranges for hyperparameter
# optimization with skopt package
ada_boost_classifier_hyperparameters_skopt = {
    'n_estimators': Integer(1e1, 3e3),
    'learning_rate': Real(1e-3, 1e1, prior='log-uniform'),
}


def weighted_roc_auc(y_true,y_pred):
    """
    Function that generates weighed ROC AUC score
    :param y_true: true label values (probability of true class)
    :param y_pred: predicted label values
    :return: score value
    """
    return roc_auc_score(y_true, y_pred, average='wighted')


def random_forest_optimizer(model_trainer):
    """
    Method for calculate optimal parameters for random forest model

    :param model_trainer: trained classifying model
    :return: optimal parameters for random forest model
    :rtype: dict
    """

    # avoid cross import
    from learner.algorithms import CLASSIFIER, REGRESSOR
    # define scorer function
    model_type = model_trainer.cv_model.model_type

    if model_type == CLASSIFIER:
        clf_rf = ensemble.RandomForestClassifier(
            max_depth=5, class_weight='balanced',
            random_state=model_trainer.seed
        )
        scorer = matthews_corrcoef

    elif model_type == REGRESSOR:
        clf_rf = ensemble.RandomForestRegressor(
            max_depth=5, random_state=model_trainer.seed)
        scorer = r2_score
    else:
        raise TypeError('Unknown modeling type: {}'.format(model_type))

    # build model
    # define tuning parameters
    tuning_parameters = [{'n_estimators': [10,25,50,100,200]}]
    # define scorer function
    new_scorer = make_scorer(scorer)
    # build a model
    model_clf_search = model_selection.GridSearchCV(
        clf_rf, tuning_parameters, cv=3, scoring=new_scorer
    ).fit(
        model_trainer.x_train['value'], model_trainer.y_train['value'][:, 0]
    )

    return model_clf_search.best_params_


def random_forest_classifier_model(optimal_parameters):
    # TODO docsting there
    # build model
    clf_rf = ensemble.RandomForestClassifier().set_params(**optimal_parameters)

    return clf_rf

# dictionary of RandomForest Classifier hyperparameters' ranges for hyperparameter
# optimization with hyperopt package
random_forest_classifier_hyperparameters = {
    'criterion': hp.choice('criterion_', ['gini', 'entropy']),
    'n_estimators': scope.int(
        hp.qloguniform('n_estimators_', np.log(1e1), np.log(3e3), 1)),
    'min_samples_split': scope.int(
        hp.qloguniform('min_samples_split_', np.log(2.0), np.log(50.0), 1)),
}

# dictionary of RandomForest Classifier hyperparameters' ranges for hyperparameter
# optimization with skopt package
random_forest_classifier_hyperparameters_skopt = {
    'n_estimators': Integer(1e1, 3e3),
    'min_samples_split': Integer(2, 50)
}


def random_forest_regressor_model(optimal_parameters):
    # TODO docsting there
    # build model
    reg_rf = ensemble.RandomForestRegressor().set_params(**optimal_parameters)

    return reg_rf

# dictionary of RandomForest Regressor hyperparameters' ranges for hyperparameter
# optimization with hyperopt package
random_forest_regressor_hyperparameters = {
    'n_estimators': scope.int(
        hp.qloguniform('n_estimators_', np.log(1e1), np.log(3e3), 1)),
    'min_samples_split': scope.int(
        hp.qloguniform('min_samples_split_', np.log(2.0), np.log(50.0), 1)),
}

# dictionary of RandomForest Regressor hyperparameters' ranges for hyperparameter
# optimization with skopt package
random_forest_regressor_hyperparameters_skopt = {
    'n_estimators': Integer(1e1, 3e3),
    'min_samples_split': Integer(2, 50)
}


def support_vector_machine_optimizer(model_trainer):
    """
    Method for calculate optimal parameters for SVM model

    :param model_trainer: trained classifying model
    :return: optimal parameters for SVM model
    :rtype: dict
    """

    # avoid cross import
    from learner.algorithms import CLASSIFIER, REGRESSOR

    # define tuning parameters
    tuning_parameters = [{
        'kernel': ['rbf'],
        'gamma': [1e-2, 1e-3],
        'C': [1,5,10]
    }]

    # define scorer function
    model_type = model_trainer.cv_model.model_type
    # build a model
    if model_type == CLASSIFIER:
        model_svm = svm.SVC(
            probability=True, class_weight='balanced',
            random_state=model_trainer.seed
        )
        scorer = matthews_corrcoef
    elif model_type == REGRESSOR:
        model_svm = svm.SVR(degree=3, gamma='auto', shrinking=True)
        scorer = r2_score
    else:
        raise TypeError('Unknown modeling type: {}'.format(model_type))

    new_scorer = make_scorer(scorer)
    model_clf_search = model_selection.GridSearchCV(
        estimator=model_svm, param_grid=tuning_parameters, cv=5,
        scoring=new_scorer
    ).fit(
        model_trainer.x_train['value'], model_trainer.y_train['value'][:, 0]
    )

    return model_clf_search.best_params_


def support_vector_machine_model_classifier(optimal_parameters):
    """
    SVM with radial base function kernel with cross-validation
    (combine into one function)
    Return best SVM stratified k-fold cross validated model

    :param optimal_parameters: optimal parameters for training model
    :return: SVM trained model
    :type optimal_parameters: dict
    """

    model_svm = svm.SVC(probability=True).set_params(**optimal_parameters)

    return model_svm


# dictionary of SVM Classifier hyperparameters' ranges for hyperparameter
# optimization with hyperopt package
support_vector_machine_classifier_hyperparameters = {
    'C': hp.loguniform('C_', np.log(1e-1), np.log(1e5)),
    'tol': hp.loguniform('tol_', np.log(1e-5), np.log(1e-2)),
    'gamma': hp.loguniform('gamma_', np.log(1e-5), np.log(1e-2)),
}


# dictionary of SVM Classifier hyperparameters' ranges for hyperparameter
# optimization with skopt package
support_vector_machine_classifier_hyperparameters_skopt = {
    'C': Real(1e-1, 1e5, prior='log-uniform'),
    'tol': Real(1e-5, 1e-2, prior='log-uniform'),
    'gamma': Real(1e-5, 1e-2, prior='log-uniform'),
}


def support_vector_machine_model_regressor(optimal_parameters):
    """
    SVM with radial base function kernel with cross-validation
    (combine into one function)
    Return best SVM stratified k-fold cross validated model

    :param optimal_parameters: optimal parameters for training model
    :return: SVM trained model
    """

    model_svm = svm.SVR().set_params(**optimal_parameters)

    return model_svm


# dictionary of SVM Regressor hyperparameters' ranges for hyperparameter
# optimization with hyperopt package
support_vector_machine_regressor_hyperparameters = {
    'C': hp.loguniform('C_', np.log(1e-1), np.log(1e5)),
    'tol': hp.loguniform('tol_', np.log(1e-5), np.log(1e-2)),
    'gamma': hp.loguniform('gamma_', np.log(1e-5), np.log(1e-2)),
    'epsilon': hp.loguniform('epsilon_', np.log(1e-3), np.log(5e1)),
}

# dictionary of SVM Regressor hyperparameters' ranges for hyperparameter
# optimization with skopt package
support_vector_machine_regressor_hyperparameters_skopt = {
    'C': Real(1e-1, 1e5, prior='log-uniform'),
    'tol': Real(1e-5, 1e-2, prior='log-uniform'),
    'gamma': Real(1e-5, 1e-2, prior='log-uniform'),
    'epsilon': Real(1e-3, 5e1, prior='log-uniform'),
}


def kernel_ridge_regressor_optimizer(regressor):
    """
    Method for calculate optimal parameters for SVM model

    :param classifier: trained classifying model
    :return: optimal parameters for KRR model
    :rtype: dict
    """
    # TODO docstring there
    # define tuning parameters
    tuning_parameters = [{
        'alpha': [1e-1, 1e-2],
        'kernel': ['chi2'],
        'gamma': [1e-3, 1e-4],
    }]

    # build a model
    model_krr = kernel_ridge.KernelRidge(degree=3)
    scorer = r2_score

    new_scorer = make_scorer(scorer)
    model_clf_search = model_selection.GridSearchCV(
        estimator=model_krr, param_grid=tuning_parameters, cv=5,
        scoring=new_scorer
    ).fit(
        regressor.x_train, regressor.y_train
    )

    return model_clf_search.best_params_


def kernel_ridge_regressor_model(optimal_parameters):
    """
    KRR with ??? with cross-validation
    (combine into one function)
    Return best SVM stratified k-fold cross validated model

    :param modeling_type: type of model. classifying or regression
    :return: SVM trained model
    """
    # TODO docstring there
    model_krr = kernel_ridge.KernelRidge().set_params(**optimal_parameters)

    return model_krr


# dictionary of KernelRidge Regressor hyperparameters' ranges for hyperparameter
# optimization with hyperopt package
kernel_ridge_regressor_hyperparameters = {
    'alpha': hp.loguniform('alpha_', np.log(1e-6), np.log(1e-1)),
    'gamma': hp.loguniform('gamma_', np.log(5e-12), np.log(5e-2)),
    'kernel': hp.choice('kernel_', ['chi2', 'laplacian', 'rbf'])
}


# dictionary of KernelRidge Regressor hyperparameters' ranges for hyperparameter
# optimization with skopt package
kernel_ridge_regressor_hyperparameters_skopt = {
    'alpha': Real(1e-6, 1e-1, prior='log-uniform'),
    'gamma': Real(5e-12, 5e-2, prior='log-uniform'),
    'kernel': Categorical(['chi2', 'laplacian', 'rbf'])
}


def elastic_network_opimizer(regressor):
    """
    Method for calculate optimal parameters for elastic network model

    :param regressor: trained regression model
    :return: optimal parameters for elastic network model
    :rtype: dict
    """

    # define regularizaton constants and k-fold for cross validation
    # a combination of L1 and L2 to use in cross validation
    l1_ratio = [0.05, 0.1, 0.15, .5, .8, .9]
    # 5 alphas
    alphas = np.logspace(-2.5, -1.5, 5)

    # Run regressor with cross-validation
    elastic_network_model = linear_model.ElasticNetCV(
        l1_ratio=l1_ratio, alphas=alphas, max_iter=5000,
        tol=0.0001, cv=5, random_state=0
    ).fit(
        regressor.x_train['value'], regressor.y_train['value'][:, 0]
    )

    return {
        'alpha': elastic_network_model.alpha_,
        'l1_ratio': elastic_network_model.l1_ratio_,
        'fit_intercept': elastic_network_model.fit_intercept,
        'normalize': elastic_network_model.normalize,
        'max_iter': elastic_network_model.max_iter,
        'copy_X': elastic_network_model.copy_X,
        'tol': elastic_network_model.tol,
        'positive': elastic_network_model.positive,
        'random_state': elastic_network_model.random_state,
        'selection': elastic_network_model.selection
    }


# dictionary of ElasticNet hyperparameters' ranges for hyperparameter
# optimization with hyperopt package
elastic_network_hyperparameters = {
    'l1_ratio': hp.uniform('l1_ratio_', 0.0, 1.0),
    'alpha': hp.loguniform('alpha_', np.log(5e-3), np.log(1e0)),
}


# dictionary of ElasticNet hyperparameters' ranges for hyperparameter
# optimization with skopt package
elastic_network_hyperparameters_skopt = {
    'l1_ratio': Real(0.0, 1.0),
    'alpha': Real(5e-3, 1e0, prior='log-uniform'),
}


def elastic_net_cv_model(optimal_parameters):
    """
    ElasticNet (Linear regression with L1 and L2 penalties)
    Return best Elastic Net model
    (Linear regression with combined L1 and L2 priors as regularizer)
    with iterative fitting along a regularization path base
    on stratified k-folds cross validation.

    :param optimal_parameters: optimal parameters for training model
    :return: ElasticNet regressor model
    :type optimal_parameters: dict
    """

    # Run regressor with cross-validation
    regressor = linear_model.ElasticNet().set_params(**optimal_parameters)

    return regressor


def kneighbors_optimizer(model_trainer):
    """
    Method for calculate optimal parameters for K neighbors model

    :param model_trainer: trained classifying model
    :return: optimal parameters for K neighbors model
    :rtype: dict
    """

    # avoid cross import
    from learner.algorithms import CLASSIFIER, REGRESSOR

    scorer = r2_score
    # define tuning parameters
    tuning_parameters = {
        'n_neighbors': range(2, 20),
        'metric': ['euclidean', 'manhattan', 'chebyshev']
    }
    # define scorer function
    new_scorer = make_scorer(scorer)
    model_type = model_trainer.model_type
    if model_type == CLASSIFIER:
        estimator = neighbors.KNeighborsClassifier(algorithm='auto')
        cv = model_selection.StratifiedKFold(5, shuffle=True)
    elif model_type == REGRESSOR:
        estimator = neighbors.KNeighborsRegressor(algorithm='auto')
        cv = model_selection.KFold(5, shuffle=True)
    else:
        raise TypeError('Unknown modeling type: {}'.format(model_type))

    # build model
    model_knn = model_selection.GridSearchCV(
        estimator=estimator, param_grid=tuning_parameters,
        cv=cv, scoring=new_scorer
    ).fit(
        model_trainer.x_train['value'], model_trainer.y_train['value'][:, 0]
    )

    return model_knn.best_params_


def kneighbors_model_classifier(optimal_parameters):
    """
    k-Nearest Neighbors Regressor with cross-validation
    Return best k-Nearest Neighbors Classifier k-fold cross validated model

    :param optimal_parameters: optimal parameters to training model
    :return: k-Nearest Neighbors Regressor
    """

    model_knn = neighbors.KNeighborsClassifier().set_params(
        **optimal_parameters)

    return model_knn


# dictionary of KNN Classifier hyperparameters' ranges for hyperparameter
# optimization with hyperopt package
kneighbors_classifier_hyperparameters = {
    'n_neighbors': scope.int(hp.quniform('n_neighbors_', 1, 20, 1)),
    'metric': hp.pchoice(
        'metric_', [
            (0.9, 'manhattan'),
            (0.07, 'euclidean'),
            (0.03, 'chebyshev'),
        ]
    )
}


# dictionary of KNN Classifier hyperparameters' ranges for hyperparameter
# optimization with skopt package
kneighbors_classifier_hyperparameters_skopt = {
    'n_neighbors': Integer(1, 20),
    'metric': Categorical(['manhattan','euclidean','chebyshev'])
}


def kneighbors_model_regressor(optimal_parameters):
    """
    k-Nearest Neighbors Regressor with cross-validation
    Return best k-Nearest Neighbors Classifier k-fold cross validated model

    :param optimal_parameters: optimal parameters to training model
    :return: k-Nearest Neighbors Regressor
    """

    model_knn = neighbors.KNeighborsRegressor().set_params(
        **optimal_parameters)

    return model_knn


# dictionary of KNN Regressor hyperparameters' ranges for hyperparameter
# optimization with hyperopt package
kneighbors_regressor_hyperparameters = {
    'n_neighbors': scope.int(hp.quniform('n_neighbors_', 1, 20, 1)),
    'metric': hp.pchoice('metric_', [
        (0.9, 'manhattan'),
        (0.07, 'euclidean'),
        (0.03, 'chebyshev'),
    ])
}


# dictionary of KNN Regressor hyperparameters' ranges for hyperparameter
# optimization with skopt package
kneighbors_regressor_hyperparameters_skopt = {
    'n_neighbors': Integer(1, 20),
    'metric': Categorical(['manhattan','euclidean','chebyshev'])
}


def xgboost_optimizer(model_trainer):
    """
    Method for calculate optimal parameters for xgboost model

    :param model_trainer: trained classifying model
    :return: optimal parameters for xgboost model
    :rtype: dict
    """

    # avoid cross import
    from learner.algorithms import CLASSIFIER, REGRESSOR

    # define tuning parameters
    params_dist_grid = make_parameters_distributions_grid()
    params_fixed = make_fixed_parameters_for_xgboost()
    n_iter = 100
    # define scorer function
    model_type = model_trainer.model_type
    if model_type == CLASSIFIER:
        cv = model_selection.StratifiedKFold(5, shuffle=True)
        estimator = xgboost.sklearn.XGBClassifier(**params_fixed)
        scorer = roc_auc_score
    elif model_type == REGRESSOR:
        cv = model_selection.KFold(5, shuffle=True)
        estimator = xgboost.sklearn.XGBRegressor(**params_fixed)
        scorer = r2_score
    else:
        raise TypeError('Unknown modeling type: {}'.format(model_type))

    # build model
    new_scorer = make_scorer(scorer)
    clf_xgb = model_selection.RandomizedSearchCV(
        estimator=estimator, random_state=model_trainer.seed, cv=cv,
        param_distributions=params_dist_grid, scoring=new_scorer, n_iter=n_iter
    ).fit(
        model_trainer.x_train['value'], model_trainer.y_train['value'][:, 0]
    )

    return clf_xgb.best_params_


def xgboost_classifier(optimal_parameters):
    """
    Extreme Gradient Boosting Classifier with cross-validation
    Return best XGBoost Classifier k-fold stratified cross validated model

    :param optimal_parameters: optimal parameters to training model
    :return: XGBoost Classifier
    """

    # define tuning parameters
    params_fixed = make_fixed_parameters_for_xgboost()

    xgboost_model = xgboost.sklearn.XGBClassifier(**params_fixed).set_params(
        **optimal_parameters)

    return xgboost_model


# dictionary of XGBoost Classifier hyperparameters' ranges for hyperparameter
# optimization with hyperopt package
xgboost_classifier_hyperparameters = {
    'max_depth': scope.int(hp.uniform('max_depth_', 3, 11)),
    'learning_rate': hp.loguniform(
        'learning_rate_', np.log(1e-3), np.log(1e-1)),
    'n_estimators': scope.int(hp.quniform('n_estimators_', 1e2, 3e3, 1e2)),
    'gamma': hp.loguniform('gamma_', np.log(1e-4), np.log(1e0)),
    'min_child_weight': scope.int(
        hp.loguniform('min_child_weight_', np.log(1e0), np.log(1e2))),
    'subsample': hp.uniform('subsample_', 0.3, 1.0),
    'colsample_bytree': hp.uniform('colsample_bytree_', 0.2, 1.0),
    'colsample_bylevel': hp.uniform('colsample_bylevel_', 0.2, 1.0),
    'reg_alpha': hp.loguniform('reg_alpha_', np.log(1e-4), np.log(1e1)),
    'reg_lambda': hp.loguniform('reg_lambda_', np.log(1e0), np.log(1e1)),
}


# dictionary of XGBoost Classifier hyperparameters' ranges for hyperparameter
# optimization with skopt package
xgboost_classifier_hyperparameters_skopt = {
    'max_depth': Integer(3, 12),
    'learning_rate': Real(1e-3, 1e-1, prior='log-uniform'),
    'n_estimators': Integer(1e2, 3e3),
    'gamma': Real(1e-4, 1e0, prior='log-uniform'),
    'min_child_weight': Integer(1e0, 1e2),
    'subsample': Real(0.3, 1.0),
    'colsample_bytree': Real(0.2, 1.0),
    'colsample_bylevel': Real(0.2, 1.0),
    'reg_alpha': Real(1e-4, 1e1, prior='log-uniform'),
    'reg_lambda': Real(1e0, 1e1, prior='log-uniform'),
}


def xgboost_regressor(optimal_parameters):
    """
    Extreme Gradient Boosting Classifier with cross-validation
    Return best XGBoost Classifier k-fold stratified cross validated model

    :param optimal_parameters: optimal parameters to training model
    :return: XGBoost Classifier
    """

    # define tuning parameters
    params_fixed = make_fixed_parameters_for_xgboost()

    xgboost_model = xgboost.sklearn.XGBRegressor(**params_fixed).set_params(
        **optimal_parameters)

    return xgboost_model

# dictionary of XGBoost Regressor hyperparameters' ranges for hyperparameter
# optimization with hyperopt package
xgboost_regressor_hyperparameters = {
    'max_depth': scope.int(hp.uniform('max_depth_', 3, 11)),
    'learning_rate': hp.loguniform('learning_rate_', np.log(1e-3), np.log(1e-1)),
    'n_estimators': scope.int(hp.quniform('n_estimators_', 1e2, 3e3, 1e2)),
    'gamma': hp.loguniform('gamma_', np.log(1e-4), np.log(1e0)),
    'min_child_weight': scope.int(
        hp.loguniform('min_child_weight_', np.log(1e0), np.log(1e2))),
    'subsample': hp.uniform('subsample_', 0.3, 1.0),
    'colsample_bytree': hp.uniform('colsample_bytree_', 0.2, 1.0),
    'colsample_bylevel': hp.uniform('colsample_bylevel_', 0.2, 1.0),
    'reg_alpha': hp.loguniform('reg_alpha_', np.log(1e-4), np.log(1e1)),
    'reg_lambda': hp.loguniform('reg_lambda_', np.log(1e0), np.log(1e1)),
}

# dictionary of XGBoost Regressor hyperparameters' ranges for hyperparameter
# optimization with skopt package
xgboost_regressor_hyperparameters_skopt = {
    'max_depth': Integer(3, 12),
    'learning_rate': Real(1e-3, 1e-1, prior='log-uniform'),
    'n_estimators': Integer(1e2, 3e3),
    'gamma': Real(1e-4, 1e0, prior='log-uniform'),
    'min_child_weight': Integer(1e0, 1e2),
    'subsample': Real(0.3, 1.0),
    'colsample_bytree': Real(0.2, 1.0),
    'colsample_bylevel': Real(0.2, 1.0),
    'reg_alpha': Real(1e-4, 1e1, prior='log-uniform'),
    'reg_lambda': Real(1e0, 1e1, prior='log-uniform'),
}


def dnn_optimize_hyperopt(
        x_train, y_train, param_grid, batch_size=256, epochs=200, n_iter=100,
        metric_name=['accuracy'], loss='binary_crossentropy', task='class'
):
    """
    Function that performs optimization of the neural network architecture
    and returs best set of parameters as well as optimization history and model
    :param x_train: pre-processed dataseries, dataframes or numpy arrays
    :param y_train: pre-processed dataseries, dataframes or numpy arrays
    :param batch_size: number of samples per gradient update
    :param epochs: number of epochs to train the model
    :param n_iter: maximum number of evaluations to find the minimum
    :param metric_name: list of metrics to be evaluated by the model
        during training and testing
    :param loss:
    :param type:
    :return: DNN model with the optimal parameters
        obtained by hyperopt optimization,
        best_params - set of the parameters with the best loss
        iters - list of the dicts with all the parameters and corresponding losses
    """
    # TODO type is a bad name!

    x_train, x_test, y_train, y_test = model_selection.train_test_split(
            x_train, y_train, test_size=0.5, random_state=42)

    def model_compiler(space):
        # TODO docstring there
        model = Sequential()
        for k in range(int(space['nh_layers']) - 1):
            if k == 0:
                model.add(Dropout(
                    space['input_drop_out'], input_shape=(x_train.shape[1],),
                    name='Input_dropout'
                ))

            model.add(Dense(
                int(space['hidden_units']), name=f'Dense_{k}',
                kernel_regularizer=l2(space['hidden_kernel_regularizer']),
                kernel_constraint=maxnorm(space['hidden_kernel_constraint']),
                kernel_initializer=space['hidden_kernel_initializer'],
            ))

            model.add(Activation(space['hidden_activation']))
            if space['hidden_batch_normalization'] is True:
                model.add(BatchNormalization())
            model.add(Dropout(space['hidden_drop_out']))

        model.add(Dense(
            space['last_units'],
            name='Dense_{}'.format(int(space["nh_layers"]) - 1),
            kernel_regularizer=l2(space['last_kernel_regularizer']),
            kernel_constraint=maxnorm(space['last_kernel_constraint']),
            kernel_initializer=space['last_kernel_initializer']
        ))
        if task is 'class':
            model.add(Activation('sigmoid'))
        if task is 'reg':
            model.add(Activation('linear'))

        if space['optimizer'] == 'SGD':
            opt = SGD(
                lr=space['l_rate'],
                momentum=space['momentum'],
                nesterov=True,
                decay=0.0000005
            )
        elif space['optimizer'] == 'Adam':
            opt = Adam(lr=space['l_rate'])
        elif space['optimizer'] == 'Nadam':
            opt = Nadam(lr=space['l_rate'])
        elif space['optimizer'] == 'RMSprop':
            opt = RMSprop(lr=space['l_rate'])
        else:
            opt = None

        model.compile(
            loss=loss, optimizer=opt, metrics=metric_name
        )

        print(model.summary())
        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss', factor=0.3, patience=3, min_lr=0.0001,
            verbose=1,epsilon=0.1
        )
        # stop training in no improving in loss for 100 steps
        stopping = EarlyStopping(
            monitor='val_loss', min_delta=0.1, patience=10, verbose=1,
            mode='auto'
        )

        model.fit(
            x_train['value'], y_train['value'], batch_size=batch_size,
            epochs=epochs, verbose=2, callbacks=[reduce_lr,stopping],
            validation_data=(x_test['value'], y_test['value'])
        )

        return model

    def objective(space):
        try:
            model = model_compiler(space)
            # loss = model.evaluate(x_test, y_test, verbose=0)[1]
            if task is 'class':
                loss = matthews_corrcoef(y_test['value'], np.round(model.predict(x_test['value'])))
                print(f'test Matthews correlation coefficient: {loss}')
            if task is 'reg':
                loss = r2_score(y_test['value'], model.predict(x_test['value']))
                print(f'test Determination coefficient: {loss}')

            return {'loss': -loss, 'params': space, 'status': STATUS_OK, 'model': model}
        except:
            return {'loss': 0, 'params': space, 'status': STATUS_FAIL, 'model': '123'}

    trials = Trials()
    results = fmin(
        objective, param_grid, algo=tpe.suggest, trials=trials, max_evals=n_iter)

    iters = trials.results
    print(iters)
    best_iter = sorted(
        iters, key=itemgetter('loss'), reverse=False
    )[0]

    best_model, best_params = best_iter['model'], best_iter['params']
    print(best_params)

    return best_model, best_params, iters


def dnn_optimize_skopt(
        x_train, y_train, param_grid, batch_size=256, epochs=200, n_iter=100,
        metric_name=['accuracy'], loss='binary_crossentropy', opt_method=gp_minimize, task='class',
):
    """

    :param x_train:
    :param y_train:
    :param batch_size:
    :param epochs:
    :param n_iter:
    :param metric_name:
    :param opt_method:
    :return:
    """
    # TODO docstring there
    x_train, x_test, y_train, y_test = model_selection.train_test_split(
            x_train, y_train, test_size=0.5, random_state=42)

    param_keys, param_vecs = zip(*param_grid.items())
    param_keys, param_vecs = list(param_keys), list(param_vecs)

    def model_compiler(space_list):
        space = dict(zip(param_keys, space_list))

        model = Sequential()
        for k in range(space['nh_layers'] - 1):
            if k == 0:
                model.add(Dropout(
                    space['input_drop_out'], input_shape=(x_train.shape[1],),
                    name='Input_dropout'
                ))

            model.add(Dense(
                int(space['hidden_units']), name=f'Dense_{k}',
                kernel_regularizer=l2(space['hidden_kernel_regularizer']),
                kernel_constraint=maxnorm(space['hidden_kernel_constraint']),
                kernel_initializer=space['hidden_kernel_initializer'],
            ))

            model.add(Activation(space['hidden_activation']))
            if space['hidden_batch_normalization'] == True:
                model.add(BatchNormalization())

            model.add(Dropout(space['hidden_drop_out']))

        model.add(Dense(
            space['last_units'],
            name='Dense_{}'.format(int(space["nh_layers"]) - 1),
            kernel_regularizer=l2(space['last_kernel_regularizer']),
            kernel_constraint=maxnorm(space['last_kernel_constraint']),
            kernel_initializer=space['last_kernel_initializer']
        ))

        if task is 'class':
            model.add(Activation('sigmoid'))
        if task is 'reg':
            model.add(Activation('linear'))

        if space['optimizer'] == 'SGD':
            opt = SGD(
                lr=space['l_rate'],
                momentum=space['momentum'],
                nesterov=True,
                decay=0.0000005
            )
        elif space['optimizer'] == 'Adam':
            opt = Adam(lr=space['l_rate'])
        elif space['optimizer'] == 'Nadam':
            opt = Nadam(lr=space['l_rate'])
        elif space['optimizer'] == 'RMSprop':
            opt = RMSprop(lr=space['l_rate'])
        else:
            opt = None

        model.compile(
            loss=loss, optimizer=opt, metrics=metric_name
        )

        print(model.summary())

        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss', factor=0.5, patience=15, min_lr=0.00001,
            verbose = 1
        )
        # stop training in no improving in loss for 100 steps
        stopping = EarlyStopping(
            monitor='val_loss', min_delta=0.01, patience=35, verbose=1,
            mode='auto'
        )

        model.fit(
            x_train['value'], y_train['value'], batch_size=batch_size,
            epochs=epochs, verbose=2, callbacks=[reduce_lr,stopping],
            validation_data=(x_test['value'], y_test['value'])
        )

        return model

    def objective(space):
        try:
            model = model_compiler(space)
            if task is 'class':
                loss = matthews_corrcoef(y_test['value'], np.round(model.predict(x_test['value'])))
                print(f'test Matthews correlation coefficient: {loss}')
            if task is 'reg':
                loss = r2_score(y_test['value'], model.predict(x_test['value']))
                print(f'test Determination coefficient: {loss}')

            return -loss
        except:
            return 0

    outcome = opt_method(objective, param_vecs, n_calls=n_iter)

    results = []
    for err, param_vec in zip(outcome.func_vals, outcome.x_iters):
        params = dict(zip(param_keys, param_vec))
        results.append({'loss': err, 'params': params})

    best_params = sorted(
        results, key=itemgetter('loss'), reverse=False
    )[0]['params']

    return model_compiler(list(best_params.values())), best_params


dnn_classifier_hyperparameters_hyperopt = {
    'input_drop_out': hp.quniform('input_drop_out_', 0, 0.95, 0.05),
    'hidden_units': hp.quniform('hidden_units_', 2, 512, 16),
    'hidden_kernel_regularizer': hp.uniform('hidden_kernel_regularizer_', 0.0000001,0.1),
    'hidden_kernel_constraint': hp.uniform('hidden_kernel_constraint_', 0.5, 6),
    'hidden_kernel_initializer': hp.choice('hidden_kernel_initializer_', [
        'lecun_uniform', 'glorot_uniform', 'he_uniform',
        'lecun_normal', 'glorot_normal', 'he_normal'
    ]),
    'hidden_batch_normalization': hp.choice('hidden_batch_normalization_', [True, False]),
    'hidden_activation': hp.choice('hidden_activation_', ['relu','selu']),
    'hidden_drop_out': hp.quniform('hidden_drop_out_', 0, 0.95, 0.05),
    'last_units': hp.choice('last_units_', [1]),
    'last_kernel_regularizer': hp.choice('last_kernel_regularizer_', [0.001, 0]),
    'last_kernel_constraint': hp.uniform('last_kernel_constraint_', 0.5, 6),
    'last_kernel_initializer': hp.choice('last_kernel_initializer_', ['he_normal']),
    'nh_layers': scope.int(hp.quniform('nh_layers_', 2, 5, 1)),
    'l_rate': hp.uniform('l_rate_', 0.00001, 0.1),
    'optimizer': hp.choice('optimizer_', ['Adam', 'Nadam', 'RMSprop', 'SGD']),
    'momentum': hp.choice('momentum_', [0.9]),
}


dnn_classifier_hyperparameters_skopt = {
    'input_drop_out': Real(0.0, 0.95),
    'hidden_units': Categorical([16, 32, 64, 128, 256, 512]),
    'hidden_kernel_regularizer': Real(1e-4, 1e-2, prior='log-uniform'),
    'hidden_kernel_constraint': Real(0.5, 6.0),
    'hidden_kernel_initializer': Categorical([
        'lecun_uniform', 'glorot_uniform', 'he_uniform',
        'lecun_normal', 'glorot_normal', 'he_normal'
    ]),
    'hidden_batch_normalization': Categorical([0, 1]),
    'hidden_drop_out': Real(0.0, 0.75),
    'hidden_activation': Categorical(['relu', 'selu']),
    'last_units': Categorical([1]),
    'last_kernel_regularizer': Real(1e-7, 1e-4, prior="log-uniform"),
    'last_kernel_constraint': Categorical([1.0, 2.0, 3.0, 4.0, 100.0]),
    'last_kernel_initializer': Categorical([
        'lecun_uniform', 'glorot_uniform', 'he_uniform',
        'lecun_normal', 'glorot_normal', 'he_normal'
    ]),
    'nh_layers': Integer(2, 5),
    'l_rate': Real(1e-4, 1e-2, prior='log-uniform'),
    'optimizer': Categorical(['Adam', 'Nadam', 'RMSprop', 'SGD']),
    'momentum': Categorical([0.9]),
}


dnn_regressor_hyperparameters_hyperopt = {
    'input_drop_out': hp.quniform('input_drop_out_', 0, 0.95, 0.05),
    'hidden_units': hp.quniform('hidden_units_', 2, 512, 16),
    'hidden_kernel_regularizer': hp.uniform('hidden_kernel_regularizer_', 0.0000001,0.1),
    'hidden_kernel_constraint': hp.uniform('hidden_kernel_constraint_', 0.5, 6),
    'hidden_kernel_initializer': hp.choice('hidden_kernel_initializer_', [
        'lecun_uniform', 'glorot_uniform', 'he_uniform',
        'lecun_normal', 'glorot_normal', 'he_normal'
    ]),
    'hidden_batch_normalization': hp.choice('hidden_batch_normalization_', [True, False]),
    'hidden_activation': hp.choice('hidden_activation_', ['relu','selu']),
    'hidden_drop_out': hp.quniform('hidden_drop_out_', 0, 0.95, 0.05),
    'last_units': hp.choice('last_units_', [1]),
    'last_kernel_regularizer': hp.choice('last_kernel_regularizer_', [0.001, 0]),
    'last_kernel_constraint': hp.uniform('last_kernel_constraint_', 0.5, 6),
    'last_kernel_initializer': hp.choice('last_kernel_initializer_', ['he_normal']),
    'nh_layers': hp.quniform('nh_layers_', 2, 5, 1),
    'l_rate': hp.uniform('l_rate_', 0.00001, 0.1),
    'optimizer': hp.choice('optimizer_', ['Adam', 'Nadam', 'RMSprop', 'SGD']),
    'momentum': hp.choice('momentum_', [0.9]),
}


dnn_regressor_hyperparameters_skopt = {
    'input_drop_out': Real(0.0, 0.95),
    'hidden_units': Categorical([16, 32, 64, 128, 256, 512]),
    'hidden_kernel_regularizer': Real(1e-4, 1e-2, prior='log-uniform'),
    'hidden_kernel_constraint': Real(0.5, 6.0),
    'hidden_kernel_initializer': Categorical([
        'lecun_uniform', 'glorot_uniform', 'he_uniform',
        'lecun_normal', 'glorot_normal', 'he_normal'
    ]),
    'hidden_batch_normalization': Categorical([0, 1]),
    'hidden_drop_out': Real(0.0, 0.75),
    'hidden_activation': Categorical(['relu', 'selu']),
    'last_units': Categorical([1]),
    'last_kernel_regularizer': Real(1e-7, 1e-4, prior="log-uniform"),
    'last_kernel_constraint': Categorical([1.0, 2.0, 3.0, 4.0, 100.0]),
    'last_kernel_initializer': Categorical([
        'lecun_uniform', 'glorot_uniform', 'he_uniform',
        'lecun_normal', 'glorot_normal', 'he_normal'
    ]),
    'nh_layers': Integer(2, 5),
    'l_rate': Real(1e-4, 1e-2, prior='log-uniform'),
    'optimizer': Categorical(['Adam', 'Nadam', 'RMSprop', 'SGD']),
    'momentum': Categorical([0.9]),
}


def cml_cross_validated_scorer(model, x_train, y_train, params, mode, kfolds):
    """
    Method to calculate an average metric with cross-validation (CML hyperparameters optimization)
    :param model:
    :param x_train:
    :param y_train:
    :param params:
    :param mode:
    :param kfolds:
    :return:
    """
    # TODO docstring there
    from learner.algorithms import CLASSIFIER, REGRESSOR
    if mode == CLASSIFIER:
        cv_score = - model_selection.cross_val_score(
            model(params), x_train, y=y_train, cv=kfolds,
            scoring=make_scorer(matthews_corrcoef)
        ).mean()
    elif mode == REGRESSOR:
        cv_score = model_selection.cross_val_score(
            model(params), x_train, y=y_train, cv=kfolds,
            scoring=make_scorer(mean_squared_error)
        ).mean()
    else:
        raise ValueError(f'unknown mode: {mode}')
    return cv_score


def cml_hyperopt_search(trainer, model, param_grid):
    """
    Method to optimize hyperparameters of CML algorithms with hyperopt package
    :param trainer:
    :param model:
    :param param_grid:
    :return:
    """
    # TODO docstring there
    def objective(params):
        # TODO docstring there
        err = cml_cross_validated_scorer(
            model, trainer.x_train['value'], trainer.y_train['value'], params,
            trainer.model_type, kfolds=3
        )
        return {'loss': err, 'params': params, 'status': STATUS_OK}

    trials = Trials()
    # TODO results unused
    results = fmin(
        objective, param_grid, algo=tpe.suggest, trials=trials,
        max_evals=trainer.n_iter_optimize
    )
    best_params = sorted(
        trials.results, key=itemgetter('loss'), reverse=False
    )[0]['params']

    return best_params


def cml_skopt_search(trainer, model, param_grid, skopt_method=gp_minimize):
    """
    Method to optimize hyperparameters of CML algorithms with skopt package
    :param trainer:
    :param model:
    :param param_grid:
    :param skopt_method:
    :return:
    """
    def objective(param_vec):
        # TODO docstring there
        params = dict(zip(param_keys, param_vec))
        err = cml_cross_validated_scorer(
            model, trainer.x_train['value'], trainer.y_train['value'], params,
            trainer.model_type, kfolds=3
        )

        return err

    results = []
    param_keys, param_vecs = map(list, zip(*param_grid.items()))
    outcome = skopt_method(objective, list(param_vecs), n_calls=trainer.n_iter_optimize)

    for err, param_vec in zip(outcome.func_vals, outcome.x_iters):
        params = dict(zip(param_keys, param_vec))
        results.append({'loss': err, 'params': params})

    best_params = sorted(
        results, key=itemgetter('loss'), reverse=False
    )[0]['params']

    return best_params


def cml_optimizer(trainer, model, param_grid):
    """
    Method to choose algorithm/package for CML hyperparameters optimization
    :param trainer:
    :param model:
    :param param_grid:
    :return:
    """

    # TODO docstring there
    if trainer.opt_method == 'parzen':
        optimizer = cml_hyperopt_search(
            trainer, model, param_grid)
    elif trainer.opt_method == 'forest':
        optimizer = cml_skopt_search(
            trainer, model, param_grid, forest_minimize)
    elif trainer.opt_method == 'gbrt':
        optimizer = cml_skopt_search(
            trainer, model, param_grid, gbrt_minimize)
    elif trainer.opt_method == 'gauss':
        optimizer = cml_skopt_search(
            trainer, model, param_grid, gp_minimize)
    else:
        raise ValueError('unknown optimizer: {}'.format(trainer.opt_method))

    return optimizer


def optimizer(trainer, model, param_grid):
    """

    :param trainer:
    :param model:
    :param param_grid:
    :return:
    """
    if trainer.model_name == 'DNN Regressor' or trainer.model_name == 'DNN Classifier':
        if trainer.model_name == 'DNN Regressor':
            task = 'reg'
            metric_name = 'mean_squared_error'
            determination_function = coeff_determination
        elif trainer.model_name == 'DNN Classifier':
            task = 'class'
            metric_name = 'binary_crossentropy'
            determination_function = 'accuracy'
        else:
            raise ValueError('unknown model_type')

        if trainer.opt_method == 'parzen':
            best_model, opt_params, iters = dnn_optimize_hyperopt(
                trainer.x_train, trainer.y_train,
                param_grid,
                n_iter=trainer.n_iter_optimize,
                loss=[metric_name],
                metric_name=[metric_name],
                task=task,
            )
            trainer.optimization_history = iters
            alpha_list = []
            counter = 0
            for iteration in trainer.optimization_history:
                params = iteration['params']
                if counter == 0:
                    headers = ['metric']
                    headers.extend(params.keys())
                    alpha_list.append(headers)
                else:
                    row = []
                    loss_value = iteration['loss']
                    row.append(loss_value)
                    row.extend(params.values())
                    alpha_list.append(row)
                counter += 1

            df = pd.DataFrame(alpha_list)
            df.to_csv('{}/params.csv'.format(trainer.sub_folder)
                      , header=None, index=False)
        elif trainer.opt_method == 'forest':
            opt_params = dnn_optimize_skopt(
                trainer.x_train, trainer.y_train,
                param_grid,
                opt_method=forest_minimize,
                n_iter=trainer.n_iter_optimize,
                loss=[metric_name],
                metric_name=[metric_name],
                task=task,
            )[1]
        elif trainer.opt_method == 'gbrt':
            opt_params = dnn_optimize_skopt(
                trainer.x_train, trainer.y_train,
                param_grid,
                opt_method=gbrt_minimize,
                n_iter=trainer.n_iter_optimize,
                loss=[metric_name],
                metric_name=[metric_name],
                task=task,
            )[1]
        elif trainer.opt_method == 'gauss':
            opt_params = dnn_optimize_skopt(
                trainer.x_train, trainer.y_train,
                param_grid,
                opt_method=gp_minimize,
                n_iter=trainer.n_iter_optimize,
                loss=[metric_name],
                metric_name=[metric_name],
                task=task,
            )[1]
        else:
            raise ValueError(f'unknown optimizer: {regressor.opt_method}')

        print("Optimal set of parameters for DNN:")
        print(opt_params)

        optimal_parameters = {
            'nodes_in_layers': int(opt_params['nh_layers'])*[int(opt_params['hidden_units'])],
            'gpu': '/gpu:0',
            'hidden_kernel_initializer': opt_params['hidden_kernel_initializer'],
            'optimizer': opt_params['optimizer'],
            'activation': opt_params['hidden_activation'],
            'l_rate': opt_params['l_rate'],
            'momentum': opt_params['momentum'],
            'beta': opt_params['hidden_kernel_regularizer'],
            'drop_out': opt_params['hidden_drop_out'],
            'input_drop_out': opt_params['input_drop_out'],
            'nh_layers': opt_params['nh_layers'],
            'k_constraint': opt_params['hidden_kernel_constraint'],
            'device': '/gpu:0',
            'model_summary': True,
            'metric_name': metric_name,
            'determination_function': determination_function,
            'warm_up_batch_size': None,
            'warm_up_lr_rate': 0.0005,
            'warm_up_momentum': 0,
            'warm_up_optimizer': 'SGD',
            'batch_norm': True,
            'input_dim': trainer.bins
        }
    else:
        if trainer.opt_method == 'parzen':
            optimal_parameters = cml_hyperopt_search(
                trainer, model, param_grid)
        elif trainer.opt_method == 'forest':
            optimal_parameters = cml_skopt_search(
                trainer, model, param_grid, forest_minimize)
        elif trainer.opt_method == 'gbrt':
            optimal_parameters = cml_skopt_search(
                trainer, model, param_grid, gbrt_minimize)
        elif trainer.opt_method == 'gauss':
            optimal_parameters = cml_skopt_search(
                trainer, model, param_grid, gp_minimize)
        else:
            raise ValueError('unknown optimizer: {}'.format(trainer.opt_method))

    return optimal_parameters


def make_parameters_distributions_for_hyperopt():
    """
    Parameters distributions for XGBoost Classifier (hyperopt optimization)

    :return:
    """
    hyperopt_grid = {
        'max_depth': hp.choice('max_depth', range(4, 13, 1)),
        'learning_rate': hp.quniform('learning_rate', 0.01, 0.5, 0.01),
        'n_estimators': hp.choice('n_estimators', range(100, 1000, 5)),
        'objective': 'binary:logistic',
        'gamma': hp.quniform('gamma', 0, 0.50, 0.01),
        'min_child_weight': hp.quniform('min_child_weight', 1, 5, 1),
        'subsample': hp.quniform('subsample', 0.1, 1, 0.01),
        'colsample_bytree': hp.quniform('colsample_bytree', 0.1, 1.0, 0.01)
    }

    return hyperopt_grid


def make_parameters_distributions_for_skopt():
    """
    Parameters distributions for XGBoost Classifier (skopt optimization)
    :return:
    """
    skopt_grid = {
        'max_depth': (4, 12),
        'learning_rate': (0.01, 0.5),
        'n_estimators': (100, 1000),
        'objective': Categorical(('binary:logistic',)),
        'gamma': (0, 0.5),
        'min_child_weight': (1, 5),
        'subsample': (0.1, 1.0),
        'colsample_bytree': (0.1, 1.0)
    }

    return skopt_grid


def get_short_optimizer(optimizer, l_rate, momentum):
    """
    Method for getting "short" optimizer function

    :param optimizer: optimizer name
    :param l_rate:
    :param momentum:
    :return: optimizer funcion
    """
    # TODO check docstring. why have difference with get_full_optimizer??

    if optimizer == 'SGD':
        opt = SGD(lr=l_rate, momentum=momentum, nesterov=True,clipnorm=1.0)
    elif optimizer == 'Adam':
        opt = Adam(lr=l_rate,clipnorm=1.0)
    elif optimizer == 'Nadam':
        opt = Nadam(lr=l_rate,clipnorm=1.0)
    elif optimizer == 'RMSprop':
        opt = RMSprop(lr=l_rate, clipnorm=1.0)
    else:
        opt = None

    return opt


def get_full_optimizer(optimizer, l_rate, momentum):
    """
    Method for get "full" optimizer function

    :param optimizer: optimizer name
    :param l_rate:
    :param momentum:
    :return: optimizer function
    """
    # TODO check docstring. why have difference with get_short_optimizer??

    if optimizer == 'SGD':
        opt = SGD(
            lr=l_rate, momentum=momentum, nesterov=False, decay=0.0000005,
            clipnorm=1.0
        )
    elif optimizer == 'Adam':
        opt = Adam(lr=l_rate, clipnorm=1.0)
    elif optimizer == 'Nadam':
        opt = Nadam(lr=l_rate, clipnorm=1.0)
    elif optimizer == 'RMSprop':
        opt = RMSprop(lr=l_rate, clipnorm=1.0)
    elif optimizer == 'Adagrad':
        opt = Adagrad(lr=l_rate, clipnorm=1.0)
    elif optimizer == 'Adadelta':
        opt = Adadelta(lr=l_rate, clipnorm=1.0)
    elif optimizer == 'Adamax':
        opt = Adamax(lr=l_rate, clipnorm=1.0)
    else:
        opt = None

    return opt


def get_activation_function(activation):
    """
    Method for getting activation function accord to function name

    :param activation:
    :param act:
    :return: activation function
    """
    # TODO check docstring. why activation and act?

    if activation == 'LeakyReLU':
        # add an advanced activation
        activation_function = LeakyReLU()
    elif activation == 'PReLU':
        # add an advanced activation
        activation_function = PReLU()
    elif activation == 'ELU':
        # add an advanced activation
        activation_function = ELU()
    elif activation == 'ThresholdedReLU':
        # add an advanced activation
        activation_function = ThresholdedReLU()
    else:
        activation_function = Activation(activation)

    return activation_function


def make_fixed_parameters_for_xgboost():
    """
    Method for make fixed parameters for xgboost modeling

    :return: fixed parameters
    """
    # TODO check docstring
    fixed_parameters = {
        'silent': 1,  # 0 means printing running messages, 1 means silent mode
    }

    # add GPU usage if xgboost compiled with GPU support
    if xgboost.__version__ == GPU_XGBOOST_VERSION:
        fixed_parameters['tree_method'] = 'gpu_hist'

    return fixed_parameters


def make_parameters_distributions_grid():
    """
    Method which make parameters distributions grid for xgboost modeling

    :return: parameters distribution grid
    """
    # TODO check docstring
    parameters_distribution_grid = {
        # maximum depth of a tree
        'max_depth': randint(1, 13),
        # step size shrinkage used in update to prevents overfitting
        'learning_rate': uniform(0.01, 0.5),
        # minimum loss reduction
        # required to make a further partition on a leaf node of the tree
        'gamma': uniform(),
        # subsample ratio of the training instance
        'subsample': uniform(),
        # subsample ratio of columns when constructing each tree
        'colsample_bytree': uniform(0.7, 0.3),
        # minimum sum of instance weight (hessian) needed in a child
        'min_child_weight': uniform(0, 10),
        # number of boosted trees to fit
        'n_estimators': randint(100, 1000)
    }

    return parameters_distribution_grid


def dnn_regressor_arguments(regressor):
    # TODO docstring there
    num_steps = 10001

    reduce_lr = ReduceLROnPlateau(
        monitor='loss', factor=0.3, patience=10, min_lr=0.00001, verbose=1,
        epsilon=0.00001
    )
    earlystopping = EarlyStopping(
        monitor='val_loss', min_delta=0.00001, patience=50, verbose=1,
        mode='auto'
    )

    # TODO temporary hardcode
    shuffle = True
    batch_size_dnn = 128

    return {
        'epochs': num_steps,
        'batch_size': batch_size_dnn,
        'validation_data': None,
        'callbacks': [earlystopping, reduce_lr],
        'shuffle': shuffle,
        'verbose': 2
    }


def dnn_multi_regressor_arguments(regressor):
    # TODO docstring there
    num_steps = 10001

    reduce_lr = ReduceLROnPlateau(
        monitor='loss', factor=0.5, patience=15, min_lr=0, verbose=1,
        epsilon=0.00001
    )
    earlystopping = EarlyStopping(
        monitor='val_loss', min_delta=0.00001, patience=100, verbose=1,
        mode='auto'
    )

    # TODO temporary hardcode
    shuffle = True
    batch_size_dnn = 128

    return {
        'epochs': num_steps,
        'batch_size': batch_size_dnn,
        'validation_data': None,
        'callbacks': [earlystopping, reduce_lr],
        'shuffle': shuffle,
        'verbose': 2
    }


def dnn_regressor_optimizer(regressor):
    """
    Method for calculate optimal parameters for DNN regression model

    :param regressor: trained regression model
    :return: optimal parameters for DNN regression model
    :rtype: dict
    """

    optimal_parameters = {
        'nodes_in_layers': [512,512,512],
        'gpu': '/gpu:0',
        'optimizer': 'Nadam',
        'hidden_kernel_initializer': 'he_normal',
        'drop_out': 0.4,
        'input_drop_out': 0.0,
        'activation': 'relu',
        'k_constraint': 5,
        'l_rate': 0.005,
        'beta': 0.0001,
        'epochs': 10001,
        'num_labels': 1,
        'cv_r2_cut_off': 0,
        'momentum': 0.9,
        'model_summary': True,
        'metric_name': 'mean_squared_error',
        'determination_function': coeff_determination,
        'warm_up_batch_size': None,
        'warm_up_lr_rate': 0.0005,
        'warm_up_momentum': 0,
        'warm_up_optimizer': 'SGD',
        'batch_norm': True,
        'input_dim': regressor.bins
    }

    return optimal_parameters


def dnn_multi_regressor_optimizer(regressor):
    """
    Method for calculate optimal parameters for DNN regression model

    :param regressor: trained regression model
    :return: optimal parameters for DNN regression model
    :rtype: dict
    """

    optimal_parameters = {
        'nodes_in_layers': [128],
        'gpu': '/gpu:0',
        'optimizer': 'Nadam',
        'drop_out': 0.5,
        'input_drop_out': 0.2,
        'activation': 'relu',
        'k_constraint': 3,
        'l_rate': 0.01,
        'beta': 0.001,
        'epochs': 10001,
        # 'num_labels': 1,
        'cv_r2_cut_off': 0,
        'momentum': 0.9,
        'model_summary': True,
        'metric_name': 'mean_squared_error',
        'determination_function': coeff_determination,
        'warm_up_batch_size': None,
        'warm_up_lr_rate': 0.0005,
        'warm_up_momentum': 0,
        'warm_up_optimizer': 'SGD',
        'batch_norm': True,
        'input_dim': regressor.bins
    }

    return optimal_parameters


def compile_dnn_reg_cv_new(optimal_parameters):
    # TODO docstring there
    n_layer = len(optimal_parameters['nodes_in_layers'])

    num_hidden = optimal_parameters['nodes_in_layers']

    model_base = TMP_model_dnn(
        num_labels=1,
        nh_layers=n_layer,
        input_dim=optimal_parameters['input_dim'],
        drop_out=optimal_parameters['drop_out'],
        input_drop_out=optimal_parameters['input_drop_out'],
        device=optimal_parameters['gpu'],
        num_hidden=num_hidden,
        activation=optimal_parameters['activation'],
        beta=optimal_parameters['beta'],
        k_constraint=optimal_parameters['k_constraint'],
        activation_out='linear',
        batch_norm=optimal_parameters['batch_norm'],
        kernel_initializer=optimal_parameters['hidden_kernel_initializer']
    )

    metric_name = optimal_parameters['metric_name']
    determination_function = optimal_parameters['determination_function']
    if optimal_parameters['warm_up_batch_size'] is not None:
        opt = get_full_optimizer(
            optimal_parameters['warm_up_optimizer'],
            optimal_parameters['warm_up_lr_rate'],
            optimal_parameters['warm_up_momentum']
        )
        if not opt:
            return 'I don\'t know the {} optimizer'.format(
                optimal_parameters['optimizer'])

        if optimal_parameters['model_summary']:
            model_base.summary()

        model_base.compile(
            loss=metric_name, optimizer=opt,
            metrics=[metric_name, determination_function]
        )

    # compile model
    opt = get_full_optimizer(
        optimal_parameters['optimizer'],
        optimal_parameters['l_rate'],
        optimal_parameters['momentum']
    )
    if not opt:
        return 'I don\'t know the {} optimizer'.format(
            optimal_parameters['optimizer'])
    if optimal_parameters['model_summary']:
        model_base.summary()
    model_base.compile(
        loss=metric_name, optimizer=opt,
        metrics=[metric_name, determination_function]
    )

    return model_base


def train_dnn_multi_reg(optimal_parameters):
    # TODO docstring there
    n_layer = len(optimal_parameters['nodes_in_layers'])

    num_hidden = optimal_parameters['nodes_in_layers']

    model_base = TMP_model_dnn(
        num_labels=optimal_parameters['num_labels'],
        nh_layers=n_layer,
        input_dim=optimal_parameters['input_dim'],
        drop_out=optimal_parameters['drop_out'],
        input_drop_out=optimal_parameters['input_drop_out'],
        device=optimal_parameters['gpu'],
        num_hidden=num_hidden,
        activation=optimal_parameters['activation'],
        beta=optimal_parameters['beta'],
        k_constraint=optimal_parameters['k_constraint'],
        activation_out='linear',
        batch_norm=optimal_parameters['batch_norm']
    )

    metric_name = optimal_parameters['metric_name']
    determination_function = optimal_parameters['determination_function']
    if optimal_parameters['warm_up_batch_size'] is not None:
        opt = get_full_optimizer(
            optimal_parameters['warm_up_optimizer'],
            optimal_parameters['warm_up_lr_rate'],
            optimal_parameters['warm_up_momentum']
        )
        if not opt:
            return 'I don\'t know the {} optimizer'.format(
                optimal_parameters['optimizer'])

        if optimal_parameters['model_summary']:
            model_base.summary()

        model_base.compile(
            loss=metric_name, optimizer=opt,
            metrics=[metric_name, determination_function]
        )

    # compile model
    opt = get_full_optimizer(
        optimal_parameters['optimizer'],
        optimal_parameters['l_rate'],
        optimal_parameters['momentum']
    )
    if not opt:
        return 'I don\'t know the {} optimizer'.format(
            optimal_parameters['optimizer'])
    if optimal_parameters['model_summary']:
        model_base.summary()
    model_base.compile(
        loss=metric_name, optimizer=opt,
        metrics=[metric_name, determination_function]
    )

    return model_base


def dnn_classifier_arguments(classifier):
    # TODO docstring there
    num_steps = 10001

    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss', factor=0.5, patience=15, min_lr=0.00001,
        verbose=1,
    )
    # stop training in no improving in loss for 100 steps
    stopping = EarlyStopping(
        monitor='val_loss', min_delta=0.0001, patience=100, verbose=1,
        mode='auto'
    )
    out_batch = BatchLogger(display=5)

    # TODO temporary hardcode
    shuffle = True
    batch_size_dnn = 256
    class_weight = class_weight_to_dict(classifier.y_train[:, 0]['value'])

    return {
        'epochs': num_steps,
        'batch_size': batch_size_dnn,
        'validation_data': None,
        'callbacks': [reduce_lr, stopping, out_batch],
        'shuffle': shuffle,
        'verbose': 2,
        'class_weight': class_weight
    }


def dnn_multi_classifier_arguments(classifier):
    # TODO docstring there
    num_steps = 10001

    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss', factor=0.5, patience=15, min_lr=0.00001,
        verbose=1
    )
    # stop training in no improving in loss for 100 steps
    stopping = EarlyStopping(
        monitor='val_loss', min_delta=0.01, patience=100, verbose=1,
        mode='auto'
    )
    out_batch = BatchLogger(display=5)

    # TODO temporary hardcode
    shuffle = True
    batch_size_dnn = 512

    return {
        'epochs': num_steps,
        'batch_size': batch_size_dnn,
        'validation_data': None,
        'callbacks': [reduce_lr, stopping, out_batch],
        'shuffle': shuffle,
        'verbose': 2,
        'class_weight': None,
    }


def dnn_classifier_optimizer(classifier):
    optimal_parameters = {
        'nodes_in_layers': [512,512,512],
        'shuffle': True,
        'gpu': '/gpu:0',
        'optimizer': 'Adam',
        'activation': 'relu',
        'l_rate': 0.005,
        'momentum': 0.9,
        'beta': 0.001,
        'drop_out': 0.3,
        'input_drop_out': 0,
        'nh_layers': 3,
        'input_dim': classifier.bins,
        'k_constraint': 4,
        'device': '/gpu:0',
        'model_summary': True,
        'metric_name': 'binary_crossentropy',
        'determination_function': 'accuracy'
    }

    return optimal_parameters


def dnn_multi_classifier_optimizer(classifier):
    """
    Method for calculate optimal parameters for DNN classifying model

    :param classifier: trained classifying model
    :return: optimal parameters for DNN classifying model
    :rtype: dict
    """

    optimal_parameters = {
        'nodes_in_layers': [32],
        'shuffle': True,
        'gpu': '/gpu:0',
        'optimizer': 'Nadam',
        'activation': 'relu',
        'l_rate': 0.01,
        'momentum': 0.9,
        'beta': 0.001,
        'drop_out': 0.1,
        'input_drop_out': 0.1,
        'nh_layers': len([32]),
        'input_dim': classifier.bins,
        'num_hidden': [32],
        'k_constraint': 4,
        'device': '/gpu:0',
        'activation_out': 'softmax',
        'model_summary': True,
        'metric_name': 'categorical_crossentropy',
        'determination_function': 'accuracy'
    }

    return optimal_parameters


def train_dnn_valid_new(optimal_parameters):
    """
    Function that creates and compiles a DNN model, given a dictionary of
    hyperparameters
    :param optimal_parameters: dictionary of hyperparameters
    :return: keras model
    """
    # TODO docstring there
    n_layer = optimal_parameters['nh_layers']
    num_hidden = optimal_parameters['nodes_in_layers']

    # constructing model
    model_dnn_clf = TMP_model_dnn(
        nh_layers=n_layer,
        input_dim=optimal_parameters['input_dim'],
        drop_out=optimal_parameters['drop_out'],
        input_drop_out=optimal_parameters['input_drop_out'],
        num_hidden=num_hidden, k_constraint=4,
        activation=optimal_parameters['activation'],
        beta=optimal_parameters['beta'],
        device=optimal_parameters['gpu'],
        activation_out='sigmoid'
    )

    # compile model
    opt = get_full_optimizer(
        optimal_parameters['optimizer'],
        optimal_parameters['l_rate'],
        optimal_parameters['momentum']
    )
    if not opt:
        return 'I don\'t know the {} optimizer'.format(
            optimal_parameters['optimizer'])

    if optimal_parameters['model_summary']:
        model_dnn_clf.summary()
    model_dnn_clf.compile(
        loss=optimal_parameters['metric_name'],
        optimizer=opt,
        metrics=[
            optimal_parameters['metric_name'],
            optimal_parameters['determination_function']
        ]
    )

    return model_dnn_clf


def train_dnn_multi_valid(optimal_parameters):
    # TODO docstring there
    n_layer = optimal_parameters['nh_layers']
    num_hidden = optimal_parameters['nodes_in_layers']

    # constructing model
    model_dnn_clf = TMP_model_dnn(
        nh_layers=n_layer,
        num_labels=optimal_parameters['num_labels'],
        input_dim=optimal_parameters['input_dim'],
        drop_out=optimal_parameters['drop_out'],
        input_drop_out=optimal_parameters['input_drop_out'],
        num_hidden=num_hidden, k_constraint=4,
        activation=optimal_parameters['activation'],
        beta=optimal_parameters['beta'],
        device=optimal_parameters['gpu'],
        activation_out='sigmoid'
    )

    # compile model
    opt = get_full_optimizer(
        optimal_parameters['optimizer'],
        optimal_parameters['l_rate'],
        optimal_parameters['momentum']
    )
    if not opt:
        return 'I don\'t know the {} optimizer'.format(
            optimal_parameters['optimizer'])

    if optimal_parameters['model_summary']:
        model_dnn_clf.summary()
    model_dnn_clf.compile(
        loss=optimal_parameters['metric_name'],
        optimizer=opt,
        metrics=[
            optimal_parameters['metric_name'],
            optimal_parameters['determination_function']
        ]
    )

    return model_dnn_clf
