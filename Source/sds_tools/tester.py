import pandas as pd
from keras.models import load_model
from sklearn.metrics import mean_squared_error, r2_score

from general_helper import coeff_determination
from learner.plotters import plot_actual_predicted


def reg_test_model(
    modelpath, x_test, y_test, sub_folder,
    modelname='RegressionModel', property_name='Property'
):
    model = load_model(
        modelpath,
        custom_objects={'coeff_determination': coeff_determination}
    )
    x_test = x_test.as_matrix()
    y_test = y_test.as_matrix()

    test_predict = model.predict(x_test)
    mse = mean_squared_error(y_test, test_predict)
    r2 = r2_score(y_test, test_predict)
    plot_actual_predicted(
        model, sub_folder, '{}'.format(modelname), '{}'.format(property_name),
        x_test, y_test, ds_type='test'
    )

    return {'MSE': mse, 'R2_score': r2}


def get_metrics(y_test, y_predicted):
    mse = mean_squared_error(y_test, y_predicted)
    r2 = r2_score(y_test, y_predicted)
    return {'MSE': mse, 'R2_score': r2}


def parse_test_dataset(filepath):
    dataset = pd.read_table(filepath, sep='\t')
    dataset = dataset[dataset.Consensus != -9999]
    dataset = dataset[dataset.expToxicValue != -9999]
    y_test = dataset['expToxicValue']
    y_predicted = dataset['Consensus']
    return y_test, y_predicted


def get_test_metrics(filepath):
    y_test, y_predicted = parse_test_dataset(filepath)
    return get_metrics(y_test, y_predicted)


def main():

    sets_list = [
        {'path': 'C:\PycharmProjects\ml-data-qsar\TEST\BCF\BCF_training.sdf', 'prop': 'Tox'},
        {'path': 'C:\PycharmProjects\ml-data-qsar\TEST\BP\BP_training.sdf', 'prop': 'Tox'},
        {'path': 'C:\PycharmProjects\ml-data-qsar\TEST\Density\Density_training.sdf', 'prop': 'Tox'},
        {'path': 'C:\PycharmProjects\ml-data-qsar\TEST\FP\FP_training.sdf', 'prop': 'Tox'},
        {'path': 'C:\PycharmProjects\ml-data-qsar\TEST\ER_LogRBA\ER_LogRBA_training.sdf', 'prop': 'Tox'},
        {'path': 'C:\PycharmProjects\ml-data-qsar\TEST\IGC50\IGC50_training.sdf', 'prop': 'Tox'},
        {'path': 'C:\PycharmProjects\ml-data-qsar\TEST\LD50\LD50_training.sdf', 'prop': 'Tox'},
        {'path': 'C:\PycharmProjects\ml-data-qsar\TEST\LC50\LC50_training.sdf', 'prop': 'Tox'},
        {'path': 'C:\PycharmProjects\ml-data-qsar\TEST\LC50DM\LC50DM_training.sdf', 'prop': 'Tox'},
        {'path': 'C:\PycharmProjects\ml-data-qsar\TEST\LD50\LD50_training.sdf', 'prop': 'Tox'},
        {'path': 'C:\PycharmProjects\ml-data-qsar\TEST\MP\MP_training.sdf', 'prop': 'Tox'},
        {'path': 'C:\PycharmProjects\ml-data-qsar\TEST\ST\ST_training.sdf', 'prop': 'Tox'},
        {'path': 'C:\PycharmProjects\ml-data-qsar\TEST\TC\TC_training.sdf', 'prop': 'Tox'},
        {'path': 'C:\PycharmProjects\ml-data-qsar\TEST\Viscosity\Viscosity_training.sdf', 'prop': 'Tox'},
        {'path': 'C:\PycharmProjects\ml-data-qsar\TEST\VP\VP_training.sdf', 'prop': 'Tox'},
        {'path': 'C:\PycharmProjects\ml-data-qsar\TEST\WS\WS_training.sdf', 'prop': 'Tox'},
    ]

    # TODO rename dict, shadows builtin type dict
    for dict in sets_list:
        try:
            set_path = dict['path'].replace(
                '_training.sdf', ' test set predictions.txt'
            )
            print(get_test_metrics(set_path))
        except:
            print('-9999')


if __name__ == '__main__':
    main()
