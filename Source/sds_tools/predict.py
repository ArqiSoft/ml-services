import os
from operator import itemgetter
from os import listdir

import numpy as np
import pandas as pd
from keras.models import load_model

from general_helper import coeff_determination
from processor import sdf_to_csv
from rdkit import Chem
from sklearn.externals import joblib

import sklearn
print(sklearn.__version__)



suppl = Chem.SDMolSupplier(
    'C:\PycharmProjects\ml-data-qsar\TEST\LC50\LC50_training.sdf')
molecules = [x for x in suppl if x is not None]
molecules = molecules

fptype = [{'Type': 'DESC'},
          {'Type': 'MACCS'},
          {'Type': 'FCFC','Size': 512,'Radius':3},
          {'Type': 'AVALON','Size': 512}]
dataframe = sdf_to_csv('LC50_prediction', fptype=fptype, molecules=molecules)


folder_path = 'C:\PycharmProjects\ml-models\\UBC\Half_LIfe_U_2018_03_18__14_24_16_DESC_MACCS_FCFC_512_3_AVALON_512_scaled___'
models_paths = [os.path.join(folder_path, x) for x in listdir(folder_path) if x.split('.')[-1] == 'h5']
transformers = [os.path.join(folder_path, x) for x in listdir(folder_path) if x.split('.')[-1] == 'sav']


predicted_test_y_vectors = []
df_predict_clf = pd.DataFrame()
for transformer in transformers:
    trans = joblib.load(transformer)

for path_to_model in models_paths:
    model_base = load_model(
        path_to_model,
        custom_objects={'coeff_determination': coeff_determination}
    )
    test_predict_tmp = model_base.predict(dataframe)
    print(test_predict_tmp)
    predicted_test_y_vectors.append(test_predict_tmp)
    print('Loading of model complete')


mean_predicted = np.mean(predicted_test_y_vectors, axis=0)
predicted_mols = itemgetter(*vectorized)(molecules)

df_predict_clf['Compound_SMILES'] = [
        Chem.MolToSmiles(mol, isomericSmiles=True) for mol in predicted_mols
        ]
# df_predict_clf['ID'] = [
#         mol.GetProp('index') for mol in predicted_mols
#         ]
df_predict_clf['value'] = mean_predicted
df_predict_clf.to_csv('predicted_LC50_DNN_SSP_test.csv')
