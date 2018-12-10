import ntpath

import numpy as np
import pandas as pd
import scipy.stats as st
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors


def get_dataset(infile):
    return [x for x in Chem.SDMolSupplier(infile) if x is not None]


def binary_classes_count(dataset,classname):
    true_counter = 0
    false_counter = 0
    for mol in dataset:
        if mol.GetProp(classname).upper() == 'TRUE' or mol.GetProp(classname) == '1':
            true_counter += 1
        elif mol.GetProp(classname).upper() == 'FALSE' or mol.GetProp(classname) == '0':
            false_counter += 1
    return true_counter,false_counter


def fcfp_1024_on_bits_average(dataset):
    num_on_bits = 0
    for mol in dataset:
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, 3, 1024)
        num_on_bits += fp.GetNumOnBits()
    return num_on_bits/len(dataset)


def average_number_of_atoms(dataset):
    num = 0
    for mol in dataset:
        num += mol.GetNumAtoms()
    return num/len(dataset)


def average_heavy_atom(dataset):
    num = []
    for mol in dataset:
        num.append(mol.GetNumHeavyAtoms())
    mean = np.mean(num)
    conf = st.t.interval(0.95, len(num)-1, loc=mean, scale=st.sem(num))
    return {'mean': mean, 'conf': conf}


def average_flexibility(dataset):
    num = []
    for mol in dataset:
        num.append(Descriptors.NumRotatableBonds(mol))
    mean = np.mean(num)
    conf = st.t.interval(0.95, len(num)-1, loc=mean, scale=st.sem(num))
    return {'mean': mean, 'conf': conf}


def amide_bonds(dataset):
    num = []
    for mol in dataset:
        num.append(rdMolDescriptors.CalcNumAmideBonds(mol))
    mean = np.mean(num)
    conf = st.t.interval(0.95, len(num)-1, loc=mean, scale=st.sem(num))
    return {'mean': mean, 'conf' : conf}


def get_min_max(dataset,valuename):
    values = []
    for mol in dataset:
        try:
            values.append(float(mol.GetProp(valuename)))
        except ValueError:
            pass
    return min(values),max(values)


def datasets_to_dataframes(paths_props_dicts,regression=False):
    data = []
    for path_prop in paths_props_dicts:
        row = []
        row.append(ntpath.basename(path_prop['path']))
        dataset = get_dataset(path_prop['path'])
        row.append(len(dataset))
        row.append(average_heavy_atom(dataset)['mean'])
        row.append(average_heavy_atom(dataset)['conf'])
        row.append(average_flexibility(dataset)['mean'])
        row.append(average_flexibility(dataset)['conf'])
        row.append(amide_bonds(dataset)['mean'])
        row.append(amide_bonds(dataset)['conf'])
        row.append(fcfp_1024_on_bits_average(dataset))
        if not regression:
            row.append(binary_classes_count(dataset,path_prop['prop']))
        else:
            row.append(get_min_max(dataset,path_prop['prop']))
        data.append(row)
    headers = ['Dataset','Entities','Average_Heavy','Confidence_int','Number_of_rot_bonds','Confidence_int','Number_of_amide_bonds','Confidence_int','BinsON_1024','1/0 or Min/Max']

    return pd.DataFrame(data,columns=headers)


#print(datasets_to_dataframes([{'path': 'C:\putty\QSAR_ready_Curated_3_4STAR_LogP_size_20.sdf', 'prop': 'Kow'}],regression=True))

# dataframe = datasets_to_dataframes([{'path':'C:\PycharmProjects\ml-data-qsar\TEST\BCF\BCF_training.sdf', 'prop':'Tox'},
#                             {'path': 'C:\PycharmProjects\ml-data-qsar\TEST\BP\BP_training.sdf', 'prop': 'Tox'},
#                             {'path': 'C:\PycharmProjects\ml-data-qsar\TEST\Density\Density_training.sdf', 'prop': 'Tox'},
#                             {'path': 'C:\PycharmProjects\ml-data-qsar\TEST\FP\FP_training.sdf','prop': 'Tox'},
#                             {'path': 'C:\PycharmProjects\ml-data-qsar\TEST\ER_LogRBA\ER_LogRBA_training.sdf','prop': 'Tox'},
#                             {'path': 'C:\PycharmProjects\ml-data-qsar\TEST\IGC50\IGC50_training.sdf','prop': 'Tox'},
#                             {'path': 'C:\PycharmProjects\ml-data-qsar\TEST\LD50\LD50_training.sdf', 'prop': 'Tox'},
#                             {'path': 'C:\PycharmProjects\ml-data-qsar\TEST\LC50\LC50_training.sdf', 'prop': 'Tox'},
#                             {'path': 'C:\PycharmProjects\ml-data-qsar\TEST\LC50DM\LC50DM_training.sdf', 'prop': 'Tox'},
#                             {'path': 'C:\PycharmProjects\ml-data-qsar\TEST\LD50\LD50_training.sdf', 'prop': 'Tox'},
#                             {'path': 'C:\PycharmProjects\ml-data-qsar\TEST\MP\MP_training.sdf', 'prop': 'Tox'},
#                             {'path': 'C:\PycharmProjects\ml-data-qsar\TEST\ST\ST_training.sdf', 'prop': 'Tox'},
#                             {'path': 'C:\PycharmProjects\ml-data-qsar\TEST\TC\TC_training.sdf', 'prop': 'Tox'},
#                             {'path': 'C:\PycharmProjects\ml-data-qsar\TEST\Viscosity\Viscosity_training.sdf', 'prop': 'Tox'},
#                             {'path': 'C:\PycharmProjects\ml-data-qsar\TEST\VP\VP_training.sdf','prop': 'Tox'},
#                             {'path': 'C:\PycharmProjects\ml-data-qsar\TEST\WS\WS_training.sdf','prop': 'Tox'},],
#                              regression=True)
# dataframe.to_csv('TEST_regression.csv')

#dataframe = datasets_to_dataframes([{'path':'C:\PycharmProjects\ml.services\Data\Eu_data_logK.sdf', 'prop':'logK'}],regression=True)
#print(dataframe)
# print(datasets_to_dataframes([{'path':'D:/ML/Eu_const/Ce_dataset.sdf', 'prop':'LogK'},
# #                              {'path':'C:/Users/mitrofjr/Downloads/Eu_data_test.sdf', 'prop':'logK'},
# #                              {'path': 'C:/putty/QSAR_ready_Curated_3_4STAR_LogP_flex_4.sdf', 'prop': 'Kow'},
# #                              {'path': 'C:/putty/QSAR_ready_Curated_3_4STAR_LogP_flex_5.sdf', 'prop': 'Kow'},
#                                 ], regression=True))
#dataframe.to_csv('TEST_regression.csv')