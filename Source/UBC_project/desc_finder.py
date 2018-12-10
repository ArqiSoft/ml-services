from os import listdir
from os.path import isfile, join

import pandas as pd
from rdkit import Chem


def get_molstring_from_library(molecule,library,value,index='VPC ID'):
    val = molecule.GetProp(value)
    try:
        val = float(val)
        vpc_id = molecule.GetProp(index)
        print(vpc_id)
        row = library.loc[library[index] == int(vpc_id)].drop(index, axis=1)
        row[value] = float(val)

        return row
    except ValueError:
        pass


def get_dataframe_from_library(file_path,library,value,index='VPC ID'):
    suppl = Chem.ForwardSDMolSupplier(file_path)
    mols = [x for x in suppl if x is not None]
    rows_list = []
    for molecule in mols:
        rows_list.append(get_molstring_from_library(molecule,library,value,index=index))

    return pd.concat(rows_list,axis=0)


def get_feature_vector_from_library(molecule,library,index='VPC ID'):
    vpc_id = molecule.GetProp('VPC ID')
    row = library.loc[library['VPCID'] == int(vpc_id)].drop('VPCID', axis=1)
    return row


def get_dataframe_from_multiple_libraries(file_path,value,library_folder,index='VPC ID'):
    suppl = Chem.ForwardSDMolSupplier(file_path)
    mols = [x for x in suppl if x is not None]
    rows_list = []
    counter = 0
    libs = [join(library_folder, lib) for lib in listdir(library_folder) if isfile(join(library_folder, lib))]
    for molecule in mols:
        val = molecule.GetProp(value)
        try:
            val = float(val)
        except ValueError:
            continue
        counter += 1
        print(counter)
        concat_list = []
        for lib in libs:
            concat_list.append(get_feature_vector_from_library(molecule,pd.read_csv(lib),index=index))
        conc_row = pd.concat(concat_list,axis=1)
        conc_row[value] = val
        rows_list.append(conc_row)

    return pd.concat(rows_list,axis=0)
