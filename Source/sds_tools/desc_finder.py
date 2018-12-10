import pandas as pd
from rdkit import Chem


def get_molstring_from_library(molecule,library,value,index='VPC ID'):
    val = molecule.GetProp(value)
    try:
        val = float(val)
        vpc_id = molecule.GetProp(index)
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

