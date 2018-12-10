from rdkit import Chem
import pandas as pd
import os
from Data.dataset_analyzer import datasets_to_dataframes
from Source.sds_tools_beta.plotters import plot_reg_target
from scipy.stats import zscore


suppl = Chem.SDMolSupplier('BF3_series_13000_original.sdf')
mols = [x for x in suppl if x is not None]

prop_names = mols[1].GetPropNames()


for property in prop_names:
    prop_val_list = []
    for mol in mols:
        try:
            value = float(mol.GetProp(property))
            prop_val_list.append(value)
        except:
            continue

    dataframe = pd.DataFrame(prop_val_list, columns=[property]).apply(
            pd.to_numeric, errors='coerce'
        ).dropna(
            axis=0, how='any'
        ).reset_index(drop=True)

    plot_reg_target(
        dataframe,'{}'.format(property),
        'C:\PycharmProjects\ml.services\Data\Plots',bins=40
    )

    # df_zscore = (dataframe - dataframe.mean()) / dataframe.std()
    dataframe['z-score'] = (dataframe[property] - dataframe[property].mean()) / dataframe[property].std(ddof=0)
    outliers = dataframe.loc[dataframe['z-score'] >=2.5]
    dataframe = dataframe[dataframe['z-score'] < 2.5][property]
    print(dataframe)

    plot_reg_target(
        dataframe,'z-scored_{}'.format(property),
        'C:\PycharmProjects\ml.services\Data\Plots',bins=40
    )









