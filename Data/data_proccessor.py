from rdkit import Chem
import pandas as pd
import os
import re


sdf = Chem.SDWriter('UBC_dataset_latest.sdf')
dataframe = pd.read_csv('C:\PycharmProjects\ml.services\Data\BF3_series_13000_original_Ushaper_flggedBy1_Eric.csv')
headers = [e for e in list(dataframe) if e not in ('Structure', 'SMILES','Mol Weight')]
counter = 0
for index,record in dataframe.iterrows():
    molecule = Chem.MolFromSmiles(str(record['SMILES']))
    if molecule is not None:
        for property in headers:
            if str(record[property]) == 'nan':
                molecule.SetProp(property, "No Data")
            else:
                molecule.SetProp(property, re.sub('[",]', '', str(record[property])))
        sdf.write(molecule)
        # if not os.path.isfile('pKaInWater_7912_{}.sdf'.format(str(t))):
        #     opened_dict['{}'.format(str(t))] = Chem.SDWriter('pKaInWater_7912_{}.sdf'.format(str(t)))
        # opened_dict[str(t)].write(molecule)
        # counter += 1
print(counter)

