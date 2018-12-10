from rdkit import Chem
from learner.fingerprints import ecfc_molstring,avalon_molstring,get_desc_data,maccs_molstring,fcfc_molstring
import numpy as np
import pandas as pd

smiles = pd.read_csv('chembl23_uniq.smi_chid',sep=' ')

molstring_list = []
counter = 0
dump_counter = 0

for id,row in smiles.iterrows():
    molecule = Chem.MolFromSmiles(row[0])
    if molecule:
        # descs, _ = get_desc_data(molecule)
        maccs = maccs_molstring(molecule,{'Size': 167})
        ecfc = ecfc_molstring(molecule,{'Radius':3, 'Size': 512})
        fcfc = fcfc_molstring(molecule,{'Radius':3, 'Size': 512})
        avalon = avalon_molstring(molecule,{'Size': 1024})
        molstring = np.concatenate([maccs,ecfc,fcfc,avalon],axis=0)
        molstring_list.append(molstring)
        counter += 1
    else:
        pass
    if counter == 100000:
        print('100K is ready')
        ar = np.vstack(molstring_list)
        np.save('features_{}'.format(dump_counter),ar)
        dump_counter += 1
        counter = 0
        molstring_list = []






