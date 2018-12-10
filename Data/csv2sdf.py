from rdkit import Chem
import pandas as pd

in_path = 'Chebi_Eu_predicted.csv'
out_path = 'Chebi_Eu_predicted_filtered_best.sdf'
prop_name = 'LogK'

def csv2sdf(in_path,out_path,prop_name):
    df = pd.read_csv(in_path, index_col=0)
    df.sort_values(by=['value'], ascending=True)
#    clust_1 = 0
#    clust_2 = 0
    suppl = []
    for i in range(df.shape[0]):
        inp = df.iloc[i]
        mol = Chem.MolFromSmiles(str(inp[0]))
        if mol is not None:
            nh = mol.GetNumHeavyAtoms()
            atoms = mol.GetAtoms()
            print(atoms)
            '''
            if 10 < nh < 60:
                mol.SetProp(prop_name,str(inp[1]))
                #if 8 < inp[1] < 10:
                #    mol.SetProp('cluster', '1')
                #    suppl.append(mol)
                #    clust_1 += 1
                if 20 < inp[1]:
                #    mol.SetProp('cluster', '2')
                    suppl.append(mol)
                #    clust_2 += 1
            '''
    w = Chem.SDWriter(out_path)
    for m in suppl: w.write(m)
    #print(clust_1, clust_2)
    return

csv2sdf(in_path=in_path,out_path=out_path,prop_name=prop_name)