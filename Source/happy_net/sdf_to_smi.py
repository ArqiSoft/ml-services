from rdkit import Chem

from Source.happy_net.SMILES_enumeration import get_mol_set

suppl = Chem.SDMolSupplier('EstrogenCHEMBL206-Binding-K.sdf')
mols = [x for x in suppl if x is not None]


out = open('Estro_active.smi','w')
len_dict = {}

for mol in mols:
    Chem.Kekulize(mol)
    if mol.GetProp('cut_off_activity') == '1':
        try:
            smile = Chem.MolToSmiles(mol,kekuleSmiles=True)
            canonical, smiles_noncan = get_mol_set(smile, tries=100)
        except Exception as e:
            print(e)
            continue
        for noncan in smiles_noncan:
            if len(noncan) in len_dict:
                len_dict[len(noncan)] += 1
            else:
                len_dict[len(noncan)] = 1
            out.write(noncan+'\n')

print(len_dict)
