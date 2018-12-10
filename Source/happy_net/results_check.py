from rdkit import Chem

suppl = Chem.SDMolSupplier('EstrogenCHEMBL206-Binding-K.sdf')
mols = [x for x in suppl if x is not None]
# mols = [Chem.MolFromSmiles(x) for x in open('all_smiles.smi','r').readlines()]
canons = [Chem.MolToSmiles(x) for x in mols if x is not None]

w = Chem.SDWriter('Estro_active_result.sdf')
lines_list = open('Estro_active_result.smi','r').readlines()
for smile in lines_list:
    try:
        molecule = Chem.MolFromSmiles(smile)
        if molecule and (Chem.MolToSmiles(molecule) not in canons):
            w.write(molecule)
    except Exception as e:
        print(e)