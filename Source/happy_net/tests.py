from rdkit import Chem

mol = Chem.MolFromSmiles('C12=C(OC(C3=CC=C(OCCN4CCOCC4)C=C3)C3=CC=C(O)C=C3CC2)C=CC(O)=C1')
print(Chem.MolToSmiles(mol))