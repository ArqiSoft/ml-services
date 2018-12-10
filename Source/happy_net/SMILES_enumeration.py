#This is a temporary bugfixed version of rdkit.Chem.Randomize.RandomizeMolBlock
#https://github.com/rdkit/rdkit/pull/1376
# This is a temporary bugfixed version of rdkit.Chem.Randomize.RandomizeMolBlock
# https://github.com/rdkit/rdkit/pull/1376
from randomizemolblock import RandomizeMolBlock
from rdkit import Chem


def RandomizeMol(mol):
	"""Function that randomizes an RDKit mol by round tripping a molblock""" 
	mb = Chem.MolToMolBlock(mol)
	mb = RandomizeMolBlock(mb) 
	return Chem.MolFromMolBlock(mb) 


def randomize_smile(smile):
	"""randomize a SMILES"""
	mol = RandomizeMol(Chem.MolFromSmiles(smile))
	Chem.Kekulize(mol)
	return Chem.MolToSmiles(mol, canonical=False,kekuleSmiles=True)


def get_mol_set(smile, tries=10000, split=True):
	"""Make a set of unique randomized SMILES"""
	s = set()
	canonical = Chem.MolToSmiles(Chem.MolFromSmiles(smile))
	s.add(canonical)
	for i in range(tries):
		s.add(randomize_smile(smile))
	print ("total %s found"%(len(s)))
	if split:
		s.remove(canonical)
		return canonical, s
	else:
		return s
			
if __name__ == "__main__":
	canonical, s = get_mol_set('Cc1ccccc1', tries=1000)
	canonical, s = get_mol_set('c1cccc(O)c1C(=O)O', tries=1000)
	canonical, s = get_mol_set('C[C@@H](C(=O)O)N', tries=1000)



	

	
	
