from rdkit import Chem
from rdkit.Chem import Descriptors

path = 'C:/putty/QSAR_ready_Curated_3_4STAR_AOP.sdf'
#path = 'D:/ML/Eu_const/EbindH_mod_2.sdf'
tag = 'OH'

def Split_by_btag(path, tag):
    '''
    Split dataset into two parts by binary (1/0 or True/False)
    tag.
    :param path: path to initial dataset in .sdf format
    :param tag: flag to split
    :return: percent of non-readable molecules
    '''
    out_train = Chem.SDWriter(path[:-4] + '_train.sdf')
    out_test = Chem.SDWriter(path[:-4] + '_test.sdf')

    suppl = Chem.SDMolSupplier(path)
    counter = 0
    fail_counter = 0
    for mol in suppl:
        if mol is None:
            fail_counter += 1
            continue
        if mol.GetProp(tag) == '1' or mol.GetProp(tag).upper() == 'TRUE':
            out_train.write(mol)
            counter += 1
        if mol.GetProp(tag) == '0' or mol.GetProp(tag).upper() == 'FALSE':
            out_test.write(mol)
            counter += 1
    correct_data = fail_counter/(fail_counter+counter)*100
    print(correct_data)
    return correct_data


def split_by_def_mol_size(path, size):
    '''
    Cut a part of dataset containing molecules with definite size +-1 heavy atom
    Returns percent of non-readable molecules
    :param path: path to initial dataset in .sdf format
    :param size: size in non-hydrogen atoms
    :return: percent of non-readable molecules
    '''
    out = Chem.SDWriter(path[:-4] + '_size_' + str(size) + '.sdf')
    suppl = Chem.SDMolSupplier(path)
    counter = 0
    fail_counter = 0
    for mol in suppl:
        if mol is None:
            continue
            fail_counter += 1
        if size-1 < mol.GetNumHeavyAtoms() <= size+1:
            out.write(mol)
            counter += 1
    correct_data = fail_counter/(fail_counter+counter)*100
    print(correct_data)
    return correct_data

def split_by_def_flex(path, flex):
    '''
    Cut a part of dataset containing molecules with definite size +-1 heavy atom
    Returns percent of non-readable molecules
    :param path: path to initial dataset in .sdf format
    :param size: size in non-hydrogen atoms
    :return: percent of non-readable molecules
    '''
    out = Chem.SDWriter(path[:-4] + '_flex_' + str(flex) + '.sdf')
    suppl = Chem.SDMolSupplier(path)
    counter = 0
    fail_counter = 0
    for mol in suppl:
        if mol is None:
            continue
            fail_counter += 1
        if flex - 0.5 < Descriptors.NumRotatableBonds(mol) <= flex + 0.5:
            out.write(mol)
            counter += 1
    correct_data = fail_counter / (fail_counter + counter) * 100
    print(correct_data)
    return correct_data

def split_by_atom(path, atom):
    '''
    Cut a part of dataset containing molecules with definite size +-1 heavy atom
    Returns percent of non-readable molecules
    :param path: path to initial dataset in .sdf format
    :param size: size in non-hydrogen atoms
    :return: percent of non-readable molecules
    '''
    out = Chem.SDWriter(path[:-4] + '_with_' + str(atom) + '.sdf')
    suppl = Chem.SDMolSupplier(path)
    for mol in suppl:
        if mol is None:
            continue
        else:
            for a in mol.GetAtoms():
                num = a.GetAtomicNum()
                if atom == num:
                    out.write(mol)
    return

def Split_by_tag(path, tag):
    '''
    Split dataset into two parts by binary (1/0 or True/False)
    tag.
    :param path: path to initial dataset in .sdf format
    :param tag: flag to split
    :return: percent of non-readable molecules
    '''
    out_mod = Chem.SDWriter(path[:-4] + '_mod.sdf')

    suppl = Chem.SDMolSupplier(path)
    counter = 0
    fail_counter = 0
    for mol in suppl:
        if mol is None:
            continue
        if mol.GetProp(tag) > 0:
            out_mod.write(mol)
            counter += 1
#    correct_data = fail_counter/(fail_counter+counter)*100
#    print(correct_data)
    return

#Split_by_tag(path=path, tag=tag)

'''
suppl = Chem.SDMolSupplier('D:/ML/Eu_const/Ligandy/Eu_data_logK.sdf')
for mol in suppl:
    if mol is not None:
        print(mol)
'''

def cut_first_n(path, n = 100):
    out_mod = Chem.SDWriter(path[:-4] + '_mod.sdf')
    suppl = Chem.SDMolSupplier(path)
    counter = 0
    fail_counter = 0
    for mol in suppl:
        if mol is None:
            fail_counter += 1
            continue
        elif counter < 100:
            counter += 1
            out_mod.write(mol)
    return (counter, fail_counter)

cut_first_n(path, 10)

def split_to_parts(path, n = 10):
    part = 1
    suppl = Chem.SDMolSupplier(path)
    out_mod = Chem.SDWriter(path[:-4] + '_mod_' + part + '.sdf')
    counter = 0
    fail_counter = 0
    for mol in suppl:
        if mol is None:
            fail_counter += 1
            continue
        elif counter < n:
            counter += 1
            out_mod.write(mol)
        elif counter == n:
            counter = 0
            part += 1
    return fail_counter

