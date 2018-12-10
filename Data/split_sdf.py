
from sklearn import model_selection


def extract_sample_mols(input_file, n = 1000, valuename = ''):
    counter = 1
    values_list = []
    mol_numbers = []

    with open(input_file, "r") as infile:
        for line in infile:
            if valuename in line:
                values_list.append(next(infile, '').strip())
                
            if line[:4] == '$$$$':
                mol_numbers.append(counter)
                counter +=1
    
    print(len(mol_numbers), len(values_list))
        
    x_train, x_test, y_train, y_test = model_selection.train_test_split(
        mol_numbers,
        values_list,
        test_size=n,
        random_state=42,
        shuffle=True)

    valid_list = x_test

    return valid_list


def write_sample_sdf(input_file, valid_list):

    sample_file=open(str(input_file[:-4]) + '_sample.sdf', 'w')
    mol = []
    i=0

    for line in open(input_file):
        mol.append(line)
        if line[:4] == '$$$$':
            i+=1
            if i in valid_list:
                for mol_line in mol:
                    sample_file.write(mol_line)
                valid_list.remove(i)
                mol = []
            else:
                mol = []
    sample_file.close()

    return


def extract_sample_dataset(input_file, n, valuename):
    valuename = '<' + valuename + '>'
    valid_list = extract_sample_mols(input_file=input_file, n = n, valuename = valuename)
    write_sample_sdf(input_file=input_file, valid_list=valid_list)
    return


input_file = '../test_in/OPERA/QSAR_ready_Curated_3_4STAR_LogP_train.sdf'
extract_sample_dataset(input_file=input_file, n=1000, valuename='Kow')