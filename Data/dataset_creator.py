import os
import pandas as pd
import pybel

"""
We use a folder with .mol files named as molecules
and a .csv with properties in columns and molecules
in rows. One of the columns should be "Name" and
contain names of .mol files
"""

path = 'D:/ML/Eu_const'

'''
#search for csv with props
filelist = os.listdir(path=path)
for file in filelist:
    if '.csv' in file:
        csv = file
'''

def create_sdf(path):
    #make df with props
    csv = path + '/index.csv'
    df = pd.read_csv(csv, index_col=0)
    records = df.to_dict(orient='records')

    #make a list of mol files
    filelist = os.listdir(path=path)
    mol_list = []
    for file in filelist:
        if '.MOL' in file.upper():
            mol_list.append(file)

    #create sdf
    dataset = path + '/dataset.sdf'
    for file in mol_list:
        abs_path = path + '/' + file
        with open(abs_path, 'r') as input:
            lines = input.readlines()
            for record in records:
                if str(os.path.basename(abs_path)[:-4]) == record['Name']:
                    for prop in record.keys():
                        lines.append('>  <' + prop + '>\n')
                        lines.append(str(record[prop]) + '\n\n')
            lines.append('$$$$\n')

        with open(dataset, 'a') as output:
            output.writelines(lines)

'''
def threeD2twoD(path):
    filelist = os.listdir(path=path)
    for file in filelist:
        if '.MOL2' in file.upper():
            mol2 = pybel.readfile("mol2", file)
            output = path + '/TwoD/' + file
            mol2.draw(show=False,filename=output)
'''


#threeD2twoD(path=path)
#print(pybel.informats())
create_sdf(path=path)
