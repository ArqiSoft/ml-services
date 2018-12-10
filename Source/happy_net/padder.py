from random import shuffle

import pandas as pd

df = pd.read_csv('Estro_act_1.smi')

mask = (df['smiles'].str.len() > 65)
df = df.loc[mask]
mask = (df['smiles'].str.len() <= 70)
df = df.loc[mask]
df = df['smiles'].values
out = open('Estro_padded_70.smi', 'w')
shuffle(df)

for smile in df:
    smile1 = smile.strip()
    while len(smile1) < 70:
        smile1 = smile1 + "/"
    if len(smile1) == 70:
        smile1 = smile1 + '\n'
        out.write(smile1)


