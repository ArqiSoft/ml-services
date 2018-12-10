from random import shuffle

import pandas as pd

df = pd.read_csv('all_smiles.smi')

df = df['smiles'].values
shuffle(df)
df = pd.DataFrame(df)
df.to_csv('all_smiles.smi',index=False)