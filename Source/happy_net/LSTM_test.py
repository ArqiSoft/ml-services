import tflearn
from tflearn.data_utils import *

# from SMILES_enumeration import get_mol_set
#
# smiles_super_line = open('smile_line.csv','w')
# smile_lines = open('all_smiles.smi','r').readlines()
# for smile in smile_lines:
#     try:
#         canonical, smiles_noncan = get_mol_set(smile, tries=50)
#     except Exception as e:
#         print(e)
#         continue
#     for smile_noncan in smiles_noncan:
#             smiles_super_line.write(smile_noncan+'\n')
# print(1)
# smiles_super_line = open('smile_strat.csv','w')
# smile_lines = open('smile_line.csv','r').readlines()
# shuffle(smile_lines)
# smiles_lines = smile_lines[:5000]
# print(1)
# for smile in smiles_lines:
#     smile1 = smile.strip()
#     # try:
#     #     canonical, smiles_noncan = get_mol_set(smile, tries=30)
#     # except Exception as e:
#     #     print(e)
#     #    continue
#     molecule = Chem.MolFromSmiles(smile1)
#     Chem.Kekulize(molecule)
#     smile1 = Chem.MolToSmiles(molecule,kekuleSmiles=True)
#     smiles_super_line.write(smile1+'\n')

# smiles_super_line = open('Estro_padded.smi','w')
# smile_lines = open('Estro.smi','r').readlines()
# shuffle(smile_lines)
# for smile in smile_lines:
#     smile1 = smile.strip()
#     if len(smile1) == 53:
#         smiles_super_line.write(smile1 + '\n')


string_utf8 = open('Estro_padded_70.smi', "r").read()
X, Y, charset = \
    string_to_semi_redundant_sequences(string_utf8, seq_maxlen=70, redun_step=70)
print(charset)
print(Y)
print(X)
g = tflearn.input_data(shape=[None, 70, len(charset)])
g = tflearn.lstm(g, 256, return_seq=True)
g = tflearn.dropout(g,0.5)
g = tflearn.lstm(g, 256)
g = tflearn.dropout(g,0.5)
g = tflearn.fully_connected(g, len(charset), activation='softmax')
g = tflearn.regression(g, optimizer='adam', loss='categorical_crossentropy',
                       learning_rate=0.001)

m = tflearn.SequenceGenerator(g, dictionary=charset,
                              seq_maxlen=70,
                              clip_gradients=5.0,
                              checkpoint_path='model_us_cities')




w_19 = open('results_1.9.txt','w')
w_15 = open('results_1.5.txt','w')
w_13 = open('results_1.3.txt','w')
w_12 = open('results_1.2.txt','w')
w_10 = open('results_1.0.txt','w')


m.fit(X, Y, validation_set=0.1, batch_size=1024,
          n_epoch=25, run_id='us_cities')


for i in range(100):
    seed = string_utf8[70*i: 70 +70*i]
    print("-- TESTING...")
    w_19.write(m.generate(70, temperature=1.9, seq_seed=seed))
    w_15.write(m.generate(70, temperature=1.5, seq_seed=seed))
    w_13.write(m.generate(70, temperature=1.3, seq_seed=seed))
    w_12.write(m.generate(70, temperature=1.2, seq_seed=seed))
    w_10.write(m.generate(70, temperature=1.0, seq_seed=seed))





