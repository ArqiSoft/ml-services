import pickle


param = pickle.load(open('params.obj','rb'))

charset = param.charset
char_to_int = param.char_to_int
smiles_len = param.smiles_len
int_to_char = param.int_to_char

pickle.dump(charset,open('charset.obj','wb'))
pickle.dump(char_to_int,open('char_to_int.obj','wb'))
pickle.dump(smiles_len,open('smiles_len.obj','wb'))
pickle.dump(int_to_char,open('int_to_char.obj','wb'))