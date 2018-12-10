

import sys

# Small LSTM Network to Generate Text for Alice in Wonderland
import numpy
from keras.callbacks import ModelCheckpoint
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.models import Sequential
# load ascii text and covert to lowercase
# filename = "smiles_20.smi"
# raw_text = open(filename).read()
# # create mapping of unique chars to integers
# chars = sorted(list(set(raw_text)))
# char_to_int = dict((c, i) for i, c in enumerate(chars))
# int_to_char = dict((i, c) for i, c in enumerate(chars))
# # summarize the loaded data
# n_chars = len(raw_text)
# n_vocab = len(chars)
# print ("Total Characters: ", n_chars)
# print ("Total Vocab: ", n_vocab)
# # prepare the dataset of input to output pairs encoded as integers
# seq_length = 20
# dataX = []
# dataY = []
# for i in range(0, n_chars - seq_length, 20):
#     seq_in = raw_text[i:i + seq_length]
#     seq_out = raw_text[i + seq_length]
#     dataX.append([char_to_int[char] for char in seq_in])
#     dataY.append(char_to_int[seq_out])
# n_patterns = len(dataX)
# print ("Total Patterns: ", n_patterns)
# # reshape X to be [samples, time steps, features]
# X = numpy.reshape(dataX, (n_patterns, seq_length, 1))
# # normalize
# X = X / float(n_vocab)
# # one hot encode the output variable
# y = np_utils.to_categorical(dataY)
# # define the LSTM model
from tflearn.data_utils import string_to_semi_redundant_sequences

string_utf8 = open('Estro_padded_73.smi', "r").read()
X, Y, charset = \
    string_to_semi_redundant_sequences(string_utf8, seq_maxlen=74, redun_step=74)

model = Sequential()
model.add(LSTM(128, input_shape=(X.shape[1], X.shape[2]), return_sequences=True))
model.add(Dropout(0.5))
model.add(LSTM(128))
model.add(Dropout(0.5))
model.add(Dense(Y.shape[1], activation='relu'))
model.compile(loss='categorical_crossentropy', optimizer='Nadam')
# define the checkpoint
filepath="weights-improvement-{epoch:02d}-{loss:.4f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [checkpoint]
# fit the model
model.fit(X, y, epochs=20, batch_size=512, callbacks=callbacks_list)
start = numpy.random.randint(0, len(dataX)-1)
pattern = dataX[start]
print("Seed:")
print("\"", ''.join([int_to_char[value] for value in pattern]), "\"")
# generate characters
for i in range(1000):
    x = numpy.reshape(pattern, (1, len(pattern), 1))
    x = x / float(n_vocab)
    prediction = model.predict(x, verbose=0)
    index = numpy.argmax(prediction)
    result = int_to_char[index]
    seq_in = [int_to_char[value] for value in pattern]
    sys.stdout.write(result)
    print(result)
    pattern.append(index)
    pattern = pattern[1:len(pattern)]
print("\nDone.")