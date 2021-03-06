'''
A stateful LSTM for text generation
Modification of https://github.com/fchollet/keras/blob/master/examples/lstm_text_generation.py
Example script to generate text from Nietzsche's writings.
At least 20 epochs are required before the generated text
starts sounding coherent.
It is recommended to run this script on GPU, as recurrent
networks are quite computationally intensive.
If you try this script on new data, make sure your corpus
has at least ~100k characters. ~1M is better.
'''

from __future__ import print_function

import random

import numpy as np
from keras.layers import Dense, Activation
from keras.layers import LSTM
from keras.models import Sequential
from keras.optimizers import RMSprop
from keras.utils import generic_utils

# Beginning of Sequence
BOS = '$'
# End Of SEQuence
EOSEQ = "$$$"

BATCH_SIZE = 1024


# Generators return EOS to indicate end of generation
def sentence_generator(lines):
    random.shuffle(lines)
    for line in lines:
        if len(line) > 5:
            yield line
    yield EOSEQ


def char_pair_generator(sentence):
    context_char = BOS
    for char in sentence:
        yield context_char, char
        context_char = char
    yield EOSEQ, EOSEQ


def get_char_dict(lines, max_chars=999999):
    assert (max_chars > 2)
    char_set = set([BOS])
    sentences = 0
    for line in sentence_generator(lines):
        char_set.update(line)
        sentences += 1
    char_indices = dict((c, i) for i, c in enumerate(char_set))
    indices_char = dict((i, c) for i, c in enumerate(char_set))
    return char_indices, indices_char, sentences

output = open("statefull_out.smi",'w')
lines = open("all_smiles.smi", 'r').readlines()

char_indices, indices_char, training_sentences = get_char_dict(lines)

print('total chars:', len(char_indices))
print('total sentences:', training_sentences)

# Train arrays
X = np.zeros((BATCH_SIZE, 1, len(char_indices)), dtype=np.bool)
Y = np.zeros((BATCH_SIZE, len(char_indices)), dtype=np.bool)


# build the model: a single LSTM
print('Build model...')
model = Sequential()
model.add(LSTM(128, batch_input_shape=(BATCH_SIZE, 1, len(char_indices)), stateful=True))
model.add(Dense(len(char_indices)))
model.add(Activation('softmax'))

optimizer = RMSprop(lr=0.01)
model.compile(loss='categorical_crossentropy', optimizer=optimizer)


def sample(preds, temperature=1.0):
    # helper function to sample an index from a probability array
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)

# train the model, output generated text after each iteration
for iteration in range(1, 40):
    print()
    print('-' * 50)
    print('Iteration', iteration)
    processed_sentences = 0
    progbar = generic_utils.Progbar(training_sentences)

    # TRAINING
    sentence_gen = sentence_generator(lines)

    char_pair_generators = []
    for i in range(BATCH_SIZE):
        # possible bug when having less then batch size sentences (neglected)
        char_pair_generators.append(char_pair_generator(next(sentence_gen)))
        processed_sentences += 1

    finished_iteration = False
    losses = []

    while not finished_iteration:
        for i in range(BATCH_SIZE):
            context_char, char = next(char_pair_generators[i])

            # refill char pair generator
            if context_char == EOSEQ:
                sentence = next(sentence_gen)
                # the iteration finished, no new sentence can be retrieved
                if sentence == EOSEQ:
                    finished_iteration = True
                    break
                char_pair_generators[i] = char_pair_generator(sentence)
                context_char, char = next(char_pair_generators[i])
                processed_sentences += 1

                if processed_sentences % 100 == 0:
                    progbar.update(processed_sentences, values=[("loss", np.mean(losses))])
                    losses = []
            X[i, 0, char_indices[context_char]] = 1
            Y[i, char_indices[char]] = 1

        loss = model.train_on_batch(X, Y)
        losses.append(loss)
        X.fill(0)
        Y.fill(0)

    # SAMPLING
    diversities = [0.2, 0.5, 1.0, 1.2]
    generated = len(diversities) * [BOS]

    for i in range(4000):
        X.fill(0)
        for k in range(len(diversities)):
            X[k, 0, char_indices[generated[k][-1]]] = 1.

        preds = model.predict(X, batch_size=BATCH_SIZE, verbose=0)

        for k in range(len(diversities)):
            next_index = sample(preds[k], diversities[k])
            char = indices_char[next_index]
            generated[k] += char

    for k in range(len(diversities)):
        print()
        print("diversity:", diversities[k])
        print(generated[k][1:])
        output.write(generated[k][1:])
        print()