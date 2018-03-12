from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.models import model_from_json
from keras.utils import np_utils
import sys
import re

import tensorflow as tf
import numpy as np
import random
import json

def sample(a, temperature=1.0):
	a = np.log(a) / temperature
	dist = np.exp(a)/np.sum(np.exp(a))
	choices = range(len(a))
	return np.random.choice(choices, p=dist)


def build_model(window, len_chars):
    model = Sequential()
    model.add(LSTM(512, return_sequences=True, input_shape=(window, len_chars)))
    model.add(Dropout(0.2))
    model.add(LSTM(512, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(len_chars))
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

    return model

rhyme = json.load(open('json/rhyme.json'))
inverted_rhyme = json.load( \
    open('json/inverted_rhyme.json'))

def end_next_rhyme(prev_rhyme):
    ending = inverted_rhyme[prev_rhyme][0]
    rhymes = rhyme[ending]
    return np.random.choice(rhymes)

def gen_ends(generated):
    end_words = [''] * 14
    for i in range(0, 12, 4):
        s = np.random.choice([line[-25:] for line in lines])
        end_words[i] = s.split(' ')[-1]
        s = np.random.choice([line[-25:] for line in lines])
        end_words[i + 1] = s.split(' ')[-1]

    #we need five seed words to rhyme with
    end_words[-1] = generated.split(' ')[-1]
    end_words[-2] = end_next_rhyme(end_words[-1])

    print(end_words)
    for i in range(14):
        if end_words[i] == '':
            end_words[i] = end_next_rhyme(end_words[i-2])
    return end_words


if __name__=='__main__':
    text = ''
    files = ['data/shakespeare.txt', 'data/more_shakespeare.txt', 'data/spenser.txt']
    for filename in files:
        with open(filename) as f:
            for line in f:
                line = line.strip()
                if len(line) > 0 and not line.isdigit():
                    text += '$' + line.lower() + '\n'
    text = re.sub('[:;,.!()?&]', '', text)
    # create mapping of unique chars to integers
    chars = sorted(list(set(text)))
    print(chars)

    char_to_int = dict((c, i) for i, c in enumerate(chars))
    int_to_char = dict((i, c) for i, c in enumerate(chars))

    maxlen = 25 # Window size

    # Train in reverse, so we can construct lines from the back for rhyme
    step = 3
    sentences = []
    next_chars = []
    for i in range(0, len(text) - maxlen, step):
        sentences.append(text[i + maxlen: i: -1])
        next_chars.append(text[i])

    model = build_model(maxlen, len(chars))

    model.load_weights("backwards_char_rnn.h5")
    model.compile(loss='categorical_crossentropy', optimizer='adam')


    # Ending sequence
    lines = text.split("\n")
    generated = np.random.choice([line[-25:] for line in lines])
    print('----- Generating with end: %s' % generated)
    end_words = gen_ends(generated)

    sentence = ''
    diversity = 0.2
    sonnet = ''
    for i in range(13, -1, -1):
        line = generated
        sentence = (sentence + '$\n' + generated[::-1])[-25:]
        syls = len(line.split(' '))
        while True:
            x = np.zeros((1, maxlen, len(chars)))
            for t, char in enumerate(sentence):
                x[0, t, char_to_int[char]] = 1.

            preds = model.predict(x, verbose=0)[0]
            next_index = sample(preds, diversity)
            next_char = int_to_char[next_index]

            # Ignore special characters
            if (next_char == '\n') or (next_char == '$'):
                sentence = sentence[1:] + next_char
                continue

            # Check syllables
            if (next_char == ' '):
                syls +=1
                if syls >= 10:
                    break

            line = next_char + line
            sentence = sentence[1:] + next_char

        if ((i + 1) % 4 == 0) or (i == 13):
            line += '.\n'
        else:
            line += ',\n'
        if (i == 12 or i == 13):
            line = '\t' + line.capitalize()
            sonnet = line + sonnet
        else:
            sonnet = line.capitalize() + sonnet

        if i != 0:
            generated = ' ' + end_words[i - 1]

    print(sonnet)
