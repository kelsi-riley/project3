# “shall i compare thee to a summer’s
# day?\n”,

import numpy as np
import sys
import random
import re
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils


# load ascii text and covert to lowercase
filename = "data/shakespeare.txt"
raw_text = open(filename).read()
raw_text = raw_text.lower()

# remove special characters
raw_text = re.sub('[!@#$.,:;?()-0123456789]', '', raw_text)
# create mapping of unique chars to integers
chars = sorted(list(set(raw_text)))
print(chars)
#create dictionaries
char_to_int = dict((c, i) for i, c in enumerate(chars))
int_to_char = dict((i, c) for i, c in enumerate(chars))
# summarize the loaded data
n_chars = len(raw_text)
n_vocab = len(chars)

# prepare the dataset of input to output pairs encoded as integers
seq_length = 40
dataX = []
dataY = []
for i in range(0, n_chars - seq_length, 1):
	seq_in = raw_text[i:i + seq_length]
	seq_out = raw_text[i + seq_length]
	dataX.append([char_to_int[char] for char in seq_in])
	dataY.append(char_to_int[seq_out])
n_patterns = len(dataX)
print ("Total Patterns: ", n_patterns)
# reshape X to be [samples, time steps, features]
X = np.reshape(dataX, (n_patterns, seq_length, 1))
# normalize
X = X / float(n_vocab)
# one hot encode the output variable
y = np_utils.to_categorical(dataY)
# define the LSTM model
model = Sequential()
model.add(LSTM(200, input_shape=(X.shape[1], X.shape[2]), return_sequences=True))
model.add(LSTM(200))
model.add(Dense(y.shape[1], activation='softmax'))
# load the network weights
filename = "weights-improvement-16-0.4146-bigger.hdf5"
model.load_weights(filename)
model.compile(loss='categorical_crossentropy', optimizer='adam')
# pick a random seed

# helper function to sample an index from a probability array
def sample(a, temperature=1.0):
	a = np.log(a) / temperature
	dist = np.exp(a)/np.sum(np.exp(a))
	choices = range(len(a))
	return np.random.choice(choices, p=dist)

# train the model, output generated text after each iteration

def generate_from_model(model):
	#start = np.random.randint(0, len(dataX)-1)

	# diversity in [0.2, 0.5, 1.0, 1.2]:
    for diversity in [1.5, 0.75, 0.25]:
        start = np.random.randint(0, len(dataX)-1)
        print()
        print('----- diversity:', diversity)

        seed = '  shall i compare thee to a summers day\n'
        sentence = []
        for c in seed:
            sentence.append(char_to_int[c])
        generated = seed
        print('----- Generating with seed: "' + generated + '"')
        sys.stdout.write(generated)
        tot_lines = 1

        while True:
            if tot_lines > 14:
                break
            x = np.reshape(sentence[0:40], (1, 40, 1))
            # normalize
            x = x / float(n_vocab)
            preds = model.predict(x, verbose=0)[0]
            next_index = sample(preds, diversity)
            next_char = int_to_char[next_index]
            # to avoid empty lines in sonnet
            if next_char == '\n' and not generated[-1] == '\n':
                tot_lines += 1
            generated += next_char
            sentence.append(next_index)
            sentence = sentence[1:len(sentence)]
            # for the format of the volta
            sys.stdout.write(next_char)
            sys.stdout.flush()
            # if we have two new lines characters, we keep generating more characters
        print()

generate_from_model(model)
