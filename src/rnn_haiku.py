import numpy as np
import sys
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils
import re

# load ascii text and covert to lowercase
filename = "data/haiku.txt"
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
seq_length = 15
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

filename = "rnn models/weights-improvement-98-0.2921-haiku.hdf5"
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
	#seeds = [x for x in dataX if x[0] == char_to_int[' ']]

	for diversity in 0.1 * np.ones(10):
		start = np.random.randint(0, len(dataX)-1)
		print()
		print('----- diversity:', diversity)

		generated = ''
		sentence = dataX[start]
		generated +=''.join([int_to_char[value] for value in sentence])
		print('----- Generating with seed: "' + ''.join([int_to_char[value] for value in sentence]) + '"')
		sys.stdout.write(generated)

		tot_lines = 0

		while True:
			if tot_lines > 2:
				break
			x = np.reshape(sentence[0:seq_length], (1, seq_length, 1))
			# normalize
			x = x / float(n_vocab)
			preds = model.predict(x, verbose=0)[0]
			next_index = sample(preds, diversity)
			next_char = int_to_char[next_index]
			generated += next_char
			if next_char == '\t' or next_char == '\n':
				tot_lines += 1
			sentence.append(next_index)
			sentence = sentence[1:len(sentence)]
			# for the format of the volta
			sys.stdout.write(next_char)
			sys.stdout.flush()
		print()

generate_from_model(model)
