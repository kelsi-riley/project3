import nltk
nltk.download('cmudict')
import string
import json

from nltk.tokenize import RegexpTokenizer
from nltk.corpus import cmudict
from nltk.util import ngrams


d = cmudict.dict()

def split_lines(filename):
    """
    Tokenizes the file and returns a list of tokens for
    each line of poetry in the file.
    """
    # Keep apostrophes and hyphens
    tokenizer = RegexpTokenizer('\w[\w|\'|-]*\w|\w')

    line_tokens = []
    with open(filename) as f:
        for line in f:
            line = line.strip()
            if (line.isdigit()):
                continue
            if (len(line) > 0):
                line = line.lower()
                tokens = tokenizer.tokenize(line)

                if len(tokens) > 1:
                    line_tokens.append(tokens)

    return line_tokens

def parse_rhyme(word):
    """
    Parses each word in a line for rhyme.
    """
    k = ''
    try:
        pronounciation = d[word][-1]
        k = ','.join(pronounciation[-2:])

    except (KeyError):
        # Can't do anything if word is not in dictionary
        pass

    return word, k

def parse_words(line):
    tot = 0
    for word in line:
        sk = parse_rhyme(word)[1]
        yield word,sk


if __name__=='__main__':
    line_tokens = []
    files = ['data/shakespeare.txt', 'data/more_shakespeare.txt', 'data/spenser.txt']
    for filename in files:
        line_tokens.extend(split_lines(filename))
    rhyme = {}

    for line in line_tokens:
        for word,sk in parse_words(line):
            # Save meter of word
            if len(sk) > 0:
                if sk in rhyme.keys():
                    rhyme[sk].add(word)
                else:
                    rhyme[sk] = set()
                    rhyme[sk].add(word)

    for k, v in rhyme.items():
       rhyme[k] = list(v)

    # Create inverse mappings to lookup words
    inverted_rhyme = {}
    for key, value in rhyme.items():
        for string in value:
            inverted_rhyme.setdefault(string, []).append(key)

    with open('json/rhyme.json', 'w') as f:
        json.dump(rhyme, f)
    with open('json/inverted_rhyme.json', 'w') as f:
        json.dump(inverted_rhyme, f)
