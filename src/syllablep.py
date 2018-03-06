def converttoint(line, wordtonum, numtoword):
    line = line.split(' ')
    converted = []
    line.append('\n')
    quote = False
    q = "'"
    for word in line:
        word = word.lower()
        if(word[0] == q[0] and (not word.replace(q[0], '') == "tis")):
            word = word.replace(q[0], '')
            quote = True
        if quote:
            if q[0] in word:
                word = word.replace(q[0], '')
                quote = False
        if word in wordtonum:
            converted.append(wordtonum[word])
        else:
            wordtonum[word] = len(wordtonum)
            numtoword[len(numtoword)] = word
            converted.append(wordtonum[word])
            # print(word)
            # print(numtoword[wordtonum[word]])
    return converted

def removechars(original):
    replacement = original
    for char in replacement:
        if char in '.,:;?()':
            replacement = replacement.replace(char, '')
    return replacement

def readshakes():
    '''
    This function reads all of the poems stored in Shakespeare.txt and
    removes punctuation, and returns two lists: one storing each poem as a
    list of words and newline characters, and another storing a list of
    numbers corresponding to each poem.
    '''
    pfile = open("data/shakespeare.txt", "r")
    pread = pfile.read().splitlines()
    reading = False
    line = 0
    poem = []
    poems = []
    poemnumbers = []
    wordtonum = {}
    numtoword = {}
    currentpoem = 0
    print(len(pread[0][0:5]))
    print(pread[0][0:5])
    for p in pread:
        if (p[0:19] == '                   '):
            currentpoem = int(p[19:len(p)])
            #print(currentpoem)
            line = 1
        else:
            if len(p) == 0:
                if len(poem) > 0:
                    poems.append(poem)
                    poemnumbers.append(currentpoem)
                    poem = []
                    line = 0

            else:
                if (currentpoem == 99):
                    if (line < 14):
                        poem.append(converttoint(removechars(p), wordtonum, numtoword))
                    else:
                        poem.append(converttoint(removechars(p[2:len(p)]), wordtonum, numtoword))
                else:
                    if (currentpoem == 126):
                        if (line < 11):
                            poem.append(converttoint(removechars(p), wordtonum, numtoword))
                        else:
                            poem.append(converttoint(removechars(p[2:len(p)]), wordtonum, numtoword))
                    else:
                        if (line < 13):
                            poem.append(converttoint(removechars(p), wordtonum, numtoword))
                        else:
                            poem.append(converttoint(removechars(p[2:len(p)]), wordtonum, numtoword))
                line += 1
    return (poems, poemnumbers, wordtonum, numtoword)


sfile = open("data/Syllable_dictionary.txt", "r")
syllableslist = sfile.read().splitlines()

for syllables in syllableslist:
    syllables = syllables.split(' ')
    print(syllables)

(ps, pns, wtn, ntw) = readshakes()
