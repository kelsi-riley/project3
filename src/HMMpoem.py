########################################
# CS/CNS/EE 155 2017
# Problem Set 5
#
# Author:       Andrew Kang
# Description:  Set 5
########################################

from HMM import unsupervised_HMM
from Utility import Utility
import processing

def generations(HMM, k, sylls, endsylls, linesylls, rhymedict, wantrhyme):
    '''
    This function generates k emissions for each HMM, where each emission
    consists of 14 lines. Then, it converts the emission from a list of
    integers into a string of words and prints it out.
    Input:
        HMM: a trained HMM
        k: the number of emissions to generate per HMM
    '''
    q = "''"
    print("{:30}".format('Generated Emissions'))
    print('#' * 70)
    # Generate k input sequences.
    for i in range(k):
        # note, I have now modified generate_emission() in HMM.py
        # so that in theory it should print out d lines, where d is the
        # argument passed into it.
        # if (linesylls):
        #     emission, states = HMM.generate_emission_set_sylls(14, sylls, endsylls)
        # else:
        #     emission, states = HMM.generate_emission(14)
        if (wantrhyme):
            emission, states = HMM.generate_emission_rhyme(14, sylls, endsylls, rhymedict)
        else:
            if (linesylls):
                emission, states = HMM.generate_emission_set_sylls(14, sylls, endsylls)
            else:
                emission, states = HMM.generate_emission(14)
        line = []
        poem = []
        count = 0
        for e in emission:
            for i in e:
                line.append(ntw[i])
            line = ' '.join([str(i) for i in line])
            # in theory, this should capitalize the first word of every line,
            # but I haven't tested it yet, so who knows.
            if (line[0] == q[0]):
                line = line[0] + line[1].upper() + line[2:]
            else:
                line = line[0].upper() + line[1:]
            # if we are in the final two lines of the poem, then we add two
            # spaces to the beginning of the line so that they are formatted
            # correctly.
            if (count >= 12):
                line = "  " + line
            poem.append(line)
            line = []
            count += 1
        poem = ''.join([str(i) for i in poem])
        print("generated poem:")
        # Print the results.
        print("{:30}".format(poem))

    print('')
    print('')


def unsupervised_generation(ps, D, n_states, N_iters, k, sylls, endsylls, rhymedict, wantrhyme, linesylls=True):
    '''
    Trains an HMM using unsupervised learning on the poems and then calls
    generations() to generate k emissions for each HMM, processing the emissions
    and printing them as strings.

    Arguments:
        ps: the list of poems, where each poem is a list of integers
            representing the tokens (words) of the poem.
        D: the number of "words" contained in ps.
        n_states: number of hidden states that the HMM should have.
        N_iters: the number of iterations the HMM should train for.
        k: the number of generations for each HMM.
    '''
    # simply tells us that we are running unsupervised learning. this isn't
    # necessary, but is nice for now.
    print('')
    print('')
    print('#' * 70)
    print("{:^70}".format("Running Unsupervised Learning With %d States") %n_states)
    print('#' * 70)
    print('')
    print('')

    # Train the HMM.
    HMM = unsupervised_HMM(ps, D, n_states, N_iters)
    HMM.save_HMM(n_states, N_iters, 1)
    # generates and prints "poems"
    #generations(HMM, k, sylls, endsylls, linesylls, rhymedict, wantrhyme)



if __name__ == '__main__':
    # reads in the shakespeare poems.
    (ps, pns, wtn, ntw, rdict) = processing.readshakes()
    (sylls, endsylls) = processing.readsylls(wtn)
    k = 5 # number of poems to generate for each model.
    #only if we are trying to rhyme yo
    wantrhyme = False
    if (wantrhyme):
        for p in ps:
            p.reverse()
            p.insert(0, 0)
            p.pop()
    # Train the HMM.
    n_states = [8, 10, 12, 14, 16] # [2, 4, 8, 16, 32]
    for n in n_states:
        unsupervised_generation(ps, len(wtn), n, 400, k, sylls, endsylls, rdict, wantrhyme)
