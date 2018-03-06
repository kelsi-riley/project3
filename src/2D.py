########################################
# CS/CNS/EE 155 2017
# Problem Set 5
#
# Author:       Andrew Kang
# Description:  Set 5
########################################

from HMM import unsupervised_HMM
from Utility import Utility
import syllablep

# THIS IS A FUNCTION I INTEND ON ACTUALLY USING BUT FOR NOW I HAVE JUST
# BASTARDIZED IT AND THEN MOVED ON TO WORKING IN MAIN()
def unsupervised_learning(n_states, N_iters):
    '''
    Trains an HMM using supervised learning on the file 'ron.txt' and
    prints the results.

    Arguments:
        n_states:   Number of hidden states that the HMM should have.
    '''
    print('')
    print('')
    print('#' * 70)
    print("{:^70}".format("Running Unsupervised Learning With %d States") %d)
    print('#' * 70)
    print('')
    print('')
    #genres, genre_map = Utility.load_ron_hidden()
    #(ps, pns, wtn, ntw) = syllablep.readshakes()

    # Train the HMM.
    HMM = unsupervised_HMM(ps, n_states, N_iters)

    return HMM



if __name__ == '__main__':
    print('')
    print('')
    print('#' * 70)
    print("{:^70}".format("Running Unsupervised Learning"))
    print('#' * 70)
    print('')
    print('')
    (ps, pns, wtn, ntw) = syllablep.readshakes()

    # Train the HMM.
    # HMM1 = unsupervised_HMM(ps, len(wtn), 2, 1000)
    # HMM1 = unsupervised_HMM(ps, len(wtn), 4, 1000)
    # HMM1 = unsupervised_HMM(ps, len(wtn), 8, 1000)
    HMM1 = unsupervised_HMM(ps, len(wtn), 16, 200)

    print("{:30}".format('Generated Emission'))
    print('#' * 70)

    # Generate k input sequences.
    for i in range(5):
        # note, I have now modified generate_emission() in HMM.py
        # so that in theory it should print out d lines, where d is the
        # argument passed into it. Unfortunately, when we train on very
        # few iterations (10), we end up with some of these lines just being
        # '\n'

        #currently, this join makes all lines after the first line start with ' '
        # I can and will fix this
        emission, states = HMM1.generate_emission(14)
        words = []
        for e in emission:
            words.append(ntw[e])
        x = ' '.join([str(i) for i in words])
        print("printing 'poem'")
        # Print the results.
        print("{:30}".format(x))

    print('')
    print('')
