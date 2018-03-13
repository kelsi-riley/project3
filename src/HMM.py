########################################
# CS/CNS/EE 155 2017
# Problem Set 5
#
# Author:       Andrew Kang
# Description:  Set 5 solutions
########################################

import random

class HiddenMarkovModel:
    '''
    Class implementation of Hidden Markov Models.
    '''

    def __init__(self, A, O):
        '''
        Initializes an HMM. Assumes the following:
            - States and observations are integers starting from 0.
            - There is a start state (see notes on A_start below). There
              is no integer associated with the start state, only
              probabilities in the vector A_start.
            - There is no end state.

        Arguments:
            A:          Transition matrix with dimensions L x L.
                        The (i, j)^th element is the probability of
                        transitioning from state i to state j. Note that
                        this does not include the starting probabilities.

            O:          Observation matrix with dimensions L x D.
                        The (i, j)^th element is the probability of
                        emitting observation j given state i.

        Parameters:
            L:          Number of states.

            D:          Number of observations.

            A:          The transition matrix.

            O:          The observation matrix.

            A_start:    Starting transition probabilities. The i^th element
                        is the probability of transitioning from the start
                        state to state i. For simplicity, we assume that
                        this distribution is uniform.
        '''

        self.L = len(A)
        self.D = len(O[0])
        self.A = A
        self.O = O
        self.A_start = [1. / self.L for _ in range(self.L)]

    def save_HMM(self, num_states, num_iters, HMMid):
        '''
        Arguments:
            num_states: the number of hidden states of the HMM.
            num_iters: the number of iterations the HMM was trained for.
            HMMid: an integer id to distinguish between different models which
                   have the same number of hidden states and training iterations.
                   for our puproses, we used 1 if the model was trained
                   backwards (for our rhyming poem generations), and 2 if it
                   was trained forwards.
        This function stores trained HMMs in text files, so that they can be
        read later and we don't have to retrain our HMM models each time we run
        our code, yielding significiant runtime savings.
        '''
        fname = str(num_states)+"_"+str(num_iters)+"_"+str(HMMid)+".txt"
        f = open(fname, "w")
        # The first line contains the numbers of states and observations,
        # seperated by a space.
        f.write(str(self.L)+" "+str(self.D)+"\n")
        # subsequent lines each represent a list of A, with the contents of the
        # list seperated by spaces, and then the remaining lines each represent
        # a list of O, with the contents of the list seperated by spaces as well.
        for a in self.A:
            f.write(' '.join([str(i) for i in a]))
            f.write("\n")
        for o in self.O:
            f.write(' '.join([str(i) for i in o]))
            f.write("\n")
        f.close()

    def forward(self, x, normalize=False):
        '''
        Uses the forward algorithm to calculate the alpha probability
        vectors corresponding to a given input sequence.

        Arguments:
            x:          Input sequence in the form of a list of length M,
                        consisting of integers ranging from 0 to D - 1.

            normalize:  Whether to normalize each set of alpha_j(i) vectors
                        at each i. This is useful to avoid underflow in
                        unsupervised learning.

        Returns:
            alphas:     Vector of alphas.

                        The (i, j)^th element of alphas is alpha_j(i),
                        i.e. the probability of observing prefix x^1:i
                        and state y^i = j.

                        e.g. alphas[1][0] corresponds to the probability
                        of observing x^1:1, i.e. the first observation,
                        given that y^1 = 0, i.e. the first state is 0.
        '''

        M = len(x)      # Length of sequence.
        alphas = [[0. for _ in range(self.L)] for _ in range(M + 1)]

        # Note that alpha_j(0) is already correct for all j's.
        # Calculate alpha_j(1) for all j's.
        for curr in range(self.L):
            alphas[1][curr] = self.A_start[curr] * self.O[curr][x[0]]

        # Calculate alphas throughout sequence.
        for t in range(1, M):
            # Iterate over all possible current states.
            for curr in range(self.L):
                prob = 0

                # Iterate over all possible previous states to accumulate
                # the probabilities of all paths from the start state to
                # the current state.
                for prev in range(self.L):
                    prob += alphas[t][prev] \
                            * self.A[prev][curr] \
                            * self.O[curr][x[t]]

                # Store the accumulated probability.
                alphas[t + 1][curr] = prob

            if normalize:
                norm = sum(alphas[t + 1])
                for curr in range(self.L):
                    alphas[t + 1][curr] /= norm

        return alphas


    def backward(self, x, normalize=False):
        '''
        Uses the backward algorithm to calculate the beta probability
        vectors corresponding to a given input sequence.

        Arguments:
            x:          Input sequence in the form of a list of length M,
                        consisting of integers ranging from 0 to D - 1.

            normalize:  Whether to normalize each set of alpha_j(i) vectors
                        at each i. This is useful to avoid underflow in
                        unsupervised learning.

        Returns:
            betas:      Vector of betas.

                        The (i, j)^th element of betas is beta_j(i), i.e.
                        the probability of observing prefix x^(i+1):M and
                        state y^i = j.

                        e.g. betas[M][0] corresponds to the probability
                        of observing x^M+1:M, i.e. no observations,
                        given that y^M = 0, i.e. the last state is 0.
        '''

        M = len(x)      # Length of sequence.
        betas = [[0. for _ in range(self.L)] for _ in range(M + 1)]

        # Initialize initial betas.
        for curr in range(self.L):
            betas[-1][curr] = 1

        # Calculate betas throughout sequence.
        for t in range(-1, -M - 1, -1):
            # Iterate over all possible current states.
            for curr in range(self.L):
                prob = 0

                # Iterate over all possible next states to accumulate
                # the probabilities of all paths from the end state to
                # the current state.
                for nxt in range(self.L):
                    if t == -M:
                        prob += betas[t][nxt] \
                                * self.A_start[nxt] \
                                * self.O[nxt][x[t]]

                    else:
                        prob += betas[t][nxt] \
                                * self.A[curr][nxt] \
                                * self.O[nxt][x[t]]

                # Store the accumulated probability.
                betas[t - 1][curr] = prob

            if normalize:
                norm = sum(betas[t - 1])
                for curr in range(self.L):
                    betas[t - 1][curr] /= norm

        return betas

    def unsupervised_learning(self, X, N_iters):
        '''
        Trains the HMM using the Baum-Welch algorithm on an unlabeled
        datset X. Note that this method does not return anything, but
        instead updates the attributes of the HMM object.

        Arguments:
            X:          A dataset consisting of input sequences in the form
                        of lists of length M, consisting of integers ranging
                        from 0 to D - 1. In other words, a list of lists.

            N_iters:    The number of iterations to train on.
        '''

        # Note that a comment starting with 'E' refers to the fact that
        # the code under the comment is part of the E-step.

        # Similarly, a comment starting with 'M' refers to the fact that
        # the code under the comment is part of the M-step.

        for iteration in range(1, N_iters + 1):

            # Numerator and denominator for the update terms of A and O.
            A_num = [[0. for i in range(self.L)] for j in range(self.L)]
            O_num = [[0. for i in range(self.D)] for j in range(self.L)]
            A_den = [0. for i in range(self.L)]
            O_den = [0. for i in range(self.L)]

            # For each input sequence:
            for x in X:
                M = len(x)
                # Compute the alpha and beta probability vectors.
                alphas = self.forward(x, normalize=True)
                betas = self.backward(x, normalize=True)

                # E: Update the expected observation probabilities for a
                # given (x, y).
                # The i^th index is P(y^t = i, x).
                for t in range(1, M + 1):
                    P_curr = [0. for _ in range(self.L)]

                    for curr in range(self.L):
                        P_curr[curr] = alphas[t][curr] * betas[t][curr]

                    # Normalize the probabilities.
                    norm = sum(P_curr)
                    for curr in range(len(P_curr)):
                        P_curr[curr] /= norm

                    for curr in range(self.L):
                        if t != M:
                            A_den[curr] += P_curr[curr]
                        O_den[curr] += P_curr[curr]
                        O_num[curr][x[t - 1]] += P_curr[curr]

                # E: Update the expectedP(y^j = a, y^j+1 = b, x) for given (x, y)
                for t in range(1, M):
                    P_curr_nxt = [[0. for _ in range(self.L)] for _ in range(self.L)]

                    for curr in range(self.L):
                        for nxt in range(self.L):
                            P_curr_nxt[curr][nxt] = alphas[t][curr] \
                                                    * self.A[curr][nxt] \
                                                    * self.O[nxt][x[t]] \
                                                    * betas[t + 1][nxt]

                    # Normalize:
                    norm = 0
                    for lst in P_curr_nxt:
                        norm += sum(lst)
                    for curr in range(self.L):
                        for nxt in range(self.L):
                            P_curr_nxt[curr][nxt] /= norm

                    # Update A_num
                    for curr in range(self.L):
                        for nxt in range(self.L):
                            A_num[curr][nxt] += P_curr_nxt[curr][nxt]

            for curr in range(self.L):
                for nxt in range(self.L):
                    self.A[curr][nxt] = A_num[curr][nxt] / A_den[curr]

            for curr in range(self.L):
                for xt in range(self.D):
                    self.O[curr][xt] = O_num[curr][xt] / O_den[curr]

    def generate_emission(self):
        '''
        Generates an emission of length 14 lines, assuming that the starting state
        is chosen uniformly at random. This emission generator does not consider
        the number of syllables per line, and a line may contain only the new
        line character when generated this way.

        Returns:
            emission:   The randomly generated emission as a list.

            states:     The randomly generated states as a list.
        '''

        emission = []
        state = random.choice(range(self.L))
        sline = []
        states = []
        line = []

        for t in range(14):
            noend = True
            while(noend):
                # Append state.
                sline.append(state)

                # Sample next observation.
                rand_var = random.uniform(0, 1)
                next_obs = 0

                while rand_var > 0:
                    rand_var -= self.O[state][next_obs]
                    next_obs += 1

                next_obs -= 1
                line.append(next_obs)
                if (next_obs == 0):
                    noend = False
                    emission.append(line)
                    line = []
                    states.append(sline)
                    sline = []

                # Sample next state.
                rand_var = random.uniform(0, 1)
                next_state = 0

                while rand_var > 0:
                    rand_var -= self.A[state][next_state]
                    next_state += 1

                next_state -= 1
                state = next_state

        return emission, states

    def generate_emission_set_sylls(self, sylls, endsylls):
        '''
        Generates an emission of length 14, assuming that the starting state
        is chosen uniformly at random. By generating an emission of length
        14, we are trying to resemble a sonnet.

        Arguments:
            sylls:      A dictionary mapping the integer representation of a
                        word to a list containing the number of syllables it
                        may have.
            endsylls:   A dictionary mapping the integer representation of a
                        word to a list containing the number of syllables it
                        might have if it is the last word of a line
        Returns:
            emission:   The randomly generated emission as a list.

            states:     The randomly generated states as a list.
        '''

        emission = []
        state = random.choice(range(self.L))
        sline = []
        states = []
        line = []
        syllables = 0
        s = []

        for t in range(14):
            noend = True
            while(noend):
                nonext = True
                count = 0
                next_syll = 0
                while (nonext and count < 30):
                    # Sample next observation.
                    rand_var = random.uniform(0, 1)
                    next_obs = 0

                    while rand_var > 0:
                        rand_var -= self.O[state][next_obs]
                        next_obs += 1

                    next_obs -= 1

                    # We can only have our next observation be 0, which
                    # corresponds to the newline character if we have 10
                    # syllables in the line.
                    if (next_obs == 0):
                        if (syllables == 10):
                            nonext = False
                        else:
                            count += 1
                    else:
                        if (next_obs in sylls):
                            # finds if the resulting line could be considered
                            # to have less than 10 syllables.
                            lessthanlim = False
                            for i in range(len(sylls[next_obs])):
                                if (syllables + sylls[next_obs][i] < 10):
                                    lessthanlim = True
                                    next_syll = sylls[next_obs][i]
                            # finds if the resulting line could be considered
                            # to have 10 syllables, with the next_obs being
                            # the final word of the line.
                            endlim = False
                            for i in range(len(endsylls[next_obs])):
                                if (syllables + endsylls[next_obs][i] == 10):
                                    endlim = True
                                    next_syll = endsylls[next_obs][i]
                            # if the line does not have less than 10 syllables,
                            # and our next observation 
                            if not lessthanlim:
                                if not endlim:
                                    count += 1
                                else:
                                    nonext = False
                            else:
                                nonext = False
                        else:
                            count += 1

                if not nonext:
                    # Append state.
                    sline.append(state)
                    line.append(next_obs)
                    syllables += next_syll
                    s.append(next_syll)

                    if (next_obs == 0):
                        noend = False
                        emission.append(line)
                        line = []
                        states.append(sline)
                        sline = []
                        syllables = 0
                        s = []

                else:
                    sline.pop()
                    line.pop()
                    syllables -= s.pop()

                rand_var = random.uniform(0,1)
                next_state = 0
                if (len(sline) == 0):
                    state = states[len(states) -1][len(states[len(states) -1]) -1]
                else:
                    state = sline[len(sline) - 1]
                while rand_var > 0:
                    rand_var -= self.A[state][next_state]
                    next_state += 1

                next_state -= 1
                state = next_state

        return emission, states

    def generate_emission_rhyme(self, M, sylls, endsylls, rhymedict):
        '''
        Generates an emission of length M, assuming that the starting state
        is chosen uniformly at random.

        Arguments:
            M:          Number of lines to generate.

        Returns:
            emission:   The randomly generated emission as a list.

            states:     The randomly generated states as a list.
        '''
        #print(rhymedict)
        rhymes = list(rhymedict.keys())
        rhymea = 0
        rhymeb = 0
        emission = []
        state = random.choice(range(self.L))
        linenum = 14
        sline = []
        states = []
        line = []
        syllables = []
        nextsyllables = []
        newsyllables = []
        tried = 0
        endlim = False
        cantrhyme = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

        while not (linenum == 0):
            noend = True
            while(noend):
                neednext = False
                #print(line)
                nonext = True
                if (syllables == []):
                    if (linenum == 14) or (linenum % 4 == 0):
                        rhymea = []
                        total = 0.0
                        for r in rhymes:
                            rhymea.append(self.O[state][r])
                            total += self.O[state][r]
                        for r in range(len(rhymes)):
                            rhymea[r] = rhymea[r]/total
                        rand_var = random.uniform(0, 1)
                        next_obs = 0
                        while rand_var > 0:
                            rand_var -= rhymea[next_obs]
                            next_obs += 1
                        next_obs -= 1
                        next_obs = rhymes[next_obs]
                        rhymea = next_obs
                        nonext = False
                    else:
                        if (linenum % 4 == 3):
                            rhymeb = []
                            total = 0.0
                            for r in rhymes:
                                rhymeb.append(self.O[state][r])
                                total += self.O[state][r]
                            for r in range(len(rhymes)):
                                rhymeb[r] = rhymeb[r]/total
                            rand_var = random.uniform(0, 1)
                            next_obs = 0
                            while rand_var > 0:
                                rand_var -= rhymeb[next_obs]
                                next_obs += 1
                            next_obs -= 1
                            next_obs = rhymes[next_obs]
                            rhymeb = next_obs
                            nonext = False
                        else:
                            if (linenum == 13) or (linenum % 4 == 2):
                                rhymeswitha = rhymedict[rhymea]
                                probs = []
                                total = 0.0
                                for r in rhymeswitha:
                                    probs.append(self.O[state][r])
                                    total += self.O[state][r]
                                if (total == 0.0):
                                    if (linenum == 13):
                                        cantrhyme[linenum-1] += 1
                                        sline = []
                                        states = []
                                        line = []
                                        emission = []
                                        syllables = []
                                        nextsyllbles = []
                                        newsyllables = []
                                        linenum = 14
                                    else:
                                        cantrhyme[linenum-1] += 1
                                        sline = []
                                        line = []
                                        emission = emission[:14-linenum-2]
                                        states = states[:14-linenum-2]
                                        syllables = []
                                        nextsyllbles = []
                                        newsyllables = []
                                        linenum += 2
                                else:
                                    for r in range(len(rhymeswitha)):
                                        probs[r] = probs[r]/total
                                    rand_var = random.uniform(0, 1)
                                    next_obs = 0
                                    while rand_var > 0:
                                        rand_var -= probs[next_obs]
                                        next_obs += 1
                                    next_obs -= 1
                                    next_obs = rhymeswitha[next_obs]
                                    nonext = False
                            else:
                                rhymeswithb = rhymedict[rhymeb]
                                probs = []
                                total = 0.0
                                for r in rhymeswithb:
                                    probs.append(self.O[state][r])
                                    total += self.O[state][r]
                                if (total == 0.0):
                                    cantrhyme[linenum-1] += 1
                                    sline = []
                                    line = []
                                    emission = emission[:14-linenum-2]
                                    states = states[:14-linenum-2]
                                    syllables = []
                                    nextsyllbles = []
                                    newsyllables = []
                                    linenum += 2
                                else:
                                    for r in range(len(rhymeswithb)):
                                        probs[r] = probs[r]/total
                                    rand_var = random.uniform(0, 1)
                                    next_obs = 0
                                    while rand_var > 0:
                                        rand_var -= probs[next_obs]
                                        next_obs += 1
                                    next_obs -= 1
                                    next_obs = rhymeswithb[next_obs]
                                    nonext = False
                    if not nonext:
                        if (next_obs in endsylls):
                            nextsyllables = endsylls[next_obs]
                            for n in nextsyllables:
                                syllables.append([n])


                # case where we aren't finding the last word of the line
                else:
                    count = 0
                    while (nonext and count < 30):
                        # Sample next observation.
                        rand_var = random.uniform(0, 1)
                        next_obs = 0

                        while rand_var > 0:
                            rand_var -= self.O[state][next_obs]
                            next_obs += 1

                        next_obs -= 1
                        if (next_obs == 0):
                            if (endlim):
                                nonext = False
                                endlim = False
                            else:
                                count += 1
                        else:
                            if (endlim):
                                nonext = False
                                endlim = False
                                noend = False
                                next_obs = 0
                            else:
                                if (next_obs in sylls):
                                    lessthanlim = False
                                    endlim = False
                                    nextsyllables = sylls[next_obs]
                                    newsyllables = []
                                    for j in range(len(syllables)):
                                        for i in range(len(nextsyllables)):
                                            nexts = list(syllables[j])
                                            nexts.append(nextsyllables[i])
                                            newsyllables.append(nexts)
                                    for j in range(len(newsyllables)):
                                        if (sum(newsyllables[j]) == 10):
                                            endlim = True
                                            syllables = list(newsyllables)
                                        else:
                                            if (sum(newsyllables[j]) < 10):
                                                lessthanlim = True
                                                syllables = list(newsyllables)


                                    if (not endlim) and (not lessthanlim):
                                        count +=1
                                    else:
                                        nonext = False


                if not nonext:
                    # Append state.
                    line.append(next_obs)
                    sline.append(state)
                    if (next_obs == 0):
                        noend = False
                        endlim = False
                        line.reverse()
                        sline.reverse()
                        emission.append(line)
                        line = []
                        states.append(sline)
                        sline = []
                        syllables = []
                        linenum -= 1

                else:
                    if (tried > 10):
                        sline = [sline[0]]
                        line = [line[0]]
                        newsyllables = []
                        for s in syllables:
                            newsyllables.append[s[0]]
                        syllables = newsyllables
                        newsyllables = []
                        tried = 0
                        neednext = True
                    else:
                        if len(sline) > 0:
                            tried += 1
                            sline.pop()
                            line.pop()
                            newsyllables = []
                            for s in syllables:
                                state = s.pop()
                                newsyllables.append(s)
                            syllables = newsyllables
                            newsyllables = []
                            neednext = True
                if (not nonext) or neednext:
                    rand_var = random.uniform(0,1)
                    next_state = 0
                    if (len(sline) == 0):
                        if (len(states)>0):
                            state = states[len(states) -1][len(states[len(states) -1]) -1]
                        else:
                            state = random.choice(range(self.L))

                    else:
                        state = sline[len(sline) - 1]
                    transprop = []
                    total = 0.0
                    for k in range(0, self.L):
                        transprop.append(self.A[state][k])
                        total += self.A[state][k]
                    for k in range(0, self.L):
                        transprop[k] = transprop[k]/total
                    while rand_var > 0:
                        rand_var -= transprop[next_state]
                        next_state += 1

                    next_state -= 1
                    state = next_state

                    if (cantrhyme[linenum-1] > 10) and linenum-4 >=0:
                        cantrhyme[linenum-1] = 0
                        sline = []
                        line = []
                        emission = emission[:14-linenum-4]
                        states = states[:14-linenum-4]
                        syllables = []
                        nextsyllbles = []
                        newsyllables = []
                        linenum += 2

        for e in emission:
            e.pop(0)
            e.append(0)
        emission.reverse()

        states.reverse()
        return emission, states

    def probability_alphas(self, x):
        '''
        Finds the maximum probability of a given input sequence using
        the forward algorithm.

        Arguments:
            x:          Input sequence in the form of a list of length M,
                        consisting of integers ranging from 0 to D - 1.

        Returns:
            prob:       Total probability that x can occur.
        '''

        # Calculate alpha vectors.
        alphas = self.forward(x)

        # alpha_j(M) gives the probability that the output sequence ends
        # in j. Summing this value over all possible states j gives the
        # total probability of x paired with any output sequence, i.e. the
        # probability of x.
        prob = sum(alphas[-1])
        return prob


    def probability_betas(self, x):
        '''
        Finds the maximum probability of a given input sequence using
        the backward algorithm.

        Arguments:
            x:          Input sequence in the form of a list of length M,
                        consisting of integers ranging from 0 to D - 1.

        Returns:
            prob:       Total probability that x can occur.
        '''

        betas = self.backward(x)

        # beta_j(0) gives the probability of the output sequence. Summing
        # this over all states and then normalizing gives the total
        # probability of x paired with any output sequence, i.e. the
        # probability of x.
        prob = sum([betas[1][k] * self.A_start[k] * self.O[k][x[0]] \
            for k in range(self.L)])

        return prob


def supervised_HMM(X, Y):
    '''
    Helper function to train a supervised HMM. The function determines the
    number of unique states and observations in the given data, initializes
    the transition and observation matrices, creates the HMM, and then runs
    the training function for supervised learning.

    Arguments:
        X:          A dataset consisting of input sequences in the form
                    of lists of variable length, consisting of integers
                    ranging from 0 to D - 1. In other words, a list of lists.

        Y:          A dataset consisting of state sequences in the form
                    of lists of variable length, consisting of integers
                    ranging from 0 to L - 1. In other words, a list of lists.
                    Note that the elements in X line up with those in Y.
    '''

    # Make a set of observations.
    observations = set()
    for x in X:
        observations |= set(x)

    # Make a set of states.
    states = set()
    for y in Y:
        states |= set(y)

    # Compute L and D.
    L = len(states)
    D = len(observations)

    # Randomly initialize and normalize matrices A and O.
    A = [[random.random() for i in range(L)] for j in range(L)]

    for i in range(len(A)):
        norm = sum(A[i])
        for j in range(len(A[i])):
            A[i][j] /= norm

    O = [[random.random() for i in range(D)] for j in range(L)]

    for i in range(len(O)):
        norm = sum(O[i])
        for j in range(len(O[i])):
            O[i][j] /= norm

    # Train an HMM with labeled data.
    HMM = HiddenMarkovModel(A, O)
    HMM.supervised_learning(X, Y)

    return HMM


def unsupervised_HMM(X, D, n_states, N_iters):
    '''
    Helper function to train an unsupervised HMM. The function determines the
    number of unique observations in the given data, initializes
    the transition and observation matrices, creates the HMM, and then runs
    the training function for unsupervised learing.

    Arguments:
        X:          A dataset consisting of input sequences in the form
                    of lists of variable length, consisting of integers
                    ranging from 0 to D - 1. In other words, a list of lists.

        n_states:   Number of hidden states to use in training.

        N_iters:    The number of iterations to train on.
    '''


    # Compute L and D.
    L = n_states

    # Randomly initialize and normalize matrices A and O.
    A = [[random.random() for i in range(L)] for j in range(L)]

    for i in range(len(A)):
        norm = sum(A[i])
        for j in range(len(A[i])):
            A[i][j] /= norm

    # Randomly initialize and normalize matrix O.
    O = [[random.random() for i in range(D)] for j in range(L)]

    for i in range(len(O)):
        norm = sum(O[i])
        for j in range(len(O[i])):
            O[i][j] /= norm

    # Train an HMM with unlabeled data.
    HMM = HiddenMarkovModel(A, O)
    HMM.unsupervised_learning(X, N_iters)

    return HMM

def unsupervised_HMM_wordcloud(X, n_states, N_iters):
    '''
    Helper function to train an unsupervised HMM. The function determines the
    number of unique observations in the given data, initializes
    the transition and observation matrices, creates the HMM, and then runs
    the training function for unsupervised learing.

    Arguments:
        X:          A dataset consisting of input sequences in the form
                    of lists of variable length, consisting of integers
                    ranging from 0 to D - 1. In other words, a list of lists.

        n_states:   Number of hidden states to use in training.

        N_iters:    The number of iterations to train on.
    '''

    # Make a set of observations.
    observations = set()
    for x in X:
        observations |= set(x)

    # Compute L and D.
    L = n_states
    D = len(observations)

    # Randomly initialize and normalize matrices A and O.
    A = [[random.random() for i in range(L)] for j in range(L)]

    for i in range(len(A)):
        norm = sum(A[i])
        for j in range(len(A[i])):
            A[i][j] /= norm

    # Randomly initialize and normalize matrix O.
    O = [[random.random() for i in range(D)] for j in range(L)]

    for i in range(len(O)):
        norm = sum(O[i])
        for j in range(len(O[i])):
            O[i][j] /= norm

    # Train an HMM with unlabeled data.
    HMM = HiddenMarkovModel(A, O)
    HMM.unsupervised_learning(X, N_iters)

    return HMM
