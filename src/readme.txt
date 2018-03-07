The part that actually runs and does a thing is in HMMpoem.py.

Most of the code I have written is in processing.py and HMMpoem.py, but I have
also also altered HMM.py a little (i.e. the unsupervised_generation() function).

I currently load in poems as lists of strings, where the strings are words and
endline characters (I have inserted the end line characters at the end of lines).

I think our next step will be to try to change it so that we get the correct
syllable count. I'm not entirely sure how to do that. At this point. We can
externally enforce it, but that seems like it would make our lines less reasonable.
Idk. 

I was thinking we could put some character at the start of the volta to indicate
that the volta is a thing.

Then we could generate our poems line by line with the last two lines restarting
from the volta state.

I dunno if that makes sense.

Additionally, we could consider externally imposing the syllable count when
generating lines.
I don't know if we have to implement that ourselves though. if not, its probability
not worth the effort. That's all. Food for thought.

Let me know if you have any ideas/suggestions/thoughts.
