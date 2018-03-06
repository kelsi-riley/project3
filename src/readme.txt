The part that actually runs and does a thing is in 2D.py
most of the code I have written is in syllablep.py
I have also altered HMM.py and 2D.py

Fear not, I will rename things and clean up my code/comment more clearly tomorrow

I currently load in poems as lists of strings, where the strings are words and
endline characters (I have inserted the end line characters at the end of lines).
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
