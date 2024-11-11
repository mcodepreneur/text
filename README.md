### Abstract
How would one use recently developed transformer language models small enough to run quickly on small mobile computers to obtain intelligent, context informed typo correction? The ml-assisted typing interface on my phone (I assume they use some kind of n-gram model) displays suggestions for words it thinks you are going to type, and the suggestions get better as you type the word, but doesn't autocorrect unless the word is misspelled. Even when the user inputs a word that is clearly out of context, the "autocorrector" doesn't intervene. My proposal is to use GPT-2 and BERT to identify when the probability of a word (GPT-2 for last word predictions, BERT for masked, bidirectional, follow up predictions) is low and there is a similar word that maximizes a combined metric of their likelyhood and similarity defined by multiplying the token prediction probabilities, taken to a power 1>p>0: 1.6, by their similarities, only where the SequenceMatching similarity (adjusted from stock because of bias where a one letter chage could be drastically more impactful to the similarity if the length of the words were small <4, because of fewer seqences sequences to be matched. To remedy this we adjust the SequenceMatching similarity by an adjustment table that maps out the approximate bias along the affected groups [len<=4] and augments the adjustment by absolute difference in length [letter difference] defining an exponential decay [adjustment\*e**(-1.5*difference)] so that the amount of adjustment given, determined by the mean of their lengths will decay exponentially as the difference in length of the words increases, due to the bias only affecting small words that are close in length.), ([0, 1]) between the predicted word and the target word is above or equal to 0.5, then applying a logarithmic transformation to the similarites, mapping it back to [0, 1]: (log(x/0.5)/log(2))**4, [figure] and defining a threshold of this new combined metric to decide whether to correct the target word. This threshold is static for the last word entered and 1 word into the context, then rises exponentially each next word from the end of the input. The target word would not be corrected to the top predicted correction, even when the the metric for that prediction passes the threshold, if the target word itself is within a set of predictions with scores above 20 percent of the top prediction.

### Files
```
text[1-5]: development/tuning files\n
text6: model graphing utility
text7: correction engine
```
