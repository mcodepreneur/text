### Abstract
How would one use recently developed transformer language models small enough to run quickly on small mobile computers to obtain intelligent, context informed typo correction? The ml-assisted typing interface on my phone (I suspect they use some kind of n-gram model) displays suggestions for words it thinks you are going to type, and the suggestions get better as you type the word, but doesn't autocorrect unless the word is misspelled. Even when the user inputs a word that is clearly out of context, the "autocorrector" doesn't intervene. My proposal is to use GPT-2 and BERT to identify when the probability of a word (GPT-2 for last word predictions, BERT for masked, bidirectional, follow up predictions) is low and there is a similar word that maximizes a combined metric of their likelihood and similarity defined by multiplying the token prediction probabilities, taken to a power 1>p>0: 1.6, by their similarities, only where the SequenceMatching similarity (adjusted from stock because of sensitivity to letter changes being drastically increased if the length of the words are small, because of fewer sequences to be matched. To remedy this we adjust the SequenceMatching similarity by an adjustment table: [1: 0.5, 2: 0.3, 3: 0.2, 4: 0.1, >4: 0], that maps out the approximate bias along the affected groups (len<=4), and augment the adjustment by the absolute difference in length (letter difference) defining an exponential decay curve (adjustment\*augment=[e**(-1.5*difference)]) so that the amount of adjustment given, determined by the mean of their lengths will decay exponentially as the difference in length of the words increases (adjustment will be 1/5th the size if the length of the words are one apart), due to the bias only affecting small words that are very close in length.), ([0, 1]) between the predicted word and the target word is above or equal to 0.5, then applying a logarithmic transformation to the similarities: (log(x/0.5)/log(2))**4, creating an exponential distribution mapping the similarities back to [0, 1], and creating and defining a threshold of this new combined metric: 0.0002, to decide whether to correct the target word. This threshold rises exponentially with each next word from the end of the input, to account for the increasing confidence of the model with increasing tailing context. The target word would not be corrected to the top predicted correction, even when the metric for that prediction passes the threshold, if the target word itself is within a set of predictions with scores above 5 percent of the top prediction. This model allows prediction architectures to be changed easily in the event of development of more efficient language modeling.

### Files (Experimental)
```
text[1-5]: development/tuning
text6:     model case graphing utility
text7:     correction engine
```
## Future Development
* Methodical fine-tuning to reduce false positives
* Incorporating supplemental traditional autocorrection libraries
* Implementation changes for executable efficiency
