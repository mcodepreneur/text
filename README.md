### Abstract
How would one use recently developed transformer language models small enough to run quickly on small mobile computers to obtain intelligent, context informed typo correction? The ml-assisted typing interface on my phone (I suspect they use some kind of n-gram model) displays suggestions for words it thinks you are going to type, and the suggestions get better as you type the word, but doesn't autocorrect unless the word is misspelled. Even when the user inputs a word that is clearly out of context, the "autocorrector" doesn't intervene. My proposal is to use foundational transformer language models to identify when the probability of a word (GPT-2 for last word predictions, BERT for masked, bidirectional, follow up predictions) is low and there is a similar word that maximizes a combined metric of their likelihood and similarity defined by multiplying the token prediction probability, taken to a power 1>p>0: 1.6, by their similarity measured by SequenceMatching ([0, 1]) (augmented from stock because of sensitivity to letter changes being drastically increased if the length of the words are small, because of fewer sequences to be matched. To remedy this we adjust the SequenceMatching similarity by an augmentation table: [1: 0.5, 2: 0.3, 3: 0.2, 4: 0.1, >4: 0], that maps out the approximate bias along the affected groups (len<=4), and adjust the augment by the absolute difference in length (letter difference) defining an exponential decay curve: agument = augment\*adjustment=[e**(-1.5*difference)], so that the amount of augmentation, determined by the mean of their lengths will decay exponentially as the difference in length of the words increases (augment will be 1/5th the size if the lengths differ by one, etc.), due to the bias only affecting small words that are very close in length), and selcting only when the similarity between predicted word and the target word is >= 0.5, then applying a logarithmic transformation to the similarities: (log(sim/0.5)/log(2))**4, creating an exponential distribution mapping the similarities back to [0, 1], and creating and defining a threshold of this new combined metric: 0.0002, to decide whether to correct the target word. This threshold rises exponentially with each next word from the end of the input, to account for the increasing confidence of the model with increasing tailing context. The target word would not be corrected to the top predicted correction, even when the metric for that prediction passes the threshold, if the target word itself is within a set of predictions with scores above 5 percent of the top prediction. This model allows prediction architectures to be changed easily in the event of more efficient language modeling being developed.

### Files (Experimental)
```
text[1-5].ipynb : development/tuning
text6.ipynb     : model case graphing utility
text7.ipynb     : correction engine
text8.py        : example real-time correction utility [test]
```
### Future Development
* Dataset creation and empirical fine-tuning
* Implementation changes for code performance
* Allows for recalculation and correcting other words based on corrected context but current implementation only gives one correction at a time
