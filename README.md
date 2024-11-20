 ### Abstract
How would one use recently developed transformer language models (token probabilities) small enough to run quickly on small mobile computers to obtain intelligent, context informed typo correction? The ml-assisted typing interface on my phone (I suspect they use some kind of n-gram model) displays suggestions for words it thinks you are going to type, and the suggestions get better as you type the word, but doesn't autocorrect unless the word is misspelled. Even when the user inputs a word that is clearly out of context, the "autocorrector" doesn't intervene. My proposal is to use foundational transformer language models to identify when the probability of a word (GPT-2 for last word predictions, BERT for masked, bidirectional, follow up predictions) is low and there is a similar word that maximizes a combined metric of their likelihood and similarity defined by multiplying the normalized token prediction probabilities $p_i=\frac{p_i}{\sum p}$, taken to a power $1>p>0$, by their similarity measured by SequenceMatching ($[0, 1]$) (augmented from stock because of sensitivity to letter changes being drastically increased if the length of the words are small, because of fewer sequences to be matched. To remedy this we adjust the SequenceMatching similarity by an augmentation table (see figure 2), that maps out the approximate bias along the affected groups ($len\leq4$), and adjust the augment by the absolute difference in length (letter difference) defining an exponential decay curve $augment=augment * e^{(-k*(difference-a))}-b$ for positive constants $a;b;k$, so that the amount of augmentation, determined by the mean of their lengths will decay exponentially as the difference in length of the words increases (augment will become a penalty if the lengths differ by $>1$), due to the bias only affecting small words that are the same length, and adjusting for when words close in length and one can become partly or wholey represented in the other, overpowering the similarity for this model), and selecting only when the similarity between predicted word and the target word is $\geq 0.5$, then applying a logarithmic transformation to the similarities $(\frac{log(sim/0.5)}{log(2)})^p$ for $p>1$ exponent $p$, creating an exponential distribution mapping the similarities back to $[0, 1]$, and creating and defining a threshold of this new combined metric to decide whether to correct the target word. This threshold rises exponentially with each next word from the end of the input, to account for the increasing confidence of the model with increasing tailing context. The target word would not be corrected to the top predicted correction, even when the metric for that prediction passes the threshold, if the target word itself (or its stem/lemma) is within a set of predictions with scores above percent $x$ of the top prediction. This model allows prediction architectures to be changed easily in the event of more efficient language modeling being developed.\
\
Using a small, custom written dataset (currently n=330), and with slightly modified metrics to account for "false true positives" (where the model made a change where one was needed, but produced the wrong output), current testing shows, within variation because metrics cannot be that accurate at this scale (among batches $\pm0.03$), an optimal recall (of cases where correction was needed, model corrected the word accurately) of 0.69, a precision (accurate corrections over total model corrections) of 0.94, and a false positive rate of 0.03, with an overall max accuracy of 0.75. This seems to indicate this model is not only relatively accurate in correcting intended words compared traditional autocorrect (TextBlob: recall ~0.47; precision ~0.05; FPR ~0.21), but also resistant to false positives, something that is always important in any application of deep models. Testing is done on relatively short contexts, 1-2 sentences or fragments of sentences, but unlike n-grams, as the length of the context given increases, the predictions will only get more accurate, theoretically even with seemingly irrelevant context. This is a noted choice partly because of the limitations of creating custom datasets, but also to make sure the model is competent with the base amount of context required to correct.
### Files (Experimental)
```
text[1-5].ipynb : development/tuning
text6.ipynb     : model case graphing utility
text7.ipynb     : correction engine, test metrics, automated parameter learning
strings.txt     : test dataset, n=330
text8.py        : example real-time correction utility
```
### Development
#### Optana Hyperparameter Learning
* Hyperparameter importances to recall
  
![recall hyperparameter importances](https://github.com/mcodepreneur/text/blob/main/figures/recall.png)\
*(figure 1)*
* Adjustment table refinement for accuracy
  
![adjustment table value importances](https://github.com/mcodepreneur/text/blob/main/figures/adjustment_importance.png)\
*(figure 2)*\
Automated fine tuning produced two optimal adjustment talbes:\
{1: 0.513, 2: 0.351, 3: 0.168, 4: 0.048}\
{1: 0.413, 2: 0.327, 3: 0.196, 4: 0.071}\
(original proposed: {1: 0.5, 2: 0.3, 3: 0.2, 4: 0.1})\

### Future Development
* Allow for recalculation and correcting other words based on corrected context
