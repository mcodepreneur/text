### Abstract
How would one use recently developed transformer language models (token probabilities) small enough to run on mobile computers to obtain intelligent, context informed typo correction? The ml-assisted typing interface on my phone (I suspect they use some kind of n-gram model) displays suggestions for words it thinks you are going to type, and the suggestions get better as you type the word, but doesn't autocorrect unless the word is misspelled. Even when the user inputs a word that is clearly out of context, the "autocorrector" doesn't intervene.\
\
My proposal is to use foundational transformer language models to identify when the probability of a word (GPT-2 for last word predictions, BERT for masked, bidirectional, follow up predictions) is low and there is a similar word that maximizes a combined metric of their likelihood and similarity defined by multiplying the normalized token prediction probabilities $p_i=\frac{p_i}{\sum p}$, taken to a power $2>prob\\_exp>0$, by their similarity measured by SequenceMatching[*](#sequencematching-augmentation) ($[0, 1]$) selecting only when the similarity between predicted word and the target word is equal to or above a bound, and applying a logarithmic transformation to the similarities $(\frac{log(sim/bound)}{log(1/bound)})^{log\\_exp}$ for $log\\_exp>1$ (see [figure 6](#logarithmic-similarity-transformation)), creating an exponential distribution mapping the similarities back to $[0, 1]$, and creating and defining a threshold of this new combined metric to decide whether to correct the target word. This threshold rises exponentially with each next word from the end of the input, to account for the increasing confidence of the model with increasing tailing context. The target word would not be corrected to the top predicted correction, even when the metric for that prediction passes the threshold, if the target word itself (or its stem/lemma) is within a set of predictions with scores above percent $x$ of the top prediction. In an additional step, if a traditional spelling algorithm marks the word as misspelled and the word was not corrected, the autocorrected version of the word . This model allows prediction architectures to be changed easily in the event of more efficient language modeling being developed.\
\
Using a small, custom written dataset (currently n=356), and with slightly modified metrics to account for "false true positives" (where the model made a change where one was needed, but produced the wrong output), current testing shows, within variation because metrics cannot be that accurate at this scale (among batches $\pm0.03$), a recall (of cases where correction was needed, model corrected the word accurately) of 0.70, a precision (accurate corrections over total model corrections) of 0.75, and a false positive rate of 0.06, with an overall accuracy of 0.75. This seems to indicate this model is not only relatively accurate in correcting intended words compared traditional autocorrect (TextBlob: recall 0.45; precision 0.07; FPR 0.19; accuracy 0.41), but also resistant to false positives, something that is always important in any application of deep models. Testing is done on relatively short contexts, 1-2 sentences or fragments of sentences, but unlike n-grams, as the length of the context given increases, the predictions will only get more accurate, theoretically even with seemingly irrelevant context. This is a noted choice partly because of the limitations of creating custom datasets, but also to make sure the model is competent with the base amount of context required to correct.
### Files (Experimental)
```
text[1-5].ipynb : development/tuning
text6.ipynb     : model case visualization utility
text7.ipynb     : correction engine, test metrics, automated parameter learning
strings.txt     : test dataset, n=356
text8.py        : example real-time correction utility
```
## Development
### SequenceMatching Augmentation
SequenceMatching is augmented from stock because of sensitivity to letter changes being drastically increased if the length of the words are small, because of fewer sequences to be matched. To remedy this we adjust the SequenceMatching similarity by an augmentation table (see [Augmentation table refinement](#augmentation-table-refinement-objectiveaccuracy)), that maps out the approximate bias along the affected groups ($len\leq4$), and adjust the augment by the absolute difference in length (letter difference) defining an exponential decay curve $augment=augment * e^{(-1.01*(difference-0.68))}-0.93$ (learned then locked constants), so that the amount of augmentation, determined by the mean of their lengths will decay exponentially as the difference in length of the words increases (augment will become a penalty if the lengths differ by $>1$), due to the bias only affecting small words that are the same length, and adjusting for when words close in length and one can become partly or wholey represented in the other.
### Optana Hyperparameter Fine-Tuning
#### Hyperparameter tuning (objective=accuracy)
Optimization history\
![optimization accuracy history](https://github.com/mcodepreneur/text/blob/main/figures/optimization_history.png)\
*(figure 1)*\
\
Hyperparameter importances\
![accuracy hyperparameter importances](https://github.com/mcodepreneur/text/blob/main/figures/accuracy_importance.png)\
*(figure 2)*\
\
Hyperparameter value search **(Parallel Coordinate Plot)**\
![accuracy hyperparameter search](https://github.com/mcodepreneur/text/blob/main/figures/accuracy_values.png)\
*(figure 3)*
#### Augmentation table refinement (objective=accuracy)
Augmentation table importances\
![augmentation table value importances](https://github.com/mcodepreneur/text/blob/main/figures/adjustment_importance.png)\
*(figure 4)*\
\
Augmentation table search **(Parallel Coordinate Plot)**\
![augmentation value search](https://github.com/mcodepreneur/text/blob/main/figures/adjustment_values.png)\
*(figure 5)*\
\
Automated fine tuning produced an optimal augmentation table (as seen in figure 5):
* {1: 0.5, 2: 0.29, 3: 0.14, 4: 0.05}
  
Original proposed: {1: 0.5, 2: 0.3, 3: 0.2, 4: 0.1}
### Logarithmic Similarity Transformation
Linear and logarithmic mappings with sim_bound=0.5\
![exponential logarithmic mappings](https://github.com/mcodepreneur/text/blob/main/figures/simmap.png)\
*(figure 6)*
### Future Development
* Allow for recalculation and correcting other words based on corrected context
* Fine tuning GPT2 and BERT once my task dataset is large enough
* Use lemmas in reverse (consider all expanded of prediction)
