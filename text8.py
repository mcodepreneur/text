## 
import tkinter as tk
import numpy as np
import pandas as pd
import torch
import threading
from transformers import GPT2TokenizerFast, GPT2LMHeadModel, AutoTokenizer, BertForMaskedLM
from torch.nn.functional import softmax
from difflib import SequenceMatcher
from autocorrect import Speller
from string import punctuation
import time
import collections.abc
import nltk
nltk.download('words')

topk = 2000  # number of top predicted tokens to retrieve (before excluding non-words) 

class GPT2:
    def __init__(self, model="gpt2"):
        self.model     =   GPT2LMHeadModel.from_pretrained(model)
        self.tokenizer = GPT2TokenizerFast.from_pretrained(model)
        self.model_id  = model
    
    def get_word_probs(self, sentence, n=topk):  # adapted from raul on stackoverflow
        inputs = self.tokenizer.encode(sentence, return_tensors="pt")
        with torch.no_grad():
            outputs = self.model(inputs)
            predictions = outputs[0]
        candidates = predictions[0, -1, :]                          # Get the next token candidates.
        topk_i = torch.topk(candidates, n).indices.tolist()         # Get the top k next token candidates.
        all_probs = torch.nn.functional.softmax(candidates, dim=-1) # Get the token probabilities for all candidates.
        topk_probs = all_probs[topk_i].tolist()                     # Filter the token probabilities for the top k candidates.
        topk_tokens = [self.tokenizer.decode([idx]).strip()         # Decode the top k candidates back to words.
                       for idx in topk_i]
        return list(zip(topk_tokens, topk_probs))

class BERT:
    def __init__(self, model="google-bert/bert-base-uncased"):
        self.model     = BertForMaskedLM.from_pretrained(model)
        self.tokenizer =   AutoTokenizer.from_pretrained(model)
        self.model_id  = model
        
    def get_word_probs(self, prompt, topk=topk):                  # Get topk masked token candidates
        inputs = self.tokenizer(prompt, return_tensors="pt")
        with torch.no_grad():
            logits = self.model(**inputs).logits
        mask_index  = (inputs.input_ids == self.tokenizer.mask_token_id)[0].nonzero(as_tuple=True)[0]
        mask_logits = logits.squeeze()[mask_index].squeeze()
        probs = softmax(mask_logits, dim=-1)
        topk = 5000
        topk_probs, topk_i = torch.topk(probs, topk, dim=-1)
        topk_tokens = np.array([self.tokenizer.decode([i]) for i in topk_i])
        return np.hstack((topk_tokens.reshape(-1,1), np.array(topk_probs).reshape(-1,1)))

M_GPT2 = GPT2("gpt2")
M_BERT = BERT("google-bert/bert-base-uncased")

def similar(a, b):
    common_len = np.ceil((len(a)+len(b))/2)
    adjustment = 0
    adjustment_table = {1: 0.5, 2: 0.3, 3: 0.2, 4: 0.1}
    if common_len in adjustment_table: adjustment = adjustment_table[common_len]*(np.e**(-k*(np.abs(len(a)-len(b))-ap))-bp)
    return SequenceMatcher(None, a, b).ratio() + adjustment
def rreplace(string, word, new_word):
    start = string.rfind(word)
    return string[0:start] + new_word + string[start+len(word):]
wl          = set(nltk.corpus.words.words())
log_map     = lambda e: np.vectorize(lambda x: np.power(np.log(x/0.5)/np.log(2), e))  # specify exponent to return vectorized mapping
after_slash = lambda x: x[(x.rfind("/")+1 if x.rfind("/") != -1 else 0):]
spell = Speller()

def correction(string, back_n):
    places = reversed(range(1,back_n+1))
    string = string.strip()
    words  = string.split()
    last_space = string.rfind(' ')
    for n in places:
        if n > len(words) or len(words) == 1: break
        spelled = False
        if n > 1:
            model  = M_BERT
            masked = "[MASK]" + words[-n][-1] if not words[-n][-1].isalpha() else "[MASK]"
            target = words[-n].strip(punctuation)
            prompt = ' '.join(words[:-n] + [masked] + words[len(words)-(n-1):])
            target = words[-n].strip(punctuation)
            if target != spell(target):
                target = spell(target)
                spelled = True
        else:
            model  = M_GPT2
            string = string.strip()
            last_space = string.rfind(' ')
            prompt = string[:last_space]
            target = string[last_space+1:].strip(punctuation)
            target = words[-n].strip(punctuation)
            if target != spell(target):
                target = spell(target)
                spelled = True
        probs  = model.get_word_probs(prompt)                
        probsp = [(str(word), float(prob), float(similar(target, word))) for word, prob in probs if word in wl]
        close_probs = [prob for prob in probsp if prob[2] > 0.5 and prob[1] >= min(0.001, probsp[consider_top][1])]
        props = [(word, (prob**prob_exp)*log_map(log_exp)(sim)) for word, prob, sim in close_probs]
        props = sorted(props, reverse=True, key=lambda x: x[1])
        props = [prop for prop in props if prop[1] > 0.000001]
        probN = threshold(n)
        make_correction = False
        if len(props) > 0 and props[0][1] > probN:
            make_correction = True
            irr_t = props[0][1] * relevency_t
            for word, score in props: 
                if score < irr_t: break
                elif target.lower() == word.lower():
                    make_correction = False
        if make_correction: return (n, props[0][0])
        if spelled: return (n, target)
    return False
    
def process_correction(string, back_n):
    corrected = string
    words     = string.split()
    is_correction = correction(string, back_n)
    if is_correction:
        n, word = is_correction
        words[-n] = word if words[-n][-1] not in punctuation else word + words[-n][-1]
        corrected = " ".join(words)
        #is_correction = correction(corrected, back_n)
    return corrected if corrected != string else False

def apply_correction(event=None):
    content = box.get("1.0", tk.END).strip("\n")
    if not content: return
        
    last_cut = content.rfind(" ")
    for punct in ",.!?":
        i = content.rfind(punct)
        if i > last_cut: last_cut = i
    for i in reversed(range(last_cut)):
        if content[i].isalpha():
            last_cut = i
            break
    split = content.find(" ", last_cut)
    if split == -1: split = 9**9
    for punct in ",.!?":
        i = content.find(punct, last_cut)
        if i != -1 and i < split: split = i
    
    text_upto = content[:split+1]
    text_new  = content[split+1:]
    def update_box():
        corrected = process_correction(text_upto, back_n)
        if corrected:
            text_new  = box.get("1.0", tk.END).strip("\n")[split+1:]
            if verbose: print(f'upto="{text_upto}", corrected="{corrected}", new="{text_new}", split=\'{content[split]}\'')
            box.delete("1.0", tk.END)
            box.insert("1.0", corrected + (' ' if corrected[-1] not in punctuation else '') + text_new)
            
    threading.Thread(target=update_box).start()

verbose = False  # print parsed text fields on correction
back_n  = 3  # number of words back from end of string, 1 is just last word
k       = 1.2  # exponent parameter for exponential decay of word length augmedented SequenceMatcher
ap      = 0.57  # exponent parameter
bp      = 1  # exponent parameter
log_exp      = 5  # exponent parameter for logarithmic mapping
prob_exp     = 1.6  # raise probability to power in ((prob**power)*log-sim)
consider_top = 100  # max top model word predictions considered
relevency_t  = 0.05  # threshold defined by portion of top proposition to exclude much smaller scored propositions for correcting
base_t       = 0.0002  # decision threshold for last word: base threshold
threshold_t  = "exponential"  # function defines decision threshold for word n from end
threshold    = {"constant":    lambda n: base_t,
                "linear":      lambda n: base_t + (base_t * (n-1)),
                "exponential": lambda n: base_t * (n**2),
                "jump-exp":    lambda n: base_t * (max(n-1,1)**2),        # jump thresholds start growing after n=2
                "jump-lin":    lambda n: base_t + (base_t * max(n-2, 0))
               }[threshold_t]

root = tk.Tk()
root.title("Text")
box = tk.Text(root, wrap='word', height=10, width=50)
box.pack()
for key in ('<KeyRelease-space>', '<KeyRelease-period>', '<KeyRelease-comma>', '<KeyRelease-exclam>', '<KeyRelease-question>'):
    box.bind(key, apply_correction)
root.mainloop()
