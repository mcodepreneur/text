## experimental real-time correction tkinter module
## implementing threading to allow the user to keep 
## typing while the model proceeses context
from torch.nn.functional import softmax
from transformers import pipeline, GPT2TokenizerFast, GPT2LMHeadModel, AutoTokenizer, BertForMaskedLM
from autocorrect import Speller
from nltk.stem import WordNetLemmatizer, PorterStemmer
from difflib import SequenceMatcher
from string import punctuation
from time import time
import tkinter as tk
import numpy as np
import pandas as pd
import threading
import torch
import nltk
nltk.download('words')

topk = 200  # number of top predicted tokens to retrieve (before excluding non-words) 

class GPT2:
    def __init__(self, model="gpt2"):
        self.model     =   GPT2LMHeadModel.from_pretrained(model)
        self.tokenizer = GPT2TokenizerFast.from_pretrained(model)
        self.model_id  = model
    
    def get_word_probs(self, sentence, n=topk):  # adapted from raun on stackoverflow
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
        return np.array(list(zip(topk_tokens, topk_probs)))

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

threshold    = {"constant":    lambda n: base_t,
                "linear":      lambda n: base_t + (base_t * (n-1)),
                "exponential": lambda n: base_t * (n**threshold_e),
                "jump-exp":    lambda n: base_t * (max(n-1,1)**threshold_e),        # jump thresholds start growing after n=2
                "jump-lin":    lambda n: base_t + (base_t * max(n-2, 0))}
lemmatizer  = WordNetLemmatizer()
lemma       = lambda x: lemmatizer.lemmatize(x)
stemmer     = PorterStemmer()
stem        = lambda x: stemmer.stem(x)
spell       = Speller()
wl          = set(nltk.corpus.words.words())
log_map     = lambda e: np.vectorize(lambda x: np.power(np.log(x/sim_bound)/np.log(1/sim_bound), e)) 
after_slash = lambda x: x[(x.rfind("/")+1 if x.rfind("/") != -1 else 0):]

def get_props(target, probs):
    probs[:, 1] = probs[:, 1].astype(float) / probs[:, 1].astype(float).sum()
    probsp = [(str(word), float(prob), float(similar(target.lower(), word.lower()))) 
              for word, prob in probs if word in wl]
    close_probs = [prob for prob in probsp if prob[2] > sim_bound and prob[1] >= 0.001]
    props = sorted([(word, (prob**prob_exp)*log_map(log_exp)(sim)) for word, prob, sim in close_probs], 
                   reverse=True, key=lambda x: x[1])
    return props

def make_correction(target, props, probN):
    make_correction = False
    if len(props) > 0 and float(props[0][1]) > probN:
        make_correction = True
        irr_t = float(props[0][1]) * relevency_t
        for word, score in props: 
            if float(score) < irr_t: break
            elif target.lower() == word.lower() or stem(target.lower()) == word.lower() or lemma(target.lower()) == word.lower():
                make_correction = False
    return make_correction

def correction(string, back_n):
    places = reversed(range(1,back_n+1))
    if back_n == 0: places = [1, 3, 2]
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
                spelled = True
        else:
            model  = M_GPT2
            string = string.strip()
            last_space = string.rfind(' ')
            prompt = string[:last_space]
            target = string[last_space+1:].strip(punctuation)
            target = words[-n].strip(punctuation)
            if target != spell(target):
                spelled = True
        probs = model.get_word_probs(prompt) 
        props = get_props(target, probs)
        probN = threshold[threshold_t](n)
        if make_correction(target, props, probN): return (n, props[0][0])
        if spelled: 
            target = spell(target)
            props = get_props(target, probs)
            if len(props) > 0 and float(props[0][1]) > probN and props[0][0] == target:
                return (n, target)
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

import json
feedback_file = "feedback.json"

def save_feedback(original, corrected, user_input):
    feedback_entry = {
        "original":   original,
        "corrected":  corrected,
        "user_input": user_input
    }
    try:
        with open(feedback_file, "r") as f:
            feedback_data = json.load(f)
    except FileNotFoundError:
        feedback_data = []
    
    feedback_data.append(feedback_entry)
    with open(feedback_file, "w") as f:
        json.dump(feedback_data, f, indent=4)
    print(f"Feedback saved: {feedback_entry}")

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
            feedback_popup(text_upto, corrected)
            
    threading.Thread(target=update_box).start()

def feedback_popup(original, corrected):
    popup = tk.Toplevel(root)
    popup.title("Feedback")

    tk.Label(popup, text=f"Original: {' '.join(original.split()[-8:])}").pack(pady=4)
    tk.Label(popup, text=f"Corrected: {' '.join(corrected.split()[-8:])}").pack(pady=4)

    tk.Label(popup, text="Was I right?").pack(pady=4)
    user_feedback = tk.StringVar(value=' '.join(corrected.split()[-8:]))

    feedback_entry = tk.Entry(popup, textvariable=user_feedback, width=40)
    feedback_entry.pack(pady=4)

    def submit_feedback():
        user_input = user_feedback.get()
        save_feedback(original.strip(), ' '.join(corrected.split()[-8:]), user_input.strip())
        popup.destroy()

    submit_btn = tk.Button(popup, text="Submit", command=submit_feedback)
    submit_btn.pack(pady=10)

verbose = False  # print parsed text fields on correction
back_n  = 4  # number of words back from end of string, 1 is just last word
k       = 1.02  # exponent parameter for exponential decay of word length augmedented SequenceMatcher
ap      = 0.68  # exponent parameter
bp      = 0.93  # exponent parameter
sim_bound    = 0.45
log_exp      = 3  # exponent parameter for logarithmic mapping
prob_exp     = 1.43  # raise probability to power in ((prob**power)*log-sim)
consider_top = 100  # max top model word predictions considered
relevency_t  = 0.054  # threshold defined by portion of top proposition to exclude much smaller scored propositions for correcting
base_t       = 0.002  # decision threshold for last word: base threshold
threshold_e  = 1.9  # exponent for exponential thresholds
threshold_t  = "exponential"  # function defines decision threshold for word n from end

root = tk.Tk()
root.title("Text")
box = tk.Text(root, wrap='word', height=10, width=50)
box.pack()
for key in ('<KeyRelease-space>', '<KeyRelease-period>', '<KeyRelease-comma>', '<KeyRelease-exclam>', '<KeyRelease-question>'):
    box.bind(key, apply_correction)
root.mainloop()