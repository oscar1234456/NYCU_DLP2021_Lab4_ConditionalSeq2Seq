import matplotlib.ticker as ticker
import numpy as np
from os import system
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu

################################
#Example inputs of compute_bleu
################################
#The target word
reference = 'accessed'
#The word generated by your model
output = 'access'

#compute BLEU-4 score
def compute_bleu(output, reference):
    cc = SmoothingFunction()
    if len(reference) == 3:
        weights = (0.33,0.33,0.33)
    else:
        weights = (0.25,0.25,0.25,0.25)
    return sentence_bleu([reference], output,weights=weights,smoothing_function=cc.method1)


"""============================================================================
example input of Gaussian_score

words = [['consult', 'consults', 'consulting', 'consulted'],
['plead', 'pleads', 'pleading', 'pleaded'],
['explain', 'explains', 'explaining', 'explained'],
['amuse', 'amuses', 'amusing', 'amused'], ....]

the order should be : simple present, third person, present progressive, past
============================================================================"""

def Gaussian_score(words):
    words_list = []
    score = 0
    yourpath = ''#should be your directory of train.txt
    with open(yourpath,'r') as fp:
        for line in fp:
            word = line.split(' ')
            word[3] = word[3].strip('\n')
            words_list.extend([word])
        for t in words:
            for i in words_list:
                if t == i:
                    score += 1
    return score/len(words)