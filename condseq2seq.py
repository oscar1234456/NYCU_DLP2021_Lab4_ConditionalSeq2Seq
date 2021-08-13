##
from __future__ import unicode_literals, print_function, division
from io import open
import random
import time
import math
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import matplotlib.pyplot as plt

plt.switch_backend('agg')
import matplotlib.ticker as ticker
import numpy as np
from os import system
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
from dataloader import WordSet
from layer import EncoderRNN, DecoderRNN, hiddenCellLinear, ConditionEmbegging
from utility import reparameter

##
"""========================================================================================
The sample.py includes the following template functions:

1. Encoder, decoder
2. Training function
3. BLEU-4 score function
4. Gaussian score function

You have to modify them to complete the lab.
In addition, there are still other functions that you have to 
implement by yourself.

1. The reparameterization trick
2. Your own dataloader (design in your own way, not necessary Pytorch Dataloader)
3. Output your results (BLEU-4 score, conversion words, Gaussian score, generation words)
4. Plot loss/score
5. Load/save weights

There are some useful tips listed in the lab assignment.
You should check them before starting your lab.
========================================================================================"""

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device:{device}")
SOS_token = 0
EOS_token = 1
# ----------Hyper Parameters----------#
hidden_size = 128
# The number of vocabulary
vocab_size = 28
latent_size = 32
condEmbedding_size = 8
condi_size = 4
teacher_forcing_ratio = 1.0
# empty_input_ratio = 0.1
KLD_weight = 0.0
LR = 0.01


def train(input_tensor, target_tensor, condition_tensor, encoder: EncoderRNN, decoder: DecoderRNN, hiddenLinear,
          cellLinear, conditionEmbedding, encoder_optimizer, decoder_optimizer,
          linear_hidden_optimizer, linear_cell_optimizer, embedding_optimizer, criterion):
    encoder_hidden = encoder.initHidden(condEmbedding_size)  # return (1,1,hidden_size) for hidden_0
    encoder_cell = encoder.initCell(condEmbedding_size)  # return (1,1,hidden_size) for cell_0

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()
    linear_hidden_optimizer.zero_grad()
    linear_cell_optimizer.zero_grad()
    embedding_optimizer.zero_grad()

    input_length = input_tensor.size(0)
    target_length = target_tensor.size(0)

    # encoder_outputs = torch.zeros(10, encoder.hidden_size, device=device)  # initl -> unuseful

    conditionEmbedded = conditionEmbedding(condition_tensor).view(1,1,-1)
    conditionHidden = torch.cat((encoder_hidden, conditionEmbedded), 2)  # concate condition to hidden
    conditionCell = torch.cat((encoder_cell, conditionEmbedded), 2)  # concate condition to cell

    encoder_hidden = (conditionHidden, conditionCell)  # make encoder_hidden as tuple

    loss = 0

    # ----------sequence to sequence part for encoder----------#
    hidden_mean, hidden_logVar, cell_mean, cell_logVar = encoder(input_tensor,
                                                                 encoder_hidden)  # encoder(input, hidden->tuple(h_0,c_0))
    # encoder_result: (hidden_mean, hidden_logVar, cell_mean, cell_logVar)

    hidden_latent = reparameter(hidden_mean, hidden_logVar)
    cell_latent = reparameter(cell_mean, cell_logVar)

    condition_hidden_latent = torch.cat((hidden_latent, conditionEmbedded), 2)
    condition_cell_latent = torch.cat((cell_latent, conditionEmbedded), 2)

    decoder_input = torch.tensor([[SOS_token]], device=device)

    condition_hidden_latent_toDe = hiddenLinear(condition_hidden_latent)
    condition_cell_latent_toDe = cellLinear(condition_cell_latent)

    decoder_hidden = (condition_hidden_latent_toDe, condition_cell_latent_toDe)

    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

    # ----------sequence to sequence part for decoder----------#
    if use_teacher_forcing:
        # Teacher forcing: Feed the target as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
            loss += criterion(decoder_output, target_tensor[di])
            decoder_input = target_tensor[di]  # Teacher forcing
            if np.isnan(loss.item()):
                print('Loss value is NaN!')

    else:
        # Without teacher forcing: use its own predictions as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
            topv, topi = decoder_output.topk(1)
            decoder_input = topi.squeeze().detach()  # detach from history as input

            loss += criterion(decoder_output, target_tensor[di])
            if decoder_input.item() == EOS_token:
                break

    loss.backward()

    encoder_optimizer.step()
    decoder_optimizer.step()
    linear_hidden_optimizer.step()
    linear_cell_optimizer.step()
    embedding_optimizer.step()

    return loss.item() / target_length


def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))


def trainIters(encoder, decoder, hiddenLinear, cellLinear, conditionEmbedding, n_iters, print_every=1000,
               plot_every=100, learning_rate=0.01):
    start = time.time()
    plot_losses = []
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every

    encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)
    linear_hidden_optimizer = optim.SGD(hiddenLinear.parameters(),
                                        lr=learning_rate)  # for latent to hidden to next decoder
    linear_cell_optimizer = optim.SGD(cellLinear.parameters(), lr=learning_rate)
    embedding_optimizeer = optim.SGD(conditionEmbedding.parameters(), lr=learning_rate)

    pairs = WordSet().getWordPair()  # read training data
    training_pairs = [random.choice(pairs) for _ in range(n_iters)]  # random choose wordpairs
    # training_pairs format: [[tensor([[1],[2],[3]...]), 1], [],.....,[]]

    criterion = nn.CrossEntropyLoss()

    for iter in range(1, n_iters + 1):
        training_pair = training_pairs[iter - 1]
        input_tensor = training_pair[0]
        target_tensor = training_pair[0]
        condition_tensor = training_pair[1]

        loss = train(input_tensor, target_tensor, condition_tensor, encoder,
                     decoder, hiddenLinear, cellLinear, conditionEmbedding, encoder_optimizer, decoder_optimizer,
                     linear_hidden_optimizer,
                     linear_cell_optimizer, embedding_optimizeer, criterion)
        print_loss_total += loss
        plot_loss_total += loss

        if iter % print_every == 0:
            print(f"print_loss_total:{print_loss_total},print_every:{print_every} ")
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            print('%s (%d %d%%) %.4f' % (timeSince(start, iter / n_iters),
                                         iter, iter / n_iters * 100, print_loss_avg))
        # TODO: Handle model save here(maybe store parameter and final output great model and save outside)


encoder1 = EncoderRNN(vocab_size, hidden_size + condEmbedding_size, latent_size).to(
    device)  # input_size, hidden_size, latent_size
decoder1 = DecoderRNN(hidden_size + condEmbedding_size, vocab_size).to(device)  # hidden_size, output_size
hiddenLinear1 = hiddenCellLinear(latent_size + condEmbedding_size, hidden_size + condEmbedding_size).to(device)
cellLinear1 = hiddenCellLinear(latent_size + condEmbedding_size, hidden_size + condEmbedding_size).to(device)
conditionEmedding1 = ConditionEmbegging(condi_size, condEmbedding_size).to(device)  # condi_size, condEmbedding_size

trainIters(encoder1, decoder1, hiddenLinear1, cellLinear1, conditionEmedding1, n_iters=75000, print_every=5000,
           learning_rate=LR)
# encoder, decoder, hiddenLinear, cellLinear, conditionEmbedding, n_iters, print_every=1000, plot_every=100, learning_rate=0.01
