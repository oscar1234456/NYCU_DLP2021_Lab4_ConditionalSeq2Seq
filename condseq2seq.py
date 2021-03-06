##
from __future__ import unicode_literals, print_function, division

import pickle
from io import open
import random
import time
import math
import torch
import torch.nn as nn
from torch import optim
import copy
import torch.nn.functional as F
import matplotlib.pyplot as plt

plt.switch_backend('agg')
import matplotlib.ticker as ticker
import numpy as np
from os import system
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
from dataloader import WordSet
from layer import EncoderRNN, DecoderRNN, hiddenCellLinear, ConditionEmbegging
from utility import reparameter, KLDLoss
from testScore import evaluateBLEU, evaluateGaussian

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
hidden_size = 256
# The number of vocabulary
vocab_size = 28
latent_size = 32
condEmbedding_size = 8
condi_size = 4
teacher_forcing_ratio = 1.0
# empty_input_ratio = 0.1
KLD_weight = 0.1
LR = 0.01


def train(input_tensor, target_tensor, condition_tensor, encoder: EncoderRNN, decoder: DecoderRNN, hiddenLinear,
          cellLinear, conditionEmbedding, encoder_optimizer, decoder_optimizer,
          linear_hidden_optimizer, linear_cell_optimizer, embedding_optimizer, criterion, use_KLD_Weight, use_TF_ratio):
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

    conditionEmbedded = conditionEmbedding(condition_tensor).view(1, 1, -1)
    conditionHidden = torch.cat((encoder_hidden, conditionEmbedded), 2)  # concate condition to hidden
    conditionCell = torch.cat((encoder_cell, conditionEmbedded), 2)  # concate condition to cell

    encoder_hidden = (conditionHidden, conditionCell)  # make encoder_hidden as tuple

    # ----------sequence to sequence part for encoder----------#
    hidden_mean, hidden_logVar, cell_mean, cell_logVar = encoder(input_tensor,
                                                                 encoder_hidden)  # encoder(input, hidden->tuple(h_0,c_0))
    # encoder_result: (hidden_mean, hidden_logVar, cell_mean, cell_logVar)

    crossEntropyLoss = 0.0
    KLDLossvalue = (KLDLoss(hidden_mean, hidden_logVar) + KLDLoss(cell_mean, cell_logVar))
    loss = KLDLossvalue * use_KLD_Weight
    hidden_latent = reparameter(hidden_mean, hidden_logVar)
    cell_latent = reparameter(cell_mean, cell_logVar)

    condition_hidden_latent = torch.cat((hidden_latent, conditionEmbedded), 2)
    condition_cell_latent = torch.cat((cell_latent, conditionEmbedded), 2)

    decoder_input = torch.tensor([[SOS_token]], device=device)

    condition_hidden_latent_toDe = hiddenLinear(condition_hidden_latent)
    condition_cell_latent_toDe = cellLinear(condition_cell_latent)

    decoder_hidden = (condition_hidden_latent_toDe, condition_cell_latent_toDe)

    use_teacher_forcing = True if random.random() < use_TF_ratio else False

    # ----------sequence to sequence part for decoder----------#
    if use_teacher_forcing:
        # Teacher forcing: Feed the target as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
            crossEntropyLoss += criterion(decoder_output, target_tensor[di])
            decoder_input = target_tensor[di]  # Teacher forcing
            if np.isnan(crossEntropyLoss.item()):
                print('Loss value is NaN!')

    else:
        # Without teacher forcing: use its own predictions as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
            topv, topi = decoder_output.topk(1)
            decoder_input = topi.squeeze().detach()  # detach from history as input

            crossEntropyLoss += criterion(decoder_output, target_tensor[di])
            if decoder_input.item() == EOS_token:
                break

    loss += crossEntropyLoss
    loss.backward()

    encoder_optimizer.step()
    decoder_optimizer.step()
    linear_hidden_optimizer.step()
    linear_cell_optimizer.step()
    embedding_optimizer.step()

    return (loss.item() / target_length), encoder, decoder, hiddenLinear, cellLinear, conditionEmbedding, KLDLossvalue, crossEntropyLoss


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
    KLDLossRecoder = [[n_iters], list()]
    crossEntropyRecoder = [[n_iters], list()]
    BLEURecoder = [[n_iters], list()]
    GaussianRecoder = [[n_iters], list()]
    KLDWeightRecoder = [[n_iters], list()]
    tfRecoder = [[n_iters], list()]

    use_TF_ratio = teacher_forcing_ratio
    use_KLD_Weight = KLD_weight

    monotonicStartIter = 190001
    monotonicFinalIter = 300000
    TFDecadeStartIter = 150001
    TFDecadeFinalIter = 300000

    use_monotonic = False
    use_cyclical = True

    cyclicalStartIter = 50000
    cyclicalInternal = 5000

    monotonicKLD_Weight_change = (0.6 - KLD_weight) / ((monotonicFinalIter - monotonicStartIter) // print_every)
    TF_Decade_Change = -(teacher_forcing_ratio - 0.8) / ((TFDecadeFinalIter - TFDecadeStartIter) // print_every)
    cycicalKLD_Weight_change = (0.6 - KLD_weight) / (cyclicalInternal//print_every)

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

    # lower_loss = 9999
    best_BLEU_score = 0.7
    best_encoder_weight = copy.deepcopy(encoder.state_dict())
    best_decoder_weight = copy.deepcopy(decoder.state_dict())
    best_hiddenLinear_weight = copy.deepcopy(hiddenLinear.state_dict())
    best_cellLinear_weight = copy.deepcopy(cellLinear.state_dict())
    best_conditionEmbedding_weight = copy.deepcopy(conditionEmbedding.state_dict())

    best_Gaussian_score = 0.3
    best_encoder_weight_gaussian = copy.deepcopy(encoder.state_dict())
    best_decoder_weight_gaussian = copy.deepcopy(decoder.state_dict())
    best_hiddenLinear_weight_gaussian = copy.deepcopy(hiddenLinear.state_dict())
    best_cellLinear_weight_gaussian = copy.deepcopy(cellLinear.state_dict())
    best_conditionEmbedding_weight_gaussian = copy.deepcopy(conditionEmbedding.state_dict())

    tempCyclicalInterval = 0
    for iter in range(1, n_iters + 1):
        training_pair = training_pairs[iter - 1]
        input_tensor = training_pair[0]
        target_tensor = training_pair[0]
        condition_tensor = training_pair[1]

        loss, encoderOut, decoderOut, hiddenLinearOut, cellLinearOut, conditionEmbeddingOut, KLDLossValue, crossEntropyLoss = train(
            input_tensor, target_tensor, condition_tensor, encoder,
            decoder, hiddenLinear, cellLinear, conditionEmbedding, encoder_optimizer, decoder_optimizer,
            linear_hidden_optimizer, linear_cell_optimizer, embedding_optimizeer, criterion,
            use_KLD_Weight, use_TF_ratio)
        print_loss_total += loss
        plot_loss_total += loss

        # BLEUScoreEvery = evaluateBLEU(encoderOut, decoderOut, hiddenLinearOut, cellLinearOut, conditionEmbeddingOut,
        #                               condEmbedding_size, show=False)
        # GaussianScoreEvery = evaluateGaussian(decoderOut, hiddenLinearOut, cellLinearOut, conditionEmbeddingOut,
        #                                       condEmbedding_size, latent_size, condi_size, show=False)

        KLDLossRecoder[1].append(KLDLossValue.item())
        crossEntropyRecoder[1].append(crossEntropyLoss.item())

        KLDWeightRecoder[1].append(use_KLD_Weight)
        tfRecoder[1].append(use_TF_ratio)

        if iter % print_every == 0:
            print("-" * 50)
            print(f"print_loss_total:{print_loss_total},print_every:{print_every}, KLDLossValue:{KLDLossValue}")
            print_loss_avg = print_loss_total / print_every
            BLEUScore = evaluateBLEU(encoderOut, decoderOut, hiddenLinearOut, cellLinearOut, conditionEmbeddingOut,
                                     condEmbedding_size)
            GaussianScore = evaluateGaussian(decoderOut, hiddenLinearOut, cellLinearOut, conditionEmbeddingOut,
                                             condEmbedding_size, latent_size, condi_size)
            BLEURecoder[1].append(BLEUScore)
            GaussianRecoder[1].append(GaussianScore)
            BLEURecoder[0].append(iter)
            GaussianRecoder[0].append(iter)
            if BLEUScore >= best_BLEU_score and GaussianScore >= best_Gaussian_score:
                best_encoder_weight = copy.deepcopy(encoderOut.state_dict())
                best_decoder_weight = copy.deepcopy(decoderOut.state_dict())
                best_hiddenLinear_weight = copy.deepcopy(hiddenLinearOut.state_dict())
                best_cellLinear_weight = copy.deepcopy(cellLinearOut.state_dict())
                best_conditionEmbedding_weight = copy.deepcopy(conditionEmbeddingOut.state_dict())
                # best_BLEU_score = BLEUScore
                print("BLEU Score UP! Save Model!")

            if GaussianScore > best_Gaussian_score:
                best_encoder_weight_gaussian = copy.deepcopy(encoderOut.state_dict())
                best_decoder_weight_gaussian = copy.deepcopy(decoderOut.state_dict())
                best_hiddenLinear_weight_gaussian = copy.deepcopy(hiddenLinearOut.state_dict())
                best_cellLinear_weight_gaussian = copy.deepcopy(cellLinearOut.state_dict())
                best_conditionEmbedding_weight_gaussian = copy.deepcopy(conditionEmbeddingOut.state_dict())
                # best_Gaussian_score = GaussianScore
                print("Gaussian_score Score UP! Save Model!")

            print_loss_total = 0
            print('%s (%d %d%%) %.4f' % (timeSince(start, iter / n_iters),
                                         iter, iter / n_iters * 100, print_loss_avg))
            if use_monotonic:
                if iter >= monotonicStartIter and iter <= monotonicFinalIter:
                    use_KLD_Weight += monotonicKLD_Weight_change
                    print(f"Change Use_KLD_Weight to {use_KLD_Weight} ")
            elif use_cyclical:
                if iter >= cyclicalStartIter:
                    if iter % 10000 == 0:
                        use_KLD_Weight = KLD_weight
                        tempCyclicalInterval = cyclicalInternal
                    else:
                        if tempCyclicalInterval >= 0:
                            use_KLD_Weight += cycicalKLD_Weight_change
                            tempCyclicalInterval -= print_every
                            print(f"Change Use_KLD_Weight to {use_KLD_Weight} ")
            if iter >= TFDecadeStartIter and iter <= TFDecadeFinalIter:
                use_TF_ratio += TF_Decade_Change
                print(f"Change Use_TF_ratio to {use_TF_ratio} ")


        # TODO: Handle model save here(maybe store parameter and final output great model and save outside)
    encoder.load_state_dict(best_encoder_weight)
    decoder.load_state_dict(best_decoder_weight)
    hiddenLinear.load_state_dict(best_hiddenLinear_weight)
    cellLinear.load_state_dict(best_cellLinear_weight)
    conditionEmbedding.load_state_dict(best_conditionEmbedding_weight)
    gaussianWeightSet = (best_encoder_weight_gaussian, best_decoder_weight_gaussian,
                         best_hiddenLinear_weight_gaussian, best_cellLinear_weight_gaussian,
                         best_conditionEmbedding_weight_gaussian)
    recoderSet = ( KLDLossRecoder,crossEntropyRecoder,BLEURecoder,GaussianRecoder,KLDWeightRecoder,tfRecoder)
    return encoder, decoder, hiddenLinear, cellLinear, conditionEmbedding, gaussianWeightSet, recoderSet


encoder1 = EncoderRNN(vocab_size, hidden_size + condEmbedding_size, latent_size).to(
    device)  # input_size, hidden_size, latent_size
decoder1 = DecoderRNN(hidden_size + condEmbedding_size, vocab_size).to(device)  # hidden_size, output_size
hiddenLinear1 = hiddenCellLinear(latent_size + condEmbedding_size, hidden_size + condEmbedding_size).to(device)
cellLinear1 = hiddenCellLinear(latent_size + condEmbedding_size, hidden_size + condEmbedding_size).to(device)
conditionEmedding1 = ConditionEmbegging(condi_size, condEmbedding_size).to(device)  # condi_size, condEmbedding_size

encoderFinal, decoderFinal, hiddenLinearFinal, cellLinearFinal, conditionEmbeddingFinal, gaussianWeightSetFinal, recoderSet = trainIters(
    encoder1, decoder1, hiddenLinear1, cellLinear1, conditionEmedding1, n_iters=300000, print_every=1000,
    learning_rate=LR)
# encoder, decoder, hiddenLinear, cellLinear, conditionEmbedding, n_iters, print_every=1000, plot_every=100, learning_rate=0.01

# Save Best model
torch.save(encoderFinal.state_dict(), 'modelWeight/0815Test17/encoderFinal_weight1.pth')
torch.save(decoderFinal.state_dict(), 'modelWeight/0815Test17/decoderFinal_weight1.pth')
torch.save(hiddenLinearFinal.state_dict(), 'modelWeight/0815Test17/hiddenLinearFinal_weight1.pth')
torch.save(cellLinearFinal.state_dict(), 'modelWeight/0815Test17/cellLinearFinal_weight1.pth')
torch.save(conditionEmbeddingFinal.state_dict(), 'modelWeight/0815Test17/conditionEmbeddingFinal_weight1.pth')

torch.save(gaussianWeightSetFinal[0], 'modelWeight/0815Test17/encoderFinal_weight1(Gaussian).pth')
torch.save(gaussianWeightSetFinal[1], 'modelWeight/0815Test17/decoderFinal_weight1(Gaussian).pth')
torch.save(gaussianWeightSetFinal[2], 'modelWeight/0815Test17/hiddenLinearFinal_weight1(Gaussian).pth')
torch.save(gaussianWeightSetFinal[3], 'modelWeight/0815Test17/cellLinearFinal_weight1(Gaussian).pth')
torch.save(gaussianWeightSetFinal[4], 'modelWeight/0815Test17/conditionEmbeddingFinal_weight1(Gaussian).pth')

##Save Training & Testing Accuracy Result
with open('modelWeight/0815Test17/recoderSet.pickle', 'wb') as f:
    pickle.dump(recoderSet, f)
