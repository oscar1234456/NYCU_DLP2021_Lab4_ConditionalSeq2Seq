##import
import torch
from layer import EncoderRNN, DecoderRNN, hiddenCellLinear, ConditionEmbegging
from testScore import evaluateBLEU, evaluateGaussian
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
##
# Testing Parameters (1)
# hidden_size = 128
# vocab_size = 28
# latent_size = 32
# condEmbedding_size = 8
# condi_size = 4

# Testing Parameters (2)
hidden_size = 256
vocab_size = 28
latent_size = 32
condEmbedding_size = 8
condi_size = 4

# Testing Parameters (3)
# hidden_size = 256
# vocab_size = 28
# latent_size = 256
# condEmbedding_size = 8
# condi_size = 4


## define model
encoder1 = EncoderRNN(vocab_size, hidden_size + condEmbedding_size, latent_size).to(device)  # input_size, hidden_size, latent_size
decoder1 = DecoderRNN(hidden_size + condEmbedding_size, vocab_size).to(device)  # hidden_size, output_size
hiddenLinear1 = hiddenCellLinear(latent_size + condEmbedding_size, hidden_size + condEmbedding_size).to(device)
cellLinear1 = hiddenCellLinear(latent_size + condEmbedding_size, hidden_size + condEmbedding_size).to(device)
conditionEmbedding1 = ConditionEmbegging(condi_size, condEmbedding_size).to(device)  # condi_size, condEmbedding_size.

##load model weight
encoder1.load_state_dict(torch.load('modelWeight/0814Test6/encoderFinal_weight1.pth'))
decoder1.load_state_dict(torch.load('modelWeight/0814Test6/decoderFinal_weight1.pth'))
hiddenLinear1.load_state_dict(torch.load('modelWeight/0814Test6/hiddenLinearFinal_weight1.pth'))
cellLinear1.load_state_dict(torch.load('modelWeight/0814Test6/cellLinearFinal_weight1.pth'))
conditionEmbedding1.load_state_dict(torch.load('modelWeight/0814Test6/conditionEmbeddingFinal_weight1.pth'))

##
evaluateBLEU(encoder1, decoder1, hiddenLinear1, cellLinear1, conditionEmbedding1,condEmbedding_size)
#encoder:EncoderRNN, decoder:DecoderRNN, hiddenLinear:hiddenCellLinear,cellLinear:hiddenCellLinear, conditionEmbedding:ConditionEmbegging, condEmbedding_size
evaluateGaussian(decoder1, hiddenLinear1, cellLinear1, conditionEmbedding1, condEmbedding_size, latent_size, condi_size)
#decoder:DecoderRNN, hiddenLinear:hiddenCellLinear,cellLinear:hiddenCellLinear, conditionEmbedding:ConditionEmbegging, condEmbedding_size, laten_size, condi_size