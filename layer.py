import torch
import torch.nn as nn
import torch.nn.functional as F
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#Encoder
class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, latent_size):
        #LSTM:(L=1, N=1, H_in = hidden_size)
        #hidden_size is included Hidden len + Condition code len
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(input_size, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size)

        self.linear_hidden_mean = nn.Linear(hidden_size, latent_size)
        self.linear_hidden_logVar = nn.Linear(hidden_size, latent_size)

        self.linear_cell_mean = nn.Linear(hidden_size, latent_size)
        self.linear_cell_logVar = nn.Linear(hidden_size, latent_size)

    def forward(self, input, hidden):
        #hidden must the tuple :(h_0, cell_0)
        # hidden_size is included Hidden len + Condition code len
        #h_0: (1,1,hidden_size)
        #cell_0:(1,1,hidden_size)
        embedded = self.embedding(input)
        for wordVec in embedded:  #iteration all letter in this word
            resizeWordVec = wordVec.view(1, 1, -1)
            output, hidden = self.lstm(resizeWordVec, hidden)  #output is unuseful
        finalHidden = hidden[0]
        finalCell = hidden[1]
        hidden_mean = self.linear_hidden_mean(finalHidden)
        hidden_logVar = self.linear_hidden_logVar(finalHidden)
        cell_mean = self.linear_cell_mean(finalCell)
        cell_logVar = self.linear_cell_logVar(finalCell)
        return hidden_mean, hidden_logVar, cell_mean, cell_logVar

    def initHidden(self, conditionCode_size):
        return torch.zeros(1, 1, self.hidden_size-conditionCode_size, device=device)
    def initCell(self,conditionCode_size):
        return torch.zeros(1, 1, self.hidden_size-conditionCode_size, device=device)

#Decoder
class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size):
        #output_size = vocab_size
        # hidden_size is included Hidden len + Condition code len (after linear)
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        # self.softmax = nn.LogSoftmax(dim=1) #we use crossEntropyLoss

    def forward(self, input, hidden):
        # hidden must the tuple :(h_0, cell_0)
        # hidden_size is included Hidden len + Condition code len
        # h_0: (1,1,hidden_size)
        # cell_0:(1,1,hidden_size)
        output = self.embedding(input).view(1, 1, -1)
        output = F.relu(output) #TODO:Evaluate
        output, hidden = self.lstm(output, hidden) #hidden is unuseful
        output = self.out(output[0]) #LSTM output:(1,1,hidden_size)
        return output, hidden

    def initHidden(self, conditionCode_size):
        return torch.zeros(1, 1, self.hidden_size - conditionCode_size, device=device)
    def initCell(self,conditionCode_size):
        return torch.zeros(1, 1, self.hidden_size-conditionCode_size, device=device)

class hiddenCellLinear(nn.Module):
    def __init__(self, input_size, output_size):
        # input: latent code+condition (latent_size + condEmbedding_size)
        # output: hidden_size for next LSTM
        super(hiddenCellLinear, self).__init__()
        self.out = nn.Linear(input_size, output_size)

    def forward(self, input):
        output = self.out(input)
        return output

class ConditionEmbegging(nn.Module):
    def __init__(self, condi_size, condEmbedding_size):
        # input: condi_size means the number of condition class (Default:4)
        # output: condEmbedding_size means the size we want to convert into
        super(ConditionEmbegging, self).__init__()
        self.embedding = nn.Embedding(condi_size, condEmbedding_size)

    def forward(self, input):
        #input must a LongTensor (condition only has 1 dim)
        embedded = self.embedding(input)
        return embedded

# if __name__ == "__main__":
#     test = EncoderRNN(1,2,3)
#     print(test)