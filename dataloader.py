import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class WordSet:
    def __init__(self):
        self.wordBank = self._readFile() #(1227*4)
        self.SOSToken = 0
        self.EOSToken = 1

    def _readFile(self):
        with open("./data/train.txt", 'r') as f:
            wordBank  = list()
            lines = f.readlines()
            for line in lines:
                wordBank.append(line.split())
        return wordBank
    def _letter2Num(self, letter):
        #ord('a'):97
        #SOS:0/EOS:1
        return ord(letter)- 97 + 2

    def _word2Tensor(self, word):
        # one word into a tensor contained number and plus eos token
        toTensorList = list()
        for letter in word:
            toTensorList.append([self._letter2Num(letter)])
        toTensorList.append([self.EOSToken])
        return torch.cuda.LongTensor(toTensorList)

    def getWordPair(self):
        #tense: {0:sp, 1:tp, 2:pg, 3:p}
        pairsList = list()
        for wordLine in self.wordBank:
            for tense, word in enumerate(wordLine):
                pairsList.append([self._word2Tensor(word), torch.cuda.LongTensor([tense])])
        return pairsList

class WordTestSet:
    def __init__(self):
        self.wordBank = self._readFile() #(1227*4)
        self.SOSToken = 0
        self.EOSToken = 1

    def _readFile(self):
        with open("./data/test.txt", 'r') as f:
            wordBank  = list()
            lines = f.readlines()
            for line in lines:
                wordBank.append(line.split())
        return wordBank
    def _letter2Num(self, letter):
        #ord('a'):97
        #SOS:0/EOS:1
        return ord(letter)- 97 + 2

    def _word2Tensor(self, word):
        # one word into a tensor contained number and plus eos token
        toTensorList = list()
        for letter in word:
            toTensorList.append([self._letter2Num(letter)])
        toTensorList.append([self.EOSToken])
        return torch.cuda.LongTensor(toTensorList)

    def getWordPair(self):
        #tense: {0:sp, 1:tp, 2:pg, 3:p}
        # sp -> p
        # sp -> pg
        # sp -> tp
        # sp -> tp
        # p  -> tp
        # sp -> pg
        # p  -> sp
        # pg -> sp
        # pg -> p
        # pg -> tp
        tenseConvert = [[0,3],[0,2],[0,1],[0,1],[3,1],[0,2],[3,0],[2,0],[2,3],[2,1]]
        pairsList = list()
        for wordLine, tenseTuple in zip(self.wordBank,tenseConvert):
            pairsList.append([[self._word2Tensor(wordLine[0]), torch.cuda.LongTensor([tenseTuple[0]]), wordLine[0]],
                            [wordLine[1], torch.cuda.LongTensor([tenseTuple[1]])]])
        return pairsList
    @staticmethod
    def vec2word(vecList):
        # input: [[tensor([3])],[tensor([2])],[tensor([3])],[EOS]]
        # SOS_token = 0
        # EOS_token = 1
        result = list()
        for letter in vecList:
            letter = int(letter.item())
            if letter != 0 and letter != 1:
                result.append(chr(letter+95))
        return "".join(result)

class WordGaussianTestSet:
    def __init__(self, latent_size, condi_size):
        self.latent_size = latent_size
        self.condi_size = condi_size
    def getGaussianLatent(self):
        result = list()
        for i in range(100):
            result.append(torch.randn(1, 1, self.latent_size, device=device))
        return result
    def getTense(self):
        result = list()
        for i in range(self.condi_size):
            result.append(torch.cuda.LongTensor([i]))
        return result


if __name__ == '__main__':
    # a = WordSet()
    # pairs = a.getWordPair()
    # print()
    # b = WordTestSet()
    # pairs = b.getWordPair()
    # print()
    print(WordTestSet.vec2word([[torch.Tensor([2])],[torch.Tensor([6])],[torch.Tensor([26])]]))