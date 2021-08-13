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


if __name__ == '__main__':
    a = WordSet()
    pairs = a.getWordPair()
    print()