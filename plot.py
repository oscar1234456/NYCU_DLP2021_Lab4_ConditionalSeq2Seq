##
import matplotlib.pyplot as plt
import pickle


##
with open('modelWeight/0815Test14/recoderSet.pickle', 'rb') as f:
    recoderSet = pickle.load(f)

##
epoch = recoderSet[0][0][0]
KLDLoss = recoderSet[0][1]
crossEntropy = recoderSet[1][1]
BLEUScore = recoderSet[2][1]
GaussianScore = recoderSet[3][1]
KLDWeight = recoderSet[4][1]
tf = recoderSet[5][1]

##
newKLDLoss = list()
newCrossEntropy = list()
for i in recoderSet[2][0][1:]:
    newKLDLoss.append(KLDLoss[i-1])
    newCrossEntropy.append(crossEntropy[i-1])

##
iter = [x+1 for x in range(epoch)]
# plt.xlim(-5,305)
# plt.ylim(64,102)
ax1 = plt.subplot()
lns1 = ax1.plot(recoderSet[2][0][1:], newKLDLoss, 'b-', label="KLD")
lns2 = ax1.plot(recoderSet[2][0][1:], newCrossEntropy, color = "orange",lineStyle = "-", label="CrossEntropy")
plt.xlabel("Iteration(s) Sample Freq: 1,000/Iter.")
plt.ylabel("Loss")
ax2 = ax1.twinx()
lns3 = ax2.plot(iter, KLDWeight, 'r--', label="KLD_Weight")
lns4 = ax2.plot(iter, tf, lineStyle='--',color="plum", label="Teacher ratio")
lns5 = ax2.scatter(recoderSet[2][0][1:], BLEUScore,color="green",marker = "." ,label="BLEU4-score")
lns6 = ax2.scatter(recoderSet[2][0][1:], GaussianScore,color="brown",marker = "." ,label="Gaussian-score")

# lns = [lns1,lns2,lns3,lns4,lns5,lns6]
# labs = ["KLD","CrossEntropy","KLD_Weight","Teacher ratio","BLEU4-score","Gaussian-score"]
plt.ylabel("Score/Weight")
ax2.legend(loc='center left')
ax1.legend(loc='upper left')
plt.title("Training loss/ratio curve (Monotonic)")
plt.show()