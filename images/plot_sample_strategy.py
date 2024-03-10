
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import scienceplots

dataset = "cifar100"
arch = "all"
data = pd.read_csv(f"./images/sample_strategy.csv", header=None, sep = "\t")
x  = ["Uniform", "0%", "10%","25%","50%", "Expert"]
methods = ["VGG11", "ResNet18"]
# Set position of bar on X axis

num_layers = len(data)
num_methods = len(methods)

barWidth = 0.4
br = [np.arange(num_layers)]
for i in range(num_methods-1):
    br.append([val+barWidth for val in br[-1] ])


fig,ax = plt.subplots()
fig.set_size_inches(10,8)
plt.style.use("science")
cs = sns.color_palette("muted")
if dataset == "cifar10":
    plt.bar(br[0], data[0],  yerr =data[1],  width = barWidth, edgecolor ='grey',  label ='VGG11 ')
    plt.bar(br[1], data[2],  yerr =data[3],    width = barWidth, edgecolor ='grey', label ='ResNet18 ')
    plt.ylim(82,90)
else:
    plt.bar(br[0], data[4],  yerr =data[5],    width = barWidth, edgecolor ='grey', label ='VGG11 ')
    plt.bar(br[1], data[6],  yerr =data[7],  width = barWidth, edgecolor ='grey',  label ='ResNet18 ')
    plt.ylim(55,65)
plt.legend(fontsize=35)
plt.ylabel("Test Accuracy", fontsize=40)
plt.xlabel("Sampling Strategy", fontsize=40)
plt.xticks([r for r in br[0]], x, rotation=60, fontsize=35)
plt.yticks(fontsize=35)
plt.tight_layout()
plt.savefig(f"./images/sample_selection_{dataset}_{arch}.pdf")
plt.show()
