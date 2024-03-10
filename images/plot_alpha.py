
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import scienceplots
dataset = "all"
arch = "all"
data = pd.read_csv(f"./images/alpha_all.csv", header=None, sep = "\t")
x  = [1000, 3000, 10000,30000,100000]

fig,ax = plt.subplots()
fig.set_size_inches(10,8)
plt.style.use("science")
cs = sns.color_palette("muted")
plt.errorbar(x, data[0],  yerr =data[1],  linestyle="-", linewidth=4, label ='VGG11-CIFAR10 ')
plt.errorbar(x, data[2],  yerr =data[3],  linestyle="-", linewidth=4, label ='ResNet18-CIFAR10 ')
plt.errorbar(x, data[4],  yerr =data[5],  linestyle="--", linewidth=4, label ='VGG11-CIFAR100 ')
plt.errorbar(x, data[6],  yerr =data[7],  linestyle="--", linewidth=4, label ='ResNet18-CIFAR100 ')

plt.ylim(0,100)
plt.legend(fontsize=30)
plt.ylabel("Test Accuracy", fontsize=40)
plt.xlabel("$\\alpha$", fontsize=40)
plt.xticks(fontsize=35)
plt.xscale("log")
plt.xticks([r for r in x], x)
plt.yticks(fontsize=35)
plt.tight_layout()
plt.savefig(f"./images/alpha_{dataset}_{arch}.pdf")
plt.show()