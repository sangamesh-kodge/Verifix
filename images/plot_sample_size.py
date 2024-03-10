
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import scienceplots
dataset = "cifar100"
arch = "all"
data = pd.read_csv(f"./images/sample_size_all.csv", header=None, sep = "\t")
x  = [100, 200, 500,1000,5000]


fig,ax = plt.subplots()
fig.set_size_inches(10,8)
plt.style.use("science")
cs = sns.color_palette("muted")
if dataset=="cifar10":
    plt.errorbar(x, data[0],  yerr =data[1],  linestyle="-", linewidth=4,  label ='VGG11 ')
    plt.errorbar(x, data[2],  yerr =data[3],  linestyle="-", linewidth=4, label ='ResNet18 ')
    plt.ylim(70,90)
else:
    plt.errorbar(x, data[4],  yerr =data[5],  linestyle="-",  linewidth=4, label ='VGG11 ')
    plt.errorbar(x, data[6],  yerr =data[7],  linestyle="-", linewidth=4, label ='ResNet18 ')
    plt.ylim(30,62)

plt.legend(fontsize=40)
plt.ylabel("Test Accuracy", fontsize=40)
plt.xlabel("Number of Samples", fontsize=40)
plt.xticks(fontsize=30)
plt.xscale("log")
plt.xticks([r for r in x], x)
plt.yticks(fontsize=30)
plt.tight_layout()
plt.savefig(f"./images/sample_size_{dataset}_{arch}.pdf")
plt.show()