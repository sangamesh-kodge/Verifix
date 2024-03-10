
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import scienceplots

# Set variables
dataset = "cifar100"
arch = "resnet18"

#Load architecute file.
data = pd.read_csv(f"./images/{arch}_compute.csv",  sep = "\t")


def conv_forward(data, dataset, samples):
    # compute = data.c1*data.c2 *data.k * data.h * data.k
    layer_wise_compute =  data.c1*data.c2 *data.k * data.h * data.k
    if dataset == "cifar100":
        layer_wise_compute[len(layer_wise_compute)-1] = 10*layer_wise_compute[len(layer_wise_compute)-1]
    compute= sum(layer_wise_compute)
    return samples*compute
single_sample_comute = conv_forward(data, dataset, 1)
clean_samples = 50000
clean_epochs = 350
compute_clean_train = 3*clean_samples*clean_epochs*single_sample_comute

retrain_samples = 37500
retrain_epochs = 350
compute_retrain_train = 3*retrain_samples*retrain_epochs*single_sample_comute

sam_samples = 50000
sam_epochs = 350
compute_sam_train = 6*sam_samples*sam_epochs*single_sample_comute


finetune_samples = 1000
finetune_epochs = 50
compute_finetune_train = 3*finetune_samples*finetune_epochs*single_sample_comute

our_samples = 1000
curvature_epochs = 10
fin = data.c1 *data.k* data.k 
fout = data.c2 
# Trusted Data curvature Compute (5 epochs)
compute_out_train = 3*clean_samples*curvature_epochs*single_sample_comute
# Representation estimation
compute_out_train += our_samples*single_sample_comute
# SVD
compute_out_train += sum(our_samples*fin*fin)
# Mr and Mf Calculation
compute_out_train += sum(2*fin**3)
# Weight Projection
compute_out_train += sum(fin*fin*fout)


compute_precent_dict = { 
    # "Finetune":compute_finetune_train,  
    "Verifix":compute_out_train, 
    "Retrain":compute_retrain_train, 
    "Vanilla":compute_clean_train, 
    "SAM":compute_sam_train, 
    }


# Set position of bar on X axis
barWidth = 0.8
br = np.arange(len(compute_precent_dict))


fig,ax = plt.subplots()
fig.set_size_inches(10,8)
plt.style.use("science")
cs = sns.color_palette("muted")
plt.bar(br, compute_precent_dict.values(),  width = barWidth, edgecolor ='grey')
value = list(compute_precent_dict.values())
for i in range(len(br)):
    plt.text(br[i],value[i]*(1.8),f"{value[i] /value[0]:0.1f}x",   ha = 'center',  fontsize=40, rotation=90)
plt.ylim(10e10,10e20)
# plt.legend( fontsize=30)
plt.ylabel("Compute Complexity \n Log Scale", fontsize=40)
plt.xticks(br,
        compute_precent_dict.keys(), rotation=45)
plt.xticks( fontsize=40)
plt.yticks(fontsize=40)
plt.yscale("log")
plt.tight_layout()
plt.savefig(f"./images/compute_{dataset}_{arch}.pdf")
plt.show()