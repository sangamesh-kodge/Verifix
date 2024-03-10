import torch
from torch import nn
from torch.nn import functional as F
from collections import OrderedDict
from copy import deepcopy
from matplotlib import pyplot as plt
from PIL import Image
import numpy as np
from numpy import pi
import random
from tqdm import tqdm
import seaborn as sns
import copy
import time 
import os
import argparse

class Linear(nn.Module):
    def __init__(self, in_feature=2, hidden_features=40, num_classes=2, layers = 10):
        super(Linear, self).__init__()
        self.fc1 = nn.Linear(in_features=in_feature, out_features=hidden_features)
        self.bn1 = nn.BatchNorm1d(hidden_features)
        self.fc = torch.nn.ModuleList([nn.Linear(in_features=hidden_features, out_features=hidden_features) for _ in range(layers-2)])
        self.bn = torch.nn.ModuleList([nn.BatchNorm1d(hidden_features) for _ in range(layers-2) ] )
        self.classifier  = nn.Linear(in_features=hidden_features, out_features=num_classes)
        self.layers = layers                   

    def forward(self,x):
        x = F.relu(self.bn1(self.fc1(x)))
        for fc, bn in zip(self.fc, self.bn):
            x = F.relu(bn(fc(x)))
        return F.log_softmax(self.classifier(x), 1)
    
    def get_activations(self, x):
        index = 2
        act = {"pre":OrderedDict(), "post":OrderedDict()}
        act["pre"]["fc1"] = deepcopy(x.clone().detach().cpu().numpy())
        x = self.fc1(x)
        act["post"]["fc1"] = deepcopy(x.clone().detach().cpu().numpy())
        x = F.relu(self.bn1(x))
        for fc, bn in zip(self.fc, self.bn):
            act["pre"][f"fc{index}"] = deepcopy(x.clone().detach().cpu().numpy())
            x = fc(x)
            act["post"][f"fc{index}"] = deepcopy(x.clone().detach().cpu().numpy())
            x = F.relu(bn(x))   
            index+=1   
        act["pre"]["classifier"] = deepcopy(x.clone().detach().cpu().numpy())
        x = self.classifier(x)
        act["post"]["classifier"] = deepcopy(x.clone().detach().cpu().numpy())
        return act 
    
    def project_weights(self, projection_mat_dict):
        index = 2
        self.fc1.weight.data = torch.mm(projection_mat_dict["post"]["fc1"].transpose(0,1), torch.mm(self.fc1.weight.data, projection_mat_dict["pre"]["fc1"].transpose(0,1)))
        self.fc1.bias.data = torch.mm(self.fc1.bias.data.unsqueeze(0), projection_mat_dict["post"]["fc1"]).squeeze(0)
        for fc in self.fc:
            fc.weight.data = torch.mm(projection_mat_dict["post"][f"fc{index}"].transpose(0,1), torch.mm(fc.weight.data, projection_mat_dict["pre"][f"fc{index}"].transpose(0,1)))
            fc.bias.data = torch.mm(fc.bias.data.unsqueeze(0), projection_mat_dict["post"][f"fc{index}"]).squeeze(0)
            index+=1
        self.classifier.weight.data =  torch.mm(self.classifier.weight.data, projection_mat_dict["pre"]["classifier"].transpose(0,1))
        return 

def get_dataset(train_samples_per_class, val_samples_per_class, test_samples_per_class, intrinsic_randomness=False):
    def get_data(N):
        theta =  np.sqrt(np.random.rand(N))*4*pi # np.linspace(0,2*pi,100) #
        r_a =(2*theta + pi)/10
        data_a = np.array([np.cos(theta)*r_a, np.sin(theta)*r_a]).T
        if intrinsic_randomness:
            x_a = data_a + np.random.randn(N,2) / 10
        else:
            x_a = data_a
        r_b = (-2*theta - pi)/10
        data_b = np.array([np.cos(theta)*r_b, np.sin(theta)*r_b]).T
        if intrinsic_randomness:
            x_b = data_b + np.random.randn(N,2) / 10
        else:
            x_b = data_b 
        data_tensor = torch.tensor(np.append(x_a, x_b,axis=0), dtype=torch.float32)
        target_tensor = torch.tensor(np.append(np.zeros((N,1)), np.ones((N,1)),axis=0), dtype=torch.long).squeeze(1)
        return (data_tensor, target_tensor)
    train_data = get_data(train_samples_per_class)
    val_data = get_data(val_samples_per_class)
    test_data = get_data(test_samples_per_class)
    return train_data, val_data, test_data
    
def get_corrupt_targets(data, fraction_corrupt):
    num_of_data_points = data[0].shape[0]
    num_of_mislabeled = int(fraction_corrupt * num_of_data_points)
    r=np.arange(num_of_data_points)
    np.random.shuffle(r)
    index_list = r[:num_of_mislabeled].tolist()
    targets = copy.deepcopy(data[1])
    for idx in index_list:
        targets[idx] =  torch.ones_like( targets[idx] ) * random.choice([val for val in range(2) if val != targets[idx] ]) 
    return index_list, targets

def train(model, train_loader, device, optimizer, schedular=None):
    model.train()
    total_loss = 0
    for data, target in train_loader:
        optimizer.zero_grad()
        data, target = data.to(device), target.to(device)
        log_prob = model(data)
        loss = F.nll_loss(log_prob, target)
        total_loss+= loss.detach().item()/len(train_loader.dataset)
        loss.backward()
        optimizer.step()
        if schedular is not None:
            schedular.step(loss)
    return total_loss
        

def test(model, test_loader, device):
    model.eval()
    total_loss = 0
    correct = 0
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        log_prob = model(data)
        pred = log_prob.argmax(dim=1, keepdim=True)
        loss = F.nll_loss(log_prob, target)
        total_loss+= loss.detach().item()/len(test_loader.dataset)
        correct += pred.eq(target.view_as(pred)).sum().item()
    acc = 100*correct/len(test_loader.dataset)
    return total_loss, acc

def plot_raw_data(data, save_path ="data.pdf"):
    fig,ax = plt.subplots()
    fig.set_size_inches(10,10)
    plt.style.use("seaborn-v0_8-talk")
    cs = sns.color_palette("muted")
    plt.scatter(data[0][:,0][(data[1]==0)], data[0][:,1][(data[1]==0)],color= lighten_color("r",1.0),label="Class 1")
    plt.scatter(data[0][:,0][(data[1]==1)], data[0][:,1][(data[1]==1)], color= lighten_color("skyblue",1.0),label="Class 2")
    plt.xticks(fontsize=30)
    plt.yticks(fontsize=30)
    plt.xlabel("Feature 1", fontsize=40)
    plt.ylabel("Feature 2", fontsize=40)
    plt.tight_layout()
    plt.savefig(save_path)
    # plt.show()
    print(f"Raw Data saved to path {save_path}")
    return 

def plot_decision_boundary(model, data, device,  save_dir, title, corrupt_targets=None):
    data_list = []
    x_points = np.arange(-3,3.005,0.01)
    y_points = np.arange(3,-3.005,-0.01)
    for x in x_points:
        for y in y_points:
            data_list.append(torch.tensor( (x,y) ) )
    data_tensor = torch.stack(data_list).float()
    predicted_class = []
    for samples in torch.split(data_tensor, 256):
        predicted_class.append(model(samples.to(device)).argmax(dim=1).long())
    predicted_class=torch.cat(predicted_class,0)
    color_lookup = [
                    torch.tensor([0.8, 0.8, 1.0]).to(device),
                    torch.tensor([0.8, 1.0, 0.8]).to(device),
                    torch.tensor([0.0, 0.0, 1.0]).to(device),
                    torch.tensor([0.0, 1.0, 0.0]).to(device),
                    torch.tensor([1.0, 0.0, 0.0]).to(device),
                    ]
    ind = 0
    decision_boundary = torch.zeros( (3, x_points.shape[0], y_points.shape[0])).to(device)
    # Plot decision boundary.
    for x, _ in enumerate(x_points):
        for y, _ in enumerate(y_points):
            decision_boundary[:, x, y ] = color_lookup[predicted_class[ind]]
            ind+=1
    # draw circles. 
    for i in range(data[0].shape[0]):
        x = data[0][i]
        y = data[1][i]
        x_i = np.round(x[0]/0.01).long()+301
        y_i = 301 - np.round(x[1]/0.01).long() # top left is  0 0  for image!
        for j in range(x_i-1, x_i+2):### To increase boldness
            for k in range(y_i-1, y_i+2):
                if j<601 and k<601:
                    if corrupt_targets is not None:
                        decision_boundary[:, j, k] =  color_lookup[2+corrupt_targets[i]]
                        # if y == corrupt_targets[i]:
                        #     decision_boundary[:, j, k] =  color_lookup[2+y]
                        # else:
                        #     decision_boundary[:, j, k] =  color_lookup[-1] # Color red if data mislabeled
                    else:
                        decision_boundary[:, j, k] =  color_lookup[2+y]                
    decision_boundary = (255 * decision_boundary).long().permute(2,1,0).cpu().numpy().astype(np.uint8)
    fig,ax = plt.subplots()
    fig.set_size_inches(10,8)
    plt.style.use("seaborn-v0_8-talk")
    cs = sns.color_palette("muted")

    plt.imshow( decision_boundary)
    x_ticks_positions = np.arange(0, x_points.shape[0]+0.5, 150.25)    
    x_ticks_labels = [str(x) for x in np.arange(-3,3.005,1.5)]
    plt.xticks(x_ticks_positions, x_ticks_labels, fontsize=30)
    y_ticks_positions = np.arange(0, y_points.shape[0]+0.5, 75.125)    
    y_ticks_labels = [str(x) for x in np.arange(3,-3.005,-0.75)]
    plt.yticks(y_ticks_positions, y_ticks_labels, fontsize=30)
    plt.xlabel("Feature 1", fontsize=40)
    plt.ylabel("Feature 2", fontsize=40)
    # plt.title(title)
    # plt.xticks()
    # plt.yticks(fontsize=30)
    plt.tight_layout()
    if not os.path.exists(f"{save_dir}/"):
        os.makedirs(f"{save_dir}/")
    plt.savefig(f"{save_dir}/demo_spiral_{title.split(':')[0].lower()}.pdf")
    # plt.show()
    print(f"Decision boundary save to path {save_dir}/demo_spiral_{title.split(':')[0].lower()}.pdf")
    return

# Plot data
def lighten_color(color, amount=0.5):
    """
    Lightens the given color by multiplying (1-luminosity) by the given amount.
    Input can be matplotlib color string, hex string, or RGB tuple.

    Examples:
    >> lighten_color('g', 0.3)
    >> lighten_color('#F034A3', 0.6)
    >> lighten_color((.3,.55,.1), 0.5)
    """
    import matplotlib.colors as mc
    import colorsys
    try:
        c = mc.cnames[color]
    except:
        c = color
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    return colorsys.hls_to_rgb(c[0], 1 - amount * (1 - c[1]), c[2])

def main():
    # dataset parameters
    train_samples_per_class = 250
    val_samples_per_class = 1000
    test_samples_per_class = 5000
    # training parameters
    epochs = 250
    lr = 1e-2
    momentum = 0.0
    nesterov = False
    gamma = 0.7
    batch_size = 512
    # model parameters
    hdim = 500
    layers = 10
    parser = argparse.ArgumentParser()
    # mislabel parameters
    parser.add_argument("--intrinsic-randomness", action="store_true", help="")
    parser.add_argument("--fraction-corrupt", default=0.1, type=float, help="")
    # Reproducability
    parser.add_argument("--seed", default=42, type=int, help="")
    args = parser.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = "cuda"
    kwargs = {
        'batch_size': batch_size,
        'num_workers': 1,
        'pin_memory': True,
        'shuffle': True
    }
    if args.intrinsic_randomness:
        save_name_extension = "noisy_spiral"
    else:
        save_name_extension = "clean_spiral"
    # Dataloaders
    data1, data2, data3 = get_dataset(train_samples_per_class, val_samples_per_class, test_samples_per_class, args.intrinsic_randomness)
    train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(data1[0], data1[1]),**kwargs)
    val_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(data2[0], data2[1]), **kwargs)
    test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(data3[0], data3[1]), **kwargs)
    ### Mislabel datapoints.
    corrupt_data_index, corrupt_targets = get_corrupt_targets(data1, fraction_corrupt = args.fraction_corrupt)
    corrupt_train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(copy.deepcopy(data1[0]), corrupt_targets),**kwargs)
    
    # Train Clean model
    clean_model = Linear(in_feature=2, hidden_features=hdim, num_classes=2, layers=layers).to(device)
    if os.path.exists(f"./pretrained_models/2DSpiral/{save_name_extension}{train_samples_per_class}_linh{hdim}l{layers}_seed{args.seed}.pt"):
        print("-"*40)
        print("Loading model trained on full clean data")
        print("-"*40)
        clean_model.load_state_dict(torch.load( f"./pretrained_models/2DSpiral/{save_name_extension}{train_samples_per_class}_linh{hdim}l{layers}_seed{args.seed}.pt" ))
        clean_test_loss, clean_test_acc = test(clean_model, test_loader, device)
        clean_test_metric = clean_test_acc
        print(f"Clean- test acc {clean_test_acc:.2f}")
        plot_decision_boundary(clean_model, data1, device,  save_dir = f"./images/{save_name_extension}{train_samples_per_class}_linh{hdim}l{layers}_corrupt{args.fraction_corrupt}_seed{args.seed}", title = f"Clean: Test acc-{clean_test_acc:.2f}" )
        
    else:
        print("-"*40)
        print("Clean data Training")
        print("-"*40)
        optimizer = torch.optim.SGD(clean_model.parameters(), lr = lr, momentum=momentum, nesterov=nesterov)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', factor=gamma)
        pbar = tqdm(range(1, epochs+1), desc='Clean Training')
        train_start_time = time.time()
        for epoch in pbar:
            train_loss = train(clean_model, train_loader, device, optimizer, scheduler)
            test_loss, test_acc = test(clean_model, val_loader, device)
            _, train_acc = test(clean_model, train_loader, device)
            pbar.set_description(f"Val(Train) Acc {test_acc:.2f}({train_acc:.2f})")    
        train_end_time = time.time()
        clean_test_loss, clean_test_acc = test(clean_model, test_loader, device)
        clean_test_metric = clean_test_acc
        print(f"Clean- test acc {clean_test_acc:.2f} train time {train_end_time-train_start_time}")
        plot_decision_boundary(clean_model, data1, device,  save_dir = f"./images/{save_name_extension}{train_samples_per_class}_linh{hdim}l{layers}_corrupt{args.fraction_corrupt}_seed{args.seed}", title = f"Clean: Test acc-{clean_test_acc:.2f}" )
        torch.save(clean_model.state_dict(), f"./pretrained_models/2DSpiral/{save_name_extension}{train_samples_per_class}_linh{hdim}l{layers}_seed{args.seed}.pt")
    
    corrupt_model =  Linear(in_feature=2, hidden_features=hdim, num_classes=2, layers=layers).to(device)
    if os.path.exists(f"./pretrained_models/2DSpiral/{save_name_extension}{train_samples_per_class}_linh{hdim}l{layers}_corrupt{args.fraction_corrupt}_seed{args.seed}.pt"):
        print("-"*40)
        print("Loading model trained on corrupt data")
        print("-"*40)
        corrupt_model.load_state_dict(torch.load( f"./pretrained_models/2DSpiral/{save_name_extension}{train_samples_per_class}_linh{hdim}l{layers}_corrupt{args.fraction_corrupt}_seed{args.seed}.pt"))
        corrupt_test_loss, corrupt_test_acc = test(corrupt_model, test_loader, device)
        corrupt_test_metric = corrupt_test_acc
        print(f"Corrupt- test acc {corrupt_test_acc:.2f}")
        plot_decision_boundary(corrupt_model, data1, device, save_dir = f"./images/{save_name_extension}{train_samples_per_class}_linh{hdim}l{layers}_corrupt{args.fraction_corrupt}_seed{args.seed}", title = f"Corrupt: Test acc-{corrupt_test_acc:.2f}", corrupt_targets=corrupt_targets)
    else:
        # Train Corrupt model
        print("-"*40)
        print("Corrupt data Training")
        print("-"*40)
        optimizer = torch.optim.SGD(corrupt_model.parameters(), lr = lr, momentum=momentum, nesterov=nesterov)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', factor=gamma)
        pbar = tqdm(range(1, epochs+1), desc='Corrupt Training')
        corrupt_start_time = time.time()
        for epoch in pbar:
            train_loss = train(corrupt_model, corrupt_train_loader, device, optimizer, scheduler)
            test_loss, test_acc = test(corrupt_model, val_loader, device)
            _, train_acc = test(corrupt_model, corrupt_train_loader, device)
            pbar.set_description(f"Val(Train) Acc {test_acc:.2f}({train_acc:.2f})")     
        corrupt_end_time = time.time()     
        corrupt_test_loss, corrupt_test_acc = test(corrupt_model, test_loader, device)
        corrupt_test_metric = corrupt_test_acc
        print(f"Corrupt- test acc {corrupt_test_acc:.2f} train time {corrupt_end_time-corrupt_start_time}")
        plot_decision_boundary(corrupt_model, data1, device, save_dir = f"./images/{save_name_extension}{train_samples_per_class}_linh{hdim}l{layers}_corrupt{args.fraction_corrupt}_seed{args.seed}", title = f"Corrupt: Test acc-{corrupt_test_acc:.2f}", corrupt_targets=corrupt_targets)
        torch.save(corrupt_model.state_dict(), f"./pretrained_models/2DSpiral/{save_name_extension}{train_samples_per_class}_linh{hdim}l{layers}_corrupt{args.fraction_corrupt}_seed{args.seed}.pt")
           

    print("-"*40)
    print("Verifix Search for best alpha_f and alpha_r on Train Set.")
    print("-"*40)
    val_loss, val_acc = test(corrupt_model, val_loader, device)
    base_metric = val_acc
    clean_data_index = np.array([int(val) for val  in np.arange( len(data1[0]) ) if val not in corrupt_data_index] )
    verifix_start_time = time.time()  
    retain_data = data1[0][clean_data_index]
    corrupt_model.eval()
    retain_act = corrupt_model.get_activations(retain_data.to(device))
    retain_mat_dict= {"pre":OrderedDict(), "post":OrderedDict()}
    retain_normalize_var_mat_dict = {"pre":OrderedDict(), "post":OrderedDict()}
    # Compute Ur
    for loc in retain_act.keys():
        for key in retain_act[loc].keys():
            activation = torch.Tensor(retain_act[loc][key]).to("cuda").transpose(0,1)
            U,S,Vh = torch.linalg.svd(activation, full_matrices=False)
            U = U.cpu().numpy()
            S = S.cpu().numpy()  
            retain_mat_dict[loc][key] = U
            retain_normalize_var_mat_dict[loc][key] = S**2 / (S**2).sum()
    forget_data = data1[0][corrupt_data_index]
    forget_act = corrupt_model.get_activations(forget_data.to(device))
    forget_mat_dict= {"pre":OrderedDict(), "post":OrderedDict()}
    forget_normalize_var_mat_dict = {"pre":OrderedDict(), "post":OrderedDict()}
    # Compute Uf
    for loc in forget_act.keys():
        for key in forget_act[loc].keys():
            activation = torch.Tensor(forget_act[loc][key]).to("cuda").transpose(0,1)
            U,S,Vh = torch.linalg.svd(activation, full_matrices=False)
            U = U.cpu().numpy()
            S = S.cpu().numpy()  
            forget_mat_dict[loc][key] = U
            forget_normalize_var_mat_dict[loc][key] = S**2 / (S**2).sum()
    best_alpha_r = 0.0
    best_alpha_f = 0.0
    best_metric = base_metric
    proj_mat_dict= {"pre":OrderedDict(), "post":OrderedDict()}
    grid_search = []
    for alpha_r in [0.3, 1, 3, 10, 30, 100, 300, 1000, 3000, 10000, 30000, 100000, 300000, 1000000, 3000000]:
        for alpha_f in [0.3, 1, 3, 10, 30, 100, 300, 1000, 3000, 10000, 30000, 100000, 300000, 1000000, 3000000]: 
            grid_search.append((alpha_r, alpha_f))    
    pbar = tqdm(grid_search, desc='Verifix Mislabeled')
    for alpha_r, alpha_f in pbar:
        for loc in retain_act.keys():
            for key in retain_act[loc].keys():
                sqrt_imp_r = torch.diag(torch.tensor( (alpha_r*retain_normalize_var_mat_dict[loc][key]/(( (alpha_r-1)*retain_normalize_var_mat_dict[loc][key]) +1))**0.5).to(device))
                Ur = torch.tensor(retain_mat_dict[loc][key]).to(device)
                Ur = torch.mm(Ur, sqrt_imp_r.float())
                Mr = torch.mm(Ur, Ur.transpose(0,1))
                sqrt_imp_f = torch.diag(torch.tensor( (alpha_f*forget_normalize_var_mat_dict[loc][key]/(( (alpha_f-1)*forget_normalize_var_mat_dict[loc][key]) +1))**0.5).to(device))
                Uf = torch.tensor(forget_mat_dict[loc][key]).to(device)
                Uf = torch.mm(Uf, sqrt_imp_f.float())
                Mf = torch.mm(Uf, Uf.transpose(0,1))
                I = torch.eye(Mf.shape[0]).to(device)
                if loc == "pre":
                    proj_mat_dict[loc][key]= I - Mf  + torch.mm(Mf, Mr)
                else:
                    proj_mat_dict[loc][key]= I 
        inference_model = deepcopy(corrupt_model)
        inference_model.project_weights(proj_mat_dict)
        val_loss, val_acc = test(inference_model, val_loader, device)
        metric = val_acc
        if metric>best_metric:
            best_metric = metric
            best_alpha_r = alpha_r
            best_alpha_f = alpha_f
            pbar.set_description(f"Best - Val acc {metric:.2f} alpha_r {best_alpha_r} alpha_f {best_alpha_f} ")
    verifix_end_time = time.time()  
    # Plot decision boundary for best model
    for loc in retain_act.keys():
        for key in retain_act[loc].keys():
            sqrt_imp_r = torch.diag(torch.tensor( (best_alpha_r*retain_normalize_var_mat_dict[loc][key]/(( (best_alpha_r-1)*retain_normalize_var_mat_dict[loc][key]) +1))**0.5).to(device))
            Ur = torch.tensor(retain_mat_dict[loc][key]).to(device)
            Ur = torch.mm(Ur, sqrt_imp_r.float())
            Mr = torch.mm(Ur, Ur.transpose(0,1))
            sqrt_imp_f = torch.diag(torch.tensor( (best_alpha_f*forget_normalize_var_mat_dict[loc][key]/(( (best_alpha_f-1)*forget_normalize_var_mat_dict[loc][key]) +1))**0.5).to(device))
            Uf = torch.tensor(forget_mat_dict[loc][key]).to(device)
            Uf = torch.mm(Uf, sqrt_imp_f.float())
            Mf = torch.mm(Uf, Uf.transpose(0,1))
            I = torch.eye(Mf.shape[0]).to(device)
            if loc == "pre":
                proj_mat_dict[loc][key]= I - Mf + torch.mm(Mf, Mr)
            else:
                proj_mat_dict[loc][key]= I 
    verifix_model = deepcopy(corrupt_model)
    verifix_model.project_weights(proj_mat_dict)
    verifix_test_loss, verifix_test_acc = test(verifix_model, test_loader, device)
    verifix_test_metric = verifix_test_acc
    print(f"Verifix- test acc {verifix_test_acc:.2f} verifix time {verifix_end_time-verifix_start_time}")
    plot_decision_boundary(verifix_model, data1, device, save_dir = f"./images/{save_name_extension}{train_samples_per_class}_linh{hdim}l{layers}_corrupt{args.fraction_corrupt}_seed{args.seed}", title = f"Verifix: Test acc-{verifix_test_acc:.2f}", corrupt_targets=corrupt_targets )

    retrain_model =  Linear(in_feature=2, hidden_features=hdim, num_classes=2, layers=layers).to(device)
    if os.path.exists(f"./pretrained_models/2DSpiral/{save_name_extension}{train_samples_per_class}_linh{hdim}l{layers}_retrained{args.fraction_corrupt}_seed{args.seed}.pt"):
        print("-"*40)
        print("Loading model retrained on clean fraction of corrupt data")
        print("-"*40)
        retrain_model.load_state_dict(torch.load( f"./pretrained_models/2DSpiral/{save_name_extension}{train_samples_per_class}_linh{hdim}l{layers}_retrained{args.fraction_corrupt}_seed{args.seed}.pt" ))
        clean_test_loss, clean_test_acc = test(retrain_model, test_loader, device)
        clean_test_metric = clean_test_acc
        print(f"Retrain- test acc {clean_test_acc:.2f} ")
        plot_decision_boundary(retrain_model, data1, device,  save_dir = f"./images/{save_name_extension}{train_samples_per_class}_linh{hdim}l{layers}_corrupt{args.fraction_corrupt}_seed{args.seed}", title = f"Retrain: Test acc-{clean_test_acc:.2f}", corrupt_targets=corrupt_targets  )
        
    else:
        # Retrain model
        print("-"*40)
        print("Retraining")
        print("-"*40)
        retain_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(data1[0][clean_data_index], data1[1][clean_data_index]),**kwargs)    
        optimizer = torch.optim.SGD(retrain_model.parameters(), lr = lr, momentum=momentum, nesterov=nesterov)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', factor=gamma)
        pbar = tqdm(range(1, epochs+1), desc='Retraining')
        retrain_start_time = time.time()    
        for epoch in pbar:
            train_loss = train(retrain_model, retain_loader, device, optimizer, scheduler)
            test_loss, test_acc = test(retrain_model, val_loader, device)
            _, train_acc = test(retrain_model, retain_loader, device)
            pbar.set_description(f"Val(Train) Acc {test_acc:.2f}({train_acc:.2f})")   
        retrain_end_time = time.time()     
        clean_test_loss, clean_test_acc = test(retrain_model, test_loader, device)
        clean_test_metric = clean_test_acc
        print(f"Retrain- test acc {clean_test_acc:.2f} train time {retrain_end_time-retrain_start_time}")
        plot_decision_boundary(retrain_model, data1, device,  save_dir = f"./images/{save_name_extension}{train_samples_per_class}_linh{hdim}l{layers}_corrupt{args.fraction_corrupt}_seed{args.seed}", title = f"Retrain: Test acc-{clean_test_acc:.2f}", corrupt_targets=corrupt_targets  )
        torch.save(retrain_model.state_dict(), f"./pretrained_models/2DSpiral/{save_name_extension}{train_samples_per_class}_linh{hdim}l{layers}_retrained{args.fraction_corrupt}_seed{args.seed}.pt")
        
if __name__ == '__main__':
    main()


        

