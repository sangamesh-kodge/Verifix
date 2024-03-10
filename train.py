### Source -> https://github.com/pytorch/examples/blob/main/mnist/main.py


from __future__ import print_function
import argparse
import torch
import torch.nn.functional as F
import torch.optim as optim
import wandb
from sklearn.metrics import confusion_matrix
import numpy as np
import os
from utils import get_dataset, get_test_set_clothing, get_model, get_mislabeled_dataset, SAM, loss_gls, EarlyStopper, IndexedDataset, MentorMixLoss, MentorNetLoss
from models.mentornet import MentorNet
import random 
import copy
from torchvision.transforms import v2
from torch.utils.data import default_collate
import torch.distributions.categorical as cat
import torch.distributions.dirichlet as diri

def train(args, model, device, indexed_train_loader, optimizer, epoch, mnet=None,  loss_p_prev=0, loss_p_second_prev=0):
    model.train()
    train_loss= 0
    for batch_idx, (data, target, v_label, index) in enumerate(indexed_train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()            
        if args.sam_rho is not None:
            # SAM Training
            # first forward-backward pass
            output = model(data)
            loss = F.cross_entropy( output, target) 
            loss.backward()
            optimizer.first_step(zero_grad=True)            
            # second forward-backward pass
            F.cross_entropy(model(data), target).backward()  # make sure to do a full forward pass
            optimizer.second_step(zero_grad=True)
        elif args.gls_smoothing is not None:
            # Label Smoothening training
            output = model(data)
            loss = loss_gls(output, target, args.gls_smoothing)
            loss.backward()
            optimizer.step() 
        elif args.mnet_gamma_p is not None:
            if args.mmix_alpha is not None:
                loss, masked_loss, loss_p_prev, loss_p_second_prev, v = MentorMixLoss(args, device, mnet, model, data, target, v_label,loss_p_prev, loss_p_second_prev, epoch-1)
                # Update v
                for count, idx in enumerate(index):
                    indexed_train_loader.dataset.v_label[idx] = v[count].long()
                masked_loss.backward()
                optimizer.step()
            else:  
                loss, masked_loss, loss_p_prev,  v = MentorNetLoss(args, device, mnet, model, data, target, v_label, loss_p_prev,  epoch-1)
                # Update v
                for count, idx in enumerate(index):
                    indexed_train_loader.dataset.v_label[idx] = v[count].long()
                masked_loss.backward()
                optimizer.step()         
        else:
            # Mixup/Vanilla Training
            output = model(data)
            loss = F.cross_entropy(output, target)
            loss.backward()
            optimizer.step()
        
        
        ### Update to logging cross-entropy loss  
        if args.mnet_gamma_p is not None:
            ### scheduler should run on masked_loss
            train_loss += masked_loss.detach().item()
        else:
            train_loss += loss.detach().item()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(indexed_train_loader.dataset),
                100. * batch_idx / len(indexed_train_loader), loss.detach().item()))
            wandb.log({"Train Set/train_loss":loss.detach().item() })
            if args.dry_run:
                break
    return train_loss, loss_p_prev, loss_p_second_prev 

def test(model, device, test_loader, num_classes=10, set_name = "Val Set"):
    model.eval()
    test_loss = 0
    correct = 0
    cm = np.zeros((num_classes,num_classes))
    dict_classwise_acc={}
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.cross_entropy(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            cm+=confusion_matrix(target.cpu().numpy(),pred.squeeze(-1).cpu().numpy(), labels=[val for val in range(num_classes)])
            correct += pred.eq(target.view_as(pred)).sum().item()
    classwise_acc = cm.diagonal()/cm.sum(axis=1)
    for i in range(1,11):
        dict_classwise_acc[str(i)] =  classwise_acc[i-1]

    test_loss /= len(test_loader.dataset)
    print(f'{set_name}: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({100. * correct / len(test_loader.dataset):.0f}%)\n')

    wandb.log({ f"{set_name}/test_loss":test_loss, 
                f"{set_name}/test_acc":100. * correct / len(test_loader.dataset),
                f"{set_name}/classwise/test_acc":dict_classwise_acc
                }
                )
    return test_loss

def main():
    # Vanilla Training settings
    parser = argparse.ArgumentParser(description='PyTorch cifar10 Example')
    parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=512, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=350, metavar='N',
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--lr', type=float, default=0.05, metavar='LR',
                        help='')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='LR',
                        help='')
    parser.add_argument('--weight-decay', type=float, default=5e-4, metavar='LR',
                        help='')
    parser.add_argument('--step-schedule', type=int, default=None, metavar='LR',
                        help='Epochs after which lr should be reduced. Try 100')
    parser.add_argument('--gamma', type=float, default=0.5, metavar='M',
                        help='Learning rate step gamma (default: 0.5) lr on plateau ')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--no-mps', action='store_true', default=False,
                        help='disables macOS GPU training')
    parser.add_argument('--dry-run', action='store_true', default=False,
                        help='quickly check a single pass')
    parser.add_argument('--seed', type=int, default=None, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=50, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--no-train-transform', action='store_false', default=False,
                        help='For Saving the current Model')
    # Network Arguments
    parser.add_argument('--arch', type=str, default='vgg11_bn', 
                        help='')    
    parser.add_argument('--model-path', type=str, default='./', 
                        help='path to save the model')
    parser.add_argument('--load-loc', type=str, default=None,
                        help='path to load the model')      
    parser.add_argument('--do-not-save', action='store_true', default=False,
                        help='For Saving the current Model') 
    # Dataset Arguments 
    parser.add_argument('--dataset', type=str, default="cifar10",
                        help='')
    parser.add_argument('--data-path', type=str, default='./', 
                        help='')
    parser.add_argument('--use-valset', type=float, default=None,
                        help='')  
    # Label noise parameters for synthetic noise injection
    parser.add_argument('--percentage-mislabeled', type=float, default=0.0, 
                        help='') 
    parser.add_argument('--clean-partition', action='store_true', default=False,
                        help='')    
    # wandb Arguments
    parser.add_argument('--project-name', type=str, default='final', 
                        help='')
    parser.add_argument('--group-name', type=str, default='train', 
                        help='')  
    parser.add_argument('--entity-name', type=str, default=None, 
                        help='')  
    # SAM Arguments
    parser.add_argument('--sam-rho',type=float, default=None, 
                        help='to do SAM') 
    # MixUp Arguments
    parser.add_argument('--mixup-alpha',type=float, default=None, 
                        help='to do MixUp') 
    # GLS Arguments
    parser.add_argument('--gls-smoothing',type=float, default=None, 
                        help='Use GLS with given smoothening rate') 
    # Early Stopping Arguments
    parser.add_argument('--estop-delta',type=float, default=None, 
                        help='change in loss to theshold for 5 checks of early stopping')   
    # MentorNet Arguments  
    parser.add_argument('--mnet-burnin', default = 10, type=int) 
    parser.add_argument('--mnet-ema', default = 0.05, type=float) 
    parser.add_argument('--mnet-gamma-p', default = None, type=float, help= "Set to run MentorNet. Suggested value 0.7")
    # MentorMix Arguments
    parser.add_argument('--mmix-alpha', default = None, type=float, help= "Set to run MentorMix. Suggested value 0.4")
    
    
    args = parser.parse_args()
    args.train_transform = not args.no_train_transform
    if args.seed == None:
        args.seed = random.randint(0, 65535)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
   
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    use_mps = not args.no_mps and torch.backends.mps.is_available()
    if use_cuda:
        device = torch.device("cuda")
    elif use_mps:
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    train_kwargs = {'batch_size': args.batch_size}
    test_kwargs = {'batch_size': args.test_batch_size}
    if use_cuda:
        cuda_kwargs = {'num_workers': 16,
                       'pin_memory': True,
                       'shuffle': True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)
    # Load the dataset.
    dataset1, dataset2 = get_dataset(args)
    # Check trained model!
    model = get_model(args, device)
    if args.load_loc is not None:            
        print("-"*40)
        print(f"Loading model from {args.load_loc}")
        load_state_dict = torch.load(args.load_loc)
        model_state_dict = model.state_dict()
        size_mismatch = set()
        missing_keys = set()
        for key in model_state_dict:
            if key in load_state_dict:
                if load_state_dict[key].shape == model_state_dict[key].shape:
                    model_state_dict[key].copy_(load_state_dict[key])
                else:
                    size_mismatch.add(key)
            else:
                if "batch" not in key and "bn" not in key:
                    missing_keys.add(key) 
        model.load_state_dict(model_state_dict)    
        print(f"Loaded keys and the following keys not loaded: \n SIZE MISMATCH ------> {size_mismatch}\n MISSING ------> {missing_keys}")
        print("-"*40)

    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model, device_ids=list(range(torch.cuda.device_count())))

    # Set the run names.
    args.group_name = f"{args.group_name}-{args.arch}-MisLabeled{args.percentage_mislabeled}"
    args.model_name = f"{args.dataset}_{args.arch}_MisLabeled{args.percentage_mislabeled}_seed{args.seed}"
    args.run_name = f"seed{args.seed}_lr{args.lr}_wd{args.weight_decay}_bsz{args.batch_size}"
    if args.clean_partition:
        args.percentage_mislabeled=0.0 
    if args.sam_rho is not None:
        args.model_name = f"{args.model_name}_sam{args.sam_rho}"
        args.group_name = f"{args.group_name}-SAM{args.sam_rho}"
        args.run_name = f"{args.run_name}_sam-rho{args.sam_rho}"        
    if args.mixup_alpha is not None:
        args.model_name = f"{args.model_name}_mixup{args.mixup_alpha}"
        args.group_name = f"{args.group_name}-MixUp{args.mixup_alpha}"
        args.run_name = f"{args.run_name}_mixup-alpha{args.mixup_alpha}"
    if args.mnet_gamma_p is not None:
        if args.mmix_alpha is not None:
            args.model_name = f"{args.model_name}-mmix{args.mnet_gamma_p}_{args.mmix_alpha}"
            args.group_name = f"{args.group_name}-MMix{args.mnet_gamma_p}_{args.mmix_alpha}"
            args.run_name = f"{args.run_name}_MentorMix{args.mnet_gamma_p}_{args.mmix_alpha}"
        else:
            args.model_name = f"{args.model_name}-mnet{args.mnet_gamma_p}"
            args.group_name = f"{args.group_name}-MNet{args.mnet_gamma_p}"
            args.run_name = f"{args.run_name}_MentorNet{args.mnet_gamma_p}"
    if args.gls_smoothing is not None:
        args.model_name = f"{args.model_name}_gls{args.gls_smoothing}"
        args.group_name = f"{args.group_name}-GLS{args.gls_smoothing}"
        args.run_name = f"{args.run_name}_gls-smoothing{args.gls_smoothing}"
    if args.estop_delta is not None:
        args.model_name = f"{args.model_name}_estop{args.estop_delta}"
        args.group_name = f"{args.group_name}-EStop{args.estop_delta}"
        args.run_name = f"{args.run_name}_early-stopping{args.estop_delta}"
    
    print("-"*40)
    print("Initialized Run Variables")
    print("-"*40)
    # Creating Synthetic Corrupt dataset if required 
    if not os.path.exists(f"{args.model_path}"):
        os.makedirs(f"{args.model_path}")
    dataset_corrupt, corrupt_samples, (index_list, old_targets, updated_targets) = get_mislabeled_dataset(copy.deepcopy(dataset1), args.percentage_mislabeled, args.num_classes, args.clean_partition, f"{args.model_path}/{args.dataset}_{args.arch}_{args.percentage_mislabeled}_seed{args.seed}")
    if "clothing" in args.dataset.lower():
        trainset_corrupt = dataset_corrupt
    elif args.use_valset is not None and args.use_valset > 0.0 and args.use_valset <=1.0:
        ### split corrupt data into train and val.
        num_of_data_points = len(dataset_corrupt)
        num_of_val_samples = int(args.use_valset  * num_of_data_points)
        r=np.arange(num_of_data_points)
        np.random.shuffle(r)
        val_index = r[:num_of_val_samples].tolist()
        if os.path.exists(f"{args.model_path}/{args.dataset}_{args.arch}_{args.percentage_mislabeled}_seed{args.seed}.corrupt_val_index"):
            val_index = torch.load(f"{args.model_path}/{args.dataset}_{args.arch}_{args.percentage_mislabeled}_seed{args.seed}.corrupt_val_index") 
        else:
            if not os.path.exists(f"{args.model_path}/"):
                os.makedirs(f"{args.model_path}/")
            torch.save(val_index, f"{args.model_path}/{args.dataset}_{args.arch}_{args.percentage_mislabeled}_seed{args.seed}.corrupt_val_index") 
        train_index = np.setdiff1d(np.arange(num_of_data_points), np.array(val_index))
        
        trainset_corrupt = torch.utils.data.Subset(dataset_corrupt, train_index)
        valset_corrupt = torch.utils.data.Subset(dataset_corrupt, val_index)
    else:
        trainset_corrupt = dataset_corrupt
        valset_corrupt = None
    print("-"*40)
    print("Dataset Created")
    print("-"*40)
    # Does mixup if mixup_alpha set.
    if args.mixup_alpha is not None:
        mixup_function = v2.MixUp(alpha=args.mixup_alpha, num_classes=args.num_classes) 
        def collate_fn(batch):
            return mixup_function(*default_collate(batch))
    else:
        def collate_fn(batch):
            return default_collate(batch)
    # Creates dataloader
    train_loader_corrupt = torch.utils.data.DataLoader(trainset_corrupt,**train_kwargs, collate_fn=collate_fn)
    if "clothing" in args.dataset.lower():
        test_set = get_test_set_clothing(args)
        val_loader_corrupt = torch.utils.data.DataLoader(dataset2,**test_kwargs)
        test_loader = torch.utils.data.DataLoader(test_set, **test_kwargs)
    elif args.use_valset is not None and args.use_valset > 0.0 and args.use_valset <=1.0:
        val_loader_corrupt = torch.utils.data.DataLoader(valset_corrupt,**test_kwargs)      
        test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)  
    else: 
        test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)  
    # Setup MentorNet model and optimizer.
    indexed_trainset_corrupt = IndexedDataset(trainset_corrupt)      
    indexed_train_loader_corrupt = torch.utils.data.DataLoader(indexed_trainset_corrupt,**train_kwargs, collate_fn=collate_fn)
    loss_p_prev = 0
    loss_p_second_prev = 0
    if args.mnet_gamma_p is not None:
        mnet = MentorNet("threshold").to(device)
    else:
        mnet = None 
    print("-"*40)
    print("DataLoader Created")
    print("-"*40)
    # Create optimizer and scheduler.
    if args.sam_rho is not None:
        base_optimizer = torch.optim.SGD
        optimizer = SAM(model.parameters(), base_optimizer, rho=args.sam_rho, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay, nesterov=True)
    else:
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay, nesterov=True)
    if args.step_schedule is not None:
        scheduler = optim.lr_scheduler.StepLR(optimizer, args.step_schedule, args.gamma)
    else:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', factor=args.gamma)
    # Initialize the EarlyStopper
    if args.estop_delta is not None:
        early_stopper = EarlyStopper(patience=5, min_delta=args.estop_delta)
    print("-"*40)
    print("Optimizer and Scheduler Created")
    print("-"*40)
    # Initializes the save path and checks if the file exits. Terminates to avoid overwriting.
    save_folder_path = os.path.join(f"{args.model_path}",f"{args.dataset}_{args.project_name}",f"{args.group_name}")
    if not os.path.exists(save_folder_path):
        os.makedirs(save_folder_path)    
    if os.path.exists( os.path.join(save_folder_path, f"{args.model_name}_final.pt")):
        if not args.do_not_save:
            raise FileExistsError     
    # Set Wandb login
    run = wandb.init(
                    # Set the project where this run will be logged
                    project=f"LabelNoiseRobustness-{args.dataset}-{args.project_name}",
                    group= args.group_name, 
                    name=args.run_name,
                    entity=args.entity_name,
                    dir = os.environ["LOCAL_HOME"],
                    # Track hyperparameters and run metadata
                    config= vars(args))
    # Train  
    min_validation_loss = np.inf
    best_model = None
    for epoch in range(1, args.epochs + 1):
        train_loss, loss_p_prev, loss_p_second_prev = train(args, model, device, indexed_train_loader_corrupt, optimizer, epoch, mnet, loss_p_prev, loss_p_second_prev)
        if scheduler:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(train_loss, epoch=(epoch+1))
            else:
                scheduler.step()
        if (args.use_valset is not None and args.use_valset > 0.0 and args.use_valset <=1.0) or "clothing" in args.dataset.lower():
            validation_loss=test(model, device, val_loader_corrupt, args.num_classes)
            if validation_loss <= min_validation_loss:
                best_model = copy.deepcopy(model) 
        else:
            best_model = copy.deepcopy(model) 
        test(model, device, test_loader, args.num_classes, set_name="Test Set")
        if args.estop_delta is not None:
            if early_stopper.early_stop(validation_loss):             
                break
            
    if not args.dry_run:
        # Save the best and final model.
        if not args.do_not_save:
            try:
                torch.save(model.module.state_dict(), os.path.join(save_folder_path, f"{args.model_name}_final.pt") )
            except:
                torch.save(model.state_dict(), os.path.join(save_folder_path, f"{args.model_name}_final.pt") )
            if args.use_valset is not None and args.use_valset > 0.0 and args.use_valset <=1.0:
                try:
                    torch.save(best_model.module.state_dict(), os.path.join(save_folder_path, f"{args.model_name}_best.pt") )
                except:
                    torch.save(best_model.state_dict(), os.path.join(save_folder_path, f"{args.model_name}_best.pt") )
        # Log the best/final model. 
        test(best_model, device, test_loader, args.num_classes, set_name="Best Model-Test Set")
        train_loader_corrupt = torch.utils.data.DataLoader(trainset_corrupt,**train_kwargs) # Remove Mixup before testing on TrainSet.
        test(best_model, device, train_loader_corrupt, args.num_classes, set_name="Best Model-Train Set")
        if args.use_valset is not None and args.use_valset > 0.0 and args.use_valset <=1.0:
            test(best_model, device, val_loader_corrupt, args.num_classes, set_name="Best Model-Val Set")
     


    wandb.finish()

if __name__ == '__main__':
    main()