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
from utils import activation_projection_based_unlearning, get_dataset, get_test_set_clothing, get_model, get_mislabeled_dataset, test
import random 
import copy

def main():
    # Training settings
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
    parser.add_argument('--train-transform', action='store_true', default=False,
                        help='For Saving the current Model')
    # Network Arguments
    parser.add_argument('--arch', type=str, default='vgg11_bn', 
                        help='')    
    parser.add_argument('--val-index-path', type=str, default='./', 
                        help='path to save the model')
    parser.add_argument('--load-loc', type=str,  required=True,
                        help='path to load the model from')      
    parser.add_argument('--save-loc', type=str, default=None,
                        help='')    
    parser.add_argument('--model-name-subscript', type=str, default=None,
                        help='')    
    parser.add_argument('--model-name', type=str, default=None,
                        help='')   
    parser.add_argument('--do-not-save', action='store_true', default=False,
                        help='For Saving the current Model') 
    # Dataset Arguments 
    parser.add_argument('--dataset', type=str, default="cifar10",
                        help='')
    parser.add_argument('--data-path', type=str, default='./', 
                        help='')
    parser.add_argument('--use-valset', action='store_true', default=False,
                        help='For Saving the current Model')  
    # Label noise parameters for synthetic noise injection
    parser.add_argument('--percentage-mislabeled', type=float, default=0.0, 
                        help='') 
    parser.add_argument('--clean-partition', action='store_true', default=False,
                        help='For Saving the current Model')    
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
    ### GPM parameters
    parser.add_argument('--projection-type', type=str, default='baseline,I-(Mf-Mi)', 
                        help='')
    parser.add_argument('--mode', type=str, default="baseline,sap", metavar='EPS',
                        help='')
    parser.add_argument('--scale-coff', type=str, default= "1,3,10,30,100", metavar='SCF',
                        help='importance co-efficeint (default: 0)')
    parser.add_argument('--mode-forget', type=str, default=None, metavar='EPS',
                        help='')
    parser.add_argument('--scale-coff-forget', type=str, default= "0.5,0.75,0.9,1,3", metavar='SCF',
                        help='importance co-efficeint (default: 0)')
    parser.add_argument('--forget-samples', type=int, default=1350, metavar='EPS',
                        help='')
    parser.add_argument('--retain-samples', type=int, default=150, metavar='EPS',
                        help='')
    parser.add_argument('--max-samples', type=int, default=50000, metavar='EPS',
                        help='')
    parser.add_argument('--max-batch-size', type=int, default=150, metavar='EPS',
                        help='')
    parser.add_argument('--gpm-eps', type=float, default=0.99, metavar='EPS',
                        help='')
    parser.add_argument('--start-layer', type=str, default="0", metavar='EPS',
                        help='')
    parser.add_argument('--end-layer', type=str, default="0", metavar='EPS',
                        help='')    
    parser.add_argument('--projection-location', type=str, default="pre", metavar='EPS',
                        help='') 
    parser.add_argument('--train-set-mode', action='store_true', default=False,
                        help='For Saving the current Model')
    parser.add_argument('--val-set-samples', type=int, default=100000, metavar='EPS',
                        help='') 
    ### Act Supression parameters
    parser.add_argument('--num-rounds', type=int, default=1, 
                        help='') 
    parser.add_argument('--use-curvature',action="store_true", default=False,
                        help='') 
    parser.add_argument('--percentile-low-curve', type=float, default=0.0,
                        help='')   
    parser.add_argument('--do-not-project-classifier',action="store_true", default=False,
                        help='') 
    parser.add_argument('--plot-cm', action='store_true', default=False,
                        help='For Saving the current Model')   
    args = parser.parse_args()
    if args.seed == None:
        args.seed = random.randint(0, 65535)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    args.val_set_mode = args.use_valset 
    args.project_classifier = not args.do_not_project_classifier
    # Process the arguments for Verifix. 
    if "," in args.scale_coff:
        args.scale_coff=[float(val) for val in args.scale_coff.split(",")]        
    else:
        args.scale_coff = [float(args.scale_coff)]
    if "," in args.scale_coff_forget:
        args.scale_coff_forget=[float(val) for val in args.scale_coff_forget.split(",")]        
    else:
        args.scale_coff_forget = [float(args.scale_coff_forget)]
    if "," in args.projection_type:
        args.projection_type=[val for val in args.projection_type.split(",")]        
    else:
        args.projection_type = [args.projection_type]
    if "I-(Mf-Mi)" not in args.projection_type:
        args.scale_coff_forget = [1]
    if "," in args.mode:
        args.mode=[val for val in args.mode.split(",")]        
    else:
        args.mode = [args.mode]
    if args.mode_forget is not None:
        if "," in args.mode_forget:
            args.mode_forget=[val for val in args.mode_forget.split(",")]            
        else:
            args.mode_forget = [args.mode_forget]
    else:
        args.mode_forget = [None]
    if "," in args.start_layer:
        args.start_layer=[int(val) for val in args.start_layer.split(",")]        
    else:
        args.start_layer = [int(args.start_layer)]
    if "," in args.end_layer:
        args.end_layer=[int(val) for val in args.end_layer.split(",")]        
    else:
        args.end_layer = [int(args.end_layer)]
    if "," in args.projection_location:
        args.projection_location=[val for val in args.projection_location.split(",")]        
    else:
        args.projection_location = [args.projection_location]    
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
    # Set the run names.
    if args.clean_partition:
        args.percentage_mislabeled=0.0
    if args.use_curvature:
        args.group_name = f"Curve-per{args.percentile_low_curve}-{args.group_name}"
    if args.model_name is None:
        args.group_name = f"{args.group_name}-{args.arch}_MisLabeled{args.percentage_mislabeled}-seed{args.seed}"
        args.model_name = f"{args.dataset}_{args.arch}_MisLabeled{args.percentage_mislabeled}_seed{args.seed}"
        args.run_name = f"seed{args.seed}_lr{args.lr}_wd{args.weight_decay}_bsz{args.batch_size}" 
        if args.sam_rho is not None:
            args.model_name = f"{args.model_name}_sam{args.sam_rho}"
            args.group_name = f"{args.group_name}-SAM{args.sam_rho}"
            args.run_name = f"{args.run_name}_sam-rho{args.sam_rho}"        
        if args.mixup_alpha is not None:
            args.model_name = f"{args.model_name}_mixup{args.mixup_alpha}"
            args.group_name = f"{args.group_name}-MixUp{args.mixup_alpha}"
            args.run_name = f"{args.run_name}_mixup-alpha{args.mixup_alpha}"
        if args.gls_smoothing is not None:
            args.model_name = f"{args.model_name}_gls{args.gls_smoothing}"
            args.group_name = f"{args.group_name}-GLS{args.gls_smoothing}"
            args.run_name = f"{args.run_name}_gls-smoothing{args.gls_smoothing}"
        if args.estop_delta is not None:
            args.model_name = f"{args.model_name}_estop{args.estop_delta}"
            args.group_name = f"{args.group_name}-EStop{args.estop_delta}"
            args.run_name = f"{args.run_name}_early-stopping{args.estop_delta}"
        if args.mnet_gamma_p is not None:
            if args.mmix_alpha is not None:
                args.model_name = f"{args.model_name}-mmix{args.mnet_gamma_p}_{args.mmix_alpha}"
                args.group_name = f"{args.group_name}-MMix{args.mnet_gamma_p}_{args.mmix_alpha}"
                args.run_name = f"{args.run_name}_MentorMix{args.mnet_gamma_p}_{args.mmix_alpha}"
            else:
                args.model_name = f"{args.model_name}-mnet{args.mnet_gamma_p}"
                args.group_name = f"{args.group_name}-MNet{args.mnet_gamma_p}"
                args.run_name = f"{args.run_name}_MentorNet{args.mnet_gamma_p}"
        if args.model_name_subscript:
            args.model_name = f"{args.model_name}_{args.model_name_subscript}"
    else:
        args.group_name = f"{args.group_name}-{args.arch}_MisLabeled{args.percentage_mislabeled}-seed{args.seed}"
        args.run_name = f"seed{args.seed}_lr{args.lr}_wd{args.weight_decay}_bsz{args.batch_size}"
    
    # Creating Synthetic Corrupt dataset if required
    dataset_corrupt, corrupt_samples, (index_list, old_targets, updated_targets) = get_mislabeled_dataset(copy.deepcopy(dataset1), args.percentage_mislabeled, args.num_classes, False,f"{args.val_index_path}/{args.dataset}_{args.arch}_{args.percentage_mislabeled}_seed{args.seed}")
    if "clothing" in args.dataset.lower():
        args.use_valset = True
        trainset = dataset_corrupt
        valset = dataset2
        train_loader = torch.utils.data.DataLoader(trainset,**train_kwargs)
        val_loader = torch.utils.data.DataLoader(valset, **test_kwargs)
        valset_clean = None
        valset_corrupt = None
        trainset_clean = None
        trainset_corrupt = None
        train_loader_clean = []
        train_loader_corrupt  = [] 
        val_loader_clean = []
        val_loader_corrupt = [] 
    elif args.use_valset:
        if os.path.exists(f"{args.val_index_path}/{args.dataset}_{args.arch}_{args.percentage_mislabeled}_seed{args.seed}.corrupt_val_index"):
            val_index = torch.load(f"{args.val_index_path}/{args.dataset}_{args.arch}_{args.percentage_mislabeled}_seed{args.seed}.corrupt_val_index") 
        else:
            ## Validation Split was not used during the training.

            num_of_data_points = len(dataset_corrupt)
            num_of_val_samples = 100000 
            r=np.arange(num_of_data_points)
            np.random.shuffle(r)
            val_index = r[:num_of_val_samples].tolist() 
        val_index = np.array(val_index)
        train_index = np.setdiff1d(np.arange(len(dataset_corrupt)), val_index)
        trainset = torch.utils.data.Subset(dataset_corrupt, train_index)
        valset = torch.utils.data.Subset(dataset_corrupt, val_index)
        # Fraction of trainset used as valset.
        train_loader = torch.utils.data.DataLoader(trainset,**train_kwargs)
        val_loader = torch.utils.data.DataLoader(valset, **test_kwargs)
        if index_list:
            index_list = np.array(index_list)
            train_corrupt_index = np.intersect1d(train_index, index_list)
            val_corrupt_index = np.intersect1d(val_index, index_list)
            trainset_corrupt = torch.utils.data.Subset(dataset_corrupt, train_corrupt_index)
            valset_corrupt = torch.utils.data.Subset(dataset_corrupt, val_corrupt_index)
            train_loader_corrupt  = torch.utils.data.DataLoader(trainset_corrupt,**train_kwargs) 
            val_loader_corrupt = torch.utils.data.DataLoader(valset_corrupt, **test_kwargs)

            train_clean_index = np.setdiff1d(train_index,index_list)     
            val_clean_index = np.setdiff1d(val_index,index_list) 
            trainset_clean = torch.utils.data.Subset(dataset_corrupt, train_clean_index)
            valset_clean = torch.utils.data.Subset(dataset_corrupt, val_clean_index)
            train_loader_clean = torch.utils.data.DataLoader(trainset_clean,**train_kwargs)   
            val_loader_clean = torch.utils.data.DataLoader(valset_clean, **test_kwargs)
        
        else:
            train_loader_corrupt  = []
            val_loader_corrupt = []
            train_loader_clean  = []
            val_loader_clean = []
    else:
        train_index = np.arange(len(dataset_corrupt))
        trainset = dataset_corrupt
        train_loader = torch.utils.data.DataLoader(trainset,**train_kwargs)
        valset = None
        valset_clean = None
        valset_corrupt = None
        if index_list:
            index_list = np.array(index_list)
            train_clean_index = np.setdiff1d(train_index,index_list)     
            trainset_clean = torch.utils.data.Subset(dataset_corrupt, train_clean_index)
            train_loader_clean = torch.utils.data.DataLoader(trainset_clean,**train_kwargs)            
            train_corrupt_index = np.intersect1d(train_index, index_list)
            trainset_corrupt = torch.utils.data.Subset(dataset_corrupt, train_corrupt_index)
            train_loader_corrupt  = torch.utils.data.DataLoader(trainset_corrupt,**train_kwargs) 
        else:
            trainset_clean = None
            trainset_corrupt = None
            train_loader_clean = []
            train_loader_corrupt  = [] 
            val_loader = []
            val_loader_clean = []
            val_loader_corrupt = []  
    if "clothing" in args.dataset.lower():
        test_set = get_test_set_clothing(args)
        test_loader = torch.utils.data.DataLoader(test_set, **test_kwargs)  
    else: 
        test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)    
    print("Dataloader Created")
    # Check trained model!
    model = get_model(args, device)
    model.load_state_dict(torch.load( os.path.join(args.load_loc,f"{args.model_name}.pt" ) ) )
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model, device_ids=list(range(torch.cuda.device_count())))
    print("Model Loaded")

    unlearnt_model = model
    prev_recur_proj_mat = None
    for round in range(args.num_rounds):
        _, next_recur_proj_mat = activation_projection_based_unlearning(args, unlearnt_model, 
                                                                                     (train_loader, train_loader_clean, train_loader_corrupt), 
                                                                                     (val_loader, val_loader_clean, val_loader_corrupt), 
                                                                                     test_loader, 
                                                                                     dataset1, 
                                                                                     device, 
                                                                                     round = round, 
                                                                                     prev_recur_proj_mat=prev_recur_proj_mat )
        if next_recur_proj_mat is not None:
            prev_recur_proj_mat = next_recur_proj_mat
            unlearnt_model = copy.deepcopy(model)
            try:
                unlearnt_model.module.project_weights(prev_recur_proj_mat, args.project_classifier)
            except:
                unlearnt_model.project_weights(prev_recur_proj_mat, args.project_classifier)
        else:
            unlearnt_model = copy.deepcopy(model)
            if prev_recur_proj_mat is not None:
                try:
                    unlearnt_model.module.project_weights(prev_recur_proj_mat, args.project_classifier)
                except:
                    unlearnt_model.project_weights(prev_recur_proj_mat, args.project_classifier)
            break
        args.scale_coff = [val*10 for val in args.scale_coff]
        if "I-(Mf-Mi)" in args.projection_type:
            args.scale_coff_forget = [val*10 for val in args.scale_coff_forget]
        else:
            args.scale_coff_forget = [1]

    run = wandb.init(
                    # Set the project where this run will be logged
                    project=f"Verifix-{args.dataset}-{args.project_name}",
                    group= f"report-{args.projection_location[-1]}layer-{args.group_name}",
                    name=f"{args.run_name}_final_test_acc",
                    entity=args.entity_name,
                    dir = os.environ["LOCAL_HOME"],
                    # Track hyperparameters and run metadata
                    config= vars(args)
        )  
    test_our_acc, test_loss = test(unlearnt_model, device, test_loader, class_label_names=args.class_label_names, num_classes = args.num_classes, set_name="Test Set", verbose=False)
    test_acc, test_loss = test(model, device, test_loader, class_label_names=args.class_label_names, num_classes = args.num_classes, set_name="Test Set", verbose=False)
    wandb.log({"Final/Baseline-acc":test_acc,
               "Final/Verifix-acc":test_our_acc,
               })
    wandb.finish()   
    try:
        if args.save_loc is not None:
            torch.save(unlearnt_model.module.state_dict(),  os.path.join(args.save_loc,f"{args.model_name}_verifix.pt") )
        elif not args.do_not_save:
            torch.save(unlearnt_model.module.state_dict(),  os.path.join(args.load_loc,f"{args.model_name}_verifix.pt") )
    except:
        if args.save_loc is not None:
            torch.save(unlearnt_model.state_dict(),  os.path.join(args.save_loc,f"{args.model_name}_verifix.pt") )
        elif not args.do_not_save:
            torch.save(unlearnt_model.state_dict(),  os.path.join(args.load_loc,f"{args.model_name}_verifix.pt") )


if __name__ == '__main__':
    main()