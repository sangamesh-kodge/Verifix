
from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from .model_utils import auto_get_activations, auto_project_weights

class Linear5(nn.Module):
    def __init__(self, in_feature=784, hidden_feature=784,n_class = 10, bias=False):
        super(Linear5, self).__init__()
        self.fc1 = nn.Linear(in_feature, hidden_feature, bias=bias)
        self.fc2 = nn.Linear(hidden_feature, hidden_feature, bias=bias)
        self.fc3 = nn.Linear(hidden_feature, hidden_feature, bias=bias)
        self.fc4 = nn.Linear(hidden_feature, hidden_feature, bias=bias)
        self.fc5 = nn.Linear(hidden_feature, n_class, bias=bias)

        # stores the activations for gpm
        # self.act=OrderedDict()

    def forward(self, x):        
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.dropout(F.relu(x),p=.1)
        x = self.fc2(x)
        x = F.dropout(F.relu(x),p=.1)
        x = self.fc3(x)
        x = F.dropout(F.relu(x),p=.1)
        x = self.fc4(x)
        x = F.dropout(F.relu(x),p=.1)
        x = self.fc5(x)
        # x =  F.log_softmax(x, dim=1)
        return x

    def get_activations(self, x, prev_recur_proj_mat = None):
        act= {"pre":OrderedDict(), "post":OrderedDict()}
        x = torch.flatten(x, 1)
        act, x = auto_get_activations(x, self.fc1, 'fc1', prev_recur_proj_mat, act)
        x = F.dropout(F.relu(x),p=.1)
        act, x = auto_get_activations(x, self.fc2, 'fc2', prev_recur_proj_mat, act)
        x = F.dropout(F.relu(x),p=.1)
        act, x = auto_get_activations(x, self.fc3, 'fc3', prev_recur_proj_mat, act)
        x = F.dropout(F.relu(x),p=.1)
        act, x = auto_get_activations(x, self.fc4, 'fc4', prev_recur_proj_mat, act)
        x = F.dropout(F.relu(x),p=.1)
        act, x = auto_get_activations(x, self.fc5, 'fc5', prev_recur_proj_mat, act)
        return act
        
    def project_weights(self, projection_mat_dict, proj_classifier=True):
        auto_project_weights(self.fc1, f"fc1", projection_mat_dict) 
        auto_project_weights(self.fc2, f"fc2", projection_mat_dict) 
        auto_project_weights(self.fc3, f"fc3", projection_mat_dict) 
        auto_project_weights(self.fc4, f"fc4", projection_mat_dict) 
        auto_project_weights(self.fc5, f"fc5", projection_mat_dict, proj_classifier) 
        return 