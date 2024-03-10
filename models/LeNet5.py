from torch.nn import Module
from torch import nn
import torch
import torch.nn.functional as F
from collections import OrderedDict
from .model_utils import auto_get_activations, auto_project_weights


class LeNet5(Module):
    def __init__(self, bias=False):
        super(LeNet5, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5, bias=bias)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(6, 16, 5, bias=bias)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(256, 120, bias=bias)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(120, 84, bias=bias)
        self.relu4 = nn.ReLU()
        self.fc3 = nn.Linear(84, 10, bias=bias)
        self.relu5 = nn.ReLU()

    def forward(self, x):
        y = self.conv1(x)
        y = self.relu1(y)
        y = self.pool1(y)
        y = self.conv2(y)
        y = self.relu2(y)
        y = self.pool2(y)
        y = y.view(y.shape[0], -1)
        y = self.fc1(y)
        y = self.relu3(y)
        y = self.fc2(y)
        y = self.relu4(y)
        y = self.fc3(y)
        y = self.relu5(y)
        # y =  F.log_softmax(y, dim=1)
        return y

    def get_activations(self, x, prev_recur_proj_mat = None):
        act= {"pre":OrderedDict(), "post":OrderedDict()}
        act, y = auto_get_activations(x, self.conv1, 'conv1', prev_recur_proj_mat, act)
        y = self.relu1(y)
        y = self.pool1(y)
        act, y = auto_get_activations(x, self.conv2, 'conv2', prev_recur_proj_mat, act)
        y = self.relu2(y)
        y = self.pool2(y)
        y = y.view(y.shape[0], -1)
        act, y = auto_get_activations(x, self.fc1, 'fc1', prev_recur_proj_mat, act) 
        y = self.relu3(y)
        act, y = auto_get_activations(x, self.fc2, 'fc2', prev_recur_proj_mat, act) 
        y = self.relu4(y)
        act, y = auto_get_activations(x, self.fc3, 'fc3', prev_recur_proj_mat, act) 
        return act
    
    def project_weights(self, projection_mat_dict, proj_classifier=True):
        auto_project_weights(self.conv1, f"conv1", projection_mat_dict) 
        auto_project_weights(self.conv2, f"conv2", projection_mat_dict)
        auto_project_weights(self.fc1, f"fc1", projection_mat_dict) 
        auto_project_weights(self.fc2, f"fc2", projection_mat_dict) 
        auto_project_weights(self.fc3, f"fc3", projection_mat_dict, proj_classifier)
        return 