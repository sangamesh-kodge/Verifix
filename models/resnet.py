
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from .model_utils import auto_get_activations, auto_project_weights

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.downsample = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.downsample(x)
        out = F.relu(out)
        return out

    def get_activations(self, x, block_key, prev_recur_proj_mat = None, act = None):
        act, out = auto_get_activations(x, self.conv1, f"{block_key}.conv1", prev_recur_proj_mat, act)
        out = F.relu(self.bn1(out))
        act, out = auto_get_activations(out, self.conv2, f"{block_key}.conv2", prev_recur_proj_mat, act)
        out = self.bn2(out) 
        act, x = auto_get_activations(x, self.downsample, f"{block_key}.downsample", prev_recur_proj_mat, act)
        out +=x
        out = F.relu(out)
        return act, out
    
    def project_weights(self, block_key, projection_mat_dict):
        auto_project_weights(self.conv1, f"{block_key}.conv1", projection_mat_dict) 
        auto_project_weights(self.conv2, f"{block_key}.conv2", projection_mat_dict) 
        auto_project_weights(self.downsample, f"{block_key}.downsample", projection_mat_dict) 
        


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion *
                               planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.downsample = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.downsample(x)
        out = F.relu(out)
        return out

    def get_activations(self, x, block_key, prev_recur_proj_mat = None, act = None):
        act, out = auto_get_activations(x, self.conv1, f"{block_key}.conv1", prev_recur_proj_mat, act)
        out = F.relu(self.bn1(out))
        act, out = auto_get_activations(out, self.conv2, f"{block_key}.conv2", prev_recur_proj_mat, act)
        out = F.relu(self.bn2(out) )
        act, out = auto_get_activations(out, self.conv3, f"{block_key}.conv3", prev_recur_proj_mat, act)
        out = self.bn3(out) 
        act, x = auto_get_activations(x, self.downsample, f"{block_key}.downsample", prev_recur_proj_mat, act)
        out +=x
        out = F.relu(out)
        return act, out



    def project_weights(self, block_key, projection_mat_dict):
        auto_project_weights(self.conv1, f"{block_key}.conv1", projection_mat_dict) 
        auto_project_weights(self.conv2, f"{block_key}.conv2", projection_mat_dict) 
        auto_project_weights(self.conv3, f"{block_key}.conv3", projection_mat_dict) 
        auto_project_weights(self.downsample, f"{block_key}.downsample", projection_mat_dict) 
        
        
        
class ResNet_cifar(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, do_log_softmax=False):
        super(ResNet_cifar, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.fc = nn.Linear(512*block.expansion, num_classes)
        self.do_log_softmax = do_log_softmax

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        if self.do_log_softmax:
            out =  F.log_softmax(out, dim=1)
        return out
    
    def get_activations(self, x, prev_recur_proj_mat=None):   
        act={"pre":OrderedDict(), "post":OrderedDict()} 
        act, out = auto_get_activations(x, self.conv1, "conv1", prev_recur_proj_mat, act)
        out = F.relu(self.bn1(out))
        act, out = auto_get_activations(out, self.layer1, "layer1", prev_recur_proj_mat, act)
        act, out = auto_get_activations(out, self.layer2, "layer2", prev_recur_proj_mat, act)
        act, out = auto_get_activations(out, self.layer3, "layer3", prev_recur_proj_mat, act)
        act, out = auto_get_activations(out, self.layer4, "layer4", prev_recur_proj_mat, act)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        act, out = auto_get_activations(out, self.fc, f"fc", prev_recur_proj_mat, act)
        return act
    
    def project_weights(self, projection_mat_dict, proj_classifier=False):
        auto_project_weights(self.conv1, f"conv1", projection_mat_dict) 
        auto_project_weights(self.layer1, f"layer1", projection_mat_dict) 
        auto_project_weights(self.layer2, f"layer2", projection_mat_dict) 
        auto_project_weights(self.layer3, f"layer3", projection_mat_dict) 
        auto_project_weights(self.layer4, f"layer4", projection_mat_dict) 
        auto_project_weights(self.fc, f"fc", projection_mat_dict, proj_classifier)
        return 
    

    

class ResNet_imagenet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, do_log_softmax=False):
        super(ResNet_imagenet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7,
                               stride=2, padding=3, bias=False)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512*block.expansion, num_classes)
        self.do_log_softmax=do_log_softmax
    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.maxpool(F.relu(self.bn1(self.conv1(x))))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)

        if self.do_log_softmax:
            out =  F.log_softmax(out, dim=1)
        return out
    
    
    def get_activations(self, x, prev_recur_proj_mat=None):   
        act={"pre":OrderedDict(), "post":OrderedDict()} 
        act, out = auto_get_activations(x, self.conv1, "conv1", prev_recur_proj_mat, act)
        out =  self.maxpool(F.relu(self.bn1(out)))
        act, out = auto_get_activations(out, self.layer1, "layer1", prev_recur_proj_mat, act)
        act, out = auto_get_activations(out, self.layer2, "layer2", prev_recur_proj_mat, act)
        act, out = auto_get_activations(out, self.layer3, "layer3", prev_recur_proj_mat, act)
        act, out = auto_get_activations(out, self.layer4, "layer4", prev_recur_proj_mat, act)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        act, out = auto_get_activations(out, self.fc, f"fc", prev_recur_proj_mat, act)
        return act
    
    def project_weights(self, projection_mat_dict, proj_classifier=False):
        auto_project_weights(self.conv1, f"conv1", projection_mat_dict) 
        auto_project_weights(self.layer1, f"layer1", projection_mat_dict) 
        auto_project_weights(self.layer2, f"layer2", projection_mat_dict) 
        auto_project_weights(self.layer3, f"layer3", projection_mat_dict) 
        auto_project_weights(self.layer4, f"layer4", projection_mat_dict) 
        auto_project_weights(self.fc, f"fc", projection_mat_dict, proj_classifier=proj_classifier)
        return 
    

    
    
def ResNet18(num_classes=1000, dataset = "imagenet", do_log_softmax=False):
    if "imagenet" in dataset.lower() or "webvision" in dataset.lower() or "clothing" in dataset.lower():
        ResNet = ResNet_imagenet
    elif "cifar" in dataset.lower():
        ResNet = ResNet_cifar
    else:
        raise ValueError
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes, do_log_softmax=do_log_softmax)


def ResNet34(num_classes=1000, dataset = "imagenet"):
    if "imagenet" in dataset.lower() or "webvision" in dataset.lower() or "clothing" in dataset.lower():
        ResNet = ResNet_imagenet
    elif "cifar" in dataset.lower():
        ResNet = ResNet_cifar
    else:
        raise ValueError
    return ResNet(BasicBlock, [3, 4, 6, 3], num_classes=num_classes)


def ResNet50(num_classes=1000, dataset = "imagenet"):
    if "imagenet" in dataset.lower() or "webvision" in dataset.lower() or "clothing" in dataset.lower():
        ResNet = ResNet_imagenet
    elif "cifar" in dataset.lower():
        ResNet = ResNet_cifar
    else:
        raise ValueError
    return ResNet(Bottleneck, [3, 4, 6, 3], num_classes=num_classes)


def ResNet101(num_classes=1000, dataset = "imagenet"):
    if "imagenet" in dataset.lower() or "webvision" in dataset.lower() or "clothing" in dataset.lower():
        ResNet = ResNet_imagenet
    elif "cifar" in dataset.lower():
        ResNet = ResNet_cifar
    else:
        raise ValueError
    return ResNet(Bottleneck, [3, 4, 23, 3], num_classes=num_classes)


def ResNet152(num_classes=1000, dataset = "imagenet"):
    if "imagenet" in dataset.lower() or "webvision" in dataset.lower() or "clothing" in dataset.lower():
        ResNet = ResNet_imagenet
    elif "cifar" in dataset.lower():
        ResNet = ResNet_cifar
    else:
        raise ValueError
    return ResNet(Bottleneck, [3, 8, 36, 3], num_classes=num_classes)


def test():
    net = ResNet18()
    y = net(torch.randn(1, 3, 32, 32))
    print(y.size())

# test()