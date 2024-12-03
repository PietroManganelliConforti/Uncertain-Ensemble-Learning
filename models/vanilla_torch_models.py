import torch
import  torch.nn as nn


class resnet18v(nn.Module):
    def __init__(self, num_classes=10, **kwargs):

        pretrained = kwargs.get('pretrained', False)
        super(resnet18v, self).__init__()
        self.model = torch.hub.load('pytorch/vision:v0.6.0', 'resnet18', pretrained=pretrained)
        self.model.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        return self.model(x)
    
class resnet34v(nn.Module):
    def __init__(self, num_classes=10, **kwargs):

        pretrained = kwargs.get('pretrained', False)
        super(resnet34v, self).__init__()
        self.model = torch.hub.load('pytorch/vision:v0.6.0', 'resnet34', pretrained=pretrained)
        self.model.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        return self.model(x)
    
class resnet50v(nn.Module):
    def __init__(self, num_classes=10, **kwargs):
        pretrained = kwargs.get('pretrained', False)
        super(resnet50v, self).__init__()
        self.model = torch.hub.load('pytorch/vision:v0.6.0', 'resnet50', pretrained=pretrained)
        self.model.fc = nn.Linear(2048, num_classes)

    def forward(self, x):
        return self.model(x)