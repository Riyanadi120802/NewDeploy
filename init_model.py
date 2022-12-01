import torch.nn as nn
import torchvision.models as models
from collections import OrderedDict


def Model():
    model_ft = models.mobilenet_v2(pretrained=True)

    for param in model_ft.parameters():
        param.requires_grad = False

    newClassifier = nn.Sequential(
        OrderedDict([
            ('0', nn.Dropout(p=0.2, inplace=False)),
            ('1', nn.Linear(in_features=1280, out_features=11, bias=True))
        ])
    )

    model_ft.classifier = newClassifier

    return model_ft
