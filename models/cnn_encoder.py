import torch
import torch.nn as nn
import torchvision.models as models



class CNNEncoder(nn.Module):

    def __init__(self):

        super().__init__()


        # pretrained ResNet50
        resnet = models.resnet50(
            weights="DEFAULT"
        )


        # remove classifier
        self.features = nn.Sequential(
            *list(resnet.children())[:-1]
        )


    def forward(self,x):

        """
        x:
        batch*frames,3,224,224
        """


        x = self.features(x)


        # flatten
        x = torch.flatten(
            x,
            1
        )


        return x