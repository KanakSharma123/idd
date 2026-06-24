import torch
import torch.nn as nn

from models.spatial_temporal import (
    SpatialAttention,
    TemporalAttention
)



class ISTVTv2(nn.Module):


    def __init__(self):

        super().__init__()



        self.projection = nn.Linear(

            2048,

            512

        )


        self.spatial = SpatialAttention()



        self.temporal = TemporalAttention()



        self.classifier = nn.Sequential(

            nn.Linear(
                512,
                128
            ),

            nn.ReLU(),


            nn.Linear(
                128,
                2
            )

        )



    def forward(self,x):


        # CNN features

        x=self.projection(x)



        # spatial learning

        x=self.spatial(x)



        # temporal learning

        x,attention=self.temporal(x)



        # video representation

        x=torch.mean(
            x,
            dim=1
        )


        output=self.classifier(x)



        return output,attention