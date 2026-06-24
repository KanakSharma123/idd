import torch
import torch.nn as nn

from models.cnn_encoder import CNNEncoder




class VideoEncoder(nn.Module):


    def __init__(self):

        super().__init__()


        self.cnn = CNNEncoder()



    def forward(self,x):

        """
        x:

        batch,
        frames,
        channels,
        height,
        width

        """


        B,T,C,H,W = x.shape



        # combine frames

        x = x.reshape(
            B*T,
            C,
            H,
            W
        )



        # CNN

        features = self.cnn(x)



        # restore sequence

        features = features.reshape(
            B,
            T,
            -1
        )


        return features