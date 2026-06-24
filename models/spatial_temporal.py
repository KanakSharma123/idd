import torch
import torch.nn as nn



class SpatialAttention(nn.Module):

    def __init__(
        self,
        dim=512,
        heads=8
    ):

        super().__init__()


        layer = nn.TransformerEncoderLayer(

            d_model=dim,

            nhead=heads,

            batch_first=True

        )


        self.encoder = nn.TransformerEncoder(

            layer,

            num_layers=2

        )



    def forward(self,x):

        return self.encoder(x)
    
class TemporalAttention(nn.Module):


    def __init__(
        self,
        dim=512,
        heads=8
    ):

        super().__init__()


        self.attention = nn.MultiheadAttention(

            embed_dim=dim,

            num_heads=heads,

            batch_first=True

        )



    def forward(self,x):


        out,weights = self.attention(

            x,

            x,

            x

        )


        return out,weights