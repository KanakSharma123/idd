import torch
import torch.nn as nn



class ISTVT(nn.Module):


    def __init__(
        self,
        feature_dim=2048,
        hidden_dim=512,
        num_layers=4,
        num_heads=8
    ):

        super().__init__()



        self.embedding = nn.Linear(
            feature_dim,
            hidden_dim
        )



        encoder_layer = nn.TransformerEncoderLayer(

            d_model=hidden_dim,

            nhead=num_heads,

            dim_feedforward=2048,

            dropout=0.1,

            batch_first=True

        )



        self.transformer = nn.TransformerEncoder(

            encoder_layer,

            num_layers=num_layers

        )



        self.classifier = nn.Sequential(

            nn.Linear(
                hidden_dim,
                256
            ),

            nn.ReLU(),

            nn.Dropout(0.3),


            nn.Linear(
                256,
                2
            )

        )





    def forward(self,x):


        x=self.embedding(x)



        x=self.transformer(x)



        x=torch.mean(
            x,
            dim=1
        )


        output=self.classifier(x)


        return output