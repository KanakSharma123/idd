import torch.nn as nn

from models.video_encoder import VideoEncoder
from models.istvt import ISTVT



class DeepfakeDetector(nn.Module):


    def __init__(self):

        super().__init__()


        self.cnn = VideoEncoder()


        self.transformer = ISTVT()



    def forward(self,x):


        features = self.cnn(x)


        prediction = self.transformer(features)

        return prediction