import torch

from dataset.dataloader import DeepfakeDataset
from models.video_encoder import VideoEncoder



dataset = DeepfakeDataset(
    "data/faces"
)



video,label = dataset[0]


# add batch dimension

video = video.unsqueeze(0)



print(
    "Input:",
    video.shape
)



model = VideoEncoder()



out = model(video)



print(
    "CNN output:",
    out.shape
)