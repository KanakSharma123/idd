import torch

from dataset.dataloader import DeepfakeDataset

from models.full_model import DeepfakeDetector




dataset = DeepfakeDataset(
    "data/faces"
)



video,label = dataset[0]


video = video.unsqueeze(0)



print(
    "Input:",
    video.shape
)



model = DeepfakeDetector()



output=model(video)



print(
    "Prediction:",
    output.shape
)

print(output)