import torch

from dataset.dataloader import (
    DeepfakeDataset,
    create_split
)

from models.full_model import DeepfakeDetector



# get validation samples
_, val_samples = create_split(
    "data/faces"
)


dataset = DeepfakeDataset(
    "data/faces",
    val_samples
)



video, label = dataset[0]


# add batch dimension

video = video.unsqueeze(0)



print(
    "Input:",
    video.shape
)



model = DeepfakeDetector()


model.eval()



with torch.no_grad():

    output, attention = model(video)



print(
    "Prediction:",
    output.shape
)


print(
    "Attention:",
    attention.shape
)