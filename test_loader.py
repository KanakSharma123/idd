from dataset.dataloader import DeepfakeDataset



dataset = DeepfakeDataset(
    "data/faces"
)



print(
    "Videos:",
    len(dataset)
)



x,y = dataset[0]



print(
    "Tensor:",
    x.shape
)


print(
    "Label:",
    y
)