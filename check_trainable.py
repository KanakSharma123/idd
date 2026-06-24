from models.full_model import DeepfakeDetector


model=DeepfakeDetector()


total=0
trainable=0


for name,param in model.named_parameters():


    total+=param.numel()


    if param.requires_grad:

        trainable+=param.numel()



print(
"Total parameters:",
total
)


print(
"Trainable parameters:",
trainable
)