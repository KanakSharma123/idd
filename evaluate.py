import torch
from torch.utils.data import DataLoader

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)


from dataset.dataloader import (
    DeepfakeDataset,
    create_split
)

from models.full_model import DeepfakeDetector




device="cuda" if torch.cuda.is_available() else "cpu"



_,val_samples=create_split(
    "data/faces"
)



dataset=DeepfakeDataset(
    "data/faces",
    val_samples
)


loader=DataLoader(
    dataset,
    batch_size=2
)



model=DeepfakeDetector()


model.load_state_dict(
    torch.load(
        "deepfake_detector.pth"
    )
)


model.to(device)

model.eval()



y_true=[]

y_pred=[]

y_prob=[]



with torch.no_grad():


    for videos,labels in loader:


        videos=videos.to(device)


        outputs=model(videos)



        probs=torch.softmax(
            outputs,
            dim=1
        )[:,1]



        preds=torch.argmax(
            outputs,
            dim=1
        )



        y_true.extend(
            labels.numpy()
        )


        y_pred.extend(
            preds.cpu().numpy()
        )


        y_prob.extend(
            probs.cpu().numpy()
        )




print("All Labels:", y_true)
print("All Predictions:", y_pred)

print(
"Accuracy:",
accuracy_score(
y_true,
y_pred
)
)



print(
"Precision:",
precision_score(
y_true,
y_pred
)
)


print(
"Recall:",
recall_score(
y_true,
y_pred
)
)


print(
"F1:",
f1_score(
y_true,
y_pred
)
)


print(
"AUC:",
roc_auc_score(
y_true,
y_prob
)
)
