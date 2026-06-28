import cv2
import os
import torch

from PIL import Image
from torchvision import transforms


transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor()
])



def extract_frames(video_path, num_frames=6):

    cap = cv2.VideoCapture(video_path)

    total = int(
        cap.get(cv2.CAP_PROP_FRAME_COUNT)
    )


    frame_indices = torch.linspace(
        0,
        total-1,
        num_frames
    ).long()



    frames = []


    for idx in frame_indices:

        cap.set(
            cv2.CAP_PROP_POS_FRAMES,
            idx.item()
        )


        ret, frame = cap.read()


        if ret:

            frame = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB
            )


            image = Image.fromarray(frame)


            image = transform(image)


            frames.append(image)



    cap.release()


    return torch.stack(frames)

frames = extract_frames(
    "data/videos/249_280.mp4"
)

print(frames.shape)
from models.full_model import DeepfakeDetector



device = "cuda" if torch.cuda.is_available() else "cpu"



model = DeepfakeDetector()


model.load_state_dict(
    torch.load(
        "deepfake_detector.pth",
        map_location=device
    )
)


model.to(device)

model.eval()



video = extract_frames(
    "data/videos/012_026.mp4"
)



# add batch

video = video.unsqueeze(0)


video = video.to(device)



print(
    "Model input:",
    video.shape
)



output = model(video)



prob = torch.softmax(
    output,
    dim=1
)



confidence, prediction = torch.max(
    prob,
    dim=1
)



print(
    "Prediction:",
    prediction.item()
)


print(
    "Confidence:",
    confidence.item()*100
)