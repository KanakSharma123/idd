import torch
import cv2
import os
import numpy as np

from PIL import Image
from facenet_pytorch import MTCNN
from torchvision import transforms

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



mtcnn = MTCNN(
    image_size=224,
    margin=20
)



transform = transforms.Compose([

    transforms.Resize(
        (224,224)
    ),

    transforms.ToTensor()

])




def process_video(video_path):


    cap=cv2.VideoCapture(video_path)


    total=int(
        cap.get(
            cv2.CAP_PROP_FRAME_COUNT
        )
    )


    indexes=np.linspace(
        0,
        total-1,
        6
    ).astype(int)



    frames=[]


    for idx in indexes:


        cap.set(
            cv2.CAP_PROP_POS_FRAMES,
            idx
        )


        ret,frame=cap.read()


        if ret:


            frame=cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB
            )


            img=Image.fromarray(frame)


            face=mtcnn(img)



            if face is not None:

                frames.append(face)


    cap.release()



    if len(frames)!=6:

        return None



    frames=torch.stack(
        frames
    )


    return frames.unsqueeze(0)






def predict(video):


    x=process_video(video)



    if x is None:

        return "Face not detected"



    x=x.to(device)



    with torch.no_grad():

        output=model(x)


        prob=torch.softmax(
            output,
            dim=1
        )


        confidence=torch.max(
            prob
        ).item()



        prediction=torch.argmax(
            output
        ).item()



    label=[
        "REAL",
        "FAKE"
    ][prediction]



    return label, confidence