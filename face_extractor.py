import cv2
import torch

from PIL import Image
from facenet_pytorch import MTCNN
from torchvision import transforms



device = "cuda" if torch.cuda.is_available() else "cpu"



mtcnn = MTCNN(
    image_size=224,
    margin=20,
    keep_all=False,
    device=device
)



transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor()
])



def extract_faces(frames):

    faces = []


    for frame in frames:


        # tensor -> image

        frame = frame.permute(
            1,2,0
        ).numpy()



        frame = (frame*255).astype(
            "uint8"
        )


        img = Image.fromarray(
            frame
        )



        face = mtcnn(img)

    if face is None:

        face = transform(img)

    else:
        # MTCNN gives [-1,1], convert to [0,1]
        face = (face + 1) / 2


    faces.append(face)



    return torch.stack(faces)