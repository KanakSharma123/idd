import torch
import cv2
import numpy as np

from PIL import Image
from torchvision import transforms

from models.full_model import DeepfakeDetector
from video_inference import extract_frames
from face_extractor import extract_faces



device = "cuda" if torch.cuda.is_available() else "cpu"



# -------------------------
# Load Model
# -------------------------

model = DeepfakeDetector()


model.load_state_dict(
    torch.load(
        "deepfake_detector.pth",
        map_location=device
    )
)


model.to(device)
model.eval()



# -------------------------
# GradCAM storage
# -------------------------

activations = {}
gradients = {}



# -------------------------
# Hooks
# -------------------------

def forward_hook(module, inp, output):

    activations["value"] = output



def backward_hook(module, grad_in, grad_out):

    gradients["value"] = grad_out[0]




target_layer = model.cnn.cnn.features[7]


target_layer.register_forward_hook(
    forward_hook
)


target_layer.register_full_backward_hook(
    backward_hook
)




# -------------------------
# Generate heatmap
# -------------------------

def generate_heatmap(video_path):


    # Extract real frames

    frames = extract_frames(
        video_path
    )
    frames = extract_faces(frames)

    print(
        "Extracted frames:",
        frames.shape
    )


    # add batch dimension

    video = frames.unsqueeze(0)


    video = video.to(device)


    print(
        "Model input:",
        video.shape
    )



    # ---------------------
    # Forward
    # ---------------------

    output = model(video)



    pred = output.argmax(dim=1)



    confidence = torch.softmax(
        output,
        dim=1
    )[0][pred].item()*100



    print(
        "Prediction:",
        pred.item()
    )


    print(
        "Confidence:",
        round(confidence,2),
        "%"
    )



    # ---------------------
    # Backward
    # ---------------------

    model.zero_grad()


    score = output[0,pred]


    score.backward()



    # ---------------------
    # Get activations
    # ---------------------

    acts = activations["value"]

    grads = gradients["value"]



    print(
        "Activation:",
        acts.shape
    )


    print(
        "Gradient:",
        grads.shape
    )



    # ---------------------
    # GradCAM
    # ---------------------

    weights = grads.mean(
        dim=(2,3),
        keepdim=True
    )



    cam = (
        weights * acts
    ).sum(dim=1)



    cam = torch.relu(cam)



    cam = cam.detach().cpu().numpy()



    # (frames,7,7)

    frame_cams = cam



    # ---------------------
    # Frame importance
    # ---------------------

    frame_scores = frame_cams.mean(
        axis=(1,2)
    )


    important_frame = np.argmax(
        frame_scores
    )



    print(
        "Frame scores:",
        frame_scores
    )


    print(
        "Most suspicious frame:",
        important_frame+1
    )



    # select CAM

    cam = frame_cams[
        important_frame
    ]



    cam = cam - cam.min()


    cam = cam / (
        cam.max()+1e-8
    )



    cam = cv2.resize(
        cam,
        (224,224)
    )



    # ---------------------
    # Use selected frame
    # ---------------------

    img = frames[
        important_frame
    ]


    img = img.permute(
        1,2,0
    ).cpu().numpy()



    img = np.clip(
        img,
        0,
        1
    )



    # ---------------------
    # Overlay
    # ---------------------

    heatmap = cv2.applyColorMap(
        np.uint8(255*cam),
        cv2.COLORMAP_JET
    )


    heatmap = heatmap[:,:,::-1]/255.0



    overlay = (
        0.5*heatmap +
        0.5*img
    )



    overlay = np.uint8(
        overlay*255
    )



    cv2.imwrite(
        "heatmap.jpg",
        cv2.cvtColor(
            overlay,
            cv2.COLOR_RGB2BGR
        )
    )


    print(
        "Saved heatmap.jpg"
    )





# -------------------------
# Run
# -------------------------

generate_heatmap(
    "data/videos/249_280.mp4"
)