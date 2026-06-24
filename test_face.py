from video_inference import extract_frames
from face_extractor import extract_faces



frames = extract_frames(
    "data/videos/249_280.mp4"
)



print(
    "Before:",
    frames.shape
)



faces = extract_faces(
    frames
)



print(
    "After:",
    faces.shape
)