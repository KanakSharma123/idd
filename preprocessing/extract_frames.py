import cv2
import os
import numpy as np
from tqdm import tqdm


NUM_FRAMES = 6



def extract_frames(video_path, save_path):

    os.makedirs(save_path, exist_ok=True)

    cap = cv2.VideoCapture(video_path)


    total_frames = int(
        cap.get(cv2.CAP_PROP_FRAME_COUNT)
    )


    if total_frames < NUM_FRAMES:
        print("Skipping:", video_path)
        return



    frame_indexes = np.linspace(
        0,
        total_frames-1,
        NUM_FRAMES
    ).astype(int)



    for i, frame_id in enumerate(frame_indexes):

        cap.set(
            cv2.CAP_PROP_POS_FRAMES,
            frame_id
        )


        ret, frame = cap.read()


        if ret:

            cv2.imwrite(
                os.path.join(
                    save_path,
                    f"{i}.jpg"
                ),
                frame
            )



    cap.release()





def process_dataset():

    classes = [
        "real",
        "fake"
    ]


    for label in classes:


        input_folder = os.path.join(
            "data/raw",
            label
        )


        output_folder = os.path.join(
            "data/processed",
            label
        )



        videos = os.listdir(input_folder)



        for video in tqdm(
            videos,
            desc=f"Processing {label}"
        ):


            if video.endswith(".mp4"):


                video_name = video.replace(
                    ".mp4",
                    ""
                )


                extract_frames(

                    os.path.join(
                        input_folder,
                        video
                    ),

                    os.path.join(
                        output_folder,
                        video_name
                    )

                )





if __name__ == "__main__":

    process_dataset()