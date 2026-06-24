from facenet_pytorch import MTCNN
from PIL import Image
import os
from tqdm import tqdm



mtcnn = MTCNN(
    image_size=224,
    margin=20
)





def process_faces():


    classes=[
        "real",
        "fake"
    ]



    for label in classes:


        video_folder = os.path.join(
            "data/processed",
            label
        )


        videos=os.listdir(video_folder)



        for video in tqdm(
            videos,
            desc=f"Faces {label}"
        ):


            input_path=os.path.join(
                video_folder,
                video
            )


            output_path=os.path.join(
                "data/faces",
                label,
                video
            )


            os.makedirs(
                output_path,
                exist_ok=True
            )



            frames=os.listdir(input_path)



            for frame in frames:


                img=Image.open(
                    os.path.join(
                        input_path,
                        frame
                    )
                )


                face=mtcnn(img)



                if face is not None:


                    face = face.permute(
                        1,
                        2,
                        0
                    )


                    face = (
                        face.numpy()*255
                    ).astype(
                        "uint8"
                    )



                    Image.fromarray(
                        face
                    ).save(
                        os.path.join(
                            output_path,
                            frame
                        )
                    )





if __name__=="__main__":

    process_faces()