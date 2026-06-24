import streamlit as st

from inference import predict



st.title(
    "ISTVT Inspired Deepfake Detector"
)



st.write(
"""
Upload a video and the model will classify it.
"""
)



video = st.file_uploader(

    "Upload video",

    type=[
        "mp4",
        "avi"
    ]

)




if video:


    with open(
        "temp.mp4",
        "wb"
    ) as f:

        f.write(
            video.read()
        )



    result=predict(
        "temp.mp4"
    )



    if result=="Face not detected":

        st.error(
            "No face detected"
        )



    else:


        label,confidence=result



        if label=="FAKE":

            st.error(
                f"""
                Prediction:
                {label}

                Confidence:
                {confidence:.2f}
                """
            )


        else:


            st.success(

                f"""
                Prediction:
                {label}

                Confidence:
                {confidence:.2f}
                """

            )