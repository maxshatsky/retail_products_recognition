import streamlit as st
import subprocess
import os

# Step 1: Upload picture
st.title("Object Detection")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg"])

if uploaded_file is not None:
    # Save the uploaded picture as 'uploaded_pic.jpg'
    with open("uploaded_pic.jpg", "wb") as f:
        f.write(uploaded_file.getvalue())

    st.write("Image uploaded successfully!")

    # Step 2: Run detection script
    st.write("Running object detection on the uploaded image...")

    # for windows
    # process = subprocess.Popen("rmdir /s /q server_predictions", shell=True, stdout=subprocess.PIPE)

    # for unix
    process = subprocess.Popen("rm -rf server_predictions", shell=True, stdout=subprocess.PIPE)
    command = "python detect.py"
    command_parameters = " --weights weights/weights_trained_by_sam.pt --source uploaded_pic.jpg"
    command_parameters += " --conf 0.1 --no-trace --save-txt"
    command_parameters += " --save-conf  --project server_predictions"
    process = subprocess.Popen(
        command+command_parameters,
        shell=True,
        stdout=subprocess.PIPE
    )
    process.wait()

    st.write("Object detection completed!")

    # Step 3: Display resulting image
    result_image_path = "server_predictions/exp/uploaded_pic.jpg"

    if os.path.isfile(result_image_path):
        result_image = open(result_image_path, "rb").read()
        st.image(result_image, caption="Resulting Image", use_column_width=True)
    else:
        st.write("No resulting image found.")

else:
    st.write("Please upload an image.")
