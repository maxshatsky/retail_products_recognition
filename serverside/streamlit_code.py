import streamlit as st
import subprocess
import os
import tensorflow as tf
import pandas as pd
import numpy as np
import cv2
from matplotlib import plt


def convert_to_x1y1x2y2_format(boxes):
    x, y, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    x1 = x - w / 2
    y1 = y - h / 2
    x2 = x + w / 2
    y2 = y + h / 2
    return np.stack((x1, y1, x2, y2), axis=1)


def convert_to_xywh_format(boxes):
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    x = (x1 + x2) / 2
    y = (y1 + y2) / 2
    w = x2 - x1
    h = y2 - y1
    return np.stack((x, y, w, h), axis=1)


def non_max_suppression_tf(boxes, scores, iou_threshold):
    selected_indices = tf.image.non_max_suppression(
        convert_to_x1y1x2y2_format(boxes),
        scores,
        max_output_size=100,
        iou_threshold=iou_threshold
    )
    # selected_boxes = tf.gather(boxes, selected_indices)
    # selected_scores = tf.gather(scores, selected_indices)
    # return selected_boxes, selected_scores
    return selected_indices


def save_bboxed_images(
    image_path,
    bbox_data,
    output_folder = 'cropped_images'
):
    img = plt.imread(image_path)

    os.makedirs(output_folder, exist_ok=True)

    grouped = bbox_data.groupby('label')


    for label, group in grouped:
        print(f"label is {label}, the type is {type(label)}\n\n")
        for idx, row in group.iterrows():
            w = row['w'] * img.shape[1]
            h = row['h'] * img.shape[0]
            x = row['x'] * img.shape[1]-w/2
            y = row['y'] * img.shape[0]-h/2

            # Crop and save the image
            x1, y1 = int(x), int(y)
            x2, y2 = int(x + w), int(y + h)
            cropped_img = img[y1:y2, x1:x2]
            cropped_img = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2RGB)
            os.makedirs(os.path.join(output_folder, str(label)), exist_ok=True)
            output_path = os.path.join(output_folder, str(label), f'label_{label}_bbox_{idx}.png')
            cv2.imwrite(output_path, cropped_img)


st.title("Retail product object detection")

uploaded_file = st.file_uploader("Upload an image!", type=["jpg", "jpeg"])

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

    result_image_path = "server_predictions/exp/uploaded_pic.jpg"

    # Save cropped images.
    data = pd.read_csv(
        result_image_path[:-3]+"txt",
        sep=' ',
        names=[
            'label',
            'x',
            'y',
            'w',
            'h',
            'conf'
        ]
    )

    boxes = data[['x', 'y', 'w', 'h']].to_numpy()
    scores = data['conf'].to_numpy()

    selected_indices = non_max_suppression_tf(
        boxes=boxes,
        scores=scores,
        iou_threshold=0.5
    )

    filtered_data = data.iloc[selected_indices].reset_index(drop=True)

    save_bboxed_images(
        image_path=result_image_path,
        bbox_data=filtered_data,
        output_folder = 'cropped_images'
    )


    # Step 3: Display resulting image
    if os.path.isfile(result_image_path):
        result_image = open(result_image_path, "rb").read()
        st.image(result_image, caption="Resulting Image", use_column_width=True)
    else:
        st.write("No resulting image found.")

else:
    st.write("Please upload an image.")
