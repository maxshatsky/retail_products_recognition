import streamlit as st
import subprocess
import os
import tensorflow as tf
import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt
import uuid
from PIL import Image


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
        output_folder='cropped_images'
):
    img = plt.imread(image_path)

    os.makedirs(output_folder, exist_ok=True)

    grouped = bbox_data.groupby('label')

    for label, group in grouped:
        print(f"label is {label}, the type is {type(label)}\n\n")
        for idx, row in group.iterrows():
            w = row['w'] * img.shape[1]
            h = row['h'] * img.shape[0]
            x = row['x'] * img.shape[1] - w / 2
            y = row['y'] * img.shape[0] - h / 2

            # Crop and save the image
            x1, y1 = int(x), int(y)
            x2, y2 = int(x + w), int(y + h)
            cropped_img = img[y1:y2, x1:x2]
            cropped_img = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2RGB)
            os.makedirs(os.path.join(output_folder, str(label)), exist_ok=True)
            output_path = os.path.join(output_folder, str(label), f'label_{label}_bbox_{idx}.png')
            cv2.imwrite(output_path, cropped_img)


def load_images_from_folder(folder_path):
    images = []
    for filename in os.listdir(folder_path):
        img_path = os.path.join(folder_path, filename)
        img = Image.open(img_path)
        images.append(img)
    return images


def show_cropped_images(
        base_folder
):
    st.title("Detected objects:")

    # base_folder = "images"
    subfolders = [f.path for f in os.scandir(base_folder) if f.is_dir()]

    for subfolder in subfolders:
        st.subheader(subfolder)

        images = load_images_from_folder(subfolder)
        num_images = len(images)
        num_cols = min(num_images, 5)  # Limit the number of columns to 5
        num_rows = (num_images + num_cols - 1) // num_cols

        for row in range(num_rows):
            # cols = st.beta_columns(num_cols)
            cols = st.columns(num_cols)
            for col in range(num_cols):
                index = row * num_cols + col
                if index < num_images:
                    cols[col].image(images[index], use_column_width=True)


st.title("Retail product object detection")

uploaded_file = st.file_uploader("Upload an image!", type=["jpg", "jpeg"])

if uploaded_file is not None:
    # Save the uploaded picture as 'uploaded_pic.jpg'
    unique_foldername = f"{str(uuid.uuid4())}"
    unique_filename = unique_foldername + "/uploaded_pic.jpg"

    # But first delete it if exists
    # for windows
    # process = subprocess.Popen(
    #     "rmdir /s /q " + unique_foldername,
    #     shell=True,
    #     stdout=subprocess.PIPE
    # )

    process = subprocess.run(
        "rmdir /s /q " + unique_foldername,
        shell=True
    )

    # for unix
    # process = subprocess.Popen("rm -rf " + unique_foldername, shell=True, stdout=subprocess.PIPE)

    print(f"unique_foldername = {unique_foldername}")
    print(f"unique_filename = {unique_filename}")

    os.makedirs(unique_foldername, exist_ok=True)
    with open(unique_filename, "wb") as f:
        f.write(uploaded_file.getvalue())

    st.write("Image uploaded successfully!")

    # Step 2: Run detection script
    st.write("Running object detection on the uploaded image...")

    command = "python detect.py"
    command_parameters = " --weights weights/weights_trained_by_sam.pt"
    command_parameters += " --source " + unique_filename
    command_parameters += " --conf 0.1 --no-trace --save-txt"
    command_parameters += " --save-conf  --project " + unique_foldername
    process = subprocess.Popen(
        command + command_parameters,
        shell=True,
        stdout=subprocess.PIPE
    )
    process.wait()

    st.write("Object detection completed!")

    result_image_path = unique_foldername + "/exp/uploaded_pic.jpg"

    # Step 3: Display resulting image
    if os.path.isfile(result_image_path):
        result_image = open(result_image_path, "rb").read()
        st.image(result_image, caption="YOLOv7 resulting Image", use_column_width=True)
    else:
        st.write("No resulting image found.")

    # Save cropped images.
    data = pd.read_csv(
        unique_foldername + "/exp/labels/uploaded_pic.txt",
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
        iou_threshold=0.2
    )

    filtered_data = data.iloc[selected_indices].reset_index(drop=True)

    save_bboxed_images(
        image_path=unique_filename,
        bbox_data=filtered_data,
        output_folder=unique_foldername + '/cropped_images'
    )

    show_cropped_images(
        unique_foldername + '/cropped_images'
    )



else:
    st.write("Please upload an image.")
