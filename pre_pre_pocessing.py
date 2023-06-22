import os
import random

import cv2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import numpy
import translators as ts
import json
import xarray.core.computation
from deep_translator import GoogleTranslator
import os
from PIL import Image, ImageDraw
import cv2
from tensorflow import keras
import pickle

def translation(row):
    translated = GoogleTranslator().translate(text=row['name'])
    return translated
def translation_clas(row):
    translated = GoogleTranslator().translate(text=row['clas'])
    return translated
def get_merged_ds(directory):
    instances_train2019 = os.path.join(directory, 'instances_train2019.json')
    with open(instances_train2019, 'r') as f :
        json_data = json.load(f)
    image_names = pd.DataFrame(json_data['images'])
    photos_annotations = pd.DataFrame(json_data['annotations'])
    pd.DataFrame(json_data['categories'])
    name_df = pd.DataFrame(json_data['__raw_Chinese_name_df'])
    name_df['en_name'] = name_df.apply(translation, axis=1)
    name_df['en_clas'] = name_df.apply(translation_clas, axis=1)
    all_data = photos_annotations.merge(image_names).merge(name_df)
    return all_data

def get_nparray_of_image(directory, filename):
    image_path = os.path.join(directory, filename)
    image = Image.open(image_path).convert('RGB')
    return np.array(image)


def make_array_of_all_images(annotation_dataset,directory):
    all_image_list=[]
    for i,filename in enumerate(annotation_dataset['file_name']):
        image_array=get_nparray_of_image(directory, filename)
        all_image_list.append(image_array)
        print(f"read {i}")
        # if i>10000:
        #     break
    all_image_array=np.array(all_image_list)
    return all_image_array

def make_array_of_one_image(data_row,directory):
    image_list=[]
    image_array=get_nparray_of_image(directory, data_row['file_name'])
    image_list.append(image_array)
    image_array=np.array(image_list)
    return image_array
def blur_and_decrease_shape (image_directory,
                             image_name,
                             blur_amount,
                             width,
                             height,
                             output_directory) :
    """
     Applies Gaussian blur to an image and resizes it, then saves the processed image in a specified directory.

     Parameters:
     image_path (str): The directory path where the input image is located.
     image_name (str): The name of the input image file.
     blur_amount (int): The size of the kernel used for the Gaussian blur. This must be an odd number.
     decrease_factor (int): The factor by which to decrease the size of the image.
     output_directory (str): The directory where the processed image will be saved.

     Returns:
     None
     """

    # Create full image path
    full_image_path = os.path.join(image_directory, image_name)

    # Load the image
    image = cv2.imread(full_image_path)

    # Apply blur
    blurred_image = cv2.GaussianBlur(image, (blur_amount, blur_amount), 0)

    # # Decrease shape
    # width = int(image.shape[1] / decrease_factor)
    # height = int(image.shape[0] / decrease_factor)
    resized_image = cv2.resize(blurred_image, (width, height))

    # Check if the output directory exists
    if not os.path.exists(output_directory) :
        os.makedirs(output_directory)

    # Create the output path with the provided image name
    output_path = os.path.join(output_directory, image_name)

    # Save the processed image
    cv2.imwrite(output_path, resized_image)


def use_the_baseline_CNN():
    model = keras.models.load_model('model.h5')
    return model
def one_picture(annotations_data_set,num,image_directory_to_save, itinial_photo_dir):
    file_name=annotations_data_set.loc[num,'file_name']
    blur_and_decrease_shape(image_name=file_name,
                            image_directory=itinial_photo_dir,
                            blur_amount=15,
                            width=80,
                            height=80,
                            output_directory=image_directory_to_save)
    initial_full_image_path = os.path.join(itinial_photo_dir, file_name)
    resized_full_image_path = os.path.join(image_directory_to_save, file_name)
    image = cv2.imread(initial_full_image_path)
    resized_image=cv2.imread(resized_full_image_path)
    cv2.imshow("Original Image", image)
    cv2.imshow("Processed Image", resized_image)
    output_path = os.path.join(image_directory_to_save, file_name)
    output_path_full_scale = os.path.join(image_directory_to_save, 'full_scale'+file_name)
    cv2.imwrite(output_path, resized_image)
    cv2.imwrite(output_path_full_scale, image)
    np_im_array=make_array_of_one_image(data_row=annotations_data_set.loc[num,:],
                                        directory=image_directory_to_save)
    print('data_row',annotations_data_set.loc[num,:])
    print('np_im_array.shape',np_im_array.shape)
    model=use_the_baseline_CNN()
    with open('label_encoder.pkl', 'rb') as file :
        label_encoder = pickle.load(file)
    y_predict=model.predict(np_im_array)
    one_hot_decode = np.argmax(np.where(y_predict > 0.5, 1, 0), axis=1)
    y_prediction_decoded = list(label_encoder.inverse_transform(one_hot_decode))
    print('y_prediction_decoded',y_prediction_decoded)
    print('annotations_data_set',annotations_data_set.loc[num,'en_name'])
    return y_prediction_decoded
def main():
    # data=get_merged_ds('/Users/alexanderzaznobin/Desktop/ds4 prj/')
    # data.to_csv('/Users/alexanderzaznobin/Desktop/ds4 prj/data.csv', index=False)
    data_set=pd.read_csv('/Users/alexanderzaznobin/Desktop/ds4 prj/data.csv')
    # all_image_list=make_array_of_all_images(annotation_dataset=data,
    #                          directory='/Users/alexanderzaznobin/Desktop/ds4 prj/train2019_ppp/')
    # np.save('/Users/alexanderzaznobin/Desktop/ds4 prj/all_image_list.npy', all_image_list)
    # all_image_list = np.load('/Users/alexanderzaznobin/Desktop/ds4 prj/all_image_list.npy')
    # print(all_image_list.shape)
    # for i, file_name in enumerate(data['file_name']):
    #     blur_and_decrease_shape(image_name=file_name,
    #                             image_directory='/Users/alexanderzaznobin/Desktop/ds4 prj/train2019/',
    #                             blur_amount=15,
    #                             width=80,
    #                             height=80,
    #                             output_directory='/Users/alexanderzaznobin/Desktop/ds4 prj/train2019_ppp/')
    r_int = random.randint(1, data_set.shape[0])
    print(r_int)
    one_picture(annotations_data_set=data_set,
                     num=r_int,
                     image_directory_to_save='/Users/alexanderzaznobin/Desktop/ds4 prj/test_model/',
                     itinial_photo_dir='/Users/alexanderzaznobin/Desktop/ds4 prj/train2019/')


if __name__ == "__main__":
    main()
