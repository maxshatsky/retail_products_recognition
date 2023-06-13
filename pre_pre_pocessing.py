import os
import cv2


def blur_and_decrease_shape (image_directory,
                             image_name,
                             blur_amount,
                             decrease_factor,
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

    # Decrease shape
    width = int(image.shape[1] / decrease_factor)
    height = int(image.shape[0] / decrease_factor)
    resized_image = cv2.resize(blurred_image, (width, height))

    # Check if the output directory exists
    if not os.path.exists(output_directory) :
        os.makedirs(output_directory)

    # Create the output path with the provided image name
    output_path = os.path.join(output_directory, image_name)

    # Save the processed image
    cv2.imwrite(output_path, resized_image)

    # Display the original and processed images
    cv2.imshow("Original Image", image)
    cv2.imshow("Processed Image", resized_image)



def main():
    blur_and_decrease_shape(image_name='20180824-15-44-39-474.jpg',
                            image_directory='/Users/alexanderzaznobin/Desktop/ds4 prj/test2019/',
                            blur_amount=15,
                            decrease_factor=4,
                            output_directory='/Users/alexanderzaznobin/Desktop/ds4 prj/ppp/')

if __name__ == "__main__":
    main()
