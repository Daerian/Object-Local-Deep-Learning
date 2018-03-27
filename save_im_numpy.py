import tensorflow as tf
import numpy as np
import os
# import cv2
# import matplotlib.image as mpimg
from PIL import Image
from PIL import ImageFilter
import scipy.misc

def main(unused_arg):
    gaussian_blur = 1
    path = "./datasets/VOC_2007/traindata/VOC2007/JPEGImages/"
    path_to_image_names = "./datasets/VOC_2007/traindata/VOC2007/ImageSets/Main/aeroplane_val.txt"
    # image_names = os.listdir(path)
    image_names = open(path_to_image_names).readlines()

    amount_of_train_images = len(image_names)
    nn_size = 227

    cropped_nn_images = np.zeros(shape=(amount_of_train_images, nn_size, nn_size, 3))
    #nn_images = np.zeros(shape=(amount_of_train_images, nn_size, nn_size, 3))
    sess = tf.InteractiveSession()
    # for i in range(len(image_names)):
    print("Starting cropping/blurring/nn_interpolation images..")
    for i in range(amount_of_train_images):
        print(i)
        # Get image
        filename = image_names[i].split(' ')[0] + '.jpg'
        image_loc = path + "/" + filename
        # image = mpimg.imread(image_loc)
        image = Image.open(image_loc)

        image_size = image.size
        resize_num = 500
        if (image_size[0] > image_size[1]):
            resize_num = image_size[1]
        else:
            resize_num = image_size[0]


        # Crop image
        t = tf.image.resize_image_with_crop_or_pad(image, resize_num, resize_num)
        cropped = t.eval()

        # Convert from numpy to PIL image
        pil_cropped_image = Image.fromarray(cropped)

        # Blur image
        blurred = pil_cropped_image.filter(ImageFilter.GaussianBlur(gaussian_blur))

        # Apply nearest neighbour interpolation
        p = tf.convert_to_tensor(blurred)
        nn_im = tf.image.resize_nearest_neighbor([p],
            (nn_size, nn_size))

        # Store in respective arrays to be saved
        cropped_nn_images[i] = nn_im[0].eval()

        # blur image, not using padding
        #blurred = image.filter(ImageFilter.GaussianBlur(gaussian_blur))
        # Convert back to tensor and do nearest_neighbor interpolation
        #p = tf.convert_to_tensor(blurred)
        #nn_im = tf.image.resize_nearest_neighbor([p],
         #   (nn_size, nn_size))
        # nn_images[i] = nn_im[0].eval()

    print("Saving cropped/blurred/nn_interpolation..")
    sess.close()
    np.save('./VOC_data/voc07_cv_only_blurred_nn_cropped.npy', cropped_nn_images)
    print("Done!")
    # np.save('./VOC_data/voc07_train_only_blurred_nn.npy', nn_images)

    #padded_images = None
    # blurred_padded_images = None

    # path = "./datasets/VOC_2007/testdata/VOC2007/JPEGImages"
    # path_to_image_names = "./datasets/VOC_2007/testdata/VOC2007/ImageSets/Main/aeroplane_test.txt"
    # image_names = open(path_to_image_names).readlines()
    # amount_of_train_images = len(image_names)

    # ## NOTE: images aren't actually cropped, they are padded, but called cropped
    # ## as the intention was to use this code to crop images to 227*227 and then use gaussian blur
    # cropped_images = np.zeros(shape=(amount_of_train_images, 500, 500, 3))
    # blurred_cropped_images = np.zeros(shape=(amount_of_train_images, 500, 500, 3))

    # sess = tf.InteractiveSession()
    # # for i in range(len(image_names)):
    # print("Starting cropping/blurring images..")
    # for i in range(amount_of_train_images):
    #     print(i)
    #     # Get image
    #     filename = image_names[i].split(' ')[0] + '.jpg'
    #     image_loc = path + "/" + filename
    #     # image = mpimg.imread(image_loc)
    #     image = Image.open(image_loc)

    #     # crop image
    #     t = tf.image.resize_image_with_crop_or_pad(image, 500, 500)
    #     cropped = t.eval()

    #     # Convert from numpy to PIL image
    #     pil_cropped_image = Image.fromarray(cropped)

    #     # Blur image
    #     blurred = pil_cropped_image.filter(ImageFilter.GaussianBlur(gaussian_blur))

    #     # Store in respective arrays to be saved
    #     cropped_images[i] = cropped
    #     blurred_cropped_images[i] = blurred

    # print("Saving cropped/blurred..")
    # sess.close()
    # np.save('./VOC_data/voc07_test_only_padded.npy', cropped_images)
    # np.save('./VOC_data/voc07_test_only_blurred_cropped.npy', blurred_cropped_images)

    #sess.close()


    # traindata = np.load("./VOC_data/voc07_train1.npy")

    # sess = tf.InteractiveSession()

    # l = traindata.shape
    # print("starting")
    # train_padded = np.array([])
    # for i in range(2501):
    #     print(i)
    #     t = tf.image.resize_image_with_crop_or_pad(traindata[i], 227, 227)
    #     # t = tf.image.resize_images(traindata[i], 227, 227)
    #     padded = t.eval()
    #     print(padded.shape)
    #     train_padded = np.append(train_padded, padded)
    # sess.close()
    # np.save("./VOC_data/voc07_train_resize_227.npy", train_padded)
    # print("saving image..")
    # scipy.misc.imsave('./outfile.jpg', train_padded[0])
    

    # testdata = np.load("./VOC_data/voc07_test.npy")
    # sess = tf.InteractiveSession()

    # l = testdata.shape
    # test_padded = np.array([])
    # for i in range(l[0]):
    #     print(i)
    #     t = tf.image.resize_image_with_crop_or_pad(testdata[i], 500, 500)
    #     padded = t.eval()
    #     print(padded.shape)
    #     test_padded = np.append(test_padded, padded)
    # np.save('./VOC_data/voc07_test_padded.npy', test_padded)
    # sess.close()



if __name__ == "__main__":
    tf.app.run(main=main)

