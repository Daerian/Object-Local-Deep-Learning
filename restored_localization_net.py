import tensorflow as tf
import numpy as np
import time
from datetime import timedelta
import os
from PIL import Image
from PIL import ImageFilter
import scipy.misc as sm

img_size = 227

num_channels = 3

# Number of classes, one class for each of 10 digits.
num_classes = 20



def main(unused_args):
    path = "./datasets/VOC_2007/traindata/VOC2007/JPEGImages/"
    gaussian_blur = 1
    nn_size = 227
    # path_to_image_names = "./datasets/VOC_2007/traindata/VOC2007/ImageSets/Main/aeroplane_train.txt"
    # # Get image names
    # image_names = open(path_to_image_names).readlines()
    # Get first image
    # filename = image_names[0].split(' ')[0] + '.jpg'

    image_loc = path + "/000033.jpg"
    # Open image
    image = Image.open(image_loc)

    # Crop based on which side of the image is smaller
    image_size = image.size
    resize_num = 500
    if (image_size[0] > image_size[1]):
        resize_num = image_size[1]
    else:
        resize_num = image_size[0]

    # saver = tf.train.Saver()
    saver = tf.train.import_meta_graph("./275_voc07_model.ckpt.meta")
    with tf.Session() as sess:
        # Open our saved graph
        saver.restore(sess, "./275_voc07_model.ckpt")

        graph = tf.get_default_graph()

        # for n in graph.as_graph_def().node:
        #     print(n.name)
        x = graph.get_tensor_by_name("x:0")
        # print(x)
        outputs = graph.get_tensor_by_name("outputs/Relu:0")
        # print(outputs)

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
        print("Finished pre-processing image")

        pre_proc_im = nn_im.eval()
        sm.imsave("./pre_proc_im.jpg", pre_proc_im[0])

        pred = sess.run(x, feed_dict={x: pre_proc_im})

        print(pred)

if __name__ == '__main__':
    tf.app.run(main=main)
