import tensorflow as tf
import numpy as np
import os
# import cv2
import matplotlib.image as mpimg

# voc_train_location = "/cmshome/korapaty/Desktop/cscd94_space/Object-Local-Deep-Learning/datasets/VOC_2007/traindata/VOC2007/JPEGImages/000005.jpg"

# voc_queue = tf.train.string_input_producer(
#   tf.train.match_filenames_once(voc_train_location))


# image_reader = tf.WholeFileReader()


# _, image_file = image_reader.read(voc_queue)

# voc_image = tf.image.decode_jpeg(
#     contents = image_file,
#     channels = 3)


# with tf.Session() as sess:
#     init = tf.global_variables_initializer()
#     init.run()

#     coord = tf.train.Coordinator()
#     threads = tf.train.start_queue_runners(coord=coord)

#     print("######################################################")
#     t = sess.run([voc_image])

#     print(t)

#     coord.request_stop()
#     coord.join(threads)

path = "./datasets/VOC_2007/testdata/VOC2007/JPEGImages/"
image_names = os.listdir(path)
imagesz = []
for i in range(len(image_names)):
    print(i)
    image_loc = path + "/" + image_names[i]

    image = mpimg.imread(image_loc)
    # print(image)
    imagesz.append(image)

data = np.array(imagesz)

# print(data)

np.save('./VOC_data/voc07_test.npy', data)
