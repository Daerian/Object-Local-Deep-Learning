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

# path = "./datasets/VOC_2007/testdata/VOC2007/JPEGImages/"
# image_names = os.listdir(path)
# imagesz = []
# for i in range(len(image_names)):
#     print(i)
#     image_loc = path + "/" + image_names[i]

#     image = mpimg.imread(image_loc)
#     # print(image)
#     imagesz.append(image)

# data = np.array(imagesz)

# # print(data)

# np.save('./VOC_data/voc07_test.npy', data)

def main(unused_arg):

    traindata = np.load("./VOC_data/voc07_train1.npy")

    sess = tf.InteractiveSession()

    l = traindata.shape

    train_padded = np.array([])
    for i in range(l[0]):
        print(i)
        t = tf.image.resize_image_with_crop_or_pad(traindata[i], 500, 500)
        padded = t.eval()
        print(padded.shape)
        train_padded = np.append(train_padded, padded)
    np.save('./VOC_data/voc07_train_padded.npy', train_padded)


    testdata = np.load("./VOC_data/voc07_test.npy")
    l = testdata.shape
    test_padded = np.array([])
    for i in range(l[0]):
        print(i)
        t = tf.image.resize_image_with_crop_or_pad(testdata[i], 500, 500)
        padded = t.eval()
        print(padded.shape)
        test_padded = np.append(test_padded, padded)
    np.save('./VOC_data/voc07_test_padded.npy', test_padded)
    sess.close()
    


if __name__ == "__main__":
    tf.app.run(main=main)

